"""Run the minimal adapter gates and frozen-model experiment in conda sdn.

python -B -m slice_alloc.tools.joint_run [calibrate|train|evaluate|all]
No old configuration, policy weights, service profile or VNF runtime is read.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from statistics import mean
from time import perf_counter
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import unittest
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from ..joint import (Request,Compiler,JointEnv,AtomicLedger,RESOURCES,CAPACITIES,CACHE_CAPACITIES,
                     KINDS,PERIOD,DEADLINE,choose,request_hash,write_tasks)
from ..src.qlearning import DoubleDQN,tensor_observation
from ..src.policy import CandidatePolicy
from .platform_verify import build,NS3,radio_accounting,export_radio_evidence,sha

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'slice_alloc/results/joint'
PROTOCOL=dict(scope='four-node static/bounded-speed IP/NDN repeated-object adapter; NOT full thesis platform',
    calibrate_seeds=[201,202,203],confirm_seeds=[301,302,303],initializations=[11,22,33],
    train_scenario_seeds=[1000,4999],validation_seeds=[61000,61001,61002,61003],
    test_seeds=list(range(72000,72008)),test_wifi_seed_offset=80000,
    loads=[.4,.5,.6],requests=24,steps_per_model=32768,checkpoint_interval=8192,
    price_grid=[0.,.15,.3,.6,1.,2.],gamma=1.,capacity=CAPACITIES,cache_capacity=CACHE_CAPACITIES,
    deadline_seconds=DEADLINE,period_seconds=PERIOD,airtime_margin=1.15,
    calibration_scope='ERP6 unicast, 1000B single-chunk, 30-character fixed-format names, two consumers, bounded concurrent contracts',
    reward='delayed admitted weighted count in fast MODEL; independent packet replay determines actual SLA success',
    selection='validation model utility only; no test ranking changes the workload/objective/policy',
    statistical_unit='scenario seed; average loads and RL initializations within seed',
    limitations=['no general function placement/path search yet','logical state quota lacks native per-slice FIB/PIT enforcement',
                 'no CPU quota executor','ID mechanism remains in separate protocol regression, not this adapter',
                 'native replay evaluates configured pinned cache and shared medium; not a universal admission guarantee'])


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def source_hashes():
    return {n:sha(ROOT/n) for n in ('slice_alloc/joint.py','slice_alloc/tools/joint_run.py',
        'slice_alloc/execution/protocol_probe.cc','slice_alloc/src/qlearning.py','slice_alloc/src/policy.py',
        'slice_alloc/protocol_contracts.py','slice_alloc/tools/platform_verify.py')}


def native_case(task):
    name,path,seed,speed=task
    prefix=OUT/'packets'/name;prefix.parent.mkdir(parents=True,exist_ok=True)
    command=[str(NS3/'build/scratch/protocol-probe'),f'--output={prefix}',f'--seed={seed}',
        '--wireless=1','--uniformL2=1','--peerUnicast=1','--equalRetry=1','--cache=1',
        f'--replayFile={path}','--backhaulMbps=1','--cacheEntries=4',f'--speed={speed}']
    env=dict(os.environ,LD_LIBRARY_PATH=str(NS3/'build/lib'))
    process=subprocess.run(command,env=env,capture_output=True,text=True)
    Path(str(prefix)+'-process.log').write_text(process.stdout+process.stderr)
    if process.returncode:raise RuntimeError(f'native replay failed: {name}: {process.stderr[-1500:]}')
    rows=list(csv.DictReader(Path(str(prefix)+'-events.csv').open()))
    frames=list(csv.DictReader(Path(str(prefix)+'-frames.csv').open()))
    summary=json.loads(Path(str(prefix)+'-summary.json').read_text())
    inputs=[]
    for line in Path(path).read_text().splitlines():
        at,node,mode,obj,start,end=line.split();inputs.append((float(at),int(node),mode,obj,float(start),float(end)))
    completions={}
    for r in rows:
        if r['event'] in ('app_data','object_ip_data'):
            key=(int(r['node']),r['name'])
            completions.setdefault(key,[]).append((float(r['time']),float(r['detail'])))
    slices={}
    for at,node,mode,obj,start,end in inputs:
        group=slices.setdefault(obj,dict(expected=0,delivered=0,on_time=0,latencies=[]))
        group['expected']+=1
        matches=completions.get((node,obj),[])
        if matches:
            match=next((i for i,(when,delay) in enumerate(matches) if abs(when-delay-at)<=1e-7),None)
            if match is None:raise ValueError('completion does not match any declared task birth')
            when,delay=matches.pop(match)
            # Independently reconstruct birth from raw completion time/delay.
            if abs(when-delay-at)>1e-7:raise ValueError('completion-to-task matching failed')
            group['delivered']+=1;group['on_time']+=int(delay<=DEADLINE+1e-9);group['latencies'].append(delay)
    for group in slices.values():group['success']=group['on_time']==group['expected']
    radio=export_radio_evidence(radio_accounting(rows),frames)
    checks=dict(phy_clocks_agree=radio['independent_phy_clocks_agree'],
        frame_counts_match=len(frames)==radio['phy_mpdu_events'],
        frame_bytes_match=sum(int(r['bytes']) for r in frames)==radio['phy_frame_bytes'],
        all_task_emits_present=summary['ndn_sent']+summary['object_ip_sent']==len(inputs),
        no_unclassified_completions=not any(completions.values()))
    if not all(checks.values()):raise ValueError(f'native raw record mismatch: {name}, {checks}')
    modes={mode for _,_,mode,_,_,_ in inputs}
    network={}
    if modes=={'ndn'}:
        roles={(0,257):'origin>relay',(1,257):'relay>origin',(1,258):'relay>a',
               (1,259):'relay>b',(2,257):'a>relay',(3,257):'b>relay'}
        for r in rows:
            key=roles.get((int(r['node']),int(r['face'])))
            if key and r['event'] in ('out_interest','out_data'):
                network['bytes:'+key]=network.get('bytes:'+key,0)+int(r['bytes'])
    elif modes=={'ip'}:
        for r in rows:
            if r['event']=='device_tx' and r['face']=='0' and r['node'] in ('0','1'):
                key='bytes:'+('origin>relay' if r['node']=='0' else 'relay>origin')
                network[key]=network.get(key,0)+int(r['bytes'])-2  # PPP link header excluded
    result=dict(name=name,seed=seed,speed=speed,checks=checks,passed=all(checks.values()),
        expected=len(inputs),delivered=sum(x['delivered'] for x in slices.values()),
        on_time=sum(x['on_time'] for x in slices.values()),strict_slices=sum(x['success'] for x in slices.values()),
        slices=slices,radio=radio,network_bytes=network,
        ndn_interest_bytes=sorted({int(r['bytes']) for r in rows if r['event']=='out_interest'}),
        ndn_data_header_bytes=sorted({int(r['bytes'])-1000 for r in rows if r['event']=='out_data'}),
        source_payload_bytes=sum(int(r['bytes']) for r in rows if r['event'] in ('producer_payload','object_ip_produce')),
        cache_payload_peaks=max((int(r['bytes']) for r in rows if r['event']=='cs_payload'),default=0),
        lease_releases=sum(r['event']=='replay_lease_release' for r in rows),
        files={str(p.relative_to(OUT)):sha(p) for p in sorted(prefix.parent.glob(prefix.name+'-*'))},
        input_file=str(Path(path).relative_to(OUT)),input_sha256=sha(path),command=command)
    return result


def calibration_input(name,kind,rounds,concurrent):
    requests=[Request(i,1.+i*.004,rounds,(kind,)) for i in range(concurrent)]
    path=OUT/'inputs'/f'{name}.tsv';path.parent.mkdir(parents=True,exist_ok=True)
    # Input task construction precedes calibration. It makes no reservation or
    # feasibility claim and therefore does not bootstrap from an unfit profile.
    rows=[]
    for r in requests:
        for j in range(rounds):
            for member in range(2):
                rows.append((r.arrival+j*PERIOD+.001*member,2+member,'ip' if kind=='ip' else 'ndn',r.name,
                             r.arrival if kind=='ndn_pin' else 0.,r.end if kind=='ndn_pin' else 0.))
    path.write_text(''.join('\t'.join(map(str,row))+'\n' for row in sorted(rows)))
    return path,requests


def calibrate():
    OUT.mkdir(parents=True,exist_ok=True)
    platform_info=build();write(OUT/'protocol.json',PROTOCOL)
    tasks=[];metadata={}
    for split,seeds in (('cal',PROTOCOL['calibrate_seeds']),('hold',PROTOCOL['confirm_seeds'])):
        for kind in KINDS:
            for rounds in (2,4,7):
                for concurrent in (1,3):
                    for seed in seeds:
                        name=f'{split}_{kind}_{rounds}_{concurrent}_{seed}'
                        path,requests=calibration_input(name,kind,rounds,concurrent)
                        tasks.append((name,path,seed,0 if split=='cal' else 5.))
                        metadata[name]=(split,kind,requests)
    with ThreadPoolExecutor(4) as pool:rows=list(pool.map(native_case,tasks))
    fit=[r for r in rows if metadata[r['name']][0]=='cal']
    hold=[r for r in rows if metadata[r['name']][0]=='hold']
    measured=max(r['radio']['tx_seconds_from_state']/sum(q.rounds for q in metadata[r['name']][2]) for r in fit)
    headers={v for r in fit for v in r['ndn_data_header_bytes']};interests={v for r in fit for v in r['ndn_interest_bytes']}
    if len(headers)!=1 or len(interests)!=1:raise ValueError('wire format is outside fixed-size adapter support')
    profile=dict(passed=True,heldout_passed=True,airtime_seconds_per_pair_round=measured*PROTOCOL['airtime_margin'],
        ndn_data_header_bytes=next(iter(headers)),ndn_interest_bytes=next(iter(interests)),
        fitted_max_seconds=measured,margin=PROTOCOL['airtime_margin'],scope=PROTOCOL['calibration_scope'],
        calibration_seeds=PROTOCOL['calibrate_seeds'],confirmation_seeds=PROTOCOL['confirm_seeds'],
        universal_service_guarantee=False)
    compiler=Compiler(profile)
    for row in rows:
        split,kind,requests=metadata[row['name']]
        plans=[compiler.compile(r,kind) for r in requests]
        expected={}
        for p in plans:
            for key,value in p.reservation.items():
                if key.startswith('bytes:'):expected[key]=expected.get(key,0.)+value
        keys=set(row['network_bytes']) if kind=='ip' else set(expected)|set(row['network_bytes'])
        byte_match=all(expected.get(k,0)==row['network_bytes'].get(k,0) for k in keys)
        allowance=profile['airtime_seconds_per_pair_round']*sum(r.rounds for r in requests)
        row['contract_checks']=dict(network_bytes_exact=byte_match,
            measured_tx_within_empirical_allowance=row['radio']['tx_seconds_from_state']<=allowance+1e-9,
            all_tasks_within_declared_deadline=row['on_time']==row['expected'],
            pinned_leases_released=row['lease_releases']==(len(requests) if kind=='ndn_pin' else 0),
            scope_is_fixed_1000B=len(row['ndn_data_header_bytes'])<=1)
        row['model_allowance_seconds']=allowance
        row['expected_network_bytes']=expected
    profile['passed']=all(all(r['contract_checks'].values()) for r in fit)
    profile['heldout_passed']=all(all(r['contract_checks'].values()) for r in hold)
    write(OUT/'calibration.json',dict(profile=profile,cases=rows,platform=platform_info,
        source_sha256=source_hashes(),protocol=PROTOCOL))
    print(json.dumps(dict(calibration_passed=profile['passed'],heldout_passed=profile['heldout_passed'],cases=len(rows),profile=profile),ensure_ascii=False),flush=True)
    if not profile['passed'] or not profile['heldout_passed']:
        print([(r['name'],r['contract_checks']) for r in rows if not all(r['contract_checks'].values())],flush=True)
        raise ValueError('calibration gate failed; do not start RL training')


def profile():
    data=json.loads((OUT/'calibration.json').read_text())
    if data['source_sha256']!=source_hashes():raise ValueError('source changed after calibration; rerun gates before training')
    Compiler(data['profile'])
    return data['profile']


def episode(env,method,seed,price=0.,model=None,trace=False):
    obs,_=env.reset(seed=seed);reward=0.;started=perf_counter();invalid=0
    while not env.done:
        action=int(model.predict(obs,deterministic=True)[0]) if model is not None else choose(env,method,price)
        obs,r,_,_,_=env.step(action);reward+=r
    accepted=[p for _,p in env.decisions if p is not None]
    assert reward==sum(p.request.weight for p in accepted)
    assert not env.ledger.active and not env.ledger.cache.leases
    row=dict(seed=seed,load=env.load,method=method,requests=len(env.requests),accepted=len(accepted),
        model_utility=reward,model_acceptance=len(accepted)/len(env.requests),
        input_sha256=request_hash(env.requests),decision_ms=1000*(perf_counter()-started)/len(env.requests),
        rejected_no_candidate=env.no_candidate,policy_rejected=env.policy_rejected,
        selections={k:sum(p.kind==k for p in accepted) for k in KINDS},
        decisions=[dict(id=r.id,arrival=r.arrival,rounds=r.rounds,allowed=r.allowed,
            selected=p.kind if p else None) for r,p in env.decisions])
    return (row,accepted) if trace else row


def validation(model=None,method='ddqn',price=0.):
    results=[episode(JointEnv(profile(),count=PROTOCOL['requests'],load=load),method,seed,price,model)
             for load in PROTOCOL['loads'] for seed in PROTOCOL['validation_seeds']]
    return dict(utility=mean(r['model_utility'] for r in results),rows=results)


class ValidationCallback(BaseCallback):
    def __init__(self,path):super().__init__();self.path=path;self.best=-np.inf;self.history=[]
    def record(self):
        result=validation(self.model,'ppo');self.history.append(dict(steps=self.num_timesteps,utility=result['utility']))
        if result['utility']>self.best:self.best=result['utility'];self.model.save(self.path)
        print(f'PPO {self.path.stem} steps={self.num_timesteps} validation={result["utility"]:.3f}',flush=True)
    def _on_training_start(self):self.record()
    def _on_step(self):
        if self.num_timesteps%PROTOCOL['checkpoint_interval']==0:self.record()
        return True


def train():
    torch.set_num_threads(1);prof=profile();(OUT/'models').mkdir(exist_ok=True)
    baseline_trials=[]
    for method in ('resource_price','adaptive_price'):
        for price in PROTOCOL['price_grid']:
            r=validation(method=method,price=price);baseline_trials.append(dict(method=method,price=price,utility=r['utility']))
    selected={m:max((x for x in baseline_trials if x['method']==m),key=lambda x:x['utility']) for m in ('resource_price','adaptive_price')}
    write(OUT/'baselines.json',dict(trials=baseline_trials,selected=selected))
    models=[]
    for algorithm in ('ddqn','ppo'):
        for initialization in PROTOCOL['initializations']:
            path=OUT/'models'/f'{algorithm}_{initialization}'
            env=JointEnv(prof,training=True,count=PROTOCOL['requests'],training_loads=PROTOCOL['loads'])
            if algorithm=='ddqn':
                agent=DoubleDQN(env.state_dim,env.candidate_dim,seed=initialization)
                obs,_=env.reset(seed=initialization);history=[];best=-np.inf
                for step in range(1,PROTOCOL['steps_per_model']+1):
                    epsilon=max(.05,1.-step/(PROTOCOL['steps_per_model']*.7))
                    action=agent.explore(obs,epsilon);nxt,reward,done,_,_=env.step(action)
                    agent.remember(obs,action,reward,nxt,done)
                    if step>=1024 and step%4==0:agent.update()
                    obs=env.reset()[0] if done else nxt
                    if step%PROTOCOL['checkpoint_interval']==0:
                        result=validation(agent);history.append(dict(steps=step,utility=result['utility']))
                        if result['utility']>best:best=result['utility'];agent.save(path.with_suffix('.pt'))
                        print(f'DDQN {initialization} steps={step} validation={result["utility"]:.3f}',flush=True)
                saved=DoubleDQN.load(path.with_suffix('.pt'))
                sample,_=env.reset(seed=1500)
                if saved.predict(sample)[0] not in np.flatnonzero(sample['mask']):raise ValueError('restored model violates mask')
                model_path=path.with_suffix('.pt')
            else:
                agent=PPO(CandidatePolicy,env,seed=initialization,device='cpu',verbose=0,
                    n_steps=512,batch_size=128,n_epochs=6,gamma=1.,learning_rate=3e-4,ent_coef=.01)
                callback=ValidationCallback(path);agent.learn(PROTOCOL['steps_per_model'],callback=callback)
                history=callback.history;best=callback.best;model_path=path.with_suffix('.zip')
            models.append(dict(algorithm=algorithm,initialization=initialization,file=str(model_path.relative_to(OUT)),
                sha256=sha(model_path),steps=PROTOCOL['steps_per_model'],validation_utility=best,history=history))
            write(OUT/'training.json',dict(models=models,source_sha256=source_hashes(),calibration_sha256=sha(OUT/'calibration.json'),
                protocol=PROTOCOL,environment=dict(conda='sdn',python=platform.python_version(),torch=torch.__version__,device='cpu')))


def offline_reference(requests,prof):
    compiler=Compiler(prof);plans=[compiler.compile(r,k) for r in requests for k in r.allowed]
    times=sorted({t for p in plans for s in p.segments for t in (s.start,s.end)})
    rows=[];bounds=[]
    for r in requests:
        rows.append([float(p.request.id==r.id) for p in plans]);bounds.append(1.)
    for at in times:
        for index,key in enumerate(RESOURCES):
            rows.append([sum(s.usage[index] for s in p.segments if s.start<=at<s.end) for p in plans]);bounds.append(CAPACITIES[key])
        rows.append([1000. if p.kind=='ndn_pin' and p.request.arrival<=at<p.request.end else 0. for p in plans]);bounds.append(4000.)
    result=milp(-np.array([p.request.weight for p in plans]),integrality=np.ones(len(plans)),
        bounds=Bounds(0,1),constraints=LinearConstraint(csr_matrix(rows),-np.inf,np.array(bounds)),options={'time_limit':15.})
    return dict(requests=len(requests),objective=-float(result.fun) if result.fun is not None else None,
        upper_bound=-float(result.mip_dual_bound) if result.mip_dual_bound is not None else None,
        optimal=result.status==0,status=int(result.status),scope='offline model reference only; future requests known, unique per-slice objects, empirical airtime contract')


def evaluate():
    torch.set_num_threads(1);prof=profile();training=json.loads((OUT/'training.json').read_text())
    if training['source_sha256']!=source_hashes() or training['calibration_sha256']!=sha(OUT/'calibration.json'):
        raise ValueError('frozen policies do not match current source/calibration')
    baselines=json.loads((OUT/'baselines.json').read_text())['selected']
    methods=[(m,m,0.,None) for m in ('first_fit','greedy')]
    methods += [(m,m,baselines[m]['price'],None) for m in baselines]
    for record in training['models']:
        path=OUT/record['file']
        if sha(path)!=record['sha256']:raise ValueError('model changed')
        model=DoubleDQN.load(path) if record['algorithm']=='ddqn' else PPO.load(path,device='cpu')
        methods.append((f'{record["algorithm"]}_{record["initialization"]}',record['algorithm'],0.,model))
    rows=[];tasks=[]
    for seed in PROTOCOL['test_seeds']:
        for load in PROTOCOL['loads']:
            for label,method,price,model in methods:
                env=JointEnv(prof,count=PROTOCOL['requests'],load=load)
                row,plans=episode(env,method,seed,price,model,True);row['method']=label
                name=f'test_{seed}_{load:g}_{label}';path=OUT/'inputs'/f'{name}.tsv';write_tasks(path,plans)
                row['case']=name;rows.append(row)
                tasks.append((name,path,seed+PROTOCOL['test_wifi_seed_offset'],0 if seed%2==0 else 5.))
    with ThreadPoolExecutor(4) as pool:packets=list(pool.map(native_case,tasks))
    by_name={x['name']:x for x in packets}
    for row in rows:
        packet=by_name[row['case']]
        if packet['strict_slices']>row['accepted']:raise ValueError('invalid slice denominator')
        row.update(packet_success=packet['strict_slices']/row['requests'],
            admitted_packet_success=packet['strict_slices']/max(row['accepted'],1),
            on_time_fraction=packet['on_time']/max(packet['expected'],1),
            packet_input_sha256=packet['input_sha256'])
    summaries={}
    for label in [m[0] for m in methods]:
        group=[r for r in rows if r['method']==label]
        summaries[label]={k:mean(r[k] for r in group) for k in ('model_acceptance','packet_success','admitted_packet_success','on_time_fraction','decision_ms')}
    intervals={};rng=np.random.default_rng(9721)
    for method in ('ddqn','ppo','greedy','adaptive_price'):
        differences=[]
        for seed in PROTOCOL['test_seeds']:
            base=mean(r['packet_success'] for r in rows if r['seed']==seed and r['method']=='resource_price')
            treatment=mean(r['packet_success'] for r in rows if r['seed']==seed and
                (r['method'].startswith(method+'_') if method in ('ddqn','ppo') else r['method']==method))
            differences.append(treatment-base)
        boot=np.mean(rng.choice(differences,(5000,len(differences)),replace=True),axis=1)
        intervals[method]=dict(mean=mean(differences),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),seeds=len(differences))
    references=[]
    for seed in PROTOCOL['test_seeds'][:2]:
        env=JointEnv(prof,count=8);env.reset(seed=seed)
        references.append(dict(seed=seed,**offline_reference(env.requests,prof)))
    write(OUT/'comparison.json',dict(scope=PROTOCOL['scope'],protocol=PROTOCOL,summary=summaries,
        paired_vs_resource_price=intervals,episodes=rows,packet_cases=packets,offline_references=references,
        source_sha256=source_hashes(),training_sha256=sha(OUT/'training.json'),calibration_sha256=sha(OUT/'calibration.json')))
    print(json.dumps(dict(summary=summaries,paired_vs_resource_price=intervals,packet_runs=len(packets)),ensure_ascii=False,indent=2),flush=True)


def verify():
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('slice_alloc.tests.test_joint_contracts')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/'tests.log').write_text(stream.getvalue())
    if not result.wasSuccessful():raise ValueError(stream.getvalue())
    data=json.loads((OUT/'comparison.json').read_text());cal=json.loads((OUT/'calibration.json').read_text())
    checks=dict(tests_passed=result.wasSuccessful(),source_matches=data['source_sha256']==source_hashes(),
        calibration_passed=cal['profile']['passed'] and cal['profile']['heldout_passed'],
        training_matches=data['training_sha256']==sha(OUT/'training.json'),
        calibration_matches=data['calibration_sha256']==sha(OUT/'calibration.json'))
    files=0
    for row in cal['cases']+data['packet_cases']:
        for path,digest in row['files'].items():
            if sha(OUT/path)!=digest:raise ValueError('raw packet file changed: '+path)
            files+=1
        if sha(OUT/row['input_file'])!=row['input_sha256']:raise ValueError('task manifest changed')
    write(OUT/'verification.json',dict(passed=all(checks.values()),checks=checks,tests=result.testsRun,
        calibration_packet_runs=len(cal['cases']),test_packet_runs=len(data['packet_cases']),
        raw_file_hashes_checked=files,models=6,training_steps=6*PROTOCOL['steps_per_model'],
        comparison_sha256=sha(OUT/'comparison.json'),source_sha256=source_hashes(),
        scope=PROTOCOL['scope'],remaining=PROTOCOL['limitations']))
    if not all(checks.values()):raise ValueError(checks)
    print('Joint adapter verification passed:',json.dumps(checks),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('calibrate','train','evaluate','verify','all'),default='all',nargs='?')
    phase=parser.parse_args().phase
    if phase=='all':
        calibrate();train();evaluate();verify()
    else:globals()[phase]()


if __name__=='__main__':main()
