"""Build, execute, train and independently audit the NEW three-modality platform.

python -B -m slice_alloc.tools.trimodal_run all
Old four-node results are never inputs to new platform scores.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from ..trimodal import (Request, Ledger, compile_plan, candidates, TrimodalEnv,
    MODES, RESOURCE_KEYS, CAPACITIES, write_tasks, select_rule, validate_metadata, request_digest)
from .platform_verify import ROOT, NS3, sha

OUT = ROOT/'slice_alloc/results/trimodal'
PYTHON = '/home/allen/miniconda3/envs/sdn/bin/python'
SOURCES = ('slice_alloc/trimodal.py', 'slice_alloc/tools/trimodal_run.py',
    'slice_alloc/execution/trimodal_scenario.cc', 'slice_alloc/src/qlearning.py',
    'slice_alloc/src/policy.py', 'slice_alloc/tests/test_trimodal.py')
PROTOCOL = dict(schema='trimodal-experiment-v1', uavs=12, main_scale_reason='4/12-UAV native timing measured; 12-UAV median about 0.2s permits full held-out matrix',
    loads=[.5, 1., 1.5], request_count=24, inits=[11,12,13,14,15],
    train_seeds=list(range(101,133)), validation_seeds=list(range(2001,2011)),
    test_seeds=list(range(4001,4031)), radio_repeats=[0,1],
    ddqn_steps=8192, ppo_steps=8192, max_candidates=16,
    methods=['firstfit','greedy','price','coverage','sequential','DDQN','PPO'],
    training_objective='model admission proxy; actual native replay determines all-request strict SLA',
    sla='all generated packets/objects delivered within 0.8s; all offered requests in denominator',
    speed_test=[5.,8.,11.], start_x_test=[-10.,0.,10.],
    forbidden='no real future trajectories in candidate generation/training observations',
    stats='paired bootstrap over 30 independent scenario seeds; average radio repetitions and inits within each scenario',
    note='restricted experimental EID/RLOC uses 25-byte header, not interoperable LISP; fixed period contracts; one AP handover envelope')


def write(path, value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sources():
    return {p:sha(ROOT/p) for p in SOURCES}


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'slice_alloc/execution/trimodal_scenario.cc'
    shutil.copy2(source,NS3/'scratch/trimodal-scenario.cc')
    t=time.perf_counter()
    p=subprocess.run(['/usr/bin/python3','waf','-j4','--targets=trimodal-scenario'],cwd=NS3,capture_output=True,text=True)
    (OUT/'build.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise RuntimeError('native build failed; see results/trimodal/build.log')
    return dict(seconds=time.perf_counter()-t,source_sha256=sha(source),binary_sha256=sha(NS3/'build/scratch/trimodal-scenario'))


def native(spec):
    name,plans,seed,uavs,speed,start_x,negative=spec
    prefix=OUT/'packets'/name;prefix.parent.mkdir(parents=True,exist_ok=True)
    taskpath=OUT/'inputs'/f'{name}.tsv';write_tasks(taskpath,plans)
    finish=max((p.request.drain for p in plans),default=20.)+1
    cmd=[str(NS3/'build/scratch/trimodal-scenario'),f'--output={prefix}',f'--tasks={taskpath}',
         f'--seed={seed}',f'--uavs={uavs}',f'--speed={speed}',f'--startX={start_x}',
         f'--stop={finish}',f'--negative={int(negative)}']
    t=time.perf_counter();env=dict(os.environ,LD_LIBRARY_PATH=str(NS3/'build/lib'))
    p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120)
    elapsed=time.perf_counter()-t
    Path(str(prefix)+'-process.log').write_text(p.stdout+p.stderr)
    if p.returncode:raise RuntimeError(f'native case failed: {name}: {p.stderr[-1200:]}')
    rows=list(csv.DictReader(Path(str(prefix)+'-events.csv').open()))
    summary=json.loads(Path(str(prefix)+'-summary.json').read_text())
    byid={p.request.id:p for p in plans}
    emits={(int(r['slice']),int(r['seq'])):float(r['time']) for r in rows if r['event']=='emit'}
    completions={}
    for r in rows:
        if r['event']=='complete':
            k=(int(r['slice']),int(r['seq']))
            if k not in emits:raise AssertionError('unmatched completion')
            if k in completions:raise AssertionError('duplicate completion')
            if abs(float(r['time'])-float(r['detail'])-emits[k])>1e-7:raise AssertionError('birth/latency mismatch')
            completions[k]=float(r['detail'])
    groups={}
    for sid,plan in byid.items():
        expected=[k for k in emits if k[0]==sid]
        ontime=sum(k in completions and completions[k]<=plan.request.deadline+1e-9 for k in expected)
        groups[str(sid)]=dict(mode=plan.request.mode,expected=len(expected),delivered=sum(k in completions for k in expected),
            on_time=ontime,success=ontime==len(expected),edge=plan.edge,server=plan.server,cache=plan.cache,
            latencies=[completions[k] for k in expected if k in completions])
    # Independent same-layer decoder: emitted IPv4 datagrams must include
    # IPv4(20)+UDP(8)+actual custom Header(25), plus the declared data payload.
    iprows=[r for r in rows if r['event'].startswith('ip_tx:')]
    bytecheck=all(int(r['bytes'])==({'4':1090,'5':1037,'7':1053}.get(r['event'].split(':')[1],53)) for r in iprows)
    emitted=sum(p.request.count for p in plans)
    checks=dict(all_declared_emissions=summary['generated']==len(emits)==emitted,
        all_deliveries_classified=summary['delivered']==len(completions),
        ipv4_udp_custom_header_bytes_exact=bytecheck,
        release_all_slices=sum(r['event']=='release' for r in rows)==len(plans),
        no_state_after_drain=summary['remaining_mappings']==summary['remaining_caches']==summary['remaining_pending']==summary['remaining_control']==summary['remaining_app_pending']==summary['remaining_slice_fib']==0,
        correct_query_server=all(int(r['node'])==byid[int(r['slice'])].server for r in rows if r['event']=='udp_receive' and r['detail']=='2'),
        all_id_packets_decapsulated=sum(r['event']=='decap' for r in rows)==sum(r['event']=='udp_receive' and r['detail']=='4' for r in rows),
        correct_data_tunnel=all(int(r['node'])==byid[int(r['slice'])].edge for r in rows if r['event']=='encap'),
        no_namespace_loss_without_injection=negative or not any(r['event']=='namespace_drop' for r in rows))
    # PHY clocks are independent: packet begin/end vs PHY TX state duration.
    active={};intervals=[]
    for r in rows:
        k=(r['node'],r['seq'])
        if r['event']=='phy_begin':
            if k in active:raise AssertionError('overlapping packet TX clock')
            active[k]=float(r['time'])
        if r['event']=='phy_end':
            if k not in active:raise AssertionError('unmatched PHY TX end')
            intervals.append(float(r['time'])-active.pop(k))
    clocks=sum(float(r['detail']) for r in rows if r['event']=='phy_tx')
    checks['independent_phy_clocks_match']=not active and abs(sum(intervals)-clocks)<1e-7
    interfaces={(r['node'],r['seq']):r['bytes'] for r in rows if r['event']=='interface'}
    ip_expected=Counter()
    for r in iprows:
        peer=interfaces.get((r['node'],r['detail']))
        if peer is not None:ip_expected[(r['node'],peer)]+=int(r['bytes'])
    ip_wire=Counter()
    for r in rows:
        if r['event']=='wire_ip':ip_wire[(r['node'],r['seq'])]+=int(r['bytes'])
    checks['ipv4_per_link_independent_trace_bytes_match']=ip_expected==ip_wire
    checks['no_undeclared_budget_overrun']=not any(r['event']=='budget_drop' for r in rows)
    # Check model state ceilings against actual execution peaks.
    checks['authoritative_and_cache_capacity']=all(int(r['bytes'])<=12 and float(r['detail'])<=12 for r in rows if r['event']=='map_state')
    associations=[dict(time=float(r['time']),uav=int(r['node']),ap=int(r['bytes']),version=int(r['seq']),locator=int(float(r['detail']))) for r in rows if r['event']=='associate']
    ndn_sizes={kind:sorted({int(r['bytes']) for r in rows if r['event']==kind and int(r['slice']) in byid}) for kind in ('ndn_interest','ndn_data')}
    result=dict(name=name,seed=seed,uavs=uavs,speed=speed,start_x=start_x,seconds=elapsed,
        checks=checks,passed=all(checks.values()),slices=groups,summary=summary,associations=associations,
        events=len(rows),ndn_sizes=ndn_sizes,
        native_ndn_source_data=sum(r['event']=='ndn_data' and r['node']=='0' and r['detail']!='999' for r in rows),
        cache_entries=[dict(node=int(r['node']),slice=int(r['slice']),bytes=int(r['bytes'])) for r in rows if r['event']=='cache_entry'],
        counts=dict(Counter(r['event'] for r in rows)),
        udp_bytes_by_message=dict(Counter({k:sum(int(r['bytes']) for r in rows if r['event']=='udp_send' and r['detail']==k)
                                     for k in sorted({r['detail'] for r in rows if r['event']=='udp_send'})})),
        phy_tx_seconds=sum(float(r['detail']) for r in rows if r['event']=='phy_tx'),
        ipv4_transmitted_bytes=sum(int(r['bytes']) for r in iprows),
        files={str(x.relative_to(OUT)):sha(x) for x in prefix.parent.glob(name+'-*')},
        task_file=str(taskpath.relative_to(OUT)),task_sha256=sha(taskpath))
    if not result['passed']:raise AssertionError(f'{name} accounting/install failed {checks}')
    return result


def mechanisms():
    platform=build();specs=[]
    for mode in MODES:
        for edge in (1,2):
            for server in ((1,2) if mode=='id' else (edge,)):
                for cache in ((0,edge) if mode=='ndn' else (0,)):
                    r=Request(1,1,mode,5,2.,18.,.5,2.)
                    p=compile_plan(r,edge,server,cache)
                    # Mechanism trials isolate execution, not admission capacity.
                    for motion,speed in (('static',0.),('handover',8.)):
                        name=f'mechanism_{mode}_{edge}_{server}_{cache}_{motion}'
                        specs.append((name,[p],301,4,speed,0.,False))
    # Multi-slice same EID across tenants and modality coexistence.
    ps=[compile_plan(Request(i+1,i%2+1,m,5 if i<4 else 6,2.,18.,.5,2.),e,s,c)
        for i,(m,e,s,c) in enumerate([('ip',1,1,0),('id',1,2,0),('id',2,1,0),('ndn',1,1,1),('ndn',2,2,2)])]
    specs.append(('mechanism_coexist_negative',ps,302,4,8.,0.,True))
    for n in (4,12):
        ps=[compile_plan(Request(i+1,i%3+1,MODES[i%3],5+i,2.,18.,.5,2.),1+i%2,1+i%2,0) for i in range(n)]
        for seed in (303,304,305):specs.append((f'cost_{n}_{seed}',ps,seed,n,8.,0.,False))
    started=time.perf_counter()
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(native,specs))
    checks={}
    for row in results:
        if row['name'].startswith('mechanism_'):
            moving='handover' in row['name'] or 'coexist' in row['name']
            aa=[a for a in row['associations'] if a['uav']==5]
            if moving:
                checks[row['name']+'_actual_ap_change']=len({a['ap'] for a in aa})==2 and len({a['locator'] for a in aa})==2
                checks[row['name']+'_same_endpoint']=all(a['uav']==5 for a in aa)
            checks[row['name']+'_all_mechanism_tasks_delivered']=all(g['success'] for g in row['slices'].values())
            if 'id' in row['name']:
                checks[row['name']+'_query_executed']=row['counts'].get('query',0)>0
            if 'ndn' in row['name']:
                plan=specs[[s[0] for s in specs].index(row['name'])][1][0]
                checks[row['name']+'_cache_location']=all(e['node']==plan.cache for e in row['cache_entries'])
                if plan.cache:
                    checks[row['name']+'_cache_reuses_object']=row['native_ndn_source_data']<row['summary']['delivered']
                else:
                    checks[row['name']+'_cache_disabled']=not row['cache_entries']
    neg=next(r for r in results if r['name']=='mechanism_coexist_negative')
    for e in ('namespace_drop','stale_register','late_drop'):
        checks['negative_'+e]=neg['counts'].get(e,0)>0
    data=next(r for r in results if r['name']=='mechanism_ndn_1_1_0_static')
    profile=dict(ndn_data_bytes=min(data['ndn_sizes']['ndn_data']),ndn_interest_bytes=min(data['ndn_sizes']['ndn_interest']),
                 tx_seconds_per_datagram=.0025,scope='conservative first version analytical TX allowance; independent packet replay reports violations')
    costs={str(n):dict(median_seconds=float(np.median([r['seconds'] for r in results if r['name'].startswith(f'cost_{n}_')])),
                      p95_seconds=float(np.percentile([r['seconds'] for r in results if r['name'].startswith(f'cost_{n}_')],95))) for n in (4,12)}
    # Functional candidate count, measured after the actual generator exists.
    env=TrimodalEnv(profile=profile);env.reset(seed=1);counts=[]
    while env.index<len(env.requests):
        counts.append(env.current_log);env.step(select_rule(env,'price'))
    write(OUT/'mechanisms.json',dict(passed=all(checks.values()),checks=checks,platform=platform,results=results,
        profile=profile,costs=costs,candidate_measurements=counts,wall_seconds=time.perf_counter()-started,sources=sources()))
    write(OUT/'profile.json',profile)
    if not all(checks.values()):raise AssertionError('mechanism acceptance failed; see mechanisms.json')
    # Small main workload is an intentional scope, not a claim that 12 is impossible.
    write(OUT/'protocol.json',PROTOCOL)
    return results


def profile():return json.loads((OUT/'profile.json').read_text())


def run_policy(method,scenario_seed,load,model=None,init=None):
    env=TrimodalEnv(uavs=PROTOCOL['uavs'],profile=profile())
    obs,_=env.reset(options=dict(scenario_seed=scenario_seed,load=load));beg=time.perf_counter()
    while env.index<len(env.requests):
        a=model.predict(obs,deterministic=True)[0] if model else select_rule(env,method)
        obs,_,_,_,_=env.step(int(a))
    return dict(method=method,init=init,scenario_seed=scenario_seed,load=load,
        requests=len(env.requests),admitted=len(env.ledger.plans),request_sha256=request_digest(env.requests),
        model_acceptance=len(env.ledger.plans)/len(env.requests),decision_ms=1000*(time.perf_counter()-beg)/len(env.requests),
        decisions=[dict(request=asdict(r),selected=p.key if p else None) for r,p in env.decisions],
        candidate_stats=env.candidate_log),env.ledger.plans


def train():
    import torch
    from stable_baselines3 import PPO
    from ..src.qlearning import DoubleDQN
    from ..src.policy import CandidatePolicy
    torch.set_num_threads(1)
    logs=[]
    for init in PROTOCOL['inits']:
        for method in ('DDQN','PPO'):
            env=TrimodalEnv(seeds=PROTOCOL['train_seeds'],uavs=PROTOCOL['uavs'],profile=profile())
            modeldir=OUT/'models'/f'{method}_{init}';modeldir.mkdir(parents=True,exist_ok=True)
            started=time.perf_counter()
            if method=='DDQN':
                model=DoubleDQN(env.state_dim,env.candidate_dim,seed=init)
                obs,_=env.reset(seed=init);loss=[]
                for step in range(PROTOCOL['ddqn_steps']):
                    epsilon=max(.08,1.-step/(PROTOCOL['ddqn_steps']*.75))
                    a=model.explore(obs,epsilon);nxt,reward,done,_,_=env.step(a)
                    model.remember(obs,a,reward,nxt,done);obs=nxt
                    if len(model.replay)>=128 and step%4==0:loss.append(model.update(128))
                    if done:obs,_=env.reset()
                model.save(modeldir/'model.pt')
                detail=dict(updates=model.updates,mean_loss=float(np.mean(loss[-100:])))
            else:
                model=PPO(CandidatePolicy,env,seed=init,device='cpu',n_steps=512,batch_size=128,
                          n_epochs=5,learning_rate=3e-4,verbose=0)
                model.learn(PROTOCOL['ppo_steps']);model.save(modeldir/'model')
                detail=dict(timesteps=model.num_timesteps)
            write(modeldir/'metadata.json',env.metadata_contract)
            validation=[run_policy(method,s,l,model,init)[0] for s in PROTOCOL['validation_seeds'] for l in PROTOCOL['loads']]
            row=dict(method=method,init=init,seconds=time.perf_counter()-started,**detail,
                     validation_mean_acceptance=float(np.mean([r['model_acceptance'] for r in validation])),validation=validation)
            logs.append(row);write(OUT/'training.json',dict(rows=logs,protocol_sha256=sha(OUT/'protocol.json'),sources=sources()))
            print(f'{method} init={init} trained {row["seconds"]:.1f}s; validation model acceptance={row["validation_mean_acceptance"]:.3f}',flush=True)
    return logs


def evaluate():
    import torch
    from stable_baselines3 import PPO
    from ..src.qlearning import DoubleDQN
    torch.set_num_threads(1)
    specs=[];decisions=[]
    for method in PROTOCOL['methods']:
        for init in (PROTOCOL['inits'] if method in ('DDQN','PPO') else [None]):
            model=None
            if init is not None:
                d=OUT/'models'/f'{method}_{init}'
                validate_metadata(d/'metadata.json',TrimodalEnv(uavs=PROTOCOL['uavs'],profile=profile()))
                model=DoubleDQN.load(d/'model.pt') if method=='DDQN' else PPO.load(d/'model',device='cpu')
            for scenario in PROTOCOL['test_seeds']:
                for load in PROTOCOL['loads']:
                    row,plans=run_policy(method,scenario,load,model,init)
                    for repeat in PROTOCOL['radio_repeats']:
                        name=f'test_{method}_{init}_{scenario}_{load}_{repeat}'
                        # Identical physical environment across methods and initializations.
                        rng=np.random.default_rng(scenario)
                        speed=float(rng.choice(PROTOCOL['speed_test']));start=float(rng.choice(PROTOCOL['start_x_test']))
                        specs.append((name,plans,scenario*2+repeat,PROTOCOL['uavs'],speed,start,False))
                        decisions.append(dict(row,name=name,radio_repeat=repeat))
    write(OUT/'decisions.json',dict(protocol_sha256=sha(OUT/'protocol.json'),rows=decisions,sources=sources()))
    packet=[];beg=time.perf_counter()
    with ThreadPoolExecutor(4) as pool:
        for i,r in enumerate(pool.map(native,specs)):
            packet.append(r)
            if (i+1)%100==0:
                print(f'Native held-out replay {i+1}/{len(specs)} elapsed={time.perf_counter()-beg:.1f}s',flush=True)
                write(OUT/'evaluation.partial.json',dict(rows=packet))
    write(OUT/'evaluation.json',dict(rows=packet,wall_seconds=time.perf_counter()-beg,sources=sources(),protocol_sha256=sha(OUT/'protocol.json')))
    (OUT/'evaluation.partial.json').unlink(missing_ok=True)
    return packet


def analyze():
    decisions=json.loads((OUT/'decisions.json').read_text())['rows']
    packets={r['name']:r for r in json.loads((OUT/'evaluation.json').read_text())['rows']}
    merged=[]
    for d in decisions:
        r=packets[d['name']];success=sum(g['success'] for g in r['slices'].values())
        modes={m:dict(requests=sum(x['request']['mode']==m for x in d['decisions']),
                     success=sum(g['success'] for g in r['slices'].values() if g['mode']==m)) for m in MODES}
        merged.append(dict(method=d['method'],init=d['init'],scenario=d['scenario_seed'],load=d['load'],
            repeat=d['radio_repeat'],strict_success=success/d['requests'],admission=d['model_acceptance'],
            post_admission_failures=d['admitted']-success,decision_ms=d['decision_ms'],modes=modes,
            phy_tx_seconds=r['phy_tx_seconds'],name=r['name']))
    scores={}
    scenario_scores={}
    for method in PROTOCOL['methods']:
        rows=[r for r in merged if r['method']==method]
        scores[method]=dict(strict_success=float(np.mean([r['strict_success'] for r in rows])),
            admission=float(np.mean([r['admission'] for r in rows])),
            per_load={str(l):float(np.mean([r['strict_success'] for r in rows if r['load']==l])) for l in PROTOCOL['loads']},
            per_mode={m:float(np.mean([r['modes'][m]['success']/r['modes'][m]['requests'] for r in rows])) for m in MODES},
            mean_decision_ms=float(np.mean([r['decision_ms'] for r in rows])),
            post_admission_failures=sum(r['post_admission_failures'] for r in rows),
            total_offered_decisions=len(rows)*PROTOCOL['request_count'])
        inits=[float(np.mean([r['strict_success'] for r in rows if r['init']==i])) for i in PROTOCOL['inits']] if method in ('DDQN','PPO') else []
        scores[method]['all_initializations']=inits
        if inits:scores[method]['initialization_sample_std_pp']=100*float(np.std(inits,ddof=1))
        scenario_scores[method]=np.array([np.mean([r['strict_success'] for r in rows if r['scenario']==s]) for s in PROTOCOL['test_seeds']])
    rng=np.random.default_rng(9801);comparisons=[]
    for a in ('DDQN','PPO'):
        for b in PROTOCOL['methods']:
            if a==b:continue
            diff=100*(scenario_scores[a]-scenario_scores[b])
            boots=np.mean(rng.choice(diff,(10000,len(diff)),replace=True),axis=1)
            comparisons.append(dict(a=a,b=b,mean_pp=float(np.mean(diff)),ci95_pp=np.quantile(boots,[.025,.975]).tolist()))
    result=dict(scores=scores,paired_comparisons=comparisons,scenario_unit_count=len(PROTOCOL['test_seeds']),
                packet_replays=len(packets),all_packet_accounting_passed=all(r['passed'] for r in packets.values()),rows=merged,
                protocol_sha256=sha(OUT/'protocol.json'),sources=sources())
    write(OUT/'analysis.json',result)
    text=['# 三模态平台：实际独立评测结果','',
        '本结果来自新双AP/双边缘原生执行器，与旧四节点成绩分开。训练奖励是模型接纳代理，以下成功率按全部申请、完整切片包级SLA计算。',
        '',f'固定规模：12架无人机；5个训练初始化；30个独立业务/轨迹种子；3档负载；每场景2次无线重复；共{len(packets)}次包级重放。',
        '', '| 方法 | 完整切片成功率 | 模型接纳率 | IP | ID | NDN |', '|---|---:|---:|---:|---:|---:|']
    for m,s in scores.items():
        text.append('| '+m+' | '+' | '.join(f'{100*v:.2f}%' for v in [s['strict_success'],s['admission'],*(s['per_mode'][x] for x in MODES)])+' |')
    text+=['','配对区间以30个独立场景为重采样单位；先在场景内平均无线重复、负载和初始化，不把重复观测当独立样本。','']
    for c in comparisons:
        if c['a']=='DDQN':text.append(f'- DDQN−{c["b"]}：{c["mean_pp"]:+.2f} 个百分点，95%区间 [{c["ci95_pp"][0]:.2f}, {c["ci95_pp"][1]:.2f}]。')
    text+=['','不预设RL优于规则；模型接纳与真实SLA的差异、全部初始化和失败均保留在analysis.json。自定义ID头为25B，另加UDP/IP 28B；不是LISP标准互操作实现。',
           '资源预约由公开业务合同和允许接入集合计算，不读取真实未来轨迹；空口额度是有限场景的保守模型，实际PHY发送时间另存，未声称普适无线硬保证。','']
    (OUT/'report.md').write_text('\n'.join(text))
    return result


def audit():
    expected={}
    for n in ('mechanisms.json','training.json','decisions.json','evaluation.json','analysis.json'):
        d=json.loads((OUT/n).read_text())
        if d['sources']!=sources():raise AssertionError(f'stale source fingerprints: {n}')
        expected[n]=sha(OUT/n)
    mechanisms_data=json.loads((OUT/'mechanisms.json').read_text())
    if not mechanisms_data['passed']:raise AssertionError('mechanisms failed')
    if sha(NS3/'build/scratch/trimodal-scenario')!=mechanisms_data['platform']['binary_sha256']:raise AssertionError('native executable changed')
    evaluations=json.loads((OUT/'evaluation.json').read_text())['rows']
    rows=mechanisms_data['results']+evaluations
    for r in rows:
        if not r['passed']:raise AssertionError(r['name'])
        for name,value in r['files'].items():
            if sha(OUT/name)!=value:raise AssertionError('raw file fingerprint mismatch: '+name)
        if sha(OUT/r['task_file'])!=r['task_sha256']:raise AssertionError('task file fingerprint mismatch')
    protocol=json.loads((OUT/'protocol.json').read_text())
    if protocol!=PROTOCOL:raise AssertionError('frozen protocol changed')
    test=set(protocol['test_seeds'])
    if test&set(protocol['train_seeds']) or test&set(protocol['validation_seeds']):raise AssertionError('seed leakage')
    decisions=json.loads((OUT/'decisions.json').read_text())['rows']
    keys={(r['method'],r['init'],r['scenario_seed'],r['load'],r['radio_repeat']) for r in decisions}
    target=(5+2*len(PROTOCOL['inits']))*len(test)*len(PROTOCOL['loads'])*len(PROTOCOL['radio_repeats'])
    if len(keys)!=len(decisions) or len(keys)!=target:raise AssertionError('incomplete experimental matrix')
    shared={}
    for r in decisions:
        k=r['scenario_seed'],r['load']
        if k in shared and shared[k]!=r['request_sha256']:raise AssertionError('unpaired workload')
        shared[k]=r['request_sha256']
        if any(x['truncated'] or x['search_limit'] for x in r['candidate_stats']):raise AssertionError('unexpected candidate truncation')
    for m in ('DDQN','PPO'):
        for i in PROTOCOL['inits']:
            d=OUT/'models'/f'{m}_{i}';validate_metadata(d/'metadata.json',TrimodalEnv(uavs=PROTOCOL['uavs'],profile=profile()))
            for p in d.iterdir():expected[str(p.relative_to(OUT))]=sha(p)
    test_run=subprocess.run([PYTHON,'-B','-m','unittest','discover','-s','slice_alloc/tests','-q'],cwd=ROOT,
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),capture_output=True,text=True)
    (OUT/'tests.log').write_text(test_run.stdout+test_run.stderr)
    if test_run.returncode:raise AssertionError('regression tests failed; see tests.log')
    expected['tests.log']=sha(OUT/'tests.log');expected['report.md']=sha(OUT/'report.md');expected['protocol.json']=sha(OUT/'protocol.json')
    result=dict(passed=True,sources=sources(),artifacts=expected,packet_cases=len(rows),evaluation_cases=len(evaluations),
                paired_scenario_count=len(test),all_matrix_entries_present=True,all_raw_files_match=True,
                training_testing_disjoint=True,model_metadata_match=True,tests_passed=True,
                scope='new three-modality dual-AP platform; analytical reservations plus independent packet SLA evaluation')
    write(OUT/'final_audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['mechanisms','train','evaluate','analyze','audit','all']);a=p.parse_args()
    for name in (['mechanisms','train','evaluate','analyze','audit'] if a.command=='all' else [a.command]):
        result=globals()[name]()
        print(name,'completed',flush=True)

if __name__=='__main__':main()
