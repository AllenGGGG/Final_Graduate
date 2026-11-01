"""Native low-altitude allocation v4: observed motion and marginal resources.

Run as python -m slice_alloc.tools.lowalt_allocation COMMAND. v3 is immutable.
All policy inputs come from the current request, past observations and contracts.
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import copy
import gzip
import json
import math
import tempfile
import time
from pathlib import Path
import numpy as np
from gymnasium import spaces
from slice_alloc import patrol
from slice_alloc.tools import allocation_study as previous
from slice_alloc.tools.patrol_run import sha, write

ROOT=previous.ROOT
OUT=ROOT/'slice_alloc/results/allocation_v4'
LOADS=(64,80)
INITS=(11,23,37,53,71)
STEPS=32768
DEV=tuple(range(7510,7518))
TEST=tuple(range(8500,8530))
METHODS=('proposed','without_motion','plain')
BASELINES=previous.BASELINES+('marginal_price',)
STATE_DIM=previous.STATE_DIM+10
CANDIDATE_DIM=previous.CANDIDATE_DIM
SOURCES=previous.SOURCES+('slice_alloc/tools/lowalt_allocation.py',)


def hashes():return {p:sha(ROOT/p) for p in SOURCES}


def marginal_calendar(ledger,plan):
    """Exact cost of adding this contract, including shared cache union/expiry."""
    start,stop=plan.request.start,plan.request.drain
    points={start,stop}
    for p in ledger.plans+[plan]:
        points.update(t for a,b,_ in p.segments for t in (a,b) if start<t<stop)
        if p.lease:points.update(t for t in p.lease[-2:] if start<t<stop)
    points=sorted(points);integral=np.zeros(len(patrol.RESOURCES));cost=0.
    for a,b in zip(points,points[1:]):
        base=ledger.usage((a+b)/2)
        marginal=(ledger.usage((a+b)/2,plan)-base)/ledger.capacity
        integral+=(b-a)*marginal
        cost+=(b-a)*float(np.dot(marginal,1+4*(base/ledger.capacity)**2))
    return integral,float(cost/(stop-start))


class LowaltEnv(previous.AllocationEnv):
    def __init__(self,*args,motion=True,**kwargs):
        self.motion_enabled=motion
        self.motion_frame=None;self.motion_values={}
        self.previous_positions={};self.previous_time=None;self.previous_aps={}
        self.observed_switches=collections.Counter();self.association_since={}
        super().__init__(*args,**kwargs)
        self.observation_space=spaces.Dict({
            'state':spaces.Box(-10,10,(STATE_DIM,),np.float32),
            'candidates':spaces.Box(-10,10,(patrol.MAX_ACTIONS,CANDIDATE_DIM),np.float32),
            'mask':spaces.Box(0,1,(patrol.MAX_ACTIONS,),np.float32)})

    def reset(self,seed=None,options=None):
        self.motion_frame=None;self.motion_values={};self.previous_positions={}
        self.previous_time=None;self.previous_aps={};self.observed_switches=collections.Counter()
        self.association_since={}
        if options is None or 'scenario' not in options:
            cfg=patrol.scenario(int(self.rng.integers(2000000,3000000)),
                                int(self.rng.choice(self.loads)),'mixed')
            options=dict(scenario=cfg)
        return super().reset(seed=seed,options=options)

    def observe_motion(self):
        now=self.current['time'];key=(now,self.current['request_id'])
        if self.motion_frame==key:return
        dt=now-self.previous_time if self.previous_time is not None else 0.
        values={}
        for node,position in self.current['positions'].items():
            pos=np.asarray(position,float);ap=self.current['associations'][node]
            known=node in self.previous_positions and dt>1e-9
            velocity=(pos-self.previous_positions[node])/dt if known else np.zeros(3)
            if node not in self.previous_aps or ap!=self.previous_aps[node]:
                if node in self.previous_aps:self.observed_switches[node]+=1
                self.association_since[node]=now
            distances=[float(np.linalg.norm(pos-np.array([x,0.,10.]))) for x in (0.,150.)]
            if ap in (3,4):
                radial=pos-np.array([0. if ap==3 else 150.,0.,10.])
                distance=float(np.linalg.norm(radial))
                outward=float(np.dot(velocity,radial)/max(distance,1e-9))
            else:distance=120.;outward=0.
            mission=self.cfg['public']['missions'][node]
            values[node]=np.array([distance/120,min(distances)/120,
                                  float(np.clip(velocity[0]/11,-1,1)),
                                  float(np.clip(outward/11,-1,1)),float(known),
                                  min(10,self.observed_switches[node])/2,
                                  float(mission['role']=='patrol'),float(ap==0),
                                  sum(d<=120 for d in distances)/2,
                                  min(8,now-self.association_since[node])/8],float)
        self.motion_values=values;self.motion_frame=key
        self.previous_positions={n:np.asarray(p,float) for n,p in self.current['positions'].items()}
        self.previous_time=now;self.previous_aps=dict(self.current['associations'])

    def encode(self):
        old=super().encode();state=np.zeros(STATE_DIM,np.float32)
        state[:previous.STATE_DIM]=old['state']
        if self.current['kind']!='done':
            self.observe_motion()
            request=self.requests[self.current['request_id']]
            if self.motion_enabled:
                state[previous.STATE_DIM:]=np.mean([self.motion_values[str(n)] for n in request.members],axis=0)
            for i,p in enumerate(self.options,1):
                integral,cost=marginal_calendar(self.ledger,p)
                old['candidates'][i,patrol.CANDIDATE_DIM:patrol.CANDIDATE_DIM+len(patrol.RESOURCES)]=integral/8
                old['candidates'][i,-3]=cost
        return dict(state=state,candidates=old['candidates'],mask=old['mask'])


def choose(method,env,model,obs):
    if method=='marginal_price':
        if not env.options:return 0
        return min(range(1,len(env.options)+1),key=lambda a:float(obs['candidates'][a,-3]))
    if method in previous.BASELINES:return previous.choose(method,env)
    return int(model.predict(obs,deterministic=True)[0])


def episode(cfg,method,model=None,prefix=None):
    env=previous.AllocationEnv() if method=='previous_ppo' else LowaltEnv(motion=method!='without_motion')
    try:
        obs,_=env.reset(options=dict(scenario=cfg,prefix=prefix,trace=prefix is not None))
        done=False
        while not done:obs,_,done,_,_=env.step(choose(method,env,model,obs))
        result=env.result()
        if prefix is not None:
            write(str(prefix)+'-result.json',result);f=Path(str(prefix)+'-events.csv')
            with gzip.open(str(f)+'.gz','wb',compresslevel=3) as dest:dest.write(f.read_bytes())
            f.unlink()
        return result
    finally:env.close()


def train_one(spec):
    import torch
    from stable_baselines3 import PPO
    from slice_alloc.src.allocation_policy import ResourcePriorPolicy
    method,init,steps=spec;assert method in METHODS;torch.set_num_threads(1)
    env=LowaltEnv(seed=init+100,motion=method!='without_motion')
    start=time.perf_counter()
    model=PPO(ResourcePriorPolicy,env,learning_rate=1e-4,n_steps=512,batch_size=128,
              n_epochs=5,gamma=1.,gae_lambda=1.,ent_coef=.005,clip_range=.1,
              seed=init,device='cpu',verbose=0,
              policy_kwargs=dict(prior_scale=0. if method=='plain' else 4.,reject_prior=-6.))
    model.learn(total_timesteps=steps)
    path=OUT/'models'/f'{method}-{init}.zip';path.parent.mkdir(exist_ok=True,parents=True)
    model.save(path)
    info=dict(method=method,initialization=init,steps=steps,checkpoint=str(path.relative_to(OUT)),
              model_sha256=sha(path),sources=hashes(),seconds=time.perf_counter()-start,
              training_scenarios=env.episode_specs,reward='unscaled native settled SLA success count',
              demonstration_episodes=0,reference_episodes=0,motion=method!='without_motion')
    env.close();write(path.with_suffix('.json'),info)
    print(json.dumps({k:v for k,v in info.items() if k not in ('sources','training_scenarios')}),flush=True)
    return info


def model_path(method,init):
    return previous.OUT/'models'/f'prior_ppo-{init}-{STEPS}.zip' if method=='previous_ppo' else OUT/'models'/f'{method}-{init}.zip'


def eval_group(spec):
    method,init,seeds,trace=spec;model=None
    if method not in BASELINES:
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1);model=PPO.load(model_path(method,init),device='cpu')
    rows=[]
    for seed in seeds:
        for load in LOADS:
            cfg=patrol.scenario(seed,load,'mixed');tag=f'{method}-{init or 0}/mixed-{load}-{seed}'
            prefix=OUT/'native'/tag if trace else None
            result=episode(cfg,method,model,prefix)
            row=dict(method=method,initialization=init,seed=seed,load=load,profile='mixed',
                     requests=result['requests'],success=result['success'],admitted=result['admitted'],
                     rejections=result['rejections'],by_mode=result['by_mode'],native_seconds=result['native_seconds'])
            if trace:
                row.update(path=tag,summary=json.loads(Path(str(prefix)+'-summary.json').read_text()))
                for suffix,key in [('-scenario.json','scenario_sha256'),('-result.json','result_sha256'),
                                   ('-events.csv.gz','events_sha256'),('-summary.json','summary_sha256')]:
                    row[key]=sha(str(prefix)+suffix)
            rows.append(row)
    return rows


def evaluate(methods,inits,seeds,jobs,trace=False):
    specs=[(m,None,seeds,trace) for m in BASELINES]
    specs += [(m,i,seeds,trace) for m in methods for i in inits]
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        for group in pool.map(eval_group,specs):rows.extend(group)
    return rows


def analyze(rows,methods,seeds):
    stats={};matrices={}
    for m in BASELINES+tuple(methods):
        group=[r for r in rows if r['method']==m]
        initializations=sorted({r['initialization'] for r in group},key=lambda i:i or 0)
        matrices[m]=np.array([[sum(r['success'] for r in group if r['initialization']==i and r['seed']==s)/
                              sum(r['requests'] for r in group if r['initialization']==i and r['seed']==s)
                              for s in seeds] for i in initializations])
        rates=matrices[m].mean(1);n=sum(r['requests'] for r in group)
        stats[m]=dict(success_rate=sum(r['success'] for r in group)/n,
                      admission_rate=sum(r['admitted'] for r in group)/n,
                      by_load={str(load):sum(r['success'] for r in group if r['load']==load)/
                               sum(r['requests'] for r in group if r['load']==load) for load in LOADS},
                      by_mode={mode:sum(r['by_mode'][mode]['success'] for r in group)/
                                    sum(r['by_mode'][mode]['requests'] for r in group) for mode in ('ip','id','ndn')},
                      initialization_rates=rates.tolist(),initialization_std=float(np.std(rates,ddof=1)) if len(rates)>1 else None,
                      requests=n,rejections=dict(sum((collections.Counter(r['rejections']) for r in group),collections.Counter())))
    comparisons={};rng=np.random.default_rng(17709)
    for m in methods:
        comparisons[m]={}
        for other in stats:
            if m==other:continue
            a,b=matrices[m],matrices[other];difference=a.mean(0)-b.mean(0)
            boot=rng.choice(difference,(10000,len(seeds)),replace=True).mean(1);nested=[]
            for _ in range(4000):
                ss=rng.integers(0,len(seeds),len(seeds));aa=rng.integers(0,len(a),len(a));bb=rng.integers(0,len(b),len(b))
                nested.append(a[np.ix_(aa,ss)].mean()-b[np.ix_(bb,ss)].mean())
            comparisons[m][other]=dict(difference=float(difference.mean()),paired_scenario_ci95=np.quantile(boot,[.025,.975]).tolist(),
                                      initialization_and_scenario_ci95=np.quantile(nested,[.025,.975]).tolist())
    return dict(stats=stats,comparisons=comparisons,episodes=len(rows),decisions=sum(r['requests'] for r in rows))


def freeze():
    assert not (OUT/'evaluation.json').exists(),'Never revise an opened final test.'
    checkpoints=[]
    for method in METHODS+('previous_ppo',):
        for init in INITS:
            path=model_path(method,init)
            info_path=path.with_suffix('.json');info=json.loads(info_path.read_text())
            if method!='previous_ppo':assert info['sources']==hashes()
            else:assert info['sources']==previous.hashes()
            assert info['steps']==STEPS and info['model_sha256']==sha(path)
            checkpoints.append(dict(method=method,initialization=init,checkpoint=str(path.relative_to(ROOT)),
                                    model_sha256=sha(path),manifest=str(info_path.relative_to(ROOT)),manifest_sha256=sha(info_path)))
    protocol=dict(version='allocation-v4',methods=METHODS+('previous_ppo',),baselines=BASELINES,
                  primary_method='proposed',loads=LOADS,initializations=INITS,steps=STEPS,
                  development_seeds=DEV,test_seeds=TEST,training_seed_range=[2000000,2999999],
                  sources=hashes(),binary_sha256=sha(patrol.NS3/'build/scratch/patrol-scenario'),
                  checkpoints=checkpoints,development_sha256=sha(OUT/'development.json'),
                  design_sha256=sha(ROOT/'slice_alloc/lowalt_design_v4.md'),
                  reward='Unscaled native SLA successes; no demonstrations or reference runs',
                  primary_metric='Successful requests / all arrivals including rejection',
                  hardware_sla_and_workload='identical patrol-v2 physical model and v3 mixed 64/80 contracts',
                  hyperparameters=dict(learning_rate=1e-4,n_steps=512,batch_size=128,n_epochs=5,gamma=1.,gae_lambda=1.,ent_coef=.005,clip_range=.1),
                  frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),reserved_three_part_seeds=list(range(9000,9030)))
    write(OUT/'protocol.json',protocol)


def verify(jobs):
    from slice_alloc.tools import patrol_verify as auditor
    from stable_baselines3 import PPO
    protocol=json.loads((OUT/'protocol.json').read_text());data=json.loads((OUT/'evaluation.json').read_text())
    assert protocol['sources']==hashes()==data['sources']
    assert protocol['binary_sha256']==sha(patrol.NS3/'build/scratch/patrol-scenario')
    assert protocol['design_sha256']==sha(ROOT/'slice_alloc/lowalt_design_v4.md')
    assert data['protocol_sha256']==sha(OUT/'protocol.json')
    auditor.OUT=OUT
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:raw=list(pool.map(auditor.check_episode,data['rows']))
    inputs=collections.defaultdict(set)
    for r in data['rows']:inputs[r['seed'],r['load']].add(r['scenario_sha256'])
    assert len(inputs)==len(TEST)*len(LOADS) and all(len(x)==1 for x in inputs.values())
    replays=[]
    for c in protocol['checkpoints']:
        assert sha(ROOT/c['checkpoint'])==c['model_sha256'] and sha(ROOT/c['manifest'])==c['manifest_sha256']
        row=next(r for r in data['rows'] if r['method']==c['method'] and r['initialization']==c['initialization'])
        prefix=OUT/'native'/row['path'];original=json.loads(Path(str(prefix)+'-result.json').read_text())
        cfg=json.loads(Path(str(prefix)+'-scenario.json').read_text());model=PPO.load(ROOT/c['checkpoint'],device='cpu')
        with tempfile.TemporaryDirectory(prefix='lowalt-v4-replay-') as tmp:
            target=Path(tmp)/'episode';again=episode(cfg,c['method'],model,target)
            for k in ('decisions','observations','settled'):assert original[k]==again[k]
            with gzip.open(str(prefix)+'-events.csv.gz','rb') as a,gzip.open(str(target)+'-events.csv.gz','rb') as b:assert a.read()==b.read()
        replays.append(dict(method=c['method'],initialization=c['initialization'],exact_events=True))
    write(OUT/'raw_reconstruction.json',raw)
    write(OUT/'audit.json',dict(passed=True,sources=hashes(),same_inputs=True,raw_packet_sla=True,
          episodes=len(data['rows']),decisions=sum(r['requests'] for r in data['rows']),exact_native_policy_replays=replays,
          artifacts={name:sha(OUT/name) for name in ('protocol.json','evaluation.json','analysis.json','raw_reconstruction.json')}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['train','dev','freeze','evaluate','analyze','verify'])
    parser.add_argument('--methods',nargs='+',default=list(METHODS));parser.add_argument('--inits',nargs='+',type=int,default=list(INITS))
    parser.add_argument('--steps',type=int,default=STEPS);parser.add_argument('--jobs',type=int,default=10);args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if args.command=='train':
        assert not (OUT/'protocol.json').exists(),'Frozen study cannot retrain.'
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            records=list(pool.map(train_one,[(m,i,args.steps) for m in args.methods for i in args.inits]))
        write(OUT/'training.json',records)
    elif args.command=='dev':
        rows=evaluate(tuple(args.methods)+('previous_ppo',),args.inits,DEV,args.jobs)
        write(OUT/'development.json',dict(sources=hashes(),rows=rows))
        for method in BASELINES+tuple(args.methods)+('previous_ppo',):
            group=[r for r in rows if r['method']==method]
            print(method,100*sum(r['success'] for r in group)/sum(r['requests'] for r in group),flush=True)
    elif args.command=='freeze':freeze()
    elif args.command=='evaluate':
        p=json.loads((OUT/'protocol.json').read_text());assert p['sources']==hashes()
        assert not (OUT/'evaluation.json').exists(),'Do not repeat an opened final test.'
        rows=evaluate(p['methods'],p['initializations'],p['test_seeds'],args.jobs,True)
        write(OUT/'evaluation.json',dict(protocol_sha256=sha(OUT/'protocol.json'),sources=hashes(),rows=rows))
    elif args.command=='analyze':
        data=json.loads((OUT/'evaluation.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
        assert data['protocol_sha256']==sha(OUT/'protocol.json') and p['sources']==hashes()
        result=analyze(data['rows'],p['methods'],p['test_seeds']);result.update(evaluation_sha256=sha(OUT/'evaluation.json'),protocol_sha256=sha(OUT/'protocol.json'))
        write(OUT/'analysis.json',result)
        print(json.dumps(result['stats'],ensure_ascii=False,indent=2),flush=True)
    else:verify(args.jobs)


if __name__=='__main__':main()
