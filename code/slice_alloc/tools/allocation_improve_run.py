"""Bounded development -> validation -> fresh test, without changing physics."""
import argparse
import concurrent.futures as cf
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from slice_alloc import allocation_final as base
from slice_alloc import guided_allocation as guided
from slice_alloc.tools import allocation_simple_run as old
from slice_alloc.tools import allocation_simple_report as audit

PACKAGE=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('LOWALT_IMPROVE_OUTPUT',str(PACKAGE/'results/improvement'))).resolve()
INITS=(11,23,37,53,71,89,101,127)
LOADS=(48,80)
DEV=tuple(range(221000,221010))
VALIDATION=tuple(range(222000,222060))
TEST=tuple(range(223000,223060))
GRID=list(itertools.product((0.,.5,1.,2.),(0.,.25,.5),(0.,1.)))
RULES=('greedy','nearest','first_fit','native','modal_greedy')
SOURCES=tuple(dict.fromkeys((*old.SOURCES,'tools/allocation_simple_report.py',
    'guided_allocation.py','tools/allocation_improve_run.py','tests/test_guided_allocation.py')))
write=old.write
read=old.read
sha=old.sha


def parameters():
    p=old.parameters();p.update(learning_rate=3e-5,buffer_size=50000,
        exploration_initial_eps=.1,exploration_final_eps=.02,exploration_steps=32768)
    p['replay_buffer_kwargs']=dict(horizon=32,discount=1.)
    return p


def freeze():
    protocol=dict(source_hashes={n:sha(PACKAGE/n) for n in SOURCES},
        binary_sha256=sha(base.NS3/'build/scratch/allocation-final-scenario'),
        initializations=INITS,loads=LOADS,development=DEV,validation=VALIDATION,test=TEST,
        rule_grid=GRID,budgets=[0,16384,32768,65536],
        physics='unchanged 72f5ec9 low-altitude native multimodal network',
        primary='guided RL minus validation-selected modal-aware greedy',
        selection='all initializations retained; per-init highest validation SLA, tie earlier checkpoint',
        promotion='paired two-way bootstrap lower bound >0; admitted violation rate not worse',
        changes='explicit rule prior, zero residual initialization, epsilon .1 -> .02, n-step32; no reward relabelling',
        training_seed_range=[61000000,62000000],
        rule_selection='24 formulas on development, select best on development; verify on separate validation',
        disclosure='8 RL initializations, 4 checkpoints each incl rule at zero; not a guarantee against rules on test',
        hyperparameters={k:v for k,v in parameters().items() if k!='replay_buffer_class'},
        replay_buffer_class='slice_alloc.distributional_allocation.NStepDictReplay',
        references=['https://ojs.aaai.org/index.php/AAAI/article/view/11757',
                    'https://papers.nips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html'])
    expected=json.loads(json.dumps(protocol))
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==expected,'frozen experiment changed'
    else:write(path,protocol)
    return expected


def check():
    p=read(OUT/'protocol.json')
    assert all(sha(PACKAGE/n)==h for n,h in p['source_hashes'].items())
    assert sha(base.NS3/'build/scratch/allocation-final-scenario')==p['binary_sha256']


class TrainEnv(guided.GuidedEnv):
    def reset(self,seed=None,options=None):
        opts=dict(options or {})
        if 'scenario' not in opts:
            opts['scenario']=base.scenario(int(self.rng.integers(61000000,62000000)),int(self.rng.choice(LOADS)))
        return super().reset(seed=seed,options=opts)


def episode(seed,load,method,parameters,model=None,trace=False,init=0):
    cfg=base.scenario(seed,load)
    prefix=OUT/'native'/f'{method}-{init}'/f'{load}-{seed}' if trace else None
    env=guided.GuidedEnv(rule_parameters=parameters)
    try:
        obs,_=env.reset(options=dict(scenario=cfg,prefix=prefix,trace=trace))
        done=False;reward=0.;choices=[]
        while not done:
            action=int(model.predict(obs,deterministic=True)[0]) if model is not None else guided.rule_action(method,env,parameters)
            assert obs['mask'][action]>.5
            choices.append(action)
            obs,r,done,truncated,_=env.step(action);assert not truncated
            reward+=r
        result=env.result();assert reward==result['success']
        result.update(method=method,init=init,seed=seed,load=load,
                      scenario_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest())
        if trace:
            write(str(prefix)+'-result.json',result)
            raw=Path(str(prefix)+'-events.csv')
            with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
            raw.unlink()
            write(str(prefix)+'-metrics.json',{k:v for k,v in result.items() if k not in ('settled','decisions','observations')})
            return audit.verify_episode(OUT,{k:result[k] for k in ('method','init','seed','load','requests','success','admitted','by_service')})
        return {k:result[k] for k in ('method','init','seed','load','requests','success','admitted','by_service')}
    finally:env.close()


def score(model,parameters,seeds=VALIDATION):
    rows=[episode(s,l,'guided',parameters,model) for s in seeds for l in LOADS]
    return dict(rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows),rows=rows)


def rule_task(spec):
    name,parameters=spec
    rows=[episode(s,l,name,parameters) for s in DEV for l in LOADS]
    return dict(method=name,parameters=parameters,rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows),rows=rows)


def diagnose(workers):
    freeze()
    specs=[('modal_greedy',p) for p in GRID]+[(m,(0.,0.,0.)) for m in RULES if m!='modal_greedy']
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(rule_task,specs))
    selected=max(rows[:24],key=lambda x:x['rate'])
    write(OUT/'diagnosis.json',dict(candidates=rows,selected=selected,
        caveat='best random policy replay is only a feasible lower bound, not an optimality ceiling; no ceiling claim'))
    print('DIAGNOSIS',[(r['method'],r['parameters'],r['rate']) for r in rows],flush=True)


class Checkpoints(BaseCallback):
    def __init__(self,folder,parameters,records):
        super().__init__();self.folder=folder;self.parameters=parameters;self.records=records
    def _on_step(self):
        if self.num_timesteps not in (16384,32768,65536):return True
        path=self.folder/f'{self.num_timesteps}.zip';self.model.save(path)
        value=score(self.model,self.parameters)
        self.records.append(dict(step=self.num_timesteps,path=str(path.relative_to(OUT)),sha256=sha(path),validation=value))
        write(self.folder/'progress.json',self.records)
        print('CHECKPOINT',self.model.seed,self.num_timesteps,value['rate'],flush=True)
        return True


def train(init):
    torch.set_num_threads(1);check()
    rule_parameters=read(OUT/'diagnosis.json')['selected']['parameters']
    folder=OUT/'training'/str(init);folder.mkdir(parents=True,exist_ok=True)
    env=TrainEnv(seed=init+200,loads=LOADS,rule_parameters=rule_parameters)
    params=parameters()
    model=guided.GuidedDQN(guided.PriorPolicy,env,**params,seed=init,device='cpu')
    try:
        zero=folder/'0.zip';model.save(zero)
        # Rule parity is tested at every actual decision on the first validation scene.
        obs,_=env.reset(options={'scenario':base.scenario(VALIDATION[0],LOADS[0])})
        done=False
        while not done:
            action=int(model.predict(obs,deterministic=True)[0])
            assert action==guided.rule_action('modal_greedy',env,rule_parameters)
            obs,_,done,_,_=env.step(action)
        records=[dict(step=0,path=str(zero.relative_to(OUT)),sha256=sha(zero),validation=score(model,rule_parameters))]
        callback=Checkpoints(folder,rule_parameters,records)
        start=time.perf_counter();model.learn(65536,callback=callback)
        assert model._n_updates>0
        selected=max(records,key=lambda x:(x['validation']['rate'],-x['step']))
        target=OUT/'models'/f'guided-{init}.zip';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(OUT/selected['path'],target)
        reload=guided.GuidedDQN.load(target,device='cpu');assert score(reload,rule_parameters)==selected['validation']
        result=dict(init=init,selected=selected,checkpoints=records,steps=model.num_timesteps,
                    updates=model._n_updates,model=str(target.relative_to(OUT)),sha256=sha(target),
                    seconds=time.perf_counter()-start,training_seeds=env.episode_specs,
                    actual_hyperparameters={k:v for k,v in params.items() if k!='replay_buffer_class'})
        write(folder/'training.json',result);print('TRAINED',init,selected['step'],selected['validation']['rate'],flush=True)
        return result
    finally:env.close()


def evaluate_group(spec):
    method,init=spec;torch.set_num_threads(1)
    parameters=read(OUT/'diagnosis.json')['selected']['parameters']
    model=guided.GuidedDQN.load(OUT/'models'/f'guided-{init}.zip',device='cpu') if method=='guided' else None
    return [episode(s,l,method,parameters,model,trace=True,init=init) for s in TEST for l in LOADS]


def report(rows):
    methods=('guided',*RULES);seeds=list(TEST)
    lookup={(r['method'],r['init'],r['seed'],r['load']):r for r in rows}
    rng=np.random.default_rng(20261105)
    sd=rng.integers(len(seeds),size=(20000,len(seeds)));ids=rng.integers(len(INITS),size=(20000,len(INITS)))
    boot={};analysis=dict(methods={},comparisons={},per_initialization={},per_load={})
    for m in methods:
        inits=INITS if m=='guided' else (0,)
        matrix=np.array([[sum(lookup[m,i,s,l]['success'] for l in LOADS)/sum(LOADS) for s in seeds] for i in inits])
        boot[m]=matrix[ids[:,:,None],sd[:,None,:]].mean((1,2)) if m=='guided' else matrix[0,sd].mean(1)
        rr=[r for r in rows if r['method']==m];v=audit.summarize(rr)
        v['violation_rate_admitted']=v['admitted_violations']/v['admitted'];v['ci95']=np.quantile(boot[m],[.025,.975]).tolist()
        v['by_service']={service:{k:sum(r['by_service'][service][k] for r in rr) for k in rr[0]['by_service'][service]} for service in rr[0]['by_service']}
        analysis['methods'][m]=v
        analysis['per_load'][m]={str(l):audit.summarize([r for r in rr if r['load']==l]) for l in LOADS}
        if m=='guided':analysis['per_initialization']={str(i):audit.summarize([r for r in rr if r['init']==i]) for i in INITS}
    for m in RULES:
        diff=boot['guided']-boot[m]
        point=analysis['methods']['guided']['success_rate']-analysis['methods'][m]['success_rate']
        analysis['comparisons']['guided-'+m]=dict(difference=point,ci95=np.quantile(diff,[.025,.975]).tolist(),
            bootstrap_centered_p=float((1+np.count_nonzero(np.abs(diff-point)>=abs(point)))/(len(diff)+1)))
    secondary=[(k,v) for k,v in analysis['comparisons'].items() if k!='guided-modal_greedy']
    adjusted=0.
    for rank,(key,v) in enumerate(sorted(secondary,key=lambda kv:kv[1]['bootstrap_centered_p'])):
        adjusted=max(adjusted,min(1.,(len(secondary)-rank)*v['bootstrap_centered_p']))
        v['holm_adjusted_p']=adjusted
    primary=analysis['comparisons']['guided-modal_greedy'];g=analysis['methods']['guided'];b=analysis['methods']['modal_greedy']
    analysis['promoted']=primary['ci95'][0]>0 and g['violation_rate_admitted']<=b['violation_rate_admitted']
    write(OUT/'analysis.json',analysis)
    print('FINAL',json.dumps(analysis,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('diagnose','train','evaluate'));parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.command=='diagnose':diagnose(args.workers);return
    check()
    if args.command=='train':
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(train,INITS))
        write(OUT/'training_complete.json',rows);return
    assert len(read(OUT/'training_complete.json'))==len(INITS)
    for row in read(OUT/'training_complete.json'):
        assert sha(OUT/row['model'])==row['sha256']
    gate=OUT/'test_started.json'
    assert not gate.exists(),'formal test already opened; no automatic retest'
    write(gate,dict(protocol_sha256=sha(OUT/'protocol.json'),training_sha256=sha(OUT/'training_complete.json'),diagnosis_sha256=sha(OUT/'diagnosis.json')))
    specs=[('guided',i) for i in INITS]+[(m,0) for m in RULES]
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:rows=[r for group in pool.map(evaluate_group,specs) for r in group]
    write(OUT/'evaluation.json',rows);report(rows)


if __name__=='__main__':main()
