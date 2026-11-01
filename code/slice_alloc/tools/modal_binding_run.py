"""Frozen, native, service-to-modality study. Training/validation/test disjoint."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from slice_alloc import modal_binding as b
from slice_alloc.distributional_allocation import MaskedDoubleQRDQN,CandidateQRPolicy,NStepDictReplay

ROOT=b.p.ROOT
OUT=ROOT/'slice_alloc/results/modal_binding'
INITS=(11,23,37,53,71)
LOADS=(48,80)
VALIDATION=tuple(range(22000,22008))
TEST=tuple(range(23000,23030))
OOD=tuple(range(24000,24015))
BUDGETS=(8192,16384,32768)
RULES=('first_fit','fixed_price','joint_price')
THRESHOLDS=(.25,.5,1.,2.,4.,8.,None)
SOURCES=('slice_alloc/modal_binding.py','slice_alloc/execution/binding_scenario.cc',
    'slice_alloc/tools/modal_binding_run.py','slice_alloc/distributional_allocation.py',
    'slice_alloc/patrol.py','slice_alloc/initial_allocation.py',
    'slice_alloc/tests/test_modal_binding.py','slice_alloc/design.md')
HYPER=dict(learning_rate=1e-4,buffer_size=20000,learning_starts=1024,batch_size=128,
    train_freq=4,gradient_steps=1,gamma=1.,target_update_interval=1000,
    exploration_initial_eps=1.,exploration_final_eps=.05,exploration_steps=16384,
    max_grad_norm=10.,policy_kwargs=dict(n_quantiles=32,net_arch=[96]),
    replay_buffer_class=NStepDictReplay,replay_buffer_kwargs=dict(horizon=8,discount=1.))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(Path(path).read_text())
def hashes():return {name:sha(ROOT/name) for name in SOURCES}


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(schema=b.PROTOCOL,sources=hashes(),initializations=INITS,loads=LOADS,
        budgets=BUDGETS,validation=VALIDATION,test=TEST,ood=OOD,training_range=[20000000,21000000],
        rules=RULES,threshold_grid=THRESHOLDS,
        checkpoint_selection='highest validation success; tie earliest; every initialization retained',
        baseline_selection='same validation seeds, threshold grid; tie highest threshold',
        reward='sum of actual native request SLA successes; reject counts failure; no shaping',
        primary='SLA successful requests / ALL arrivals',
        M='capacity-normalized marginal reservation seconds / successful request; plus measured wire bytes and PHY TX time',
        V='frozen policies on held-out mixes, unseen load112 and unseen object6/deadline.35 contract',
        claim='no theoretical optimality or guaranteed superiority over cross-modal prices',
        binary_sha256=sha(b.p.NS3/'build/scratch/binding-scenario'))
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==json.loads(json.dumps(protocol))
    else:
        write(OUT/'protocol.json',protocol)
        for name in SOURCES:
            target=OUT/'frozen'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    return protocol


def episode(cfg,method,model=None,prefix=None,threshold=None,fixed=False,motion=True):
    env=b.BindingEnv(fixed=fixed,motion=motion);timings=[]
    try:
        obs,_=env.reset(options=dict(scenario=cfg,prefix=prefix,trace=prefix is not None));done=False;reward=0.
        while not done:
            start=time.perf_counter()
            action=int(model.predict(obs,deterministic=True)[0]) if model else b.baseline(method,env,math.inf if threshold is None else threshold)
            timings.append(time.perf_counter()-start)
            assert obs['mask'][action]>.5
            obs,r,done,_,_=env.step(action);reward+=r
        result=env.result();assert reward==result['success']
        result['decision_ms']=1000*float(np.mean(timings));result['scenario_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        if prefix:
            write(str(prefix)+'-result.json',result)
            raw=Path(str(prefix)+'-events.csv')
            with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
            raw.unlink()
            result.update(audit(prefix,result,cfg))
            write(str(prefix)+'-metrics.json',{k:v for k,v in result.items() if k not in ('decisions','settled','observations')})
        return result
    finally:env.close()


def audit(prefix,result,cfg):
    # Reconstruct expected and on-time counts independently from packet events.
    emitted={};ontime={};completed=set();wire=0;phy=0.;origin=0
    with gzip.open(str(prefix)+'-events.csv.gz','rt') as stream:
        for row in csv.DictReader(stream):
            event=row['event'];sid=int(row['slice']);seq=int(row['seq']);key=(sid,seq)
            if event=='emit':
                assert key not in emitted;emitted[key]=float(row['time'])
            if event=='complete':
                assert key in emitted and key not in completed;completed.add(key)
                request=cfg['requests'][sid-1]
                if float(row['time'])-emitted[key]<=request['deadline']+1e-8:ontime[sid]=ontime.get(sid,0)+1
            if event in ('wire_ip','wire_ndn'):
                wire+=int(row['bytes'])
                if int(row['node'])==0:origin+=int(row['bytes'])
            if event=='phy_tx':phy+=float(row['detail'])
    for s in result['settled']:
        r=cfg['requests'][s['id']-1];expected=sum(sid==s['id'] for sid,seq in emitted)
        assert expected==s['expected']
        assert ontime.get(s['id'],0)==s['on_time']
        assert s['success']==(ontime.get(s['id'],0)+1e-9>=expected*r['minimum_ratio'])
    summary=read(str(prefix)+'-summary.json')
    assert not any(summary[k] for k in ('remaining_mapping','remaining_mapcache','remaining_pending','remaining_application','remaining_cache_entries'))
    return dict(audited=True,wire_bytes=wire,origin_wire_bytes=origin,phy_tx_seconds=phy,
        native_summary=summary)


def score(model,fixed=False,motion=True):
    rows=[]
    for seed in VALIDATION:
        for load in LOADS:
            r=episode(b.scenario(seed,load),'learned',model,fixed=fixed,motion=motion)
            rows.append(dict(seed=seed,load=load,success=r['success'],requests=r['requests']))
    return dict(rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows),rows=rows)


class Checkpoints(BaseCallback):
    def __init__(self,folder,fixed,motion):super().__init__();self.folder=folder;self.fixed=fixed;self.motion=motion;self.records=[]
    def _on_step(self):
        if self.num_timesteps not in BUDGETS:return True
        path=self.folder/f'{self.num_timesteps}.zip';self.model.save(path)
        val=score(self.model,self.fixed,self.motion)
        self.records.append(dict(step=self.num_timesteps,path=str(path.relative_to(ROOT)),sha256=sha(path),validation=val))
        write(self.folder/'progress.json',self.records)
        print('CHECKPOINT',self.folder.parent.name,self.folder.name,self.num_timesteps,round(val['rate'],5),flush=True)
        return True


def train(spec):
    method,init=spec;torch.set_num_threads(1);fixed=method=='fixed_qrdqn';motion=method!='no_motion'
    folder=OUT/'training'/method/str(init);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training.json').exists():return read(folder/'training.json')
    env=b.BindingEnv(seed=init+100,loads=LOADS,fixed=fixed,motion=motion)
    model=MaskedDoubleQRDQN(CandidateQRPolicy,env,**HYPER,seed=init,device='cpu',verbose=0)
    before=model.quantile_net.deployment[-1].weight.detach().clone();cb=Checkpoints(folder,fixed,motion)
    start=time.perf_counter()
    try:
        model.learn(max(BUDGETS),callback=cb)
        change=float(torch.linalg.vector_norm(model.quantile_net.deployment[-1].weight.detach()-before));assert change>0
        best=max(cb.records,key=lambda r:(r['validation']['rate'],-r['step']))
        selected=OUT/'models'/f'{method}-{init}.zip';selected.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/best['path'],selected)
        again=MaskedDoubleQRDQN.load(selected,device='cpu');assert score(again,fixed,motion)==best['validation']
        result=dict(method=method,init=init,steps=model.num_timesteps,updates=model._n_updates,
            weight_change=change,seconds=time.perf_counter()-start,stages=cb.records,selected=best,
            model=str(selected.relative_to(ROOT)),model_sha256=sha(selected),training_seeds=env.episode_specs,
            protocol_sha256=sha(OUT/'protocol.json'))
        write(folder/'training.json',result);print('TRAINED',method,init,flush=True);return result
    finally:env.close()


def tune_rule(method):
    rows=[]
    for threshold in (None,) if method=='first_fit' else THRESHOLDS:
        successes=total=0
        for seed in VALIDATION:
            for load in LOADS:
                r=episode(b.scenario(seed,load),method,threshold=threshold);successes+=r['success'];total+=r['requests']
        rows.append(dict(threshold=threshold,rate=successes/total))
    chosen=max(rows,key=lambda x:(x['rate'],math.inf if x['threshold'] is None else x['threshold']))
    value=dict(method=method,chosen=chosen,grid=rows);write(OUT/'rules'/f'{method}.json',value)
    print('RULE',method,chosen,flush=True);return value


def evaluate_group(spec):
    method,init,scope,mix,load,seeds=spec;torch.set_num_threads(1)
    model=None;threshold=None
    if method not in RULES:model=MaskedDoubleQRDQN.load(OUT/'models'/f'{method}-{init}.zip',device='cpu')
    else:threshold=read(OUT/'rules'/f'{method}.json')['chosen']['threshold']
    rows=[]
    for seed in seeds:
        prefix=OUT/'native'/f'{method}-{init}'/f'{scope}-{mix}-{load}-{seed}'
        metrics=Path(str(prefix)+'-metrics.json')
        if metrics.exists():r=read(metrics)
        else:
            r=episode(b.scenario(seed,load,mix),method,model,prefix,threshold,
                fixed=method=='fixed_qrdqn',motion=method!='no_motion')
        rows.append({k:v for k,v in r.items() if k not in ('decisions','settled','observations')})
        rows[-1].update(method=method,init=init,scope=scope,mix=mix,load=load,seed=seed)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','train','evaluate']);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    torch.set_num_threads(1);freeze()
    if args.command=='train':
        tasks=[(m,i) for m in ('joint_qrdqn','fixed_qrdqn','no_motion') for i in INITS]
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(train,t) for t in tasks]+[pool.submit(tune_rule,m) for m in RULES]
            for future in cf.as_completed(futures):future.result()
    if args.command=='evaluate':
        # A completion manifest locks ALL model and baseline choices before test opens.
        manifests=[read(OUT/'training'/m/str(i)/'training.json') for m in ('joint_qrdqn','fixed_qrdqn','no_motion') for i in INITS]
        write(OUT/'selection.json',dict(models=manifests,rules=[read(OUT/'rules'/f'{m}.json') for m in RULES]))
        choices=[(m,i) for m in ('joint_qrdqn','fixed_qrdqn','no_motion') for i in INITS]+[(m,0) for m in RULES]
        specs=[]
        for method,init in choices:
            for load in LOADS:specs.append((method,init,'test','balanced',load,TEST))
            for mix,load in [('object_heavy',80),('mobile_heavy',80),('balanced',112),('novel',80)]:specs.append((method,init,'ood',mix,load,OOD))
        rows=[]
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for group in pool.map(evaluate_group,specs):rows.extend(group)
        write(OUT/'evaluation.json',rows)
        print('EVALUATED',len(rows),flush=True)

if __name__=='__main__':main()
