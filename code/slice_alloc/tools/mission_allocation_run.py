"""Mission workload: freeze -> baseline/development -> masked PPO validation.

All native links and packet contracts are unchanged. No formal test is opened
by this tool's preparation or training commands.
"""
import argparse
import concurrent.futures as cf
import itertools
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from slice_alloc import mission_allocation as mission
from slice_alloc.mission_policy import MissionPolicy
from slice_alloc.tools import allocation_simple_run as old

PACKAGE=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('LOWALT_MISSION_OUTPUT',str(PACKAGE/'results/mission'))).resolve()
INITS=(11,23,37,53,71,89,101,127)
LOADS=(48,80,112)
DEV=tuple(range(321000,321010))
VALIDATION=tuple(range(322000,322060))
TEST=tuple(range(323000,323060))
GRID=list(itertools.product((0.,.5,1.,2.),(0.,.25,.5),(0.,1.)))
THRESHOLDS=(.25,.5,1.,2.,4.,8.,None)
SOURCES=tuple(dict.fromkeys((*old.SOURCES,'guided_allocation.py','mission_allocation.py',
    'mission_policy.py','tools/mission_allocation_run.py','tests/test_mission_allocation.py')))
read=old.read;write=old.write;sha=old.sha


def parameters():
    return dict(learning_rate=5e-5,n_steps=1024,batch_size=128,n_epochs=5,gamma=1.,gae_lambda=1.,
        clip_range=.1,ent_coef=.002,vf_coef=.5,max_grad_norm=.5,target_kl=.02,
        policy_kwargs=dict(net_arch=dict(pi=[],vf=[96,96]),ortho_init=False))


def freeze():
    protocol=dict(sources={n:sha(PACKAGE/n) for n in SOURCES},
        binary_sha256=sha(mission.base.NS3/'build/scratch/allocation-final-scenario'),
        initializations=INITS,loads=LOADS,development=DEV,validation=VALIDATION,test=TEST,
        training_range=[71000000,72000000],hyperparameters=parameters(),checkpoints=[0,16384,32768,65536],
        workload='steady20%, upload-first50%, retrieval-first30%; unchanged service mixture, durations, rate, payload and SLA',
        private_profiles='hidden from policy; state adds only past three-second arrival counts per business',
        rule_grid=GRID,admission_price_thresholds=THRESHOLDS,
        prior_selection='best of24modal formulas then7price thresholds on10development scenes; no test selection',
        baseline_selection='full24modal grid and7price thresholds plus common rules on60validation scenes',
        model_selection='all8initializations retained; per-init best validation actual SLA, tie earlier checkpoint',
        reward='potential-shaped native success: Phi=.5observedactive, gamma1, initial/terminalPhi0; total exactly SLA success count',
        primary='RL minus validation-selected best rule family, including admission price',
        acceptance='paired95CI lower>0, learned policy better than its zero-step prior, admitted violation no worse',
        scope='firstpart only; no prediction, QoS controller, resource tiers, overbooking or auxiliary heads')
    import json
    normalized=json.loads(json.dumps(protocol));path=OUT/'protocol.json'
    if path.exists():assert read(path)==normalized
    else:
        write(path,protocol)
        for n in SOURCES:
            target=OUT/'frozen'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PACKAGE/n,target)
    return normalized


def check():
    p=read(OUT/'protocol.json')
    assert all(sha(PACKAGE/n)==h and sha(OUT/'frozen'/n)==h for n,h in p['sources'].items())
    assert sha(mission.base.NS3/'build/scratch/allocation-final-scenario')==p['binary_sha256']


def episode(seed,load,name,parameters=(2.,.5,1.),threshold=None,model=None):
    cfg=mission.scenario(seed,load)
    env=mission.MissionEnv(rule_parameters=parameters,admission_threshold=threshold if model else None)
    try:
        obs,_=env.reset(options={'scenario':cfg});done=False;reward=0.
        while not done:
            action=int(model.predict(obs,action_masks=obs['mask']>.5,deterministic=True)[0]) if model else mission.rule_action(name,env,parameters,threshold)
            assert obs['mask'][action]>.5
            obs,r,done,truncated,_=env.step(action);assert not truncated;reward+=r
        result=env.result();assert result['success']==reward
        return dict(seed=seed,load=load,profile=cfg['private']['demand_profile'],
            **{k:result[k] for k in ('requests','success','admitted','by_service')})
    finally:env.close()


def rule_task(spec):
    name,parameters,threshold,seeds=spec
    rows=[episode(s,l,name,parameters,threshold) for s in seeds for l in LOADS]
    result=dict(method=name,parameters=parameters,threshold=threshold,rows=rows,
        rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows))
    return result


def prepare(workers):
    freeze()
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        modal=list(pool.map(rule_task,[('modal_greedy',p,None,DEV) for p in GRID]))
        chosen=max(modal,key=lambda x:x['rate'])
        price=list(pool.map(rule_task,[('price',chosen['parameters'],t,DEV) for t in THRESHOLDS]))
        winner=max([chosen,*price],key=lambda x:x['rate'])
        prior_validation=rule_task((winner['method'],winner['parameters'],winner['threshold'],VALIDATION))
    write(OUT/'diagnosis.json',dict(modal_candidates=modal,price_candidates=price,selected=winner))
    write(OUT/'prior_validation.json',prior_validation)
    print('PREPARED',winner['method'],winner['parameters'],winner['threshold'],winner['rate'],'VALIDATION',prior_validation['rate'],flush=True)


def baseline_task(spec):
    result=rule_task((*spec,VALIDATION))
    name='-'.join(map(str,spec[:3])).replace(' ','')
    write(OUT/'baseline_validation'/f'{name}.json',result)
    print('BASELINE',spec,result['rate'],flush=True)
    return result


def baselines(workers):
    check();chosen=read(OUT/'diagnosis.json')['selected']['parameters']
    specs=[('modal_greedy',p,None) for p in GRID]+[('price',chosen,t) for t in THRESHOLDS]
    specs += [(n,chosen,None) for n in ('greedy','nearest','first_fit','native')]
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(baseline_task,specs))
    write(OUT/'baseline_selection.json',dict(candidates=rows,selected=max(rows,key=lambda x:x['rate'])))


def score(model,prior):
    rows=[episode(s,l,'ppo',prior['parameters'],prior['threshold'],model) for s in VALIDATION for l in LOADS]
    return dict(rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows),rows=rows)


class Checkpoints(BaseCallback):
    def __init__(self,init,folder,prior,records):
        super().__init__();self.init=init;self.folder=folder;self.prior=prior;self.records=records
    def _on_step(self):
        if self.num_timesteps not in (16384,32768,65536):return True
        p=self.folder/f'{self.num_timesteps}.zip';self.model.save(p)
        value=score(self.model,self.prior)
        self.records.append(dict(step=self.num_timesteps,path=str(p.relative_to(OUT)),sha256=sha(p),validation=value))
        write(self.folder/'progress.json',self.records)
        print('CHECKPOINT',self.init,self.num_timesteps,value['rate'],flush=True)
        return True


def train(init):
    torch.set_num_threads(1);check();prior=read(OUT/'diagnosis.json')['selected']
    folder=OUT/'training'/str(init);folder.mkdir(parents=True,exist_ok=True)
    env=mission.MissionEnv(seed=init+300,loads=LOADS,rule_parameters=prior['parameters'],admission_threshold=prior['threshold'])
    model=MaskablePPO(MissionPolicy,env,**parameters(),seed=init,device='cpu')
    try:
        zero=folder/'0.zip';model.save(zero)
        obs,_=env.reset(options={'scenario':mission.scenario(VALIDATION[0],LOADS[0])});done=False
        while not done:
            a=int(model.predict(obs,action_masks=obs['mask']>.5,deterministic=True)[0])
            assert a==mission.rule_action(prior['method'],env,prior['parameters'],prior['threshold'])
            obs,_,done,_,_=env.step(a)
        cached=read(OUT/'prior_validation.json')
        records=[dict(step=0,path=str(zero.relative_to(OUT)),sha256=sha(zero),validation=dict(rate=cached['rate'],rows=cached['rows']))]
        callback=Checkpoints(init,folder,prior,records)
        before=model.policy.action_net.residual[-1].weight.detach().clone();started=time.perf_counter()
        model.learn(65536,callback=callback)
        assert model._n_updates>0 and not torch.equal(before,model.policy.action_net.residual[-1].weight)
        selected=max(records,key=lambda x:(x['validation']['rate'],-x['step']))
        target=OUT/'models'/f'ppo-{init}.zip';target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OUT/selected['path'],target)
        loaded=MaskablePPO.load(target,device='cpu');assert score(loaded,prior)==selected['validation']
        record=dict(init=init,steps=model.num_timesteps,updates=model._n_updates,selected=selected,
            checkpoints=records,model=str(target.relative_to(OUT)),sha256=sha(target),
            seconds=time.perf_counter()-started,all_environment_resets=env.episode_specs)
        write(folder/'training.json',record)
        print('TRAINED',init,selected['step'],selected['validation']['rate'],flush=True)
        return record
    finally:env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','baseline','train'));parser.add_argument('--workers',type=int,default=8);parser.add_argument('--initializations',nargs='+',type=int,default=list(INITS))
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.workers)
    elif args.command=='baseline':baselines(args.workers)
    else:
        assert set(args.initializations)<=set(INITS)
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(train,args.initializations))
        write(OUT/('training_complete.json' if set(args.initializations)==set(INITS) else 'pilot_complete.json'),rows)


if __name__=='__main__':main()
