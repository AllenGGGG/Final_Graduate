"""One bounded independent-Double-DQN experiment, not a rule-guided policy."""
import argparse,concurrent.futures as cf,copy,gzip,hashlib,json,os,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
from slice_alloc import allocation_credit as envs
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.distributional_allocation import NStepDictReplay
from slice_alloc.tools import allocation_simple_run as old

PKG=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('LOWALT_CREDIT_OUTPUT',str(PKG/'results/credit'))).resolve()
REFERENCE=PKG/'results/final'
INITS=(11,23,37,53,71);PILOT=(11,23)
LOADS=(32,64,96);DEV=tuple(range(441000,441012));VAL=tuple(range(442000,442060));TEST=tuple(range(443000,443060))
CONFIGS={'short':(8,0.),'long':(32,0.),'progress':(8,.5),'long_progress':(32,.5)}
BUDGETS=(8192,16384,32768)
SOURCES=(*old.SOURCES,'allocation_credit.py','tools/allocation_credit_run.py','tests/test_allocation_credit.py')
read=old.read;write=old.write;sha=old.sha

def freeze():
    p=dict(title='面向低空智联的多模态网络切片技术',scope='firstpart independent Masked Double-DQN only',
        sources={n:sha(PKG/n) for n in SOURCES},binary_sha256=sha(envs.base.NS3/'build/scratch/allocation-final-scenario'),
        reference_models={str(i):sha(REFERENCE/f'models/double_dqn-{i}.zip') for i in INITS},
        initializations=INITS,pilot_initializations=PILOT,loads=LOADS,development=DEV,validation=VAL,test=TEST,
        training_seed_range=[81000000,82000000],configs=CONFIGS,budgets=BUDGETS,
        actual_training=dict(learning_rate=5e-5,replay_capacity=50000,learning_starts=1024,batch_size=128,train_freq=4,gradient_steps=1,
            gamma=1.,target_update_interval=1000,exploration_initial_eps=.1,exploration_final_eps=.02,exploration_steps=16384,
            network='unchanged96-wide shared candidate scalar network',optimizer='Adam states retained from independent RL checkpoint'),
        design='unchanged original uniform arrivals, physical network and packet contracts; three natural load levels',
        prior='none; initialize exclusively from already trained independent Double-DQN, never from a heuristic',
        reward='native SLA settlement plus Phi(next)-Phi(current); Phi=.5sum observed on_time/contract.expected for active contracts; initial/terminal0; gamma1',
        selection='pilot2 per configuration; expand exactly one family to all5 if paired validation mean exceeds matching zero-step RL by>=0.2percentage_points; tie simpler no-potential family; all init checkpoints validation max, tie earlier',
        baselines=['greedy','nearest','first_fit','native'],primary='RL versus Greedy; also report native semantic rule and learning versus its initial RL checkpoint',
        no_test_tuning=True)
    p=json.loads(json.dumps(p));file=OUT/'protocol.json'
    if file.exists():assert read(file)==p
    else:
        write(file,p)
        for n in SOURCES:
            t=OUT/'frozen'/n;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PKG/n,t)
    return p

def check():
    p=read(OUT/'protocol.json')
    for n,h in p['sources'].items():assert sha(PKG/n)==sha(OUT/'frozen'/n)==h,n
    assert sha(envs.base.NS3/'build/scratch/allocation-final-scenario')==p['binary_sha256']
    for i,h in p['reference_models'].items():assert sha(REFERENCE/f'models/double_dqn-{i}.zip')==h

def episode(seed,load,method,model=None,weight=0.,trace=None):
    env=envs.CreditEnv(progress_weight=weight)
    try:
        obs,_=env.reset(options=dict(scenario=envs.base.scenario(seed,load),prefix=trace,trace=trace is not None))
        done=False;reward=0.
        while not done:
            action=int(model.predict(obs,deterministic=True)[0]) if model else (envs.native_action(env) if method=='native' else old.choose(method,env))
            assert obs['mask'][action]>.5
            obs,r,done,truncated,_=env.step(action);assert not truncated;reward+=r
        result=env.result();assert abs(reward-result['success'])<1e-8
        return result
    finally:env.close()

def summarize(rows):return sum(r['success'] for r in rows)/sum(r['requests'] for r in rows)
def score(model,weight=0.,seeds=VAL):
    rows=[]
    for s in seeds:
        for l in LOADS:
            r=episode(s,l,'double_dqn',model,weight)
            rows.append(dict(seed=s,load=l,**{k:r[k] for k in ('requests','success','admitted','by_service')}))
    return dict(rate=summarize(rows),rows=rows)

def dev_task(spec):
    method,init=spec;torch.set_num_threads(1)
    model=MaskedDoubleDQN.load(REFERENCE/f'models/double_dqn-{init}.zip',device='cpu') if method=='double_dqn' else None
    rows=[];delays=[]
    for s in DEV:
        for l in LOADS:
            r=episode(s,l,method,model)
            rows.append(dict(seed=s,load=l,**{k:r[k] for k in ('requests','success','admitted','by_service')}))
            index={d['request']:j for j,d in enumerate(r['decisions'])}
            for k,msg in enumerate(r['observations'][1:]):
                for settled in msg['observation']['settled']:delays.append(k-index[settled['id']]+1)
    if model is not None:
        write(OUT/'initial_validation'/f'{init}.json',score(model))
    return dict(method=method,init=init,rate=summarize(rows),rows=rows,
        settlement_delays=delays,fraction_beyond_eight=float(np.mean(np.array(delays)>8)) if delays else 0.)

def prepare(workers):
    freeze()
    specs=[('double_dqn',i) for i in INITS]+[(m,0) for m in ('greedy','nearest','first_fit','native')]
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(dev_task,specs))
    write(OUT/'development.json',rows)
    for r in rows:print('DEVELOPMENT',r['method'],r['init'],r['rate'],'credit>8',r['fraction_beyond_eight'],flush=True)

class Checkpoints(BaseCallback):
    def __init__(self,config,init,folder,records):super().__init__();self.config=config;self.init=init;self.folder=folder;self.records=records
    def _on_step(self):
        if self.num_timesteps not in BUDGETS:return True
        p=self.folder/f'{self.num_timesteps}.zip';self.model.save(p)
        v=score(self.model,CONFIGS[self.config][1]);self.records.append(dict(step=self.num_timesteps,path=str(p.relative_to(OUT)),sha256=sha(p),validation=v))
        write(self.folder/'progress.json',self.records)
        print('CHECKPOINT',self.config,self.init,self.num_timesteps,v['rate'],flush=True)
        return True

def train(spec):
    config,init=spec;torch.set_num_threads(1);check()
    folder=OUT/'training'/config/str(init);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training.json').exists():return read(folder/'training.json')
    horizon,weight=CONFIGS[config]
    env=envs.CreditEnv(seed=init+2000,loads=LOADS,progress_weight=weight)
    model=MaskedDoubleDQN.load(REFERENCE/f'models/double_dqn-{init}.zip',device='cpu')
    model.set_env(env);model.learning_rate=5e-5;model.lr_schedule=get_schedule_fn(5e-5)
    model.learning_starts=1024;model.exploration_initial_eps=.1;model.exploration_final_eps=.02;model.exploration_steps=16384
    model.set_random_seed(init+2000);model._n_calls=0;model._n_updates=0
    model.buffer_size=50000;model.replay_buffer_kwargs=dict(horizon=horizon,discount=1.)
    model.replay_buffer=NStepDictReplay(50000,model.observation_space,model.action_space,device='cpu',n_envs=1,horizon=horizon,discount=1.)
    records=[]
    try:
        p=folder/'0.zip';model.save(p)
        cache=OUT/'initial_validation'/f'{init}.json'
        if cache.exists():value=read(cache)
        else:value=score(model);write(cache,value)
        records.append(dict(step=0,path=str(p.relative_to(OUT)),sha256=sha(p),validation=value))
        before=model.quantile_net.deployment[-1].weight.detach().clone();started=time.perf_counter()
        model.learn(max(BUDGETS),callback=Checkpoints(config,init,folder,records),reset_num_timesteps=True)
        assert model._n_updates>0 and not torch.equal(before,model.quantile_net.deployment[-1].weight)
        best=max(records,key=lambda r:(r['validation']['rate'],-r['step']))
        dst=OUT/'models'/config/f'double_dqn-{init}.zip';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OUT/best['path'],dst)
        loaded=MaskedDoubleDQN.load(dst,device='cpu');assert score(loaded)==best['validation']
        row=dict(config=config,init=init,steps=model.num_timesteps,updates=model._n_updates,seconds=time.perf_counter()-started,
            selected=best,checkpoints=records,model=str(dst.relative_to(OUT)),model_sha256=sha(dst),training_seeds=env.episode_specs)
        write(folder/'training.json',row);print('TRAINED',config,init,best['step'],best['validation']['rate'],flush=True);return row
    finally:env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','pilot','expand'));p.add_argument('--workers',type=int,default=8);p.add_argument('--config',choices=CONFIGS)
    args=p.parse_args()
    if args.command=='prepare':prepare(args.workers);return
    check();assert not (OUT/'test_started.json').exists()
    specs=[(c,i) for c in CONFIGS for i in PILOT] if args.command=='pilot' else [(args.config,i) for i in INITS if i not in PILOT]
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(train,specs))
    write(OUT/('pilot_complete.json' if args.command=='pilot' else 'expansion_complete.json'),rows)

if __name__=='__main__':main()
