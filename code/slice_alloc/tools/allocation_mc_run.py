"""Bounded independent Double-DQN full-return replay pilot."""
import argparse,concurrent.futures as cf,copy,os,shutil,time
from pathlib import Path
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn
from slice_alloc import allocation_credit as envs
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.allocation_mc import MonteCarloDictReplay
from slice_alloc.tools import allocation_credit_run as common

OUT=common.PKG/'results/monte_carlo';INITS=(11,23);LOADS=(32,64,96);VAL=common.VAL;BUDGETS=(8192,16384,32768)

def freeze():
 base=common.read(common.PKG/'results/credit/protocol.json')
 sources={**base['sources'],'allocation_mc.py':'','tools/allocation_mc_run.py':'','tests/test_allocation_mc.py':''}
 for n in tuple(sources):
  if not sources[n]:sources[n]=common.sha(common.PKG/n)
 protocol=dict(base, sources=sources,initializations=list(INITS),loads=list(LOADS),validation=list(VAL),training='start from independently trained RL checkpoint; replace n-step replay with full terminal return-to-go; same network and native reward',budgets=list(BUDGETS),formal_test_opened=False)
 p=OUT/'protocol.json'
 if p.exists():assert common.read(p)==protocol
 else:
  common.write(p,protocol)
  for n in sources:
   q=OUT/'frozen'/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(common.PKG/n,q)
 return protocol

def score(model):return common.score(model,0.,VAL)
class Check(BaseCallback):
 def __init__(self,folder,records):super().__init__();self.folder=folder;self.records=records
 def _on_step(self):
  if self.num_timesteps not in BUDGETS:return True
  p=self.folder/f'{self.num_timesteps}.zip';self.model.save(p);v=score(self.model);self.records.append(dict(step=self.num_timesteps,path=str(p.relative_to(OUT)),sha256=common.sha(p),validation=v));common.write(self.folder/'progress.json',self.records);print('CHECKPOINT',self.folder.name,self.num_timesteps,v['rate'],flush=True);return True

def train(init):
 freeze();torch.set_num_threads(1);folder=OUT/'training'/str(init);folder.mkdir(parents=True,exist_ok=True)
 if (folder/'training.json').exists():return common.read(folder/'training.json')
 env=envs.CreditEnv(seed=init+9000,loads=LOADS)
 model=MaskedDoubleDQN.load(common.PKG/'results/final/models'/f'double_dqn-{init}.zip',device='cpu');model.set_env(env)
 model.learning_rate=5e-5;model.lr_schedule=get_schedule_fn(5e-5);model.learning_starts=1024;model.exploration_initial_eps=.1;model.exploration_final_eps=.02;model.exploration_steps=16384;model.set_random_seed(init+9000);model._n_calls=0;model._n_updates=0
 model.replay_buffer=MonteCarloDictReplay(50000,model.observation_space,model.action_space,device='cpu',n_envs=1)
 records=[];p=folder/'0.zip';model.save(p);zero=score(model);records.append(dict(step=0,path=str(p.relative_to(OUT)),sha256=common.sha(p),validation=zero));before=model.quantile_net.deployment[-1].weight.detach().clone()
 try:
  model.learn(max(BUDGETS),callback=Check(folder,records),reset_num_timesteps=True);assert model._n_updates>0 and not torch.equal(before,model.quantile_net.deployment[-1].weight)
  best=max(records,key=lambda r:(r['validation']['rate'],-r['step']));dst=OUT/'models'/f'double_dqn-{init}.zip';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OUT/best['path'],dst)
  row=dict(init=init,steps=32768,updates=model._n_updates,selected=best,checkpoints=records,model=str(dst.relative_to(OUT)),model_sha256=common.sha(dst),returns_committed=model.replay_buffer.returns_committed)
  common.write(folder/'training.json',row);print('TRAINED',init,best['step'],best['validation']['rate'],flush=True);return row
 finally:env.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);a=p.parse_args();freeze()
 with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(train,INITS))
 common.write(OUT/'training_complete.json',rows)
