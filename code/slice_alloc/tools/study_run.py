"""Frozen sequential pipeline, resumable only with matching sources/artifacts."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import itertools
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from slice_alloc import study as s
from slice_alloc.forecast import GRU,Predictor,windows,baseline_predictions,loss,metrics
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.distributional_allocation import CandidateQRPolicy,NStepDictReplay

PARAMETERS=list(itertools.product((0.,.5,1.,2.),(0.,.5,1.),(0.,1.)))
HYPER=dict(learning_rate=1e-4,buffer_size=20000,learning_starts=1024,batch_size=128,
 train_freq=4,gradient_steps=1,gamma=1.,target_update_interval=1000,
 exploration_initial_eps=1.,exploration_final_eps=.05,exploration_steps=32768,
 max_grad_norm=10.,policy_kwargs=dict(n_quantiles=1,net_arch=[96]),
 replay_buffer_class=NStepDictReplay,replay_buffer_kwargs=dict(horizon=8,discount=1.))

def source_hashes():
 return {str(p.relative_to(s.PACKAGE)):s.sha(p) for p in sorted(s.PACKAGE.rglob('*')) if p.is_file()
  and p.suffix in ('.py','.cc','.hpp','.json','.txt','.md')
  and p.relative_to(s.PACKAGE).parts[0] not in ('results','final') and '__pycache__' not in p.parts
  and p.name!='modal_binding_run.py' and not p.name.startswith(('SIMPLE_DESIGN','REFINE_PROTOCOL'))
  and not p.name.startswith(('allocation_simple','allocation_refine','test_allocation_simple','test_allocation_refine'))}

def preregister():
 s.OUT.mkdir(parents=True,exist_ok=True)
 import importlib.metadata as metadata
 p=dict(environment={name:metadata.version(name) for name in ('numpy','torch','gymnasium','stable-baselines3','sb3-contrib','jsonschema')},native_libraries={q.name:s.sha(q) for q in sorted((s.b.p.NS3/'build/lib').glob('libns3*.so'))},title='面向低空智联的多模态网络切片技术',initializations=s.INITS,loads=s.LOADS,mobility=s.MOBILITY,
  validation=s.VALIDATION,test=s.TEST,budgets=s.BUDGETS,variants=s.VARIANTS,rules=s.RULES,
  sources=source_hashes(),binary_sha256=s.sha(s.b.p.NS3/'build/scratch/allocation-final-scenario'),
  main_comparison='A - modal_greedy; SLA successful / all arrivals; scenario-cluster and initialization bootstrap',
  secondary='A-fixed, A-no_motion (mobile only), AP-A, AP_predictive-AP_passive; Holm correction',
  train_range=[51000000,52000000],forecast_train=list(range(112000,112060)),forecast_validation=list(range(113000,113018)),
  baseline_grid=PARAMETERS,baseline_budget=24,capacity_scales=[1.,.75,1.5],
  capacity_gate={'low_forced_max':.08,'high_forced_min':.08,'high_forced_max':.4,'success_gap_min':.05},
  forecast={'sample_seconds':.1,'history_seconds':3,'horizon_seconds':1,'target':'actual bytes attempting to enter each directed device queue / physical capacity / second',
   'architecture':'GRU(input20,hidden32), ordered P50/P90; seed20261103; epochs30, batch256, Adam1e-3',
   'integration':'forecast at each admission; held at most1s for100ms QoS control; stale forecast falls back to passive'},
  qos={'scheduler':'device deficit round robin, quantum512 bytes, shared100 packet buffer','default_weight':1,'raised_weight':4,'cooldown_seconds':.5,
   'passive':'head wait > .25*deadline','predictive':'passive OR forecast90>.8 AND head wait>.1*deadline',
   'shared_content':'one physical NDN packet charged once to earliest active matching slice'},
  checkpoint_selection='validation aggregate SLA only; tie earliest; all initializations retained',
  reward='native completed SLA count, reject and violation zero; no shaping',
  hyperparameters={**HYPER,'replay_buffer_class':'NStepDictReplay'})
 p=__import__('json').loads(__import__('json').dumps(p))
 path=s.OUT/'protocol.json'
 if path.exists():assert s.read(path)==p,'experiment sources/config changed'
 else:
  s.write(path,p)
  for name in p['sources']:
   dest=s.OUT/'frozen'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s.PACKAGE/name,dest)
 return p

def run_tests():
 marker=s.OUT/'tests_complete.json';value={'protocol_sha256':s.sha(s.OUT/'protocol.json')}
 if marker.exists():assert s.read(marker)==value;return
 with (s.OUT/'tests.log').open('w') as stream:
  subprocess.run([sys.executable,'-m','unittest','slice_alloc.tests.test_study','slice_alloc.tests.test_allocation_value','slice_alloc.tests.test_patrol'],cwd=s.PACKAGE.parent,stdout=stream,stderr=subprocess.STDOUT,check=True)
 s.write(marker,value)

def diagnostic_task(spec):
 seed,load,mobility,scale=spec;before=time.perf_counter()
 result=s.episode(s.scenario(seed,load,mobility,scale=scale),'greedy')
 forced=sum(not r['decision']['accept'] and not r['feasible'] for r in result['decisions'])
 return dict(seed=seed,load=load,mobility=mobility,success=result['success'],requests=load,forced=forced,seconds=time.perf_counter()-before)

def capacity(workers):
 path=s.OUT/'capacity.json'
 if path.exists():return s.read(path)['scale']
 attempts=[]
 for scale in (1.,.75,1.5):
  specs=list(itertools.product(range(111000,111006),s.LOADS,s.MOBILITY,[scale]))
  with cf.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'),max_workers=workers) as pool:rows=list(pool.map(diagnostic_task,specs))
  groups={load:[r for r in rows if r['load']==load] for load in s.LOADS}
  rates={load:sum(r['success'] for r in rr)/sum(r['requests'] for r in rr) for load,rr in groups.items()}
  forced={load:sum(r['forced'] for r in rr)/sum(r['requests'] for r in rr) for load,rr in groups.items()}
  passed=forced[32]<=.08 and .08<=forced[96]<=.4 and rates[32]-rates[96]>=.05
  attempts.append(dict(scale=scale,rates=rates,forced=forced,passed=passed,rows=rows))
  print('CAPACITY',scale,rates,forced,passed,flush=True)
  if passed:
   s.write(path,dict(scale=scale,attempts=attempts));return scale
 s.write(s.OUT/'capacity_failed.json',attempts)
 raise RuntimeError('preregistered capacity criterion not met; no formal training/test opened')

def data_task(spec):
 split,index,seed,scale=spec;load=s.LOADS[index%3];mobility=s.MOBILITY[(index//3)%3];method=s.RULES[index%3]
 prefix=s.OUT/'forecast_data'/split/str(seed)
 if not Path(str(prefix)+'-telemetry.jsonl.gz').exists():s.episode(s.scenario(seed,load,mobility,scale=scale),method,prefix=prefix)
 return dict(seed=seed,load=load,mobility=mobility,method=method,prefix=str(prefix.relative_to(s.OUT)))

def prepare_data(workers,scale):
 path=s.OUT/'forecast_data.json'
 if path.exists():return
 specs=[(split,i,seed,scale) for split,seeds in [('train',range(112000,112060)),('validation',range(113000,113018))] for i,seed in enumerate(seeds)]
 with cf.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'),max_workers=workers) as pool:rows=list(pool.map(data_task,specs))
 s.write(path,rows)

def load_data(split):
 xx=[];yy=[]
 for row in s.read(s.OUT/'forecast_data.json'):
  if split not in row['prefix']:continue
  x,y,_=windows(str(s.OUT/row['prefix'])+'-telemetry.jsonl.gz');xx.append(x);yy.append(y)
 return np.concatenate(xx),np.concatenate(yy)

def train_forecast():
 path=s.OUT/'forecast_complete.json'
 if path.exists():
  assert s.read(path)['sha256']==s.sha(s.OUT/'forecast.pt');return
 torch.set_num_threads(1);torch.manual_seed(20261103);rng=np.random.default_rng(20261103)
 x,y=load_data('train');vx,vy=load_data('validation');model=GRU();optim=torch.optim.Adam(model.parameters(),lr=1e-3)
 best=float('inf');history=[]
 for epoch in range(30):
  model.train();order=rng.permutation(len(x));train_loss=[]
  for start in range(0,len(x),256):
   idx=order[start:start+256];value=loss(model(torch.from_numpy(x[idx])),torch.from_numpy(y[idx]));optim.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optim.step();train_loss.append(float(value.detach()))
  model.eval()
  with torch.inference_mode():pred=torch.cat([model(torch.from_numpy(vx[k:k+1024])) for k in range(0,len(vx),1024)]).numpy()
  val=metrics(pred,vy);history.append(dict(epoch=epoch+1,training_loss=float(np.mean(train_loss)),validation=val))
  if val['pinball']<best:
   best=val['pinball'];torch.save({'state_dict':model.state_dict(),'epoch':epoch+1},s.OUT/'forecast.pt')
 print('FORECAST',best,flush=True)
 # Baseline P90 is validation-calibrated using one scalar residual per link.
 calibration={}
 for name,pred in baseline_predictions(vx).items():calibration[name]=np.maximum(0,np.quantile(vy-pred,.9,axis=0)).tolist()
 s.write(path,dict(sha256=s.sha(s.OUT/'forecast.pt'),training_windows=len(x),validation_windows=len(vx),history=history,baseline_p90_residual=calibration))

def baseline_task(spec):
 parameters,scale=spec;rows=[]
 for k,seed in enumerate(s.VALIDATION):
  for load in s.LOADS:
   result=s.episode(s.scenario(seed,load,s.MOBILITY[k%3],scale=scale),'modal_greedy',parameters=parameters)
   rows.append({'success':result['success'],'requests':load})
 return dict(parameters=parameters,rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows),rows=rows)

def select_baseline(workers,scale):
 path=s.OUT/'baseline_selection.json'
 if path.exists():return
 with cf.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'),max_workers=workers) as pool:rows=list(pool.map(baseline_task,[(p,scale) for p in PARAMETERS]))
 best=max(rows,key=lambda r:r['rate']);s.write(path,dict(selected=best,all_trials=rows,trial_count=len(rows)))
 print('BASELINE_SELECTED',best['parameters'],best['rate'],flush=True)

def validation(model,variant,scale,predictor):
 rows=[]
 for k,seed in enumerate(s.VALIDATION):
  for load in s.LOADS:
   result=s.episode(s.scenario(seed,load,s.MOBILITY[k%3],scale=scale),variant,model,predictor)
   rows.append(dict(seed=seed,load=load,mobility=s.MOBILITY[k%3],success=result['success'],requests=load))
 return dict(rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows),rows=rows)

class Checkpoints(BaseCallback):
 def __init__(self,folder,variant,scale,predictor):
  super().__init__();self.folder=folder;self.variant=variant;self.scale=scale;self.predictor=predictor;self.records=[]
 def _on_step(self):
  if self.num_timesteps in s.BUDGETS:
   path=self.folder/f'{self.num_timesteps}.zip';self.model.save(path)
   value=validation(self.model,self.variant,self.scale,self.predictor)
   self.records.append(dict(step=self.num_timesteps,path=str(path.relative_to(s.OUT)),sha256=s.sha(path),validation=value))
   s.write(self.folder/'progress.json',self.records)
   print('CHECKPOINT',self.variant,self.model.seed,self.num_timesteps,value['rate'],flush=True)
  return True

def train_task(spec):
 variant,init,scale=spec;torch.set_num_threads(1);folder=s.OUT/'training'/f'{variant}-{init}';path=folder/'complete.json'
 if path.exists():
  value=s.read(path);assert value['protocol_sha256']==s.sha(s.OUT/'protocol.json');assert value['sha256']==s.sha(s.OUT/value['model']);return value
 predictor=Predictor(s.OUT/'forecast.pt') if variant=='AP' else None
 env=s.Env(variant=variant,predictor=predictor,seed=init+100,scale=scale)
 model=MaskedDoubleDQN(CandidateQRPolicy,env,**HYPER,seed=init,device='cpu');callback=Checkpoints(folder,variant,scale,predictor)
 before=model.quantile_net.deployment[-1].weight.detach().clone();started=time.perf_counter()
 try:
  model.learn(max(s.BUDGETS),callback=callback)
  assert not torch.equal(before,model.quantile_net.deployment[-1].weight.detach())
  best=max(callback.records,key=lambda v:(v['validation']['rate'],-v['step']))
  dest=s.OUT/'models'/f'{variant}-{init}.zip';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s.OUT/best['path'],dest)
  reload=MaskedDoubleDQN.load(dest,device='cpu');assert validation(reload,variant,scale,predictor)==best['validation']
  value=dict(variant=variant,init=init,steps=model.num_timesteps,updates=model._n_updates,seconds=time.perf_counter()-started,
   model=str(dest.relative_to(s.OUT)),sha256=s.sha(dest),checkpoints=callback.records,selected=best,training_seeds=env.episode_specs,protocol_sha256=s.sha(s.OUT/'protocol.json'))
  s.write(path,value);return value
 finally:env.close()

def eval_task(spec):
 method,init,scale=spec;torch.set_num_threads(1)
 variant='AP' if method.startswith('AP') else method
 model=MaskedDoubleDQN.load(s.OUT/'models'/f'{variant}-{init}.zip',device='cpu') if method not in s.RULES else None
 predictor=Predictor(s.OUT/'forecast.pt') if method.startswith('AP') else None
 parameters=s.read(s.OUT/'baseline_selection.json')['selected']['parameters'];rows=[]
 for seed in s.TEST:
  for mobility in s.MOBILITY:
   if method=='no_motion' and mobility=='stationary':continue
   for load in s.LOADS:
    prefix=s.OUT/'native'/f'{method}-{init}'/f'{mobility}-{load}-{seed}'
    marker=Path(str(prefix)+'-complete.json')
    if marker.exists():row=s.read(marker)
    else:
     qos='passive' if method=='AP_passive' else 'predictive' if method=='AP_predictive' else 'none'
     result=s.episode(s.scenario(seed,load,mobility,qos,scale),method,model,predictor,prefix,parameters)
     row=dict(method=method,init=init,seed=seed,mobility=mobility,load=load,prefix=str(prefix.relative_to(s.OUT)),
      requests=load,success=result['success'],admitted=result['admitted'],by_service=result['by_service'])
     s.write(marker,row)
    rows.append(row)
 return rows

def prepare_test():
 paths=['protocol.json','tests_complete.json','capacity.json','forecast_complete.json','forecast.pt','baseline_selection.json','training_complete.json']
 snapshot={name:s.sha(s.OUT/name) for name in paths}
 for row in s.read(s.OUT/'training_complete.json'):
  assert row['steps']==max(s.BUDGETS) and row['updates']>0
  assert s.sha(s.OUT/row['model'])==row['sha256']
 marker=s.OUT/'test_started.json'
 if marker.exists():assert s.read(marker)==snapshot
 else:s.write(marker,snapshot)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','train','evaluate','all']);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
 torch.set_num_threads(1);preregister();run_tests();scale=capacity(args.workers)
 if args.command in ('prepare','train','all'):
  assert not (s.OUT/'test_started.json').exists() or (s.OUT/'training_complete.json').exists()
  prepare_data(args.workers,scale);train_forecast();select_baseline(args.workers,scale)
 if args.command in ('train','all') and not (s.OUT/'training_complete.json').exists():
  assert not (s.OUT/'test_started.json').exists()
  with cf.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'),max_workers=args.workers) as pool:
   rows=list(pool.map(train_task,[(v,i,scale) for v in s.VARIANTS for i in s.INITS]))
  s.write(s.OUT/'training_complete.json',rows)
 if args.command in ('evaluate','all'):
  prepare_test();specs=[(v,i,scale) for v in (*s.VARIANTS,'AP_passive','AP_predictive') for i in s.INITS]+[(v,0,scale) for v in s.RULES]
  with cf.ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'),max_workers=args.workers) as pool:rows=[r for group in pool.map(eval_task,specs) for r in group]
  s.write(s.OUT/'evaluation.json',rows);print('EVALUATED',len(rows),flush=True)
 if args.command=='all':
  from slice_alloc.tools.study_report import report
  report(args.workers)

if __name__=='__main__':main()
