"""Predeclared, validation-only test of averaging learned QR-DQN values."""
import concurrent.futures
import json
from pathlib import Path
import numpy as np
import torch
from slice_alloc.tools import distributional_run as d
from slice_alloc.distributional_allocation import MaskedDoubleQRDQN

OUT=d.ROOT/'slice_alloc/results/qrdqn_ensemble'
SEEDS=tuple(range(11600,11616))
METHODS=('mean','vote','single','marginal_price')

class Ensemble:
 def __init__(self,models,method):self.models=models;self.method=method
 def predict(self,obs,deterministic=True):
  with torch.no_grad():
   values=[]
   for model in self.models:
    tensor,_=model.policy.obs_to_tensor(obs)
    values.append(model.quantile_net(tensor).mean(1)[0].cpu().numpy())
   values=np.stack(values);valid=obs['mask']>.5
   values[:,~valid]=-np.inf
   mean=values.mean(0)
   if self.method=='vote':
    counts=np.bincount(values.argmax(1),minlength=len(mean))
    eligible=valid&(counts==counts[valid].max())
    mean=np.where(eligible,mean,-np.inf)
   return np.asarray(int(mean.argmax())),None

def run(spec):
 method,seed=spec;torch.set_num_threads(1)
 records=d.read(d.OUT/'protocol.json')['models'];models=[]
 if method!='marginal_price':
  for r in records:
   if r['method']=='qrdqn':
    models.append(MaskedDoubleQRDQN.load(d.ROOT/r['path'],device='cpu'))
 results=[]
 for load in d.LOADS:
  if method=='single':
   for model in models:
    r=d.episode(d.make_scenario(seed,load),'qrdqn',model)
    results.append(dict(method=method,seed=seed,load=load,success=r['success'],requests=r['requests']))
  else:
   model=Ensemble(models,method) if models else None
   r=d.episode(d.make_scenario(seed,load),'qrdqn' if models else method,model)
   results.append(dict(method=method,seed=seed,load=load,success=r['success'],requests=r['requests']))
 return results

if __name__=='__main__':
 torch.set_num_threads(1)
 plan=dict(scope='Validation only, no formal claim or selection using previous test results',seeds=SEEDS,methods=METHODS,
  models=d.read(d.OUT/'protocol.json')['models'],source_sha256=d.sha(__file__),training_steps=0,
  rationale='Reduce initialization variance by averaging learned values; vote sensitivity check, no rule scores',
  prospective_test_seeds=list(range(11700,11740)),sources=d.hashes())
 assert not (OUT/'validation.json').exists()
 d.write(OUT/'plan.json',plan)
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
  rows=[r for group in pool.map(run,[(m,s) for m in METHODS for s in SEEDS]) for r in group]
 stats={m:sum(r['success'] for r in rows if r['method']==m)/sum(r['requests'] for r in rows if r['method']==m) for m in METHODS}
 d.write(OUT/'validation.json',dict(rows=rows,stats=stats,source_sha256=d.sha(__file__)))
 print('VALIDATION ONLY',json.dumps(stats),flush=True)
