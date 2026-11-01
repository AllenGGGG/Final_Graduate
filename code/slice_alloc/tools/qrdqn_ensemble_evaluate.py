"""New holdout evaluation of the validation-selected learned-value ensemble."""
import concurrent.futures,gzip
import numpy as np
import torch
from slice_alloc.tools import distributional_run as d
from slice_alloc.tools.qrdqn_ensemble_probe import Ensemble
from slice_alloc.distributional_allocation import MaskedDoubleQRDQN

OUT=d.ROOT/'slice_alloc/results/qrdqn_ensemble'
TEST=tuple(range(11700,11740))

def worker(spec):
 method,record=spec;torch.set_num_threads(1);d.native.OUT=OUT;model=None
 if method=='ensemble':
  records=[x for x in d.read(d.OUT/'protocol.json')['models'] if x['method']=='qrdqn']
  model=Ensemble([MaskedDoubleQRDQN.load(d.ROOT/r['path'],device='cpu') for r in records],'mean')
 elif record:model=MaskedDoubleQRDQN.load(d.ROOT/record['path'],device='cpu')
 rows=[]
 for seed in TEST:
  for load in d.LOADS:
   cfg=d.make_scenario(seed,load);init=record['initialization'] if record else None
   tag=f'holdout/{method}-{init or 0}/{load}-{seed}';prefix=OUT/'native'/tag
   result=d.episode(cfg,'qrdqn' if model else method,model,prefix)
   rows.append(d.native.row_for(cfg,method,result,tag,initialization=init,seed=seed,load=load,mix='balanced',scope='holdout'))
 return rows

def audit(row):
 d.native.OUT=OUT
 return d.native.raw_check(row)

if __name__=='__main__':
 torch.set_num_threads(1)
 assert not (OUT/'analysis.json').exists()
 dev=d.read(OUT/'validation.json')['stats'];chosen=max(('mean','vote'),key=lambda x:dev[x]);assert chosen=='mean'
 records=[r for r in d.read(d.OUT/'protocol.json')['models'] if r['method']=='qrdqn']
 plan=dict(scope='Fresh holdout after validation-only selection; fixed five-model ensemble',
  chosen=chosen,validation_sha256=d.sha(OUT/'validation.json'),test_seeds=TEST,loads=d.LOADS,
  models=records,sources=d.hashes(),script_sha256=d.sha(__file__),
  ensemble_script_sha256=d.sha(d.ROOT/'slice_alloc/tools/qrdqn_ensemble_probe.py'),
  resampling='Scenario-paired bootstrap conditional on the fixed five trained models; no claim of five independently trained ensembles')
 d.write(OUT/'holdout_plan.json',plan)
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
  rows=[r for group in pool.map(worker,[('ensemble',None)]+[('single',r) for r in records]+[('marginal_price',None)]) for r in group]
 d.write(OUT/'evaluation.json',rows)
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:raw=list(pool.map(audit,rows))
 d.write(OUT/'raw_audit.json',raw)
 inputs={}
 for r in rows:
  fingerprint=d.sha(OUT/'native'/(r['path']+'-scenario.json'))
  assert inputs.setdefault((r['seed'],r['load']),fingerprint)==fingerprint
 stats={m:d.native.aggregate([r for r in rows if r['method']==m]) for m in ('ensemble','single','marginal_price')}
 byseed={m:np.array([sum(r['success'] for r in rows if r['method']==m and r['seed']==s)/sum(r['requests'] for r in rows if r['method']==m and r['seed']==s) for s in TEST]) for m in stats}
 rng=np.random.default_rng(1170040);comparisons={}
 for other in ('single','marginal_price'):
  differences=byseed['ensemble']-byseed[other];samples=[]
  for _ in range(10000):samples.append(float(differences[rng.integers(0,len(TEST),len(TEST))].mean()))
  comparisons[other]=dict(difference=float(differences.mean()),ci95=np.quantile(samples,[.025,.975]).tolist())
 assert len(rows)==560
 assert d.hashes()==plan['sources'] and d.sha(__file__)==plan['script_sha256']
 assert all(d.sha(d.ROOT/r['path'])==r['sha256'] for r in records)
 result=dict(passed=True,stats=stats,comparisons=comparisons,episodes=len(rows),decisions=sum(r['requests'] for r in rows),
  same_inputs=True,all_raw_audits_passed=True,plan_sha256=d.sha(OUT/'holdout_plan.json'),scope=plan['resampling'])
 d.write(OUT/'analysis.json',result)
 print('FRESH HOLDOUT',result,flush=True)
