"""Read-only final verification of paired statistics, retraining and ensemble study."""
import contextlib,io,json,runpy
from pathlib import Path
import numpy as np
import torch
from slice_alloc.tools import distributional_run as d
from slice_alloc.distributional_allocation import MaskedDoubleQRDQN

def main():
 torch.set_num_threads(1);base=d.OUT;out=d.ROOT/'slice_alloc/results/qrdqn_ensemble'
 with contextlib.redirect_stdout(io.StringIO()) as captured:
  runpy.run_path(str(base/'independent_verify.py'),run_name='__main__')
 assert json.loads(captured.getvalue())['passed']
 delivery=d.read(out/'delivery.json');assert delivery['passed'] and delivery['sources']==d.hashes()
 for name,h in delivery['artifacts'].items():assert d.sha(out/name)==h
 for name,h in delivery['documents'].items():assert d.sha(d.ROOT/name)==h
 for name,h in delivery['implementation'].items():assert d.sha(d.ROOT/name)==h
 plan=d.read(out/'holdout_plan.json');assert plan['script_sha256']==d.sha(d.ROOT/'slice_alloc/tools/qrdqn_ensemble_evaluate.py')
 assert plan['ensemble_script_sha256']==d.sha(d.ROOT/'slice_alloc/tools/qrdqn_ensemble_probe.py')
 evaluation=d.read(out/'evaluation.json');analysis=d.read(out/'analysis.json');extra=d.read(out/'additional_rules_evaluation.json')
 raw=d.read(out/'raw_audit.json')+d.read(out/'additional_rules_raw_audit.json')
 assert len(evaluation)==560 and len(extra)==240 and len(raw)==800
 inputs={}
 for row,check in zip(evaluation+extra,raw):
  assert row['path']==check['path'] and row['success']==check['request_success']
  fingerprint=d.sha(out/'native'/(row['path']+'-scenario.json'))
  assert inputs.setdefault((row['seed'],row['load']),fingerprint)==fingerprint
 vectors={}
 for method,s in analysis['stats'].items():
  group=[r for r in evaluation if r['method']==method];n=sum(r['requests'] for r in group)
  assert sum(r['success'] for r in group)/n==s['success_rate']
  vectors[method]=np.array([sum(r['success'] for r in group if r['seed']==s)/sum(r['requests'] for r in group if r['seed']==s) for s in plan['test_seeds']])
 rng=np.random.default_rng(1170040)
 for method,record in analysis['comparisons'].items():
  diff=vectors['ensemble']-vectors[method]
  samples=[diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(10000)]
  assert abs(diff.mean()-record['difference'])<1e-12
  np.testing.assert_allclose(np.quantile(samples,[.025,.975]),record['ci95'],rtol=0,atol=1e-12)
 r=base/'statistical_correction/retraining';reproduction=d.read(r/'final_verification.json');assert reproduction['passed']
 assert reproduction['episodes']==300 and reproduction['success_rate']==d.read(base/'analysis.json')['stats']['qrdqn']['success_rate']
 for name,h in reproduction['files'].items():assert d.sha(r/name)==h
 training=d.read(r/'training.json');reference=d.read(base/'training.json');assert len(training)==5 and sum(x['trained_steps'] for x in training)==655360
 checked=0
 for record in training:
  assert record['sources']==d.hashes()
  prior=next(x for x in reference if x['initialization']==record['initialization'])
  assert record['training_scenarios']==prior['training_scenarios']
  for stage in record['stages']:
   old=next(x for x in prior['stages'] if x['step']==stage['step'])
   assert stage['validation']==old['validation']
   assert d.sha(d.ROOT/stage['path'])==stage['sha256']
   a=MaskedDoubleQRDQN.load(d.ROOT/stage['path'],device='cpu');b=MaskedDoubleQRDQN.load(d.ROOT/old['path'],device='cpu')
   for name,v in a.policy.state_dict().items():torch.testing.assert_close(v,b.policy.state_dict()[name],atol=0,rtol=0)
   checked+=1
 assert checked==15
 assert 'Ran 136 tests' in (out/'tests.log').read_text() and (out/'tests.log').read_text().rstrip().endswith('OK')
 docs=d.read(d.ROOT/'slice_alloc/results/document_verification.json')
 assert docs['passed'] and docs['delivery_sha256']==d.sha(out/'delivery.json') and docs['documents']==delivery['documents']
 print(json.dumps(dict(passed=True,paired_statistics_recomputed=True,training_weights_exact_at_all_15_checkpoints=True,
  from_scratch_training_steps=655360,retrained_formal_episodes_exact=300,ensemble_raw_audited_episodes=800,
  tests=136,source_model_document_hashes_valid=True,ensemble_success=analysis['stats']['ensemble']['success_rate'],
  comparisons=analysis['comparisons']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
