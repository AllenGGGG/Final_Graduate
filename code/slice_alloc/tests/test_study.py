import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import torch
from slice_alloc import study as s
from slice_alloc.forecast import GRU,history_array,windows,loss

class Contracts(unittest.TestCase):
 def test_waypoints(self):
  for mobility in s.MOBILITY:
   cfg=s.scenario(3001,32,mobility)
   self.assertEqual(len(cfg['requests']),32)
   self.assertEqual(len(cfg['private']['motions']),0 if mobility=='stationary' else 4)
   for points in cfg['private']['waypoints'].values():self.assertTrue(all(p[2]==80 for p in points))
   if mobility=='local':self.assertTrue(all(max(p[0] for p in v)<=45 for v in cfg['private']['waypoints'].values()))
 def test_history_is_causal(self):
  history=[{'load':[i]*10,'queue':[i/10]*10} for i in range(50)]
  a=history_array(history)
  self.assertEqual(a.shape,(30,20));self.assertEqual(a[0,0],20);self.assertEqual(a[-1,0],49)
  self.assertEqual(history_array([]).sum(),0)
 def test_quantile_order_and_gradients(self):
  model=GRU();pred=model(torch.rand(8,30,20));self.assertEqual(pred.shape,(8,10,2))
  self.assertTrue(bool((pred[:,:,1]>=pred[:,:,0]).all()))
  value=loss(pred,torch.rand(8,10));value.backward();self.assertTrue(torch.isfinite(value))
 def test_future_label_excludes_current(self):
  import gzip
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'x.gz'
   with gzip.open(p,'wt') as f:
    for i in range(45):f.write(json.dumps({'time':i*.1,'load':[float(i)]*10,'queue':[0.]*10})+'\n')
   x,y,t=windows(p);self.assertEqual(x[29,-1,0],29);self.assertEqual(y[29,0],34.5);self.assertEqual(y[0,0],5.5)

class Native(unittest.TestCase):
 def test_drr_weight_changes_real_service(self):
  binary=s.b.p.NS3/'build/scratch/allocation-final-scenario'
  values=[]
  for w in (1,4):
   r=subprocess.run([str(binary),f'--queueProbe={w}'],capture_output=True,text=True,check=True,env=dict(os.environ,LD_LIBRARY_PATH=str(s.b.p.NS3/'build/lib')))
   values.append(json.loads(r.stdout))
  self.assertLessEqual(abs(values[0]['slice1']-values[0]['slice2']),1)
  self.assertGreater(values[1]['slice1'],values[0]['slice1']+4)
  self.assertGreater(values[1]['slice2'],0)
  self.assertEqual(values[1]['remaining'],55)
 def test_native_three_modalities(self):
  with tempfile.TemporaryDirectory() as tmp:
   cfg=s.scenario(3011,8,'cross');result=s.episode(cfg,'native',prefix=Path(tmp)/'trial')
   self.assertEqual(result['requests'],8)
   summary=s.read(Path(tmp)/'trial-summary.json')
   self.assertEqual(len(summary['queues']),10)
   self.assertTrue(all(q['input']==q['served']+q['dropped']+q['remaining'] for q in summary['queues']))
   self.assertGreater(sum(q['input'] for q in summary['queues']),0)
   self.assertEqual(sum(q['unclassified'] for q in summary['queues']),0)
   observations=result['observations']
   for item in observations:
    obs=item['observation']
    self.assertTrue(all(row['time']<=obs['time']+1e-8 for row in obs['load_history']))
 def test_native_rl_training_and_reload(self):
  from slice_alloc.tools.study_run import HYPER
  from slice_alloc.allocation_value import MaskedDoubleDQN
  from slice_alloc.distributional_allocation import CandidateQRPolicy
  torch.set_num_threads(1)
  env=s.Env(seed=44);params={**HYPER,'learning_starts':16,'buffer_size':256,'batch_size':16}
  model=MaskedDoubleDQN(CandidateQRPolicy,env,**params,seed=44,device='cpu')
  before=model.quantile_net.deployment[-1].weight.detach().clone()
  try:
   model.learn(96)
   self.assertGreater(model._n_updates,0)
   self.assertFalse(torch.equal(before,model.quantile_net.deployment[-1].weight.detach()))
   with tempfile.TemporaryDirectory() as tmp:
    model.save(Path(tmp)/'model');loaded=MaskedDoubleDQN.load(Path(tmp)/'model',device='cpu')
    obs,_=env.reset(options={'scenario':s.scenario(3017,8)})
    self.assertEqual(int(model.predict(obs,deterministic=True)[0]),int(loaded.predict(obs,deterministic=True)[0]))
  finally:env.close()
 def test_no_motion_features_and_action_parity(self):
  cfg=s.scenario(3015,8,'cross')
  a=s.Env();n=s.Env(variant='no_motion')
  try:
   oa,_=a.reset(options={'scenario':cfg});on,_=n.reset(options={'scenario':cfg})
   self.assertTrue(np.array_equal(oa['mask'],on['mask']))
   self.assertTrue(np.array_equal(oa['candidates'],on['candidates']))
   self.assertEqual(on['state'][s.b.STATE_DIM:s.b.STATE_DIM+96].sum(),0)
   self.assertFalse(np.array_equal(oa['state'],on['state']))
  finally:a.close();n.close()

if __name__=='__main__':unittest.main()
