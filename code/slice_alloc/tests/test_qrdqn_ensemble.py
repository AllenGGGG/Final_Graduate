"""Generic masking and permutation behavior for learned-value aggregation."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from slice_alloc.tools.qrdqn_ensemble_probe import Ensemble

class LearnedValueEnsembleTests(unittest.TestCase):
 def model(self,values):
  return SimpleNamespace(policy=SimpleNamespace(obs_to_tensor=lambda obs:(obs,False)),
   quantile_net=lambda obs:torch.tensor(values,dtype=torch.float32)[None,None].expand(1,8,-1))
 def test_value_average_and_vote_respect_masks(self):
  models=[self.model([0,2,100]),self.model([0,4,200])]
  obs=dict(mask=np.array([1,1,0],dtype=np.float32))
  for method in ('mean','vote'):
   action,_=Ensemble(models,method).predict(obs)
   self.assertEqual(int(action),1)
  obs['mask'][:]=[1,0,0]
  self.assertEqual(int(Ensemble(models,'mean').predict(obs)[0]),0)
 def test_mean_and_vote_are_distinct_predeclared_aggregators(self):
  models=[self.model([0,2,1]),self.model([0,2,1]),self.model([0,0,10])]
  obs=dict(mask=np.ones(3,dtype=np.float32))
  self.assertEqual(int(Ensemble(models,'mean').predict(obs)[0]),2)
  self.assertEqual(int(Ensemble(models,'vote').predict(obs)[0]),1)
 def test_action_permutation_equivariance(self):
  values=np.array([[0,3,1,2,4],[0,2,3,1,5]])
  order=np.array([0,3,1,4,2]);obs=dict(mask=np.ones(5,dtype=np.float32))
  original=int(Ensemble([self.model(v) for v in values],'mean').predict(obs)[0])
  permuted=int(Ensemble([self.model(v[order]) for v in values],'mean').predict(obs)[0])
  self.assertEqual(original,int(order[permuted]))

if __name__=='__main__':unittest.main()
