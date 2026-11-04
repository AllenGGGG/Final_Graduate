import unittest
import numpy as np
from slice_alloc.allocation_credit import CreditEnv
from slice_alloc.allocation_mc import MonteCarloDictReplay

class MonteCarloTests(unittest.TestCase):
 def test_terminal_return_to_go_flushes_all_transitions(self):
  env=CreditEnv(loads=(8,))
  try:
   buf=MonteCarloDictReplay(32,env.observation_space,env.action_space,device='cpu',n_envs=1)
   obs={k:np.zeros((1,*s.shape),np.float32) for k,s in env.observation_space.spaces.items()};obs['mask'][:,0]=1
   for i in range(4):buf.add(obs,obs,np.array([0]),np.array([float(i+1)]),np.array([i==3]),[{}])
   # The replay buffer commits the reversed episode in insertion order; each
   # stored reward is still the exact return-to-go for its transition.
   np.testing.assert_array_equal(buf.rewards[:4,0],[4,7,9,10]);np.testing.assert_array_equal(buf.dones[:4,0],1)
   self.assertEqual(buf.returns_committed,4);self.assertFalse(buf.pending)
  finally:env.close()

 def test_returns_stay_with_their_own_action_and_episode(self):
  env=CreditEnv(loads=(8,))
  try:
   buf=MonteCarloDictReplay(32,env.observation_space,env.action_space,device='cpu',n_envs=1)
   def add(marker,reward,done):
    obs={k:np.zeros((1,*s.shape),np.float32) for k,s in env.observation_space.spaces.items()}
    obs['state'][0,0]=marker;obs['mask'][:,0]=1
    buf.add(obs,obs,np.array([marker]),np.array([reward]),np.array([done]),[{}])
   add(1,2.,False);self.assertEqual(buf.size(),0)
   add(2,5.,True);add(3,11.,True)
   labels={int(buf.observations['state'][i,0,0]):(int(buf.actions[i,0,0]),float(buf.rewards[i,0])) for i in range(3)}
   self.assertEqual(labels,{1:(1,7.),2:(2,5.),3:(3,11.)})
   samples=buf._get_samples(np.arange(3))
   self.assertTrue(bool((samples.dones==1).all()))
   self.assertTrue(bool((samples.discounts==0).all()))
  finally:env.close()
if __name__=='__main__':unittest.main()
