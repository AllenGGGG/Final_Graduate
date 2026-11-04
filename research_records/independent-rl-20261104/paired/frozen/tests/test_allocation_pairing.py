import unittest
import numpy as np
import torch
from slice_alloc.allocation_pairing import PairedEnv,PairedReplay,PairedDoubleDQN,observation_key
from slice_alloc.distributional_allocation import CandidateQRPolicy


class PairingTests(unittest.TestCase):
    def test_native_replay_prefix_and_real_chosen_actions(self):
        torch.set_num_threads(1);env=PairedEnv(seed=23,loads=(16,))
        try:
            model=PairedDoubleDQN(CandidateQRPolicy,env,learning_starts=16,buffer_size=256,batch_size=16,
                train_freq=4,policy_kwargs=dict(net_arch=[16]),replay_buffer_class=PairedReplay,
                replay_buffer_kwargs=dict(horizon=8,discount=1.),seed=11,device='cpu')
            model.learn(128)
            self.assertGreater(model._n_updates,0)
            self.assertGreater(len(env.pair_proofs),0)
            self.assertTrue(all(81000000<=p['seed']<82000000 for p in env.pair_proofs))
            self.assertGreater(len(model.replay_buffer.paired_keys),0)
        finally:env.close()

    def test_paired_slots_remain_valid_after_ring_overwrites(self):
        env=PairedEnv()
        try:
            buf=PairedReplay(50,env.observation_space,env.action_space,n_envs=1,device='cpu',horizon=1,discount=1.)
            for turn in range(240):
                obs={k:np.zeros((1,*s.shape),np.float32) for k,s in env.observation_space.spaces.items()}
                obs['state'][0,0]=(turn//2)%20
                obs['mask'][0,:2]=1;act=turn%2
                buf.add(obs,obs,np.array([act]),np.array([act],float),np.array([True]),[{}])
                for key in buf.paired_keys:
                    for a,pos in buf.state_actions[key].items():
                        self.assertEqual(int(buf.actions[pos,0,0]),a)
                        current={k:v[pos] for k,v in buf.observations.items()}
                        self.assertEqual(observation_key(current),key)
            self.assertEqual(buf.sample(32).actions.shape,(32,1))
            self.assertGreater(buf.paired_batches,0)
        finally:env.close()

if __name__=='__main__':unittest.main()
