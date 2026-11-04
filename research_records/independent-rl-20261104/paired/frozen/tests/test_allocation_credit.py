"""Native causal progress shaping and delayed-return conservation."""
import unittest
import numpy as np
from slice_alloc.allocation_credit import CreditEnv,base
from slice_alloc.distributional_allocation import NStepDictReplay
from slice_alloc.tools.allocation_simple_run import greedy_action


class CreditTests(unittest.TestCase):
    def test_native_progress_is_causal_and_conserves_success_return(self):
        results=[];rewards=[];largest=0.
        for weight in (0.,.5):
            env=CreditEnv(progress_weight=weight)
            try:
                obs,_=env.reset(options={'scenario':base.scenario(449999,24)});done=False;rr=[]
                self.assertEqual(env.potential(),0.)
                while not done:
                    potential=env.potential();largest=max(largest,potential)
                    # Unarrived contracts may change; the potential sees active counters only.
                    active={x['id'] for x in env.current['active']}
                    untouched={i:r for i,r in env.requests.items() if i not in active}
                    for i in untouched:env.requests[i]=None
                    self.assertEqual(env.potential(),potential)
                    env.requests.update(untouched)
                    obs,r,done,_,_=env.step(greedy_action(env));rr.append(r)
                self.assertEqual(env.potential(),0.)
                self.assertAlmostEqual(sum(rr),env.result()['success'])
                results.append(env.result());rewards.append(rr)
            finally:env.close()
        self.assertGreater(largest,0.)
        self.assertEqual(results[0]['decisions'],results[1]['decisions'])
        self.assertEqual(results[0]['settled'],results[1]['settled'])
        self.assertFalse(np.allclose(rewards[0],rewards[1]))

    def test_long_replay_flushes_real_delayed_rewards_at_terminal(self):
        env=CreditEnv()
        try:
            replay=NStepDictReplay(100,env.observation_space,env.action_space,device='cpu',n_envs=1,horizon=32,discount=1.)
            obs={k:np.zeros((1,*space.shape),np.float32) for k,space in env.observation_space.spaces.items()}
            obs['mask'][:,0]=1
            for i in range(12):replay.add(obs,obs,np.array([0]),np.array([float(i==11)]),np.array([i==11]),[{}])
            self.assertEqual(replay.size(),12)
            np.testing.assert_array_equal(replay.rewards[:12,0],np.ones(12))
            np.testing.assert_array_equal(replay.dones[:12,0],np.ones(12))
            self.assertFalse(replay.pending)
        finally:env.close()

if __name__=='__main__':unittest.main()
