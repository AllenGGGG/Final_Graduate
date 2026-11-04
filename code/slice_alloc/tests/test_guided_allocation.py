"""Native checks for the rule prior, masks and actual TD learning."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from slice_alloc import guided_allocation as g
from slice_alloc import allocation_final as base
from slice_alloc.tools import allocation_simple_run as old


class GuidedTests(unittest.TestCase):
    def test_exact_rule_start_native_masks_and_reward(self):
        torch.set_num_threads(1)
        env=g.GuidedEnv(rule_parameters=(1.,.25,1.))
        try:
            model=g.GuidedDQN(g.PriorPolicy,env,**old.parameters(),seed=11,device='cpu')
            obs,_=env.reset(options={'scenario':base.scenario(229001,24)})
            done=False;reward=0.
            while not done:
                action=int(model.predict(obs,deterministic=True)[0])
                self.assertEqual(action,g.rule_action('modal_greedy',env,(1.,.25,1.)))
                self.assertEqual(obs['mask'][action],1.)
                obs,r,done,_,_=env.step(action);reward+=r
            self.assertEqual(reward,env.result()['success'])
        finally:env.close()

    def test_real_td_update_and_roundtrip(self):
        torch.set_num_threads(1)
        env=g.GuidedEnv(loads=(8,),rule_parameters=(1.,.25,1.))
        try:
            params=old.parameters();params.update(learning_starts=8,batch_size=8,buffer_size=128,exploration_initial_eps=.1)
            model=g.GuidedDQN(g.PriorPolicy,env,**params,seed=23,device='cpu')
            before=model.quantile_net.deployment[-1].weight.detach().clone()
            model.learn(64)
            self.assertGreater(model._n_updates,0)
            self.assertFalse(torch.equal(before,model.quantile_net.deployment[-1].weight))
            obs,_=env.reset(options={'scenario':base.scenario(229002,8)})
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'model.zip';model.save(path)
                again=g.GuidedDQN.load(path,device='cpu')
            tensor,_=model.policy.obs_to_tensor(obs)
            torch.testing.assert_close(model.quantile_net(tensor),again.quantile_net(tensor),atol=0,rtol=0)
        finally:env.close()


if __name__=='__main__':unittest.main()
