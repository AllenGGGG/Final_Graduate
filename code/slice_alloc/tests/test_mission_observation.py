"""Prevent future-profile leakage and preserve an admission-rule start."""
from copy import deepcopy
import unittest
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from slice_alloc import mission_allocation as m
from slice_alloc.mission_policy import MissionPolicy


class MissionObservationTests(unittest.TestCase):
    def test_private_profile_label_does_not_change_observations(self):
        cfg=m.scenario(329004,8);other=deepcopy(cfg)
        other['private']['demand_profile']='unavailable_future_label'
        envs=[m.MissionEnv(),m.MissionEnv()]
        try:
            obs=[env.reset(options={'scenario':c})[0] for env,c in zip(envs,(cfg,other))]
            done=False
            while not done:
                for key in obs[0]:np.testing.assert_array_equal(obs[0][key],obs[1][key])
                actions=[m.rule_action('modal_greedy',env) for env in envs]
                self.assertEqual(actions[0],actions[1])
                transitions=[env.step(a) for env,a in zip(envs,actions)]
                self.assertEqual(transitions[0][2],transitions[1][2])
                done=transitions[0][2];obs=[x[0] for x in transitions]
        finally:
            for env in envs:env.close()

    def test_zero_actor_matches_admission_price_including_rejection(self):
        torch.set_num_threads(1)
        env=m.MissionEnv(rule_parameters=(2.,.5,1.),admission_threshold=.25)
        try:
            model=MaskablePPO(MissionPolicy,env,n_steps=32,batch_size=16,
                policy_kwargs=dict(net_arch=dict(pi=[],vf=[96,96]),ortho_init=False),seed=11,device='cpu')
            obs,_=env.reset(options={'scenario':m.scenario(329005,24)})
            done=False;voluntary=0
            while not done:
                action=int(model.predict(obs,action_masks=obs['mask']>.5,deterministic=True)[0])
                self.assertEqual(action,m.rule_action('price',env,(2.,.5,1.),.25))
                voluntary+=int(action==0 and bool(env.options))
                obs,_,done,_,_=env.step(action)
            self.assertGreater(voluntary,0)
        finally:env.close()


if __name__=='__main__':unittest.main()
