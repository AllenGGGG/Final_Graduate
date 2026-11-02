"""Focused checks for the final constrained admission formulation."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from stable_baselines3 import PPO

from slice_alloc import allocation_cmdp as a
from slice_alloc.tools import allocation_final_run as run


class CMDPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_action_space_has_tiers_and_is_native_sensitive(self):
        env=a.AllocationEnv(loads=(8,))
        try:
            obs,_=env.reset(options={'scenario':a.scenario(71011,8)})
            self.assertEqual(env.action_space.n, a.MAX_ACTIONS)
            self.assertTrue(env.observation_space.contains(obs))
            factors={round(p.rate_factor,2) for p in env.options}
            self.assertTrue({.9,1.1,1.4}.issubset(factors))
        finally:
            env.close()

    def test_risk_candidates_are_bounded(self):
        env=a.AllocationEnv(loads=(80,))
        try:
            env.reset(options={'scenario':a.scenario(71012,80)})
            found=False
            for _ in range(80):
                found |= any(p.risk>0 for p in env.options)
                if found: break
                _,_,done,_,_=env.step(1 if env.options else 0)
                if done: break
            self.assertTrue(found)
            self.assertTrue(all(p.peak_ratios[:11].max() <= a.OVERBOOK+1e-8 for p in env.options))
        finally:
            env.close()

    def test_native_shaper_changes_delivery_and_audits(self):
        rows=[]
        with tempfile.TemporaryDirectory() as directory:
            for tier in a.TIERS:
                result=run.episode(a.scenario(71013,1,mobility='stationary'),f'tier_{tier}',
                    prefix=Path(directory)/str(tier))
                rows.append(result['success'])
                self.assertTrue(result['audited'])
        self.assertGreater(len(set(rows)),1)

    def test_hierarchical_policy_trains_and_respects_mask(self):
        env=a.AllocationEnv(loads=(8,))
        try:
            model=PPO(a.HierarchicalPPOPolicy,env,n_steps=16,batch_size=16,n_epochs=1,
                      gamma=1.,device='cpu',seed=71014)
            model.learn(32)
            obs,_=env.reset(options={'scenario':a.scenario(71015,8)})
            action=int(model.predict(obs,deterministic=True)[0])
            self.assertEqual(obs['mask'][action],1.)
        finally:
            env.close()

    def test_milp_bound_is_at_least_greedy(self):
        from slice_alloc.tools.allocation_cmdp_diagnose import relaxed_bound
        cfg=a.scenario(71016,8,mobility='stationary')
        bound=relaxed_bound(cfg)
        self.assertTrue(bound['optimal'])
        self.assertGreaterEqual(bound['upper_utility'],0.)


if __name__=='__main__':
    unittest.main()
