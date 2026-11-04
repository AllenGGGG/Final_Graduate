"""Native contract and training checks for the simplified allocation runner."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from slice_alloc import allocation_final as a
from slice_alloc.tools import allocation_simple_run as run
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.distributional_allocation import CandidateQRPolicy


class SimpleAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_sla_reward_and_audit_for_every_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            for method in ("greedy", "nearest", "first_fit"):
                result = run.episode(run.scenario(990011, 8), method, prefix=Path(folder) / method)
                self.assertTrue(result["audited"])
                self.assertGreater(result["success"], 0)
                self.assertTrue(all("rate_factor" not in r["decision"] for r in result["decisions"]))

    def test_physical_mask_and_business_compatibility(self):
        env = a.AllocationEnv(no_calendar=True)
        try:
            obs, _ = env.reset(options={"scenario": run.scenario(990012, 24)})
            done = False
            self.assertEqual(env.action_space.n, 11)
            while not done:
                request = env.requests[env.current["request_id"]]
                self.assertTrue(all(p.request.mode in a.COMPATIBLE[request.service] for p in env.options))
                self.assertTrue(all(env.ledger.feasible(p) for p in env.options))
                self.assertTrue(env.observation_space.contains(obs))
                action = run.greedy_action(env)
                self.assertEqual(obs["mask"][action], 1.)
                obs, _, done, _, _ = env.step(action)
        finally:
            env.close()

    def test_native_double_dqn_training_and_reload(self):
        env = a.AllocationEnv(seed=990013, loads=(8,), no_calendar=True)
        try:
            params = run.parameters()
            params.update(learning_starts=8, batch_size=8, buffer_size=128,
                          target_update_interval=16, exploration_steps=64)
            params["policy_kwargs"]["net_arch"] = [16]
            model = MaskedDoubleDQN(CandidateQRPolicy, env, **params, seed=11, device="cpu")
            before = model.quantile_net.deployment[-1].weight.detach().clone()
            model.learn(64)
            self.assertFalse(torch.equal(before, model.quantile_net.deployment[-1].weight))
            self.assertGreater(model._n_updates, 0)
            obs, _ = env.reset(options={"scenario": run.scenario(990014, 8)})
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "model.zip"
                model.save(path)
                again = MaskedDoubleDQN.load(path, device="cpu")
            self.assertEqual(int(model.predict(obs, deterministic=True)[0]),
                             int(again.predict(obs, deterministic=True)[0]))
        finally:
            env.close()

    def test_freeze_detects_actual_training_parameter_changes(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(run, "OUT", Path(folder)):
            protocol = run.freeze()
            self.assertEqual(protocol["hyperparameters"]["policy_kwargs"]["n_quantiles"], 1)
            with patch.dict(run.HYPER, learning_rate=.123):
                with self.assertRaises(AssertionError):
                    run.freeze()


if __name__ == "__main__":
    unittest.main()
