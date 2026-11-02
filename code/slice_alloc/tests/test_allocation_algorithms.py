"""Learning, legal actions and persistence for additional RL candidates."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from stable_baselines3 import A2C, PPO

from slice_alloc.allocation_final_policy import CandidatePPOPolicy
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.distributional_allocation import CandidateQRPolicy, NStepDictReplay
from slice_alloc.tools import allocation_final_run as run


class TinyCandidates(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(dict(state=spaces.Box(-1, 1, (4,), np.float32),
            candidates=spaces.Box(-1, 1, (4, 2), np.float32), mask=spaces.Box(0, 1, (4,), np.float32)))
        self.action_space = spaces.Discrete(4)
        self.actions = []

    def observation(self):
        return dict(state=np.zeros(4, np.float32),
            candidates=np.array([[0, 0], [1, 0], [0, 1], [0, 0]], np.float32),
            mask=np.array([1, 1, 0, 0], np.float32))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        return self.observation(), {}

    def step(self, action):
        if action not in (0, 1):
            raise AssertionError('An infeasible action reached the environment')
        self.actions.append(int(action))
        self.time += 1
        return self.observation(), float(action == 1), self.time == 4, False, {}


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def dqn(self, **kwargs):
        return MaskedDoubleDQN(CandidateQRPolicy, TinyCandidates(),
            buffer_size=256, learning_starts=16, batch_size=16, train_freq=4,
            target_update_interval=32, exploration_steps=64, learning_rate=1e-3,
            replay_buffer_class=NStepDictReplay, replay_buffer_kwargs=dict(horizon=3, discount=1.),
            policy_kwargs=dict(net_arch=[16], **kwargs), seed=11, device='cpu')

    def test_scalar_double_target_masks_online_choice_and_evaluates_target(self):
        model = self.dqn()
        class Fixed(nn.Module):
            def __init__(self, values):
                super().__init__()
                self.values = values
            def forward(self, obs):
                return torch.tensor(self.values).float()[None, None].expand(2, 1, -1)
        model.quantile_net = Fixed([0, 8, 5, 999])
        model.quantile_net_target = Fixed([1, 3, 99, 999])
        replay = SimpleNamespace(next_observations={'mask': torch.tensor([[1, 1, 1, 0]]*2)},
            rewards=torch.tensor([[2.], [6.]]), dones=torch.tensor([[0.], [1.]]),
            discounts=torch.tensor([[.5], [.5]]))
        torch.testing.assert_close(model.bootstrap_quantiles(replay), torch.tensor([[3.5], [6.]]))

    def test_scalar_network_rejects_multiple_outputs(self):
        with self.assertRaises(ValueError):
            self.dqn(n_quantiles=2)

    def test_scalar_exploration_and_greedy_respect_mask(self):
        model = self.dqn()
        obs = TinyCandidates().observation()
        obs['mask'][:] = 0
        obs['mask'][0] = 1
        for epsilon in (0., 1.):
            model.exploration_rate = epsilon
            self.assertTrue(all(int(model.predict(obs)[0]) == 0 for _ in range(20)))

    def test_scalar_real_update_and_exact_model_roundtrip(self):
        model = self.dqn()
        before = model.quantile_net.deployment[-1].weight.detach().clone()
        model.learn(128)
        self.assertGreater(model._n_updates, 0)
        self.assertFalse(torch.equal(before, model.quantile_net.deployment[-1].weight))
        obs = TinyCandidates().observation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.zip'
            model.save(path)
            again = MaskedDoubleDQN.load(path, device='cpu')
        self.assertEqual(again.n_quantiles, 1)
        tensor, _ = model.policy.obs_to_tensor(obs)
        torch.testing.assert_close(model.quantile_net(tensor), again.quantile_net(tensor), atol=0, rtol=0)

    def test_a2c_learns_with_masked_candidate_policy_and_reloads(self):
        env = TinyCandidates()
        model = A2C(CandidatePPOPolicy, env, n_steps=16, gamma=1., seed=11, device='cpu')
        before = model.policy.actor[-1].weight.detach().clone()
        model.learn(128)
        self.assertTrue(env.actions and set(env.actions) <= {0, 1})
        self.assertFalse(torch.equal(before, model.policy.actor[-1].weight))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'a2c.zip'
            model.save(path)
            again = A2C.load(path, device='cpu')
        obs = env.observation()
        self.assertEqual(int(model.predict(obs, deterministic=True)[0]),
                         int(again.predict(obs, deterministic=True)[0]))

    def test_registry_distinguishes_scalar_and_distributional_training(self):
        self.assertEqual(run.ALGORITHMS, ('candidate_ppo',))
        self.assertIs(run.MODEL_CLASSES['candidate_ppo'], PPO)
        self.assertTrue(run.POLICIES['candidate_ppo'].__name__.endswith('HierarchicalPPOPolicy'))

    def test_baseline_choice_is_frozen_before_new_test(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run, 'OUT', Path(directory)):
            protocol = run.freeze()
            self.assertEqual(protocol['primary_rule'], 'greedy')
            self.assertIn('greedy', protocol['main_rules'])
            self.assertEqual(set(protocol['supplemental_rules']), {'price', 'semantic'})
            self.assertTrue(set(protocol['test']).isdisjoint(range(43000, 43030)))
            with patch.object(run, 'PRIMARY_RULE', 'price'):
                with self.assertRaises(AssertionError):
                    run.freeze()


if __name__ == '__main__':
    unittest.main()
