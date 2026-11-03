"""Scalar masked Double-DQN and replay invariants for the final allocator."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.distributional_allocation import CandidateQRPolicy, NStepDictReplay

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


class ScalarTests(unittest.TestCase):
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

    def test_nstep_terminal_flush_keeps_episodes_separate(self):
        model=self.dqn()
        buf=model.replay_buffer
        obs={k:v[None] for k,v in TinyCandidates().observation().items()}
        for i in range(4):
            buf.add(obs,obs,np.array([1]),np.array([i+1.]),np.array([i==3]),[{}])
        np.testing.assert_array_equal(buf.rewards[:4,0],[6,9,7,4])
        np.testing.assert_array_equal(buf.dones[:4,0],[0,1,1,1])
        self.assertEqual(len(buf.pending),0)
        buf.add(obs,obs,np.array([0]),np.array([100.]),np.array([True]),[{}])
        self.assertEqual(buf.rewards[4,0],100.)

    def test_candidate_order_does_not_change_action_values(self):
        model=self.dqn()
        obs=TinyCandidates().observation()
        tensor,_=model.policy.obs_to_tensor(obs)
        order=torch.tensor([0,3,1,2])
        changed={**tensor,'candidates':tensor['candidates'][:,order],'mask':tensor['mask'][:,order]}
        torch.testing.assert_close(model.quantile_net(changed),model.quantile_net(tensor)[:,:,order],atol=1e-6,rtol=1e-5)

if __name__=='__main__':
    unittest.main()
