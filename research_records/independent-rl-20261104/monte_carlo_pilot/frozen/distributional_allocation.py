"""Masked Double QR-DQN for finite, variable sets of admissible actions.

The environment supplies constraints; the network learns every action value.
No resource-price score, admission preference or protocol rule enters the model.
"""
from collections import deque
import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from sb3_contrib import QRDQN
from sb3_contrib.common.utils import quantile_huber_loss
from sb3_contrib.qrdqn.policies import QRDQNPolicy, QuantileNetwork
from stable_baselines3.common.buffers import DictReplayBuffer


class RuleFreeObservation(gym.ObservationWrapper):
    """Discard the external rule's scalar before collection, replay or inference."""
    def __init__(self, env):
        super().__init__(env)
        from slice_alloc.initial_allocation import PRICE_INDEX
        self.price_index = PRICE_INDEX
        original = env.observation_space.spaces
        candidate = original['candidates']
        self.observation_space = spaces.Dict({**original, 'candidates': spaces.Box(
            np.delete(candidate.low, PRICE_INDEX, axis=-1),
            np.delete(candidate.high, PRICE_INDEX, axis=-1), dtype=np.float32)})

    def observation(self, observation):
        return {**observation, 'candidates': np.delete(observation['candidates'], self.price_index, axis=-1)}


class NStepDictReplay(DictReplayBuffer):
    """N-step returns for one dictionary-observation environment, with terminal flush."""
    def __init__(self, *args, horizon=3, discount=1., **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1 or horizon < 1 or not 0 < discount <= 1:
            raise ValueError('Require one environment, positive horizon and discount in (0,1]')
        self.horizon, self.discount = int(horizon), float(discount)
        self.pending = deque()
        self.return_discounts = np.zeros((self.buffer_size, 1), np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        if any(info.get('TimeLimit.truncated', False) for info in infos):
            raise ValueError('This replay requires real terminal states, not time-limit truncation')
        self.pending.append(copy.deepcopy((obs, next_obs, action, reward, done, infos)))
        if len(self.pending) >= self.horizon:
            self._commit()
        if bool(done[0]):
            while self.pending:
                self._commit()

    def _commit(self):
        window = list(self.pending)[:self.horizon]
        first, last = window[0], window[-1]
        reward = sum(self.discount**i * item[3] for i, item in enumerate(window))
        self.return_discounts[self.pos] = self.discount**len(window)
        super().add(first[0], last[1], first[2], reward, last[4], last[5])
        self.pending.popleft()

    def _get_samples(self, batch_inds, env=None):
        samples = super()._get_samples(batch_inds, env)
        return samples._replace(discounts=self.to_torch(self.return_discounts[batch_inds]))


class CandidateQuantileNetwork(QuantileNetwork):
    """Shared action encoder and invariant pooling, independent of protocol labels."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantile_net = nn.Identity()
        width = self.net_arch[0] if self.net_arch else 128
        schema = self.observation_space.spaces
        self.context_keys = tuple(k for k in sorted(schema) if k not in ('candidates', 'mask', 'needs'))
        context_dim = sum(int(np.prod(schema[k].shape)) for k in self.context_keys)
        action_dim = schema['candidates'].shape[-1]
        if 'needs' in schema:
            action_dim += int(np.prod(schema['needs'].shape[1:]))
        self.context_encoder = nn.Sequential(nn.Linear(context_dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.context_fusion = nn.Sequential(nn.Linear(3*width, width), nn.ReLU())
        self.deployment = nn.Sequential(nn.Linear(2*width, width), nn.ReLU(), nn.Linear(width, self.n_quantiles))
        self.rejection = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, self.n_quantiles))

    def forward(self, obs):
        context = self.context_encoder(torch.cat([obs[k].float().flatten(1) for k in self.context_keys], dim=-1))
        actions = obs['candidates'][:, 1:].float()
        if 'needs' in obs:
            actions = torch.cat([actions, obs['needs'][:, 1:].float().flatten(2)], dim=-1)
        encoded = self.action_encoder(actions)
        valid = obs['mask'][:, 1:] > .5
        mean = (encoded*valid[:, :, None]).sum(1)/valid.sum(1, keepdim=True).clamp(min=1)
        maximum = encoded.masked_fill(~valid[:, :, None], -1e9).max(1).values
        maximum = torch.where(valid.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        context = self.context_fusion(torch.cat([context, mean, maximum], dim=-1))
        values = self.deployment(torch.cat([context[:, None].expand(-1, encoded.shape[1], -1), encoded], dim=-1))
        return torch.cat([self.rejection(context)[:, None], values], dim=1).transpose(1, 2)

    def _predict(self, observation, deterministic=True):
        means = self(observation).mean(1).masked_fill(observation['mask'] <= .5, -torch.inf)
        return means.argmax(1)


class CandidateQRPolicy(QRDQNPolicy):
    def make_quantile_net(self):
        args = self._update_features_extractor(self.net_args, features_extractor=None)
        return CandidateQuantileNetwork(**args).to(self.device)


class MaskedDoubleQRDQN(QRDQN):
    """Masks apply to exploration, greedy inference and Double-Q bootstrap."""
    def __init__(self, *args, exploration_steps=16384, **kwargs):
        self.exploration_steps = int(exploration_steps)
        if self.exploration_steps <= 0:
            raise ValueError('exploration_steps must be positive')
        if 'policy_kwargs' in kwargs:
            kwargs['policy_kwargs'] = copy.deepcopy(kwargs['policy_kwargs'])
        super().__init__(*args, **kwargs)

    def _on_step(self):
        super()._on_step()
        fraction = min(1., self.num_timesteps/self.exploration_steps)
        self.exploration_rate = self.exploration_initial_eps + fraction*(self.exploration_final_eps-self.exploration_initial_eps)
        self.logger.record('rollout/exploration_rate', self.exploration_rate)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        tensor, vectorized = self.policy.obs_to_tensor(observation)
        if not deterministic and np.random.rand() < self.exploration_rate:
            masks = tensor['mask'].detach().cpu().numpy()
            actions = np.array([np.random.choice(np.flatnonzero(row > .5)) for row in masks])
            return (actions if vectorized else actions.reshape(())), state
        return self.policy.predict(observation, state, episode_start, deterministic)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            actions = np.array([np.random.choice(np.flatnonzero(row > .5)) for row in self._last_obs['mask']])
        else:
            actions, _ = self.predict(self._last_obs, deterministic=False)
        return actions, actions.copy()

    def bootstrap_quantiles(self, replay):
        with torch.no_grad():
            # Selection uses online estimates; evaluation uses the target network.
            online = self.quantile_net(replay.next_observations).mean(1)
            online = online.masked_fill(replay.next_observations['mask'] <= .5, -torch.inf)
            actions = online.argmax(1)[:, None, None].expand(-1, self.n_quantiles, 1)
            future = self.quantile_net_target(replay.next_observations).gather(2, actions).squeeze(2)
            discount = replay.discounts if replay.discounts is not None else self.gamma
            return replay.rewards + (1-replay.dones)*discount*future

    def train(self, gradient_steps, batch_size=128):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            target = self.bootstrap_quantiles(replay)
            actions = replay.actions[..., None].long().expand(-1, self.n_quantiles, 1)
            current = self.quantile_net(replay.observations).gather(2, actions).squeeze(2)
            loss = quantile_huber_loss(current, target, sum_over_quantiles=True)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite quantile regression loss')
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(float(loss.detach()))
        self._n_updates += gradient_steps
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/loss', float(np.mean(losses)))
