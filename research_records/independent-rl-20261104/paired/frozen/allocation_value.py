"""Scalar Double-DQN using the same masked candidate encoder and replay."""
import copy

import numpy as np
import torch
from torch.nn import functional as F

from slice_alloc.distributional_allocation import MaskedDoubleQRDQN


class MaskedDoubleDQN(MaskedDoubleQRDQN):
    """Expected scalar returns trained with the Huber temporal-difference loss.

    The one-output candidate network reuses the QR implementation's encoder;
    this class replaces quantile regression with ordinary scalar TD learning.
    """
    def __init__(self, *args, **kwargs):
        policy = copy.deepcopy(kwargs.get('policy_kwargs', {}))
        if policy.get('n_quantiles', 1) != 1:
            raise ValueError('Scalar Double-DQN requires exactly one output')
        policy['n_quantiles'] = 1
        kwargs['policy_kwargs'] = policy
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps, batch_size=128):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            target = self.bootstrap_quantiles(replay)
            current = self.quantile_net(replay.observations).gather(
                2, replay.actions.long()[:, None]).squeeze(2)
            loss = F.smooth_l1_loss(current, target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite scalar TD loss')
            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(float(loss.detach()))
        self._n_updates += gradient_steps
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/loss', float(np.mean(losses)))
