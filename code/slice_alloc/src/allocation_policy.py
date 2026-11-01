"""Resource-prior PPO for initial placement, with an explicit no-learning ablation.

The prior is a public, current-state resource-price formula, not an oracle or
demonstration dataset.  PPO learns a residual over it from actual SLA settlements.
"""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution


class ResourcePriorPolicy(ActorCriticPolicy):
    def __init__(self, *args, prior_scale=4., reject_prior=-6., **kwargs):
        self.prior_scale = prior_scale
        self.reject_prior = reject_prior
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        sd = self.observation_space['state'].shape[0]
        cd = self.observation_space['candidates'].shape[1]
        self.state_encoder = nn.Sequential(nn.Linear(sd, 96), nn.Tanh(), nn.Linear(96, 64), nn.Tanh())
        self.candidate_encoder = nn.Sequential(nn.Linear(cd, 64), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Exact zero residual at initialization: deterministic policy equals price.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.orthogonal_(self.critic[-1].weight, 1.)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _outputs(self, obs):
        raw = obs['candidates'].float()
        mask = obs['mask'] > .5
        s = self.state_encoder(obs['state'].float())
        c = self.candidate_encoder(raw)
        pairs = torch.cat([s[:, None].expand(-1, c.shape[1], -1), c], -1)
        logits = self.actor(pairs).squeeze(-1)
        if self.prior_scale:
            # Last three fields: pre-admission price, observed reuse, duration.
            costs = raw[:, :, -3]
            valid = mask.clone()
            valid[:, 0] = False
            best = costs.masked_fill(~valid, 1e6).min(1, keepdim=True).values
            prior = -self.prior_scale * (costs - best)
            prior[:, 0] = self.reject_prior
            logits = logits + prior
        logits = logits.masked_fill(~mask, -1e9)
        pooled = (c * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        value = self.critic(torch.cat([s, pooled], -1))
        return CategoricalDistribution(self.action_space.n).proba_distribution(logits), value

    def forward(self, obs, deterministic=False):
        distribution, values = self._outputs(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, values, distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        distribution, values = self._outputs(obs)
        return values, distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        return self._outputs(obs)[0]

    def predict_values(self, obs):
        return self._outputs(obs)[1]
