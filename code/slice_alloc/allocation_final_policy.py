"""PPO with shared candidate scoring and masks, without an expert prior."""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution


class CandidatePPOPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        width = 96
        dim = self.observation_space['state'].shape[0]
        candidate_dim = self.observation_space['candidates'].shape[-1]
        self.context = nn.Sequential(nn.Linear(dim, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.candidate = nn.Sequential(nn.Linear(candidate_dim, width), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(3*width, width), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(2*width, width), nn.Tanh(), nn.Linear(width, 1))
        self.reject = nn.Linear(width, 1)
        self.critic = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.zeros_(module.bias)
        for layer in (self.actor[-1], self.reject):
            nn.init.orthogonal_(layer.weight, .01)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _outputs(self, obs):
        context = self.context(obs['state'].float())
        candidates = self.candidate(obs['candidates'][:, 1:].float())
        mask = obs['mask'][:, 1:] > .5
        mean = (candidates*mask[..., None]).sum(1)/mask.sum(1, keepdim=True).clamp(min=1)
        maximum = candidates.masked_fill(~mask[..., None], -1e9).max(1).values
        maximum = torch.where(mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        context = self.fusion(torch.cat([context, mean, maximum], -1))
        pairs = torch.cat([context[:, None].expand(-1, candidates.shape[1], -1), candidates], -1)
        logits = torch.cat([self.reject(context), self.actor(pairs).squeeze(-1)], -1)
        logits = logits.masked_fill(obs['mask'] <= .5, -torch.inf)
        dist = CategoricalDistribution(self.action_space.n).proba_distribution(logits)
        return dist, self.critic(context)

    def forward(self, obs, deterministic=False):
        dist, value = self._outputs(obs)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, value, dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        dist, value = self._outputs(obs)
        return value, dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        return self._outputs(obs)[0]

    def predict_values(self, obs):
        return self._outputs(obs)[1]
