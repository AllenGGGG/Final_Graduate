"""Candidate-content actor with masking in sampling, PPO updates and inference.

Uses the installed SB3 PPO optimizer; does not require sb3-contrib. Slot order
has no learned embedding. A permuted candidate set permutes actor probabilities.
"""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution


class CandidatePolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        state_dim = self.observation_space["state"].shape[0]
        candidate_dim = self.observation_space["candidates"].shape[1]
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.candidate_encoder = nn.Sequential(nn.Linear(candidate_dim, 64), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2)); nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _outputs(self, obs):
        state = self.state_encoder(obs["state"].float())
        candidates = self.candidate_encoder(obs["candidates"].float())
        mask = obs["mask"] > .5
        if not bool(torch.all(mask.any(dim=1))):
            raise ValueError("every state must have at least the reject action")
        expanded = state.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        logits = self.actor(torch.cat([expanded, candidates], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        pooled = (candidates*mask.unsqueeze(-1)).sum(dim=1)/mask.sum(dim=1, keepdim=True)
        values = self.critic(torch.cat([state, pooled], dim=-1))
        return CategoricalDistribution(self.action_space.n).proba_distribution(logits), values

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
