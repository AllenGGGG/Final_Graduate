"""Standard masked PPO with a transparent rule-initialized actor.

No outcome model, future observations or expert labels enter inference.
"""
import numpy as np
import torch
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy


class PriorActor(nn.Module):
    def __init__(self,features_dim,prior_start,prior_size):
        super().__init__();self.prior_start=prior_start;self.prior_size=prior_size
        self.residual=nn.Sequential(nn.Linear(features_dim,96),nn.Tanh(),
            nn.Linear(96,96),nn.Tanh(),nn.Linear(96,prior_size))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)

    def forward(self,features):
        return self.residual(features)+2.5*features[:,self.prior_start:self.prior_start+self.prior_size]


class MissionPolicy(MaskableMultiInputActorCriticPolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        offset=0
        for key,space in self.observation_space.spaces.items():
            if key=='prior':break
            offset+=int(np.prod(space.shape))
        assert self.mlp_extractor.latent_dim_pi==self.features_dim
        self.action_net=PriorActor(self.features_dim,offset,self.action_space.n)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)
