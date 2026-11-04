"""Masked PPO actor scores each available configuration with shared weights."""
import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
from slice_alloc.distributional_allocation import CandidateQuantileNetwork


class CandidateActor(nn.Module):
    def __init__(self,observation_space,action_space):
        super().__init__()
        self.shapes={key:space.shape for key,space in observation_space.spaces.items()}
        stripped=spaces.Dict({key:space for key,space in observation_space.spaces.items() if key!='prior'})
        extractor=CombinedExtractor(stripped)
        self.scorer=CandidateQuantileNetwork(stripped,action_space,extractor,extractor.features_dim,
            n_quantiles=1,net_arch=[96],normalize_images=False)
        for layer in (self.scorer.deployment[-1],self.scorer.rejection[-1]):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    @property
    def residual(self):
        return self.scorer.deployment

    def forward(self,features):
        obs={};offset=0
        for key,shape in self.shapes.items():
            size=int(np.prod(shape));obs[key]=features[:,offset:offset+size].reshape(-1,*shape);offset+=size
        return self.scorer(obs).squeeze(1)+2.5*obs['prior']


class CandidateMissionPolicy(MaskableMultiInputActorCriticPolicy):
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        assert self.mlp_extractor.latent_dim_pi==self.features_dim
        self.action_net=CandidateActor(self.observation_space,self.action_space)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)
