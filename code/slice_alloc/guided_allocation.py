"""Rule-initialized scalar Double-DQN, with unchanged native packet physics.

The rule is an explicit prior, not an expert label or a guaranteed policy bound.
Delayed native rewards are left intact: arbitrary relabelling is not RUDDER.
"""
import numpy as np
import torch
from torch import nn
from gymnasium import spaces

from slice_alloc import allocation_final as base
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.distributional_allocation import CandidateQuantileNetwork, CandidateQRPolicy


def costs(env, parameters):
    """Causal modal-aware rule: resource cost, observed mobility and caching."""
    beta, cache_penalty, pressure_weight = parameters
    if not env.options:
        return np.empty(0, dtype=np.float32)
    request = env.requests[env.current['request_id']]
    speed = np.mean([np.linalg.norm(env.velocities[str(n)][:2]) for n in request.members])
    # Use the already committed ledger, never future arrivals or private motion.
    committed = sum((p.peak for p in env.ledger.plans), np.zeros_like(env.ledger.capacity))
    pressure = np.minimum(2., committed / env.ledger.capacity)
    answer = []
    for plan in env.options:
        normalized = plan.peak / env.ledger.capacity
        resource = float(np.sum(normalized * (1. + pressure_weight * pressure)))
        risk = float(plan.request.mode == 'ip' and request.service != 'telemetry')
        risk *= min(1., speed * (request.stop - request.start) / 120.)
        no_cache = request.service == 'object' and not (plan.request.mode == 'ndn' and plan.cache)
        answer.append(resource + beta * risk + cache_penalty * no_cache)
    return np.asarray(answer, dtype=np.float32)


def rule_action(name, env, parameters=(0., 0., 0.)):
    if not env.options:
        return 0
    if name == 'first_fit':
        return 1
    if name == 'nearest':
        r = env.requests[env.current['request_id']]
        aps = [env.current['associations'][str(n)] - 2 for n in r.members]
        return min(range(len(env.options)), key=lambda i: (
            sum(env.options[i].edge != ap for ap in aps), costs(env, (0., 0., 0.))[i])) + 1
    if name == 'native':
        r = env.requests[env.current['request_id']]
        desired = {'telemetry': 'ip', 'command': 'id', 'object': 'ndn'}[r.service]
        eligible = [i for i, p in enumerate(env.options) if p.request.mode == desired]
        # No native configuration available: admit with another compatible one.
        eligible = eligible or list(range(len(env.options)))
        return min(eligible, key=lambda i: (
            r.service == 'object' and not env.options[i].cache,
            costs(env, (0., 0., 0.))[i])) + 1
    if name not in ('greedy', 'modal_greedy'):
        raise ValueError(name)
    return int(np.argmin(costs(env, parameters if name == 'modal_greedy' else (0., 0., 0.)))) + 1


class GuidedEnv(base.AllocationEnv):
    def __init__(self, *args, rule_parameters=(0., 0., 0.), **kwargs):
        self.rule_parameters = tuple(rule_parameters)
        kwargs.setdefault('no_calendar', True)
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Dict({**self.observation_space.spaces,
            'prior': spaces.Box(-100., 100., (11,), dtype=np.float32)})

    def encode(self):
        obs = super().encode()
        prior = np.zeros(11, dtype=np.float32)
        if self.current['kind'] != 'done' and self.options:
            scores = costs(self, self.rule_parameters)
            prior[1:1+len(scores)] = -np.clip(scores - scores.min(), 0., 50.)
            prior[0] = -1.
        return {**obs, 'prior': prior}


class PriorNetwork(CandidateQuantileNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Exact initial policy parity with the tuned rule, including tie order.
        for layer in (self.deployment[-1], self.rejection[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, obs):
        return super().forward(obs) + obs['prior'][:, None, :]


class PriorPolicy(CandidateQRPolicy):
    def make_quantile_net(self):
        args = self._update_features_extractor(self.net_args, features_extractor=None)
        return PriorNetwork(**args).to(self.device)


class GuidedDQN(MaskedDoubleDQN):
    """Standard masked Double-Q TD, conservative exploration around rule start."""
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            actions = self._last_obs['prior'].copy()
            actions[self._last_obs['mask'] <= .5] = -np.inf
            selected = actions.argmax(1)
            return selected, selected.copy()
        return super()._sample_action(learning_starts, action_noise, n_envs)
