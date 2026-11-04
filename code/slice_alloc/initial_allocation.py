"""First-part model: native trimodal slices allocated by PPO.

Only patrol.py supplies the physical executor and slice contracts. No previous
allocation experiment, checkpoint or diagnostic script is a dependency.
"""
from __future__ import annotations

import collections

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from slice_alloc import patrol

STATE_DIM = patrol.STATE_DIM + 2 * len(patrol.RESOURCES) + 10
CANDIDATE_DIM = patrol.CANDIDATE_DIM + len(patrol.RESOURCES) + 3
RESOURCE_DIM = 5
NEED_DIM = 5
MEMBER_DIM = 16
MAX_MEMBERS = 2
CALENDAR_TIMES = (0., 1., 3., 6.)
SHARED = np.array([i for i, name in enumerate(patrol.RESOURCES)
                   if name.startswith('wire:') or name == 'airtime'])
MODES = ('ip', 'id', 'ndn')


def make_scenario(seed, load, mix='balanced'):
    profiles = dict(balanced='mixed', ip_heavy='bandwidth',
                    id_heavy='mapping', ndn_heavy='cache')
    cfg = patrol.scenario(seed, load, profiles[mix])
    # Change request mix only: never inherit the stress profile's smaller
    # physical capacities. Every mix uses exactly the same infrastructure.
    cfg['public'] = patrol.public_config('mixed')
    patrol.validate('scenario', cfg)
    return cfg


class QuotaLedger(patrol.Ledger):
    """Fixed admission quotas on shared communication resources only."""
    def __init__(self, public, fractions=None):
        super().__init__(public)
        self.fractions = fractions or dict.fromkeys(MODES, 1 / 3)
        if set(self.fractions) != set(MODES) or any(v < 0 for v in self.fractions.values()):
            raise ValueError('Invalid modality quotas')
        if not np.isclose(sum(self.fractions.values()), 1):
            raise ValueError('Quotas must sum to the original physical capacity')

    def feasible(self, extra):
        if not super().feasible(extra):
            return False
        plans = self.plans + [extra]
        points = sorted({t for p in plans for a, b, _ in p.segments for t in (a, b)})
        for a, b in zip(points, points[1:]):
            if b <= extra.request.start or a >= extra.request.drain:
                continue
            t = (a + b) / 2
            usage = np.zeros(len(patrol.RESOURCES))
            for p in plans:
                if p.request.mode == extra.request.mode:
                    for start, end, vector in p.segments:
                        if start <= t < end:
                            usage += vector
            if np.any(usage[SHARED] > self.capacity[SHARED] *
                      self.fractions[extra.request.mode] + 1e-8):
                return False
        return True


def marginal_cost(ledger, plan):
    start, end = plan.request.start, plan.request.drain
    points = {start, end}
    for existing in ledger.plans + [plan]:
        points.update(t for a, b, _ in existing.segments for t in (a, b) if start < t < end)
        if existing.lease:
            points.update(t for t in existing.lease[-2:] if start < t < end)
    integral = np.zeros(len(patrol.RESOURCES))
    price = 0.
    points = sorted(points)
    for a, b in zip(points, points[1:]):
        usage = ledger.usage((a + b) / 2)
        addition = (ledger.usage((a + b) / 2, plan) - usage) / ledger.capacity
        integral += (b - a) * addition
        price += (b - a) * float(np.dot(addition, 1 + 4 * (usage / ledger.capacity) ** 2))
    return integral, price / (end - start)


class InitialAllocationEnv(patrol.PatrolEnv):
    def __init__(self, seed=1, loads=(64, 80), partitioned=False):
        super().__init__(seed=seed, loads=loads, profiles=('mixed',), shaping=False)
        self.partitioned = partitioned
        self._clear_history()
        self.observation_space = spaces.Dict({
            'state': spaces.Box(-10, 10, (STATE_DIM,), np.float32),
            'candidates': spaces.Box(-10, 10, (patrol.MAX_ACTIONS, CANDIDATE_DIM), np.float32),
            'resources': spaces.Box(-10, 10, (len(patrol.RESOURCES), RESOURCE_DIM), np.float32),
            'needs': spaces.Box(-10, 10, (patrol.MAX_ACTIONS, len(patrol.RESOURCES), NEED_DIM), np.float32),
            'members': spaces.Box(-10, 10, (MAX_MEMBERS, MEMBER_DIM), np.float32),
            'member_mask': spaces.Box(0, 1, (MAX_MEMBERS,), np.float32),
            'mask': spaces.Box(0, 1, (patrol.MAX_ACTIONS,), np.float32)})

    def _clear_history(self):
        self.seen = collections.Counter()
        self.positions = {}
        self.previous_time = None
        self.previous_aps = {}
        self.switches = collections.Counter()
        self.association_since = {}
        self.motion_frame = None
        self.motion = {}

    def reset(self, seed=None, options=None):
        self._clear_history()
        if options is None or 'scenario' not in options:
            cfg = make_scenario(int(self.rng.integers(6000000, 7000000)),
                                int(self.rng.choice(self.loads)))
            options = dict(scenario=cfg)
        return super().reset(seed=seed, options=options)

    def _observe_motion(self):
        now = self.current['time']
        frame = (now, self.current['request_id'])
        if frame == self.motion_frame:
            return
        dt = now - self.previous_time if self.previous_time is not None else 0.
        for node, raw in self.current['positions'].items():
            pos = np.asarray(raw, float)
            ap = self.current['associations'][node]
            known = node in self.positions and dt > 1e-9
            velocity = (pos - self.positions[node]) / dt if known else np.zeros(3)
            if node not in self.previous_aps or ap != self.previous_aps[node]:
                if node in self.previous_aps:
                    self.switches[node] += 1
                self.association_since[node] = now
            distances = [float(np.linalg.norm(pos - [x, 0., 10.])) for x in (0., 150.)]
            if ap in (3, 4):
                radial = pos - [0. if ap == 3 else 150., 0., 10.]
                distance = float(np.linalg.norm(radial))
                outward = float(np.dot(velocity, radial) / max(distance, 1e-9))
            else:
                distance, outward = 120., 0.
            mission = self.cfg['public']['missions'][node]
            self.motion[node] = np.array([
                distance / 120, min(distances) / 120,
                np.clip(velocity[0] / 11, -1, 1), np.clip(outward / 11, -1, 1),
                float(known), min(10, self.switches[node]) / 2,
                float(mission['role'] == 'patrol'), float(ap == 0),
                sum(d <= 120 for d in distances) / 2,
                min(8, now - self.association_since[node]) / 8])
        self.motion_frame = frame
        self.positions = {n: np.asarray(p, float) for n, p in self.current['positions'].items()}
        self.previous_time = now
        self.previous_aps = dict(self.current['associations'])

    def encode(self):
        if self.partitioned and not isinstance(self.ledger, QuotaLedger):
            assert not self.ledger.plans
            self.ledger = QuotaLedger(self.cfg['public'])
        original = super().encode()
        state = np.zeros(STATE_DIM, np.float32)
        rows = np.zeros((patrol.MAX_ACTIONS, CANDIDATE_DIM), np.float32)
        resources = np.zeros((len(patrol.RESOURCES), RESOURCE_DIM), np.float32)
        needs = np.zeros((patrol.MAX_ACTIONS, len(patrol.RESOURCES), NEED_DIM), np.float32)
        members = np.zeros((MAX_MEMBERS, MEMBER_DIM), np.float32)
        member_mask = np.zeros(MAX_MEMBERS, np.float32)
        state[:patrol.STATE_DIM] = original['state']
        rows[:, :patrol.CANDIDATE_DIM] = original['candidates']
        if self.current['kind'] != 'done':
            now = self.current['time']
            request = self.requests[self.current['request_id']]
            for i, horizon in enumerate((3., 6.)):
                at = patrol.STATE_DIM + i * len(patrol.RESOURCES)
                state[at:at + len(patrol.RESOURCES)] = self.ledger.usage(now + horizon) / self.ledger.capacity
            self._observe_motion()
            state[-10:] = np.mean([self.motion[str(n)] for n in request.members], axis=0)
            if len(request.members) > MAX_MEMBERS:
                raise ValueError('Request exceeds declared endpoint capacity')
            for k, n in enumerate(request.members):
                key = str(n)
                ap = self.current['associations'][key]
                members[k] = np.r_[self.motion[key],
                    [float(ap == x) for x in (0, 3, 4)],
                    np.asarray(self.current['positions'][key]) / 150.]
                member_mask[k] = 1
            usages = [self.ledger.usage(now + dt) for dt in CALENDAR_TIMES]
            resources[:, :4] = np.stack(usages, axis=1) / self.ledger.capacity[:, None]
            resources[:, 4] = self.ledger.capacity / patrol.capacities(patrol.public_config())
            for i, option in enumerate(self.options, 1):
                integral, price = marginal_cost(self.ledger, option)
                needs[i, :, 0] = option.peak / self.ledger.capacity
                for k, dt in enumerate(CALENDAR_TIMES):
                    needs[i, :, k + 1] = (self.ledger.usage(now + dt, option) - usages[k]) / self.ledger.capacity
                rows[i, patrol.CANDIDATE_DIM:patrol.CANDIDATE_DIM + len(patrol.RESOURCES)] = integral / 8
                rows[i, -3:] = [price,
                    min(10, self.seen[request.tenant, request.content, request.version]) / 10
                    if request.mode == 'ndn' else 0., (request.drain - now) / 8]
        return dict(state=state, candidates=rows, mask=original['mask'],
                    resources=resources, needs=needs, members=members, member_mask=member_mask)

    def step(self, action):
        request = self.requests[self.current['request_id']]
        if request.mode == 'ndn':
            self.seen[request.tenant, request.content, request.version] += 1
        return super().step(action)


PRICE_INDEX = -3  # Used by the external rule; excluded from PPO's network inputs.


class RewardUnit(gym.Wrapper):
    """Constant reward units, preserving the SLA objective exactly."""
    def __init__(self, env, unit=80.):
        super().__init__(env)
        if not np.isfinite(unit) or unit <= 0:
            raise ValueError('reward unit must be positive and finite')
        self.unit = float(unit)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward / self.unit, terminated, truncated, {
            **info, 'native_sla_reward': reward}


class HierarchicalCategorical(CategoricalDistribution):
    """Joint training probabilities; sequential gate/deployment evaluation."""
    def __init__(self, reject_score, deployment_scores, valid):
        super().__init__(deployment_scores.shape[1] + 1)
        self.reject_score = reject_score.squeeze(-1)
        self.valid = valid
        self.deployment_scores = deployment_scores.masked_fill(~valid, -1e9)
        available = valid.any(-1)
        conditional = torch.log_softmax(self.deployment_scores, dim=-1)
        reject_logp = torch.nn.functional.logsigmoid(self.reject_score)
        admit_logp = torch.nn.functional.logsigmoid(-self.reject_score)
        joint = torch.cat([reject_logp[:, None], admit_logp[:, None] + conditional], dim=-1)
        joint[:, 1:] = joint[:, 1:].masked_fill(~valid, -1e9)
        joint = torch.where(available[:, None], joint,
                            torch.cat([torch.zeros_like(reject_logp[:, None]),
                                       torch.full_like(deployment_scores, -1e9)], dim=-1))
        self.proba_distribution(joint)

    def mode(self):
        deployment = self.deployment_scores.argmax(-1) + 1
        accept = (self.reject_score <= 0.) & self.valid.any(-1)
        return torch.where(accept, deployment, torch.zeros_like(deployment))


class AllocationEncoder(nn.Module):
    """Request/candidate attention over per-resource calendar and endpoint sets."""
    def __init__(self, width, lifecycle):
        super().__init__()
        self.lifecycle = lifecycle
        self.width = width
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.candidate_encoder = nn.Sequential(nn.Linear(CANDIDATE_DIM, width), nn.Tanh())
        self.resource_encoder = nn.Sequential(nn.Linear(RESOURCE_DIM + NEED_DIM, width), nn.Tanh())
        self.resource_identity = nn.Embedding(len(patrol.RESOURCES), width)
        self.resource_attention = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0.)
        self.candidate_norm = nn.LayerNorm(width)
        self.member_encoder = nn.Sequential(nn.Linear(MEMBER_DIM, width), nn.Tanh())
        self.member_attention = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0.)
        self.context = nn.Sequential(nn.Linear(3 * width + 1, width), nn.Tanh())

    def forward(self, obs):
        state = obs['state'].float().clone()
        raw = obs['candidates'].float().clone()
        raw[:, :, PRICE_INDEX] = 0.
        resources = obs['resources'].float().clone()
        needs = obs['needs'].float().clone()
        if not self.lifecycle:
            # Remove explicit per-resource future calendars in all input paths;
            # keep the current request duration and old aggregate held counts.
            state[:, patrol.STATE_DIM:patrol.STATE_DIM + 2 * len(patrol.RESOURCES)] = 0.
            resources[:, :, 1:4] = resources[:, :, :1]
            needs[:, :, :, 2:5] = needs[:, :, :, 1:2]
        mask = obs['mask'][:, 1:] > .5
        encoded = self.state_encoder(state)
        candidates = self.candidate_encoder(raw[:, 1:])
        batch, count, width = candidates.shape
        shared = resources[:, None].expand(-1, count, -1, -1)
        tokens = self.resource_encoder(torch.cat([shared, needs[:, 1:]], dim=-1))
        tokens = tokens + self.resource_identity.weight[None, None]
        attended, _ = self.resource_attention(candidates.reshape(batch * count, 1, width),
            tokens.reshape(batch * count, len(patrol.RESOURCES), width),
            tokens.reshape(batch * count, len(patrol.RESOURCES), width), need_weights=False)
        candidates = self.candidate_norm(candidates + attended.reshape(batch, count, width))
        pooled = (candidates * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        member_mask = obs['member_mask'] > .5
        has_members = member_mask.any(-1)
        safe_mask = member_mask.clone()
        safe_mask[:, 0] |= ~has_members
        members = self.member_encoder(obs['members'].float())
        member_context, _ = self.member_attention(encoded[:, None], members, members,
            key_padding_mask=~safe_mask, need_weights=False)
        member_context = member_context[:, 0] * has_members[:, None]
        context = self.context(torch.cat([encoded, pooled, member_context,
            mask.sum(1, keepdim=True).float() / (patrol.MAX_ACTIONS - 1)], dim=-1))
        pairs = torch.cat([context[:, None].expand(-1, count, -1), candidates], dim=-1)
        return context, pairs


class AllocationPolicy(ActorCriticPolicy):
    """One PPO with learned admission and conditional deployment probabilities."""
    def __init__(self, *args, width=64, hierarchical=True, lifecycle=True, **kwargs):
        for obsolete in ('delta', 'price_weight', 'reject_bias'):
            if obsolete in kwargs:
                raise ValueError(f'{obsolete} belongs to a historical policy, not this PPO')
        if not isinstance(width, int) or width < 8 or width % 4:
            raise ValueError('width must be a positive multiple of four >= 8')
        self.width = width
        self.hierarchical = bool(hierarchical)
        self.lifecycle = bool(lifecycle)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        w = self.width
        self.encoder = AllocationEncoder(w, self.lifecycle)
        self.value_encoder = AllocationEncoder(w, self.lifecycle)
        self.actor = nn.Sequential(nn.Linear(2*w, w), nn.Tanh(), nn.Linear(w, 1))
        self.reject = nn.Sequential(nn.Linear(w, w), nn.Tanh(), nn.Linear(w, 1))
        self.critic = nn.Sequential(nn.Linear(w, w), nn.Tanh(), nn.Linear(w, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=.1)
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        nn.init.orthogonal_(self.reject[-1].weight, .01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        result = super()._get_constructor_parameters()
        result.update(width=self.width, hierarchical=self.hierarchical, lifecycle=self.lifecycle)
        return result

    def _outputs(self, obs):
        mask = obs['mask'] > .5
        context, pairs = self.encoder(obs)
        deployments = self.actor(pairs).squeeze(-1)
        rejection = self.reject(context)
        if self.hierarchical:
            distribution = HierarchicalCategorical(rejection, deployments, mask[:, 1:])
        else:
            logits = torch.cat([rejection, deployments], dim=-1).masked_fill(~mask, -1e9)
            distribution = CategoricalDistribution(self.action_space.n).proba_distribution(logits)
        value_context, _ = self.value_encoder(obs)
        value = self.critic(value_context)
        return distribution, value

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
