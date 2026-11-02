"""Initial admission/placement with real ingress shaping and measured CMDP costs.

Inspired by HRL-ACRA's admission/embedding factorization and CLARA's separation
of instantaneous resource feasibility from cumulative service constraints. This
is an adaptation, not a reproduction of either algorithm or its guarantees.
"""
from copy import deepcopy
import collections

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from slice_alloc import allocation_final as base
from slice_alloc.initial_allocation import HierarchicalCategorical, marginal_cost
from slice_alloc.allocation_final_policy import CandidatePPOPolicy

b = base.b
p = b.p
COMPATIBLE = base.COMPATIBLE
WEIGHTS = dict(telemetry=2., command=3., object=1.)
TIERS = (.90, 1.10, 1.40)
OVERBOOK = 1.12
RISK_BUDGET = .02  # mean admitted peak excess per arrival; not packet loss
VIOLATION_BUDGET = dict(telemetry=.15, object=.10, command=.10)
SERVICES = tuple(b.SERVICES)
MAX_ACTIONS = 1 + (b.MAX_ACTIONS-1)*len(TIERS)
EXTRA_STATE = 9
EXTRA_CANDIDATE = 6
TRAIN_RANGE = (61000000, 62000000)
NS3 = p.NS3


def constraint_excess(rows):
    result = []
    for service in SERVICES:
        admitted = sum(r['service_admitted'][service] for r in rows)
        success = sum(r['by_service'][service]['success'] for r in rows)
        result.append(max(0.,(admitted-success)/max(1,admitted)-VIOLATION_BUDGET[service]))
    result.append(max(0.,sum(r['risk_sum'] for r in rows)/sum(r['requests'] for r in rows)-RISK_BUDGET))
    return result


def shaping_adequate(request, factor):
    """Simple baseline checks its own deterministic source-shaping delay.

    It uses the current contract, not hidden traffic or future arrivals. It
    ignores network delay except a fixed 4ms allowance and is not an SLA proof.
    """
    rate = factor*request.size/request.period
    burst = max(request.payload,factor*request.size)
    finish = 0.; on_time = 0
    for tick in range(request.rounds):
        now = tick*request.period
        for chunk in range(request.chunks):
            at = now+.001*chunk
            finish = max(at,finish)
            delay = max(0.,finish-(burst-request.payload)/rate-at)
            on_time += delay+.004 <= request.deadline
            finish += request.payload/rate
    return on_time+1e-9 >= request.rounds*request.chunks*request.minimum_ratio


def scenario(seed, load=48, mix='balanced', mobility='mixed', distance_loss=True):
    cfg = base.scenario(seed, load, mix, mobility, distance_loss)
    cfg['public'].update(ingress_shaping=True, admission_overbook=OVERBOOK,
                         service_weights=WEIGHTS, rate_tiers=list(TIERS))
    return cfg


def ratios(ledger, plan):
    """Exact peak at all reservation AND shared-cache lease boundaries."""
    points = {plan.request.start, plan.request.drain}
    for old in ledger.plans+[plan]:
        points.update(t for a, z, _ in old.segments for t in (a,z)
                      if plan.request.start < t < plan.request.drain)
        if old.lease:
            points.update(t for t in old.lease[-2:]
                          if plan.request.start < t < plan.request.drain)
    points = sorted(points)
    return np.max([ledger.usage((a+z)/2,plan)/ledger.capacity
                   for a,z in zip(points,points[1:])], axis=0)


class AdmissionLedger(p.Ledger):
    """Only divisible communication reservations may exceed nominal capacity.

    Mapping entries and cached bytes remain hard physical constraints. The
    1.12 communication ceiling is a design parameter, not extra PHY capacity.
    """
    def __init__(self, pub, limit=OVERBOOK):
        super().__init__(pub)
        if not np.isfinite(limit) or limit < 1:
            raise ValueError('Invalid admission limit')
        self.limits = np.ones(len(p.RESOURCES))
        self.limits[:11] = limit

    def feasible(self, plan):
        return bool(np.all(ratios(self,plan) <= self.limits+1e-8))


class Session(base.FinalSession):
    binary_name = 'allocation-cmdp-scenario'


class AllocationEnv(base.AllocationEnv):
    session_class = Session

    def __init__(self, *args, strict=False, standard_only=False,
                 constrained=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.strict, self.standard_only = strict, standard_only
        self.constrained = constrained
        self.multipliers = np.ones(4) if constrained else np.zeros(4)
        self.pending_multipliers = self.multipliers.copy()
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Dict({
            'state': spaces.Box(-100,100,
                (self.observation_space['state'].shape[0]+EXTRA_STATE,),np.float32),
            'candidates': spaces.Box(-100,100,
                (MAX_ACTIONS,b.CANDIDATE_DIM+EXTRA_CANDIDATE),np.float32),
            'mask': spaces.Box(0,1,(MAX_ACTIONS,),np.float32)})

    def reset(self, seed=None, options=None):
        opts = dict(options or {})
        if 'scenario' not in opts:
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            opts['scenario'] = scenario(int(self.rng.integers(*TRAIN_RANGE)),
                                        int(self.rng.choice(self.loads)))
        self.multipliers = self.pending_multipliers.copy()
        self.utility = self.training_return = self.risk_sum = 0.
        self.raw_cost = np.zeros(4)
        self.seen = collections.Counter()
        return super().reset(seed=seed,options=opts)

    def set_multipliers(self, values):
        values = np.asarray(values,float)
        if values.shape != (4,) or np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError('Invalid multipliers')
        self.pending_multipliers = values.copy() if self.constrained else np.zeros(4)

    def _candidates(self, r):
        if any(self.current['associations'].get(str(n),0) not in
               self.ledger.pub['allowed_aps'][str(n)] for n in r.members):
            return []
        modes = (b.native_mode(r),) if self.fixed else base.COMPATIBLE[r.service]
        plans = []
        # Current association is observed; future AP sequence is never read.
        public = deepcopy(self.ledger.pub)
        public['allowed_aps'] = {str(n):[int(ap)] if ap else
            public['allowed_aps'][str(n)] for n,ap in self.current['associations'].items()}
        for mode in modes:
            for edge in (1,2):
                choices = ((s,0) for s in (1,2)) if mode=='id' else (
                    ((edge,c) for c in (0,edge)) if mode=='ndn' else ((edge,0),))
                for server,cache in choices:
                    if r.service=='telemetry':
                        original = p.compile_plan(r.bound('ip'),edge,server,0,public,self.current)
                    else:
                        original = b.compile_plan(r,mode,edge,server,cache,public,self.current)
                    for factor in (TIERS[1],) if self.standard_only else TIERS:
                        plan = deepcopy(original)
                        for _,_,usage in plan.segments:
                            usage[:11] *= factor
                        plan.rate_factor = factor
                        plan.peak_ratios = ratios(self.ledger,plan)
                        plan.risk = max(0.,float(plan.peak_ratios[:11].max())-1.)
                        if self.ledger.feasible(plan) and (not self.quota or self.quota_feasible(plan)):
                            plans.append(plan)
        return plans

    def encode(self):
        if not isinstance(self.ledger,AdmissionLedger):
            assert not self.ledger.plans
            self.ledger = AdmissionLedger(self.ledger.pub,1. if self.strict else OVERBOOK)
        # The parent constructs its legacy-sized rows; temporarily collect
        # only base encodings then build expanded candidates here.
        self._expanded_options = None
        original = super().encode()
        state = np.r_[original['state'],np.zeros(EXTRA_STATE,np.float32)]
        rows = np.zeros((MAX_ACTIONS,b.CANDIDATE_DIM+EXTRA_CANDIDATE),np.float32)
        mask = np.zeros(MAX_ACTIONS,np.float32); mask[0] = 1.
        if self.current['kind']=='done':
            return dict(state=state,candidates=rows,mask=mask)
        r = self.requests[self.current['request_id']]
        state[-EXTRA_STATE:] = [WEIGHTS[r.service]/3.,
            *(self.multipliers/20.),*(self.seen[s]/80. for s in SERVICES),
            self.risk_sum/max(1,sum(self.seen.values()))]
        self.options = self._expanded_options
        for i,plan in enumerate(self.options,1):
            integral,_ = marginal_cost(self.ledger,plan)
            mode = plan.request.mode
            rows[i] = np.r_[plan.peak/self.ledger.capacity,integral/8.,
                [mode==s for s in b.MODES],[plan.edge==s for s in (1,2)],
                [plan.server==s for s in (1,2)],[plan.cache==s for s in (1,2)],plan.warm,1.,
                plan.rate_factor/1.4,plan.risk/.12,
                float(plan.peak_ratios[:11].max()),float(plan.peak_ratios[11:].max()),
                WEIGHTS[r.service]/3.,r.size/r.period/100000.]
            mask[i] = 1.
        return dict(state=state.astype(np.float32),candidates=rows,mask=mask)

    # BindingEnv encodes at most ten rows. Preserve the full set without
    # truncating it from the observation delivered to a policy.
    def get_candidates(self, r):
        plans = self._candidates(r)
        self._expanded_options = plans
        return plans[:b.MAX_ACTIONS-1]

    def step(self, action):
        action = int(action)
        if not 0 <= action <= len(self.options):
            raise ValueError('Masked action')
        r = self.requests[self.current['request_id']]
        self.seen[r.service] += 1
        costs = np.zeros(4); costs[-1] = -RISK_BUDGET
        risk = 0.
        if action:
            plan = self.options[action-1]
            integral,_ = marginal_cost(self.ledger,plan)
            self.resource_integral += integral
            self.ledger.add(plan)
            risk = plan.risk
            costs[-1] += risk
            decision = {**plan.decision(),'schema':b.PROTOCOL,'mode':plan.request.mode,
                        'rate_factor':plan.rate_factor}
        else:
            decision = dict(schema=b.PROTOCOL,request_id=r.id,accept=False,
                            edge=0,server=0,cache=0,mode='ip',rate_factor=1.)
        self.records.append(dict(request=r.id,service=r.service,decision=decision,
            feasible=len(self.options),risk=risk,strict_feasible=sum(x.risk<=1e-8 for x in self.options)))
        self.current = self.session.decide(decision)
        utility = 0.
        settled_counts = np.zeros(3)
        failures = np.zeros(3)
        for item in self.current['settled']:
            self.settled[item['id']] = item
            service = self.requests[item['id']].service
            k = SERVICES.index(service)
            settled_counts[k] += 1
            failures[k] += not item['success']
            costs[k] += float(not item['success'])-VIOLATION_BUDGET[service]
            utility += WEIGHTS[service]*float(item['success'])
        reward = (utility-float(np.dot(self.multipliers,costs)))/3.
        self.utility += utility
        self.training_return += reward
        self.risk_sum += risk
        self.raw_cost += costs
        done = self.current['kind']=='done'
        return self.encode(),reward,done,False,dict(utility=utility,
            costs=costs.tolist(),settled_counts=settled_counts.tolist(),
            failures=failures.tolist(),risk=risk,multipliers=self.multipliers.tolist())

    def result(self):
        result = super().result()
        total = sum(WEIGHTS[r.service] for r in self.requests.values())
        result.update(utility=self.utility,utility_available=total,
            utility_rate=self.utility/total,reward_total=self.training_return,
            risk_sum=self.risk_sum,cost_sum=self.raw_cost.tolist(),
            risk_admissions=sum(x['risk']>1e-8 for x in self.records),
            tiers=dict(collections.Counter(str(x['decision']['rate_factor'])
                for x in self.records if x['decision']['accept'])),
            service_admitted={s:sum(self.requests[k].service==s for k in self.settled) for s in SERVICES},
            multipliers=self.multipliers.tolist())
        return result


class HierarchicalPPOPolicy(CandidatePPOPolicy):
    def _outputs(self, obs):
        context = self.context(obs['state'].float())
        candidates = self.candidate(obs['candidates'][:,1:].float())
        mask = obs['mask'][:,1:]>.5
        mean = (candidates*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
        maximum = candidates.masked_fill(~mask[...,None],-1e9).max(1).values
        maximum = torch.where(mask.any(1,keepdim=True),maximum,torch.zeros_like(maximum))
        context = self.fusion(torch.cat([context,mean,maximum],-1))
        pairs = torch.cat([context[:,None].expand(-1,candidates.shape[1],-1),candidates],-1)
        distribution = HierarchicalCategorical(self.reject(context),self.actor(pairs).squeeze(-1),mask)
        return distribution,self.critic(context)
