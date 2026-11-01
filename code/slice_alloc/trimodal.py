"""Versioned, bounded, executable three-modality allocation model.

No real future trajectory is an input. Native packet replay is authoritative for
SLA; this fast environment optimizes model admission, explicitly a proxy.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import time
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np

SCHEMA = 'trimodal-v1'
MODES = ('ip', 'id', 'ndn')
EDGES = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4))
WIRED = tuple((a, b) for a, b in EDGES for a, b in ((a, b), (b, a)))
CAPACITIES = {**{f'wire:{a}>{b}': 125000. for a, b in WIRED},
              'airtime:shared': .9,
              **{f'fib:{n}': 48. for n in (1, 2, 3, 4)},
              **{f'map:{n}': 12. for n in (1, 2)},
              **{f'mapcache:{n}': 12. for n in (1, 2)},
              **{f'cache:{n}': 16000. for n in (1, 2)}}
RESOURCE_KEYS = tuple(CAPACITIES)
MAX_CANDIDATES = 16
HEADER = 25  # actual v1 experiment header: type + six uint32 fields
PAYLOAD = 1000
SOURCE_DATAGRAM = PAYLOAD + HEADER + 28
IP_DATAGRAM = PAYLOAD + 9 + 28
ID_DATAGRAM = IP_DATAGRAM + HEADER + 28
RETRY = .2
TTL = .6


@dataclass(frozen=True)
class Request:
    id: int
    tenant: int
    mode: str
    uav: int
    start: float
    end: float
    period: float
    deadline: float
    attachments: tuple = (3, 4)

    def __post_init__(self):
        if (self.mode not in MODES or self.id <= 0 or self.uav < 5 or
                self.start < 1 or self.end <= self.start+.05 or
                self.period <= 0 or self.deadline <= 0 or
                not self.attachments or not set(self.attachments) <= {3, 4}):
            raise ValueError('invalid traffic contract')

    @property
    def drain(self):
        return self.end + self.deadline + .01

    @property
    def count(self):
        return len(np.arange(self.start+.05, self.end-1e-8, self.period))


@dataclass(frozen=True)
class Plan:
    request: Request
    edge: int
    server: int
    cache: int
    usage: tuple
    paths: tuple

    @property
    def key(self):
        return self.edge, self.server, self.cache


def path(a, b):
    """Fixed loop-free ground route matching the executable forwarding rules."""
    if a == b:
        return ()
    # Selected functional edge is represented explicitly by caller composition.
    queue = [(a, ())]
    visited = {a}
    for node, hops in queue:
        for x, y in WIRED:
            if x == node and y not in visited:
                p = hops + ((x, y),)
                if y == b:
                    return p
                visited.add(y)
                queue.append((y, p))
    raise ValueError('no legal ground path')


def compile_plan(r, edge, server, cache, profile=None):
    if edge not in (1, 2) or server not in (1, 2) or cache not in (0, edge):
        raise ValueError('unsupported executable placement')
    if r.mode != 'ndn' and cache:
        raise ValueError('cache is NDN-only')
    if r.mode != 'id' and server != edge:
        raise ValueError('only ID has a separate mapping service')
    profile = profile or {'ndn_data_bytes': 1070, 'ndn_interest_bytes': 48,
                          'tx_seconds_per_datagram': .0025}
    u = dict.fromkeys(RESOURCE_KEYS, 0.)
    attempts = math.ceil(r.deadline / RETRY) + 1
    interval = r.period
    scenarios = []
    allpaths = []
    for ap in r.attachments:
        v = dict.fromkeys(RESOURCE_KEYS, 0.)
        data_path = path(0, edge) + path(edge, ap)
        allpaths.append(data_path)
        def add(hops, byte_rate):
            for a, b in hops:
                v[f'wire:{a}>{b}'] += byte_rate
        if r.mode == 'ndn':
            # Pinned immutable content: one upstream fill per slice, no free
            # wireless hit; each subsequent demand still crosses AP--UAV.
            interest = profile['ndn_interest_bytes'] + len(str(r.id))-1
            data = profile['ndn_data_bytes'] + len(str(r.id))-1
            upstream = attempts / (r.end-r.start) if cache else attempts/interval
            add(path(edge, 0), upstream * interest)
            add(path(0, edge), upstream * data)
            add(path(ap, edge), attempts/interval * interest)
            add(path(edge, ap), attempts/interval * data)
            v['airtime:shared'] = attempts/interval * profile['tx_seconds_per_datagram'] * 2
        else:
            add(path(0, edge), SOURCE_DATAGRAM / interval)
            add(path(edge, ap), attempts * (ID_DATAGRAM if r.mode == 'id' else IP_DATAGRAM) / interval)
            add(path(ap, edge), attempts * (HEADER+28) / interval)  # ACK
            # Bounded initial + one handover registration, with ACK and retry.
            control = 24 * (HEADER+28) / (r.end-r.start)
            add(path(ap, server), control)
            add(path(server, ap), control)
            if r.mode == 'id':
                # Demand-triggered queries, pessimistic retry bound per demand.
                add(path(edge, server), attempts * (HEADER+28) / interval)
                add(path(server, edge), attempts * (HEADER+28) / interval)
            v['airtime:shared'] = (attempts/interval*2 + 48/(r.end-r.start)) * profile['tx_seconds_per_datagram']
        # Transition/management allowance is shared, never an extra AP channel.
        v['airtime:shared'] += .01
        scenarios.append(v)
    for k in u:
        u[k] = max(v[k] for v in scenarios)
    for n in (1, 2, 3, 4):
        u[f'fib:{n}'] = float(r.mode == 'ndn') * (2 if n in (edge, *r.attachments) else 1)
    if r.mode in ('ip', 'id'):
        u[f'map:{server}'] = 1.
    if r.mode == 'id':
        u[f'mapcache:{edge}'] = 1.
    if cache:
        u[f'cache:{cache}'] = PAYLOAD
    return Plan(r, edge, server, cache, tuple(u[k] for k in RESOURCE_KEYS), tuple(allpaths))


class Ledger:
    def __init__(self):
        self.plans = []
        self.capacity = np.array(list(CAPACITIES.values()))

    def usage(self, at):
        active = [p.usage for p in self.plans if p.request.start <= at < p.request.drain]
        return np.sum(active, axis=0) if active else np.zeros(len(RESOURCE_KEYS))

    def feasible(self, p):
        r = p.request
        times = {r.start}
        times.update(x for q in self.plans for x in (q.request.start, q.request.drain)
                     if r.start <= x < r.drain)
        return all(np.all(self.usage(t) + p.usage <= self.capacity + 1e-9) for t in times)

    def commit(self, p):
        if not self.feasible(p):
            raise ValueError('atomic reservation failed')
        self.plans.append(p)


def candidates(r, ledger, profile=None):
    start = time.perf_counter()
    choices = [(e, s, 0) for e in (1, 2) for s in (1, 2)] if r.mode == 'id' else [
        (e, e, c) for e in (1, 2) for c in ((0, e) if r.mode == 'ndn' else (0,))]
    compiled = [compile_plan(r, *c, profile) for c in choices]
    feasible = [p for p in compiled if ledger.feasible(p)]
    result = feasible[:MAX_CANDIDATES]
    return result, dict(enumerated=len(compiled), checked=len(compiled), feasible=len(feasible),
                        retained=len(result), truncated=len(feasible)>MAX_CANDIDATES,
                        search_limit=False, seconds=time.perf_counter()-start)


def workload(seed, load=1., uavs=4, count=24):
    rng = np.random.default_rng(seed)
    modes = list(MODES) * math.ceil(count/3)
    rng.shuffle(modes)
    at = 2.
    rows = []
    for i in range(count):
        at += float(rng.uniform(.45, .85)) / load
        mode = modes[i]
        node = int(rng.integers(5, 5+uavs))
        duration = float(rng.choice((3., 5., 7.)))
        # Endpoints except node5 are stationary but the publicly permitted
        # attachment set is a contract, not a future-trajectory oracle.
        # UAVs are not uniformly dual-covered: declared home-side UAVs only
        # require their nearer AP, so which edge serves them actually
        # determines admissibility, not just internal resource bookkeeping.
        # A UAV crossing coverage areas still declares both as permitted.
        side = (node-5) % 2
        attachments = (3,) if side == 0 else (4,) if rng.random() < .7 else (3, 4)
        rows.append(Request(i+1, i%3+1, mode, node, at, at+duration,
                            float(rng.choice((.3, .5, .8))), .8, attachments))
    return rows


def write_tasks(pathname, plans):
    p = Path(pathname)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# trimodal-task-v1 id tenant mode uav start stop period deadline edge server cache']
    for plan in plans:
        r = plan.request
        lines.append(' '.join(map(str, (r.id, r.tenant, r.mode, r.uav, r.start, r.end,
                                       r.period, r.deadline, plan.edge, plan.server, plan.cache))))
    p.write_text('\n'.join(lines)+'\n')


class TrimodalEnv(gym.Env):
    candidate_dim = len(RESOURCE_KEYS) + 9
    state_dim = len(RESOURCE_KEYS) + 8

    def __init__(self, seeds=(101, 102, 103), loads=(.5, 1., 1.5), uavs=4, count=24, profile=None):
        self.seeds, self.loads, self.uavs, self.count = seeds, loads, uavs, count
        self.profile = profile
        self.rng = np.random.default_rng(71)
        self.action_space = spaces.Discrete(MAX_CANDIDATES+1)
        self.observation_space = spaces.Dict({
            'state': spaces.Box(0, 10, (self.state_dim,), np.float32),
            'candidates': spaces.Box(0, 10, (MAX_CANDIDATES+1, self.candidate_dim), np.float32),
            'mask': spaces.Box(0, 1, (MAX_CANDIDATES+1,), np.float32)})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        self.requests = workload(options.get('scenario_seed', int(self.rng.choice(self.seeds))),
                                 options.get('load', float(self.rng.choice(self.loads))), self.uavs, self.count)
        self.ledger = Ledger()
        self.index, self.decisions = 0, []
        self.candidate_log = []
        return self.observation(), {}

    def observation(self):
        state = np.zeros(self.state_dim, np.float32)
        rows = np.zeros((MAX_CANDIDATES+1, self.candidate_dim), np.float32)
        mask = np.zeros(MAX_CANDIDATES+1, np.float32)
        rows[0, 0] = 1
        mask[0] = 1
        self.options = []
        if self.index < len(self.requests):
            r = self.requests[self.index]
            usage = self.ledger.usage(r.start)/self.ledger.capacity
            state[:len(usage)] = usage
            state[len(usage):] = [*(float(r.mode == m) for m in MODES),
                                   (r.end-r.start)/10, r.period, r.start/60, self.index/self.count, r.deadline]
            self.options, info = candidates(r, self.ledger, self.profile)
            self.current_log = info
            for i, p in enumerate(self.options, 1):
                v = np.array(p.usage)/self.ledger.capacity
                rows[i] = [0, *(float(r.mode == m) for m in MODES),
                           p.edge/2, p.server/2, p.cache/2, (r.end-r.start)/10, r.period, *v]
                mask[i] = 1
        return {'state': state, 'candidates': rows, 'mask': mask}

    def step(self, action):
        action = int(action)
        if not 0 <= action <= len(self.options):
            raise ValueError('masked action selected')
        p = self.options[action-1] if action else None
        if p:
            self.ledger.commit(p)
        self.decisions.append((self.requests[self.index], p))
        self.candidate_log.append(self.current_log)
        self.index += 1
        done = self.index == len(self.requests)
        return self.observation(), float(p is not None), done, False, {}

    @property
    def metadata_contract(self):
        return dict(schema=SCHEMA, resources=RESOURCE_KEYS, state_dim=self.state_dim,
                    candidate_dim=self.candidate_dim, max_candidates=MAX_CANDIDATES,
                    uavs=self.uavs, reward='model-admission-proxy', profile=self.profile)


def select_rule(env, method):
    if not env.options:
        return 0
    if method == 'firstfit':
        return 1
    usage = env.ledger.usage(env.requests[env.index].start)/env.ledger.capacity
    costs = []
    for p in env.options:
        v = np.array(p.usage)/env.ledger.capacity
        if method == 'price':
            cost = np.sum(v/(1.01-usage))
        elif method == 'sequential':
            # Frozen edge-first stage: lowest current edge wire utilization;
            # then cheapest complete configuration only on that edge.
            preferred = 1 if usage[0]+usage[2] <= usage[1]+usage[3] else 2
            cost = 100*(p.edge != preferred)+np.sum(v)
        elif method == 'coverage':
            r = p.request
            home = 1 if r.uav == 5 or (r.uav-5)%2 else 2
            cost = .05*(p.edge != home)+np.sum(v)
        else:
            # Ground topology and cost are edge1/edge2-symmetric by construction
            # (same wire capacities, shared airtime pool); a static tie-break
            # would make this rule structurally blind to edge2. Break ties by
            # comparing EACH edge's own current wire occupancy (its source
            # link plus its AP link), so 'greedy' reacts to actual occupancy
            # instead of candidate list order.
            edge_wire_idx = {
                1: [i for i, k in enumerate(RESOURCE_KEYS) if k in ('wire:0>1', 'wire:1>0', 'wire:1>3', 'wire:3>1')],
                2: [i for i, k in enumerate(RESOURCE_KEYS) if k in ('wire:0>2', 'wire:2>0', 'wire:2>4', 'wire:4>2')]}
            own_edge_load = usage[edge_wire_idx[p.edge]].sum()
            cost = np.sum(v) + 1e-6 * own_edge_load
        costs.append(cost)
    return int(np.argmin(costs))+1


def validate_metadata(pathname, env):
    saved = json.loads(Path(pathname).read_text())
    current = json.loads(json.dumps(env.metadata_contract))
    if saved != current:
        raise ValueError('incompatible model schema/resources/profile')


def request_digest(requests):
    return hashlib.sha256(json.dumps([asdict(r) for r in requests], sort_keys=True).encode()).hexdigest()
