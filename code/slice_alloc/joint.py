"""Minimal protocol-contract allocator, separate from superseded VNF fixtures.

This first adapter targets the measured four-node IP/NDN object topology. It
does not implement the planned general moving 30-UAV mapper, CPU scheduler,
identity-slice adapter or tenant-specific FIB/PIT limits. Model admission and
native packet success are intentionally separate measurements.
"""
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from threading import RLock
from types import MappingProxyType
import hashlib
import json
import gymnasium as gym
import numpy as np
from .protocol_contracts import CacheLease, CacheLedger, Link, Reservation, object_reservation

KINDS = ('ip', 'ndn_cold', 'ndn_pin')
RESOURCES = ('backhaul', 'airtime', 'forwarding_state')
CAPACITIES = {'backhaul': 1e6, 'airtime': .4, 'forwarding_state': 48.}
CACHE_CAPACITIES = {'relay': 4000}
PERIOD = .12
DEADLINE = .15
NAME = '/tenantA/slice00/map/v1/chunk0'


@dataclass(frozen=True)
class Request:
    id: int
    arrival: float
    rounds: int
    allowed: tuple[str, ...]
    weight: float = 1.

    def __post_init__(self):
        if (type(self.id) is not int or not 0 <= self.id < 100 or not isfinite(self.arrival)
                or self.arrival < .1 or type(self.rounds) is not int or self.rounds < 1
                or not self.allowed or not set(self.allowed) <= set(KINDS)
                or not isfinite(self.weight) or self.weight <= 0):
            raise ValueError('invalid declared slice request')

    @property
    def name(self): return f'/tenantA/slice{self.id:02d}/map/v1/chunk0'

    @property
    def end(self): return self.arrival + (self.rounds-1)*PERIOD + .001 + DEADLINE


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    usage: tuple[float, ...]

    def __post_init__(self):
        if (not isfinite(self.start) or not isfinite(self.end) or self.end <= self.start
                or len(self.usage) != len(RESOURCES) or any(not isfinite(v) or v < 0 for v in self.usage)):
            raise ValueError('invalid resource interval')


@dataclass(frozen=True)
class Plan:
    request: Request
    kind: str
    segments: tuple[Segment, ...]
    reservation: Reservation

    @property
    def area(self):
        return sum((np.array(s.usage)*(s.end-s.start) for s in self.segments), np.zeros(len(RESOURCES)))

    @property
    def peak(self): return np.max([s.usage for s in self.segments], axis=0)


def links():
    result = {}
    for a,b,domain in (('origin','relay',None),('relay','a','ag'),('relay','b','ag')):
        for x,y in ((a,b),(b,a)):
            key=f'{x}>{y}'; result[key]=Link(key,x,y,f'{y}>{x}',1e6 if domain is None else 6e6,domain)
    return result


class Compiler:
    """Compile declared repeated objects, charging cold fill before cache use.

    Source-to-relay traffic uses a shared wired direction budget (both directions
    summed conservatively). Airtime comes from an independent empirical profile,
    not the serialization lower bound. The fixed state units are an explicit
    logical quota, not a measured CPU/physical-memory claim.
    """
    def __init__(self, profile):
        if not profile.get('passed') or not profile.get('heldout_passed'):
            raise ValueError('an independently checked calibration profile is required')
        self.profile=profile
        self.air=float(profile['airtime_seconds_per_pair_round'])
        self.data_header=int(profile['ndn_data_header_bytes'])
        self.interest=int(profile['ndn_interest_bytes'])
        self.paths={'a':('origin>relay','relay>a'),'b':('origin>relay','relay>b')}

    def compile(self, request, kind):
        if kind not in request.allowed: raise ValueError('protocol substitution is not permitted')
        pin=kind=='ndn_pin'
        lease=CacheLease('tenantA',f'slice{request.id:02d}','relay','map','v1',1000,request.arrival,request.end,True)
        amounts={}; segments=[]
        required=[]
        for round_id in range(request.rounds):
            start=request.arrival+round_id*PERIOD
            # Derive both sides from the same integer round boundary. Repeated
            # floating additions can create tiny overlaps and double a quota.
            end=request.end if round_id==request.rounds-1 else request.arrival+(round_id+1)*PERIOD
            warm=pin and round_id>0
            part=object_reservation(links=links(),paths=self.paths,tenant='tenantA',slice_id=f'slice{request.id:02d}',
                name='map',version='v1',payload_bytes=1000,start=start,end=end,
                modality='ip_unicast' if kind=='ip' else 'ndn',synchronous=True,
                leases=(),pin_requests=(lease,) if warm else (),
                data_header_bytes=32 if kind=='ip' else self.data_header,interest_bytes=self.interest)
            for key,value in part.items():
                if not key.startswith('new_pinned_cache_bytes:'): amounts[key]=amounts.get(key,0.)+value
            # IP also requests objects over UDP/IP. The generic object kernel
            # models its Data path only; this adapter adds actual request headers.
            if kind=='ip':
                for path in self.paths.values():
                    for edge in path:
                        reverse=links()[edge].reverse
                        key='bytes:'+reverse
                        amounts[key]=amounts.get(key,0.)+len(request.name)+32
            upstream=(part.get('bytes:origin>relay',0.)+part.get('bytes:relay>origin',0.))
            if kind=='ip': upstream+=2*(len(request.name)+32)
            state=2. if kind=='ip' else 6.
            # Reserve the declared burst in at most the SLA interval. A profile
            # is an empirical workload allowance, not a universal Wi-Fi service curve.
            window=min(PERIOD,end-start,DEADLINE)
            segments.append(Segment(start,end,(upstream*8/window,self.air/window,state)))
            required.extend(part.required_leases)
        dependencies=()
        if required:
            dependencies=(CacheLease(lease.tenant,lease.slice_id,lease.node,lease.name,lease.version,lease.bytes,
                min(l.valid_from for l in required),max(l.valid_until for l in required),True),)
        reservation=Reservation(amounts,dependencies,(lease,) if pin else ())
        return Plan(request,kind,tuple(segments),reservation)


class AtomicLedger:
    """One locked transaction for interval resources and object cache ownership."""
    def __init__(self, capacities=None, cache_capacities=None):
        self.capacities=dict(CAPACITIES if capacities is None else capacities)
        self.cache_capacities=dict(CACHE_CAPACITIES if cache_capacities is None else cache_capacities)
        if set(self.capacities)!=set(RESOURCES) or any(not isfinite(v) or v<=0 for v in self.capacities.values()):
            raise ValueError('complete positive resource capacities required')
        self.cache=CacheLedger();self.active={};self.version=0;self._lock=RLock()

    @property
    def capacity(self):return np.array([self.capacities[k] for k in RESOURCES])

    def used(self, at):
        return sum((np.array(s.usage) for p in self.active.values() for s in p.segments
                    if s.start<=at<s.end),np.zeros(len(RESOURCES)))

    def _trial(self, owner, plan):
        if owner in self.active:raise ValueError('duplicate allocation owner')
        proposed=tuple(self.active.values())+(plan,)
        segments=[s for p in proposed for s in p.segments]
        starts=np.array([s.start for s in segments]);ends=np.array([s.end for s in segments])
        times=np.unique(starts)
        used=((times[:,None]>=starts)&(times[:,None]<ends))@np.array([s.usage for s in segments])
        if np.any(used>self.capacity+1e-10):raise ValueError('interval resource capacity exceeded')
        trial=deepcopy(self.cache)
        trial.commit(owner,plan.reservation,self.cache_capacities)
        return trial

    def feasible(self, plan):
        with self._lock:
            try:self._trial(str(plan.request.id),plan)
            except ValueError:return False
            return True

    def commit(self, plan, expected_version=None):
        with self._lock:
            if expected_version is not None and expected_version!=self.version:
                raise ValueError('stale snapshot; regenerate candidates')
            owner=str(plan.request.id)
            trial=self._trial(owner,plan)
            self.cache=trial;self.active[owner]=plan;self.version+=1

    def release(self, owner):
        with self._lock:
            owner=str(owner)
            if owner not in self.active:raise KeyError(owner)
            trial=deepcopy(self.cache);trial.release(owner)
            self.cache=trial;del self.active[owner];self.version+=1


def candidates(compiler,ledger,request):
    return [p for k in request.allowed if ledger.feasible(p:=compiler.compile(request,k))]


def workload(seed,count=48,load=1.):
    rng=np.random.default_rng(seed)
    t=1.;result=[]
    for i in range(count):
        t+=float(rng.exponential(.035/load))
        mode=int(rng.choice(3,p=(.4,.4,.2)))
        allowed=(('ip',),('ndn_cold','ndn_pin'),KINDS)[mode]
        result.append(Request(i,t,int(rng.choice((2,4,7))),allowed))
    return tuple(result)


def request_hash(requests):
    return hashlib.sha256(json.dumps([dict(id=r.id,arrival=r.arrival,rounds=r.rounds,allowed=r.allowed,weight=r.weight)
        for r in requests],sort_keys=True).encode()).hexdigest()


def write_tasks(path,plans):
    rows=[]
    for p in plans:
        r=p.request
        for j in range(r.rounds):
            for member in range(2):
                rows.append((r.arrival+j*PERIOD+.001*member,2+member,'ip' if p.kind=='ip' else 'ndn',r.name,
                             r.arrival if p.kind=='ndn_pin' else 0.,r.end if p.kind=='ndn_pin' else 0.))
    from pathlib import Path
    Path(path).write_text(''.join('\t'.join(map(str,row))+'\n' for row in sorted(rows)))


class JointEnv(gym.Env):
    """Fast finite-horizon admission MODEL; packet rewards are reported separately.

    Only declared current contracts, active leases and release times are visible.
    Rewards settle at release; sum rewards equals admitted model utility. The
    native replay, not this environment, determines actual SLA success.
    """
    def __init__(self,profile,training=False,count=48,load=1.,training_loads=None):
        super().__init__();self.compiler=Compiler(profile);self.training=training;self.count=count;self.load=load
        # Fixed single-load construction stays exact for evaluation/diagnostics
        # (explicit load=, never overwritten below). training_loads is only
        # consulted when training=True, so no existing caller is affected.
        self.training_loads=tuple(training_loads) if training_loads else (load,)
        self.action_space=gym.spaces.Discrete(4)
        self.state_dim=18;self.candidate_dim=15
        self.observation_space=gym.spaces.Dict({
            'state':gym.spaces.Box(-np.inf,np.inf,(self.state_dim,),np.float32),
            'candidates':gym.spaces.Box(-np.inf,np.inf,(4,self.candidate_dim),np.float32),
            'mask':gym.spaces.Box(0,1,(4,),np.float32)})

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options=options or {}
        chosen=options.get('scenario_seed',int(self.np_random.integers(1000,5000)) if self.training else seed or 70000)
        if self.training and 'requests' not in options:
            self.load=float(self.np_random.choice(self.training_loads))
        self.requests=tuple(options.get('requests',workload(chosen,self.count,self.load)))
        self.ledger=AtomicLedger();self.i=0;self.done=False;self.decisions=[];self.settled=0.;self.arrived=np.zeros(3);self.accepted=np.zeros(3)
        self.no_candidate=0;self.policy_rejected=0
        self.now=self.requests[0].arrival;self.plans=candidates(self.compiler,self.ledger,self.requests[0])
        return self.observation(),{}

    def observation(self):
        out=np.zeros((4,self.candidate_dim),np.float32);out[0,-1]=1.;mask=np.zeros(4,np.float32);mask[0]=1.
        state=np.zeros(self.state_dim,np.float32)
        state[:3]=self.ledger.used(self.now)/self.ledger.capacity
        state[3]=self.ledger.cache.usage(self.now).get('relay',0)/CACHE_CAPACITIES['relay']
        if self.done:return {'state':state,'candidates':out,'mask':mask}
        r=self.requests[self.i]
        future=sum((np.array(s.usage)*max(0.,s.end-max(self.now,s.start))
                    for p in self.ledger.active.values() for s in p.segments),np.zeros(3))/self.ledger.capacity
        state[4:7]=future
        state[7:10]=self.arrived/max(self.i,1)
        state[10:13]=self.accepted/np.maximum(self.arrived,1.)
        state[13:]=[r.rounds/7.,r.end-r.arrival,len(self.ledger.active)/48.,self.i/self.count,self.now/10.]
        for j,p in enumerate(self.plans,1):
            mask[j]=1.;out[j,:3]=p.peak/self.ledger.capacity
            out[j,3:6]=p.area/self.ledger.capacity
            out[j,6]=1000./4000 if p.kind=='ndn_pin' else 0.
            out[j,7:10]=np.eye(3)[KINDS.index(p.kind)]
            out[j,10:14]=[r.rounds/7.,r.weight,(r.end-r.arrival),
                         sum(v for k,v in p.reservation.items() if k.startswith('bytes:origin'))/20000.]
        return {'state':state,'candidates':out,'mask':mask}

    def step(self,action):
        if self.done or type(action) not in (int,np.int64,np.int32) or not 0<=action<=len(self.plans):
            raise ValueError('invalid masked action')
        r=self.requests[self.i];category=0 if r.allowed==('ip',) else 1 if 'ip' not in r.allowed else 2
        if not self.plans:self.no_candidate+=1
        elif not action:self.policy_rejected+=1
        self.arrived[category]+=1
        plan=self.plans[action-1] if action else None
        if plan:self.ledger.commit(plan);self.accepted[category]+=1
        self.decisions.append((r,plan));self.i+=1;self.done=self.i==len(self.requests)
        next_time=float('inf') if self.done else self.requests[self.i].arrival
        reward=0.
        for owner,p in tuple(self.ledger.active.items()):
            if p.request.end<=next_time:
                reward+=p.request.weight;self.ledger.release(owner)
        self.settled+=reward
        if not self.done:self.now=next_time
        self.plans=[] if self.done else candidates(self.compiler,self.ledger,self.requests[self.i])
        return self.observation(),reward,self.done,False,{}


def choose(env,method='greedy',price=.3):
    if not env.plans:return 0
    capacity=env.ledger.capacity
    scores=[]
    for p in env.plans:
        cost=float((p.area/capacity).sum())+(p.request.end-p.request.arrival)*(p.kind=='ndn_pin')*.25
        if method=='first_fit':return 1
        if method=='greedy':score=p.request.weight-1e-6*cost
        else:
            # Integrate the opportunity price across this plan's known intervals,
            # not just the currently free resource snapshot.
            penalty=0.
            for segment in p.segments:
                u=env.ledger.used(segment.start)/capacity;q=np.array(segment.usage)/capacity
                barrier=np.log((1.05-u)/np.maximum(1.05-u-q,1e-9))
                if method=='adaptive_price':barrier*=1.+u
                penalty+=(segment.end-segment.start)*barrier.sum()
            penalty+=(p.request.end-p.request.arrival)*(p.kind=='ndn_pin')*.25
            score=p.request.weight-price*penalty
        scores.append(score)
    return int(np.argmax([0.]+scores))
