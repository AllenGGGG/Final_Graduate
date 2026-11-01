"""Paper-scoped patrol experiment. Native radio observations, fixed modalities.

Public contracts are deliberately separated from the native provider's private
trajectory. This module never uses private fields when constructing candidates.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import collections
import itertools
import json
import math
import os
import select
import subprocess
import tempfile
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
NS3 = Path(os.environ.get('LOWALT_NS3_ROOT', str(Path.home()/'.local/share/lowalt-ndnsim/ns-3'))).expanduser().resolve()
PROTOCOL = 'patrol-v2'
SCHEMA = json.loads((Path(__file__).with_name('patrol_contract.schema.json')).read_text())
VALIDATORS = {name: Draft202012Validator({'$ref': '#/$defs/'+name, '$defs': SCHEMA['$defs']}) for name in SCHEMA['$defs']}
WIRES = ((0,1),(1,0),(0,2),(2,0),(1,2),(2,1),(1,3),(3,1),(2,4),(4,2))
RESOURCES = tuple(f'wire:{a}:{b}' for a,b in WIRES)+('airtime','mapping:1','mapping:2','mapcache:1','mapcache:2','cache:1','cache:2')
INDEX = {s:i for i,s in enumerate(RESOURCES)}
PROFILES = ('mixed','bandwidth','mapping','cache')
MAX_ACTIONS = 5  # reject + 2 edges x {two servers OR no-cache/cache}
STATE_DIM = len(RESOURCES)+16
# Dense reward-shaping scale for the congestion term in step(); calibrated
# below so that one full-success settlement (reward=1) still dominates the
# shaping term accumulated over a whole episode. See step() for rationale.
SHAPING_WEIGHT = 0.02
CANDIDATE_DIM = len(RESOURCES)+13

def validate(kind, value):
    VALIDATORS[kind].validate(value)

def generate_header():
    keys = sorted({k for d in SCHEMA['$defs'].values() for k in d['properties']})
    return '// Generated from patrol_contract.schema.json; do not edit.\n#pragma once\nnamespace contract {\nconstexpr const char* PROTOCOL="'+PROTOCOL+'";\n'+''.join(f'constexpr const char* {k.upper()}="{k}";\n' for k in keys)+'}\n'

@dataclass(frozen=True)
class Request:
    id: int
    tenant: int
    mode: str
    members: tuple[int,...]
    start: float
    stop: float
    period: float
    deadline: float
    payload: int
    chunks: int
    content: int
    version: int = 1
    minimum_ratio: float = .95

    @property
    def drain(self): return self.stop+self.deadline+.1
    @property
    def size(self): return self.payload*self.chunks
    @property
    def prefix(self): return f'/mission/t{self.tenant}/map{self.content}/v{self.version}'
    @property
    def rounds(self): return max(1,math.ceil((self.stop-self.start-.04-1e-8)/self.period))
    @property
    def expected(self): return self.rounds*len(self.members)*(self.chunks if self.mode=='ndn' else 1)
    def json(self):
        d=asdict(self);d['members']=list(self.members);validate('request',d);return d


def public_config(profile='mixed', uavs=12):
    # Tenant mapping quotas are service limits; not the physical RAM limit of an edge.
    missions={str(n):dict(role='patrol' if (n-5)%3==0 else 'hover',x_min=30. if (n-5)%3==0 else 20. if (n-5)%2==0 else 145.,x_max=150. if (n-5)%3==0 else 20. if (n-5)%2==0 else 145.,altitude=80.) for n in range(5,5+uavs)}
    allowed={n:[ap for ap,x in ((3,0.),(4,150.)) if max(m['x_min'],x-math.sqrt(120**2-70**2))<=min(m['x_max'],x+math.sqrt(120**2-70**2))] for n,m in missions.items()}
    return dict(uavs=uavs, mapping_entries=2 if profile=='mapping' else 8,mapcache_entries=8,missions=missions,
                cache_bytes=8000 if profile=='cache' else 16000,
                backhaul_mbps=.45 if profile=='bandwidth' else 1.,
                cross_mbps=.5, access_mbps=2., profile=profile,
                allowed_aps=allowed,
                max_handover=2, mapping_ttl=.6,airtime_budget=.65)


def scenario(seed=1, load=24, profile='mixed', uavs=12):
    rng=np.random.default_rng(seed)
    pub=public_config(profile,uavs)
    positions={};motions=[]
    for n in range(5,5+uavs):
        mission=pub['missions'][str(n)];mobile=mission['role']=='patrol'
        x=float(rng.uniform(mission['x_min'],55)) if mobile else mission['x_min']
        positions[str(n)]=[x,float(rng.uniform(-3,3)),80.]
        if mobile: motions.append(dict(node=n,speed=float(rng.choice([5.,8.,11.])),right=True,start=float(rng.uniform(0,.5))))
    probs={'mixed':(.34,.33,.33),'bandwidth':(.75,.125,.125),'mapping':(.125,.75,.125),'cache':(.125,.125,.75)}[profile]
    reqs=[]
    # The same horizon at both loads; increasing load increases arrival count.
    times=sorted(rng.uniform(2.,18.,load))
    for i,at in enumerate(times,1):
        mode=str(rng.choice(['ip','id','ndn'],p=probs));tenant=int(rng.integers(1,3))
        endpoint=int(rng.integers(5,5+uavs));duration=float(rng.choice([3.,5.,7.]))
        if mode=='ndn':
            members=tuple(sorted(int(v) for v in rng.choice(np.arange(5,5+uavs),size=min(2,uavs),replace=False)))
            content=int(rng.integers(1,4));chunks=4 if content!=3 else 8
            r=Request(i,tenant,mode,members,float(at),float(at+duration),2.,.8,1000,chunks,content,minimum_ratio=1.)
        elif mode=='id':
            r=Request(i,tenant,mode,(endpoint,),float(at),float(at+duration),.15,.6,256,1,i,minimum_ratio=.95)
        else:
            r=Request(i,tenant,mode,(endpoint,),float(at),float(at+duration),float(rng.choice([.02,.04,.08])),.2,1000,1,i,minimum_ratio=.95)
        reqs.append(r.json())
    return dict(schema=PROTOCOL,radio_seed=seed+1,public=pub,private=dict(initial_positions=positions,motions=motions),requests=reqs)


def request_from_json(d):
    validate('request',d)
    return Request(**{**d,'members':tuple(d['members'])})


def capacities(pub):
    values=[]
    for a,b in WIRES:
        mbps=pub['backhaul_mbps'] if 0 in (a,b) else pub['cross_mbps'] if {a,b}=={1,2} else pub['access_mbps']
        values.append(mbps*1e6/8)
    # Shared channel, both APs. 0.65 is a conservative admission budget, not PHY throughput.
    return np.array(values+[pub['airtime_budget'],pub['mapping_entries'],pub['mapping_entries'],pub['mapcache_entries'],pub['mapcache_entries'],pub['cache_bytes'],pub['cache_bytes']],float)


def path(a,b):
    if a==b:return ()
    adjacency={0:(1,2),1:(0,2,3),2:(0,1,4),3:(1,),4:(2,)}
    q=collections.deque([(a,())]);seen={a}
    while q:
        node,edges=q.popleft()
        for nxt in adjacency[node]:
            if nxt==b:return edges+((node,nxt),)
            if nxt not in seen:seen.add(nxt);q.append((nxt,edges+((node,nxt),)))
    raise ValueError((a,b))

@dataclass
class Plan:
    request: Request
    edge: int
    server: int
    cache: int
    segments: list[tuple[float,float,np.ndarray]]
    warm: bool
    @property
    def peak(self):
        result=np.max([s[2] for s in self.segments],axis=0)
        if self.cache:result[INDEX[f'cache:{self.cache}']]=self.request.size
        return result
    @property
    def lease(self):
        r=self.request
        return (self.cache,r.tenant,r.content,r.version,r.size,r.start,r.drain) if self.cache else None
    def decision(self):
        d=dict(schema=PROTOCOL,request_id=self.request.id,accept=True,edge=self.edge,server=self.server,cache=self.cache);validate('decision',d);return d

class Ledger:
    def __init__(self,pub):self.pub=pub;self.capacity=capacities(pub);self.plans=[]
    def usage(self,t,extra=None):
        plans=self.plans+([extra] if extra else []);out=np.zeros(len(RESOURCES));leases={}
        for p in plans:
            for a,b,u in p.segments:
                if a<=t<b:out+=u
            if p.lease:
                n,tenant,c,v,size,a,b=p.lease
                if a<=t<b:leases[n,tenant,c,v]=size
        for (n,*_),size in leases.items():out[INDEX[f'cache:{n}']]+=size
        return out
    def feasible(self,p):
        points=sorted({v for x in self.plans+[p] for s in x.segments for v in s[:2]}|{v for x in self.plans+[p] if x.lease for v in x.lease[-2:]})
        return all(np.all(self.usage((a+b)/2,p)<=self.capacity+1e-8) for a,b in zip(points,points[1:]) if a<p.request.drain and b>p.request.start)
    def add(self,p):
        if not self.feasible(p):raise ValueError('infeasible transaction')
        self.plans.append(p)
    def release(self,t):self.plans=[p for p in self.plans if p.request.drain>t+1e-8]


def compile_plan(r,edge,server,cache,pub,observation):
    warm=any(x['node']==edge and (x['tenant'],x['content'],x['version'])==(r.tenant,r.content,r.version) and x['bytes']>=r.size for x in observation['cached']) if r.mode=='ndn' else False
    def add_wire(v,a,b,rate):
        for x,y in path(a,b):v[INDEX[f'wire:{x}:{y}']]+=rate
    def air_cost(size,packets):return packets*(.00015+(size+80)*8/24e6)
    # Per-AP envelopes are mutually exclusive. A single shared air budget is used.
    variants=[]
    choices=[pub['allowed_aps'][str(n)] for n in r.members]
    for aps in itertools.product(*choices):
        u=np.zeros(len(RESOURCES))
        for n,ap in zip(r.members,aps):
            if r.mode=='ip':
                rate=(r.payload+37)/r.period;add_wire(u,ap,edge,rate);add_wire(u,edge,0,rate)
                u[INDEX['airtime']]+=air_cost(r.payload+37,1/r.period)
            elif r.mode=='id':
                # Up to 2 data attempts, query/registration retries and acknowledgements.
                add_wire(u,0,edge,(r.payload+53)/r.period)
                add_wire(u,edge,ap,2*(r.payload+90)/r.period)
                add_wire(u,ap,edge,2*53/r.period)
                control=10*53+12*53*(1+pub['max_handover'])/(r.stop-r.start)
                add_wire(u,ap,server,control);add_wire(u,server,ap,control)
                add_wire(u,edge,server,10*53);add_wire(u,server,edge,10*53)
                u[INDEX['airtime']]+=air_cost(r.payload+90,2/r.period)+air_cost(53,2/r.period+2*control/53)
                u[INDEX[f'mapping:{server}']]=1
            else:
                attempts=2;round_rate=r.chunks/r.period
                # Native wire encodings are variable; use declared upper bounds.
                add_wire(u,ap,edge,attempts*100*round_rate)
                add_wire(u,edge,ap,attempts*(r.payload+128)*round_rate)
                if not warm:
                    add_wire(u,edge,0,attempts*100*round_rate)
                    add_wire(u,0,edge,attempts*(r.payload+128)*round_rate)
                u[INDEX['airtime']]+=air_cost(100,attempts*round_rate)+air_cost(r.payload+128,attempts*round_rate)
        variants.append(u)
    rate=np.max(variants,axis=0)
    state=np.zeros(len(RESOURCES))
    if r.mode=='id':
        state[INDEX[f'mapping:{server}']]=1;state[INDEX[f'mapcache:{edge}']]=1;rate[INDEX[f'mapping:{server}']]=0
    # Cache discounted only when a complete object already exists and a current
    # lease plus this lease retain it for this plan's service interval.
    return Plan(r,edge,server,cache,[(r.start,r.drain,state+rate)],warm)


def candidates(r,ledger,observation):
    pub=ledger.pub
    if any(observation['associations'].get(str(n),0) not in pub['allowed_aps'][str(n)] for n in r.members):return []
    result=[]
    for edge in (1,2):
        for server,cache in ((s,0) for s in (1,2)) if r.mode=='id' else ((edge,c) for c in (0,edge)) if r.mode=='ndn' else ((edge,0),):
            p=compile_plan(r,edge,server,cache,pub,observation)
            # Without a lease, an observed cache object may disappear at its owner's
            # expiry, so it cannot justify discounting a new no-cache plan.
            if p.warm and not p.cache:
                cold={**observation,'cached':[]};p=compile_plan(r,edge,server,cache,pub,cold)
            if ledger.feasible(p):result.append(p)
    return result

class NativeSession:
    def __init__(self,cfg,prefix=None,trace=False):
        validate('scenario',cfg)
        self.temp=None
        if prefix is None:self.temp=tempfile.TemporaryDirectory(prefix='lowalt-patrol-');prefix=Path(self.temp.name)/'episode'
        self.prefix=Path(prefix);self.prefix.parent.mkdir(parents=True,exist_ok=True)
        self.config_path=Path(str(self.prefix)+'-scenario.json');self.config_path.write_text(json.dumps(cfg,indent=2)+'\n')
        self.stderr=open(str(self.prefix)+'-stderr.log','w+');self.messages=[];self.started=time.perf_counter()
        self.process=subprocess.Popen([str(NS3/'build/scratch/patrol-scenario'),f'--config={self.config_path}',f'--output={self.prefix}',f'--trace={int(trace)}'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1,env=dict(os.environ,LD_LIBRARY_PATH=str(NS3/'build/lib')))
        self.observation=self.read()
    def read(self):
        ready,_,_=select.select([self.process.stdout],[],[],45)
        line=self.process.stdout.readline() if ready else ''
        if not line:
            self.stderr.flush();self.stderr.seek(0);err=self.stderr.read();self.close();raise RuntimeError('native observation missing: '+err[-4000:])
        value=json.loads(line);validate('observation',value);self.messages.append(dict(observation=value));return value
    def decide(self,d):
        validate('decision',d)
        if d['request_id']!=self.observation['request_id']:raise ValueError('wrong current request')
        self.messages[-1]['decision']=d;self.process.stdin.write(json.dumps(d)+'\n');self.process.stdin.flush();self.observation=self.read()
        if self.observation['kind']=='done':
            if self.process.wait(timeout=10):raise RuntimeError('native provider failed')
            self.seconds=time.perf_counter()-self.started
        return self.observation
    def close(self):
        if hasattr(self,'process'):
            if self.process.poll() is None:self.process.terminate();self.process.wait(timeout=10)
            for stream in (self.process.stdin,self.process.stdout):stream.close()
        if hasattr(self,'stderr'):self.stderr.close()
        if self.temp:self.temp.cleanup();self.temp=None

class PatrolEnv(gym.Env):
    metadata={}
    def __init__(self,seed=1,loads=(24,40),profiles=PROFILES,uavs=12,shaping=True):
        self.rng=np.random.default_rng(seed);self.loads=loads;self.profiles=profiles;self.uavs=uavs;self.session=None;self.episode_specs=[];self.shaping=shaping
        self.action_space=spaces.Discrete(MAX_ACTIONS)
        self.observation_space=spaces.Dict({'state':spaces.Box(-10,10,(STATE_DIM,),np.float32),'candidates':spaces.Box(-10,10,(MAX_ACTIONS,CANDIDATE_DIM),np.float32),'mask':spaces.Box(0,1,(MAX_ACTIONS,),np.float32)})
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if self.session:self.session.close()
        cfg=options['scenario'] if options and 'scenario' in options else scenario(int(self.rng.integers(10000,1000000)),int(self.rng.choice(self.loads)),str(self.rng.choice(self.profiles)),self.uavs)
        self.episode_specs.append(dict(seed=cfg['radio_seed']-1,profile=cfg['public']['profile'],load=len(cfg['requests'])))
        self.cfg=cfg;self.requests={r['id']:request_from_json(r) for r in cfg['requests']};self.ledger=Ledger(cfg['public']);self.records=[];self.settled={};self.rejections=collections.Counter();self.times=[];self.pressure=collections.Counter();self.peak_usage=np.zeros(len(RESOURCES))
        self.session=NativeSession(cfg,prefix=options.get('prefix') if options else None,trace=options.get('trace',False) if options else False)
        self.current=self.session.observation
        return self.encode(),{}
    def encode(self):
        state=np.zeros(STATE_DIM,np.float32);rows=np.zeros((MAX_ACTIONS,CANDIDATE_DIM),np.float32);mask=np.zeros(MAX_ACTIONS,np.float32);mask[0]=1
        if self.current['kind']=='done':return dict(state=state,candidates=rows,mask=mask)
        r=self.requests[self.current['request_id']];now=self.current['time'];self.ledger.release(now);state[:len(RESOURCES)]=self.ledger.usage(now)/self.ledger.capacity
        tail=[r.mode=='ip',r.mode=='id',r.mode=='ndn',(r.stop-r.start)/8,r.payload/1000,r.chunks/8,len(r.members)/4,r.period/2,r.deadline,r.start/30,np.mean([self.current['associations'][str(n)]==3 for n in r.members]),np.mean([self.current['associations'][str(n)]==4 for n in r.members])]
        # Dynamic context: when currently held reservations free up, and how
        # many admitted-but-not-yet-released requests are still in flight.
        # Both derived only from self.ledger.plans (already-admitted plans),
        # never from len(self.requests) or len(self.records): either would
        # leak the total/arrived request count, which is not public unless
        # explicitly declared as a contract field (it is not, here).
        held=[p.request.drain-now for p in self.ledger.plans]
        tail+= [len(held)/8,
                (min(held)/8 if held else 0.),
                (float(np.mean(held))/8 if held else 0.),
                (max(held)/8 if held else 0.)]
        state[len(RESOURCES):]=tail
        self.options=candidates(r,self.ledger,self.current)
        self.peak_usage=np.maximum(self.peak_usage,self.ledger.usage(self.current['time'])/self.ledger.capacity)
        for edge in (1,2):
            for server,cache in ((s,0) for s in (1,2)) if r.mode=='id' else ((edge,c) for c in (0,edge)) if r.mode=='ndn' else ((edge,0),):
                p=compile_plan(r,edge,server,cache,self.ledger.pub,self.current)
                for start,stop,_ in p.segments:
                    points=sorted({start,stop}|{x for old in self.ledger.plans for seg in old.segments for x in seg[:2] if start<x<stop})
                    for a,b in zip(points,points[1:]):
                        for index in np.flatnonzero(self.ledger.usage((a+b)/2,p)>self.ledger.capacity+1e-8):self.pressure[RESOURCES[index]]+=1
        # Candidate-feature engineering (history-pressure exposure + recent
        # rejection rate) was tried and retracted: a 32768-step probe showed
        # no improvement (and a slight DDQN regression), so it is reverted
        # here rather than left silently residual in CANDIDATE_DIM.
        for i,p in enumerate(self.options,1):
            mask[i]=1;rows[i,:len(RESOURCES)]=p.peak/self.ledger.capacity
            rows[i,len(RESOURCES):]=[p.edge==1,p.edge==2,p.server==1,p.server==2,p.cache==1,p.cache==2,p.warm,r.mode=='ip',r.mode=='id',r.mode=='ndn',(r.stop-r.start)/8,len(r.members)/4,1.]
        rows[0,-1]=-1.
        return dict(state=state,candidates=rows,mask=mask)
    def step(self,action):
        r=self.requests[self.current['request_id']];action=int(action);started=time.perf_counter()
        if action<0 or action>len(self.options):raise ValueError('masked action')
        # Reward shaping: the real SLA-success signal below is sparse (nonzero
        # only when the native simulator settles a request), so most of the
        # 24-40 decision steps per episode carry zero learning signal. This
        # adds a small dense term, the negative of the same congestion-price
        # score the 'price' rule uses to rank candidates (peak usage with a
        # quadratic congestion penalty, plus an explicit cache cost). It is
        # observable BEFORE settlement (uses only current occupancy and this
        # candidate's own declared usage, no future information), and is
        # scaled far below 1.0 so the true success reward still dominates.
        shaping=0.
        if action:
            p=self.options[action-1];self.ledger.add(p);d=p.decision()
            u=self.ledger.usage(r.start)/self.ledger.capacity
            congestion=float(np.dot(p.peak/self.ledger.capacity,1+4*u*u))
            if p.cache:congestion+=r.size/self.ledger.capacity[INDEX[f'cache:{p.cache}']]*.1
            if self.shaping:shaping=-SHAPING_WEIGHT*congestion
        else:
            d=dict(schema=PROTOCOL,request_id=r.id,accept=False,edge=0,server=0,cache=0)
            self.rejections['no_feasible' if not self.options else 'active']+=1
        self.records.append(dict(request=r.id,mode=r.mode,decision=d,feasible=len(self.options),time=self.current['time']))
        self.current=self.session.decide(d);reward=shaping
        for s in self.current['settled']:self.settled[s['id']]=s;reward+=float(s['success'])
        done=self.current['kind']=='done'
        return self.encode(),reward,done,False,{}
    def result(self):
        if self.current['kind']!='done':raise ValueError('incomplete episode')
        modes={}
        for mode in ('ip','id','ndn'):
            ids=[r.id for r in self.requests.values() if r.mode==mode];admitted=[i for i in ids if i in self.settled]
            modes[mode]=dict(requests=len(ids),admitted=len(admitted),success=sum(self.settled[i]['success'] for i in admitted),expected=sum(self.settled[i]['expected'] for i in admitted),on_time=sum(self.settled[i]['on_time'] for i in admitted))
        return dict(requests=len(self.requests),admitted=len(self.settled),success=sum(s['success'] for s in self.settled.values()),by_mode=modes,rejections=dict(self.rejections),pressure=dict(self.pressure),peak_usage=dict(zip(RESOURCES,self.peak_usage.tolist())),native_seconds=self.session.seconds,decisions=self.records,settled=list(self.settled.values()),observations=self.session.messages)
    def close(self):
        if self.session:self.session.close();self.session=None


def baseline(name,env):
    opts=env.options
    if not opts:return 0
    r=env.requests[env.current['request_id']]
    def price(p):
        u=env.ledger.usage(r.start)/env.ledger.capacity
        score=float(np.dot(p.peak/env.ledger.capacity,1+4*u*u))
        if p.cache:score+=r.size/env.ledger.capacity[INDEX[f'cache:{p.cache}']]*.1
        return score
    if name=='first_fit':return 1
    if name=='near_edge':
        aps=[env.current['associations'][str(n)]-2 for n in r.members]
        return min(range(len(opts)),key=lambda i:(sum(e!=opts[i].edge for e in aps),price(opts[i])))+1
    if name=='greedy':return min(range(len(opts)),key=lambda i:np.sum(opts[i].peak/env.ledger.capacity))+1
    if name=='price':return min(range(len(opts)),key=lambda i:price(opts[i]))+1
    if name=='cache_aware':
        # Covers a strong rule that anticipates repeated public-map requests.
        return min(range(len(opts)),key=lambda i:(0 if (r.mode=='ndn' and opts[i].cache) or r.mode!='ndn' else 1,price(opts[i])))+1
    raise ValueError(name)
