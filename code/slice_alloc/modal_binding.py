"""Research point 1: semantic service contracts, joint modality and slice placement.

No future arrivals/positions enter encode or candidates. Physical execution is
ns-3/NFD; reservations are conservative envelopes, not packet SLA guarantees.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
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
from slice_alloc import patrol as p
from slice_alloc.initial_allocation import marginal_cost

PROTOCOL = 'binding-v1'
MODES = ('ip', 'id', 'ndn')
SERVICES = ('telemetry', 'object', 'command')
MAX_ACTIONS = 11  # reject + IP(2) + ID(4) + NDN(4)
STATE_DIM = 4*len(p.RESOURCES)+23
CANDIDATE_DIM = 2*len(p.RESOURCES)+11

@dataclass(frozen=True)
class Service:
    id: int
    tenant: int
    service: str
    members: tuple[int, ...]
    start: float
    stop: float
    period: float
    deadline: float
    payload: int
    chunks: int
    content: int
    reusable: bool
    version: int = 1
    minimum_ratio: float = .95

    @property
    def drain(self): return self.stop+self.deadline+.1
    @property
    def size(self): return self.payload*self.chunks
    @property
    def rounds(self): return max(1, math.ceil((self.stop-self.start-.04-1e-8)/self.period))
    @property
    def expected(self): return self.rounds*len(self.members)*self.chunks
    def json(self): return {**asdict(self), 'members': list(self.members)}
    def bound(self, mode):
        return p.Request(self.id, self.tenant, mode, self.members, self.start,
            self.stop, self.period, self.deadline, self.payload, self.chunks,
            self.content, self.version, self.minimum_ratio)


def scenario(seed, load=48, mix='balanced'):
    # Mobility and topology stay the same across all mixes and methods.
    cfg=p.scenario(seed,1,'mixed'); rng=np.random.default_rng(seed+93001)
    weights={'balanced':(.35,.4,.25), 'object_heavy':(.15,.7,.15),
             'mobile_heavy':(.25,.2,.55), 'novel':(0.,1.,0.)}[mix]
    services=[]
    for i,at in enumerate(sorted(rng.uniform(2.,18.,load)),1):
        kind=str(rng.choice(SERVICES,p=weights)); endpoint=int(rng.integers(5,17))
        tenant=int(rng.integers(1,3)); duration=float(rng.choice([3.,5.,7.]))
        if kind=='object':
            count=int(rng.choice([1,2])); members=tuple(sorted(int(n) for n in rng.choice(np.arange(5,17),count,replace=False)))
            # Held-out novel contract combines reusable large payload with tight SLA.
            chunks=int(rng.choice([4,8])) if mix!='novel' else 6
            services.append(Service(i,tenant,kind,members,float(at),float(at+duration),
                2.,.8 if mix!='novel' else .35,1000,chunks,int(rng.integers(1,4)),True,version=chunks,minimum_ratio=1.).json())
        else:
            services.append(Service(i,tenant,kind,(endpoint,),float(at),float(at+duration),
                float(rng.choice([.02,.04,.08])) if kind=='telemetry' else .3,.2 if kind=='telemetry' else .6,
                1000 if kind=='telemetry' else 512,1,1000+i,False).json())
    cfg.update(schema=PROTOCOL,requests=services)
    validate(cfg)
    return cfg


def validate(cfg):
    if cfg['schema']!=PROTOCOL: raise ValueError('schema mismatch')
    ids=set();last=-1
    for raw in cfg['requests']:
        if 'mode' in raw: raise ValueError('request must not prescribe a modality')
        r=Service(**{**raw,'members':tuple(raw['members'])})
        if r.service not in SERVICES or r.id in ids or r.start<last: raise ValueError('invalid ordered service')
        if not 1<=len(r.members)<=2 or len(set(r.members))!=len(r.members): raise ValueError('invalid endpoints')
        if r.stop<=r.start+.04 or min(r.period,r.deadline)<=0 or not 64<=r.payload<=1000 or not 1<=r.chunks<=32: raise ValueError('invalid traffic')
        if not 0<r.minimum_ratio<=1: raise ValueError('invalid SLA')
        if not r.reusable and r.chunks!=1: raise ValueError('live contract uses one chunk')
        ids.add(r.id);last=r.start


class Session(p.NativeSession):
    def __init__(self,cfg,prefix=None,trace=False):
        validate(cfg);self.temp=None
        if prefix is None:
            self.temp=tempfile.TemporaryDirectory(prefix='binding-');prefix=Path(self.temp.name)/'episode'
        self.prefix=Path(prefix);self.prefix.parent.mkdir(parents=True,exist_ok=True)
        self.config_path=Path(str(self.prefix)+'-scenario.json');self.config_path.write_text(json.dumps(cfg,indent=2)+'\n')
        self.stderr=open(str(self.prefix)+'-stderr.log','w+');self.messages=[];self.started=time.perf_counter()
        self.process=subprocess.Popen([str(p.NS3/'build/scratch/binding-scenario'),f'--config={self.config_path}',
            f'--output={self.prefix}',f'--trace={int(trace)}'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.stderr,text=True,bufsize=1,env=dict(os.environ,LD_LIBRARY_PATH=str(p.NS3/'build/lib')))
        self.observation=self.read()
    def read(self):
        ready,_,_=select.select([self.process.stdout],[],[],45)
        line=self.process.stdout.readline() if ready else ''
        if not line:
            self.stderr.flush();self.stderr.seek(0);err=self.stderr.read();self.close();raise RuntimeError(err[-4000:])
        value=json.loads(line)
        if value['schema']!=PROTOCOL:raise ValueError('observation schema mismatch')
        self.messages.append(dict(observation=value));return value
    def decide(self,d):
        if d['schema']!=PROTOCOL or d['request_id']!=self.observation['request_id']:raise ValueError('decision mismatch')
        if d['accept'] and d['mode'] not in MODES:raise ValueError('invalid modality')
        self.messages[-1]['decision']=d;self.process.stdin.write(json.dumps(d)+'\n');self.process.stdin.flush()
        self.observation=self.read()
        if self.observation['kind']=='done':
            if self.process.wait(timeout=10):raise RuntimeError('native failed')
            self.seconds=time.perf_counter()-self.started
        return self.observation


def compile_plan(r,mode,edge,server,cache,pub,obs):
    bound=r.bound(mode)
    if mode=='id':
        plan=p.compile_plan(replace(bound,period=r.period/r.chunks),edge,server,0,pub,obs)
        plan.request=bound
        return plan
    if mode=='ndn':
        # Only actual complete, leased, reusable objects justify a warm discount.
        cold=obs if r.reusable and cache else {**obs,'cached':[]}
        plan=p.compile_plan(bound,edge,server,cache,pub,cold)
        return plan
    variants=[]
    for aps in itertools.product(*(pub['allowed_aps'][str(n)] for n in r.members)):
        use=np.zeros(len(p.RESOURCES));rate=(r.payload+53)*r.chunks/r.period
        for ap in aps:
            for a,b in p.path(0,edge)+p.path(edge,ap):use[p.INDEX[f'wire:{a}:{b}']]+=rate
            use[p.INDEX['airtime']]+=(.00015+(r.payload+80)*8/24e6)*r.chunks/r.period
        variants.append(use)
    return p.Plan(bound,edge,edge,0,[(r.start,r.drain,np.max(variants,axis=0))],False)


def candidates(r,ledger,obs,fixed=False):
    if any(obs['associations'].get(str(n),0) not in ledger.pub['allowed_aps'][str(n)] for n in r.members):return []
    modes=(native_mode(r),) if fixed else MODES
    plans=[]
    for mode in modes:
        if mode=='id' and len(r.members)!=1:continue
        for edge in (1,2):
            choices=((s,0) for s in (1,2)) if mode=='id' else ((edge,c) for c in (0,edge)) if mode=='ndn' and r.reusable else ((edge,0),)
            for server,cache in choices:
                plan=compile_plan(r,mode,edge,server,cache,ledger.pub,obs)
                if ledger.feasible(plan):plans.append(plan)
    return plans


def native_mode(r): return 'ndn' if r.reusable else 'id' if r.service=='command' else 'ip'


class BindingEnv(gym.Env):
    def __init__(self,seed=1,loads=(32,48),fixed=False,motion=True):
        self.rng=np.random.default_rng(seed);self.loads=loads;self.fixed=fixed;self.motion_enabled=motion;self.session=None
        self.action_space=spaces.Discrete(MAX_ACTIONS)
        self.observation_space=spaces.Dict({'state':spaces.Box(-10,10,(STATE_DIM,),np.float32),
            'candidates':spaces.Box(-10,10,(MAX_ACTIONS,CANDIDATE_DIM),np.float32),
            'mask':spaces.Box(0,1,(MAX_ACTIONS,),np.float32)})
        self.episode_specs=[]
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if self.session:self.session.close()
        if seed is not None:self.rng=np.random.default_rng(seed)
        opts=options or {};cfg=opts.get('scenario')
        if cfg is None:cfg=scenario(int(self.rng.integers(20000000,21000000)),int(self.rng.choice(self.loads)))
        self.episode_specs.append(cfg['radio_seed']-1)
        self.cfg=cfg;self.requests={r['id']:Service(**{**r,'members':tuple(r['members'])}) for r in cfg['requests']}
        self.ledger=p.Ledger(cfg['public']);self.records=[];self.settled={};self.previous={};self.previous_time=None
        self.motion_frame=None;self.velocities={};self.resource_integral=np.zeros(len(p.RESOURCES))
        self.session=Session(cfg,opts.get('prefix'),opts.get('trace',False));self.current=self.session.observation
        return self.encode(),{}
    def encode(self):
        state=np.zeros(STATE_DIM,np.float32);rows=np.zeros((MAX_ACTIONS,CANDIDATE_DIM),np.float32);mask=np.zeros(MAX_ACTIONS,np.float32);mask[0]=1
        self.options=[]
        if self.current['kind']=='done':return dict(state=state,candidates=rows,mask=mask)
        now=self.current['time'];r=self.requests[self.current['request_id']];self.ledger.release(now)
        usage=[self.ledger.usage(now+dt)/self.ledger.capacity for dt in (0,1,3,6)]
        state[:4*len(p.RESOURCES)]=np.concatenate(usage)
        frame=(now,r.id)
        if frame!=self.motion_frame:
            dt=now-self.previous_time if self.previous_time is not None else 0
            self.velocities={n:(np.asarray(v)-self.previous[n])/dt if dt>1e-9 else np.zeros(3) for n,v in self.current['positions'].items()}
            self.previous={n:np.asarray(v) for n,v in self.current['positions'].items()};self.previous_time=now;self.motion_frame=frame
        mobility=[]
        for n in r.members:
            key=str(n);pos=np.array(self.current['positions'][key]);ap=self.current['associations'][key]
            distances=[np.linalg.norm(pos-[x,0,10]) for x in (0,150)]
            mobility.append([pos[0]/150,min(distances)/120,self.velocities[key][0]/11,
                float(self.previous_time is not None),float(self.cfg['public']['missions'][key]['role']=='patrol'),
                float(ap==3),float(ap==4),float(ap==0)])
        motion=np.mean(mobility,axis=0) if self.motion_enabled else np.zeros(8)
        state[4*len(p.RESOURCES):]=[*(float(r.service==s) for s in SERVICES),float(r.reusable),
            (r.stop-r.start)/8,r.period/2,r.deadline,r.payload/1000,r.chunks/8,len(r.members)/2,
            r.minimum_ratio,now/30,len(self.ledger.plans)/20,*motion,0.,0.]
        self.options=candidates(r,self.ledger,self.current,self.fixed)
        for i,plan in enumerate(self.options,1):
            integral,_=marginal_cost(self.ledger,plan);mode=plan.request.mode
            rows[i]=np.r_[plan.peak/self.ledger.capacity,integral/8,
                [mode==s for s in MODES],[plan.edge==s for s in (1,2)],
                [plan.server==s for s in (1,2)],[plan.cache==s for s in (1,2)],plan.warm,1.]
            mask[i]=1
        return dict(state=state,candidates=rows,mask=mask)
    def step(self,action):
        action=int(action);r=self.requests[self.current['request_id']]
        if not 0<=action<=len(self.options):raise ValueError('masked action')
        if action:
            plan=self.options[action-1];integral,_=marginal_cost(self.ledger,plan)
            self.resource_integral+=integral;self.ledger.add(plan)
            d={**plan.decision(),'schema':PROTOCOL,'mode':plan.request.mode}
            # Plan.decision validates legacy protocol first; wire uses the new schema.
        else:d=dict(schema=PROTOCOL,request_id=r.id,accept=False,edge=0,server=0,cache=0,mode='ip')
        self.records.append(dict(request=r.id,service=r.service,decision=d,feasible=len(self.options)))
        self.current=self.session.decide(d);reward=0.
        for s in self.current['settled']:self.settled[s['id']]=s;reward+=float(s['success'])
        done=self.current['kind']=='done'
        return self.encode(),reward,done,False,{}
    def result(self):
        if self.current['kind']!='done':raise ValueError('incomplete')
        success=sum(s['success'] for s in self.settled.values())
        return dict(requests=len(self.requests),success=success,admitted=len(self.settled),
            resource_time=self.resource_integral.tolist(),resource_per_success=float(self.resource_integral.sum()/max(success,1)),
            by_service={kind:dict(requests=sum(r.service==kind for r in self.requests.values()),
                success=sum(s['success'] for i,s in self.settled.items() if self.requests[i].service==kind)) for kind in SERVICES},
            modes=dict(collections.Counter(x['decision']['mode'] for x in self.records if x['decision']['accept'])),
            decisions=self.records,settled=list(self.settled.values()),observations=self.session.messages,native_seconds=self.session.seconds)
    def close(self):
        if self.session:self.session.close();self.session=None


def exponential_price(ledger,plan):
    start,end=plan.request.start,plan.request.drain
    points={start,end}
    for old in ledger.plans+[plan]:
        points.update(t for a,b,_ in old.segments for t in (a,b) if start<t<end)
        if old.lease:points.update(t for t in old.lease[-2:] if start<t<end)
    result=0.;points=sorted(points)
    for a,b in zip(points,points[1:]):
        u=ledger.usage((a+b)/2)/ledger.capacity
        delta=(ledger.usage((a+b)/2,plan)/ledger.capacity)-u
        result+=(b-a)*float(np.dot(delta,np.exp(3*u)))
    return result/(end-start)


def baseline(name,env,threshold=math.inf):
    opts=env.options
    if not opts:return 0
    r=env.requests[env.current['request_id']]
    indices=[i for i,x in enumerate(opts) if name!='fixed_price' or x.request.mode==native_mode(r)]
    if not indices:return 0
    if name=='first_fit':return indices[0]+1
    scores={i:exponential_price(env.ledger,opts[i]) for i in indices}
    best=min(indices,key=lambda i:scores[i])
    return best+1 if scores[best]<=threshold else 0
