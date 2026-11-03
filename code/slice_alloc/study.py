"""One experiment contract: allocation, measured-load prediction, runtime DRR."""
from pathlib import Path
import hashlib
import json
import os
import time
import gzip
import shutil

import numpy as np
from gymnasium import spaces
from slice_alloc import allocation_final as base, modal_binding as b

PACKAGE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('LOWALT_OUTPUT', str(PACKAGE/'results/final'))).expanduser().resolve()
INITS = (11,23,37,53,71,89,101,127)
LOADS = (32,64,96)
MOBILITY = ('stationary','local','cross')
VARIANTS = ('A','fixed','no_motion','AP')
RULES = ('greedy','native','modal_greedy','first_fit')
VALIDATION = tuple(range(121000,121024))
TEST = tuple(range(131000,131060))
BUDGETS = (32768,65536)
# Order agrees exactly with the ten native PointToPoint device queues.
WIRES = ((0,1),(1,0),(0,2),(2,0),(1,2),(2,1),(1,3),(3,1),(2,4),(4,2))

def write(path, value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name(path.name+'.tmp')
 temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def scenario(seed,load,mobility='cross',qos='none',scale=1.):
 cfg=base.scenario(seed,load,mobility='stationary' if mobility=='stationary' else 'mixed')
 cfg['public']['qos']=qos
 for key in ('backhaul_mbps','cross_mbps','access_mbps'):cfg['public'][key]*=scale
 points={}
 for motion in cfg['private']['motions']:
  n=str(motion['node']);local=mobility=='local';right=45. if local else 145.
  if local:
   cfg['private']['initial_positions'][n]=[35.,0.,80.]
   cfg['public']['missions'][n].update(x_min=30.,x_max=45.)
   cfg['public']['allowed_aps'][n]=[3]
  points[n]=[[right,8.,80.],[right,-8.,80.],[30.,-8.,80.],[30.,8.,80.]]
 cfg['private']['waypoints']=points
 return cfg

class Session(base.FinalSession):
 forecast90=None
 def decide(self,decision):
  if self.forecast90 is not None:decision={**decision,'forecast90':self.forecast90.tolist()}
  return super().decide(decision)

class Env(base.AllocationEnv):
 session_class=Session
 def __init__(self,variant='A',predictor=None,scale=1.,**kwargs):
  self.variant,self.predictor,self.scale=variant,predictor,scale
  super().__init__(no_calendar=True,no_motion=variant=='no_motion',fixed=variant=='fixed',**kwargs)
  self.observation_space=spaces.Dict({**self.observation_space.spaces,
   'state':spaces.Box(-100,100,(self.observation_space['state'].shape[0]+40,),np.float32)})
 def reset(self,seed=None,options=None):
  opts=dict(options or {})
  if 'scenario' not in opts:
   if seed is not None:self.rng=np.random.default_rng(seed)
   seed_value=int(self.rng.integers(51000000,52000000));mobility=str(self.rng.choice(MOBILITY))
   opts['scenario']=scenario(seed_value,int(self.rng.choice(LOADS)),mobility,scale=self.scale)
  return super().reset(seed=seed,options=opts)
 def encode(self):
  obs=super().encode();observed=np.zeros(20,np.float32);forecast=np.zeros(20,np.float32)
  if self.current['kind']!='done':
   history=self.current.get('load_history',[])
   if history:observed=np.asarray(history[-1]['load']+history[-1]['queue'],np.float32)
   if self.predictor is not None:
    predicted=self.predictor(history);forecast=np.asarray(predicted,np.float32).reshape(-1)
    self.session.forecast90=np.asarray(predicted)[:,1]
   if self.variant=='no_motion':
    # Remove explicit per-endpoint geometry, velocity, role and group mobility.
    offset=b.STATE_DIM+12*8
    for group in range(9):obs['state'][offset+group*7+6]=0.
  return {**obs,'state':np.r_[obs['state'],observed,forecast]}

def rule_action(name,env,parameters=(0.,0.,0.)):
 if not env.options:return 0
 r=env.requests[env.current['request_id']];plans=env.options
 def resource(plan):return float(np.sum(plan.peak/env.ledger.capacity))
 if name=='first_fit':return 1
 if name=='greedy':return min(range(len(plans)),key=lambda i:resource(plans[i]))+1
 if name=='native':
  desired={'telemetry':'ip','command':'id','object':'ndn'}[r.service]
  eligible=[i for i,p in enumerate(plans) if p.request.mode==desired]
  if not eligible:eligible=list(range(len(plans)))
  return min(eligible,key=lambda i:(r.service=='object' and plans[i].request.mode=='ndn' and not plans[i].cache,resource(plans[i])))+1
 if name!='modal_greedy':raise ValueError(name)
 alpha,beta,cache_penalty=parameters
 history=env.current.get('load_history',[])
 measured=np.array(history[-1]['load'] if history else [0.]*10)
 waiting=np.array(history[-1]['queue'] if history else [0.]*10)
 speed=np.mean([np.linalg.norm(env.velocities[str(n)][:2]) for n in r.members])
 def score(p):
  normalized=p.peak/env.ledger.capacity
  traffic=float(np.dot(normalized[:10],1+measured)+normalized[10:].sum())
  delay=float(waiting[normalized[:10]>0].sum())/r.deadline
  binding_risk=(p.request.mode=='ip' and r.service!='telemetry')*min(1.,speed*(r.stop-r.start)/120.)
  no_cache=r.service=='object' and not (p.request.mode=='ndn' and p.cache)
  return traffic+alpha*delay+beta*binding_risk+cache_penalty*no_cache
 return min(range(len(plans)),key=lambda i:score(plans[i]))+1

def episode(cfg,method,model=None,predictor=None,prefix=None,parameters=(0.,0.,0.)):
 variant=method if method in VARIANTS else 'AP' if method.startswith('AP_') else 'A'
 env=Env(variant=variant,predictor=predictor);timing=[]
 try:
  obs,_=env.reset(options={'scenario':cfg,'prefix':prefix,'trace':prefix is not None});done=False;reward=0.
  while not done:
   start=time.perf_counter()
   action=int(model.predict(obs,deterministic=True)[0]) if model else rule_action(method,env,parameters)
   timing.append(time.perf_counter()-start);assert obs['mask'][action]>.5
   obs,value,done,_,_=env.step(action);reward+=value
  result=env.result();assert abs(reward-result['success'])<1e-8
  result['decision_ms']=1000*float(np.mean(timing))
  result['scenario_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
  if prefix:
   write(str(prefix)+'-result.json',result)
   for suffix in ('events.csv','telemetry.jsonl'):
    raw=Path(str(prefix)+'-'+suffix)
    with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
    raw.unlink()
  return result
 finally:env.close()
