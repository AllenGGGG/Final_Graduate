"""Mission-stage request arrivals; identical packet network and SLA contracts.

Profiles are private scenario-generation labels, never policy observations.
Only already observed arrivals in a recent window enter the allocator state.
"""
from collections import Counter
import numpy as np
from gymnasium import spaces
from slice_alloc import allocation_final as base, modal_binding as b
from slice_alloc import guided_allocation as guided


def scenario(seed,load=80):
    cfg=base.scenario(seed,load)
    rng=np.random.default_rng(seed+180003)
    profile=str(rng.choice(('steady','upload_first','retrieval_first'),p=(.2,.5,.3)))
    cfg['private']['demand_profile']=profile
    if profile!='steady':
        for raw in cfg['requests']:
            duration=raw['stop']-raw['start']
            early=(raw['service']=='telemetry') if profile=='upload_first' else (raw['service']!='telemetry')
            raw['start']=float(rng.uniform(2.,6.) if early else rng.uniform(5.,18.))
            raw['stop']=raw['start']+duration
        cfg['requests'].sort(key=lambda r:r['start'])
    b.validate(cfg)
    return cfg


class MissionEnv(guided.GuidedEnv):
    def __init__(self,*args,**kwargs):
        self.admission_threshold=kwargs.pop('admission_threshold',None)
        kwargs.setdefault('shaping',.5)
        super().__init__(*args,**kwargs)
        original=self.observation_space.spaces
        self.observation_space=spaces.Dict({**original,'state':spaces.Box(-100,100,
            (original['state'].shape[0]+3,),np.float32)})

    def reset(self,seed=None,options=None):
        opts=dict(options or {})
        if 'scenario' not in opts:
            opts['scenario']=scenario(int(self.rng.integers(71000000,72000000)),int(self.rng.choice(self.loads)))
        return super().reset(seed=seed,options=opts)

    def encode(self):
        obs=super().encode();counts=Counter()
        if self.current['kind']!='done':
            now=self.current['time']
            for record in self.records:
                r=self.requests[record['request']]
                if now-3<=r.start<=now:counts[r.service]+=1
            if self.admission_threshold is not None and self.options:
                action=guided.rule_action('modal_greedy',self,self.rule_parameters)
                if b.exponential_price(self.ledger,self.options[action-1])>self.admission_threshold:
                    obs['prior'][1:1+len(self.options)]-=1.
                    obs['prior'][0]=0.
        return {**obs,'state':np.r_[obs['state'],np.array([counts[s]/30 for s in b.SERVICES],dtype=np.float32)].astype(np.float32)}

    def action_masks(self):
        return self.encode()['mask']>.5


def rule_action(name,env,parameters=(2.,.5,1.),threshold=None):
    if name!='price':return guided.rule_action(name,env,parameters)
    action=guided.rule_action('modal_greedy',env,parameters)
    if not action:return 0
    price=b.exponential_price(env.ledger,env.options[action-1])
    return action if threshold is None or price<=threshold else 0
