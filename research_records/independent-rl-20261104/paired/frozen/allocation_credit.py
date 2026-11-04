"""Native SLA credit assignment for the existing independent Double-DQN.

No policy prior or expert action is used. A causal potential provides packet
progress feedback, while complete-episode return remains native SLA successes.
"""
import numpy as np
from slice_alloc import allocation_final as base


class CreditEnv(base.AllocationEnv):
    def __init__(self,*args,progress_weight=0.,**kwargs):
        self.progress_weight=float(progress_weight)
        kwargs.update(no_calendar=True,shaping=0.)
        super().__init__(*args,**kwargs)

    def reset(self,seed=None,options=None):
        opts=dict(options or {})
        if 'scenario' not in opts:
            if seed is not None:self.rng=np.random.default_rng(seed)
            opts['scenario']=base.scenario(int(self.rng.integers(81000000,82000000)),int(self.rng.choice(self.loads)))
        return super().reset(seed=seed,options=opts)

    def potential(self):
        if self.current['kind']=='done':return 0.
        return self.progress_weight*sum(x['on_time']/max(1,self.requests[x['id']].expected)
                                        for x in self.current['active'])


def native_action(env):
    if not env.options:return 0
    r=env.requests[env.current['request_id']]
    mode='ip' if r.service=='telemetry' else 'id' if r.service=='command' else 'ndn'
    available=[i for i,p in enumerate(env.options) if p.request.mode==mode]
    if not available:return 0
    return min(available,key=lambda i:float(np.sum(env.options[i].peak/env.ledger.capacity)))+1
