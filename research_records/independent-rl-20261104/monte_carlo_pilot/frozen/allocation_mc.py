"""Full-return replay for independent Double-DQN.

Each completed native episode stores the exact return-to-go at every causal
(state, action) transition. No future state is observed at action time and no
heuristic action is injected; this only replaces the short n-step target.
"""
from collections import deque
import copy
import numpy as np
from slice_alloc.distributional_allocation import NStepDictReplay


class MonteCarloDictReplay(NStepDictReplay):
    """Terminal Monte-Carlo returns with episode-safe buffering."""
    def __init__(self,*args,**kwargs):
        kwargs.pop('horizon',None);kwargs.pop('discount',None)
        super().__init__(*args,horizon=1,discount=1.,**kwargs)
        self.returns_committed=0

    def add(self,obs,next_obs,action,reward,done,infos):
        if any(info.get('TimeLimit.truncated',False) for info in infos):
            raise ValueError('Monte Carlo replay requires real terminal episodes')
        self.pending.append(copy.deepcopy((obs,next_obs,action,reward,done,infos)))
        if not bool(done[0]):return
        future=0.
        transitions=list(self.pending)
        self.pending.clear()
        for obs,next_obs,action,reward,done,infos in reversed(transitions):
            future=float(np.asarray(reward).reshape(-1)[0])+future
            super(NStepDictReplay,self).add(obs,next_obs,action,np.asarray([future],np.float32),np.asarray([True]),infos)
            self.returns_committed+=1
