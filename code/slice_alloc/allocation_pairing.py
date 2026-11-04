"""Paired native exploration and replay for the same scalar Double-DQN.

A training scene is replayed once with one different legal action. Both
trajectories are genuine native interactions. Deployment uses only the learned
Q network; neither future trajectories nor a heuristic choose its actions.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import numpy as np
from slice_alloc.allocation_credit import CreditEnv
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.distributional_allocation import NStepDictReplay


class PairedEnv(CreditEnv):
    def __init__(self,*args,**kwargs):
        self.pending_branch=None;self.branch=None;self.executed=[];self.pair_proofs=[]
        super().__init__(*args,**kwargs)

    def reset(self,seed=None,options=None):
        opts=dict(options or {})
        explicit='scenario' in opts
        self.branch=None
        if not explicit and self.pending_branch is not None:
            self.branch=self.pending_branch;self.pending_branch=None
            opts['scenario']=self.branch['scenario']
        self.collect_pairs=not explicit
        self.executed=[];self.state_keys=[]
        return super().reset(seed=seed,options=opts)

    def controlled_action(self,default):
        j=len(self.executed)
        if self.branch is None or j>self.branch['target']:return int(default)
        if j<self.branch['target']:return int(self.branch['actions'][j])
        mask=self.encode()['mask']>.5
        alternatives=np.flatnonzero(mask);baseline=self.branch['actions'][j]
        alternatives=alternatives[alternatives!=baseline]
        assert len(alternatives)
        key=observation_key({k:v[None] for k,v in self.encode().items()})
        assert key==self.branch['state_keys'][j], 'branch prefix did not reproduce the native state'
        self.pair_proofs.append(dict(seed=self.cfg['radio_seed']-1,request=self.current['request_id'],
            baseline_action=baseline,state_key=key))
        return int(self.rng.choice(alternatives))

    def step(self,action):
        self.state_keys.append(observation_key({k:v[None] for k,v in self.encode().items()}))
        self.executed.append(int(action))
        obs,r,done,truncated,info=super().step(action)
        if done and self.collect_pairs and self.branch is None:
            choices=[j for j,d in enumerate(self.records) if d['feasible']>=1]
            if choices:
                self.pending_branch=dict(scenario=deepcopy(self.cfg),target=int(self.rng.choice(choices)),
                    actions=list(self.executed),state_keys=list(self.state_keys))
        return obs,r,done,truncated,info


def observation_key(obs):
    h=hashlib.blake2b(digest_size=16)
    for name in sorted(obs):h.update(name.encode());h.update(np.asarray(obs[name],dtype=np.float32).tobytes())
    return h.hexdigest()


class PairedReplay(NStepDictReplay):
    """Same Double-Q targets, with half a minibatch from equal-state action pairs."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.state_actions=defaultdict(dict);self.slot_keys=[None]*self.buffer_size
        self.paired_keys=set();self.paired_batches=0

    def _commit(self):
        obs,_,action,*_=self.pending[0]
        key=observation_key(obs);act=int(np.asarray(action).reshape(-1)[0]);pos=self.pos
        previous=self.slot_keys[pos]
        if previous is not None:
            old_key,old_act=previous;entries=self.state_actions.get(old_key,{})
            if entries.get(old_act)==pos:entries.pop(old_act)
            if len(entries)<2:self.paired_keys.discard(old_key)
            if not entries:self.state_actions.pop(old_key,None)
        super()._commit()
        self.slot_keys[pos]=(key,act);self.state_actions[key][act]=pos
        if len(self.state_actions[key])>=2:self.paired_keys.add(key)

    def sample(self,batch_size,env=None):
        if len(self.paired_keys)<16:return super().sample(batch_size,env)
        keys=tuple(sorted(self.paired_keys));indices=[]
        for _ in range(batch_size//4):
            key=keys[np.random.randint(len(keys))]
            entries=self.state_actions[key];actions=np.random.choice(tuple(entries),2,replace=False)
            indices.extend(entries[int(a)] for a in actions)
        indices.extend(np.random.randint(0,self.buffer_size if self.full else self.pos,size=batch_size-len(indices)))
        np.random.shuffle(indices);self.paired_batches+=1
        return self._get_samples(np.asarray(indices),env)


class PairedDoubleDQN(MaskedDoubleDQN):
    def _setup_learn(self,*args,**kwargs):
        self.replay_buffer_class=type(self.replay_buffer)
        return super()._setup_learn(*args,**kwargs)

    def _sample_action(self,*args,**kwargs):
        actions,_=super()._sample_action(*args,**kwargs)
        actual=self.env.env_method('controlled_action',int(actions[0]))[0]
        actions=np.array([actual]);return actions,actions.copy()
