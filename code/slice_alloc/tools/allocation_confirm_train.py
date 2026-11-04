"""Frozen, five-initialization confirmation of native-return Monte-Carlo RL.

The two pilot initializations are imported without changing policy weights.
The other three use exactly the pilot interaction, optimizer and return target.
Pilot test observations are exploratory; the final test is a separate batch.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import FloatSchedule

from slice_alloc.allocation_credit import CreditEnv
from slice_alloc.allocation_mc import MonteCarloDictReplay
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc import allocation_final as environment

PKG = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('LOWALT_ALLOCATION_OUTPUT', str(PKG/'results/final'))).resolve()
PILOT = Path(os.environ.get('LOWALT_MC_PILOT', str(Path.home()/'.local/share/lowalt-experiments/independent-rl-development-20261104/monte_carlo_pilot'))).resolve()
INITS = (11, 23, 37, 53, 71)
LOADS = (32, 64, 96)
VALIDATION = tuple(range(442000, 442060))
TEST = tuple(range(447000, 447060))
BUDGETS = (8192, 16384, 32768)
PARAMETERS = dict(learning_rate=5e-5, buffer_size=50000, learning_starts=1024,
                  batch_size=128, train_freq=4, gradient_steps=1, gamma=1.,
                  target_update_interval=1000, exploration_initial_eps=.1,
                  exploration_final_eps=.02, exploration_steps=16384)
SOURCES = ('allocation_final.py', 'allocation_value.py', 'allocation_credit.py',
           'allocation_mc.py', 'distributional_allocation.py', 'modal_binding.py',
           'modal_binding_rl.py', 'patrol.py', 'initial_allocation.py',
           'patrol_contract.schema.json', 'tools/modal_binding_run.py',
           'execution/allocation_final_scenario.cc', 'tools/allocation_confirm_train.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def freeze():
    value = dict(title='面向低空智联的多模态网络切片技术', research='第一部分：初始智能分配',
                 sources={n:sha(PKG/n) for n in SOURCES},
                 binary_sha256=sha(environment.NS3/'build/scratch/allocation-final-scenario'),
                 initializations=list(INITS), loads=list(LOADS), validation=list(VALIDATION), test=list(TEST),
                 budgets=list(BUDGETS), actual_training=PARAMETERS,
                 reference_models={str(i):sha(OUT/'reference_models'/f'double_dqn-{i}.zip') for i in INITS},
                 reward='native settled SLA successes; each completed episode contributes exact return-to-go targets',
                 method='masked candidate action-value control, Monte-Carlo targets; pretrained Double-DQN encoder',
                 replay='complete real episodes only; terminal targets, no temporal-difference bootstrap in continuation',
                 prior='independent RL only; no expert actions, rule scores or behavior cloning',
                 checkpoint_selection='maximum validation success rate; tie earliest; all five initializations',
                 exploratory_pilot=dict(initializations=[11,23], tested_seeds=[445000,445059],
                                        status='exploratory; retained separately, never pooled with confirmation'),
                 training_seed_range=[81000000,82000000],
                 primary='RL versus Greedy, paired 95% interval lower bound positive',
                 controls=['nearest','first_fit','native','initial_rl'],
                 full_publication='all four rule differences have positive paired intervals; both violation rates no higher than Greedy',
                 secondary='report Holm-corrected exploratory p-values; do not equate unadjusted intervals with corrected significance',
                 bootstrap='20000 paired resamples of scenes and initializations; loads clustered within scene',
                 no_post_test_tuning=True)
    value=json.loads(json.dumps(value))
    p=OUT/'protocol.json'
    if p.exists():
        if read(p)!=value:raise AssertionError('Actual configuration or source changed after freeze')
    else:
        assert not (OUT/'test_started.json').exists()
        write(p,value)
        for n in SOURCES:
            dest=OUT/'frozen'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PKG/n,dest)
    return value


def check():
    p=read(OUT/'protocol.json')
    assert p['actual_training']==PARAMETERS
    for n,h in p['sources'].items():assert sha(PKG/n)==sha(OUT/'frozen'/n)==h,n
    assert sha(environment.NS3/'build/scratch/allocation-final-scenario')==p['binary_sha256']
    for i,h in p['reference_models'].items():assert sha(OUT/'reference_models'/f'double_dqn-{i}.zip')==h


def score(model):
    env=CreditEnv(progress_weight=0.)
    rows=[]
    try:
        for seed in VALIDATION:
            for load in LOADS:
                obs,_=env.reset(options={'scenario':environment.scenario(seed,load)})
                done=False;reward=0.
                while not done:
                    action=int(model.predict(obs,deterministic=True)[0]);assert obs['mask'][action]>.5
                    obs,r,done,truncated,_=env.step(action);assert not truncated;reward+=r
                result=env.result();assert reward==result['success']
                rows.append(dict(seed=seed,load=load,**{k:result[k] for k in ('requests','success','admitted','by_service')}))
        return dict(rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows),rows=rows)
    finally:env.close()


def save_record(model,folder,step,validation=None):
    path=folder/f'{step}.zip';model.save(path)
    return dict(step=step,path=str(path.relative_to(OUT)),sha256=sha(path),
                validation=score(model) if validation is None else validation)


class Checkpoints(BaseCallback):
    def __init__(self,folder,records):
        super().__init__();self.folder=folder;self.records=records

    def _on_step(self):
        if self.num_timesteps in BUDGETS:
            row=save_record(self.model,self.folder,self.num_timesteps);self.records.append(row)
            write(self.folder/'progress.json',self.records)
            print('CHECKPOINT',self.folder.name,self.num_timesteps,row['validation']['rate'],flush=True)
        return True


def train(init):
    torch.set_num_threads(1);check()
    assert not (OUT/'test_started.json').exists(),'Cannot train after final test opened'
    folder=OUT/'training'/str(init);folder.mkdir(parents=True,exist_ok=True)
    completion=folder/'training.json'
    if completion.exists():
        row=read(completion)
        assert row['protocol_sha256']==sha(OUT/'protocol.json')
        for c in row['checkpoints']:assert sha(OUT/c['path'])==c['sha256']
        assert sha(OUT/row['model'])==row['model_sha256']
        return row
    if init in (11,23):
        original=read(PILOT/'training'/str(init)/'training.json')
        records=[]
        for c in original['checkpoints']:
            source=PILOT/c['path'];assert sha(source)==c['sha256']
            model=MaskedDoubleDQN.load(source,device='cpu')
            policy={n:t.clone() for n,t in model.policy.state_dict().items()}
            # Fix serialized replay metadata only. This does not alter any
            # policy/optimizer tensor or the already completed pilot training.
            model.replay_buffer_class=MonteCarloDictReplay;model.replay_buffer_kwargs={}
            record=save_record(model,folder,c['step'],c['validation'])
            loaded=MaskedDoubleDQN.load(OUT/record['path'],device='cpu')
            for n,t in policy.items():torch.testing.assert_close(t,loaded.policy.state_dict()[n],atol=0,rtol=0)
            record['pilot_source_sha256']=c['sha256'];records.append(record)
        updates=original['updates'];committed=original['returns_committed'];seconds=0.
        origin='unchanged pilot policies and validation records; serialized replay metadata repaired before final freeze/test'
        seeds=None
    else:
        env=CreditEnv(seed=init+9000,loads=LOADS,progress_weight=0.)
        model=MaskedDoubleDQN.load(OUT/'reference_models'/f'double_dqn-{init}.zip',device='cpu');model.set_env(env)
        for name,value in PARAMETERS.items():
            if name!='train_freq':setattr(model,name,value)
        assert model.train_freq.frequency==PARAMETERS['train_freq']
        model.lr_schedule=FloatSchedule(PARAMETERS['learning_rate'])
        model.replay_buffer_class=MonteCarloDictReplay;model.replay_buffer_kwargs={}
        model.replay_buffer=MonteCarloDictReplay(PARAMETERS['buffer_size'],model.observation_space,model.action_space,device='cpu',n_envs=1)
        model.set_random_seed(init+9000);model._n_calls=0;model._n_updates=0
        cache=PILOT.parent/'credit/initial_validation'/f'{init}.json'
        records=[save_record(model,folder,0,read(cache) if cache.exists() else None)]
        started=time.perf_counter();before=model.quantile_net.deployment[-1].weight.detach().clone()
        try:
            model.learn(max(BUDGETS),callback=Checkpoints(folder,records),reset_num_timesteps=True)
            assert model._n_updates>0 and not torch.equal(before,model.quantile_net.deployment[-1].weight)
            assert isinstance(model.replay_buffer,MonteCarloDictReplay)
            committed=model.replay_buffer.returns_committed;updates=model._n_updates;seconds=time.perf_counter()-started
            seeds=list(env.episode_specs)
        finally:env.close()
        origin='new full-return continuation with exactly the pilot interaction schedule'
    best=max(records,key=lambda r:(r['validation']['rate'],-r['step']))
    dest=OUT/'models'/f'double_dqn-{init}.zip';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OUT/best['path'],dest)
    row=dict(init=init,steps=max(BUDGETS),updates=updates,seconds=seconds,origin=origin,
             returns_committed=committed,selected=best,checkpoints=records,
             model=str(dest.relative_to(OUT)),model_sha256=sha(dest),training_seeds=seeds,
             protocol_sha256=sha(OUT/'protocol.json'))
    write(completion,row);print('TRAINED',init,best['step'],best['validation']['rate'],flush=True)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'reference_models').exists():shutil.copytree(PILOT/'reference_models',OUT/'reference_models')
    freeze()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(train,INITS))
    write(OUT/'training_complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),models=rows))


if __name__=='__main__':main()
