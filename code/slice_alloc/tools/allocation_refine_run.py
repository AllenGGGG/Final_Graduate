"""One bounded continuation experiment; identical physics, actions and reward."""
import argparse
import concurrent.futures as cf
from pathlib import Path
import os
import shutil
import subprocess
import sys

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn

from slice_alloc.tools import allocation_simple_run as base
from slice_alloc.allocation_value import MaskedDoubleDQN

ORIGINAL = Path(os.environ.get('LOWALT_REFERENCE_OUTPUT', str(base.ROOT/'slice_alloc/results/simple-main-20261103'))).resolve()
OUT = Path(os.environ.get('LOWALT_REFINE_OUTPUT', str(base.ROOT/'slice_alloc/results/refine-main-20261103'))).resolve()
INITS = base.INITS
TEST = tuple(range(96000,96060))
CHECKPOINTS = (49152,65536)
LEARNING_RATE = 5e-5
REPLAY_WARMUP = 1024
EXTRA_SOURCES = ('tools/allocation_refine_run.py','tools/allocation_refine_report.py',
                 'tests/test_allocation_refine.py','REFINE_PROTOCOL.md')
write, read, sha = base.write, base.read, base.sha


def freeze():
    old = read(ORIGINAL/'protocol.json')
    sources = {**old['sources'],**{n:sha(base.ROOT/'slice_alloc'/n) for n in EXTRA_SOURCES}}
    assert not set(TEST) & (set(old['test']) | set(old['validation']))
    for name,digest in old['sources'].items():
        assert sha(base.ROOT/'slice_alloc'/name)==digest, name
    assert sha(base.b.p.NS3/'build/scratch/allocation-final-scenario')==old['binary_sha256']
    protocol = dict(reference_output=str(ORIGINAL), reference_protocol_sha256=sha(ORIGINAL/'protocol.json'),
        sources=sources, binary_sha256=old['binary_sha256'], initializations=list(INITS),
        loads=old['loads'], validation=old['validation'], test=list(TEST),
        methods=['double_dqn','reference_dqn','greedy','nearest','first_fit'],
        training_seed_range=old['training_seed_range'], checkpoints=list(CHECKPOINTS),
        learning_rate=LEARNING_RATE, replay_warmup=REPLAY_WARMUP,
        start_steps=32768, end_steps=65536, hyperparameters=old['hyperparameters'],
        changes='continue each terminal 32768-step checkpoint with learning_rate=5e-5; warm a fresh replay for 1024 transitions; all other hyperparameters and environment unchanged',
        selection='each initialization: max original validation-selected checkpoint and continuation checkpoints by same validation success; tie fewer training steps; never drop an initialization',
        primary='paired double_dqn minus greedy on fresh seeds; also paired double_dqn minus reference_dqn with identical seed and initialization; 20000 bootstrap draws; no test-driven retuning',
        reference_models={str(i):sha(ORIGINAL/f'models/double_dqn-{i}.zip') for i in INITS},
        start_models={str(i):sha(ORIGINAL/f'training/{i}/32768.zip') for i in INITS})
    path=OUT/'protocol.json'
    if path.exists():
        assert read(path)==protocol,'frozen refinement protocol changed'
    else:
        write(path,protocol)
        for name in sources:
            target=OUT/'frozen'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(base.ROOT/'slice_alloc'/name,target)
    return protocol


def prepare_continuation(model, env):
    """Rebuild replay safely before the first resumed gradient step."""
    model.set_env(env)
    model.learning_rate=LEARNING_RATE
    model.lr_schedule=get_schedule_fn(LEARNING_RATE)
    model.learning_starts=model.num_timesteps+REPLAY_WARMUP
    return model


class Checkpoints(BaseCallback):
    def __init__(self,folder):
        super().__init__()
        self.folder=folder
        self.records=[]

    def _on_step(self):
        if self.num_timesteps not in CHECKPOINTS:
            return True
        path=self.folder/f'{self.num_timesteps}.zip'
        self.model.save(path)
        validation=base.score(self.model)
        self.records.append(dict(step=self.num_timesteps,path=str(path.relative_to(OUT)),
            sha256=sha(path),validation=validation,origin='continuation'))
        write(self.folder/'progress.json',self.records)
        print('CHECKPOINT',self.folder.name,self.num_timesteps,validation['rate'],flush=True)
        return True


def train(init):
    torch.set_num_threads(1)
    freeze()
    folder=OUT/'training'/str(init)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training.json').exists():
        row=read(folder/'training.json')
        assert row['protocol_sha256']==sha(OUT/'protocol.json')
        assert sha(OUT/row['model'])==row['model_sha256']
        return row
    previous=read(ORIGINAL/f'training/{init}/training.json')
    reference=ORIGINAL/previous['model']
    saved_ref=OUT/f'models/reference_dqn-{init}.zip'
    saved_ref.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(reference,saved_ref)
    env=base.environment.AllocationEnv(seed=init+100100,loads=base.LOADS,no_calendar=True)
    model=MaskedDoubleDQN.load(ORIGINAL/f'training/{init}/32768.zip',device='cpu')
    assert model.num_timesteps==32768
    model.set_random_seed(init+1000)
    prepare_continuation(model,env)
    before=model.quantile_net.deployment[-1].weight.detach().clone()
    callback=Checkpoints(folder)
    try:
        model.learn(32768,callback=callback,reset_num_timesteps=False)
        assert model.num_timesteps==65536
        assert not torch.equal(before,model.quantile_net.deployment[-1].weight.detach())
        original_choice={**previous['selected'], 'path':str(saved_ref.relative_to(OUT)),
            'sha256':sha(saved_ref),'origin':'reference'}
        choices=[original_choice]+callback.records
        best=max(choices,key=lambda x:(x['validation']['rate'],-x['step']))
        selected=OUT/f'models/double_dqn-{init}.zip'
        shutil.copy2(OUT/best['path'],selected)
        again=MaskedDoubleDQN.load(selected,device='cpu')
        assert base.score(again)==best['validation'],'selected model reload changed validation'
        row=dict(init=init,steps=model.num_timesteps,updates=model._n_updates,
            training_seeds=env.episode_specs,selected=best,checkpoints=choices,
            model=str(selected.relative_to(OUT)),model_sha256=sha(selected),
            reference_model=str(saved_ref.relative_to(OUT)),reference_model_sha256=sha(saved_ref),
            protocol_sha256=sha(OUT/'protocol.json'))
        write(folder/'training.json',row)
        print('TRAINED',init,best['origin'],best['step'],best['validation']['rate'],flush=True)
        return row
    finally:
        env.close()


def evaluate_group(spec):
    method,init=spec
    torch.set_num_threads(1)
    model=None if method in ('greedy','nearest','first_fit') else MaskedDoubleDQN.load(
        OUT/f'models/{method}-{init}.zip',device='cpu')
    rows=[]
    for seed in TEST:
        for load in base.LOADS:
            prefix=OUT/'native'/f'{method}-{init}'/f'{load}-{seed}'
            metrics=Path(str(prefix)+'-metrics.json')
            result=read(metrics) if metrics.exists() else base.episode(base.scenario(seed,load),method,model,prefix)
            rows.append(dict(method=method,init=init,seed=seed,load=load,
                **{key:result[key] for key in ('requests','success','admitted','by_service','audited')}))
    return rows


def ensure_tests():
    marker=OUT/'tests_complete.json'
    if marker.exists():
        row=read(marker)
        assert row['protocol_sha256']==sha(OUT/'protocol.json')
        assert row['log_sha256']==sha(OUT/'full-tests.log')
        return
    with (OUT/'full-tests.log').open('w') as stream:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','slice_alloc/tests','-p','test_*.py'],
            cwd=base.ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    write(marker,dict(protocol_sha256=sha(OUT/'protocol.json'),log_sha256=sha(OUT/'full-tests.log')))


def prepare_evaluation():
    completed=read(OUT/'training_complete.json')
    assert completed['protocol_sha256']==sha(OUT/'protocol.json')
    assert {r['init'] for r in completed['models']}==set(INITS)
    for row in completed['models']:
        for key in ('model','reference_model'):
            assert sha(OUT/row[key])==row[key+'_sha256']
    snapshot={n:sha(OUT/n) for n in ('protocol.json','training_complete.json','tests_complete.json')}
    marker=OUT/'test_started.json'
    if marker.exists():
        assert read(marker)==snapshot
    else:
        write(marker,snapshot)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['freeze','train','evaluate','all'])
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    torch.set_num_threads(1)
    freeze()
    if args.command=='freeze':
        return
    ensure_tests()
    if args.command in ('train','all'):
        assert not (OUT/'test_started.json').exists(),'cannot retune after formal test opens'
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            models=list(pool.map(train,INITS))
        write(OUT/'training_complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),models=models))
    if args.command in ('evaluate','all'):
        prepare_evaluation()
        specs=[(m,i) for m in ('double_dqn','reference_dqn') for i in INITS]
        specs += [(m,0) for m in ('greedy','nearest','first_fit')]
        rows=[]
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for group in pool.map(evaluate_group,specs):
                rows.extend(group)
        write(OUT/'evaluation.json',rows)
        print('EVALUATED',len(rows),flush=True)


if __name__=='__main__':
    main()
