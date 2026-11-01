"""Isolated, reproducible RL revision; old published results remain immutable."""
import argparse
import concurrent.futures as cf
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from slice_alloc import modal_binding as b
from slice_alloc.modal_binding_rl import BindingRLEnv
from slice_alloc.distributional_allocation import MaskedDoubleQRDQN, CandidateQRPolicy
from slice_alloc.tools import modal_binding_run as old
from slice_alloc.tools.modal_binding_challengers import Rule

OUT = b.p.ROOT/'slice_alloc/results/modal_binding_rl'
INITS = (11, 23, 37, 53, 71)
ABLATION_INITS = (11, 23, 37)
VALIDATION = tuple(range(26000, 26006))
TEST = tuple(range(27000, 27030))
OOD = tuple(range(28000, 28015))
LOADS = (48, 80)
BUDGETS = (8192, 16384, 32768, 65536)
RULES = ('first_fit', 'fixed_price', 'joint_price', 'ndn_only', 'semantic_rule')
METHODS = ('full', 'fixed', 'no_progress', 'no_shaping')
HYPER = copy.deepcopy(old.HYPER)
HYPER.update(exploration_steps=16384,
    replay_buffer_kwargs=dict(horizon=32, discount=1.))
SOURCES = ('slice_alloc/modal_binding_rl.py', 'slice_alloc/execution/binding_rl_scenario.cc',
    'slice_alloc/tools/modal_binding_rl_run.py', 'slice_alloc/tests/test_modal_binding_rl.py',
    'slice_alloc/modal_binding.py', 'slice_alloc/patrol.py',
    'slice_alloc/distributional_allocation.py', 'slice_alloc/initial_allocation.py',
    'slice_alloc/tools/modal_binding_challengers.py', 'slice_alloc/tools/modal_binding_run.py')
write, read, sha = old.write, old.read, old.sha


def specs():
    return [(m, i) for m in METHODS for i in (INITS if m in ('full', 'fixed') else ABLATION_INITS)]


def env_for(method, **kwargs):
    return BindingRLEnv(enhanced=method != 'no_progress',
        shaping=0. if method == 'no_shaping' else .5, fixed=method == 'fixed', **kwargs)


def freeze():
    protocol = dict(version='observed-progress-rl-v2', sources={n: sha(b.p.ROOT/n) for n in SOURCES},
        binary_sha256=sha(b.p.NS3/'build/scratch/binding-rl-scenario'),
        train_seeds=[30000000, 31000000], validation=VALIDATION, test=TEST, ood=OOD,
        loads=LOADS, methods=specs(), budgets=BUDGETS, thresholds=old.THRESHOLDS,
        primary='native SLA successes / all arrivals, rejects are failures',
        selection='validation success only; ties earliest checkpoint; all initializations retained',
        deployment='full single initialization11 predetermined, no ensemble or rule fallback',
        physics='identical v1 contracts and packet code, only observed active counters added',
        observation='current/past positions and associations; admitted slice contracts; actual progress, never future scenarios',
        potential='Phi=.5*number of active admitted slices; terminal Phi=0; gamma=1',
        return_identity='sum(r_native + Phi(next)-Phi(current)) = native episode success count',
        training='masked Double QR-DQN,32quantiles,width96,nstep32,lr1e-4,replay20000,warmup1024,batch128,trainfreq4,target1000',
        ablations='fixed disallows modality choice; no_progress uses original observations; no_shaping retains native sparse rewards',
        M='measured wire bytes, airtime and normalized reservation seconds per success; secondary, not optimized directly',
        V='separate zero-update held-out mixes/load112/new object contract, not a claimed theoretical SMV optimum',
        exclusions='no future prediction, no runtime QoS reconfiguration, no demonstration, no expert fallback',
        comparison='fresh test seeds; development choices can use old test diagnosis, so old tests are not treated as unseen again')
    path = OUT/'protocol.json'
    if path.exists():
        assert read(path) == json.loads(json.dumps(protocol)), 'Frozen protocol changed'
    else:
        write(path, protocol)
        for name in SOURCES:
            target = OUT/'frozen'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(b.p.ROOT/name, target)
    return protocol


def episode(cfg, method, model=None, prefix=None, threshold=None):
    env = env_for(method)
    timings = []
    try:
        obs, _ = env.reset(options=dict(scenario=cfg, prefix=prefix, trace=prefix is not None))
        done = False
        total = native = 0.
        while not done:
            start = time.perf_counter()
            if model is not None:
                action = int(model.predict(obs, deterministic=True)[0])
            elif method in ('ndn_only', 'semantic_rule'):
                action = int(Rule(env, method, threshold).predict(obs)[0])
            else:
                action = b.baseline(method, env, math.inf if threshold is None else threshold)
            timings.append(time.perf_counter()-start)
            assert obs['mask'][action] > .5
            obs, r, done, _, info = env.step(action)
            total += r
            native += info['native_reward']
        result = env.result()
        assert total == native == result['success']
        result['decision_ms'] = 1000*float(np.mean(timings))
        result['scenario_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        if prefix:
            write(str(prefix)+'-result.json', result)
            raw = Path(str(prefix)+'-events.csv')
            with raw.open('rb') as src, gzip.open(str(raw)+'.gz', 'wb', compresslevel=3) as dst:
                shutil.copyfileobj(src, dst)
            raw.unlink()
            result.update(old.audit(prefix, result, cfg))
            write(str(prefix)+'-metrics.json', {k: v for k, v in result.items() if k not in ('decisions', 'settled', 'observations')})
        return result
    finally:
        env.close()


def score(model, method):
    rows = []
    for seed in VALIDATION:
        for load in LOADS:
            r = episode(b.scenario(seed, load), method, model)
            rows.append(dict(seed=seed, load=load, success=r['success'], requests=r['requests']))
    return dict(rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows), rows=rows)


class Checkpoints(BaseCallback):
    def __init__(self, folder, method):
        super().__init__()
        self.folder, self.method, self.records = folder, method, []

    def _on_step(self):
        if self.num_timesteps not in BUDGETS:
            return True
        path = self.folder/f'{self.num_timesteps}.zip'
        self.model.save(path)
        val = score(self.model, self.method)
        self.records.append(dict(step=self.num_timesteps, path=str(path.relative_to(b.p.ROOT)),
            sha256=sha(path), validation=val))
        write(self.folder/'progress.json', self.records)
        print('CHECKPOINT', self.method, self.folder.name, self.num_timesteps, round(val['rate'], 5), flush=True)
        return True


def train(spec):
    method, init = spec
    torch.set_num_threads(1)
    folder = OUT/'training'/method/str(init)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'training.json').exists():
        return read(folder/'training.json')
    env = env_for(method, seed=init+100, loads=LOADS)
    model = MaskedDoubleQRDQN(CandidateQRPolicy, env, **HYPER, seed=init, device='cpu', verbose=0)
    before = model.quantile_net.deployment[-1].weight.detach().clone()
    cb = Checkpoints(folder, method)
    started = time.perf_counter()
    try:
        model.learn(max(BUDGETS), callback=cb)
        change = float(torch.linalg.vector_norm(model.quantile_net.deployment[-1].weight.detach()-before))
        assert change > 0
        best = max(cb.records, key=lambda r: (r['validation']['rate'], -r['step']))
        selected = OUT/'models'/f'{method}-{init}.zip'
        selected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(b.p.ROOT/best['path'], selected)
        again = MaskedDoubleQRDQN.load(selected, device='cpu')
        assert score(again, method) == best['validation']
        result = dict(method=method, init=init, steps=model.num_timesteps, updates=model._n_updates,
            weight_change=change, seconds=time.perf_counter()-started, stages=cb.records, selected=best,
            model=str(selected.relative_to(b.p.ROOT)), model_sha256=sha(selected),
            training_seeds=env.episode_specs, protocol_sha256=sha(OUT/'protocol.json'))
        write(folder/'training.json', result)
        print('TRAINED', method, init, flush=True)
        return result
    finally:
        env.close()


def tune_rule(method):
    grid = []
    for threshold in (None,) if method == 'first_fit' else old.THRESHOLDS:
        results = [episode(b.scenario(s, l), method, threshold=threshold) for s in VALIDATION for l in LOADS]
        grid.append(dict(threshold=threshold,
            rate=sum(r['success'] for r in results)/sum(r['requests'] for r in results)))
    chosen = max(grid, key=lambda r: (r['rate'], math.inf if r['threshold'] is None else r['threshold']))
    value = dict(method=method, chosen=chosen, grid=grid)
    write(OUT/'rules'/f'{method}.json', value)
    print('RULE', method, chosen, flush=True)


def evaluate_group(spec):
    method, init, scope, mix, load, seeds = spec
    torch.set_num_threads(1)
    model = None if method in RULES else MaskedDoubleQRDQN.load(OUT/'models'/f'{method}-{init}.zip', device='cpu')
    threshold = read(OUT/'rules'/f'{method}.json')['chosen']['threshold'] if model is None else None
    rows = []
    for seed in seeds:
        prefix = OUT/'native'/f'{method}-{init}'/f'{scope}-{mix}-{load}-{seed}'
        path = Path(str(prefix)+'-metrics.json')
        r = read(path) if path.exists() else episode(b.scenario(seed, load, mix), method, model, prefix, threshold)
        row = {k: v for k, v in r.items() if k not in ('decisions', 'settled', 'observations')}
        row.update(method=method, init=init, scope=scope, mix=mix, load=load, seed=seed)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'train', 'evaluate'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze()
    if args.command == 'train':
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(train, s) for s in specs()] + [pool.submit(tune_rule, m) for m in RULES]
            for future in cf.as_completed(futures):
                future.result()
    elif args.command == 'evaluate':
        models = [read(OUT/'training'/m/str(i)/'training.json') for m, i in specs()]
        rules = [read(OUT/'rules'/f'{m}.json') for m in RULES]
        write(OUT/'selection.json', dict(models=models, rules=rules,
            selected_rule=max(rules, key=lambda r: r['chosen']['rate'])['method']))
        cases = [('test', 'balanced', l, TEST) for l in LOADS] + [
            ('ood', mix, l, OOD) for mix, l in [('object_heavy', 80), ('mobile_heavy', 80), ('balanced', 112), ('novel', 80)]]
        groups = [(m, i, *case) for m, i in specs()+[(r, 0) for r in RULES] for case in cases]
        rows = []
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for batch in pool.map(evaluate_group, groups):
                rows.extend(batch)
        write(OUT/'evaluation.json', rows)
        print('EVALUATED', len(rows), flush=True)


if __name__ == '__main__':
    main()
