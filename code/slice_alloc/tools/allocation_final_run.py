"""Frozen train/validation/test pipeline for research point 1 only."""
import argparse
from collections import Counter
import concurrent.futures as cf
import copy
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from slice_alloc import allocation_final as a
from slice_alloc.allocation_final_policy import CandidatePPOPolicy
from slice_alloc.distributional_allocation import MaskedDoubleQRDQN, CandidateQRPolicy
from slice_alloc.tools import modal_binding_run as old

ROOT = a.b.p.ROOT
OUT = Path(os.environ.get('LOWALT_ALLOCATION_OUTPUT', str(ROOT/'slice_alloc/results/final'))).expanduser().resolve()
INITS = (11, 23, 37)
LOADS = (48, 80)
CHECKPOINTS = (8192, 16384, 32768)
VALIDATION = tuple(range(42000, 42008))
TEST = tuple(range(43000, 43030))
CONTROL = tuple(range(44000, 44010))
RULES = ('first_fit', 'nearest', 'greedy', 'price', 'semantic', 'quota')
SOURCES = ('__main__.py', 'ALLOCATION_FINAL.md', 'allocation_final.py', 'allocation_final_policy.py', 'modal_binding.py',
           'modal_binding_rl.py', 'patrol.py', 'initial_allocation.py',
           'distributional_allocation.py', 'execution/allocation_final_scenario.cc',
           'tools/allocation_final_run.py', 'tools/modal_binding_run.py',
           'tests/test_allocation_final.py', 'tests/test_allocation_final_pipeline.py')
write, read, sha = old.write, old.read, old.sha
PPO_HYPER = dict(learning_rate=3e-4, n_steps=512, batch_size=128,
    n_epochs=10, gamma=1., gae_lambda=1., clip_range=.2, ent_coef=.01)


def training_parameters(algorithm):
    if algorithm == 'candidate_ppo':
        return copy.deepcopy(PPO_HYPER)
    if algorithm == 'double_qrdqn':
        return copy.deepcopy(old.HYPER)
    raise ValueError(algorithm)


def serialize_parameters(value):
    if isinstance(value, dict):
        return {key: serialize_parameters(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_parameters(item) for item in value]
    if isinstance(value, type):
        return f'{value.__module__}.{value.__qualname__}'
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported frozen parameter: {type(value)}')


def freeze():
    protocol = dict(title='面向低空智联的多模态网络切片技术', research='智能分配',
        sources={n: sha(ROOT/'slice_alloc'/n) for n in SOURCES},
        binary_sha256=sha(a.NS3/'build/scratch/allocation-final-scenario'),
        training=[41000000, 42000000], validation=VALIDATION, test=TEST, control=CONTROL,
        algorithms=['candidate_ppo', 'double_qrdqn'], initializations=INITS,
        loads=LOADS, checkpoints=CHECKPOINTS,
        selection='mean validation success of ALL 3 initializations; ties PPO; checkpoints by validation, tie earliest',
        ablations=['fixed', 'no_motion', 'no_calendar'],
        ablation_protocol='retrain selected algorithm with identical budget and initializations',
        rule_thresholds=[1., 4., None],
        reward='raw actual SLA successes, gamma=1, no shaping or expert prior',
        objective='successful requests / all arrivals; rejected requests fail',
        scope='point 1 only; no demand prediction, proactive migration or hard QoS guarantee',
        propagation='LogDistance n=2.5 reference46.6777dB plus 120m finite footprint, TX20dBm, fixed24Mbps',
        directions='telemetry uplink IP; command downlink IP/ID; object retrieval IP/NDN',
        quotas='per-modality one third of each shared wire-direction and airtime; intrinsic state/cache quotas unchanged',
        ppo=serialize_parameters(training_parameters('candidate_ppo')),
        qrdqn=serialize_parameters(training_parameters('double_qrdqn')),
        policies=dict(candidate_ppo=serialize_parameters(CandidatePPOPolicy),
                      double_qrdqn=serialize_parameters(CandidateQRPolicy)),
        execution=dict(device='cpu', verbose=0))
    path = OUT/'protocol.json'
    normalized = json.loads(json.dumps(protocol))
    if path.exists():
        assert read(path) == normalized, 'Frozen sources/protocol changed'
    else:
        write(path, normalized)
        for name in SOURCES:
            target = OUT/'frozen'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/'slice_alloc'/name, target)
    return protocol


def environment(variant='full', **kwargs):
    return a.AllocationEnv(fixed=variant == 'fixed', no_motion=variant == 'no_motion',
                           no_calendar=variant == 'no_calendar', quota=variant == 'quota', **kwargs)


def rule_action(name, env, threshold=None):
    if not env.options:
        return 0
    r = env.requests[env.current['request_id']]
    indices = list(range(len(env.options)))
    if name in ('semantic', 'quota'):
        preferred = a.b.native_mode(r)
        if r.service == 'command' and all(env.cfg['public']['missions'][str(n)]['role'] != 'patrol' for n in r.members):
            preferred = 'ip'
        indices = [i for i in indices if env.options[i].request.mode == preferred]
        if r.reusable:
            leased = [i for i in indices if env.options[i].cache]
            if leased:
                indices = leased
    if not indices:
        return 0
    if name == 'first_fit':
        return indices[0]+1
    costs = {i: a.b.exponential_price(env.ledger, env.options[i]) for i in indices}
    if name == 'nearest':
        attached = [env.current['associations'][str(n)]-2 for n in r.members]
        best = min(indices, key=lambda i: (sum(env.options[i].edge != e for e in attached), costs[i]))
    elif name == 'greedy':
        best = min(indices, key=lambda i: float(np.sum(env.options[i].peak/env.ledger.capacity)))
    else:
        best = min(indices, key=costs.get)
    return best+1 if threshold is None or costs[best] <= threshold else 0


def episode(cfg, method, model=None, variant='full', threshold=None, prefix=None):
    env = environment('quota' if method == 'quota' else variant)
    try:
        obs, _ = env.reset(options=dict(scenario=cfg, prefix=prefix, trace=prefix is not None))
        done = False
        reward = 0.
        decisions = []
        while not done:
            started = time.perf_counter()
            action = int(model.predict(obs, deterministic=True)[0]) if model is not None else rule_action(method, env, threshold)
            decisions.append(time.perf_counter()-started)
            assert obs['mask'][action] > .5
            obs, r, done, _, _ = env.step(action)
            reward += r
        result = env.result()
        assert reward == result['success']
        result.update(decision_ms=1000*float(np.mean(decisions)),
            forced_reject=sum(not r['decision']['accept'] and not r['feasible'] for r in result['decisions']),
            voluntary_reject=sum(not r['decision']['accept'] and bool(r['feasible']) for r in result['decisions']),
            scenario_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())
        if prefix is not None:
            write(str(prefix)+'-result.json', result)
            raw = Path(str(prefix)+'-events.csv')
            with raw.open('rb') as src, gzip.open(str(raw)+'.gz', 'wb', compresslevel=3) as dst:
                shutil.copyfileobj(src, dst)
            raw.unlink()
            result.update(old.audit(prefix, result, cfg))
            result.update(audit_direction(prefix, result, cfg))
            write(str(prefix)+'-metrics.json', {k:v for k,v in result.items() if k not in ('decisions','observations','settled')})
        return result
    finally:
        env.close()


def audit_direction(prefix, result, cfg):
    requests = {r['id']:r for r in cfg['requests']}
    emitted = Counter()
    completed = Counter()
    associations = Counter()
    for row in csv.DictReader(gzip.open(str(prefix)+'-events.csv.gz', 'rt')):
        sid, node = int(row['slice']), int(row['node'])
        if row['event'] in ('associate', 'deassociate'):
            associations[row['event']] += 1
        if row['event'] == 'emit' and requests[sid]['service'] == 'telemetry':
            assert node in requests[sid]['members']
            emitted[sid] += 1
        if row['event'] == 'complete' and requests[sid]['service'] == 'telemetry':
            assert node == 0, 'telemetry must terminate at the ground platform'
            completed[sid] += 1
    for r in result['decisions']:
        if r['decision']['accept']:
            assert r['decision']['mode'] in a.COMPATIBLE[requests[r['request']]['service']]
    return dict(direction_audited=True, telemetry_generated=sum(emitted.values()),
                telemetry_delivered=sum(completed.values()), associations=dict(associations))


def score(model, variant='full'):
    rows = []
    for seed in VALIDATION:
        for load in LOADS:
            r = episode(a.scenario(seed, load), 'learned', model, variant)
            rows.append(dict(seed=seed, load=load, success=r['success'], requests=r['requests']))
    return dict(rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows), rows=rows)


def load_model(algorithm, path):
    return (PPO if algorithm == 'candidate_ppo' else MaskedDoubleQRDQN).load(path, device='cpu')


class Checkpoints(BaseCallback):
    def __init__(self, folder, variant):
        super().__init__()
        self.folder, self.variant, self.records = folder, variant, []

    def _on_step(self):
        if self.num_timesteps not in CHECKPOINTS:
            return True
        path = self.folder/f'{self.num_timesteps}.zip'
        self.model.save(path)
        validation = score(self.model, self.variant)
        self.records.append(dict(step=self.num_timesteps, path=str(path.relative_to(OUT)), sha256=sha(path), validation=validation))
        write(self.folder/'progress.json', self.records)
        print('CHECKPOINT', str(self.folder.relative_to(OUT)), self.num_timesteps, validation['rate'], flush=True)
        return True


def train(spec):
    freeze()
    algorithm, variant, init = spec
    torch.set_num_threads(1)
    folder = OUT/'training'/algorithm/variant/str(init)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'training.json').exists():
        cached = read(folder/'training.json')
        assert cached['protocol_sha256'] == sha(OUT/'protocol.json'), 'Cached training protocol mismatch'
        assert cached['model_sha256'] == sha(OUT/cached['model']), 'Cached model changed'
        return cached
    env = environment(variant, seed=init+100, loads=LOADS)
    if algorithm == 'candidate_ppo':
        model = PPO(CandidatePPOPolicy, env, **training_parameters(algorithm),
                    seed=init, device='cpu', verbose=0)
    else:
        model = MaskedDoubleQRDQN(CandidateQRPolicy, env, **training_parameters(algorithm), seed=init, device='cpu', verbose=0)
    before = torch.cat([p.detach().flatten() for p in model.policy.parameters()]).clone()
    cb = Checkpoints(folder, variant)
    started = time.perf_counter()
    try:
        model.learn(max(CHECKPOINTS), callback=cb)
        after = torch.cat([p.detach().flatten() for p in model.policy.parameters()])
        change = float(torch.linalg.vector_norm(after-before))
        assert change > 0.
        best = max(cb.records, key=lambda r:(r['validation']['rate'], -r['step']))
        selected = OUT/'models'/f'{algorithm}-{variant}-{init}.zip'
        selected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT/best['path'], selected)
        again = load_model(algorithm, selected)
        assert score(again, variant) == best['validation'], 'model reload must reproduce validation'
        result = dict(algorithm=algorithm, variant=variant, init=init, steps=model.num_timesteps,
            updates=model._n_updates, weight_change=change, seconds=time.perf_counter()-started,
            selected=best, checkpoints=cb.records, model=str(selected.relative_to(OUT)),
            model_sha256=sha(selected), training_seeds=env.episode_specs, protocol_sha256=sha(OUT/'protocol.json'))
        write(folder/'training.json', result)
        print('TRAINED', algorithm, variant, init, result['seconds'], flush=True)
        return result
    finally:
        env.close()


def tune_rule(method):
    grid = []
    for threshold in (None,) if method in ('first_fit','nearest','greedy') else (1.,4.,None):
        rows = [episode(a.scenario(s,l), method, threshold=threshold) for s in VALIDATION for l in LOADS]
        grid.append(dict(threshold=threshold, rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows)))
    chosen = max(grid, key=lambda r:(r['rate'], math.inf if r['threshold'] is None else r['threshold']))
    write(OUT/'rules'/f'{method}.json', dict(method=method, chosen=chosen, grid=grid))
    print('RULE', method, chosen, flush=True)


def evaluate_group(spec):
    method, variant, init, scope, load, seeds, mobility = spec
    torch.set_num_threads(1)
    model = None if method in RULES else load_model(method, OUT/'models'/f'{method}-{variant}-{init}.zip')
    threshold = read(OUT/'rules'/f'{method}.json')['chosen']['threshold'] if model is None else None
    rows = []
    for seed in seeds:
        prefix = OUT/'native'/f'{method}-{variant}-{init}'/f'{scope}-{load}-{seed}'
        metrics = Path(str(prefix)+'-metrics.json')
        result = read(metrics) if metrics.exists() else episode(a.scenario(seed, load, mobility=mobility), method,
            model, variant, threshold, prefix)
        row = {k:v for k,v in result.items() if k not in ('decisions','observations','settled')}
        row.update(method=method, variant=variant, init=init, scope=scope, load=load, seed=seed,
                   trace=str(prefix.relative_to(OUT)))
        rows.append(row)
    return rows


def pool_map(function, specs, workers):
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, specs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze','train','evaluate','all'])
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze()
    from slice_alloc.tools.allocation_final_pipeline import ensure_tests, prepare_evaluation
    if args.command in ('train', 'evaluate', 'all'):
        ensure_tests()
    if args.command in ('train','all'):
        results = pool_map(train, [(m,'full',i) for m in ('candidate_ppo','double_qrdqn') for i in INITS], args.workers)
        values = {m:float(np.mean([r['selected']['validation']['rate'] for r in results if r['algorithm']==m]))
                  for m in ('candidate_ppo','double_qrdqn')}
        selected = max(values, key=values.get)
        selection = dict(selected=selected, validation=values, models=results, protocol_sha256=sha(OUT/'protocol.json'))
        if (OUT/'selection.json').exists():
            assert read(OUT/'selection.json') == selection
        else:
            write(OUT/'selection.json', selection)
        print('SELECTED', selected, values, flush=True)
        pool_map(train, [(selected,v,i) for v in ('fixed','no_motion','no_calendar') for i in INITS], args.workers)
        pool_map(tune_rule, RULES, args.workers)
        write(OUT/'training_complete.json', dict(protocol_sha256=sha(OUT/'protocol.json'), selection_sha256=sha(OUT/'selection.json')))
    if args.command in ('evaluate','all'):
        assert (OUT/'training_complete.json').exists()
        prepare_evaluation()
        selected = read(OUT/'selection.json')['selected']
        models = [(m,'full',i) for m in ('candidate_ppo','double_qrdqn') for i in INITS]
        models += [(selected,v,i) for v in ('fixed','no_motion','no_calendar') for i in INITS]
        methods = models+[(r,'full',0) for r in RULES]
        groups = [(m,v,i,'test',l,TEST,'mixed') for m,v,i in methods for l in LOADS]
        controls = [(selected,'full',i) for i in INITS]+[('semantic','full',0),('price','full',0)]
        groups += [(m,v,i,'mobility_'+mobi,80,CONTROL,mobi) for m,v,i in controls for mobi in ('stationary','local','mixed')]
        write(OUT/'test_started.json', dict(time=time.time(), protocol_sha256=sha(OUT/'protocol.json'),
            selection_sha256=sha(OUT/'selection.json'), jobs=len(groups))) if not (OUT/'test_started.json').exists() else None
        batches = pool_map(evaluate_group, groups, args.workers)
        write(OUT/'evaluation.json', [r for batch in batches for r in batch])
        print('EVALUATED', sum(map(len,batches)), flush=True)


if __name__ == '__main__':
    main()
