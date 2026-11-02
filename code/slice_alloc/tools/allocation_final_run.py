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
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback

from slice_alloc import allocation_cmdp as a
from slice_alloc.allocation_final_policy import CandidatePPOPolicy
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.distributional_allocation import MaskedDoubleQRDQN, CandidateQRPolicy
from slice_alloc.tools import modal_binding_run as old

ROOT = a.b.p.ROOT
OUT = Path(os.environ.get('LOWALT_ALLOCATION_OUTPUT', str(ROOT/'slice_alloc/results/final'))).expanduser().resolve()
INITS = (11, 23, 37, 53, 71)
LOADS = (48, 80)
CHECKPOINTS = (8192,)
VALIDATION = tuple(range(72000, 72030))
TEST = tuple(range(73000, 73030))
CONTROL = tuple(range(74000, 74010))
ALGORITHMS = ('candidate_ppo',)  # method fixed from formulation, no noisy algorithm contest
ABLATIONS = ()
MAIN_RULES = ('first_fit', 'nearest', 'greedy', 'quota')
PRIMARY_RULE = 'greedy'
CONTROL_RULES = ('greedy', 'semantic', 'price')
RULES = ('first_fit', 'nearest', 'greedy', 'price', 'semantic', 'quota')
REGISTRATIONS = ('protocol.json', 'stress_protocol.json', 'auxiliary_sources.json',
    'deployment.json', 'tests_complete.json', 'selection.json', 'training_complete.json', 'diagnosis.json') + tuple(
    f'rules/{method}.json' for method in RULES)
SOURCES = ('__main__.py', 'ALLOCATION_FINAL.md', 'allocation_final.py', 'allocation_cmdp.py', 'allocation_final_policy.py', 'allocation_value.py', 'modal_binding.py',
           'modal_binding_rl.py', 'patrol.py', 'initial_allocation.py',
           'distributional_allocation.py', 'execution/allocation_final_scenario.cc', 'execution/allocation_cmdp_scenario.cc',
           'tools/allocation_final_run.py', 'tools/modal_binding_run.py',
           'tests/test_allocation_final.py', 'tests/test_allocation_final_pipeline.py',
           'tests/test_allocation_algorithms.py', 'tests/test_allocation_cmdp.py')
write, read, sha = old.write, old.read, old.sha
PPO_HYPER = dict(learning_rate=3e-4, n_steps=512, batch_size=128,
    n_epochs=10, gamma=1., gae_lambda=1., clip_range=.2, ent_coef=.01)
A2C_HYPER = dict(learning_rate=7e-4, n_steps=128, gamma=1., gae_lambda=1.,
    ent_coef=.01, vf_coef=.5, max_grad_norm=.5, normalize_advantage=True,
    use_rms_prop=True, rms_prop_eps=1e-5)
MODEL_CLASSES = dict(candidate_ppo=PPO, double_qrdqn=MaskedDoubleQRDQN,
                     double_dqn=MaskedDoubleDQN, candidate_a2c=A2C)
POLICIES = dict(candidate_ppo=a.HierarchicalPPOPolicy, double_qrdqn=CandidateQRPolicy,
                double_dqn=CandidateQRPolicy, candidate_a2c=CandidatePPOPolicy)


def training_parameters(algorithm):
    if algorithm == 'candidate_ppo':
        return copy.deepcopy(PPO_HYPER)
    if algorithm == 'double_qrdqn':
        return copy.deepcopy(old.HYPER)
    if algorithm == 'double_dqn':
        parameters = copy.deepcopy(old.HYPER)
        parameters['policy_kwargs']['n_quantiles'] = 1
        return parameters
    if algorithm == 'candidate_a2c':
        return copy.deepcopy(A2C_HYPER)
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
        binary_sha256=sha(a.p.NS3/'build/scratch/allocation-cmdp-scenario'),
        training=list(a.TRAIN_RANGE), validation=VALIDATION, test=TEST, control=CONTROL,
        algorithms=ALGORITHMS, initializations=INITS,
        loads=LOADS, checkpoints=CHECKPOINTS,
        selection='one preregistered hierarchical PPO method; checkpoints by validation utility minus 5 times positive constraint excess, tie earliest; ALL five initializations retained',
        ablations=ABLATIONS,
        ablation_protocol='retrain selected algorithm with identical budget and initializations',
        rule_thresholds=[1., 4., None],
        primary_rule=PRIMARY_RULE, main_rules=MAIN_RULES, control_rules=CONTROL_RULES,
        supplemental_rules=['price', 'semantic'],
        reward='(priority-weighted native SLA utility - adaptive Lagrange multipliers dot constraint costs)/3; gamma=1',
        weights=a.WEIGHTS, tiers=a.TIERS, overbook=a.OVERBOOK, risk_budget=a.RISK_BUDGET,
        violation_budgets=a.VIOLATION_BUDGET, dual_learning_rate=.2, dual_clip=[0.,20.],
        execution_model='per-slice per-endpoint native token-bucket ingress shaping; birth times unchanged; no hard per-hop bandwidth guarantee',
        objective='weighted successful contracts / all offered contract value; also report unweighted success and ALL service/risk constraints',
        scope='point 1 only; no demand prediction, proactive migration or hard QoS guarantee',
        propagation='LogDistance n=2.5 reference46.6777dB plus 120m finite footprint, TX20dBm, fixed24Mbps',
        directions='telemetry uplink IP; command downlink IP/ID; object retrieval IP/NDN',
        quotas='per-modality one third of each shared wire-direction and airtime; intrinsic state/cache quotas unchanged',
        ppo=serialize_parameters(training_parameters('candidate_ppo')),
        qrdqn=serialize_parameters(training_parameters('double_qrdqn')),
        dqn=serialize_parameters(training_parameters('double_dqn')),
        a2c=serialize_parameters(training_parameters('candidate_a2c')),
        policies={name:serialize_parameters(policy) for name,policy in POLICIES.items()},
        model_classes={name:serialize_parameters(model) for name,model in MODEL_CLASSES.items()},
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
    kwargs.setdefault('no_calendar', True)
    return a.AllocationEnv(fixed=variant == 'fixed', no_motion=variant == 'no_motion',
                           standard_only=variant=='standard', strict=variant=='strict',
                           quota=variant == 'quota', **kwargs)


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
    if name.startswith('tier_'):
        factor=float(name.split('_',1)[1])
        return next((i+1 for i in indices if abs(env.options[i].rate_factor-factor)<1e-8),0)
    if name == 'first_fit':
        return indices[0]+1
    adequate = [i for i in indices if a.shaping_adequate(r,getattr(env.options[i],'rate_factor',1.1))]
    if adequate:
        indices = adequate
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
    if model is not None:
        env.pending_multipliers = np.asarray(getattr(model,'constraint_multipliers',[1.]*4))
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
        assert abs(reward-result['reward_total']) < 1e-7
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
            rows.append(dict(seed=seed, load=load, success=r['success'], requests=r['requests'], utility=r['utility'],utility_available=r['utility_available'],
                             costs=r['cost_sum'],admitted=r['admitted'],risk_sum=r['risk_sum'],
                             by_service=r['by_service'],service_admitted=r['service_admitted']))
    rate=sum(r['utility'] for r in rows)/sum(r['utility_available'] for r in rows)
    excess=a.constraint_excess(rows)
    return dict(rate=rate,unweighted_rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows),
                constraint_excess=excess,selection_score=rate-5.*sum(excess), rows=rows)


def load_model(algorithm, path):
    return MODEL_CLASSES[algorithm].load(path, device='cpu')


class Checkpoints(BaseCallback):
    def __init__(self, folder, variant):
        super().__init__()
        self.folder, self.variant, self.records = folder, variant, []
        self.dual = np.ones(4)
        self.batch_infos = []
        self.dual_history = []

    def _on_step(self):
        self.batch_infos.extend(self.locals.get('infos',[]))
        self.model.constraint_multipliers = self.dual.tolist()
        if self.num_timesteps not in CHECKPOINTS:
            return True
        path = self.folder/f'{self.num_timesteps}.zip'
        self.model.save(path)
        validation = score(self.model, self.variant)
        self.records.append(dict(step=self.num_timesteps, path=str(path.relative_to(OUT)), sha256=sha(path), validation=validation))
        write(self.folder/'progress.json', self.records)
        print('CHECKPOINT', str(self.folder.relative_to(OUT)), self.num_timesteps, validation['rate'], flush=True)
        return True

    def _on_rollout_end(self):
        rows=self.batch_infos; self.batch_infos=[]
        if not rows: return
        totals=np.sum([r['costs'] for r in rows],axis=0)
        counts=np.r_[np.sum([r['settled_counts'] for r in rows],axis=0),len(rows)]
        gradient=totals/np.maximum(1.,counts)
        self.dual=np.clip(self.dual+.2*gradient,0.,20.)
        self.training_env.env_method('set_multipliers',self.dual.tolist())
        self.model.constraint_multipliers=self.dual.tolist()
        self.dual_history.append(dict(step=self.num_timesteps,multipliers=self.dual.tolist(),gradient=gradient.tolist()))


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
    model = MODEL_CLASSES[algorithm](POLICIES[algorithm], env, **training_parameters(algorithm),
                                    seed=init, device='cpu', verbose=0)
    before = torch.cat([p.detach().flatten() for p in model.policy.parameters()]).clone()
    cb = Checkpoints(folder, variant)
    started = time.perf_counter()
    try:
        model.learn(max(CHECKPOINTS), callback=cb)
        after = torch.cat([p.detach().flatten() for p in model.policy.parameters()])
        change = float(torch.linalg.vector_norm(after-before))
        assert change > 0.
        best = max(cb.records, key=lambda r:(r['validation']['selection_score'], -r['step']))
        selected = OUT/'models'/f'{algorithm}-{variant}-{init}.zip'
        selected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT/best['path'], selected)
        again = load_model(algorithm, selected)
        assert score(again, variant) == best['validation'], 'model reload must reproduce validation'
        result = dict(algorithm=algorithm, variant=variant, init=init, steps=model.num_timesteps,
            updates=model._n_updates, weight_change=change,dual_history=cb.dual_history, seconds=time.perf_counter()-started,
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
        grid.append(dict(threshold=threshold, rate=sum(r['utility'] for r in rows)/sum(r['utility_available'] for r in rows),
            excess=a.constraint_excess(rows)))
    chosen = max(grid, key=lambda r:(r['rate']-5.*sum(r['excess']), math.inf if r['threshold'] is None else r['threshold']))
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


def preregistration_snapshot():
    return {name:sha(OUT/name) for name in REGISTRATIONS}


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
        results = pool_map(train, [(m,'full',i) for m in ALGORITHMS for i in INITS], args.workers)
        values = {m:float(np.mean([r['selected']['validation']['rate'] for r in results if r['algorithm']==m]))
                  for m in ALGORITHMS}
        selected = max(values, key=values.get)
        selection = dict(selected=selected, validation=values, models=results, protocol_sha256=sha(OUT/'protocol.json'))
        if (OUT/'selection.json').exists():
            assert read(OUT/'selection.json') == selection
        else:
            write(OUT/'selection.json', selection)
        print('SELECTED', selected, values, flush=True)
        pool_map(train, [(selected,v,i) for v in ABLATIONS for i in INITS], args.workers)
        pool_map(tune_rule, RULES, args.workers)
        write(OUT/'training_complete.json', dict(protocol_sha256=sha(OUT/'protocol.json'), selection_sha256=sha(OUT/'selection.json')))
    if args.command in ('evaluate','all'):
        assert (OUT/'training_complete.json').exists()
        prepare_evaluation()
        selected = read(OUT/'selection.json')['selected']
        models = [(m,'full',i) for m in ALGORITHMS for i in INITS]
        models += [(selected,v,i) for v in ABLATIONS for i in INITS]
        methods = models+[(r,'full',0) for r in RULES]
        groups = [(m,v,i,'test',l,TEST,'mixed') for m,v,i in methods for l in LOADS]
        controls = [(selected,'full',i) for i in INITS]+[(r,'full',0) for r in CONTROL_RULES]
        groups += [(m,v,i,'mobility_'+mobi,80,CONTROL,mobi) for m,v,i in controls for mobi in ('stationary','local','mixed')]
        registered = preregistration_snapshot()
        marker = OUT/'test_started.json'
        if marker.exists():
            assert read(marker)['preregistration_sha256'] == registered, 'Preregistered inputs changed'
        else:
            write(marker, dict(time=time.time(), protocol_sha256=sha(OUT/'protocol.json'),
                selection_sha256=sha(OUT/'selection.json'), jobs=len(groups),
                preregistration_sha256=registered))
        batches = pool_map(evaluate_group, groups, args.workers)
        write(OUT/'evaluation.json', [r for batch in batches for r in batch])
        print('EVALUATED', sum(map(len,batches)), flush=True)


if __name__ == '__main__':
    main()
