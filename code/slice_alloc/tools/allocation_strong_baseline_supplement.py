"""Supplemental same-protocol comparison against stronger baselines.

Pure reporting: no tuning against the formal test seeds. modal_greedy reuses
the parameters already selected on a separate development set (seeds
221000-221009) in the archived strong-baseline study; price reuses the
existing exponential-price rule with no admission threshold. Neither rule's
parameters were chosen using these 447000-447059 test seeds or these results.
"""
import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil

import numpy as np

from slice_alloc import allocation_final as environment
from slice_alloc import modal_binding as b
from slice_alloc.allocation_credit import CreditEnv
from slice_alloc.tools import allocation_confirm_train as run
from slice_alloc.tools import allocation_simple_report as audit

OUT = run.OUT
MODAL_GREEDY_PARAMETERS = (2.0, 0.5, 0.0)  # beta, cache_penalty, pressure_weight; selected on dev seeds 221000-221009


def modal_greedy_costs(env, parameters):
    beta, cache_penalty, pressure_weight = parameters
    if not env.options:
        return np.empty(0, dtype=np.float32)
    request = env.requests[env.current['request_id']]
    speed = np.mean([np.linalg.norm(env.velocities[str(n)][:2]) for n in request.members])
    committed = sum((p.peak for p in env.ledger.plans), np.zeros_like(env.ledger.capacity))
    pressure = np.minimum(2., committed / env.ledger.capacity)
    answer = []
    for plan in env.options:
        normalized = plan.peak / env.ledger.capacity
        resource = float(np.sum(normalized * (1. + pressure_weight * pressure)))
        risk = float(plan.request.mode == 'ip' and request.service != 'telemetry')
        risk *= min(1., speed * (request.stop - request.start) / 120.)
        no_cache = request.service == 'object' and not (plan.request.mode == 'ndn' and plan.cache)
        answer.append(resource + beta * risk + cache_penalty * no_cache)
    return np.asarray(answer, dtype=np.float32)


def rule_action(method, env, parameters=MODAL_GREEDY_PARAMETERS, threshold=None):
    if not env.options:
        return 0
    if method in ('modal_greedy', 'modal_price_threshold'):
        action = int(np.argmin(modal_greedy_costs(env, parameters))) + 1
        if threshold is not None and b.exponential_price(env.ledger, env.options[action-1]) > threshold:
            return 0
        return action
    if method == 'exponential_price':
        return b.baseline('joint_price', env, threshold=math.inf)
    if method == 'greedy':
        return min(range(len(env.options)), key=lambda i:float(np.sum(env.options[i].peak/env.ledger.capacity))) + 1
    raise ValueError(method)


def action_digest(decisions):
    return hashlib.sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest()


def episode(seed, load, method, parameters=MODAL_GREEDY_PARAMETERS, threshold=None,
            output=None, trace=False, rule=None):
    cfg = environment.scenario(seed, load)
    env = CreditEnv(progress_weight=0.)
    output = Path(output) if output is not None else OUT
    prefix = output/'native'/f'{method}-0'/f'{load}-{seed}' if trace else None
    try:
        obs, _ = env.reset(options=dict(scenario=cfg, prefix=prefix, trace=trace))
        done = False
        reward = 0.
        while not done:
            action = rule_action(rule or method, env, parameters, threshold)
            assert obs['mask'][action] > .5
            obs, r, done, truncated, _ = env.step(action)
            assert not truncated
            reward += r
        result = env.result()
        assert reward == result['success']
        records = result['decisions']
        forced = sum(not r['decision']['accept'] and not r['feasible'] for r in records)
        voluntary = sum(not r['decision']['accept'] and bool(r['feasible']) for r in records)
        violations = result['admitted'] - result['success']
        assert forced + voluntary + violations + result['success'] == result['requests']
        row = dict(method=method, init=0, seed=seed, load=load,
                    action_sha256=action_digest(records),
                    requests=result['requests'], success=result['success'],
                    admitted=result['admitted'], by_service=result['by_service'])
        scenario_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        if not trace:
            return dict(**row, forced_reject=forced, voluntary_reject=voluntary,
                        admitted_violations=violations, modes=result['modes'], scenario_sha256=scenario_hash)
        result.update(scenario_sha256=scenario_hash, action_sha256=row['action_sha256'])
        run.write(str(prefix)+'-result.json',result)
        run.write(str(prefix)+'-metrics.json',dict(**row,scenario_sha256=scenario_hash))
        raw=Path(str(prefix)+'-events.csv')
        with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=3) as dst:
            shutil.copyfileobj(src,dst)
        raw.unlink()
        return audit.verify_episode(output,row)
    finally:
        env.close()


def task(spec):
    import torch
    torch.set_num_threads(1)
    seed,load,method=spec
    return episode(seed,load,method,trace=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();run.check()
    methods=('modal_greedy','exponential_price')
    specs=[(s,l,m) for m in methods for s in run.TEST for l in run.LOADS]
    with cf.ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        rows=[]
        for row in pool.map(task,specs):
            rows.append(row)
            if len(rows)%60==0:print('SUPPLEMENT_AUDITED',len(rows),len(specs),flush=True)
    run.write(OUT / 'strong_baseline_supplement.json', dict(
        caveat='modal_greedy parameters selected on separate dev seeds 221000-221009, not on this test set; reporting only, not used for primary selection',
        method_definitions={'exponential_price':'minimum exponential resource price, no admission threshold; not the archived modal-plus-admission-price rule'},
        source_sha256=run.sha(__file__),binary_sha256=run.sha(environment.NS3/'build/scratch/allocation-final-scenario'),
        primary_protocol_sha256=run.sha(OUT/'protocol.json'),
        audit=dict(native_episodes=len(rows),packet_sla=True,direction=True,reservation_replay=True),
        modal_greedy_parameters=list(MODAL_GREEDY_PARAMETERS), rows=rows))
    for method in methods:
        subset = [r for r in rows if r['method'] == method]
        rate = sum(r['success'] for r in subset) / sum(r['requests'] for r in subset)
        print(method, f'{rate:.4%}', f'{sum(r["requests"] for r in subset)} requests', flush=True)


if __name__ == '__main__':
    main()
