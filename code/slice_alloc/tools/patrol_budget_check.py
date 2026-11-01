"""Independent diagnostic: does a longer training budget close the DDQN/PPO gap?

Read-only with respect to the frozen primary result (protocol.json, training.json,
evaluation.json, analysis.json are never modified). Trains separate checkpoints
at a longer step budget and replays them on the SAME frozen test seeds/profiles,
so the only variable changed is train_steps. Not a claim that any other factor
is ruled out; this isolates one hypothesis at a time.
"""
from __future__ import annotations
import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
from slice_alloc.patrol import *
from slice_alloc.tools.patrol_run import (OUT, PROTOCOL_SPEC, BASELINES, scenario, episode,
                                          source_hashes, sha, write)

LONG_STEPS = 32768
# Exploration decays on the SHORT budget's schedule in both runs, so that the
# long run is the same process extended rather than a rescaled schedule.
REFERENCE_STEPS = 4096
BUDGET_OUT = OUT / 'budget_check'


def train_long(spec):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from slice_alloc.src.qlearning import DoubleDQN
    from slice_alloc.src.policy import CandidatePolicy
    kind, seed = spec
    env = PatrolEnv(seed=seed + 100, loads=(24, 40), profiles=PROFILES)
    t = time.perf_counter()
    out = BUDGET_OUT / 'models' / f'{kind}-{seed}'
    out.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'ddqn':
        model = DoubleDQN(STATE_DIM, CANDIDATE_DIM, seed)
        obs, _ = env.reset()
        for step in range(LONG_STEPS):
            # Anchored to REFERENCE_STEPS, not LONG_STEPS: with the old formula
            # epsilon was still ~0.76 at step 4095 in the long run versus 0.08
            # in the short run, which confounded budget with exploration and
            # made the two budgets non-comparable.
            epsilon = max(.08, .9 * (1 - step / (REFERENCE_STEPS * .8)))
            action = model.explore(obs, epsilon)
            nxt, reward, done, _, _ = env.step(action)
            model.remember(obs, action, reward, nxt, done)
            if len(model.replay) >= 128 and step % 4 == 0:
                model.update()
            obs = env.reset()[0] if done else nxt
        model.save(str(out) + '.pt')
        checkpoint = Path(str(out) + '.pt')
    else:
        model = PPO(CandidatePolicy, env, learning_rate=.0003, n_steps=256, batch_size=128,
                    n_epochs=5, gamma=1., seed=seed, device='cpu', verbose=0)
        model.learn(total_timesteps=LONG_STEPS)
        model.save(str(out) + '.zip')
        checkpoint = Path(str(out) + '.zip')
    env.close()
    info = dict(kind=kind, seed=seed, steps=LONG_STEPS, epsilon_reference_steps=REFERENCE_STEPS,
                seconds=time.perf_counter() - t, model_sha256=sha(checkpoint))
    write(str(out) + '.json', info)
    return info


def eval_long(spec):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from slice_alloc.src.qlearning import DoubleDQN
    # (method, init, seed): init selects the trained checkpoint (one of
    # PROTOCOL_SPEC['initializations']); seed selects the frozen test
    # scenario (one of PROTOCOL_SPEC['test_seeds']). These are independent
    # axes evaluated as a full cross product, mirroring patrol_run.eval_group.
    method, init, seed = spec
    model = DoubleDQN.load(BUDGET_OUT / 'models' / f'{method}-{init}.pt') if method == 'ddqn' \
        else PPO.load(BUDGET_OUT / 'models' / f'{method}-{init}.zip', device='cpu')
    rows = []
    for pindex, profile in enumerate(PROFILES):
        for load in PROTOCOL_SPEC['loads']:
            cfg = scenario(seed + pindex * 100, load, profile)
            result = episode(cfg, method, model, prefix=None, trace=False)
            rows.append(dict(method=method, initialization=init, seed=seed, profile=profile,
                             load=load, requests=result['requests'], success=result['success'],
                             admitted=result['admitted']))
    return rows


def main():
    frozen = json.loads((OUT / 'protocol.json').read_text())
    assert frozen['sources'] == source_hashes(), 'sources changed after freeze; this check compares against a stale baseline'
    primary = json.loads((OUT / 'analysis.json').read_text())

    specs = [(k, s) for k in ('ddqn', 'ppo') for s in PROTOCOL_SPEC['initializations']]
    # Skip checkpoints already trained successfully in a prior (partially
    # failed) run; re-verify each against its own recorded info/checkpoint
    # hash rather than trusting the file's mere presence.
    pending = []
    infos = []
    for kind, seed in specs:
        info_path = BUDGET_OUT / 'models' / f'{kind}-{seed}.json'
        checkpoint = BUDGET_OUT / 'models' / f"{kind}-{seed}{'.pt' if kind == 'ddqn' else '.zip'}"
        if info_path.exists() and checkpoint.exists():
            info = json.loads(info_path.read_text())
            # epsilon_reference_steps must match too: checkpoints trained under
            # the old LONG_STEPS-scaled schedule are not valid for this run.
            if (info.get('model_sha256') == sha(checkpoint) and info.get('steps') == LONG_STEPS
                    and info.get('epsilon_reference_steps') == REFERENCE_STEPS):
                infos.append(info)
                continue
        pending.append((kind, seed))
    print(f'reusing {len(infos)} already-trained checkpoints; training {len(pending)} pending', flush=True)
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(train_long, spec) for spec in pending]
            for future in concurrent.futures.as_completed(futures):
                infos.append(future.result())
    write(BUDGET_OUT / 'training.json', infos)

    eval_specs = [(k, i, s) for k in ('ddqn', 'ppo') for i in PROTOCOL_SPEC['initializations']
                  for s in PROTOCOL_SPEC['test_seeds']]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(eval_long, spec) for spec in eval_specs]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())
    write(BUDGET_OUT / 'evaluation.json', rows)

    comparison = {}
    for m in ('ddqn', 'ppo'):
        short = primary['stats'][m]['success_rate']
        short_inits = primary['stats'][m].get('initialization_rates', [])
        long_rows = [r for r in rows if r['method'] == m]
        long_rate = sum(r['success'] for r in long_rows) / sum(r['requests'] for r in long_rows)
        long_inits = []
        for init in PROTOCOL_SPEC['initializations']:
            init_rows = [r for r in long_rows if r['initialization'] == init]
            long_inits.append(sum(r['success'] for r in init_rows) / sum(r['requests'] for r in init_rows))
        comparison[m] = dict(short_budget_steps=PROTOCOL_SPEC['train_steps'],
                              short_budget_success_rate=short,
                              short_budget_initialization_rates=short_inits,
                              long_budget_steps=LONG_STEPS,
                              long_budget_success_rate=long_rate,
                              long_budget_initialization_rates=long_inits,
                              delta=long_rate - short)
    best_rule = max(BASELINES, key=lambda b: primary['stats'][b]['success_rate'])
    comparison['strongest_rule'] = dict(name=best_rule,
                                         success_rate=primary['stats'][best_rule]['success_rate'])
    comparison['scope'] = ('isolates train_steps only; same frozen environment, test seeds, '
                           'profiles and loads as the primary 4096-step result; does not rule out '
                           'architecture/reward/exploration as contributing factors')
    write(BUDGET_OUT / 'comparison.json', comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
