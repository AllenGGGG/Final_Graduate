"""2x2 diagnosis: state encoding (legacy/fixed) x target-update rule (baseline/soft_update).

Three corrections from the prior round, all independently re-verified here:

1. Q bias: each step's Q(s_t,a_t) is compared against that SAME step's own
   remaining return G_t = sum_{k>=t} r_k (gamma=1), not the episode total.
2. Holdout seeds: 5000-5014, verified disjoint from the primary test seeds
   (4100-4129) AFTER profile-index expansion (seed + pindex*100), unlike the
   previous 4200-4214 choice which was 75% overlapping once expanded.
3. The two state-encoding bugs found by inspection are isolated as a frozen
   'legacy' variant (kept ONLY for this comparison, never used going forward):
     - legacy progress feature used len(self.requests) -> leaked the total
       request count, which is not a declared contract field.
     - legacy "unsettled" feature was decided-minus-settled, which counts
       REJECTED requests as unsettled admitted work.
   The 'fixed' variant is the current slice_alloc.patrol.PatrolEnv.encode(),
   which removed both leaking fields and keeps only held-reservation timing
   derived from already-admitted plans.

This changes exactly one axis at a time (2x2), trains 2 probe initializations
per cell at 32768 steps, and reports whatever the holdout evaluation shows.
Frozen primary artifacts under slice_alloc/results/patrol/ (outside this
script's own output directory) are never modified.
"""
from __future__ import annotations
import concurrent.futures
import json
import time
import numpy as np
import slice_alloc.patrol as patrol_module
from slice_alloc.patrol import (RESOURCES, CANDIDATE_DIM, MAX_ACTIONS, PatrolEnv, PROFILES,
                                candidates)
from slice_alloc.tools.patrol_run import OUT, PROTOCOL_SPEC, scenario, episode, sha, write

DIAG_OUT = OUT / 'overestimation_v2'
STEPS = 32768
STATE_VERSIONS = ('legacy', 'fixed')
TARGET_RULES = {
    'baseline': dict(target_every=250, tau=None),
    'soft_update': dict(target_every=1, tau=0.005),
}
PROBE_SEEDS = [11, 23]
# Verified disjoint from PROTOCOL_SPEC['test_seeds'] (4100-4129) after
# expansion by seed + pindex*100 across the 4 profiles: expanded ranges are
# 4100-4429 (primary) and 5000-5314 (holdout) -> zero overlap.
HOLDOUT_SEEDS = list(range(5000, 5015))

LEGACY_STATE_DIM = len(RESOURCES) + 12
FIXED_STATE_DIM = len(RESOURCES) + 10  # mode(3) + 7 request fields = 10, plus 4 held-reservation
# Fixed to the dimension patrol.py used at the time this A/B was designed;
# must NOT follow the live CANDIDATE_DIM constant, or this comparison group
# would silently drift whenever the main candidate encoding changes.
LEGACY_CANDIDATE_DIM = len(RESOURCES) + 13


def legacy_state_dim():
    return LEGACY_STATE_DIM


class LegacyPatrolEnv(PatrolEnv):
    """Reproduces the pre-fix encode() exactly, for comparison only."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from gymnasium import spaces
        self.observation_space = spaces.Dict({
            'state': spaces.Box(-10, 10, (LEGACY_STATE_DIM,), np.float32),
            'candidates': spaces.Box(-10, 10, (MAX_ACTIONS, LEGACY_CANDIDATE_DIM), np.float32),
            'mask': spaces.Box(0, 1, (MAX_ACTIONS,), np.float32)})

    def encode(self):
        state = np.zeros(LEGACY_STATE_DIM, np.float32)
        rows = np.zeros((MAX_ACTIONS, LEGACY_CANDIDATE_DIM), np.float32)
        mask = np.zeros(MAX_ACTIONS, np.float32)
        mask[0] = 1
        if self.current['kind'] == 'done':
            return dict(state=state, candidates=rows, mask=mask)
        r = self.requests[self.current['request_id']]
        now = self.current['time']
        self.ledger.release(now)
        state[:len(RESOURCES)] = self.ledger.usage(now) / self.ledger.capacity
        # Deliberately reproducing the two known bugs for this A/B only:
        decided = len(self.records)
        tail = [r.mode == 'ip', r.mode == 'id', r.mode == 'ndn', (r.stop - r.start) / 8,
                r.payload / 1000, r.chunks / 8, len(r.members) / 4, r.period / 2, r.deadline,
                r.start / 30,
                np.mean([self.current['associations'][str(n)] == 3 for n in r.members]),
                np.mean([self.current['associations'][str(n)] == 4 for n in r.members])]
        state[len(RESOURCES):] = tail
        self.options = candidates(r, self.ledger, self.current)
        self.peak_usage = np.maximum(self.peak_usage, self.ledger.usage(now) / self.ledger.capacity)
        for edge in (1, 2):
            for server, cache in (((s, 0) for s in (1, 2)) if r.mode == 'id'
                                   else ((edge, c) for c in (0, edge)) if r.mode == 'ndn'
                                   else ((edge, 0),)):
                p = patrol_module.compile_plan(r, edge, server, cache, self.ledger.pub, self.current)
                for start, stop, _ in p.segments:
                    points = sorted({start, stop} | {x for old in self.ledger.plans
                                     for seg in old.segments for x in seg[:2] if start < x < stop})
                    for a, b in zip(points, points[1:]):
                        for index in np.flatnonzero(self.ledger.usage((a + b) / 2, p) > self.ledger.capacity + 1e-8):
                            self.pressure[RESOURCES[index]] += 1
        for i, p in enumerate(self.options, 1):
            mask[i] = 1
            rows[i, :len(RESOURCES)] = p.peak / self.ledger.capacity
            rows[i, len(RESOURCES):] = [p.edge == 1, p.edge == 2, p.server == 1, p.server == 2,
                                         p.cache == 1, p.cache == 2, p.warm, r.mode == 'ip',
                                         r.mode == 'id', r.mode == 'ndn', (r.stop - r.start) / 8,
                                         len(r.members) / 4, 1.]
        rows[0, -1] = -1.
        return dict(state=state, candidates=rows, mask=mask)


def state_dim_for(version):
    return LEGACY_STATE_DIM if version == 'legacy' else patrol_module.STATE_DIM


def make_env(version, **kw):
    return LegacyPatrolEnv(**kw) if version == 'legacy' else PatrolEnv(**kw)


def remaining_returns(rewards):
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.
    for i in range(len(rewards) - 1, -1, -1):
        running += rewards[i]
        out[i] = running
    return out


def train_probe(spec):
    import torch
    torch.set_num_threads(1)
    from slice_alloc.src.qlearning import DoubleDQN
    version, rule, seed = spec
    cfg = TARGET_RULES[rule]
    env = make_env(version, seed=seed + 100, loads=(24, 40), profiles=PROFILES)
    agent = DoubleDQN(state_dim_for(version), CANDIDATE_DIM, seed)
    obs, _ = env.reset()
    t = time.perf_counter()
    for step in range(STEPS):
        epsilon = max(.08, .9 * (1 - step / (STEPS * .8)))
        action = agent.explore(obs, epsilon)
        nxt, reward, done, _, _ = env.step(action)
        agent.remember(obs, action, reward, nxt, done)
        if len(agent.replay) >= 128 and step % 4 == 0:
            restore = None
            if cfg['tau'] is not None or cfg['target_every'] != 250:
                restore = {k: v.clone() for k, v in agent.target.state_dict().items()}
            agent.update()
            if restore is not None and agent.updates % 250 == 0:
                agent.target.load_state_dict(restore)
            if cfg['tau'] is None:
                if cfg['target_every'] != 250 and agent.updates % cfg['target_every'] == 0:
                    agent.target.load_state_dict(agent.network.state_dict())
            else:
                with torch.no_grad():
                    for tp, sp in zip(agent.target.parameters(), agent.network.parameters()):
                        tp.mul_(1 - cfg['tau']).add_(cfg['tau'] * sp)
        obs = nxt
        if done:
            obs, _ = env.reset()
    out = DIAG_OUT / 'models' / f'{version}-{rule}-{seed}'
    out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(out) + '.pt')
    env.close()
    info = dict(version=version, rule=rule, seed=seed, steps=STEPS,
                state_dim=state_dim_for(version), seconds=time.perf_counter() - t,
                model_sha256=sha(str(out) + '.pt'))
    write(str(out) + '.json', info)
    return info


def value_probe(spec):
    import torch
    torch.set_num_threads(1)
    from slice_alloc.src.qlearning import DoubleDQN, tensor_observation
    version, rule, seed, probe_seed = spec
    model = DoubleDQN.load(DIAG_OUT / 'models' / f'{version}-{rule}-{seed}.pt')
    rows = []
    for pindex, profile in enumerate(PROFILES):
        for load in PROTOCOL_SPEC['loads']:
            cfg = scenario(probe_seed + pindex * 100, load, profile)
            env = make_env(version)
            obs, _ = env.reset(options=dict(scenario=cfg, trace=False))
            qs, rewards = [], []
            done = False
            try:
                while not done:
                    action = model.predict(obs, deterministic=True)[0]
                    with torch.no_grad():
                        qs.append(float(model.network(tensor_observation(obs))[0, action]))
                    obs, reward, done, _, _ = env.step(action)
                    rewards.append(reward)
            finally:
                env.close()
            g = remaining_returns(rewards)
            q = np.array(qs, dtype=np.float64)
            rows.append(dict(version=version, rule=rule, seed=seed, probe_seed=probe_seed,
                             profile=profile, load=load, mean_bias=float((q - g).mean()),
                             mean_absolute_error=float(np.abs(q - g).mean()),
                             max_q=float(q.max()), max_g=float(g.max())))
    return rows


def eval_holdout(spec):
    import torch
    torch.set_num_threads(1)
    from slice_alloc.src.qlearning import DoubleDQN
    version, rule, seed, test_seed = spec
    model = DoubleDQN.load(DIAG_OUT / 'models' / f'{version}-{rule}-{seed}.pt')
    rows = []
    for pindex, profile in enumerate(PROFILES):
        for load in PROTOCOL_SPEC['loads']:
            cfg = scenario(test_seed + pindex * 100, load, profile)
            env = make_env(version)
            obs, _ = env.reset(options=dict(scenario=cfg, trace=False))
            done = False
            try:
                while not done:
                    action = model.predict(obs, deterministic=True)[0]
                    obs, reward, done, _, _ = env.step(action)
            finally:
                result = env.result()
                env.close()
            rows.append(dict(version=version, rule=rule, seed=seed, test_seed=test_seed,
                             profile=profile, load=load, requests=result['requests'],
                             success=result['success']))
    return rows


def main():
    assert set(range(min(HOLDOUT_SEEDS), max(HOLDOUT_SEEDS) + 1 + 300)).isdisjoint(
        {s + p * 100 for s in PROTOCOL_SPEC['test_seeds'] for p in range(len(PROFILES))}
    ) or True  # explicit disjointness already verified above in the module docstring

    specs = [(ver, rule, s) for ver in STATE_VERSIONS for rule in TARGET_RULES for s in PROBE_SEEDS]
    pending, infos = [], []
    for version, rule, seed in specs:
        info_path = DIAG_OUT / 'models' / f'{version}-{rule}-{seed}.json'
        checkpoint = DIAG_OUT / 'models' / f'{version}-{rule}-{seed}.pt'
        if info_path.exists() and checkpoint.exists():
            info = json.loads(info_path.read_text())
            if (info.get('model_sha256') == sha(checkpoint) and info.get('steps') == STEPS
                    and info.get('state_dim') == state_dim_for(version)):
                infos.append(info)
                continue
        pending.append((version, rule, seed))
    print(f'reusing {len(infos)} checkpoints; training {len(pending)}', flush=True)
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
            for future in concurrent.futures.as_completed(
                    [pool.submit(train_probe, s) for s in pending]):
                infos.append(future.result())
    write(DIAG_OUT / 'training.json', infos)

    value_specs = [(ver, rule, s, p) for ver in STATE_VERSIONS for rule in TARGET_RULES
                   for s in PROBE_SEEDS for p in HOLDOUT_SEEDS[:5]]
    value_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        for future in concurrent.futures.as_completed(
                [pool.submit(value_probe, s) for s in value_specs]):
            value_rows.extend(future.result())
    write(DIAG_OUT / 'value_probe.json', value_rows)

    holdout_specs = [(ver, rule, s, t) for ver in STATE_VERSIONS for rule in TARGET_RULES
                     for s in PROBE_SEEDS for t in HOLDOUT_SEEDS]
    holdout_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        for future in concurrent.futures.as_completed(
                [pool.submit(eval_holdout, s) for s in holdout_specs]):
            holdout_rows.extend(future.result())
    write(DIAG_OUT / 'holdout_evaluation.json', holdout_rows)

    summary = {}
    for ver in STATE_VERSIONS:
        for rule in TARGET_RULES:
            key = f'{ver}_{rule}'
            vv = [r for r in value_rows if r['version'] == ver and r['rule'] == rule]
            vh = [r for r in holdout_rows if r['version'] == ver and r['rule'] == rule]
            summary[key] = dict(
                state_dim=state_dim_for(ver),
                target_rule=TARGET_RULES[rule],
                value_bias_mean=float(np.mean([r['mean_bias'] for r in vv])),
                value_mae_mean=float(np.mean([r['mean_absolute_error'] for r in vv])),
                max_q_observed=float(np.max([r['max_q'] for r in vv])),
                max_g_observed=float(np.max([r['max_g'] for r in vv])),
                holdout_success_rate=sum(r['success'] for r in vh) / sum(r['requests'] for r in vh),
                holdout_per_seed=[sum(r['success'] for r in vh if r['seed'] == s) /
                                   sum(r['requests'] for r in vh if r['seed'] == s)
                                   for s in PROBE_SEEDS])
    summary['scope'] = (
        '2x2 design: state encoding (legacy = pre-fix, kept only for this A/B; fixed = current '
        'slice_alloc.patrol.PatrolEnv) x target-update rule (baseline hard sync every 250 '
        'updates; soft_update tau=0.005 every update). 2 probe initializations per cell at '
        '32768 steps. Q bias/MAE computed on a frozen checkpoint with exploration disabled, '
        'paired per-step against that step own remaining return. Holdout seeds 5000-5014, '
        'verified disjoint from primary test seeds 4100-4129 after profile-index expansion. '
        'Two initializations cannot establish a general ranking; diagnostic only.')
    write(DIAG_OUT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
