"""Recheck frozen v4 evidence without overwriting its published artifacts.

Run: python -B -m slice_alloc.tools.lowalt_validation --output PATH
Motion zeroing is a posthoc sensitivity probe, not a new SLA evaluation.
"""
from __future__ import annotations

import argparse
import collections
import copy
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from slice_alloc import patrol
from slice_alloc.tools import lowalt_allocation as study
from slice_alloc.tools.patrol_run import sha, write


def fingerprints():
    delivery = json.loads((study.OUT / 'delivery.json').read_text())
    checked = []
    for group, base in (('sources', study.ROOT), ('artifacts', study.OUT),
                        ('manuscript_artifacts', study.ROOT),
                        ('previous_document_artifacts', study.ROOT)):
        for name, expected in delivery[group].items():
            assert sha(base / name) == expected, (group, name)
            checked.append(dict(group=group, path=name, sha256=expected))
    completion = json.loads((study.OUT / 'completion.json').read_text())
    assert completion['delivery_sha256'] == sha(study.OUT / 'delivery.json')
    return checked


def motion_probe(data, protocol):
    """Hold saved histories and each checkpoint fixed; zero only motion inputs."""
    torch.set_num_threads(1)
    all_results = []
    for init in protocol['initializations']:
        model = PPO.load(study.OUT / 'models' / f'proposed-{init}.zip', device='cpu')
        rows = [r for r in data['rows'] if r['method'] == 'proposed'
                and r['initialization'] == init]
        observations = []
        saved_actions = []
        count_plans = changed_plans = 0
        for row in rows:
            prefix = study.OUT / 'native' / row['path']
            cfg = json.loads(Path(str(prefix) + '-scenario.json').read_text())
            result = json.loads(Path(str(prefix) + '-result.json').read_text())
            env = study.LowaltEnv()
            env.cfg = cfg
            env.requests = {r['id']: patrol.request_from_json(r) for r in cfg['requests']}
            env.ledger = patrol.Ledger(cfg['public'])
            env.pressure = collections.Counter()
            env.peak_usage = np.zeros(len(patrol.RESOURCES))
            env.records = []
            try:
                for record, pair in zip(result['decisions'], result['observations']):
                    env.current = pair['observation']
                    obs = env.encode()
                    assert env.observation_space.contains(obs)
                    observations.append(obs)
                    request = env.requests[env.current['request_id']]
                    # A software dependency probe, not a simulated trajectory:
                    # retain associations, caches and public mission contracts.
                    moved = copy.deepcopy(env.current)
                    moved['positions'] = {n: [p[0] + 1, p[1], p[2]]
                                          for n, p in env.current['positions'].items()}
                    for option in env.options:
                        args = (request, option.edge, option.server, option.cache,
                                env.ledger.pub)
                        before = patrol.compile_plan(*args, env.current)
                        after = patrol.compile_plan(*args, moved)
                        count_plans += 1
                        changed_plans += not np.array_equal(before.peak, after.peak)
                    if request.mode == 'ndn':
                        env.seen_content[request.tenant, request.content, request.version] += 1
                    decision = record['decision']
                    if decision['accept']:
                        action, plan = next((i, p) for i, p in enumerate(env.options, 1)
                                            if p.decision() == decision)
                        saved_actions.append(action)
                        env.ledger.add(plan)
                    else:
                        saved_actions.append(0)
                    env.records.append(record)
            finally:
                env.close()
        tensors = {k: torch.as_tensor(np.stack([o[k] for o in observations]))
                   for k in observations[0]}
        hidden = {k: v.clone() for k, v in tensors.items()}
        hidden['state'][:, -10:] = 0
        with torch.no_grad():
            actual = model.policy.get_distribution(tensors).distribution.probs
            masked = model.policy.get_distribution(hidden).distribution.probs
            delta = (actual - masked).abs().max(1).values
            changed = int((actual.argmax(1) != masked.argmax(1)).sum())
            # Verify reconstruction reaches the same deterministic actions as
            # the original native rollout, rather than silently probing drift.
            assert actual.argmax(1).tolist() == saved_actions
        all_results.append(dict(initialization=init, episodes=len(rows),
            decisions=len(observations), changed_actions=changed,
            nonzero_motion_observations=int((tensors['state'][:, -10:] != 0).any(1).sum()),
            maximum_action_probability_change=float(delta.max()),
            mean_maximum_action_probability_change=float(delta.mean()),
            candidate_plans=count_plans, candidate_resource_changes=changed_plans,
            model_sha256=sha(study.OUT / 'models' / f'proposed-{init}.zip')))
        print(json.dumps(all_results[-1]), flush=True)
    return dict(scope='Posthoc input sensitivity over all saved proposed test episodes; '
                      'same models and histories; not a new SLA success evaluation',
                episodes=sum(r['episodes'] for r in all_results),
                decisions=sum(r['decisions'] for r in all_results),
                changed_actions=sum(r['changed_actions'] for r in all_results),
                candidate_resource_changes=sum(r['candidate_resource_changes'] for r in all_results),
                by_initialization=all_results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'validation.json').exists(), 'Use a fresh validation directory.'
    original_out = study.OUT
    protocol = json.loads((original_out / 'protocol.json').read_text())
    data = json.loads((original_out / 'evaluation.json').read_text())
    hashes = fingerprints()
    print(f'Fingerprints checked: {len(hashes)}', flush=True)

    rebuilt = study.analyze(data['rows'], protocol['methods'], protocol['test_seeds'])
    stored = json.loads((original_out / 'analysis.json').read_text())
    for key in rebuilt:
        assert rebuilt[key] == stored[key], key
    write(output / 'recomputed_analysis.json', rebuilt)
    print(f'Recomputed {rebuilt["episodes"]} episodes and {rebuilt["decisions"]} decisions.', flush=True)

    # The existing auditor writes its results under OUT. Point it at an isolated
    # directory with read-only inputs so published audit hashes stay unchanged.
    for name in ('native', 'protocol.json', 'evaluation.json', 'analysis.json'):
        target = output / name
        assert not target.exists(), target
        target.symlink_to(original_out / name)
    study.OUT = output
    try:
        study.verify(args.jobs)
    finally:
        study.OUT = original_out
    print('Raw packet audit and 20 native checkpoint replays passed.', flush=True)
    probe = motion_probe(data, protocol)
    write(output / 'motion_probe.json', probe)
    assert fingerprints() == hashes
    report = dict(passed=True, completed_at=datetime.datetime.now().astimezone().isoformat(),
        verifier_sha256=sha(__file__), frozen_artifacts_unchanged=True,
        checked_fingerprints=hashes, analysis_matches=True,
        episodes=rebuilt['episodes'], decisions=rebuilt['decisions'],
        primary_success=rebuilt['stats']['proposed']['success_rate'],
        motion_probe=probe, original_protocol_sha256=sha(original_out / 'protocol.json'),
        artifacts={name: sha(output / name) for name in
                   ('audit.json', 'raw_reconstruction.json', 'motion_probe.json',
                    'recomputed_analysis.json')})
    write(output / 'validation.json', report)
    print('VALIDATION COMPLETE', report['primary_success'], flush=True)


if __name__ == '__main__':
    main()
