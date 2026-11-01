"""Recompute ALL packet audits and replay each selected model independently."""
import argparse
import concurrent.futures as cf
import tempfile
from pathlib import Path

import torch

from slice_alloc import modal_binding as b
from slice_alloc.distributional_allocation import MaskedDoubleQRDQN
from slice_alloc.tools import modal_binding_run as old
from slice_alloc.tools import modal_binding_rl_run as run
from slice_alloc.tools.modal_binding_rl_registry import all_specs, load, episode


def prefix_for(row):
    return run.OUT/'native'/f"{row['method']}-{row['init']}"/f"{row['scope']}-{row['mix']}-{row['load']}-{row['seed']}"


def audit_row(row):
    prefix = prefix_for(row)
    result = run.read(str(prefix)+'-result.json')
    cfg = run.read(str(prefix)+'-scenario.json')
    assert result['requests'] == len(cfg['requests'])
    assert result['admitted'] == len(result['settled'])
    assert result['success'] == sum(s['success'] for s in result['settled'])
    assert {d['request'] for d in result['decisions'] if d['decision']['accept']} == {s['id'] for s in result['settled']}
    contracts = {r['id']: b.Service(**{**r, 'members': tuple(r['members'])}) for r in cfg['requests']}
    assert all(s['expected'] == contracts[s['id']].expected for s in result['settled'])
    independent = old.audit(prefix, result, cfg)
    assert result['scenario_sha256'] == row['scenario_sha256']
    assert b.scenario(row['seed'], row['load'], row['mix']) == cfg
    for key in ('success', 'requests', 'admitted', 'resource_time'):
        assert result[key] == row[key]
    for key, value in independent.items():
        assert row[key] == value
    # Reconstruct active progress from event history for a representative replay
    # in tests; this audit independently reconstructs every terminal SLA count.
    return (row['method'], row['init'], row['scope'], row['mix'], row['load'], row['seed'])


def replay(spec):
    torch.set_num_threads(1)
    method, init = spec
    path = run.OUT/'models'/f'{method}-{init}.zip'
    manifest = run.read(run.OUT/'training'/method/str(init)/'training.json')
    assert run.sha(path) == manifest['model_sha256']
    model = load(method, path)
    cfg = b.scenario(run.TEST[0], run.LOADS[0])
    original = run.read(str(prefix_for(dict(method=method, init=init, scope='test',
        mix='balanced', load=run.LOADS[0], seed=run.TEST[0])))+'-result.json')
    with tempfile.TemporaryDirectory(prefix='binding-rl-replay-') as folder:
        prefix = Path(folder)/'episode'
        result = episode(cfg, method, model, prefix)
        for key in ('success', 'admitted', 'decisions', 'settled', 'resource_time', 'observations'):
            assert original[key] == result[key], (method, init, key)
        import gzip
        old_path = str(prefix_for(dict(method=method, init=init, scope='test',
            mix='balanced', load=run.LOADS[0], seed=run.TEST[0])))+'-events.csv.gz'
        with gzip.open(old_path, 'rb') as a, gzip.open(str(prefix)+'-events.csv.gz', 'rb') as c:
            assert a.read() == c.read(), (method, init, 'packet bytes')
    return dict(method=method, init=init, model_sha256=run.sha(path), exact_packet_replay=True, evidence='full packet-event CSV equality; not raw-frame PCAP equality')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(1)
    protocol = run.freeze()
    ppo_protocol = run.read(run.OUT/'ppo_protocol.json')
    assert run.sha(run.b.p.ROOT/'slice_alloc/tools/modal_binding_rl_ppo.py') == ppo_protocol['source_sha256']
    legacy_protocol = run.read(run.OUT/'legacy_protocol.json')
    assert run.sha(run.b.p.ROOT/'slice_alloc/tools/modal_binding_rl_legacy.py') == legacy_protocol['source_sha256']
    assert run.sha(b.p.NS3/'build/scratch/binding-scenario') == legacy_protocol['binary_sha256']
    from slice_alloc.tools.modal_binding_rl_select import compute
    rl_choice = run.read(run.OUT/'rl_selection.json')
    computed = compute()
    assert {k: rl_choice[k] for k in computed} == computed
    from datetime import datetime
    assert datetime.fromisoformat(rl_choice['locked_at']).timestamp() <= (run.OUT/'selection.json').stat().st_mtime
    selection = run.read(run.OUT/'selection.json')
    assert selection['models'] == [run.read(run.OUT/'training'/m/str(i)/'training.json') for m, i in run.specs()]
    assert selection['rules'] == [run.read(run.OUT/'rules'/f'{m}.json') for m in run.RULES]
    assert selection['selected_rule'] == max(selection['rules'], key=lambda r: r['chosen']['rate'])['method']
    import math
    for rule in selection['rules']:
        assert rule['chosen'] == max(rule['grid'], key=lambda r: (r['rate'], math.inf if r['threshold'] is None else r['threshold']))
    rows = run.read(run.OUT/'evaluation.json')
    expected = (len(all_specs())+len(run.RULES))*(len(run.TEST)*len(run.LOADS)+len(run.OOD)*4)
    assert len(rows) == expected
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        keys = list(pool.map(audit_row, rows))
        replays = list(pool.map(replay, all_specs()))
    assert len(set(keys)) == expected
    for method, init in all_specs():
        record = run.read(run.OUT/'training'/method/str(init)/'training.json')
        assert record['steps'] == (32768 if method == 'legacy' else max(run.BUDGETS)) and record['weight_change'] > 0
        if method == 'legacy':
            assert record['reused']
            assert record['protocol_sha256'] == legacy_protocol['source_protocol_sha256']
            assert record['evaluation_protocol_sha256'] == run.sha(run.OUT/'protocol.json')
            assert all(20000000 <= s < 21000000 for s in record['training_seeds'])
            assert run.sha(b.p.ROOT/record['source_model']) == record['model_sha256']
        else:
            assert record['protocol_sha256'] == run.sha(run.OUT/'protocol.json')
            assert all(30000000 <= s < 31000000 for s in record['training_seeds'])
        if method == 'ppo':
            assert record['ppo_protocol_sha256'] == run.sha(run.OUT/'ppo_protocol.json')
        for stage in record['stages']:
            assert run.sha(b.p.ROOT/stage['path']) == stage['sha256']
        best = max(record['stages'], key=lambda x: (x['validation']['rate'], -x['step']))
        assert best == record['selected']
    result = dict(passed=True, independently_reaudited_episodes=expected,
        model_replays=replays, trained_steps=sum(run.read(run.OUT/'training'/m/str(i)/'training.json')['steps'] for m, i in all_specs() if m != 'legacy'),
        reused_legacy_training_steps=163840,
        source_fingerprints=protocol['sources'], protocol_sha256=run.sha(run.OUT/'protocol.json'),
        evaluation_sha256=run.sha(run.OUT/'evaluation.json'))
    result['rl_selection_sha256'] = run.sha(run.OUT/'rl_selection.json')
    run.write(run.OUT/'verification.json', result)
    print({k: v for k, v in result.items() if k not in ('model_replays', 'source_fingerprints')})


if __name__ == '__main__':
    main()
