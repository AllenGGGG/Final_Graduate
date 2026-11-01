"""Freeze and forward-evaluate the previous RL recipe as a strong learned control."""
import concurrent.futures as cf
from datetime import datetime
from pathlib import Path
import shutil

import torch

from slice_alloc.tools import modal_binding_rl_run as run
from slice_alloc.tools import modal_binding_run as old
from slice_alloc.tools.modal_binding_rl_registry import episode, load


def freeze():
    assert not (run.OUT/'selection.json').exists()
    diagnostic = run.read(run.OUT/'legacy_validation_diagnostic.json')
    for row in diagnostic['rows']:
        init = row['initialization']
        original = old.OUT/'training/joint_qrdqn'/str(init)/'training.json'
        record = run.read(original)
        source = run.b.p.ROOT/row['source_model']
        assert run.sha(source) == record['model_sha256'] == row['sha256']
        target = run.OUT/'models'/f'legacy-{init}.zip'
        shutil.copy2(source, target)
        record.update(method='legacy', reused=True, source_training_manifest=str(original.relative_to(run.b.p.ROOT)),
            source_model=str(source.relative_to(run.b.p.ROOT)), fresh_validation=row['validation'],
            model=str(target.relative_to(run.b.p.ROOT)), evaluation_protocol_sha256=run.sha(run.OUT/'protocol.json'))
        run.write(run.OUT/'training/legacy'/str(init)/'training.json', record)
    run.write(run.OUT/'legacy_protocol.json', dict(
        reason='Pre-test fresh validation shows the existing RL is stronger than several revisions; retain a strong learned reference and permit validation-only reuse',
        created_at=datetime.now().astimezone().isoformat(), source_sha256=run.sha(__file__),
        source_protocol_sha256=run.sha(old.OUT/'protocol.json'),
        binary_sha256=run.sha(run.b.p.NS3/'build/scratch/binding-scenario'),
        training='reuse five immutable v1 single QR-DQNs; 32768 steps each,91state,8step returns,no shaping; no new legacy training',
        selection='old checkpoints remain immutable; only recipe choice uses fresh validation; all five initializations retained',
        legacy_training_steps=163840))


def evaluate_group(spec):
    init, scope, mix, count, seeds = spec
    torch.set_num_threads(1)
    model = load('legacy', run.OUT/'models'/f'legacy-{init}.zip')
    rows = []
    for seed in seeds:
        prefix = run.OUT/'native'/f'legacy-{init}'/f'{scope}-{mix}-{count}-{seed}'
        path = Path(str(prefix)+'-metrics.json')
        result = run.read(path) if path.exists() else episode(run.b.scenario(seed, count, mix), 'legacy', model, prefix)
        row = {k: v for k, v in result.items() if k not in ('decisions', 'observations', 'settled')}
        row.update(method='legacy', init=init, scope=scope, mix=mix, load=count, seed=seed)
        rows.append(row)
    return rows


def evaluate():
    cases = [('test', 'balanced', l, run.TEST) for l in run.LOADS]+[
        ('ood', mix, l, run.OOD) for mix, l in [('object_heavy', 80), ('mobile_heavy', 80), ('balanced', 112), ('novel', 80)]]
    rows = [r for r in run.read(run.OUT/'evaluation.json') if r['method'] != 'legacy']
    with cf.ProcessPoolExecutor(max_workers=10) as pool:
        for batch in pool.map(evaluate_group, [(i, *case) for i in run.INITS for case in cases]):
            rows.extend(batch)
    run.write(run.OUT/'evaluation.json', rows)
    print('LEGACY EVALUATED', len(rows), flush=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        evaluate()
    else:
        freeze()
