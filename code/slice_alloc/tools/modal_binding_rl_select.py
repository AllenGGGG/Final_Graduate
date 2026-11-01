"""Validation-only RL method selection, governed by the pre-test addendum."""
from datetime import datetime
import numpy as np

from slice_alloc.tools import modal_binding_rl_run as run


def compute():
    contract = run.read(run.OUT/'rl_selection_addendum.json')
    common = contract['comparison_initializations']
    scores = {}
    for method in contract['rl_candidates']:
        records = [run.read(run.OUT/'training'/method/str(i)/'training.json') for i in common]
        values = [r['fresh_validation']['rate'] if method == 'legacy' else r['selected']['validation']['rate'] for r in records]
        scores[method] = dict(mean=float(np.mean(values)), individual=values, initializations=common)
    # max keeps first method when equal; explicit simplest-first tie priority.
    winner = max(contract['tie_order'], key=lambda m: scores[m]['mean'])
    from slice_alloc.tools.modal_binding_rl_registry import all_specs
    return dict(method=winner, initialization=contract['deployment_initialization'], scores=scores,
        addendum_sha256=run.sha(run.OUT/'rl_selection_addendum.json'),
        ppo_protocol_sha256=run.sha(run.OUT/'ppo_protocol.json'),
        legacy_protocol_sha256=run.sha(run.OUT/'legacy_protocol.json'),
        model_sha256={f'{m}-{i}': run.read(run.OUT/'training'/m/str(i)/'training.json')['model_sha256'] for m, i in all_specs()},
        contract='mean selected-checkpoint validation success, same common initializations; no test input')


def main():
    path = run.OUT/'rl_selection.json'
    value = compute()
    if path.exists():
        saved = run.read(path)
        assert {k: saved[k] for k in value} == value
    else:
        assert not (run.OUT/'selection.json').exists(), 'Select RL before test evaluation starts'
        value['locked_at'] = datetime.now().astimezone().isoformat()
        run.write(path, value)
    print(value)


if __name__ == '__main__':
    main()
