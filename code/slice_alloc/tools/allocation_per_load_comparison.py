"""Per-load paired bootstrap comparisons (rl vs each rule), split by 32/64/96
requests. The pooled report.md comparison can hide load-dependent reversals;
this reports each load separately as its own scene-resampled interval.
"""
import json

import numpy as np

from slice_alloc.tools import allocation_confirm_train as run
from slice_alloc.tools import allocation_simple_report as audit

OUT = run.OUT
RULES = ('greedy', 'nearest', 'first_fit', 'native')


def main():
    rows = json.loads((OUT / 'evaluation.json').read_text())
    lookup = {(r['method'], r['init'], r['seed'], r['load']): r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate primary scenario')
    rng = np.random.default_rng(20261107)
    scenes = rng.integers(len(run.TEST), size=(20000, len(run.TEST)))
    inits = rng.integers(len(run.INITS), size=(20000, len(run.INITS)))

    out = {}
    for load in run.LOADS:
        out[str(load)] = {}
        rl_matrix = np.array([[lookup['rl', i, s, load]['success'] / lookup['rl', i, s, load]['requests']
                               for s in run.TEST] for i in run.INITS])
        rl_boot = rl_matrix[inits[:, :, None], scenes[:, None, :]].mean((1, 2))
        for method in RULES:
            rule_matrix = np.array([lookup[method, 0, s, load]['success'] / lookup[method, 0, s, load]['requests']
                                    for s in run.TEST])
            rule_boot = rule_matrix[scenes].mean(1)
            difference = rl_matrix.mean() - rule_matrix.mean()
            distribution = rl_boot - rule_boot
            out[str(load)][method] = dict(
                rl_rate=float(rl_matrix.mean()), rule_rate=float(rule_matrix.mean()),
                difference_pp=100 * difference,
                centered_bootstrap_p=float((1+np.count_nonzero(np.abs(distribution-difference)>=abs(difference)))/(len(distribution)+1)),
                ci95_pp=[100 * x for x in np.quantile(distribution, [.025, .975]).tolist()])
    family=[v for per_load in out.values() for v in per_load.values()];running=0.
    for rank,v in enumerate(sorted(family,key=lambda item:item['centered_bootstrap_p'])):
        running=max(running,min(1.,(len(family)-rank)*v['centered_bootstrap_p']))
        v['holm_adjusted_p_12_comparisons']=running
    run.write(OUT/'per_load_comparison.json',out)
    for load in run.LOADS:
        for method in RULES:
            v = out[str(load)][method]
            print(load, method, f"rl={v['rl_rate']:.4%}", f"rule={v['rule_rate']:.4%}",
                  f"diff={v['difference_pp']:+.3f}pp", f"ci95={v['ci95_pp']}", flush=True)


if __name__ == '__main__':
    main()
