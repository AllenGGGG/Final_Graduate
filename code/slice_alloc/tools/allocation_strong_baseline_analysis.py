"""Compute paired bootstrap comparisons between rl and the supplemental strong
baselines, using the same scene/initialization resampling as the primary
analysis. Reporting only: these baselines were not part of the frozen primary
selection and did not influence it.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from slice_alloc.tools import allocation_confirm_train as run
from slice_alloc.tools import allocation_simple_report as audit

OUT = run.OUT


def audit_counts(out,rows):
    """Recount outcomes by request ID and hash the retained native evidence."""
    hashes={};counts=defaultdict(Counter)
    for row in rows:
        prefix=out/'native'/f"{row['method']}-{row['init']}"/f"{row['load']}-{row['seed']}"
        cfg=run.read(str(prefix)+'-scenario.json');result=run.read(str(prefix)+'-result.json')
        requests={r['id']:r for r in cfg['requests']}
        decisions={d['request']:d for d in result['decisions']}
        accepted={i for i,d in decisions.items() if d['decision']['accept']}
        settled={s['id']:s for s in result['settled']}
        successful={i for i,s in settled.items() if s['success']}
        assert len(requests)==len(cfg['requests'])==len(decisions)
        assert set(requests)==set(decisions) and set(settled)==accepted
        assert len(requests)==row['requests'] and len(accepted)==row['admitted'] and len(successful)==row['success']
        services={s:dict(requests=sum(r['service']==s for r in requests.values()),
                        success=sum(requests[i]['service']==s for i in successful))
                  for s in ('telemetry','command','object')}
        assert services==row['by_service']
        assert dict(Counter(decisions[i]['decision']['mode'] for i in accepted))==row['modes']
        assert hashlib.sha256(json.dumps(result['decisions'],sort_keys=True).encode()).hexdigest()==row['action_sha256']
        counts[row['method']].update(requests=len(requests),success=len(successful),admitted=len(accepted))
        for suffix in ('-scenario.json','-result.json','-metrics.json','-events.csv.gz'):
            p=Path(str(prefix)+suffix);hashes[str(p.relative_to(out))]=run.sha(p)
    run.write(out/'independent_count_and_trace_manifest.json',dict(verified_episodes=len(rows),
        by_request_id=True,all_business_counts_match=True,all_modes_match=True,all_action_digests_match=True,
        totals={k:dict(v) for k,v in counts.items()},trace_hashes=hashes,source_sha256=run.sha(__file__)))
    print('COUNTER_AND_TRACE_AUDIT',len(rows),str(out),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-traces',action='store_true')
    args=parser.parse_args()
    primary = json.loads((OUT / 'evaluation.json').read_text())
    supplement = json.loads((OUT / 'strong_baseline_supplement.json').read_text())['rows']
    rl_rows = [r for r in primary if r['method'] == 'rl']

    rng = np.random.default_rng(20261107)
    scenes = rng.integers(len(run.TEST), size=(20000, len(run.TEST)))
    inits = rng.integers(len(run.INITS), size=(20000, len(run.INITS)))

    methods=('modal_greedy','exponential_price')
    expected={(m,s,l) for m in methods for s in run.TEST for l in run.LOADS}
    actual={(r['method'],r['seed'],r['load']) for r in supplement}
    if len(supplement)!=len(actual) or actual!=expected:
        raise ValueError('Supplement has missing or duplicate scenarios')
    # Recreate the per-(init, seed) pooled-across-load rate matrix exactly as analyze() does.
    def rl_scene_rate(init, seed):
        rows = [r for r in rl_rows if r['init'] == init and r['seed'] == seed]
        return sum(r['success'] for r in rows) / sum(r['requests'] for r in rows)
    rl_matrix = np.array([[rl_scene_rate(i, s) for s in run.TEST] for i in run.INITS])
    rl_boot = rl_matrix[inits[:, :, None], scenes[:, None, :]].mean((1, 2))

    out = dict(interval_method=(
        '20000 paired bootstrap resamples of scenes (and initializations for rl); '
        'supplemental baselines are deterministic single-run rules, not part of the frozen primary selection'))
    for method in methods:
        rows = [r for r in supplement if r['method'] == method]
        summary = audit.summarize(rows)
        summary['violation_rate_admitted'] = summary['admitted_violations'] / summary['admitted'] if summary['admitted'] else 0.
        summary['minimum_service_success'] = min(s['success'] / s['requests'] for s in summary['by_service'].values())

        def scene_rate(seed):
            rr = [r for r in rows if r['seed'] == seed]
            return sum(r['success'] for r in rr) / sum(r['requests'] for r in rr)
        matrix = np.array([scene_rate(s) for s in run.TEST])
        boot = matrix[scenes].mean(1)
        summary['ci95'] = np.quantile(boot, [.025, .975]).tolist()
        out[method] = summary

        difference = (sum(r['success'] for r in rl_rows) / sum(r['requests'] for r in rl_rows)) - summary['success_rate']
        distribution = rl_boot - boot
        out[f'rl-{method}'] = dict(difference=difference, ci95=np.quantile(distribution, [.025, .975]).tolist(),
            centered_bootstrap_p=float((1 + np.count_nonzero(np.abs(distribution - difference) >= abs(difference))) / (len(distribution) + 1)))

    # Keep the original frozen analysis intact; report an explicitly expanded
    # exploratory family rather than silently replacing its adjusted p-values.
    original=run.read(OUT/'analysis.json')['comparisons']
    family={k:v['centered_bootstrap_p'] for k,v in original.items() if k!='rl-greedy'}
    family.update({k:v['centered_bootstrap_p'] for k,v in out.items() if k.startswith('rl-')})
    adjusted={};running=0.
    for rank,(name,p) in enumerate(sorted(family.items(),key=lambda kv:kv[1])):
        running=max(running,min(1.,(len(family)-rank)*p));adjusted[name]=running
    out['expanded_exploratory_holm']=adjusted
    out['multiplicity']='six secondary comparisons together; frozen original family and primary decision unchanged'
    for method in methods:out['rl-'+method]['holm_adjusted_p_expanded']=adjusted['rl-'+method]
    run.write(OUT/'strong_baseline_comparison.json',out)
    for method in methods:
        s = out[method]
        c = out[f'rl-{method}']
        print(method, f"rate={s['success_rate']:.4%}", f"ci95={s['ci95']}",
              f"rl_diff={100*c['difference']:+.3f}pp", f"ci95={[100*x for x in c['ci95']]}", flush=True)
    if args.audit_traces:
        audit_counts(OUT,supplement)
        greedy=Path(os.environ.get('LOWALT_GREEDY_OUTPUT',str(run.PKG/'results/greedy-improvement')))
        if (greedy/'verification.json').exists():audit_counts(greedy/'test',run.read(greedy/'test.json')['rows'])


if __name__ == '__main__':
    main()
