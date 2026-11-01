"""Preregistered resource sensitivities, separate from the primary test."""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from slice_alloc.tools import allocation_final_run as run

SEEDS=tuple(range(45000,45010))


def scenario(seed,case):
    cfg=run.a.scenario(seed,80 if case=='backhaul' else 160)
    if case=='backhaul':
        cfg['public']['backhaul_mbps']=.45
    elif case=='airtime':
        for key in ('backhaul_mbps','cross_mbps','access_mbps'):
            cfg['public'][key]*=8.
        rng=np.random.default_rng(seed+123456)
        for r in cfg['requests']:
            if rng.random()<.85:
                r.update(service='telemetry',members=[r['members'][0]],payload=1000,chunks=1,
                    reusable=False,period=float(rng.choice([.02,.04,.08])),deadline=.2,minimum_ratio=.95)
    else:
        raise ValueError(case)
    return cfg


def freeze():
    protocol=dict(seeds=SEEDS,cases=['backhaul','airtime'],initializations=run.INITS,
        methods='validation-selected full RL, resource price, semantic/mobile rule',
        backhaul='80 balanced requests, backhaul reduced from1Mbps to0.45Mbps, other parameters unchanged',
        airtime='160 requests; each existing request independently forced to telemetry with probability.85, otherwise retained; all wire rates multiplied8; same PHY and application contract choices',
        interpretation='sensitivity only; does not change primary scene or count as RL algorithm improvement',
        source_sha256=run.sha(__file__),main_protocol_sha256=run.sha(run.OUT/'protocol.json'))
    path=run.OUT/'stress_protocol.json'
    normalized=json.loads(json.dumps(protocol))
    if path.exists():
        assert run.read(path)==normalized
    else:
        assert not (run.OUT/'test_started.json').exists(), 'preregister before opening primary test'
        run.write(path,protocol)
    return protocol


def evaluate(spec):
    method,init,case=spec
    torch.set_num_threads(1)
    model=None if method in run.RULES else run.load_model(method,run.OUT/'models'/f'{method}-full-{init}.zip')
    threshold=run.read(run.OUT/'rules'/f'{method}.json')['chosen']['threshold'] if model is None else None
    rows=[]
    for seed in SEEDS:
        cfg=scenario(seed,case)
        prefix=run.OUT/'native'/f'{method}-full-{init}'/f'stress-{case}-{seed}'
        path=Path(str(prefix)+'-metrics.json')
        result=run.read(path) if path.exists() else run.episode(cfg,method,model,threshold=threshold,prefix=prefix)
        row={k:v for k,v in result.items() if k not in ('decisions','observations','settled')}
        row.update(method=method,variant='full',init=init,scope='stress_'+case,load=len(cfg['requests']),seed=seed,
            trace=str(prefix.relative_to(run.OUT)))
        rows.append(row)
    return rows


def main():
    import sys
    freeze()
    if len(sys.argv)>1 and sys.argv[1]=='freeze':
        print('STRESS PREREGISTERED',flush=True)
        return
    from slice_alloc.tools.allocation_final_pipeline import prepare_evaluation
    prepare_evaluation()
    chosen=run.read(run.OUT/'selection.json')['selected']
    specs=[(m,i,c) for c in ('backhaul','airtime') for m,i in [(chosen,j) for j in run.INITS]+[('price',0),('semantic',0)]]
    with cf.ProcessPoolExecutor(max_workers=6) as pool:
        batches=list(pool.map(evaluate,specs))
    run.write(run.OUT/'stress_evaluation.json',[r for batch in batches for r in batch])
    print('STRESS EVALUATED',sum(map(len,batches)),flush=True)


if __name__=='__main__':
    main()
