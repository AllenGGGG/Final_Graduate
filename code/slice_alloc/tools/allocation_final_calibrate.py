"""Development-only diagnostics; never tunes the frozen RL/test protocol."""
from collections import Counter
import concurrent.futures as cf
import csv
import math
from pathlib import Path
import tempfile

import numpy as np

from slice_alloc.tools import allocation_final_run as run


def pressure(spec):
    seed,load=spec
    env=run.environment()
    counters=Counter()
    try:
        obs,_=env.reset(options={'scenario':run.a.scenario(seed,load)})
        done=False
        while not done:
            if not env.options:
                counters['no_feasible_requests']+=1
            r=env.requests[env.current['request_id']]
            capacity=env.ledger.capacity
            try:
                env.ledger.capacity=np.full_like(capacity,math.inf)
                raw=env.get_candidates(r)
            finally:
                env.ledger.capacity=capacity
            if not raw:
                counters['disconnected_requests']+=1
            for plan in raw:
                counters['raw_candidates']+=1
                if env.ledger.feasible(plan):
                    continue
                counters['infeasible_candidates']+=1
                points=sorted({t for p in env.ledger.plans+[plan] for segment in p.segments for t in segment[:2]}|
                              {t for p in env.ledger.plans+[plan] if p.lease for t in p.lease[-2:]})
                exceeded=set()
                for start,end in zip(points,points[1:]):
                    if start<plan.request.drain and end>plan.request.start:
                        exceeded.update(np.flatnonzero(env.ledger.usage((start+end)/2,plan)>capacity+1e-8).tolist())
                for index in exceeded:
                    counters[run.a.b.p.RESOURCES[index]]+=1
            action=run.rule_action('price',env)
            obs,_,done,_,_=env.step(action)
        result=env.result()
        return dict(seed=seed,load=load,success=result['success'],requests=result['requests'],
                    counters=dict(counters),by_service=result['by_service'])
    finally:
        env.close()


def isolated(spec):
    x,loss=spec
    cfg=run.a.scenario(39100,1,mobility='stationary',distance_loss=loss)
    cfg['private']['initial_positions']['7']=[float(x),0.,80.]
    cfg['public']['allowed_aps']['7']=[3,4]
    cfg['public']['missions']['7'].update(x_min=float(x),x_max=float(x))
    cfg['requests'][0].update(id=1,service='telemetry',members=[7],start=2.,stop=6.,
        period=.02,deadline=.2,payload=1000,chunks=1,reusable=False,minimum_ratio=.95)
    with tempfile.TemporaryDirectory() as tmp:
        prefix=Path(tmp)/'isolated'
        result=run.episode(cfg,'first_fit',prefix=prefix)
        import gzip
        delays=[]
        for row in csv.DictReader(gzip.open(str(prefix)+'-events.csv.gz','rt')):
            if row['event']=='complete':
                delays.append(float(row['detail']))
        return dict(x=x,distance_loss=loss,success=result['success'],admitted=result['admitted'],
            on_time=sum(s['on_time'] for s in result['settled']),expected=sum(s['expected'] for s in result['settled']),
            mean_delay=float(np.mean(delays)) if delays else None,phy_tx_seconds=result['phy_tx_seconds'],
            associations=result['associations'])


def main():
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(pressure,[(s,l) for s in range(39030,39034) for l in (32,48,64,80,96)]))
        isolated_rows=list(pool.map(isolated,[(x,loss) for loss in (False,True) for x in (20,55,80,95,110,145)]))
    run.write(run.OUT/'calibration.json',dict(development_rows=rows,isolated=isolated_rows,
        interpretation='candidate resource-exceedance counts can overlap and are ledger diagnostics, not proof of packet bottlenecks; no algorithm/test-dependent tuning'))
    print('CALIBRATED',len(rows),len(isolated_rows),flush=True)


if __name__=='__main__':
    main()
