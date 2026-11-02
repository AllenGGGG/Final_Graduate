"""Before training: native action sensitivity and an explicit optimistic MILP.

The MILP bounds the reservation model, NOT ns-3's globally optimal packet SLA.
It gives free warm caches, ignores finite state, permits any mission AP, uses
the cheapest tier, and assumes every admitted request succeeds. No policy gets
future arrivals; only this labelled offline relaxation does.
"""
import itertools
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from slice_alloc.tools import allocation_final_run as run


def relaxed_bound(cfg):
    a = run.a
    public = cfg['public']
    plans = []
    for raw in cfg['requests']:
        request = a.b.Service(**{**raw,'members':tuple(raw['members'])})
        unique = set()
        for aps in itertools.product(*(public['allowed_aps'][str(n)] for n in request.members)):
            pub = dict(public,allowed_aps={**public['allowed_aps'],
                **{str(n):[ap] for n,ap in zip(request.members,aps)}})
            obs = dict(cached=[dict(node=e,tenant=request.tenant,content=request.content,
                version=request.version,bytes=request.size) for e in (1,2)])
            for mode in a.COMPATIBLE[request.service]:
                for edge in (1,2):
                    if request.service=='telemetry':
                        plan=a.p.compile_plan(request.bound('ip'),edge,edge,0,pub,obs)
                    else:
                        plan=a.b.compile_plan(request,mode,edge,edge,edge if mode=='ndn' else 0,pub,obs)
                    vector=plan.peak[:11]*min(a.TIERS)
                    identity=tuple(vector)
                    if identity not in unique:
                        unique.add(identity)
                        plans.append((request,vector))
    times=sorted({t for r,_ in plans for t in (r.start,r.drain)})
    n=len(plans); count=len(cfg['requests'])
    matrix=lil_matrix((count+11*(len(times)-1),n))
    upper=np.ones(matrix.shape[0])
    capacity=a.p.capacities(public)[:11]*a.OVERBOOK
    for j,(r,vector) in enumerate(plans):
        matrix[r.id-1,j]=1
        for k,(lo,hi) in enumerate(zip(times,times[1:])):
            if r.start <= (lo+hi)/2 < r.drain:
                matrix[count+11*k:count+11*(k+1),j]=vector[:,None]/capacity[:,None]
    c=-np.array([a.WEIGHTS[r.service] for r,_ in plans])
    result=milp(c,integrality=np.ones(n),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix.tocsr(),np.full(len(upper),-np.inf),upper),
        options=dict(time_limit=30.,mip_rel_gap=1e-7))
    assert result.mip_dual_bound is not None, result.message
    offered=sum(a.WEIGHTS[r['service']] for r in cfg['requests'])
    return dict(upper_utility=min(offered,float(-result.mip_dual_bound)),
        offered_utility=offered,optimal=bool(result.success),gap=float(result.mip_gap),
        variables=n,relaxation='free warm cache, no state limits, any mission AP, lowest tier, perfect delivery')


def diagnose(out):
    sensitivity=[]
    for business,modes in run.a.COMPATIBLE.items():
        cfg=run.a.scenario(71000,1,mobility='stationary')
        raw=cfg['requests'][0]
        raw.update(id=1,service=business,members=[7],start=2.,stop=9.,period=.02 if business=='telemetry' else .3 if business=='command' else 2.,
            deadline=.2 if business=='telemetry' else .6 if business=='command' else .8,
            payload=1000,chunks=4 if business=='object' else 1,
            reusable=business=='object',minimum_ratio=.95 if business!='object' else 1.)
        cfg['private']['initial_positions']['7']=[20.,0.,80.]
        for tier in run.a.TIERS:
            result=run.episode(cfg,f'tier_{tier}',prefix=out/'diagnostic_native'/f'{business}-{tier}')
            sensitivity.append(dict(service=business,tier=tier,success=result['success'],
                utility=result['utility'],expected=sum(s['expected'] for s in result['settled']),
                on_time=sum(s['on_time'] for s in result['settled']),audited=result['audited']))
    action_rows=[]
    for seed in range(71001,71005):
        for variant in ('full','strict'):
            env=run.environment(variant)
            counts=dict(arrivals=0,risk_candidates=0,tier_candidates=0,only_risk_arrivals=0)
            try:
                _,_=env.reset(options={'scenario':run.a.scenario(seed,80)})
                done=False
                while not done:
                    counts['arrivals']+=1
                    counts['risk_candidates']+=sum(p.risk>1e-8 for p in env.options)
                    counts['only_risk_arrivals']+=bool(env.options) and all(p.risk>1e-8 for p in env.options)
                    counts['tier_candidates']+=len(env.options)
                    _,_,done,_,_=env.step(run.rule_action('greedy',env))
                result=env.result()
                action_rows.append(dict(seed=seed,variant=variant,**counts,
                    success=result['success'],utility=result['utility'],risk_admissions=result['risk_admissions']))
            finally:
                env.close()
    bounds=[]
    for seed,load in ((71005,48),(71006,80)):
        cfg=run.a.scenario(seed,load,mobility='stationary')
        bound=relaxed_bound(cfg)
        result=run.episode(cfg,'greedy',prefix=out/'diagnostic_native'/f'bound-greedy-{seed}')
        assert bound['upper_utility']+1e-7>=result['utility']
        bounds.append(dict(seed=seed,load=load,**bound,greedy_native_utility=result['utility'],
            optimistic_gap=(bound['upper_utility']-result['utility'])/bound['offered_utility']))
    changed=any(len({r['on_time'] for r in sensitivity if r['service']==s})>1 for s in run.a.COMPATIBLE)
    risk_available=any(r['risk_candidates']>0 for r in action_rows if r['variant']=='full')
    result=dict(native_tier_sensitivity=sensitivity,action_space=action_rows,offline_bounds=bounds,
        action_changes_native_delivery=changed,risk_actions_exist=risk_available,
        proceed=changed and risk_available,
        interpretation='A wide optimistic MILP gap is not evidence that RL can close it; active rejection need not be nonzero or beneficial. Test constraints and all baselines independently.')
    assert result['proceed'], result
    run.write(out/'diagnosis.json',result)
    return result


def main():
    assert not (run.OUT/'test_started.json').exists(),'Do not backdate development diagnosis'
    result=diagnose(run.OUT)
    print('DIAGNOSED',result['proceed'],result['action_changes_native_delivery'],flush=True)


if __name__=='__main__':
    main()
