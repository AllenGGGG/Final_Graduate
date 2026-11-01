"""Deployable sampled look-ahead; never reads future held-out requests.

Synthetic futures come from the declared training arrival/template law updated
by observed past arrivals. This is a comparator, not a clairvoyant teacher.
"""
from types import SimpleNamespace
from statistics import mean
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from ..joint import Request,JointEnv,candidates,choose,write_tasks,request_hash
from .joint_diagnose import clone_ledger
from .joint_run import OUT,PROTOCOL,profile,native_case,write,source_hashes
from .joint_analyze import failure_counts
from .platform_verify import sha


def sampled_action(env,horizon=4,samples=3):
    if not env.plans:return 0
    # Hash only contracts already observed. No held-out suffix, load label or
    # realized future trajectory enters the random stream or sampled demand.
    prefix=env.requests[:env.i+1]
    rng=np.random.default_rng(int(request_hash(prefix)[:16],16))
    elapsed=env.now-prefix[0].arrival
    gap=(elapsed+4*.035)/(env.i+4)
    h=min(horizon,env.count-env.i-1)
    futures=[]
    for _ in range(samples):
        at=env.now;requests=[]
        for j in range(h):
            at+=float(rng.exponential(gap));category=int(rng.choice(3,p=(.4,.4,.2)))
            allowed=(('ip',),('ndn_cold','ndn_pin'),('ip','ndn_cold','ndn_pin'))[category]
            requests.append(Request(env.requests[env.i].id+j+1,at,int(rng.choice((2,4,7))),allowed))
        futures.append(requests)
    scores=[]
    for action in range(len(env.plans)+1):
        value=[]
        for future in futures:
            ledger=clone_ledger(env.ledger);count=0.
            if action:ledger.commit(env.plans[action-1]);count+=env.plans[action-1].request.weight
            for request in future:
                for owner,p in tuple(ledger.active.items()):
                    if p.request.end<=request.arrival:ledger.release(owner)
                plans=candidates(env.compiler,ledger,request)
                selected=choose(SimpleNamespace(ledger=ledger,plans=plans),'adaptive_price',1.)
                if selected:ledger.commit(plans[selected-1]);count+=request.weight
            value.append(count)
        scores.append(mean(value))
    best=np.flatnonzero(np.isclose(scores,max(scores),rtol=0,atol=1e-10))
    fallback=choose(env,'adaptive_price',1.)
    return int(fallback if fallback in best else best[0])


def rollout_episode(seed,load,prof,config,trace=False):
    env=JointEnv(prof,count=PROTOCOL['requests'],load=load);env.reset(seed=seed)
    started=perf_counter()
    while not env.done:env.step(sampled_action(env,**config))
    accepted=[p for _,p in env.decisions if p is not None]
    row=dict(seed=seed,load=load,method='sampled_rollout',requests=len(env.requests),accepted=len(accepted),
        model_utility=sum(p.request.weight for p in accepted),model_acceptance=len(accepted)/len(env.requests),
        input_sha256=request_hash(env.requests),decision_ms=1000*(perf_counter()-started)/len(env.requests),
        rejected_no_candidate=env.no_candidate,policy_rejected=env.policy_rejected,
        decisions=[dict(id=r.id,arrival=r.arrival,rounds=r.rounds,allowed=r.allowed,selected=p.kind if p else None) for r,p in env.decisions])
    return (row,accepted) if trace else row


def main():
    prof=profile();trials=[]
    for config in (dict(horizon=4,samples=3),dict(horizon=8,samples=3)):
        rows=[]
        for seed in PROTOCOL['validation_seeds']:
            for load in PROTOCOL['loads']:
                row=rollout_episode(seed,load,prof,config);rows.append(row)
                print('validation',config,seed,load,row['accepted'],flush=True)
        trials.append(dict(config=config,model_utility=mean(r['model_utility'] for r in rows),rows=rows))
    selected=max(trials,key=lambda r:r['model_utility']);tasks=[];episodes=[]
    for seed in PROTOCOL['test_seeds']:
        for load in PROTOCOL['loads']:
            row,plans=rollout_episode(seed,load,prof,selected['config'],True)
            name=f'test_{seed}_{load:g}_sampled_rollout';path=OUT/'inputs'/f'{name}.tsv';write_tasks(path,plans)
            row['case']=name;episodes.append(row)
            tasks.append((name,path,seed+PROTOCOL['test_wifi_seed_offset'],0 if seed%2==0 else 5.))
            print('test',seed,load,row['accepted'],flush=True)
    with ThreadPoolExecutor(4) as pool:packets=list(pool.map(native_case,tasks))
    by_name={p['name']:p for p in packets}
    for row in episodes:
        packet=by_name[row['case']];row.update(packet_success=packet['strict_slices']/row['requests'],
            admitted_packet_success=packet['strict_slices']/max(row['accepted'],1),packet_input_sha256=packet['input_sha256'])
    proof=dict(source_sha256=sha(__file__),diagnostic_source_sha256=sha(__file__.replace('joint_rollout.py','joint_diagnose.py')),
        frozen_source_sha256=source_hashes(),comparison_sha256=sha(OUT/'comparison.json'),
        trials=trials,selected_config=selected['config'],episodes=episodes,packet_cases=packets,
        summary=failure_counts(episodes),mean_decision_ms=mean(r['decision_ms'] for r in episodes),
        scope='sampled online comparator with training-law futures; not the actual future, not an upper bound or intrinsic information-gap decomposition',
        demonstration_status='no DQfD/teacher update; comparator must first establish useful validation behavior')
    write(OUT/'rollout_comparison.json',proof)
    print(json.dumps(dict(selected=selected['config'],summary=proof['summary'],decision_ms=proof['mean_decision_ms']),indent=2),flush=True)


if __name__=='__main__':main()
