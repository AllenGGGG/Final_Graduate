"""Independent reconstruction of new-platform decisions and raw SLA evidence.

Run AFTER trimodal_run analyze/audit. Does not trust stored success flags and
replays frozen policies from checkpoints, not the training process objects.
"""
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .trimodal_run import OUT, ROOT, NS3, PROTOCOL, profile, write, sources, run_policy
from .platform_verify import sha
from ..trimodal import Request, Ledger, compile_plan, RESOURCE_KEYS, MODES, TrimodalEnv, validate_metadata


def main():
    import torch
    from stable_baselines3 import PPO
    from ..src.qlearning import DoubleDQN
    torch.set_num_threads(1)
    started=time.perf_counter()
    audit=json.loads((OUT/'final_audit.json').read_text())
    if not audit['passed'] or audit['sources']!=sources():raise AssertionError('upstream audit stale')
    mechanisms=json.loads((OUT/'mechanisms.json').read_text())
    eid_checks=0
    for case in mechanisms['results']:
        if case['name'].startswith('mechanism_id_') and case['speed']>0:
            aa=[a for a in case['associations'] if a['uav']==5]
            if len(aa)<2:raise AssertionError('identity handover not exercised')
            rows=list(csv.DictReader((OUT/'packets'/f'{case["name"]}-events.csv').open()))
            packets=[x for x in rows if x['event']=='decap']
            if {int(float(x['detail'])) for x in packets}!={0x0aee0005}:raise AssertionError('inner EID changed during handover')
            if not any(float(x['time'])<aa[-1]['time'] for x in packets) or not any(float(x['time'])>aa[-1]['time'] for x in packets):raise AssertionError('no delivery on both sides of handover')
            updates=[x for x in rows if x['event']=='mapping_update']
            if len({x['detail'] for x in updates})<2:raise AssertionError('locator not actually updated')
            eid_checks+=1
    decisions=json.loads((OUT/'decisions.json').read_text())['rows']
    evaluations=json.loads((OUT/'evaluation.json').read_text())['rows']
    analysis=json.loads((OUT/'analysis.json').read_text())
    byname={r['name']:r for r in evaluations}
    reconstructed=[];resource_checks=[];actual_handovers=0
    native_bytes=0;policy_checks=0;state_measurements=0
    replayed={}
    for method in PROTOCOL['methods']:
        for init in (PROTOCOL['inits'] if method in ('DDQN','PPO') else [None]):
            model=None
            if init is not None:
                d=OUT/'models'/f'{method}_{init}'
                validate_metadata(d/'metadata.json',TrimodalEnv(uavs=PROTOCOL['uavs'],profile=profile()))
                model=DoubleDQN.load(d/'model.pt') if method=='DDQN' else PPO.load(d/'model',device='cpu')
            for seed in PROTOCOL['test_seeds']:
                for load in PROTOCOL['loads']:
                    r,_=run_policy(method,seed,load,model,init)
                    replayed[(method,init,seed,load)]=r
    for d in decisions:
        if json.loads(json.dumps(replayed[(d['method'],d['init'],d['scenario_seed'],d['load'])]['decisions']))!=d['decisions']:
            raise AssertionError('checkpoint policy replay differs')
        policy_checks+=1
        r=byname[d['name']]
        ledger=Ledger();plans={}
        for x in d['decisions']:
            q=dict(x['request']);q['attachments']=tuple(q['attachments']);request=Request(**q)
            if x['selected'] is not None:
                p=compile_plan(request,*x['selected'],profile=profile())
                ledger.commit(p);plans[request.id]=p
        # Compare installed task metadata against EVERY selected plan, not just counts.
        lines=[x for x in (OUT/r['task_file']).read_text().splitlines() if x and not x.startswith('#')]
        actual_tasks=[]
        for line in lines:
            sid,tenant,mode,uav,start,end,period,deadline,edge,server,cache=line.split()
            actual_tasks.append((int(sid),int(tenant),mode,int(uav),float(start),float(end),float(period),float(deadline),int(edge),int(server),int(cache)))
        expected_tasks=[(p.request.id,p.request.tenant,p.request.mode,p.request.uav,p.request.start,p.request.end,p.request.period,p.request.deadline,p.edge,p.server,p.cache) for p in plans.values()]
        if actual_tasks!=expected_tasks:raise AssertionError('executed configuration is not selected configuration')
        expected_emits={}
        for p in plans.values():
            q=p.request
            for i,at in enumerate(np.arange(q.start+.05,q.end-1e-8,q.period),1):expected_emits[q.id,i]=float(at)
        rows=list(csv.DictReader((OUT/'packets'/f'{d["name"]}-events.csv').open()))
        for x in rows:
            if x['event'] in ('ndn_interest','ndn_data') and int(x['slice']) in plans:
                sid=int(x['slice']);k='ndn_interest_bytes' if x['event']=='ndn_interest' else 'ndn_data_bytes'
                if int(x['bytes'])!=profile()[k]+len(str(sid))-1:raise AssertionError('NDN encoded name/header size differs from contract')
        observed_emits={(int(x['slice']),int(x['seq'])):float(x['time']) for x in rows if x['event']=='emit'}
        if expected_emits.keys()!=observed_emits.keys() or any(abs(v-observed_emits[k])>1e-7 for k,v in expected_emits.items()):raise AssertionError('emission schedule differs')
        completed={}
        for x in rows:
            if x['event']=='complete':
                k=int(x['slice']),int(x['seq'])
                if k in completed or k not in expected_emits:raise AssertionError('duplicate or foreign completion')
                latency=float(x['time'])-expected_emits[k]
                if abs(latency-float(x['detail']))>1e-7:raise AssertionError('completion latency differs')
                completed[k]=latency
        success=0
        for sid,p in plans.items():
            keys=[k for k in expected_emits if k[0]==sid]
            ontime=sum(k in completed and completed[k]<=p.request.deadline+1e-8 for k in keys)
            ok=ontime==len(keys)
            if ok!=r['slices'][str(sid)]['success'] or ontime!=r['slices'][str(sid)]['on_time']:raise AssertionError('stored SLA differs from raw packets')
            success+=ok
        # Integral reservations cover bounded transmission work, not instantaneous
        # burst-free wireless service. Report excesses rather than rewriting data.
        allowed=dict.fromkeys(RESOURCE_KEYS,0.)
        for p in plans.values():
            for k,v in zip(RESOURCE_KEYS,p.usage):allowed[k]+=v*(p.request.drain-p.request.start)
        wired=Counter();active_phy=0.
        for x in rows:
            if x['event'] in ('wire_ip','wire_ndn'):
                wired[f'wire:{x["node"]}>{x["seq"]}']+=int(x['bytes'])
            if x['event']=='phy_tx' and any(p.request.start<=float(x['time'])<p.request.drain for p in plans.values()):active_phy+=float(x['detail'])
            if x['event']=='map_state':state_measurements+=1
        wire_excess={k:float(v-allowed[k]) for k,v in wired.items() if v>allowed[k]+1e-6}
        radio_excess=max(0.,active_phy-allowed['airtime:shared'])
        resource_checks.append(dict(name=d['name'],wire_integral_excess_bytes=wire_excess,phy_integral_excess_seconds=radio_excess))
        actual_handovers+=int(len({a['ap'] for a in r['associations'] if a['uav']==5})==2)
        native_bytes+=sum(wired.values())
        reconstructed.append(dict(method=d['method'],init=d['init'],scenario=d['scenario_seed'],load=d['load'],repeat=d['radio_repeat'],success=success/d['requests']))
    for m in PROTOCOL['methods']:
        mean=float(np.mean([x['success'] for x in reconstructed if x['method']==m]))
        if abs(mean-analysis['scores'][m]['strict_success'])>1e-12:raise AssertionError('reported mean differs')
    # Collect native resident memory separately: process wall time alone does not
    # establish memory scalability. These use saved mechanism inputs, not test data.
    measurements=[]
    for n in (4,12):
        for seed in (303,304,305):
            name=f'memory_{n}_{seed}';prefix=OUT/'performance'/name;prefix.parent.mkdir(exist_ok=True)
            inp=OUT/'inputs'/f'cost_{n}_{seed}.tsv';metric=Path(str(prefix)+'-time.txt')
            cmd=['/usr/bin/time','-f','%e %U %S %M','-o',str(metric),str(NS3/'build/scratch/trimodal-scenario'),
                 f'--output={prefix}',f'--tasks={inp}',f'--seed={seed}',f'--uavs={n}','--speed=8','--startX=0']
            p=subprocess.run(cmd,env=dict(os.environ,LD_LIBRARY_PATH=str(NS3/'build/lib')),capture_output=True,text=True,timeout=120)
            if p.returncode:raise AssertionError('native memory measurement failed')
            wall,user,system,rss=map(float,metric.read_text().split())
            measurements.append(dict(uavs=n,seed=seed,wall_seconds=wall,user_seconds=user,system_seconds=system,max_rss_kib=int(rss)))
    artifacts={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'performance').iterdir())}
    for name in ('protocol.json','profile.json','training.json','mechanisms.json','decisions.json','evaluation.json','analysis.json','report.md','final_audit.json','tests.log'):
        artifacts[name]=sha(OUT/name)
    for p in (OUT/'models').rglob('*'):
        if p.is_file():artifacts[str(p.relative_to(OUT))]=sha(p)
    resource_violations=sum(bool(x['wire_integral_excess_bytes']) or x['phy_integral_excess_seconds']>1e-9 for x in resource_checks)
    result=dict(passed=True,independent_raw_sla_reconstruction=True,raw_cases=len(reconstructed),
        policy_checkpoint_replays=policy_checks,stable_eid_handover_cases=eid_checks,all_selected_plans_atomically_feasible=True,
        all_installed_tasks_match_plans=True,scenarios_with_actual_handovers=actual_handovers,
        measured_ground_network_bytes=native_bytes,state_measurements=state_measurements,
        measured_resource_integral_violation_cases=resource_violations,resource_details=resource_checks,
        performance=measurements,artifacts=artifacts,sources=sources(),verifier_sha256=sha(__file__),
        native_binary_sha256=sha(NS3/'build/scratch/trimodal-scenario'),seconds=time.perf_counter()-started,
        scope='independent native SLA, plan/task matching and checkpoint repeatability; integral resource checks are not an instantaneous or universal wireless guarantee')
    write(OUT/'independent_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','sources','resource_details','performance')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
