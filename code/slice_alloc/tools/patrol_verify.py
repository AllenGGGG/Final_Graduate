"""Independent raw-packet reconstruction and frozen-policy/native replay audit."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import numpy as np
from slice_alloc.patrol import *
from .patrol_run import OUT, PROTOCOL_SPEC, BASELINES, sha, source_hashes, write, episode


def raw_rows(prefix):
    path=Path(str(prefix)+'-events.csv.gz')
    return list(csv.DictReader(gzip.open(path,'rt') if path.exists() else open(str(prefix)+'-events.csv')))


def check_episode(row):
    prefix=OUT/'native'/row['path']
    for suffix,key in [('-result.json','result_sha256'),('-events.csv.gz','events_sha256'),('-scenario.json','scenario_sha256'),('-summary.json','summary_sha256')]:assert sha(str(prefix)+suffix)==row[key],str(prefix)+suffix
    cfg=json.loads(Path(str(prefix)+'-scenario.json').read_text());validate('scenario',cfg)
    result=json.loads(Path(str(prefix)+'-result.json').read_text());requests={r['id']:request_from_json(r) for r in cfg['requests']};events=raw_rows(prefix)
    emissions=collections.defaultdict(dict);completions=collections.defaultdict(dict);installed={};wire=collections.Counter();air=0.;versions=collections.defaultdict(list)
    for x in events:
        sid=int(x['slice']);seq=int(x['seq']);t=float(x['time']);n=int(x['node']);b=int(x['bytes']);detail=float(x['detail'])
        if x['event']=='emit':
            assert seq not in emissions[sid];emissions[sid][seq]=(t,n,b)
        elif x['event']=='complete':
            assert seq not in completions[sid];completions[sid][seq]=(t,n,b,detail)
        elif x['event']=='install':
            assert sid not in installed;installed[sid]=(seq,int(detail),b)
        elif x['event'] in ('wire_ip','wire_ndn'):wire[n,seq]+=b
        elif x['event']=='phy_tx':air+=detail
        elif x['event']=='associate':versions[n].append((t,seq,b))
        elif x['event']=='udp_send':
            r=requests[sid];kind=int(detail)
            expected=r.payload+37 if kind==5 else r.payload+90 if kind==4 else r.payload+53 if kind==7 else 53
            assert b==expected,(row['path'],x,expected)
        elif x['event']=='ndn_interest':assert b<=100
        elif x['event']=='ndn_data':assert b<=1128
        assert x['event'] not in ('mapping_capacity_drop','mapcache_capacity_drop','namespace_drop','inner_packet_drop'),(row['path'],x)
    reconstructed={}
    for sid,(edge,server,cache) in installed.items():
        r=requests[sid];assert len(emissions[sid])==r.expected
        on_time=0
        for seq,(at,n,size,delay) in completions[sid].items():
            born,origin,payload=emissions[sid][seq];assert abs(at-born-delay)<2e-8
            assert size==payload==r.payload
            assert n==0 if r.mode=='ip' else n==origin
            if at-born<=r.deadline+1e-9:on_time+=1
        reconstructed[sid]=dict(id=sid,expected=r.expected,on_time=on_time,success=on_time+1e-9>=r.expected*r.minimum_ratio)
    assert reconstructed=={s['id']:s for s in result['settled']}
    assert row['success']==sum(s['success'] for s in reconstructed.values())
    assert row['admitted']==len(installed)==result['admitted']
    assert len(result['decisions'])==row['requests']==len(requests)
    ledger=Ledger(cfg['public'])
    for record,pair in zip(result['decisions'],result['observations']):
        o=pair['observation'];d=pair['decision'];validate('observation',o);validate('decision',d);assert record['decision']==d
        sid=d['request_id'];assert sid==o['request_id'];ledger.release(o['time'])
        if d['accept']:
            assert installed[sid]==(d['edge'],d['server'],d['cache'])
            assert all(o['associations'][str(n)] in cfg['public']['allowed_aps'][str(n)] for n in requests[sid].members)
            plan=compile_plan(requests[sid],d['edge'],d['server'],d['cache'],cfg['public'],o)
            if plan.warm and not plan.cache:plan=compile_plan(requests[sid],d['edge'],d['server'],d['cache'],cfg['public'],{**o,'cached':[]})
            ledger.add(plan)
        else:assert sid not in installed and sid not in emissions
    summary=row['summary']
    assert all(summary[k]==0 for k in ('remaining_mapping','remaining_mapcache','remaining_pending','remaining_application','remaining_cache_entries'))
    assert summary['generated']==sum(len(v) for v in emissions.values())
    assert summary['delivered']==sum(len(v) for v in completions.values())
    assert summary['budget_drops']==0
    return dict(path=row['path'],wire_bytes={f'{a}>{b}':v for (a,b),v in wire.items()},radio_tx_seconds=air,associations={str(k):v for k,v in versions.items()},request_success=row['success'])


def mechanism_checks():
    m=json.loads((OUT/'mechanisms.json').read_text());assert m['passed'];checks={}
    for name in ('ip_near','ip_far'):
        rows=raw_rows(OUT/'mechanisms'/name);expected=m['cases'][name]['result']['settled'][0]['expected'];wire=collections.Counter()
        for r in rows:
            if r['event']=='wire_ip':wire[int(r['node']),int(r['seq'])]+=int(r['bytes'])
        path_expected=[(3,1),(1,0)] if name=='ip_near' else [(3,1),(1,2),(2,0)]
        assert wire==collections.Counter({edge:expected*1037 for edge in path_expected})
        checks[name+'_exact_network_layer_bytes']=True
    for name in ('id_local','id_remote'):
        rows=raw_rows(OUT/'mechanisms'/name)
        associations=[r for r in rows if r['event']=='associate' and r['node']=='5']
        updates=[r for r in rows if r['event']=='mapping_update']
        decaps=[r for r in rows if r['event']=='decap']
        assert {r['bytes'] for r in associations}=={'3','4'}
        assert len({r['seq'] for r in updates})>=2
        assert {int(float(r['detail'])) for r in decaps}=={0x0aee0000+5}
        switch=float(associations[1]['time']);assert any(float(r['time'])<switch for r in decaps) and any(float(r['time'])>switch for r in decaps)
        checks[name+'_same_eid_before_and_after_actual_association']=True
        queries=[r for r in rows if r['event']=='query'];assert queries
        checks[name+'_queries_are_observed_demand_events']=len(queries)
    a=m['cases']['ndn_cache']['summary'];b=m['cases']['ndn_cold']['summary']
    assert a['ndn_source_payload_bytes']==4000 and b['ndn_source_payload_bytes']==28000
    def lastmile(name):return sum(int(r['bytes']) for r in raw_rows(OUT/'mechanisms'/name) if r['event']=='ndn_data' and r['node'] in ('3','4'))
    assert lastmile('ndn_cache')==lastmile('ndn_cold')>0
    checks['cache_keeps_last_mile_cost']=True
    checks['cache_source_payload_reduction']=1-a['ndn_source_payload_bytes']/b['ndn_source_payload_bytes']
    dev=json.loads((OUT/'development.json').read_text())
    for profile,resource in [('bandwidth','wire:'),('mapping','mapping:'),('cache','cache:')]:
        assert any(any(k.startswith(resource) and v>0 for k,v in d['pressure'].items()) for d in dev if d['profile']==profile)
        checks[profile+'_actual_candidate_resource_conflicts']=True
    # The future trajectory differs only after the first two decision times.
    observations=[]
    for speed in (5.,11.):
        cfg=scenario(1500,1,'mixed',4);cfg['private']['motions']=[dict(node=5,speed=speed,right=True,start=6.)]
        cfg['requests']=[Request(i,1,'ip',(7,),float(i+1),float(i+2),.08,.2,1000,1,i).json() for i in (1,2)]
        native=NativeSession(cfg);past=[]
        while native.observation['kind']!='done':
            past.append(native.observation);sid=native.observation['request_id'];native.decide(dict(schema=PROTOCOL,request_id=sid,accept=False,edge=0,server=0,cache=0))
        native.close();observations.append(past)
    assert observations[0]==observations[1]
    checks['native_future_change_does_not_change_past_observations']=True
    return checks


def policy_replay(spec):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from slice_alloc.src.qlearning import DoubleDQN
    method,init,profile,load,seed=spec
    prefix=OUT/'native'/f'{method}-{init or 0}/{profile}-{load}-{seed}';cfg=json.loads(Path(str(prefix)+'-scenario.json').read_text())
    model=DoubleDQN.load(OUT/'models'/f'ddqn-{init}.pt') if method=='ddqn' else PPO.load(OUT/'models'/f'ppo-{init}.zip',device='cpu') if method=='ppo' else None
    original=json.loads(Path(str(prefix)+'-result.json').read_text())
    with tempfile.TemporaryDirectory(prefix='lowalt-patrol-replay-') as tmp:
        out=Path(tmp)/'episode';again=episode(cfg,method,model,out,trace=True)
        assert again['decisions']==original['decisions']
        assert again['settled']==original['settled']
        assert again['observations']==original['observations']
        with gzip.open(str(prefix)+'-events.csv.gz','rb') as f:first=f.read()
        with gzip.open(str(out)+'-events.csv.gz','rb') as f:second=f.read()
        assert first==second,('non-reproducible native events',spec)
    return dict(method=method,init=init,profile=profile,load=load,seed=seed,decisions=len(original['decisions']),exact_native_event_replay=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=8);parser.add_argument('--validation-only',action='store_true');args=parser.parse_args()
    if args.validation_only:
        data=json.loads((OUT/'validation.json').read_text());assert data['protocol_sha256']==sha(OUT/'protocol.json');assert len(data['rows'])==640
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:verified=list(pool.map(check_episode,data['rows']))
        write(OUT/'validation_audit.json',dict(passed=True,episodes=len(verified),validation_sha256=sha(OUT/'validation.json'),verifier_sha256=sha(__file__)))
        print('validation raw SLA and contract audit: 640/640',flush=True);return
    frozen=json.loads((OUT/'protocol.json').read_text());assert frozen['sources']==source_hashes()
    assert frozen['binary_sha256']==sha(NS3/'build/scratch/patrol-scenario')
    data=json.loads((OUT/'evaluation.json').read_text());assert data['sources']==frozen['sources'];assert data['protocol_sha256']==sha(OUT/'protocol.json')
    validation=json.loads((OUT/'validation_audit.json').read_text());assert validation['passed'] and validation['validation_sha256']==sha(OUT/'validation.json') and validation['verifier_sha256']==sha(__file__)
    checks=mechanism_checks();rows=data['rows'];assert len(rows)==3600
    # Each method sees precisely the same private scenario, not copied outcomes.
    inputs=collections.defaultdict(set)
    for r in rows:inputs[r['profile'],r['load'],r['seed']].add(r['scenario_sha256'])
    assert len(inputs)==240 and all(len(v)==1 for v in inputs.values())
    trained=json.loads((OUT/'training.json').read_text());assert len(trained)==10
    for t in trained:
        suffix='.pt' if t['kind']=='ddqn' else '.zip';assert t['model_sha256']==sha(OUT/'models'/f"{t['kind']}-{t['seed']}{suffix}")
        assert all(10000<=s['seed']<1000000 for s in t['training_scenarios'])
    reconstructed=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i,r in enumerate(pool.map(check_episode,rows),1):
            reconstructed.append(r)
            if i%200==0:print(f'raw SLA/contract audit {i}/{len(rows)}',flush=True)
    specs=[(m,i,p,40,4100) for m in ('ddqn','ppo') for i in PROTOCOL_SPEC['initializations'] for p in PROFILES]+[(m,None,'mixed',24,4100) for m in BASELINES]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:replays=list(pool.map(policy_replay,specs))
    write(OUT/'resource_reconstruction.json',reconstructed)
    result=dict(passed=True,episodes=len(rows),decisions=sum(r['requests'] for r in rows),checks=checks,policy_native_replays=replays,same_private_inputs_across_methods=True,raw_sla_reconstruction=True,all_accepted_reservations_revalidated=True,sources=source_hashes(),verifier_sha256=sha(__file__),artifacts={name:sha(OUT/name) for name in ('protocol.json','mechanisms.json','development.json','training.json','evaluation.json','analysis.json','resource_reconstruction.json','validation.json','validation_audit.json')})
    write(OUT/'independent_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('sources','artifacts','policy_native_replays')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
