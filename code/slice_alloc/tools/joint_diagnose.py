"""Read-only replay instrumentation and reservation-model sensitivity.

Never changes the frozen environment, calibration, policies or packet outcomes.
Enumeration completeness refers only to the declared three-contract catalogue.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import mean
import gzip
import json
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import csr_matrix
from stable_baselines3 import PPO
import torch
from ..joint import (AtomicLedger,Compiler,JointEnv,RESOURCES,CAPACITIES,CACHE_CAPACITIES,
                     PERIOD,choose,request_hash)
from ..protocol_contracts import cache_usage
from ..src.qlearning import DoubleDQN,tensor_observation
from .joint_run import OUT,PROTOCOL,profile,write,source_hashes
from .platform_verify import sha


def clone_ledger(ledger):
    other=AtomicLedger(ledger.capacities,ledger.cache_capacities)
    other.active=dict(ledger.active);other.cache=deepcopy(ledger.cache);other.version=ledger.version
    return other


def plan_slack(ledger,plan):
    segments=[s for p in tuple(ledger.active.values())+(plan,) for s in p.segments]
    times=sorted({s.start for s in segments if s.start>=plan.request.arrival})
    usage=np.array([sum((np.array(s.usage) for s in segments if s.start<=t<s.end),np.zeros(3)) for t in times])
    # Include future commitments, not only occupancy at the request arrival.
    minimum=np.min(ledger.capacity-usage,axis=0)
    leases=ledger.cache.leases+plan.reservation.new_leases+plan.reservation.required_leases
    cache_times=sorted({plan.request.arrival}|{l.valid_from for l in leases if l.valid_from>=plan.request.arrival})
    cache_min=min((ledger.cache_capacities['relay']-cache_usage(leases,t).get('relay',0)
                   for t in cache_times),default=ledger.cache_capacities['relay'])
    slack={key:float(minimum[i]) for i,key in enumerate(RESOURCES)};slack['cache_payload_bytes']=float(cache_min)
    normalized={key:float(minimum[i]/ledger.capacity[i]) for i,key in enumerate(RESOURCES)}
    normalized['cache_payload_bytes']=cache_min/ledger.cache_capacities['relay']
    blocked=[key for key,value in slack.items() if value < -1e-9]
    return dict(minimum_interval_slack=slack,normalized_interval_slack=normalized,
                blocking_dimensions=blocked,feasible=ledger.feasible(plan))


def decision_diagnostic(env,action):
    request=env.requests[env.i]
    alternatives={kind:plan_slack(env.ledger,env.compiler.compile(request,kind)) for kind in request.allowed}
    found=[kind for kind,r in alternatives.items() if r['feasible']]
    if found!=[p.kind for p in env.plans]:raise ValueError('candidate generator omitted/reordered an allowed feasible contract')
    return dict(id=request.id,arrival=request.arrival,rounds=request.rounds,allowed=request.allowed,
        selected=env.plans[action-1].kind if action else None,feasible_candidates=len(found),
        enumerated_candidates=len(alternatives),enumeration_status='exhaustive' if found else 'infeasible',
        enumeration_scope='all allowed fixed-path contracts; NOT general function/path search',
        rejection_reason=None if action else ('no_candidate' if not found else 'policy'),
        binding_dimension=sorted({d for r in alternatives.values() for d in r['blocking_dimensions']}) if not found else [],
        envelope_slack=alternatives,ledger_version=env.ledger.version,
        scope='counterfactual slack at this policy history; blocking dimensions may overlap and are not additive causal shares')


class ConditionalCompiler(Compiler):
    def __init__(self,prof,per_kind=None):super().__init__(prof);self.per_kind=per_kind or {}
    def compile(self,request,kind):
        old=self.air
        try:
            self.air=self.per_kind.get(kind,old)
            return super().compile(request,kind)
        finally:self.air=old


def solve_reference(requests,compiler,fixed=()):
    """Exact frozen MODEL reference, with an independently checked witness."""
    plans=[compiler.compile(r,k) for r in requests for k in r.allowed]
    all_plans=tuple(plans)+tuple(fixed)
    times=sorted({s.start for p in all_plans for s in p.segments})
    matrix=[];upper=[]
    for r in requests:
        matrix.append([float(p.request.id==r.id) for p in plans]);upper.append(1.)
    for at in times:
        for index,key in enumerate(RESOURCES):
            reserved=sum(s.usage[index] for p in fixed for s in p.segments if s.start<=at<s.end)
            matrix.append([sum(s.usage[index] for s in p.segments if s.start<=at<s.end)/CAPACITIES[key] for p in plans])
            upper.append((CAPACITIES[key]-reserved)/CAPACITIES[key])
        reserved=sum(1000 for p in fixed if p.kind=='ndn_pin' and p.request.arrival<=at<p.request.end)
        matrix.append([.25 if p.kind=='ndn_pin' and p.request.arrival<=at<p.request.end else 0. for p in plans])
        upper.append((4000-reserved)/4000)
    result=milp(-np.array([p.request.weight for p in plans]),integrality=np.ones(len(plans)),
        bounds=Bounds(0,1),constraints=LinearConstraint(csr_matrix(matrix),-np.inf,np.array(upper)),
        options={'time_limit':30.})
    if result.status!=0:raise ValueError('reference not proven optimal: '+str(result.message))
    selected=[p for p,x in zip(plans,result.x) if x>.5]
    ledger=AtomicLedger()
    for p in sorted(tuple(fixed)+tuple(selected),key=lambda p:p.request.arrival):ledger.commit(p)
    if abs(sum(p.request.weight for p in selected)+result.fun)>1e-6:raise ValueError('invalid optimal witness')
    return dict(objective=round(-float(result.fun),9),upper_bound=-float(result.mip_dual_bound),
        optimal=True,status=int(result.status),witness=[dict(id=p.request.id,kind=p.kind) for p in selected],
        witness_interval_and_cache_check=True)


def main():
    torch.set_num_threads(1);prof=profile();data=json.loads((OUT/'comparison.json').read_text())
    training=json.loads((OUT/'training.json').read_text());cal=json.loads((OUT/'calibration.json').read_text())
    if data['source_sha256']!=source_hashes():raise ValueError('frozen source mismatch')
    stability_path=OUT/'stability_comparison.json'
    if stability_path.exists():
        extended=json.loads(stability_path.read_text())
        training=json.loads((OUT/'stability.json').read_text())
        if extended['comparison_sha256']!=sha(OUT/'comparison.json') or extended['training_sha256']!=sha(OUT/'stability.json'):
            raise ValueError('stale stability input')
        data=extended
    models={}
    for record in training['models']:
        path=OUT/record['file']
        if sha(path)!=record['sha256']:raise ValueError('model mismatch')
        models[f"{record['algorithm']}_{record['initialization']}"]=DoubleDQN.load(path) if record['algorithm']=='ddqn' else PPO.load(path,device='cpu')
    prices=json.loads((OUT/'baselines.json').read_text())['selected']
    counts=dict(decisions=0,no_candidate=0,policy=0,infeasible_at_commit=0,search_limit=0,catalog_omissions=0)
    blocked={};flexible={};flexible_by_method={};q_errors={};decisions_path=OUT/'decision_diagnostics.jsonl.gz'
    with gzip.open(decisions_path,'wt') as output:
        for row in data['episodes']:
            env=JointEnv(prof,count=row['requests'],load=row['load']);obs,_=env.reset(seed=row['seed'])
            if request_hash(env.requests)!=row['input_sha256']:raise ValueError('replay request mismatch')
            method=row['method'];model=models.get(method)
            price=prices[method]['price'] if method in prices else 0.
            episode_records=[]
            for original in row['decisions']:
                action=int(model.predict(obs,deterministic=True)[0]) if model else choose(env,method,price)
                record=decision_diagnostic(env,action)
                if isinstance(model,DoubleDQN):
                    with torch.no_grad():values=model.network(tensor_observation(obs))[0].numpy()
                    record['q_values']=[float(v) if valid else None for v,valid in zip(values,obs['mask'])]
                if record['selected']!=original['selected']:raise ValueError('frozen decision not reproduced')
                counts['decisions']+=1
                if record['rejection_reason']:counts[record['rejection_reason']]+=1
                for d in record['binding_dimension']:blocked[d]=blocked.get(d,0)+1
                if 'ip' in record['allowed'] and len(record['allowed'])>1:
                    key=record['selected'] or 'reject';flexible[key]=flexible.get(key,0)+1
                    group=flexible_by_method.setdefault(method,{})
                    group[key]=group.get(key,0)+1
                try:obs,reward,_,_,_=env.step(action)
                except ValueError:
                    counts['infeasible_at_commit']+=1;raise
                record['reward_after_step']=reward;record['action']=action;episode_records.append(record)
            remaining_return=0.
            for record in reversed(episode_records):
                remaining_return+=record['reward_after_step']
                record['realized_future_return']=remaining_return
                if 'q_values' in record:
                    error=record['q_values'][record['action']]-remaining_return
                    record['chosen_q_minus_realized_return']=error;q_errors.setdefault(method,[]).append(error)
            for record in episode_records:output.write(json.dumps(dict(case=row['case'],method=method,**record),allow_nan=False)+'\n')
    # Only development calibration is used for conditional coefficient estimates.
    maxima={}
    for row in cal['cases']:
        if not row['name'].startswith('cal_'):continue
        for kind in ('ip','ndn_cold','ndn_pin'):
            prefix='cal_'+kind+'_'
            if row['name'].startswith(prefix):
                rounds,concurrent,_=map(int,row['name'][len(prefix):].split('_'))
                maxima[kind]=max(maxima.get(kind,0.),row['radio']['tx_seconds_from_state']/(rounds*concurrent))
                break
    hold=[]
    for row in cal['cases']:
        if not row['name'].startswith('hold_'):continue
        for kind in maxima:
            prefix='hold_'+kind+'_'
            if row['name'].startswith(prefix):
                rounds,concurrent,_=map(int,row['name'][len(prefix):].split('_'))
                hold.append(dict(case=row['name'],kind=kind,actual=row['radio']['tx_seconds_from_state'],
                    proposed_allowance=maxima[kind]*1.15*rounds*concurrent,
                    covered=row['radio']['tx_seconds_from_state']<=maxima[kind]*1.15*rounds*concurrent+1e-9))
                break
    variants={
        'current_global_15pct':(prof,{}),
        'global_fit_max_no_margin':({**prof,'airtime_seconds_per_pair_round':prof['fitted_max_seconds']},{}),
        'conditional_fit_max_15pct':(prof,{k:v*1.15 for k,v in maxima.items()}),
        'conditional_fit_max_no_margin':(prof,maxima),
        'zero_airtime_constraint_ablation':({**prof,'airtime_seconds_per_pair_round':0.},{})}
    references=[]
    for variant,(p,conditional) in variants.items():
        compiler=ConditionalCompiler(p,conditional)
        for seed in PROTOCOL['test_seeds']:
            for load in PROTOCOL['loads']:
                env=JointEnv(prof,count=PROTOCOL['requests'],load=load);env.reset(seed=seed)
                refs=solve_reference(env.requests,compiler)
                references.append(dict(variant=variant,seed=seed,load=load,requests=len(env.requests),input_sha256=request_hash(env.requests),**refs))
        print(variant,sum(r['objective'] for r in references if r['variant']==variant),flush=True)
    original_refs=json.loads((OUT/'analysis.json').read_text())['offline_references']
    for old in original_refs:
        new=next(r for r in references if r['variant']=='current_global_15pct' and r['seed']==old['seed'] and r['load']==old['load'])
        if abs(old['objective']-new['objective'])>1e-6:raise ValueError('independent model reference mismatch')
    summaries={name:dict(optimal_model_count=sum(r['objective'] for r in references if r['variant']==name),requests=576,
        optimal_model_rate=sum(r['objective'] for r in references if r['variant']==name)/576) for name in variants}
    transfer=json.loads((OUT/'analysis.json').read_text())['model_transfer']
    ratios=[r['actual_tx']/r['empirical_allowance'] for r in transfer if r['empirical_allowance']>0]
    proof=dict(passed=True,source_sha256=sha(__file__),frozen_source_sha256=source_hashes(),
        comparison_sha256=sha(OUT/'comparison.json'),analysis_sha256=sha(OUT/'analysis.json'),
        stability_comparison_sha256=sha(stability_path) if stability_path.exists() else None,
        decision_file=decisions_path.name,decision_file_sha256=sha(decisions_path),counts=counts,
        rejection_binding_dimensions_nonexclusive=blocked,flexible_interface_selections=flexible,
        flexible_interface_selections_by_method=flexible_by_method,
        realized_return_q_diagnostics={key:dict(states=len(values),mean_bias=mean(values),
            mean_absolute_error=mean(abs(v) for v in values),rmse=float(np.sqrt(np.mean(np.square(values)))),
            scope='chosen Q versus one realized held-out return; not true expected-Q calibration, not a diagnosis of training exploration collapse')
            for key,values in q_errors.items()},
        conditional_fit_coefficients=maxima,conditional_holdout_checks=hold,
        envelope_sensitivity=summaries,offline_references=references,
        realized_total_tx_allowance_ratio=dict(minimum=min(ratios),median=float(np.median(ratios)),maximum=max(ratios)),
        scope='read-only diagnostic; envelope changes move the model optimum and are not additive attribution of the original online gap',
        not_identified=['physical SLA capacity bound','causal additive A/B/C loss shares','unseen-path candidate sufficiency',
                        'per-task counterfactual airtime for rejected requests','intrinsic online information gap'],
        caveats=['all comparison cases already use aligned ERP6 unicast; earlier native broadcast/PHY confounding cannot explain variation within this matrix',
                 'candidate catalogue is exhaustive only for fixed paths and three contracts; no candidate search-limit parameter exists',
                 'conditional profiles and zero-margin/zero-airtime variants are sensitivity experiments, not deployed guarantees',
                 'realized total PHY time is policy-dependent and non-additive under contention; not an ex-ante rejected-request coefficient'])
    write(OUT/'allocation_diagnostics.json',proof)
    print(json.dumps(dict(counts=counts,envelope_sensitivity=summaries,blocking=blocked),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
