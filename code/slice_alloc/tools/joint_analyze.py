"""Descriptive held-out diagnostics; never selects policies or retunes inputs."""
from statistics import mean
import json
import numpy as np
from ..joint import JointEnv,request_hash
from .joint_run import OUT,PROTOCOL,profile,offline_reference,write
from .platform_verify import sha


def failure_counts(rows):
    """Mutually exclusive counts, all rates use requests unless named otherwise."""
    requested=accepted=successful=no_candidate=policy_rejected=0
    for row in rows:
        n=row['requests'];a=row['accepted'];s=round(row['packet_success']*n)
        if abs(s-row['packet_success']*n)>1e-7 or not 0<=s<=a<=n:
            raise ValueError('invalid integer slice outcomes')
        if row['rejected_no_candidate']+row['policy_rejected']+a!=n:
            raise ValueError('rejection causes do not partition requests')
        requested+=n;accepted+=a;successful+=s
        no_candidate+=row['rejected_no_candidate'];policy_rejected+=row['policy_rejected']
    if not requested:raise ValueError('empty outcome group')
    violated=accepted-successful;failed=requested-successful
    return dict(requests=requested,accepted=accepted,successful=successful,
        no_candidate=no_candidate,policy_rejected=policy_rejected,admitted_violations=violated,
        failed=failed,no_candidate_rate=no_candidate/requested,policy_rejected_rate=policy_rejected/requested,
        admitted_violation_request_rate=violated/requested,success_rate=successful/requested,
        admitted_success_rate=successful/accepted if accepted else None,
        rejection_share_of_failures=(no_candidate+policy_rejected)/failed if failed else None,
        scope='no-candidate is conditional on the online allocation history, not an irreducible capacity bound; active rejection alone does not establish future-resource planning')


def reference_comparison(rows,references):
    """Match scenario inputs before normalizing against an optimal MODEL count."""
    lookup={}
    for ref in references:
        key=(ref['seed'],ref['load'])
        if key in lookup:raise ValueError('duplicate reference scenario')
        if not ref['optimal']:raise ValueError('unproven optimum cannot be used as an exact reference')
        lookup[key]=ref
    requested=oracle=admitted=successful=0
    by_method={}
    for row in rows:
        key=(row['seed'],row['load']);ref=lookup[key]
        if row['input_sha256']!=ref['input_sha256'] or row['requests']!=ref['requests']:
            raise ValueError('online and offline request contracts differ')
        method=by_method.setdefault(row['method'],set())
        if key in method:raise ValueError('duplicate online scenario')
        method.add(key)
        if row['accepted']>ref['objective']+1e-6:raise ValueError('online model exceeds claimed optimal reference')
        requested+=row['requests'];oracle+=ref['objective'];admitted+=row['accepted']
        successful+=round(row['requests']*row['packet_success'])
    if not rows or any(keys!=set(lookup) for keys in by_method.values()):
        raise ValueError('reference coverage must match all scenarios for each initialization')
    return dict(scenarios=len(lookup),policy_runs=len(rows),requests_with_policy_repetition=requested,
        reference_model_count_with_policy_repetition=oracle,reference_model_rate=oracle/requested,
        model_acceptance=admitted/requested,packet_success=successful/requested,
        model_fraction_of_offline_model=admitted/oracle,packet_fraction_of_offline_model=successful/oracle,
        online_model_gap_request_rate=(oracle-admitted)/requested,
        execution_loss_request_rate=(admitted-successful)/requested,
        scope='offline reference is optimal only for the frozen reservation model; packet/reference ratio is descriptive, not a physical SLA bound or an online approximation guarantee')


def main():
    data=json.loads((OUT/'comparison.json').read_text());prof=profile()
    rows=data['episodes'];rng=np.random.default_rng(91173)
    def family(label):
        return [r for r in rows if r['method'].startswith(label+'_')] if label in ('ddqn','ppo') else [r for r in rows if r['method']==label]
    paired={}
    for treatment,baseline in [(a,b) for a in ('ddqn','ppo') for b in ('greedy','resource_price','adaptive_price')]+[('ddqn','ppo')]:
        differences=[]
        for seed in PROTOCOL['test_seeds']:
            differences.append(mean(r['packet_success'] for r in family(treatment) if r['seed']==seed)
                -mean(r['packet_success'] for r in family(baseline) if r['seed']==seed))
        boot=np.mean(rng.choice(differences,(10000,len(differences)),replace=True),axis=1)
        paired[treatment+'-'+baseline]=dict(mean=mean(differences),low=float(np.quantile(boot,.025)),
            high=float(np.quantile(boot,.975)),seeds=len(differences),differences=differences)
    references=[]
    for seed in PROTOCOL['test_seeds']:
        for load in PROTOCOL['loads']:
            env=JointEnv(prof,count=PROTOCOL['requests'],load=load);env.reset(seed=seed)
            ref=dict(seed=seed,load=load,input_sha256=request_hash(env.requests),**offline_reference(env.requests,prof))
            references.append(ref)
            print(json.dumps(dict(seed=seed,load=load,objective=ref['objective'],optimal=ref['optimal'])),flush=True)
    labels=['first_fit','greedy','resource_price','adaptive_price','ddqn','ppo']+sorted({r['method'] for r in rows if r['method'].startswith(('ddqn_','ppo_'))})
    failure_decomposition={label:failure_counts(family(label)) for label in labels}
    matched_references={label:reference_comparison(family(label),references) for label in labels}
    by_load={str(load):{label:failure_counts([r for r in family(label) if r['load']==load]) for label in labels}
             for load in PROTOCOL['loads']}
    initialization_summary={}
    for label in ('ddqn','ppo'):
        values=[failure_decomposition[f'{label}_{i}']['success_rate'] for i in PROTOCOL['initializations']]
        initialization_summary[label]=dict(mean=mean(values),sample_std=float(np.std(values,ddof=1)),
            minimum=min(values),maximum=max(values),initializations=PROTOCOL['initializations'],rates=values,
            scope='sample standard deviation across three frozen initializations, not a confidence interval or a test-selection criterion')
    by_name={p['name']:p for p in data['packet_cases']}
    transfer=[]
    for r in rows:
        packet=by_name[r['case']]
        rounds=sum(d['rounds'] for d in r['decisions'] if d['selected'])
        allowance=rounds*prof['airtime_seconds_per_pair_round']
        transfer.append(dict(case=r['case'],model_admitted=r['accepted'],actual_success=packet['strict_slices'],
            actual_tx=packet['radio']['tx_seconds_from_state'],empirical_allowance=allowance,
            allowance_exceeded=packet['radio']['tx_seconds_from_state']>allowance+1e-9,
            mac_retries=packet['radio']['link_layer']['business_mac_retries']))
    proof=dict(role='descriptive post-evaluation diagnostics; no policy selection or parameter retuning',
        source_sha256=sha(__file__),comparison_sha256=sha(OUT/'comparison.json'),paired=paired,
        offline_references=references,model_transfer=transfer,
        failure_decomposition=failure_decomposition,matched_references=matched_references,
        failure_decomposition_by_load=by_load,initialization_summary=initialization_summary,
        cases_with_sla_failure=sum(r['actual_success']<r['model_admitted'] for r in transfer),
        cases_exceeding_empirical_tx_allowance=sum(r['allowance_exceeded'] for r in transfer),
        statistical_scope='eight independent request seeds, loads and initialization averaged within seed; descriptive bootstrap intervals, no familywise significance claim')
    write(OUT/'analysis.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('model_transfer','paired')},ensure_ascii=False,indent=2),flush=True)
    print(json.dumps(paired,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
