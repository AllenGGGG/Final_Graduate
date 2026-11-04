"""Recount arrivals and business outcomes directly by request ID.

This additional check reads original scenarios and settled contracts, rather
than trusting service summaries or assuming a request ID is an array index.
It does not select policies, change a metric, or rerun the simulator.
"""
from collections import Counter,defaultdict

from slice_alloc.tools import allocation_confirm_train as run


def main():
    rows=run.read(run.OUT/'evaluation.json');analysis=run.read(run.OUT/'analysis.json')
    counts=defaultdict(Counter);business=defaultdict(lambda:defaultdict(Counter));modes=defaultdict(Counter)
    for row in rows:
        prefix=run.OUT/'native'/f"{row['method']}-{row['init']}"/f"{row['load']}-{row['seed']}"
        cfg=run.read(str(prefix)+'-scenario.json');result=run.read(str(prefix)+'-result.json')
        requests={r['id']:r for r in cfg['requests']}
        decisions={d['request']:d for d in result['decisions']}
        assert len(requests)==len(cfg['requests'])==len(decisions)==row['requests']
        assert set(requests)==set(decisions)
        accepted={i for i,d in decisions.items() if d['decision']['accept']}
        settled={s['id']:s for s in result['settled']};assert set(settled)==accepted
        successful={i for i,s in settled.items() if s['success']}
        assert len(accepted)==row['admitted'] and len(successful)==row['success']
        by_service={s:dict(requests=sum(r['service']==s for r in requests.values()),
                           success=sum(requests[i]['service']==s for i in successful))
                    for s in ('telemetry','command','object')}
        assert row['by_service']==result['by_service']==by_service
        method=row['method'];counts[method].update(requests=len(requests),success=len(successful),admitted=len(accepted))
        for service,value in by_service.items():business[method][service].update(value)
        modes[method].update(decisions[i]['decision']['mode'] for i in accepted)
    for method,value in analysis['methods'].items():
        for key,total in counts[method].items():assert total==value[key]
        for service,total in business[method].items():assert dict(total)==value['by_service'][service]
        assert dict(modes[method])==value['modes']
        assert value['success_rate']==counts[method]['success']/counts[method]['requests']
        assert value['admission_rate']==counts[method]['admitted']/counts[method]['requests']
    run.write(run.OUT/'independent_counter_check.json',dict(
        verified_episodes=len(rows),request_id_lookup=True,all_service_counts_match=True,
        all_aggregate_counts_match=True,all_modality_counts_match=True,source_sha256=run.sha(__file__)))
    print('COUNTER_CHECK',len(rows),'all request, business and modality counts match')


if __name__=='__main__':main()
