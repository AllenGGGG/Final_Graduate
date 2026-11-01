"""Reconstruct current diagnostic evidence and regenerate the sole MD/DOCX."""
from collections import defaultdict
from pathlib import Path
import csv
import gzip
import io
import json
import unittest
from .joint_run import ROOT,OUT,write
from .platform_verify import sha,NS3


def read(path):return json.loads(Path(path).read_text())


def main():
    modules=['slice_alloc.tests.'+name for name in ('test_design_contracts','test_joint_contracts','test_joint_analysis','test_joint_diagnosis','test_report')]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(m) for m in modules)
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/'allocation-diagnostic-tests.log').write_text(stream.getvalue())
    if not tests.wasSuccessful():raise ValueError(stream.getvalue())
    d=read(OUT/'allocation_diagnostics.json');s=read(OUT/'stability_comparison.json');r=read(OUT/'rollout_comparison.json')
    training=read(OUT/'stability.json');cal=read(OUT/'calibration.json')
    comparison=read(OUT/'comparison.json');analysis=read(OUT/'analysis.json')
    assert analysis['source_sha256']==sha(ROOT/'slice_alloc/tools/joint_analyze.py')
    assert analysis['comparison_sha256']==sha(OUT/'comparison.json')
    assert d['comparison_sha256']==s['comparison_sha256']==r['comparison_sha256']==sha(OUT/'comparison.json')
    assert d['analysis_sha256']==sha(OUT/'analysis.json')
    assert d['source_sha256']==sha(ROOT/'slice_alloc/tools/joint_diagnose.py')
    assert r['source_sha256']==sha(ROOT/'slice_alloc/tools/joint_rollout.py') and r['diagnostic_source_sha256']==d['source_sha256']
    assert s['source_sha256']==training['source_sha256']==sha(ROOT/'slice_alloc/tools/joint_stability.py')
    assert s['training_sha256']==sha(OUT/'stability.json') and d['stability_comparison_sha256']==sha(OUT/'stability_comparison.json')
    assert sha(OUT/d['decision_file'])==d['decision_file_sha256']
    bycase=defaultdict(list);decisions=0
    with gzip.open(OUT/d['decision_file'],'rt') as f:
        for line in f:
            trace=json.loads(line);bycase[trace['case']].append(trace);decisions+=1
            assert trace['feasible_candidates']==sum(p['feasible'] for p in trace['envelope_slack'].values())
            assert trace['enumerated_candidates']==len(trace['allowed'])
            assert (trace['enumeration_status']=='infeasible')==(trace['feasible_candidates']==0)
            if trace['rejection_reason']=='no_candidate':assert trace['feasible_candidates']==0
            if trace['rejection_reason']=='policy':assert trace['feasible_candidates']>0
    assert decisions==d['counts']['decisions']==11520
    for row in s['episodes']:
        assert [(x['id'],x['selected']) for x in bycase[row['case']]]==[(x['id'],x['selected']) for x in row['decisions']]
    timing=read(OUT/'reject_timing_agreement.json')
    assert timing['passed'] and timing['source_sha256']==sha(ROOT/'slice_alloc/tools/joint_reject_timing.py')
    assert timing['stability_comparison_sha256']==sha(OUT/'stability_comparison.json')
    assert timing['rollout_comparison_sha256']==sha(OUT/'rollout_comparison.json')
    assert timing['rollout_config']==r['selected_config']
    assert timing['frozen_source_sha256']==training['frozen_source_sha256']
    expected_rejections={(case,x['id']):(x['method'],x['arrival'])
        for case,traces in bycase.items() for x in traces
        if x['method'].startswith('ddqn_') and x['rejection_reason']=='policy'}
    observed_rejections={(x['case'],x['id']):(x['method'],x['arrival']) for x in timing['agreement_rows']}
    assert len(timing['agreement_rows'])==len(observed_rejections) and observed_rejections==expected_rejections
    from .joint_reject_timing import summarize_agreement
    rates={f"ddqn_{x['initialization']}":x['success_rate'] for x in s['summary']['ddqn']['per_initialization']}
    summaries,aggregate=summarize_agreement(timing['agreement_rows'],rates)
    assert summaries==timing['per_initialization_agreement'] and aggregate==timing['aggregate']
    assert aggregate['policy_rejections']==sum(x['policy_rejected'] for x in s['summary']['ddqn']['per_initialization'])
    from .joint_analyze import reference_comparison
    load_stratification={}
    references=[ref for ref in d['offline_references'] if ref['variant']=='current_global_15pct']
    for load in sorted({row['load'] for row in s['episodes']}):
        refs=[ref for ref in references if ref['load']==load]
        methods={}
        for method in ('greedy','resource_price','adaptive_price','ddqn','ppo'):
            rows=[row for row in s['episodes'] if row['load']==load and
                  (row['method'].startswith(method+'_') if method in ('ddqn','ppo') else row['method']==method)]
            methods[method]=reference_comparison(rows,refs)
        load_stratification[str(load)]=dict(
            mean_arrival_interval_seconds=.035/load,
            unique_request_scenarios=len(refs),requests=sum(ref['requests'] for ref in refs),
            offline_model_count=sum(ref['objective'] for ref in refs),methods=methods)
    assert len(d['offline_references'])==120 and all(ref['optimal'] and ref['witness_interval_and_cache_check'] for ref in d['offline_references'])
    assert len(training['models'])==16 and training['new_training_steps']==327680 and training['total_training_steps']==524288
    # Revalidate the original diagnostic layer against CURRENT inputs rather
    # than retaining its old passing flag and stale source/result fingerprints.
    from .joint_analyze import failure_counts
    for method in ('first_fit','greedy','resource_price','adaptive_price','ddqn','ppo'):
        rows=[row for row in comparison['episodes'] if
              (row['method'].startswith(method+'_') if method in ('ddqn','ppo') else row['method']==method)]
        assert failure_counts(rows)==analysis['failure_decomposition'][method]
        assert reference_comparison(rows,analysis['offline_references'])==analysis['matched_references'][method]
    analysis_suite=unittest.defaultTestLoader.loadTestsFromName('slice_alloc.tests.test_joint_analysis')
    diagnostic_proof=dict(passed=True,tests=analysis_suite.countTestCases(),
        tests_sha256=sha(ROOT/'slice_alloc/tests/test_joint_analysis.py'),
        analysis_source_sha256=sha(ROOT/'slice_alloc/tools/joint_analyze.py'),
        analysis_sha256=sha(OUT/'analysis.json'),comparison_sha256=sha(OUT/'comparison.json'),
        optimal_reference_scenarios=len(analysis['offline_references']),
        unique_requested_slices=sum(ref['requests'] for ref in analysis['offline_references']),
        optimal_model_admissions=sum(ref['objective'] for ref in analysis['offline_references']),
        original_policy_and_packet_outputs_changed=False,
        scope='current all-scenario model-reference pairing and failure denominators; not a physical network bound')
    write(OUT/'diagnostic_verification.json',diagnostic_proof)
    artifacts={}
    def retain(p,expected=None):
        digest=sha(p)
        if expected is not None:assert digest==expected,str(p)
        artifacts[str(Path(p).relative_to(ROOT))]=digest
    for name in ('allocation_diagnostics.json','decision_diagnostics.jsonl.gz','stability.json','stability_comparison.json','rollout_comparison.json','reject_timing_agreement.json','allocation-diagnostic-tests.log','diagnostic_verification.json'):
        retain(OUT/name)
    for name in ('joint_diagnose.py','joint_stability.py','joint_rollout.py','joint_reject_timing.py','joint_audit.py','report.py'):
        retain(ROOT/'slice_alloc/tools'/name)
    for name in ('test_joint_diagnosis.py','test_joint_analysis.py','test_report.py'):retain(ROOT/'slice_alloc/tests'/name)
    retain(ROOT/'slice_alloc/design.md')
    for model in training['models']:retain(OUT/model['file'],model['sha256'])
    for name,digest in training['frozen_source_sha256'].items():assert sha(ROOT/name)==digest
    cases=cal['cases']+s['packet_cases']+r['packet_cases'];files=0
    assert len(cases)==612 and len({p['name'] for p in cases})==612
    for case in cases:
        for name,digest in case['files'].items():retain(OUT/name,digest);files+=1
        retain(OUT/case['input_file'],case['input_sha256'])
        tasks=defaultdict(list);observed=defaultdict(list)
        for line in (OUT/case['input_file']).read_text().splitlines():
            at,node,mode,name,start,end=line.split();tasks[int(node),name].append(float(at))
        with (OUT/'packets'/f"{case['name']}-events.csv").open() as f:
            for row in csv.DictReader(f):
                if row['event'] in ('app_data','object_ip_data'):
                    delay=float(row['detail']);observed[int(row['node']),row['name']].append((float(row['time'])-delay,delay))
        outcomes=defaultdict(lambda:[0,0,0])
        for key,starts in tasks.items():
            for at in starts:
                group=outcomes[key[1]];group[0]+=1
                matches=[i for i,(birth,_) in enumerate(observed[key]) if abs(birth-at)<1e-7]
                assert len(matches)<=1
                if matches:
                    _,delay=observed[key].pop(matches[0]);group[1]+=1;group[2]+=delay<=.150000001
        assert not any(observed.values())
        assert sum(v[0] for v in outcomes.values())==case['expected']
        assert sum(v[1] for v in outcomes.values())==case['delivered']
        assert sum(v[2] for v in outcomes.values())==case['on_time']
        assert sum(v[0]==v[2] for v in outcomes.values())==case['strict_slices']
    assert sha(NS3/'build/scratch/protocol-probe')==cal['platform']['binary_sha256']
    for name,digest in cal['platform']['library_sha256'].items():assert sha(NS3/'build/lib'/name)==digest
    proof=dict(passed=True,tests=tests.testsRun,artifact_sha256=artifacts,models=16,total_training_steps=524288,new_training_steps=327680,
        native_joint_cases_reconstructed=len(cases),raw_file_hashes_checked=files,decision_rows_checked=decisions,
        model_sensitivity_milps=120,new_native_comparison_runs=264,current_native_comparison_runs=504,
        rejection_timing_decisions_checked=aggregate['policy_rejections'],
        load_stratification=load_stratification,
        scope='read-only allocation diagnostics, unchanged-protocol eight-initialization extension and sampled online comparator; not complete thesis acceptance',
        unresolved=['no additive causal split of the online gap','no general path/function catalogue','no universal deadline service guarantee',
                    'no intrinsic online information-gap bound','no DQfD change without validated teacher design'])
    write(OUT/'allocation_diagnostic_verification.json',proof)
    from .report import main as report
    report()
    doc=read(ROOT/'slice_alloc/results/document_verification.json');assert doc['passed']
    assert doc['allocation_diagnostic_verification_sha256']==sha(OUT/'allocation_diagnostic_verification.json')
    for suffix in ('md','docx'):assert doc[suffix+'_sha256']==sha(ROOT/f'面向低空智联的多模态网络切片技术.{suffix}')
    regression=read(ROOT/'slice_alloc/results/platform/verification.json')
    failures=sum(c['strict_slices']<len(c['slices']) for c in s['packet_cases']+r['packet_cases'])
    sources=dict(training['frozen_source_sha256'])
    for relative in ('slice_alloc/tools/report.py','slice_alloc/tools/export_document.py',
                     'slice_alloc/tools/joint_audit.py','slice_alloc/tools/joint_analyze.py',
                     'slice_alloc/tools/joint_diagnose.py','slice_alloc/tools/joint_reject_timing.py',
                     'slice_alloc/design.md'):
        sources[relative]=sha(ROOT/relative)
    f=dict(passed=True,sources=sources,
        comparison_sha256=sha(OUT/'comparison.json'),analysis_sha256=sha(OUT/'analysis.json'),
        diagnostic_verification_sha256=sha(OUT/'diagnostic_verification.json'),
        native_joint_cases_reconstructed=len(cases),raw_file_hashes=files,models=len(training['models']),
        libraries=len(cal['platform']['library_sha256']),regression_cases=regression['cases'],
        regression_assertions=regression['assertions'],
        document_verification_sha256=sha(ROOT/'slice_alloc/results/document_verification.json'),
        allocation_diagnostic_verification_sha256=sha(OUT/'allocation_diagnostic_verification.json'),
        total_training_steps=training['total_training_steps'],new_training_steps=training['new_training_steps'],decision_rows_checked=decisions,
        three_initialization_sla_violation_cases=analysis['cases_with_sla_failure'],
        current_comparison_sla_violation_cases=failures,sla_violation_cases_retained=failures,
        scope='current 612 joint calibration/evaluation cases plus existing protocol regression; artifact and completion reconstruction, not universal SLA acceptance')
    write(OUT/'final_audit.json',f)
    # Check the saved manifest itself, including each source, before reporting.
    saved=read(OUT/'final_audit.json')
    for relative,digest in saved['sources'].items():assert sha(ROOT/relative)==digest,relative
    for name in ('comparison','analysis','diagnostic_verification','allocation_diagnostic_verification'):
        assert saved[name+'_sha256']==sha(OUT/(name+'.json')),name
    assert saved['document_verification_sha256']==sha(ROOT/'slice_alloc/results/document_verification.json')
    print(json.dumps({k:v for k,v in proof.items() if k!='artifact_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
