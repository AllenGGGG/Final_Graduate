"""Finish development, freeze once, evaluate once, audit and publish v3.

Selection sees development rows only. This entrypoint and its selection rule are
bound into the protocol BEFORE any final test is opened.
"""
import collections
import concurrent.futures
import gzip
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from slice_alloc.tools import allocation_study as study
from slice_alloc.tools import allocation_admission as training
from slice_alloc.tools.patrol_run import sha,write

CANDIDATES=('prior_ppo','normalized_prior_ppo','normalized_admission_ppo',
            'paired_safe_ppo','paired_ppo','normalized_plain_ppo','plain_ppo')


def command(module,*args):
    print('RUN',module,*args,flush=True)
    subprocess.run([sys.executable,'-B','-m',module,*args],check=True,cwd=study.ROOT)


def parallel_audit():
    from stable_baselines3 import PPO
    from slice_alloc.tools import patrol_verify as auditor
    p=json.loads((study.OUT/'protocol.json').read_text())
    assert p['sources']==study.hashes()
    assert p['finalization_source_sha256']==sha(__file__)
    assert p['binary_sha256']==sha(study.NS3/'build/scratch/patrol-scenario')
    data=json.loads((study.OUT/'evaluation.json').read_text())
    assert data['protocol_sha256']==sha(study.OUT/'protocol.json') and data['sources']==study.hashes()
    analysis=json.loads((study.OUT/'analysis.json').read_text())
    assert analysis['evaluation_sha256']==sha(study.OUT/'evaluation.json')
    training.check_training_sources()
    auditor.OUT=study.OUT
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        reconstructed=list(pool.map(auditor.check_episode,data['rows']))
    grouped=collections.defaultdict(set)
    for r in data['rows']:grouped[r['seed'],r['load']].add(r['scenario_sha256'])
    assert len(grouped)==len(p['test_seeds'])*len(p['loads']) and all(len(v)==1 for v in grouped.values())
    replay_results=[]
    for c in p['checkpoints']:
        assert sha(study.OUT/c['checkpoint'])==c['model_sha256']
        r=next(x for x in data['rows'] if x['method']==c['method'] and x['initialization']==c['initialization'])
        prefix=study.OUT/'native'/r['path']
        original=json.loads(Path(str(prefix)+'-result.json').read_text())
        cfg=json.loads(Path(str(prefix)+'-scenario.json').read_text())
        model=PPO.load(study.OUT/c['checkpoint'],device='cpu')
        with tempfile.TemporaryDirectory(prefix='allocation-final-replay-') as tmp:
            out=Path(tmp)/'episode';again=study.run_episode(cfg,c['method'],model,out)
            for key in ('decisions','observations','settled'):assert again[key]==original[key]
            with gzip.open(str(out)+'-events.csv.gz','rb') as a,gzip.open(str(prefix)+'-events.csv.gz','rb') as b:
                assert a.read()==b.read()
        replay_results.append(dict(method=c['method'],initialization=c['initialization'],exact_events=True))
    write(study.OUT/'raw_reconstruction.json',reconstructed)
    write(study.OUT/'audit.json',dict(passed=True,episodes=len(data['rows']),
         decisions=sum(r['requests'] for r in data['rows']),sources=study.hashes(),
         same_inputs=True,raw_packet_sla=True,all_reservations_rechecked=True,
         exact_native_policy_replays=replay_results,verifier_sha256=sha(__file__),
         artifacts={n:sha(study.OUT/n) for n in ('protocol.json','evaluation.json','analysis.json',
                                              'raw_reconstruction.json','training_audit.json')}))
    print('All raw-packet, reservation, fingerprint and replay checks passed.',flush=True)


def main():
    assert not (study.OUT/'evaluation.json').exists(), 'Never repeat an opened final test.'
    expected=[study.OUT/'models'/f'{m}-{i}-32768.json' for m in CANDIDATES for i in study.INITS]
    while True:
        ready=[]
        for f in expected:
            try:json.loads(f.read_text());ready.append(f)
            except (FileNotFoundError,json.JSONDecodeError):pass
        if len(ready)==len(expected):break
        print('Waiting for complete fixed-budget models:',len(ready),'/',len(expected),flush=True)
        time.sleep(30)
    command('slice_alloc.tools.allocation_study','dev','--methods',*CANDIDATES,
            '--inits',*[str(i) for i in study.INITS],'--jobs','10')
    rows=json.loads((study.OUT/'development-32768.json').read_text())
    scores={m:sum(r['success'] for r in rows if r['method']==m)/
              sum(r['requests'] for r in rows if r['method']==m) for m in CANDIDATES}
    # Candidate order breaks exact ties in favor of the simpler unpaired prior.
    primary=max(CANDIDATES,key=scores.get)
    methods=list(dict.fromkeys([primary,'prior_ppo','plain_ppo','normalized_plain_ppo']))
    if primary in ('paired_ppo','normalized_admission_ppo'):methods.append('normalized_admission_ppo')
    if primary in ('paired_safe_ppo','normalized_prior_ppo'):methods.append('normalized_prior_ppo')
    methods=list(dict.fromkeys(methods))
    write(study.OUT/'selection.json',dict(primary_method=primary,development_scores=scores,
          rule='maximum complete five-initialization development SLA mean; predefined order breaks exact ties',
          final_methods=methods,final_test_not_opened=True,source_sha256=sha(__file__)))
    print('DEVELOPMENT SELECTION',primary,scores,flush=True)
    command('slice_alloc.tools.allocation_study','freeze','--methods',*methods,
            '--inits',*[str(i) for i in study.INITS])
    training.bind_training_sources()
    p=json.loads((study.OUT/'protocol.json').read_text())
    p.update(primary_method=primary,reference_ppo_method='normalized_plain_ppo' if
             primary.startswith(('normalized','paired')) else 'plain_ppo',
             finalization_source_sha256=sha(__file__),selection_sha256=sha(study.OUT/'selection.json'),
             calibration_high_sha256=sha(study.OUT/'load_calibration_high.json'),
             reserved_three_part_final_seeds=list(range(9000,9030)))
    write(study.OUT/'protocol.json',p)
    training.check_training_sources()
    command('slice_alloc.tools.allocation_study','evaluate','--jobs','10')
    command('slice_alloc.tools.allocation_study','analyze')
    parallel_audit()
    command('slice_alloc.tools.allocation_study','report')
    command('slice_alloc.tools.allocation_publish')
    delivery=json.loads((study.OUT/'delivery.json').read_text())
    delivery['finalization_source_sha256']=sha(__file__)
    delivery['training_audit_sha256']=sha(study.OUT/'training_audit.json')
    for k,v in delivery['artifacts'].items():assert sha(study.OUT/k)==v
    for k,v in delivery['manuscript_artifacts'].items():assert sha(study.ROOT/k)==v
    write(study.OUT/'delivery.json',delivery)
    print('COMPLETE: final results, full budget, raw SLA audit, manuscript and Word agree.',flush=True)


if __name__=='__main__':main()
