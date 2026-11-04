"""Stronger baseline selection and fresh test, registered before opening test."""
import argparse
import concurrent.futures as cf
import json
import shutil
import sys
from pathlib import Path
import torch
from slice_alloc import guided_allocation as guided
from slice_alloc.tools import allocation_improve_run as run


def rule_validation(parameters):
    rows=[run.episode(s,l,'modal_greedy',parameters) for s in run.VALIDATION for l in run.LOADS]
    result=dict(parameters=parameters,rows=rows,
        rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows))
    name='-'.join(str(x) for x in parameters)
    run.write(run.OUT/'baseline_validation'/f'{name}.json',result)
    print('RULE VALIDATION',parameters,result['rate'],flush=True)
    return result


def evaluate_group(spec):
    method,init=spec;torch.set_num_threads(1)
    # Policy input prior remains exactly its training-time rule; strong baseline
    # gets the best of the full grid on validation, independently of that prior.
    prior=run.read(run.OUT/'diagnosis.json')['selected']['parameters']
    chosen=run.read(run.OUT/'baseline_selection.json')['selected']['parameters']
    parameters=chosen if method=='modal_greedy' else prior
    if method=='guided':
        model=guided.GuidedDQN.load(run.OUT/'models'/f'guided-{init}.zip',device='cpu')
    elif method=='prior':
        model=guided.GuidedDQN.load(run.OUT/'training'/str(run.INITS[0])/'0.zip',device='cpu')
    else:model=None
    return [run.episode(s,l,method,parameters,model,trace=True,init=init) for s in run.TEST for l in run.LOADS]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('baseline','evaluate'));parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();run.check()
    registration=dict(source_sha256=run.sha(__file__),protocol_sha256=run.sha(run.OUT/'protocol.json'),
        full_grid_validation=True,validation=list(run.VALIDATION),
        primary='guided RL minus best of all24 modal-aware rule formulas on validation',
        prior='development-selected rule stays fixed in all RL input observations',
        test=list(run.TEST),loads=list(run.LOADS),initializations=list(run.INITS),
        ablation='prior: exact zero-step policy on same scenes, to isolate learning from rule initialization',
        expected_episodes=1680,
        interpretation='additional rule tuning and zero-step ablation declared before testing; strengthens comparison')
    path=run.OUT/'evaluation_registration.json'
    if path.exists():assert run.read(path)==registration
    else:run.write(path,registration)
    if args.command=='baseline':
        assert not(run.OUT/'test_started.json').exists()
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(rule_validation,run.GRID))
        run.write(run.OUT/'baseline_selection.json',dict(candidates=rows,selected=max(rows,key=lambda x:x['rate'])))
        return
    training=run.read(run.OUT/'training_complete.json');assert len(training)==len(run.INITS)
    assert {r['init'] for r in training}==set(run.INITS)
    for row in training:
        assert row['steps']==65536 and row['updates']>0
        assert run.sha(run.OUT/row['model'])==row['sha256']
    gate=run.OUT/'test_started.json';assert not gate.exists(),'already opened formal test'
    log=(run.OUT/'full-tests.log').read_text();assert '\nOK\n' in log and '\nFAILED' not in log
    run.write(gate,{name:run.sha(run.OUT/name) for name in (
        'protocol.json','training_complete.json','diagnosis.json','baseline_selection.json',
        'evaluation_registration.json','full-tests.log')})
    specs=[('guided',i) for i in run.INITS]+[(m,0) for m in (*run.RULES,'prior')]
    rows=[]
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for group in pool.map(evaluate_group,specs):
            rows.extend(group);print('AUDITED',len(rows),1680,flush=True)
    run.write(run.OUT/'evaluation.json',rows)
    for seed in run.TEST:
        for load in run.LOADS:
            assert len({r['scenario_sha256'] for r in rows if r['seed']==seed and r['load']==load})==1
    run.report(rows)


if __name__=='__main__':main()
