"""Independent packet audits and paired statistics for the five-seed RL study."""
import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import torch

from slice_alloc.allocation_credit import CreditEnv, native_action
from slice_alloc.allocation_mc import MonteCarloDictReplay
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.tools import allocation_confirm_train as run
from slice_alloc.tools import allocation_simple_report as audit

RULES = ('greedy','nearest','first_fit','native')
LEARNED = ('rl','initial_rl')


def action_digest(decisions):
    return hashlib.sha256(json.dumps(decisions,sort_keys=True).encode()).hexdigest()


def rule_action(method,env):
    if not env.options:return 0
    cost=lambda i:float(np.sum(env.options[i].peak/env.ledger.capacity))
    if method=='native':return native_action(env)
    if method=='first_fit':return 1
    if method=='greedy':return min(range(len(env.options)),key=cost)+1
    if method=='nearest':
        r=env.requests[env.current['request_id']]
        aps=[env.current['associations'][str(n)]-2 for n in r.members]
        return min(range(len(env.options)),key=lambda i:(sum(env.options[i].edge!=ap for ap in aps),cost(i)))+1
    raise ValueError(method)


def model_path(method,init):
    return run.OUT/('models' if method=='rl' else 'reference_models')/f'double_dqn-{init}.zip'


def episode(seed,load,method,init=0,model=None,trace=True):
    cfg=run.environment.scenario(seed,load)
    prefix=run.OUT/'native'/f'{method}-{init}'/f'{load}-{seed}' if trace else None
    env=CreditEnv(progress_weight=0.)
    try:
        obs,_=env.reset(options=dict(scenario=cfg,prefix=prefix,trace=trace))
        done=False;reward=0.
        while not done:
            action=int(model.predict(obs,deterministic=True)[0]) if model is not None else rule_action(method,env)
            assert obs['mask'][action]>.5
            obs,r,done,truncated,_=env.step(action);assert not truncated;reward+=r
        result=env.result();assert reward==result['success']
        scenario_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        decision_hash=action_digest(result['decisions'])
        result.update(scenario_sha256=scenario_hash,action_sha256=decision_hash)
        row=dict(method=method,init=init,seed=seed,load=load,action_sha256=decision_hash,
                 **{k:result[k] for k in ('requests','success','admitted','by_service')})
        if not trace:return row
        run.write(str(prefix)+'-result.json',result)
        raw=Path(str(prefix)+'-events.csv')
        with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
        raw.unlink()
        run.write(str(prefix)+'-metrics.json',dict(**row,scenario_sha256=scenario_hash))
        return audit.verify_episode(run.OUT,row)
    finally:env.close()


def specifications():
    return [(m,i) for m in LEARNED for i in run.INITS]+[(m,0) for m in RULES]


def expected_keys():
    return {(m,i,s,l) for m,i in specifications() for s in run.TEST for l in run.LOADS}


def tests():
    """A log is evidence only after the real test process exits successfully."""
    marker=run.OUT/'tests_passed.json'
    tests_hash={str(p.relative_to(run.PKG)):run.sha(p) for p in sorted((run.PKG/'tests').glob('test_*.py'))}
    if marker.exists():
        prior=run.read(marker)
        if prior['test_sources']==tests_hash and prior['log_sha256']==run.sha(run.OUT/'full-tests.log'):
            assert '\nOK\n' in (run.OUT/'full-tests.log').read_text()
            return prior
    assert not (run.OUT/'test_started.json').exists(),'Cannot change verification tests after opening the final test'
    with (run.OUT/'full-tests.log').open('w') as log:
        proc=subprocess.run([sys.executable,'-m','unittest','discover','-s','slice_alloc/tests','-p','test_*.py','-v'],
                            cwd=run.PKG.parent,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError('Code tests failed; see full-tests.log. No completion marker written.')
    log=(run.OUT/'full-tests.log').read_text();assert '\nOK\n' in log
    value=dict(test_sources=tests_hash,log_sha256=run.sha(run.OUT/'full-tests.log'),
               passed_tests=int(re.search(r'Ran (\d+) tests',log)[1]))
    run.write(marker,value)
    return value


def prepare_artifacts():
    """Repair pilot serialization metadata before registration, preserving tensors.

    Pilot training used a 50k replay object, but its saved buffer_size attribute
    still said 20k. Inference weights and training results are unchanged. Keep
    the original serialized files and record every conversion before testing.
    """
    marker=run.OUT/'artifact_metadata_repair.json'
    if marker.exists():return
    assert not (run.OUT/'test_started.json').exists()
    completion=run.read(run.OUT/'training_complete.json');changes=[]
    for row in completion['models']:
        if row['init'] not in (11,23):continue
        for checkpoint in row['checkpoints']:
            path=run.OUT/checkpoint['path'];assert run.sha(path)==checkpoint['sha256']
            model=MaskedDoubleDQN.load(path,device='cpu')
            if model.buffer_size==run.PARAMETERS['buffer_size']:continue
            original=run.OUT/'provenance/serialized-before-metadata-fix'/checkpoint['path']
            original.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,original)
            before={n:t.clone() for n,t in model.policy.state_dict().items()}
            previous=model.buffer_size;model.buffer_size=run.PARAMETERS['buffer_size'];model.save(path)
            loaded=MaskedDoubleDQN.load(path,device='cpu')
            for name,tensor in before.items():torch.testing.assert_close(tensor,loaded.policy.state_dict()[name],atol=0,rtol=0)
            changes.append(dict(path=checkpoint['path'],before_sha256=checkpoint['sha256'],after_sha256=run.sha(path),
                                old_serialized_buffer_size=previous,actual_training_buffer_size=model.buffer_size,
                                all_policy_tensors_identical=True,original=str(original.relative_to(run.OUT))))
            checkpoint['sha256']=run.sha(path)
        row['selected']=max(row['checkpoints'],key=lambda c:(c['validation']['rate'],-c['step']))
        shutil.copy2(run.OUT/row['selected']['path'],run.OUT/row['model'])
        row['model_sha256']=run.sha(run.OUT/row['model'])
        run.write(run.OUT/'training'/str(row['init'])/'training.json',row)
    run.write(run.OUT/'training_complete.json',completion)
    run.write(marker,dict(reason='correct saved replay capacity to match actual pilot replay; no policy or target changes',changes=changes))


def check_training():
    data=run.read(run.OUT/'training_complete.json')
    assert data['protocol_sha256']==run.sha(run.OUT/'protocol.json')
    rows=data['models'];assert len(rows)==len(run.INITS) and {r['init'] for r in rows}==set(run.INITS)
    actual=[];torch.set_num_threads(1)
    for row in rows:
        assert row['selected']==max(row['checkpoints'],key=lambda c:(c['validation']['rate'],-c['step']))
        assert row['steps']==max(run.BUDGETS) and row['updates']>0 and row['returns_committed']>0
        for c in row['checkpoints']:
            assert run.sha(run.OUT/c['path'])==c['sha256']
            validation=c['validation']['rows']
            assert len(validation)==len(run.VALIDATION)*len(run.LOADS)
            assert {(r['seed'],r['load']) for r in validation}=={(s,l) for s in run.VALIDATION for l in run.LOADS}
            assert c['validation']['rate']==sum(r['success'] for r in validation)/sum(r['requests'] for r in validation)
        assert run.sha(run.OUT/row['model'])==row['model_sha256']
        model=MaskedDoubleDQN.load(run.OUT/row['model'],device='cpu')
        assert model.replay_buffer_class is MonteCarloDictReplay and model.replay_buffer_kwargs=={}
        assert model.buffer_size==run.PARAMETERS['buffer_size']
        assert model.n_quantiles==1 and model.gamma==1. and model.learning_rate==run.PARAMETERS['learning_rate']
        assert model.batch_size==run.PARAMETERS['batch_size'] and model.train_freq.frequency==run.PARAMETERS['train_freq']
        initial=MaskedDoubleDQN.load(model_path('initial_rl',row['init']),device='cpu')
        if row['selected']['step']==0:
            for n,t in model.policy.state_dict().items():torch.testing.assert_close(t,initial.policy.state_dict()[n],atol=0,rtol=0)
        else:assert model.num_timesteps==row['selected']['step']
        if row['training_seeds'] is not None:assert all(81000000<=s<82000000 for s in row['training_seeds'])
        actual.append(dict(init=row['init'],pretrained_steps=initial.num_timesteps,
                           selected_additional_steps=row['selected']['step'],model_sha256=row['model_sha256'],
                           origin=row['origin'],replay_class='MonteCarloDictReplay'))
    return actual


def register():
    run.check();prepare_artifacts();check_training();test_record=tests()
    value=dict(source_sha256=run.sha(__file__),audit_sha256=run.sha(audit.__file__),
               protocol_sha256=run.sha(run.OUT/'protocol.json'),tests_sha256=run.sha(run.OUT/'tests_passed.json'),
               artifact_metadata_repair_sha256=run.sha(run.OUT/'artifact_metadata_repair.json'),
               entry_sha256=run.sha(run.PKG/'__main__.py'),
               test=list(run.TEST),loads=list(run.LOADS),initializations=list(run.INITS),
               expected_episodes=len(expected_keys()),inference_probes=6*len(run.INITS),
               primary='rl-greedy; paired 95% bootstrap interval lower>0',
               full_publication='all rule comparison intervals positive and both violation measures no worse than Greedy',
               secondary='Holm-adjusted approximate bootstrap p-values; no claim from an unadjusted interval alone',
               bootstrap_seed=20261107,bootstrap_draws=20000,
               exploratory_history='two pilot inits previously tested on 445000..445059; no pilot test cases pooled here')
    path=run.OUT/'evaluation_registration.json'
    if path.exists():assert run.read(path)==value
    else:
        assert not (run.OUT/'test_started.json').exists()
        run.write(path,value)
    return test_record


def test_snapshot():
    names=('protocol.json','training_complete.json','evaluation_registration.json','tests_passed.json','full-tests.log','artifact_metadata_repair.json')
    return {n:run.sha(run.OUT/n) for n in names}


def group(spec):
    method,init=spec;torch.set_num_threads(1)
    model=MaskedDoubleDQN.load(model_path(method,init),device='cpu') if method in LEARNED else None
    rows=[]
    for seed in run.TEST:
        for load in run.LOADS:rows.append(episode(seed,load,method,init,model))
    return rows


def evaluate(workers):
    register();gate=run.OUT/'test_started.json';snapshot=test_snapshot()
    if gate.exists():assert run.read(gate)==snapshot,'Frozen inputs changed; refusing to resume'
    else:run.write(gate,snapshot)
    if (run.OUT/'evaluation.json').exists():return
    rows=[]
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for batch in pool.map(group,specifications()):
            rows.extend(batch);print('AUDITED',len(rows),len(expected_keys()),flush=True)
    assert len(rows)==len(expected_keys())
    run.write(run.OUT/'evaluation.json',rows)


def analyze(rows):
    lookup={(r['method'],r['init'],r['seed'],r['load']):r for r in rows}
    assert len(lookup)==len(rows)==len(expected_keys()) and set(lookup)==expected_keys()
    rng=np.random.default_rng(20261107)
    scenes=rng.integers(len(run.TEST),size=(20000,len(run.TEST)))
    inits=rng.integers(len(run.INITS),size=(20000,len(run.INITS)))
    out=dict(methods={},comparisons={},per_initialization={},per_load={},
             interval_method='20000 paired bootstrap resamples of scenes and initializations; all three loads are one scene cluster')
    boots={}
    for method in (*LEARNED,*RULES):
        initializations=(0,) if method in RULES else run.INITS
        matrix=np.array([[sum(lookup[method,i,s,l]['success'] for l in run.LOADS)/sum(run.LOADS)
                          for s in run.TEST] for i in initializations])
        boots[method]=matrix[0,scenes].mean(1) if method in RULES else matrix[inits[:,:,None],scenes[:,None,:]].mean((1,2))
        rr=[r for r in rows if r['method']==method];summary=audit.summarize(rr)
        summary['violation_rate_admitted']=summary['admitted_violations']/summary['admitted'] if summary['admitted'] else 0.
        summary['minimum_service_success']=min(s['success']/s['requests'] for s in summary['by_service'].values())
        summary['ci95']=np.quantile(boots[method],[.025,.975]).tolist();out['methods'][method]=summary
        out['per_load'][method]={str(l):audit.summarize([r for r in rr if r['load']==l]) for l in run.LOADS}
        if method in LEARNED:out['per_initialization'][method]={str(i):audit.summarize([r for r in rr if r['init']==i]) for i in run.INITS}
    for method in ('initial_rl',*RULES):
        difference=out['methods']['rl']['success_rate']-out['methods'][method]['success_rate']
        distribution=boots['rl']-boots[method]
        out['comparisons']['rl-'+method]=dict(difference=difference,ci95=np.quantile(distribution,[.025,.975]).tolist(),
            centered_bootstrap_p=float((1+np.count_nonzero(np.abs(distribution-difference)>=abs(difference)))/(len(distribution)+1)))
    secondary=[(k,v) for k,v in out['comparisons'].items() if k!='rl-greedy'];running=0.
    for rank,(k,v) in enumerate(sorted(secondary,key=lambda item:item[1]['centered_bootstrap_p'])):
        running=max(running,min(1.,(len(secondary)-rank)*v['centered_bootstrap_p']));v['holm_adjusted_p']=running
    rl=out['methods']['rl'];greedy=out['methods']['greedy']
    gates={m+'_ci_positive':out['comparisons']['rl-'+m]['ci95'][0]>0 for m in RULES}
    gates.update(violation_all_not_worse=rl['violation_rate_all']<=greedy['violation_rate_all'],
                 violation_admitted_not_worse=rl['violation_rate_admitted']<=greedy['violation_rate_admitted'])
    out['publication_gates']=gates;out['full_publication_passed']=all(gates.values())
    out['primary_comparison_passed']=gates['greedy_ci_positive'] and gates['violation_all_not_worse'] and gates['violation_admitted_not_worse']
    return out


def audit_row(row):
    extra=('forced_reject','voluntary_reject','admitted_violations','modes','scenario_sha256')
    verified=audit.verify_episode(run.OUT,{k:v for k,v in row.items() if k not in extra})
    assert verified==row
    prefix=run.OUT/'native'/f"{row['method']}-{row['init']}"/f"{row['load']}-{row['seed']}"
    assert action_digest(run.read(str(prefix)+'-result.json')['decisions'])==row['action_sha256']
    return verified


def probe(init):
    torch.set_num_threads(1);model=MaskedDoubleDQN.load(model_path('rl',init),device='cpu')
    return [episode(s,l,'rl',init,model,False) for s in (run.TEST[0],run.TEST[-1]) for l in run.LOADS]


def render(analysis,verification):
    lines=['# 第一部分：五个初始化的独立验证','',
           '同一低空巡检平台、IP/EID-RLOC/NDN、11槽掩码动作。主方法使用预训练独立Double-DQN的候选编码网络，继续用完整回合真实回报训练。续训阶段为蒙特卡洛动作价值控制，终局目标不含Double-Q自举；不把它误称为新的Double-DQN算法。','',
           '测试种子447000—447059，32/64/96请求；五个初始化全部保留，验证集选检查点。两个初始化的先导测试445000—445059已公开，未并入本批；本批是这轮开发之后的独立确认，不宣称没有过往试验。','',
           '| 方法 | SLA成功率 | 95%区间 | 接纳率 | 违约/全部到达 | 违约/已接纳 |','|---|---:|---:|---:|---:|---:|']
    for name,r in analysis['methods'].items():
        lo,hi=r['ci95'];lines.append(f"| {name} | {r['success_rate']:.3%} | [{lo:.3%},{hi:.3%}] | {r['admission_rate']:.3%} | {r['violation_rate_all']:.3%} | {r['violation_rate_admitted']:.3%} |")
    lines+=['','| 配对比较 | 差值（百分点） | 95%区间（百分点） | Holm校正近似p |','|---|---:|---:|---:|']
    for name,r in analysis['comparisons'].items():
        lo,hi=r['ci95'];p=f"{r['holm_adjusted_p']:.4f}" if 'holm_adjusted_p' in r else '主比较未校正'
        lines.append(f"| {name} | {100*r['difference']:+.3f} | [{100*lo:+.3f},{100*hi:+.3f}] | {p} |")
    lines+=['',f"主比较通过：{analysis['primary_comparison_passed']}；完整发布条件通过：{analysis['full_publication_passed']}。",'',
            f"{verification['verified_native_episodes']}个原生回合逐包SLA、方向、预约重放全部通过；{verification['inference_probes']}次模型重载重放一致；{verification['passed_tests']}项代码测试通过。",'',
            '所有拒绝都计入成功率分母。违约按全部到达和已接纳分别统计；各业务、各负载、每个初始化及最低业务成功率见analysis.json。总体改善不等于每类业务均改善，不能省略遥测等弱项。', '',
            '切片在本部分指准入预约和配置，未实现逐切片确定性带宽隔离；EID-RLOC为简化实现。不同历史工作负载不直接比较。未比较或曾经表现更强的规则不能声称已经被超过。']
    (run.OUT/'report.md').write_text('\n'.join(lines)+'\n')


def verify(workers):
    test_record=register();assert run.read(run.OUT/'test_started.json')==test_snapshot()
    actual=check_training();rows=run.read(run.OUT/'evaluation.json')
    keys={(r['method'],r['init'],r['seed'],r['load']) for r in rows}
    assert len(rows)==len(expected_keys()) and keys==expected_keys()
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for n,_ in enumerate(pool.map(audit_row,rows),1):
            if n%180==0:print('VERIFIED',n,len(rows),flush=True)
        probes=[r for batch in pool.map(probe,run.INITS) for r in batch]
    lookup={(r['method'],r['init'],r['seed'],r['load']):r for r in rows}
    for row in probes:
        original=lookup[row['method'],row['init'],row['seed'],row['load']]
        for k in ('action_sha256','success','admitted','by_service'):assert row[k]==original[k]
    for s in run.TEST:
        for l in run.LOADS:assert len({r['scenario_sha256'] for r in rows if r['seed']==s and r['load']==l})==1
    analysis=analyze(rows)
    verification=dict(verified_native_episodes=len(rows),inference_probes=len(probes),actual_models=actual,
        all_packet_audits_pass=True,all_direction_audits_pass=True,all_reservation_replays_pass=True,
        all_methods_share_scenarios=True,all_initializations_reported=True,new_independent_test=True,
        passed_tests=test_record['passed_tests'],primary_comparison_passed=analysis['primary_comparison_passed'],
        full_publication_passed=analysis['full_publication_passed'],publication_gates=analysis['publication_gates'])
    run.write(run.OUT/'analysis.json',analysis);run.write(run.OUT/'verification.json',verification);render(analysis,verification)
    print('FINAL',analysis['primary_comparison_passed'],analysis['full_publication_passed'],analysis['comparisons'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('register','evaluate','verify'));p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    if args.command=='register':register()
    elif args.command=='evaluate':evaluate(args.workers)
    else:verify(args.workers)


if __name__=='__main__':main()
