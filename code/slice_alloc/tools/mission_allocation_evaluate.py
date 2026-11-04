"""Registered fresh testing, packet replay and policy inference verification."""
import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import shutil
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from slice_alloc import mission_allocation as mission
from slice_alloc.tools import mission_allocation_run as run
from slice_alloc.tools import allocation_simple_report as audit
from slice_alloc.tools import mission_episode_audit

RULES=('greedy','nearest','first_fit','native','modal_greedy','price','prior')


def audit_row(row):
    # The shared auditor reconstructs these fields, including the scenario hash.
    reconstructed=('scenario_sha256','forced_reject','voluntary_reject','admitted_violations','modes')
    core={k:v for k,v in row.items() if k not in reconstructed}
    verified=mission_episode_audit.verify_episode(run.OUT,core)
    for key in reconstructed:
        if key in row:assert row[key]==verified[key],key
    return verified


def register():
    run.check()
    registration=dict(source_sha256=run.sha(__file__),protocol_sha256=run.sha(run.OUT/'protocol.json'),
        test=list(run.TEST),loads=list(run.LOADS),initializations=list(run.INITS),expected_episodes=2700,
        primary='PPO versus validation-selected strongest rule family, including admission price',
        ablation='exact zero-step prior policy on same scenes',
        bootstrap='20000 paired draws of scenario clusters and RL initializations; all three loads remain together',
        guardrails='primary and learned-vs-prior confidence lower bound positive; conditional and arrival violation rates no worse than primary rule',
        family_selection='one complete8init family selected on validation; see family_selection_registration.json and family_selection.json',
        inference_probes='rerun first and last test seed, each load and initialization; compare exact decisions and totals',
        secondary='exploratory centered bootstrap p-values, Holm correction')
    path=run.OUT/'evaluation_registration.json'
    if path.exists():
        previous=run.read(path)
        if previous!=registration:
            amendment=run.read(run.OUT/'evaluation_audit_amendment.json')
            assert amendment['original_registration_sha256']==run.sha(path)
            assert amendment['original_source_sha256']==previous['source_sha256']
            assert amendment['repaired_source_sha256']==registration['source_sha256']
            assert amendment['audit_adapter_sha256']==run.sha(mission_episode_audit.__file__)
            assert {k:v for k,v in previous.items() if k!='source_sha256'}=={k:v for k,v in registration.items() if k!='source_sha256'}
    else:run.write(path,registration)


def choose_config(method):
    prior=run.read(run.OUT/'diagnosis.json')['selected']
    if method in ('ppo','prior'):return prior
    rows=run.read(run.OUT/'baseline_selection.json')['candidates']
    return max((r for r in rows if r['method']==method),key=lambda r:r['rate'])


def episode(seed,load,method,init=0,model=None,trace=True):
    selected=choose_config(method);parameters=selected['parameters'];threshold=selected['threshold']
    cfg=mission.scenario(seed,load)
    prefix=run.OUT/'native'/f'{method}-{init}'/f'{load}-{seed}' if trace else None
    env=mission.MissionEnv(rule_parameters=parameters,admission_threshold=threshold if model else None)
    try:
        obs,_=env.reset(options=dict(scenario=cfg,prefix=prefix,trace=trace));done=False;reward=0.
        while not done:
            action=int(model.predict(obs,action_masks=obs['mask']>.5,deterministic=True)[0]) if model else mission.rule_action(method,env,parameters,threshold)
            assert obs['mask'][action]>.5
            obs,r,done,truncated,_=env.step(action);assert not truncated;reward+=r
        result=env.result();assert reward==result['success']
        result.update(method=method,init=init,seed=seed,load=load,profile=cfg['private']['demand_profile'],
            scenario_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
            action_sha256=hashlib.sha256(json.dumps(result['decisions'],sort_keys=True).encode()).hexdigest())
        row={k:result[k] for k in ('method','init','seed','load','profile','scenario_sha256','action_sha256','requests','success','admitted','by_service')}
        if trace:
            run.write(str(prefix)+'-result.json',result)
            raw=Path(str(prefix)+'-events.csv')
            with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
            raw.unlink();run.write(str(prefix)+'-metrics.json',row)
            return audit_row(row)
        return row
    finally:env.close()


def evaluate_group(spec):
    method,init=spec;torch.set_num_threads(1)
    if method=='ppo':model=MaskablePPO.load(run.OUT/'models'/f'ppo-{init}.zip',device='cpu')
    elif method=='prior':model=MaskablePPO.load(run.OUT/'training'/str(run.INITS[0])/'0.zip',device='cpu')
    else:model=None
    return [episode(s,l,method,init,model) for s in run.TEST for l in run.LOADS]


def evaluate(workers,resume=False):
    register();training=run.read(run.OUT/'training_complete.json')
    assert {r['init'] for r in training}==set(run.INITS)
    for r in training:
        assert r['steps']==65536 and r['updates']>0
        assert run.sha(run.OUT/r['model'])==r['sha256']
    gate=run.OUT/'test_started.json'
    if resume:
        assert gate.exists() and (run.OUT/'evaluation_audit_amendment.json').exists()
        assert not (run.OUT/'evaluation.json').exists()
        for name,h in run.read(gate).items():assert run.sha(run.OUT/name)==h
    else:assert not gate.exists(),'formal test already opened; verify existing data instead'
    log=(run.OUT/'full-tests.log').read_text();assert '\nOK\n' in log and '\nFAILED' not in log
    if not resume:
        run.write(gate,{n:run.sha(run.OUT/n) for n in ('protocol.json','training_complete.json',
            'diagnosis.json','prior_validation.json','baseline_selection.json','evaluation_registration.json',
            'family_selection_registration.json','family_selection.json','full-tests.log')})
    specs=[('ppo',i) for i in run.INITS]+[(m,0) for m in RULES]
    rows=[]
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for group in pool.map(evaluate_group,specs):
            rows.extend(group);print('AUDITED',len(rows),2700,flush=True)
    run.write(run.OUT/'evaluation.json',rows)


def verify_row(row):
    verified=audit_row(row)
    prefix=run.OUT/'native'/f"{row['method']}-{row['init']}"/f"{row['load']}-{row['seed']}"
    result=run.read(str(prefix)+'-result.json')
    assert hashlib.sha256(json.dumps(result['decisions'],sort_keys=True).encode()).hexdigest()==row['action_sha256']
    return verified


def inference_probe(init):
    torch.set_num_threads(1);model=MaskablePPO.load(run.OUT/'models'/f'ppo-{init}.zip',device='cpu')
    return [episode(s,l,'ppo',init,model,trace=False) for s in (run.TEST[0],run.TEST[-1]) for l in run.LOADS]


def analyze(rows):
    methods=('ppo',*RULES);lookup={(r['method'],r['init'],r['seed'],r['load']):r for r in rows}
    rng=np.random.default_rng(20261105)
    sd=rng.integers(len(run.TEST),size=(20000,len(run.TEST)));ids=rng.integers(len(run.INITS),size=(20000,len(run.INITS)))
    boots={};a=dict(methods={},comparisons={},per_initialization={},per_load={},per_profile={})
    for m in methods:
        inits=run.INITS if m=='ppo' else (0,)
        matrix=np.array([[sum(lookup[m,i,s,l]['success'] for l in run.LOADS)/sum(run.LOADS) for s in run.TEST] for i in inits])
        boots[m]=matrix[ids[:,:,None],sd[:,None,:]].mean((1,2)) if m=='ppo' else matrix[0,sd].mean(1)
        rr=[r for r in rows if r['method']==m];v=audit.summarize(rr)
        v['violation_rate_admitted']=v['admitted_violations']/v['admitted'] if v['admitted'] else 0.
        v['ci95']=np.quantile(boots[m],[.025,.975]).tolist()
        v['by_service']={s:{k:sum(r['by_service'][s][k] for r in rr) for k in ('requests','success')} for s in ('telemetry','command','object')}
        v['minimum_service_success']=min(x['success']/x['requests'] for x in v['by_service'].values())
        a['methods'][m]=v
        a['per_load'][m]={str(l):audit.summarize([r for r in rr if r['load']==l]) for l in run.LOADS}
        a['per_profile'][m]={p:audit.summarize([r for r in rr if r['profile']==p]) for p in ('steady','upload_first','retrieval_first') if any(r['profile']==p for r in rr)}
        if m=='ppo':a['per_initialization']={str(i):audit.summarize([r for r in rr if r['init']==i]) for i in inits}
    for m in RULES:
        point=a['methods']['ppo']['success_rate']-a['methods'][m]['success_rate'];diff=boots['ppo']-boots[m]
        a['comparisons']['ppo-'+m]=dict(difference=point,ci95=np.quantile(diff,[.025,.975]).tolist(),
            centered_bootstrap_p=float((1+np.count_nonzero(np.abs(diff-point)>=abs(point)))/(len(diff)+1)))
    primary=run.read(run.OUT/'baseline_selection.json')['selected']['method'];a['primary_rule']=primary
    secondary=[(k,v) for k,v in a['comparisons'].items() if k!='ppo-'+primary];adjusted=0.
    for rank,(key,v) in enumerate(sorted(secondary,key=lambda kv:kv[1]['centered_bootstrap_p'])):
        adjusted=max(adjusted,min(1.,(len(secondary)-rank)*v['centered_bootstrap_p']));v['holm_adjusted_p']=adjusted
    learned=a['methods']['ppo'];b=a['methods'][primary]
    gates=dict(primary_ci_positive=a['comparisons']['ppo-'+primary]['ci95'][0]>0,
        learning_ci_positive=a['comparisons']['ppo-prior']['ci95'][0]>0,
        admitted_violation_not_worse=learned['violation_rate_admitted']<=b['violation_rate_admitted'],
        arrival_violation_not_worse=learned['violation_rate_all']<=b['violation_rate_all'])
    a['promotion']=gates;a['promoted']=all(gates.values());return a


def verify(workers):
    register()
    for n,h in run.read(run.OUT/'test_started.json').items():assert run.sha(run.OUT/n)==h
    models=run.read(run.OUT/'training_complete.json');assert {r['init'] for r in models}==set(run.INITS)
    actual=[];torch.set_num_threads(1)
    for r in models:
        assert r['selected']==max(r['checkpoints'],key=lambda x:(x['validation']['rate'],-x['step']))
        for checkpoint in r['checkpoints']:assert run.sha(run.OUT/checkpoint['path'])==checkpoint['sha256']
        assert run.sha(run.OUT/r['model'])==r['sha256']
        model=MaskablePPO.load(run.OUT/r['model'],device='cpu')
        assert model.gamma==1. and model.gae_lambda==1. and model.num_timesteps==r['selected']['step']
        assert r['steps']==65536 and r['updates']>0
        assert r['all_environment_resets'][0]==run.VALIDATION[0]
        assert all(71000000<=s<72000000 for s in r['all_environment_resets'][1:])
        if model.num_timesteps:assert bool(torch.any(model.policy.action_net.residual[-1].weight!=0))
        actual.append(dict(init=r['init'],steps=model.num_timesteps,learning_rate=model.learning_rate,
            rollout_steps=model.n_steps,gamma=model.gamma,gae_lambda=model.gae_lambda,sha256=r['sha256']))
    rows=run.read(run.OUT/'evaluation.json');expected={(m,i,s,l) for m in ('ppo',*RULES) for i in (run.INITS if m=='ppo' else (0,)) for s in run.TEST for l in run.LOADS}
    assert len(rows)==len(expected)==2700
    assert {(r['method'],r['init'],r['seed'],r['load']) for r in rows}==expected
    verified=[]
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for n,r in enumerate(pool.map(verify_row,rows),1):
            verified.append(r)
            if n%180==0:print('VERIFIED',n,len(rows),flush=True)
        probes=[p for group in pool.map(inference_probe,run.INITS) for p in group]
    lookup={(r['method'],r['init'],r['seed'],r['load']):r for r in rows}
    for probe in probes:
        row=lookup[probe['method'],probe['init'],probe['seed'],probe['load']]
        for key in ('action_sha256','success','admitted','by_service'):assert probe[key]==row[key],(probe,key)
    for s in run.TEST:
        for l in run.LOADS:assert len({r['scenario_sha256'] for r in verified if r['seed']==s and r['load']==l})==1
    a=analyze(verified);run.write(run.OUT/'analysis.json',a)
    log=(run.OUT/'full-tests.log').read_text();assert '\nOK\n' in log
    v=dict(passed_tests=int(re.search(r'Ran (\d+) tests',log)[1]),verified_native_episodes=len(rows),
        inference_probes=len(probes),actual_models=actual,new_independent_test=True,
        all_packet_audits_pass=True,all_direction_audits_pass=True,all_reservation_replays_pass=True,
        all_methods_share_scenarios=True,all_initializations_reported=True,promotion=a['promotion'])
    run.write(run.OUT/'verification.json',v)
    text=['# 面向低空智联的多模态网络切片技术：第一部分验证','',
        '12架无人机、双AP、双边缘及IP/简化EID-RLOC/原生NDN保持原执行实现。任务阶段只改变请求到达时间，业务数量、合同持续时间、速率、报文大小及SLA保持不变。阶段标签不进入控制器，输入仅含当前请求、资源、已观测交付、位置关联及历史到达计数。','',
        '主方法为单个规则引导Masked PPO，选择拒绝或模态、边缘、映射/缓存位置。显式规则起点、标准策略梯度与价值函数，不称从零学习。无预测器、辅助头、资源档位、超订或对偶惩罚。','',
        'gamma=1、GAE lambda=1，沿实际rollout回溯奖励，在未终止的rollout边界使用价值自举。势函数为0.5×已观测活跃合同数，首尾为0，整局奖励严格等于真实SLA成功数。8个初始化全部保留，只在60个验证场景选择检查点。','',
        '24组模态规则、7个准入价格阈值及常用规则均在验证集比较；主对照为验证成功率最高的规则族：'+a['primary_rule']+'。测试采用未打开的323000—323059种子、48/80/112三负载，共2700回合，另有48回合模型推理重放。','',
        '| 方法 | SLA成功率 | 接纳率 | 违约／全部到达 | 违约／已接纳 | 最低业务成功率 |','|---|---:|---:|---:|---:|---:|']
    for name,r in a['methods'].items():text.append(f"| {name} | {r['success_rate']:.2%} | {r['admission_rate']:.2%} | {r['violation_rate_all']:.2%} | {r['violation_rate_admitted']:.2%} | {r['minimum_service_success']:.2%} |")
    for name,r in a['comparisons'].items():text+=['',f"{name}：{100*r['difference']:+.2f}个百分点，配对95%区间[{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}]。"]
    text+=['','主比较为验证选定的最强规则；次要比较的中心化bootstrap近似p值及Holm校正见analysis.json。双重重采样场景和初始化，场景内三负载保持配对。分业务、负载、阶段及每个初始化完整报告。','',
        f"全部发布条件通过：{a['promoted']}；未通过时不替换已验证主线。",'',
        '本场景为固定高度的小规模低空巡检简化平台；EID-RLOC为简化实现。资源账本是准入预约，不是逐切片带宽保证。遥测仅IP，多模态选择收益不能归给遥测。新的任务负载结果不能与历史均匀负载成绩直接相减归因于算法。']
    (run.OUT/'report.md').write_text('\n'.join(text)+'\n')
    print('FINAL',a['promoted'],a['primary_rule'],a['comparisons']['ppo-'+a['primary_rule']],a['promotion'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('register','evaluate','verify'));parser.add_argument('--workers',type=int,default=8);parser.add_argument('--resume-audit-repair',action='store_true')
    args=parser.parse_args()
    if args.command=='register':register()
    elif args.command=='evaluate':evaluate(args.workers,args.resume_audit_repair)
    else:verify(args.workers)


if __name__=='__main__':main()
