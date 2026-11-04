"""Independent raw-packet audit, paired statistics and promotion decision."""
import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import re
import numpy as np
import torch
from slice_alloc import guided_allocation as guided
from slice_alloc.tools import allocation_improve_run as run
from slice_alloc.tools import allocation_simple_report as native


def verify_task(row):
    return native.verify_episode(run.OUT,row)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();run.check();torch.set_num_threads(1)
    registration=run.read(run.OUT/'analysis_registration.json')
    assert registration['source_sha256']==run.sha(__file__)
    assert registration['protocol_sha256']==run.sha(run.OUT/'protocol.json')
    evaluation_registration=run.read(run.OUT/'evaluation_registration.json')
    assert evaluation_registration['source_sha256']==run.sha(Path(__file__).with_name('allocation_improve_evaluate.py'))
    for name,h in run.read(run.OUT/'test_started.json').items():assert run.sha(run.OUT/name)==h
    models=run.read(run.OUT/'training_complete.json')
    assert {r['init'] for r in models}==set(run.INITS)
    actual=[]
    for r in models:
        assert r['selected']==max(r['checkpoints'],key=lambda x:(x['validation']['rate'],-x['step']))
        for checkpoint in r['checkpoints']:assert run.sha(run.OUT/checkpoint['path'])==checkpoint['sha256']
        model=guided.GuidedDQN.load(run.OUT/r['model'],device='cpu')
        assert run.sha(run.OUT/r['model'])==r['sha256']
        assert model.num_timesteps==r['selected']['step'] and model.n_quantiles==1 and model.gamma==1.
        assert r['steps']==65536 and r['updates']>0
        # The first entry is a deterministic validation-parity reset, outside
        # learn(). SB3 subsequently resets the training env before collecting.
        specs=r['training_seeds'];assert specs[0]==run.VALIDATION[0]
        assert all(61000000<=s<62000000 for s in specs[1:])
        actual.append(dict(init=r['init'],selected_step=model.num_timesteps,
            learning_rate=model.learning_rate,updates=r['updates'],sha256=r['sha256']))
    rows=run.read(run.OUT/'evaluation.json')
    methods=('guided',*run.RULES,'prior')
    expected={(m,i,s,l) for m in methods for i in (run.INITS if m=='guided' else (0,)) for s in run.TEST for l in run.LOADS}
    assert len(rows)==len(expected)==1680
    assert {(r['method'],r['init'],r['seed'],r['load']) for r in rows}==expected
    verified=[]
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n,r in enumerate(pool.map(verify_task,rows),1):
            verified.append(r)
            if n%120==0:print('VERIFIED',n,len(rows),flush=True)
    for s in run.TEST:
        for l in run.LOADS:assert len({r['scenario_sha256'] for r in verified if r['seed']==s and r['load']==l})==1
    lookup={(r['method'],r['init'],r['seed'],r['load']):r for r in verified}
    rng=np.random.default_rng(20261105)
    sd=rng.integers(len(run.TEST),size=(20000,len(run.TEST)));ids=rng.integers(len(run.INITS),size=(20000,len(run.INITS)))
    boots={};analysis=dict(methods={},comparisons={},per_initialization={},per_load={})
    for m in methods:
        inits=run.INITS if m=='guided' else (0,)
        matrix=np.array([[sum(lookup[m,i,s,l]['success'] for l in run.LOADS)/sum(run.LOADS) for s in run.TEST] for i in inits])
        boots[m]=matrix[ids[:,:,None],sd[:,None,:]].mean((1,2)) if m=='guided' else matrix[0,sd].mean(1)
        rr=[r for r in verified if r['method']==m];v=native.summarize(rr)
        assert v['success']+v['admitted_violations']==v['admitted']
        v['violation_rate_admitted']=v['admitted_violations']/v['admitted']
        v['ci95']=np.quantile(boots[m],[.025,.975]).tolist()
        v['by_service']={s:{k:sum(r['by_service'][s][k] for r in rr) for k in ('requests','success')} for s in ('telemetry','command','object')}
        v['minimum_service_success']=min(t['success']/t['requests'] for t in v['by_service'].values())
        analysis['methods'][m]=v
        analysis['per_load'][m]={str(l):native.summarize([r for r in rr if r['load']==l]) for l in run.LOADS}
        if m=='guided':analysis['per_initialization']={str(i):native.summarize([r for r in rr if r['init']==i]) for i in inits}
    for m in methods[1:]:
        d=analysis['methods']['guided']['success_rate']-analysis['methods'][m]['success_rate'];samples=boots['guided']-boots[m]
        analysis['comparisons']['guided-'+m]=dict(difference=d,ci95=np.quantile(samples,[.025,.975]).tolist(),
            centered_bootstrap_p=float((1+np.count_nonzero(np.abs(samples-d)>=abs(d)))/(len(samples)+1)))
    secondary=[(k,v) for k,v in analysis['comparisons'].items() if k!='guided-modal_greedy'];adjusted=0.
    for rank,(key,v) in enumerate(sorted(secondary,key=lambda kv:kv[1]['centered_bootstrap_p'])):
        adjusted=max(adjusted,min(1.,(len(secondary)-rank)*v['centered_bootstrap_p']));v['holm_adjusted_p']=adjusted
    g=analysis['methods']['guided'];b=analysis['methods']['modal_greedy']
    gates=dict(primary_ci_positive=analysis['comparisons']['guided-modal_greedy']['ci95'][0]>0,
        learning_effect_ci_positive=analysis['comparisons']['guided-prior']['ci95'][0]>0,
        admitted_violation_not_worse=g['violation_rate_admitted']<=b['violation_rate_admitted'],
        arrival_violation_not_worse=g['violation_rate_all']<=b['violation_rate_all'],
        all_raw_audits_pass=True,all_initializations_retained=True)
    analysis['promotion']=gates;analysis['promoted']=all(gates.values())
    run.write(run.OUT/'audited_analysis.json',analysis)
    test_log=(run.OUT/'full-tests.log').read_text();assert '\nOK\n' in test_log
    verification=dict(verified_native_episodes=len(rows),actual_models=actual,
        all_packet_audits_pass=True,all_direction_audits_pass=True,all_reservation_replays_pass=True,
        all_methods_share_scenarios=True,new_independent_test=True,passed_tests=int(re.search(r'Ran (\d+) tests',test_log)[1]),
        analysis_source_sha256=run.sha(__file__),promotion=gates)
    run.write(run.OUT/'verification.json',verification)
    report=['# 第一部分：规则引导的多模态切片Double-DQN验证结果','',
        '低空巡检背景、IP/简化EID-RLOC/原生NDN、合同与容量保持不变。RL选择拒绝或模态、边缘、映射/缓存位置。规则先验显式进入Q值；这是规则引导的RL，不称从零学习。','',
        '24组模态感知规则在60个验证场景上选优，全部8个RL初始化保留，测试使用未用于历史实验的223000—223059种子与48/80两负载。包括规则起点消融，共1680回合；所有回合重新逐包、方向和预约审计。','',
        '| 方法 | SLA成功率 | 违约／全部到达 | 违约／已接纳 | 最低业务成功率 |','|---|---:|---:|---:|---:|']
    for m,v in analysis['methods'].items():report.append(f"| {m} | {v['success_rate']:.2%} | {v['violation_rate_all']:.2%} | {v['violation_rate_admitted']:.2%} | {v['minimum_service_success']:.2%} |")
    for key,v in analysis['comparisons'].items():
        report+=['',f"{key}：{100*v['difference']:+.2f}个百分点，配对95%区间[{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]。"]
    report+=['','主比较为RL对验证选定的模态感知Greedy；其他比较的探索性中心化bootstrap近似p值和Holm校正在audited_analysis.json。场景和初始化双重重采样，场景内两负载保持配对。','',
        f"通过全部发布条件：{analysis['promoted']}。未通过时不替换已验证主线，不将负面结果删除。",'',
        '账本是准入预约，不是逐切片带宽保证；固定高度小规模场景不作为低空信道创新。各业务候选受兼容性限制，遥测仅IP。资源档位、超订、预测器、辅助头及运行期QoS不在本轮。']
    (run.OUT/'report.md').write_text('\n'.join(report)+'\n')
    print('AUDIT RESULT',analysis['promoted'],analysis['comparisons']['guided-modal_greedy'],gates,flush=True)


if __name__=='__main__':main()
