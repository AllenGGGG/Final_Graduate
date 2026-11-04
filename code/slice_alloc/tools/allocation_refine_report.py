"""Verify all fresh native runs and compare refined/reference RL fairly."""
import argparse
import concurrent.futures as cf
from pathlib import Path
import re

import numpy as np

from slice_alloc.tools import allocation_simple_report as audit
from slice_alloc.tools import allocation_refine_run as run


def verify_task(spec):
    out,row=spec
    return audit.verify_episode(out,row)


def analyze(rows,protocol):
    inits,seeds,loads=protocol['initializations'],protocol['test'],protocol['loads']
    lookup={(r['method'],r['init'],r['seed'],r['load']):r for r in rows}
    def rate(method,init,seed):
        rr=[lookup[method,init,seed,load] for load in loads]
        return sum(r['success'] for r in rr)/sum(r['requests'] for r in rr)
    matrices={m:np.array([[rate(m,i,s) for s in seeds] for i in
        (inits if m in ('double_dqn','reference_dqn') else [0])]) for m in protocol['methods']}
    rng=np.random.default_rng(20261104)
    sd=rng.integers(len(seeds),size=(20000,len(seeds)))
    ids=rng.integers(len(inits),size=(20000,len(inits)))
    boots={m:(matrix[ids[:,:,None],sd[:,None,:]].mean((1,2)) if len(matrix)>1
        else matrix[0,sd].mean(1)) for m,matrix in matrices.items()}
    result=dict(methods={},comparisons={},per_initialization={},per_load={},
        bootstrap='20000 draws; paired scenario clusters and paired reference/refined initialization; primary refined-greedy; secondary comparisons not multiplicity adjusted')
    for m in protocol['methods']:
        rr=[r for r in rows if r['method']==m]
        result['methods'][m]={**audit.summarize(rr),'ci95':np.quantile(boots[m],[.025,.975]).tolist()}
        result['per_load'][m]={str(l):audit.summarize([r for r in rr if r['load']==l]) for l in loads}
        if len(matrices[m])>1:
            result['per_initialization'][m]={str(i):audit.summarize([r for r in rr if r['init']==i]) for i in inits}
    for lhs,rhs in [('double_dqn','greedy'),('reference_dqn','greedy'),('double_dqn','reference_dqn'),
                    ('double_dqn','nearest'),('double_dqn','first_fit')]:
        result['comparisons'][lhs+'-'+rhs]=dict(
            difference=float(matrices[lhs].mean()-matrices[rhs].mean()),
            ci95=np.quantile(boots[lhs]-boots[rhs],[.025,.975]).tolist())
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    out=args.output.resolve()
    protocol=run.read(out/'protocol.json')
    for name,digest in protocol['sources'].items():
        assert run.sha(run.base.ROOT/'slice_alloc'/name)==digest,name
        assert run.sha(out/'frozen'/name)==digest,name
    assert run.sha(run.base.b.p.NS3/'build/scratch/allocation-final-scenario')==protocol['binary_sha256']
    for name,digest in run.read(out/'test_started.json').items():
        assert run.sha(out/name)==digest,name
    log=(out/'full-tests.log').read_text()
    assert '\nOK\n' in log and '\nFAILED ' not in log
    test_count=int(re.search(r'Ran (\d+) tests',log)[1])
    completed=run.read(out/'training_complete.json')
    choices={}
    assert {r['init'] for r in completed['models']}==set(protocol['initializations'])
    for row in completed['models']:
        assert row['steps']==65536 and row['updates']>0
        assert row['protocol_sha256']==run.sha(out/'protocol.json')
        for key in ('model','reference_model'):
            assert run.sha(out/row[key])==row[key+'_sha256']
        assert row['reference_model_sha256']==protocol['reference_models'][str(row['init'])]
        assert row['selected']==max(row['checkpoints'],key=lambda x:(x['validation']['rate'],-x['step']))
        for checkpoint in row['checkpoints']:
            assert run.sha(out/checkpoint['path'])==checkpoint['sha256']
        assert all(protocol['training_seed_range'][0]<=s<protocol['training_seed_range'][1] for s in row['training_seeds'])
        choices[str(row['init'])]=dict(step=row['selected']['step'],origin=row['selected']['origin'],
            validation_rate=row['selected']['validation']['rate'])
    rows=run.read(out/'evaluation.json')
    expected={(m,i,s,l) for m in protocol['methods'] for i in
        (protocol['initializations'] if m in ('double_dqn','reference_dqn') else [0])
        for s in protocol['test'] for l in protocol['loads']}
    assert len(rows)==len(expected)
    assert {(r['method'],r['init'],r['seed'],r['load']) for r in rows}==expected
    verified=[]
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n,row in enumerate(pool.map(verify_task,((out,r) for r in rows)),1):
            verified.append(row)
            if n%120==0:
                print('VERIFIED',n,len(rows),flush=True)
    for seed in protocol['test']:
        for load in protocol['loads']:
            assert len({r['scenario_sha256'] for r in verified if r['seed']==seed and r['load']==load})==1
    analysis=analyze(verified,protocol)
    analysis['selection']=choices
    run.write(out/'analysis.json',analysis)
    verification=dict(passed_tests=test_count,verified_native_episodes=len(verified),
        full_training_complete=True,fresh_test_complete=True,all_packet_audits_pass=True,
        all_direction_audits_pass=True,all_reservation_replays_pass=True,
        all_methods_share_scenarios=True,all_initializations_reported=True,
        protocol_sha256=run.sha(out/'protocol.json'))
    run.write(out/'verification.json',verification)
    report=['# 面向低空智联的多模态网络切片技术：最终验证版','',
        '## 设计思路','',
        '固定低空巡检背景：12架无人机、双AP、双边缘，4架巡检、8架驻留，高度80m。ns-3真实运行无线竞争、移动关联与三种网络机制。遥测采用IP上行，命令可选IP/EID-RLOC，地图可选IP/NDN及边缘缓存。机制间保持相同业务合同与SLA。','',
        '请求到达时，控制器选择拒绝或一个完整的模态、边缘、映射/缓存位置配置。17维账本约束资源，动作数组11槽，业务兼容和资源可行性由掩码检查。Masked Double-DQN依据当前资源、请求、端点历史移动及已观测交付计数，为候选配置估计长期成功合同数。奖励只来自实际满足SLA的合同；拒绝仍计入到达分母。','',
        '本轮唯一训练优化是将后期学习率降至5e-5，并从各初始化32768步终点续训至65536步。新回放先填充1024步，计入总预算。原版验证最优模型也保留为候选，只按原验证集选检查点，同分取较早者。没有新增网络模块、资源档位、超订或拉格朗日约束。','',
        '资源预测和运行期QoS控制属于后续研究点，本轮只验证初始分配。资源账本是准入预约，不是逐切片确定性QoS保证。IP下行未实现连接迁移，EID-RLOC为简化实现；结论适用于该小规模仿真场景。','',
        '## 实际结果','',
        '全部5个初始化保留。使用60个新测试种子、48/80两负载；原版与优化版各600回合，三条规则各120回合，共1560回合。原版和优化版用相同场景及初始化配对，避免把换测试集造成的差异归因于训练优化。','',
        '| 方法 | SLA成功率 | 95%区间 | 接纳率 | 主动拒绝率 |',
        '|---|---:|---:|---:|---:|']
    names=dict(double_dqn='最终选择RL',reference_dqn='原版RL',greedy='Greedy',nearest='Nearest',first_fit='First-Fit')
    for method,values in analysis['methods'].items():
        lo,hi=values['ci95']
        report.append(f"| {names[method]} | {values['success_rate']:.2%} | [{lo:.2%}, {hi:.2%}] | {values['admission_rate']:.2%} | {values['voluntary_reject_rate']:.2%} |")
    for key,label in [('double_dqn-greedy','最终RL − Greedy'),('reference_dqn-greedy','原版RL − Greedy'),('double_dqn-reference_dqn','最终RL − 原版RL')]:
        row=analysis['comparisons'][key]
        lo,hi=row['ci95']
        report += ['',f"{label}：{100*row['difference']:+.2f}个百分点，配对95%区间[{100*lo:+.2f}, {100*hi:+.2f}]。"]
    report += ['', '主比较为最终RL对Greedy；原版对照和其他规则为补充比较，未作多重比较校正。区间同时重采样场景和初始化，同场景两负载一起抽样，原版与优化版的初始化保持配对。','',
        '## 验证与复现','',
        f"{test_count}项测试通过；{len(verified)}个正式回合全部重新核对完整发包数、逐包SLA、遥测方向、资源预约和状态释放。模型、检查点、输入、决策和逐包数据保留在本输出目录。",'',
        '设计协议见REFINE_PROTOCOL.md；全部分业务、分负载、初始化和选择结果见analysis.json；独立核验见verification.json。protocol.json和frozen目录保留完整参数与源码。','',
        '续训总预算约为原版两倍，本结果不能单独归因于降低学习率，也不构成一种新的DQN算法。只按验证集完成一次有界优化，未用新正式测试再次调参。']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    print((out/'report.md').read_text(),flush=True)


if __name__=='__main__':
    main()
