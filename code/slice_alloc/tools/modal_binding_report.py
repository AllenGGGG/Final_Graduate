"""Read-only statistical analysis and thesis artifacts for the frozen binding study."""
from __future__ import annotations
import collections
import json
from pathlib import Path
import numpy as np
from slice_alloc.tools.modal_binding_run import OUT,ROOT,INITS,RULES,read,write,sha,hashes

LABELS={'joint_qrdqn':'联合模态QR-DQN','fixed_qrdqn':'固定模态QR-DQN',
    'no_motion':'去移动观测QR-DQN','first_fit':'首次可行',
    'fixed_price':'固定模态＋指数价格','joint_price':'跨模态＋指数价格',
    'ndn_only':'仅NDN（可复用对象优先缓存）','semantic_rule':'业务语义与移动规则'}

def summary(rows):
    models=sorted({r['init'] for r in rows});n=len(models)
    total=sum(r['requests'] for r in rows)/n;success=sum(r['success'] for r in rows)/n
    return dict(rate=success/total,success=success,requests=total,
        admitted=sum(r['admitted'] for r in rows)/n/total,
        reservation_per_success=sum(sum(r['resource_time']) for r in rows)/n/max(success,1),
        wire_per_arrival=sum(r['wire_bytes'] for r in rows)/n/total,
        wire_per_success=sum(r['wire_bytes'] for r in rows)/n/max(success,1),
        origin_per_arrival=sum(r['origin_wire_bytes'] for r in rows)/n/total,
        phy_per_arrival=sum(r['phy_tx_seconds'] for r in rows)/n/total,
        decision_ms=float(np.mean([r['decision_ms'] for r in rows])),
        individual_rates={str(i):sum(r['success'] for r in rows if r['init']==i)/sum(r['requests'] for r in rows if r['init']==i) for i in models},
        modes={mode:sum(r['modes'].get(mode,0) for r in rows)/n for mode in ('ip','id','ndn')},
        by_service={s:dict(rate=sum(r['by_service'][s]['success'] for r in rows)/max(1,sum(r['by_service'][s]['requests'] for r in rows)),
            requests=sum(r['by_service'][s]['requests'] for r in rows)/n) for s in ('telemetry','object','command')})


def paired(left,right,samples=10000):
    # Same seed's two loads form one cluster. Initialization resampled separately.
    seeds=sorted({r['seed'] for r in left});li=sorted({r['init'] for r in left});ri=sorted({r['init'] for r in right})
    def matrix(rows,inits):
        lookup=collections.defaultdict(float)
        for r in rows:lookup[r['init'],r['seed']]+=r['success']
        return np.array([[lookup[i,s] for s in seeds] for i in inits])
    l=matrix(left,li);r=matrix(right,ri)
    denominator=np.array([sum(x['requests'] for x in left if x['seed']==s and x['init']==li[0]) for s in seeds])
    rng=np.random.default_rng(50931);draws=[]
    for _ in range(samples):
        ix=rng.integers(len(seeds),size=len(seeds));ia=rng.integers(len(li),size=len(li));ib=rng.integers(len(ri),size=len(ri))
        # Paired initialization labels for two training variants, when matched.
        if li==ri:ib=ia
        draws.append((l[ia][:,ix].mean(0).sum()-r[ib][:,ix].mean(0).sum())/denominator[ix].sum())
    return dict(difference=summary(left)['rate']-summary(right)['rate'],
        ci95=np.quantile(draws,[.025,.975]).tolist(),clusters=len(seeds),initializations=len(li))


def main():
    rows=read(OUT/'evaluation.json')
    for name in ('ndn_only','semantic_rule'):rows+=read(OUT/f'{name}-evaluation.json')
    assert all(r['audited'] for r in rows)
    methods=list(LABELS);test=[r for r in rows if r['scope']=='test']
    stats={m:summary([r for r in test if r['method']==m]) for m in methods}
    rules=[read(OUT/'rules'/f'{m}.json') for m in (*RULES,'ndn_only','semantic_rule')]
    best=max(rules,key=lambda r:r['chosen']['rate'])['method']
    differences={m:paired([r for r in test if r['method']=='joint_qrdqn'],[r for r in test if r['method']==m]) for m in methods if m!='joint_qrdqn'}
    ood={}
    for mix,load in [('object_heavy',80),('mobile_heavy',80),('balanced',112),('novel',80)]:
        group=[r for r in rows if r['scope']=='ood' and r['mix']==mix and r['load']==load]
        ood[f'{mix}-{load}']={m:summary([r for r in group if r['method']==m]) for m in methods}
    study=dict(test=stats,paired=differences,best_validation_rule=best,ood=ood,
        by_load={str(load):{m:summary([r for r in test if r['load']==load and r['method']==m]) for m in methods} for load in (48,80)},
        episodes=len(rows),audited_episodes=sum(r['audited'] for r in rows))
    abandoned=[OUT.with_name('modal_binding_light_dev'),OUT.with_name('modal_binding_inconsistent_dev')]
    lower_bound=sum(max(x['step'] for x in read(path)) for folder in abandoned for path in (folder/'training').glob('*/*/progress.json'))
    study['development_cost']=dict(archived_directories=[str(folder.relative_to(ROOT)) for folder in abandoned],
        completed_checkpoint_interactions_lower_bound=lower_bound,
        exact_interrupted_step_count_available=False,
        final_training_interactions=491520)
    write(OUT/'analysis.json',study)
    from slice_alloc.tools.modal_binding_allocate import select
    deployment=select();write(OUT/'recommended_policy.json',deployment)
    figures(study)
    q=stats['joint_qrdqn'];ref=stats[best];delta=differences[best]
    fmt=lambda x:f'{100*x:.2f}%'
    if delta['ci95'][0]>0:
        interpretation='本轮总体留出结果支持联合学习方法优于验证集选定规则；这一结论仅限声明的拓扑、合同和分布，分组回退另报。'
    elif delta['ci95'][1]<0:
        interpretation='本轮联合学习方法低于验证集选定规则，不能写成强化学习优胜成果。跨模态联合分配建模与原生实现有效，但学习器作为性能改进主张未获支持；规则版本应作为实际使用的参考。'
    else:
        interpretation='配对区间覆盖零，本轮不能确认联合学习方法优于验证集选定规则。应将结果描述为可行实现与方法比较，而非显著性能突破。'
    lines=['# 面向低空智联的多模态网络切片技术：研究点1最终实验报告','',
        '本轮仅完成智能分配。资源预测、运行期主动QoS保障尚未实现。', '',
        f"联合模态单模型QR-DQN的常规留出SLA成功率为 **{fmt(q['rate'])}**；验证集选定的最强规则为{LABELS[best]}，测试为 **{fmt(ref['rate'])}**。差值 **{100*delta['difference']:+.2f}个百分点**，场景与初始化配对重采样95%区间 **[{100*delta['ci95'][0]:+.2f}, {100*delta['ci95'][1]:+.2f}]个百分点**。",'',
        interpretation,'',
        f"本轮最终推荐 **{LABELS[deployment['method']]}**，按验证集选择方法，平局优先无需模型训练的规则。推荐决策在正式测试前规定，不依据测试挑选优胜者；推理耗时另行实测，不假设规则推理一定更快。完整学习训练结果仍报告，不能把规则收益写成神经网络收益。",'',
        '该结果是5次独立训练的平均，部署时使用一个模型，不进行五模型集成。QR-DQN与Double-Q是已有算法，贡献定位为业务语义驱动的跨模态切片联合分配建模、原生执行及可审计评测。', '',
        '## 设计与适用范围','',
        '![系统与研究范围](figures/system.png)','',
        '12架无人机、双AP和双互连边缘；三类统一下行业务不携带模态标签。动作联合选择接纳、IP/受限EID-RLOC/NDN和功能位置，共享17维容量账本。IP接纳时固定定位，ID被动更新定位，NDN按名字获取；同一业务跨模态的载荷、消费者与SLA一致。双消费者合同屏蔽仅实现单端点的ID候选。', '',
        '本轮固定高度、范围传播和有限拓扑，未实现CPU调度或租户无线调度器；资源预约不能替代包级SLA验证。IP固定定位是本轮协议合同，结论不能推广为所有IP传输无法支持移动。', '',
        '租户隔离在本轮表现为独立命名空间、共享对象键和合同账本，未实现认证、恶意访问防护或完整的安全隔离。','',
        '理论依据是[R05的四域映射与SMV目标](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT250949)，求解采用[QR-DQN](https://arxiv.org/abs/1710.10044)与[Double DQN](https://arxiv.org/abs/1509.06461)。价格规则是本实验实现的exp(3u)基线，未直接继承其他论文竞争比。', '',
        '## 常规留出结果','',
        '| 方法 | SLA成功率 | 接纳率 | 预约秒/成功合同 | 有线KB/成功合同 | 平均决策ms |',
        '|---|---:|---:|---:|---:|---:|']
    for m in methods:
        s=stats[m];lines.append(f"| {LABELS[m]} | {fmt(s['rate'])} | {fmt(s['admitted'])} | {s['reservation_per_success']:.3f} | {s['wire_per_success']/1000:.2f} | {s['decision_ms']:.3f} |")
    lines+=['','### 按业务和负载分组','',
        '| 方法 | 态势图像流 | 共享地图 | 任务命令 | 负载48 | 负载80 |','|---|---:|---:|---:|---:|---:|']
    for m in methods:
        values=[stats[m]['by_service'][s]['rate'] for s in ('telemetry','object','command')]
        values += [study['by_load'][str(load)][m]['rate'] for load in (48,80)]
        lines.append('| '+LABELS[m]+' | '+' | '.join(fmt(v) for v in values)+' |')
    lines+=['','预约秒是17维容量归一化边际预约积分的约定代理，不是实际货币或能耗。有线字节统计全部有线跳的发送，包括转发和控制报文。决策耗时先取各回合平均再跨回合平均，在同时运行多个评测进程的CPU环境测量，不构成精确硬件基准。','',
        '![常规留出SLA与通信成本](figures/test.png)','',
        '## 独立训练与消融','',
        '| 方法 | 初始化11 | 初始化23 | 初始化37 | 初始化53 | 初始化71 |','|---|---:|---:|---:|---:|---:|']
    for m in methods[:3]:lines.append('| '+LABELS[m]+' | '+' | '.join(fmt(stats[m]['individual_rates'][str(i)]) for i in INITS)+' |')
    for m in ('fixed_qrdqn','no_motion'):
        d=differences[m];lines+=['',f"联合方法相对{LABELS[m]}：{100*d['difference']:+.2f}个百分点，95%区间[{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}]。"]
    lines+=['','不同训练版本共享初始化标签和同一场景种子进行配对；同种子两档负载构成一簇。不把同场景的5个模型结果当成5个独立场景。区间跨零时，不宣称显著提升。','',
        '![验证学习曲线](figures/learning.png)','',
        '学习曲线实线为5初始化均值，阴影为最小至最大值范围，不是置信区间。正式差值区间由独立场景及初始化的配对重采样计算。','',
        '## 分布外适应性','',
        '| 留出变化 | 联合QR-DQN | 固定QR-DQN | 去移动观测 | 固定价格 | 跨模态价格 | 仅NDN | 语义规则 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    names={'object_heavy-80':'地图占比70%','mobile_heavy-80':'命令占比55%','balanced-112':'未见负载112','novel-80':'未见6块/350ms合同组合'}
    for key,value in ood.items():lines.append('| '+names[key]+' | '+' | '.join(fmt(value[m]['rate']) for m in ('joint_qrdqn','fixed_qrdqn','no_motion','fixed_price','joint_price','ndn_only','semantic_rule'))+' |')
    lines+=['','以上留出变化是V的工程操作化，未覆盖全新业务类别，也不等于理论包容性得分。全部变化均不再训练、不再改阈值。']
    controls=read(OUT/'controlled.json')
    lines+=['','## 协议机制的受控验证','',
        '| 相同命令合同 | IP按时率 | EID-RLOC按时率 | NDN按时率 |','|---|---:|---:|---:|']
    for moving,label in ((False,'静止'),(True,'移动')):
        values=[next(r['delivery_ratio'] for r in controls if r['case']=='mobility' and r['moving']==moving and r['mode']==mode) for mode in ('ip','id','ndn')]
        lines.append('| '+label+' | '+' | '.join(fmt(v) for v in values)+' |')
    lines+=['','同一地图合同、两个静止消费者，三种配置均满足SLA。源端出站字节分别为：'+
        '、'.join(f"{r['mode']}（缓存{r['cache']}）{r['origin_wire_bytes']}B" for r in controls if r['case']=='reuse')+
        '。这是固定场景机制验证，不当作总体算法增益或通用网络结论。']
    lines+=['',
        '## 训练、选择和验真','',
        '三种学习版本各5初始化、每模型32768步，总491520训练交互。在8192、16384、32768步验证，按8个验证种子和两档负载的SLA选检查点；全部初始化保留。价格规则只在相同验证集选择准入阈值。训练不使用规则示范、价格先验、人工模态奖励或未来信息。', '',
        f'测试前轻流量开发版因缺少承载取舍而归档，另一开发批因同名对象大小不一致而废止，均不继承权重。两批已写盘检查点至少对应{lower_bound}次训练交互，实际中断步数更高且未完整记录，不能隐藏这部分开销或计作正式结果。正式版以对象分块数区分版本，保证同名同版本不可变；修订仅依据开发数据，正式测试此前未打开。', '',
        '训练种子20000000—20999999，验证22000—22007，常规测试23000—23029，分布外24000—24014。常规60个场景、分布外60个场景，每方法/初始化使用相同输入。', '',
        '在训练进行、正式测试尚未打开时，追加冻结仅NDN和业务语义/移动规则两条强基线；源码指纹与规则见challengers-protocol.json。两者使用同一验证集和阈值网格，单独写定选择清单后才测试。最强规则按五条规则的验证成绩选择，不依据测试挑选。', '',
        f"全部 **{len(rows)}** 个正式评测回合通过逐包重建，emit/complete与合同应交付数量、按时数量及SLA一致；映射、缓存与应用待处理状态释放检查通过。源码、原生二进制、模型和测试输入均记录SHA256。", '',
        '[冻结协议](protocol.json) · [全部检查点与选择](selection.json) · [完整统计](analysis.json) · [逐回合数据](evaluation.json) · [设计](../../design.md)', '',
        '## 结论使用边界','',
        '结果允许出现局部回退；应同时参考业务分组、负载分组和分布外表，不能把总体提升写成每个场景均提升。旧版固定模态的79.31%采用另一套业务合同，禁止用新旧百分比声称增益。运行期切换模态、需求分位数预测、CVaR和主动QoS保障未实施，不写入本轮实验结论。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    from slice_alloc.tools.export_document import to_docx
    to_docx((OUT/'report.md').read_text().replace('(figures/','(slice_alloc/results/modal_binding/figures/'),destination=OUT/'report.docx')
    (ROOT/'README.md').write_text(f'''# 面向低空智联的多模态网络切片技术

本轮完成研究点1：低空智联下的多模态网络切片智能分配。业务语义驱动接纳、模态绑定及功能部署，共享17维资源账本，在ns-3/NFD中执行。研究点2、3尚未实现。

按冻结验证集推荐：**{LABELS[deployment['method']]}**。学习方法作为完整实测对照保留，不将规则收益计入学习算法成果。

5次独立训练的单模型QR-DQN平均SLA成功率 **{fmt(q['rate'])}**；验证集选定规则 **{fmt(ref['rate'])}**；差值 **{100*delta['difference']:+.2f}个百分点**，95%区间[{100*delta['ci95'][0]:+.2f},{100*delta['ci95'][1]:+.2f}]。完整结果包括资源开销、消融和分布外回退。

[最终设计](slice_alloc/design.md) · [最终报告](slice_alloc/results/modal_binding/report.md) · [Word报告](slice_alloc/results/modal_binding/report.docx) · [完整统计](slice_alloc/results/modal_binding/analysis.json) · [复现](slice_alloc/results/modal_binding/REPRODUCE.md)

旧固定模态结果仅为历史实验；与新版业务合同不同，不能直接比较成功率。旧原始证据保留，不作为当前主线。
''')
    print(json.dumps(dict(rate=q['rate'],baseline=ref['rate'],difference=delta),ensure_ascii=False))


def figures(study):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']=['Noto Sans CJK JP','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    methods=list(LABELS);x=np.arange(len(methods));test=study['test']
    fig,ax=plt.subplots(1,2,figsize=(13,4.5))
    ax[0].bar(x,[test[m]['rate']*100 for m in methods],color=['#267d94']*3+['#9da8b7']*3)
    ax[0].set_ylabel('SLA成功率 (%)');ax[0].set_ylim(0,100)
    ax[1].bar(x,[test[m]['wire_per_success']/1000 for m in methods],color='#c08c45');ax[1].set_ylabel('有线KB / 成功合同')
    for a in ax:a.set_xticks(x,labels=[LABELS[m] for m in methods],rotation=35,ha='right');a.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(folder/'test.png',dpi=200);fig.savefig(folder/'test.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for method in methods[:3]:
        info=[read(OUT/'training'/method/str(i)/'training.json') for i in INITS]
        y=np.array([[r['validation']['rate'] for r in m['stages']] for m in info])*100
        steps=[r['step'] for r in info[0]['stages']]
        ax.plot(steps,y.mean(0),marker='o',label=LABELS[method]);ax.fill_between(steps,y.min(0),y.max(0),alpha=.12)
    ax.set_xlabel('每模型训练交互数');ax.set_ylabel('验证SLA成功率 (%)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'learning.png',dpi=200);fig.savefig(folder/'learning.svg');plt.close(fig)

if __name__=='__main__':main()
