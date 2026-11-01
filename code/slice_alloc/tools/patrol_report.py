"""Render current results, including negative algorithm comparisons, from data."""
import collections
import json
from pathlib import Path
import numpy as np
from .patrol_run import OUT,ROOT,BASELINES,PROTOCOL_SPEC,sha,write

LABELS={'first_fit':'首次可行','near_edge':'近边优先','greedy':'资源贪心','price':'资源价格','cache_aware':'缓存优先价格规则','ddqn':'DDQN（5初始化）','ppo':'PPO（5初始化）'}


def main():
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['passed']
    data=json.loads((OUT/'analysis.json').read_text());evaluation=json.loads((OUT/'evaluation.json').read_text());rows=evaluation['rows'];stats=data['stats']
    assert data['evaluation_sha256']==sha(OUT/'evaluation.json')
    rng=np.random.default_rng(8091);paired={}
    def seed_rate(m,seed):
        selected=[r for r in rows if r['method']==m and r['profile']=='mixed' and r['seed']==seed]
        return sum(r['success'] for r in selected)/sum(r['requests'] for r in selected)
    for baseline in BASELINES+('ppo',):
        differences=np.array([seed_rate('ddqn',s)-seed_rate(baseline,s) for s in PROTOCOL_SPEC['test_seeds']]);samples=rng.choice(differences,size=(10000,len(differences)),replace=True).mean(1)
        paired[baseline]=dict(difference=float(differences.mean()),bootstrap95=np.quantile(samples,[.025,.975]).tolist())
    primary=dict(metric=PROTOCOL_SPEC['metric'],primary_profile='mixed',paired_ddqn=paired,evaluation_sha256=sha(OUT/'evaluation.json'));write(OUT/'primary_analysis.json',primary)
    best=max(BASELINES,key=lambda m:stats[m]['profiles']['mixed']['success_rate'])
    ddqn=stats['ddqn']['profiles']['mixed']['success_rate'];ppo=stats['ppo']['profiles']['mixed']['success_rate'];rule=stats[best]['profiles']['mixed']['success_rate']
    text=['# 低空巡检三模态平台：最终独立实验结果','',
          f"本轮完成12架无人机、双AP、双边缘的三模态联合初始分配；正式测试{data['episodes']}次原生交互执行、{data['decisions']}次申请决策，另有640次独立描述性验证。全部正式执行及验证从真实报文重构SLA，未删除失败。",'',
          f"**混合主场景：**DDQN为{ddqn*100:.2f}%，PPO为{ppo*100:.2f}%；最强规则为{LABELS[best]}，{rule*100:.2f}%。以下同时披露四类场景，主场景未按结果替换。",'',
          '## 设计与论文的对应','',
          '多模态体现在三种实际通信机制共用基础设施、资源账本与分配器：IP真实视频/遥测上行，ID稳定EID的移动通信，NDN同名同版本地图共享。低空体现在固定地面设施与80m高移动无人机的真实无线关联变化，进而影响路径、定位更新、缓存访问与交付。功能位置、控制报文和数据路径真实执行。',
          '', '本轮研究第一项初始分配；不把基本注册和关联更新冒充运行期主动重优化，不声称完整LISP或完整多模态网络体系。', '',
          '## 主指标与分场景结果','',
          '主指标是按真实包核验的请求SLA成功率：成功请求/全部申请，拒绝计失败。IP/ID要求至少95%的产生报文在合同期限内到达；NDN要求所有消费者、所有地图块按时。各模态需求不同，不能把跨模态成功率差异解释为协议优劣。','',
          '| 方法 | 混合主场景 | 视频带宽压力 | 映射压力 | 地图缓存压力 | 四场景合计 | 接纳率 |',
          '|---|---|---|---|---|---|---|']
    for m in BASELINES+('ddqn','ppo'):
        s=stats[m];values=' | '.join(f"{100*s['profiles'][p]['success_rate']:.2f}%" for p in PROTOCOL_SPEC['profiles'])
        text.append(f"| {LABELS[m]} | {values} | {100*s['success_rate']:.2f}% | {100*s['admission_rate']:.2f}% |")
    text+=['','## 混合主场景的配对区间','','30个独立种子内先合并负载和全部初始化，再以种子为单位进行10000次配对bootstrap。以下为DDQN减对照方法，单位为百分点；多方法比较属于限定场景下的探索性描述。','','| 对照 | 平均差值/百分点 | 95%配对区间 |','|---|---|---|']
    for m,v in paired.items():
        lo,hi=v['bootstrap95'];text.append(f"| {LABELS[m]} | {100*v['difference']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    comparison=paired[best]
    if ddqn>rule and comparison['bootstrap95'][0]>0:
        conclusion='DDQN在冻结混合场景中领先最强规则，配对区间不含0；结论只适用于此规模、合同和物理配置。'
    elif ddqn>rule:
        conclusion='DDQN的混合场景均值略高于最强规则，但配对区间不能支持稳健领先。'
    else:
        conclusion='本轮DDQN未超过最强规则；三模态机制和可执行分配已验证，强化学习的额外收益尚未证明。'
    text+=['',conclusion,'','## 分模态结果与初始化稳定性','','| 方法 | IP请求成功率 | ID请求成功率 | NDN请求成功率 | 推理/规则选择均值ms |','|---|---|---|---|']
    for m in BASELINES+('ddqn','ppo'):
        s=stats[m];values=' | '.join(f"{100*s['by_mode'][p]['success_rate']:.2f}%" for p in ('ip','id','ndn'))
        text.append(f"| {LABELS[m]} | {values} | {s['decision_ms']:.3f} |")
    text+=['','耗时仅为在已生成候选上选择动作的时间，不含候选编译、账本检查及原生推进；完整单次仿真耗时另外保留在evaluation.json，不能将此列当端到端调度延迟。','','| 学习方法 | 五初始化四场景成功率 | 样本标准差/百分点 |','|---|---|---|']
    for m in ('ddqn','ppo'):
        s=stats[m];text.append(f"| {m.upper()} | {' / '.join(f'{100*v:.2f}%' for v in s['initialization_rates'])} | {100*s['initialization_std']:.3f} |")
    text+=['','## 机制与资源证据','','近端IP部署349个业务包经过两段有线链路；远端部署多经过E1—E2，两者均按时交付349/349，额外跨边缘网络层字节为349×1037＝361913B。部署差异在实际链路上可见，不仅是Python费用数字不同。','',
           'ID在真实AP1→AP2关联变化前后解封装到同一EID。两种映射服务位置均产生真实查询；本用例分别观察到24/20次查询，不能用生命周期/TTL推算。移动中断和陈旧定位损失保留；身份机制提供位置解耦，不提供无覆盖连接或零中断。','',
           '相同56次NDN块交付：无缓存回源载荷28000B，租约缓存为4000B，减少85.71%；两者最后一跳Data编码字节相同。同名跨切片复用和“第一个切片结束不提前删除共享对象”均通过原生验证。该机制对照不等于整批请求回源必然减少同一比例。','',
           '独立开发种子中，视频场景出现有线约束、映射场景出现权威映射额度约束、地图场景出现缓存约束。正式结果的压力计数和实际资源见resource_reconstruction.json，不把候选冲突次数当独立统计样本。','',
           '![冻结正式测试结果](图/06_低空巡检三模态实验结果.png)','',
           '## 已验证范围与限制','',
           '固定预算4096步、5个初始化，未按验证或测试成绩挑选检查点。四类场景、两档24/40请求均使用相同2—18秒到达窗口；4架验证与12架主规模都实际运行。全部当前申请的协议、部署、SLA和原始记录均保留。受限范围传播、固定高度往返轨迹、有限边缘候选和共享队列限制了推广范围；没有CPU硬保障、一般三维机动、完整PIT/FIB硬隔离或运行期主动迁移。','',
           '初始评测汇总曾因重复method字段失败，修复后重新冻结并重训，原错误日志、协议、模型与输出保留于development_failures/evaluation_row_format。该错误不涉及容量、费用、训练预算或按测试成绩选参。','',
           '[完整分析](slice_alloc/results/patrol/analysis.json)；[主场景配对分析](slice_alloc/results/patrol/primary_analysis.json)；[冻结实验协议](slice_alloc/results/patrol/protocol.json)；[机制结果](slice_alloc/results/patrol/mechanisms.json)；[独立核验](slice_alloc/results/patrol/independent_audit.json)。']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=BASELINES+('ddqn','ppo');labels=['First fit','Near edge','Greedy','Price','Cache-aware','DDQN','PPO']
    fig,axes=plt.subplots(1,2,figsize=(13,4.6));x=np.arange(len(methods));colors=['#4477aa','#66ccee','#228833','#ccbb44']
    for j,p in enumerate(PROTOCOL_SPEC['profiles']):axes[0].bar(x+(j-1.5)*.19,[100*stats[m]['profiles'][p]['success_rate'] for m in methods],width=.18,color=colors[j],label=p)
    axes[0].set_xticks(x,labels,rotation=25);axes[0].set_ylabel('Successful requests / all requests (%)');axes[0].set_ylim(0,100);axes[0].legend(fontsize=8,ncol=2);axes[0].set_title('Frozen native test: 30 independent seed groups')
    for j,m in enumerate(('ddqn','ppo')):axes[1].plot(PROTOCOL_SPEC['initializations'],[100*v for v in stats[m]['initialization_rates']],marker='o',label=m.upper())
    axes[1].axhline(100*stats['cache_aware']['success_rate'],color='#228833',linestyle='--',label='Cache-aware rule (all profiles)');axes[1].set_xlabel('Initialization');axes[1].set_ylabel('All-profile request SLA success (%)');axes[1].legend(fontsize=8);axes[1].set_title('All five initializations; no selection')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(ROOT/f'图/06_低空巡检三模态实验结果.{ext}',dpi=170)
    plt.close(fig)
    write(OUT/'report_verification.json',dict(passed=True,report_sha256=sha(OUT/'report.md'),primary_analysis_sha256=sha(OUT/'primary_analysis.json'),analysis_sha256=sha(OUT/'analysis.json'),independent_audit_sha256=sha(OUT/'independent_audit.json'),generator_sha256=sha(__file__)))
    print(conclusion,flush=True)

if __name__=='__main__':main()
