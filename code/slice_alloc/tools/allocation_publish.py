"""Publish the verified v3 first-part study, preserving the previous manuscript."""
import json
import shutil
import hashlib
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote
from slice_alloc.tools.allocation_study import OUT, ROOT, hashes
from slice_alloc.tools.patrol_run import sha, write
from slice_alloc.tools.export_document import to_docx


LABELS={'prior_ppo':'本方法：资源先验PPO','plain_ppo':'普通PPO（消融）',
        'normalized_plain_ppo':'普通PPO奖励缩放（补充消融）','first_fit':'首次可行',
        'near_edge':'近边优先','greedy':'资源贪心','price':'价格规则',
        'cache_aware':'缓存优先','lifetime_price':'生命周期价格规则'}
MAIN_METHODS=('prior_ppo','plain_ppo','first_fit','price','lifetime_price')


def table_rows(analysis,methods,internal_names=False):
    lines=['| 方法 | 总成功率 | 64请求 | 80请求 | 接纳率 | 初始化标准差/百分点 |',
           '|---|---:|---:|---:|---:|---:|']
    for m in methods:
        s=analysis['stats'][m];label=LABELS.get(m,m)
        if internal_names:label+=f'（{m}）'
        std='—' if s['initialization_std'] is None else f'{100*s["initialization_std"]:.3f}'
        lines.append(f'| {label} | {100*s["success_rate"]:.2f}% | {100*s["by_load"]["64"]:.2f}% | '
                     f'{100*s["by_load"]["80"]:.2f}% | {100*s["admission_rate"]:.2f}% | {std} |')
    return '\n'.join(lines)


def report_text(analysis,protocol,evaluation):
    primary=protocol['primary_method'];assert primary=='prior_ppo'
    rows=[r for r in evaluation['rows'] if r['method']==primary]
    total=sum(r['requests'] for r in rows)
    rejected=sum(r['requests']-r['admitted'] for r in rows)
    failed=sum(r['admitted']-r['success'] for r in rows)
    score=100*analysis['stats'][primary]['success_rate']
    lines=['# 基于PPO的低空三模态切片联合分配：实验报告','',
           f'第一部分采用一个主方法：基于资源先验的PPO，冻结测试成功率{score:.2f}%。普通PPO是去掉先验的消融，价格规则等是传统基线。所有方法共用IP／EID-RLOC／NDN业务、双边缘双AP、12架无人机和同一资源账本。','',
           '## 主方法与对照','',
           '主方法使用一个候选感知Actor-Critic网络。资源价格作为动作偏好，PPO学习可调整的评分；三种模态共用同一策略。奖励为真实结算的SLA成功数。主方法没有模仿预训练、奖励塑形或参照回报训练。',
           '普通PPO去掉整个先验项（含拒绝初始偏置），其余网络、观测、奖励和策略交互预算相同。价格规则是无学习对照；生命周期价格规则还利用已经签订合同的释放时间。','',
           '## 主要实验结果','',table_rows(analysis,MAIN_METHODS),'',
           '主指标为满足SLA的请求数／全部到达请求数，拒绝计失败。64/80请求均在2—18秒到达；物理容量和SLA未调整。两负载合计进入75%—82%目标范围，不表示每种负载或模态都进入该范围。旧24/40负载及四类场景合计成绩不能直接当作本方法的优化前成绩。','',
           '## 改进是否有效','',
           '| 本方法−对照 | 差值/百分点 | 初始化＋场景95%区间 |',
           '|---|---:|---|']
    for other in ('plain_ppo','first_fit','price','lifetime_price'):
        c=analysis['comparisons'][primary][other];lo,hi=c['initialization_and_scenario_ci95']
        lines.append(f'| {LABELS[other]} | {100*c["difference"]:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
    difference=100*analysis['comparisons'][primary]['plain_ppo']['difference']
    best=max(protocol['baselines'],key=lambda m:analysis['stats'][m]['success_rate'])
    lines+=['',f'本方法相对普通PPO提升{difference:.2f}个百分点，区间不含0。最强规则为{LABELS[best]}，'
            f'{100*analysis["stats"][best]["success_rate"]:.2f}%；本方法尚未证明优于最强规则。价格先验改善了PPO，但不能把先验贡献全归为强化学习的收益。','',
            '主方法全部初始化：'+' / '.join(f'{100*v:.2f}%' for v in analysis['stats'][primary]['initialization_rates'])+'。','',
            '## 三模态与失败构成','',
            '| 固定模态 | 请求数 | 接纳数 | SLA成功数 | 成功率 |','|---|---:|---:|---:|---:|']
    for mode in ('ip','id','ndn'):
        n=sum(r['by_mode'][mode]['requests'] for r in rows)
        admitted=sum(r['by_mode'][mode]['admitted'] for r in rows)
        successful=sum(r['by_mode'][mode]['success'] for r in rows)
        lines.append(f'| {mode} | {n} | {admitted} | {successful} | {100*successful/n:.2f}% |')
    lines+=['','不同模态的合同、方向和业务量不同，分模态成绩不作为协议优劣排名。',
            f'主方法全部{total}次请求中，拒绝{rejected}次（{100*rejected/total:.2f}%），接纳后SLA失败{failed}次（占全部请求的{100*failed/total:.2f}%）。这不是后续模块必然能挽回的收益。','',
            '## 实验与验证','',
            f'主方法与普通PPO各5个初始化，每模型32768次策略交互，学习率1e-4、rollout512、batch128、5轮更新、gamma=1、GAE lambda=1、entropy=0.005、clip=0.1。配置在开发7000—7007上选定，冻结后仅执行最终8000—8029。正式执行{analysis["episodes"]}次仿真、{analysis["decisions"]}次请求决策；全部结果保留。',
            '原始包记录重建SLA、所有已接纳预约复验、相同输入核对与15个检查点原生事件重放均通过。有限观测不保证完整Markov性，平均工作量预约不等于无线逐包硬保障。','',
            '## 与后续研究的关系','',
            '第一部分输出初始部署、资源预约和运行记录；第二部分以后预测实际资源需求，第三部分以后利用预测进行运行期QoS调整。后两项尚未完成，不赋予预期成功率。累计消融须沿用相同物理资源、业务和SLA，并使用预留的9000—9029种子。','',
            '## 附录：完整记录与开发成本','',
            '以下是冻结结果的完整记录，不是多个论文主算法。奖励缩放、潜势和配对参照等未采用的尝试保留用于复现。表中内部标识仅用于定位模型文件。','',
            table_rows(analysis,list(analysis['stats']),True),'',
            '| 本方法−对照（内部标识） | 差值/百分点 | 场景95%区间 | 初始化＋场景95%区间 |',
            '|---|---:|---|---|']
    for other,c in analysis['comparisons'][primary].items():
        lo,hi=c['paired_scenario_ci95'];l2,h2=c['initialization_and_scenario_ci95']
        lines.append(f'| {other} | {100*c["difference"]:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | [{100*l2:+.2f}, {100*h2:+.2f}] |')
    budget=protocol['development_training_budget']
    lines+=['',f'全部开发尝试共{budget["models"]}个模型、{budget["native_decisions"]}次策略交互，额外参照交互{budget["reference_decisions"]}次，另有512步接口冒烟测试。未采用的配对版本额外运行price产生训练标签，因此其总原生仿真预算更大；主方法不使用该机制。','',
            '[当前设计](../../design.md) · [冻结协议](protocol.json) · [完整结果](analysis.json) · [逐包核验](audit.json) · [训练指纹](training_audit.json) · [负载扫描](load_calibration.json)']
    return '\n'.join(lines)+'\n'


def root_links(markdown,base):
    def replace(match):
        target=match.group(2)
        if '://' in target or target.startswith('#'):return match.group(0)
        path,separator,fragment=target.partition('#')
        relative=(base/unquote(path)).resolve().relative_to(ROOT.resolve())
        return match.group(1)+str(relative)+(separator+fragment if separator else '')+')'
    return re.sub(r'(!?\[[^\]]*\]\()([^)]+)\)',replace,markdown)


def main():
    audit=json.loads((OUT/'audit.json').read_text())
    delivery=json.loads((OUT/'delivery.json').read_text())
    assert audit['passed'] and audit['sources']==hashes()
    assert all(sha(OUT/name)==value for name,value in delivery['artifacts'].items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    analysis=json.loads((OUT/'analysis.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    primary=protocol['primary_method']
    best_rule=max(protocol['baselines'],key=lambda m:analysis['stats'][m]['success_rate'])
    text=report_text(analysis,protocol,json.loads((OUT/'evaluation.json').read_text()))
    (OUT/'report.md').write_text(text)
    to_docx(text).save(OUT/'report.docx')
    for name in ('report.md','report.docx'):delivery['artifacts'][name]=sha(OUT/name)
    methods=list(MAIN_METHODS)
    plot_labels={'prior_ppo':'Proposed PPO','plain_ppo':'Plain PPO','first_fit':'First fit',
                 'price':'Price rule','lifetime_price':'Lifetime price'}
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    x=np.arange(len(methods))
    for j,load in enumerate(protocol['loads']):
        axes[0].bar(x+(j-.5)*.36,[100*analysis['stats'][m]['by_load'][str(load)] for m in methods],
                    width=.36,label=f'{load} arrivals')
    axes[0].set_xticks(x,[plot_labels[m] for m in methods],rotation=25,ha='right',fontsize=9)
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Successful SLA requests / all arrivals (%)')
    axes[0].legend();axes[0].set_title('Frozen primary test, unchanged physical resources')
    for method in (primary,'plain_ppo'):
        values=analysis['stats'][method]['initialization_rates']
        axes[1].plot(protocol['initializations'],np.array(values)*100,marker='o',label=plot_labels[method])
    axes[1].axhline(100*analysis['stats']['price']['success_rate'],linestyle='--',color='black',label='Price rule')
    if best_rule!='price':
        axes[1].axhline(100*analysis['stats'][best_rule]['success_rate'],linestyle=':',color='gray',label='Best rule: '+plot_labels.get(best_rule,best_rule))
    axes[1].set_ylim(0,100);axes[1].set_xlabel('Training initialization')
    axes[1].set_ylabel('Primary success (%)');axes[1].legend(fontsize=8)
    axes[1].set_title('Every initialization retained')
    fig.tight_layout()
    figure='图/07_v3初始分配冻结实验结果'
    for ext in ('png','svg'):fig.savefig(ROOT/(figure+'.'+ext),dpi=170)
    plt.close(fig)
    old=OUT/'previous_manuscript'
    title='面向低空智联的多模态网络切片技术'
    document_record=ROOT/'slice_alloc/results/document_verification.json'
    canonical=(ROOT/'slice_alloc/design.md').read_text()
    assert canonical.startswith('# 基于PPO的低空三模态网络切片联合分配')
    design=root_links(canonical,ROOT/'slice_alloc')
    report=root_links((OUT/'report.md').read_text().split('\n',1)[1],OUT)
    manuscript=design+'\n## 第一部分：当前冻结实验结果\n'+report+'\n![PPO主方法与对照]('+figure+'.png)\n'
    (ROOT/(title+'.md')).write_text(manuscript)
    to_docx(manuscript).save(ROOT/(title+'.docx'))
    a=json.loads((OUT/'analysis.json').read_text())
    readme=f'''# 面向低空智联的多模态网络切片技术

第一部分只采用一个主算法：**基于资源先验的PPO联合分配**。它在同一策略中处理IP、EID/RLOC和NDN请求，决定是否接纳、功能放在哪个边缘以及缓存配置。普通PPO是消融对照，价格规则是传统基线。

场景为双边缘、双AP和12架巡检无人机。真实移动影响关联、路径和定位开销；三模态竞争同一个资源账本。当前负载64／80请求，沿用原物理容量和SLA。

| 角色 | 方法 | 冻结成功率 |
|---|---|---:|
| 主算法 | 资源先验PPO | {100*a['stats']['prior_ppo']['success_rate']:.2f}% |
| 消融 | 普通PPO，去掉整个先验项 | {100*a['stats']['plain_ppo']['success_rate']:.2f}% |
| 传统基线 | 价格规则 | {100*a['stats']['price']['success_rate']:.2f}% |
| 传统强基线 | 生命周期价格规则 | {100*a['stats']['lifetime_price']['success_rate']:.2f}% |

主算法达到75%—82%的合计目标，相对普通PPO提升{100*a['comparisons']['prior_ppo']['plain_ppo']['difference']:.2f}个百分点，尚未证明超过最强规则。完整结果、补充消融和开发成本在报告附录。

[完整设计](slice_alloc/design.md) · [实验报告](slice_alloc/results/allocation_v3/report.md) · [Word正文]({title}.docx) · [正文]({title}.md) · [交付核验](slice_alloc/results/allocation_v3/delivery.json)

正式执行{a['episodes']}次、{a['decisions']}次请求决策，全部原始结果保留。主方法与普通PPO各5个初始化、每模型32768次策略交互，原始SLA奖励，无模仿预训练或参照回报训练。

## 核验与复现

使用sdn环境执行 `python -B -m unittest discover -s slice_alloc/tests -q`；训练来源核对为 `python -B -m slice_alloc.tools.allocation_admission check`。算法实现见 [PPO策略](slice_alloc/src/allocation_policy.py)，训练、候选与观测见 [实验入口](slice_alloc/tools/allocation_study.py)。

实际训练参数、源码版本、模型、种子和完整预算见[冻结协议](slice_alloc/results/allocation_v3/protocol.json)，原始包核对见[逐包核验](slice_alloc/results/allocation_v3/audit.json)。开发命令入口仍保留用于追溯，不能在已经打开的最终测试集上继续调参或覆盖模型；新研究须使用新目录和留出集。

## 论文三部分

第一部分输出初始部署、预约和运行记录；第二部分以后预测实际资源需求；第三部分以后根据预测进行运行期QoS调整。后两部分尚未实现，也没有实验成功率。后续累计消融保持相同物理资源、业务和SLA。

旧实验与未采用开发尝试保留作历史。冻结实验数据不变，本次只将设计、图表和正文统一为一个PPO主方法。此前表述保存在 `slice_alloc/results/allocation_v3/editorial_before_single_ppo/`，更早旧稿及Word恢复记录在 `previous_manuscript/`。
'''
    (ROOT/'README.md').write_text(readme)
    checked_links=0
    for file in (ROOT/'README.md',ROOT/(title+'.md'),OUT/'report.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',file.read_text()):
            if '://' in target or target.startswith('#'):continue
            path=unquote(target.split('#',1)[0].strip('<>'))
            assert (file.parent/path).exists(),(str(file),target)
            checked_links+=1
    with zipfile.ZipFile(ROOT/(title+'.docx')) as z:
        xml=ET.fromstring(z.read('word/document.xml'))
        word_text=''.join(t.text or '' for t in xml.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'))
        for method,s in a['stats'].items():
            assert method in word_text
            assert f'{100*s["success_rate"]:.2f}%' in word_text
        assert any(hashlib.sha256(z.read(n)).hexdigest()==sha(ROOT/(figure+'.png'))
                   for n in z.namelist() if n.startswith('word/media/'))
    write(document_record,dict(passed=True,scope='Current v3 first-part report and manuscript only; previous record archived',
          local_links_checked=checked_links,word_contains_current_method_scores=True,
          word_contains_current_result_figure=True,analysis_sha256=sha(OUT/'analysis.json'),
          protocol_sha256=sha(OUT/'protocol.json'),audit_sha256=sha(OUT/'audit.json'),
          md_sha256=sha(ROOT/(title+'.md')),docx_sha256=sha(ROOT/(title+'.docx')),
          design_source_sha256=sha(ROOT/'slice_alloc/design.md'),generator_sha256=sha(__file__),
          renderer_sha256=sha(ROOT/'slice_alloc/tools/export_document.py')))
    files=['.gitignore','README.md','slice_alloc/design.md','slice_alloc/allocation_design_v3.md',
           'slice_alloc/tools/allocation_publish.py','slice_alloc/tests/test_allocation_study.py',
           'slice_alloc/results/document_verification.json','slice_alloc/tools/export_document.py',
           'slice_alloc/tools/report.py','slice_alloc/tests/test_report.py',
           title+'.md',title+'.docx',figure+'.png',figure+'.svg']
    delivery['manuscript_artifacts']={f:sha(ROOT/f) for f in files}
    delivery['previous_manuscript_artifacts']={str(f.relative_to(ROOT)):sha(f) for f in old.iterdir() if f.is_file()}
    write(OUT/'delivery.json',delivery)
    print('Current report, design, README and Word synchronized; previous manuscript preserved.')


if __name__=='__main__':main()
