"""Objective reporting for the frozen research-point-1 RL revision."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from slice_alloc.tools import modal_binding_rl_run as run
from slice_alloc.tools.modal_binding_rl_registry import all_specs
from slice_alloc.tools.modal_binding_report import summary

LABELS = dict(full='增强观测＋整形RL', fixed='固定模态RL', no_progress='旧观测＋整形RL',
    no_shaping='增强观测无整形RL', first_fit='首次可行', fixed_price='固定模态指数定价',
    joint_price='联合模态指数定价', ndn_only='仅NDN', semantic_rule='业务语义与移动规则', ppo='Maskable PPO', legacy='既有联合模态RL（8步）')


def paired(left, right, samples=20000):
    li = sorted({r['init'] for r in left})
    ri = sorted({r['init'] for r in right})
    # Train-variant ablations use common initializations, not unpaired 5-vs-3.
    matched = len(ri) > 1
    if matched:
        li = ri = sorted(set(li) & set(ri))
        left = [r for r in left if r['init'] in li]
        right = [r for r in right if r['init'] in ri]
    seeds = sorted({r['seed'] for r in left})
    assert set(seeds) == {r['seed'] for r in right}

    def matrix(rows, inits):
        data = defaultdict(float)
        for row in rows:
            data[row['init'], row['seed']] += row['success']
        return np.array([[data[i, s] for s in seeds] for i in inits])

    a, c = matrix(left, li), matrix(right, ri)
    denominator = np.array([sum(r['requests'] for r in left if r['seed'] == s and r['init'] == li[0]) for s in seeds])
    rng = np.random.default_rng(912631)
    draws = []
    for _ in range(samples):
        scene = rng.integers(len(seeds), size=len(seeds))
        ai = rng.integers(len(li), size=len(li))
        ci = ai if matched else rng.integers(len(ri), size=len(ri))
        draws.append((a[ai][:, scene].mean(0).sum()-c[ci][:, scene].mean(0).sum())/denominator[scene].sum())
    return dict(difference=summary(left)['rate']-summary(right)['rate'],
        ci95=np.quantile(draws, [.025, .975]).tolist(), clusters=len(seeds),
        left_initializations=li, right_initializations=ri, matched_initializations=matched)


def plots(study):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    out = run.OUT/'figures'
    out.mkdir(exist_ok=True)
    methods = list(LABELS)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    values = [study['test'][m]['rate']*100 for m in methods]
    ax.barh([LABELS[m] for m in methods], values, color=['#126782' if m == study['deployment']['method'] else '#a3bac3' for m in methods])
    for i, v in enumerate(values):
        ax.text(v+.5, i, f'{v:.2f}%', va='center', fontsize=9)
    ax.set(xlim=(0, 100), xlabel='SLA成功合同 / 全部到达合同（%）')
    ax.invert_yaxis()
    fig.savefig(out/'test.png', dpi=180)
    fig.savefig(out/'test.svg')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for method in (*run.METHODS, 'ppo'):
        records = [run.read(run.OUT/'training'/m/str(i)/'training.json') for m, i in all_specs() if m == method]
        rates = np.array([[r['validation']['rate'] for r in x['stages']] for x in records])*100
        ax.plot(run.BUDGETS, rates.mean(0), marker='o', label=f'{LABELS[method]}（n={len(records)}）')
        ax.fill_between(run.BUDGETS, rates.min(0), rates.max(0), alpha=.1)
    ax.set(xlabel='每模型训练决策数', ylabel='验证集SLA成功率（%）')
    ax.legend(fontsize=9)
    fig.savefig(out/'learning.png', dpi=180)
    fig.savefig(out/'learning.svg')
    plt.close(fig)


def main():
    rows = run.read(run.OUT/'evaluation.json')
    selection = run.read(run.OUT/'selection.json')
    verification = run.read(run.OUT/'verification.json')
    rl_choice = run.read(run.OUT/'rl_selection.json')
    main_method = rl_choice['method']
    assert verification['passed'] and verification['evaluation_sha256'] == run.sha(run.OUT/'evaluation.json')
    test = [r for r in rows if r['scope'] == 'test']
    stats = {m: summary([r for r in test if r['method'] == m]) for m in LABELS}
    comparisons = {m: paired([r for r in test if r['method'] == main_method],
        [r for r in test if r['method'] == m]) for m in LABELS if m != main_method}
    full_ablations = {m: paired([r for r in test if r['method'] == 'full'],
        [r for r in test if r['method'] == m]) for m in ('fixed', 'no_progress', 'no_shaping')}
    ood = {}
    for mix, load in [('object_heavy', 80), ('mobile_heavy', 80), ('balanced', 112), ('novel', 80)]:
        group = [r for r in rows if r['scope'] == 'ood' and r['mix'] == mix and r['load'] == load]
        ood[f'{mix}-{load}'] = {m: summary([r for r in group if r['method'] == m]) for m in LABELS}
    best = selection['selected_rule']
    study = dict(test=stats, paired=comparisons, full_ablations=full_ablations, best_validation_rule=best, ood=ood,
        by_load={str(l): {m: summary([r for r in test if r['method'] == m and r['load'] == l]) for m in LABELS} for l in run.LOADS},
        episodes=len(rows), training_steps=verification['trained_steps'],
        deployment=rl_choice,
        note='Ablation comparisons use only common initializations. Primary comparison is validation-selected rule; other intervals are exploratory, not multiplicity adjusted.')
    run.write(run.OUT/'analysis.json', study)
    plots(study)
    q, ref, delta = stats[main_method], stats[best], comparisons[best]
    n_inits = len(q['individual_rates'])
    if delta['ci95'][0] > 0:
        verdict = '在本轮已声明环境和新留出样本内，主要比较支持RL优于验证集选定规则；不推广为任意网络上的算法保证。'
    elif delta['ci95'][1] < 0:
        verdict = '本轮RL低于验证集选定规则，改进后的学习设计仍不足以支持总体性能优越性。可以使用RL作为第一研究点的方法实现，但不能写成已优于强规则的性能成果。'
    else:
        verdict = '主要比较的区间覆盖零，不能确认RL显著优于验证集选定规则。可以将RL作为研究点1的方法实现，性能结论须保留这一限制。'
    fmt = lambda x: f'{100*x:.2f}%'
    lines = ['# 研究点1：RL修订版最终结果', '',
        f"验证集选定的联合模态方法为 **{LABELS[main_method]}**，其{n_inits}个独立单模型在新常规测试上的平均SLA成功率为 **{fmt(q['rate'])}**；验证集选定规则“{LABELS[best]}”为 **{fmt(ref['rate'])}**。差值 **{100*delta['difference']:+.2f}个百分点**，场景/初始化配对bootstrap的95%区间 **[{100*delta['ci95'][0]:+.2f}, {100*delta['ci95'][1]:+.2f}]个百分点**。", '', verdict, '',
        f'本轮以RL为研究点1的主方法，不使用ensemble、专家规则兜底或示范训练。主部署初始化11预先固定；{n_inits}模型均值只是研究统计，不是集成推理。', '',
        '在新测试开启前补充冻结rl_selection_addendum.json、ppo_protocol.json与legacy_protocol.json：三个修订QR-DQN、Maskable PPO及不可改动的既有联合模态QR-DQN，按共同初始化11/23/37的新验证均值选择，平局采用预先声明顺序。既有模型的检查点仍按原验证集选定，不在新验证集重新挑其检查点；全部5个初始化保留。原protocol中固定full为主方法的条款被补充协议替代；物理合同和新测试划分不变。PPO与QR架构不同，配方间比较不能当作只改变优化器的消融。选择锁定见rl_selection.json，代码检查其时间早于正式测试清单。', '',
        f"预定部署单模型初始化11的常规测试成功率为 **{fmt(q['individual_rates']['11'])}**。全部初始化："+', '.join(f"{i}: {fmt(r)}" for i, r in q['individual_rates'].items())+'。', '',
        '![常规测试](figures/test.png)', '', '## 设计与范围', '',
        '![低空多模态切片系统](figures/system.png)', '',
        '低空场景、业务合同、IP/受限EID-RLOC/NDN执行机制和17维共享资源容量沿用v1。请求只有业务语义，RL联合决定接纳、承载模态、边缘/映射/缓存位置。切片为租户/SLA/生命周期/资源预约合同；没有每租户无线调度器或确定性QoS保证。所有业务是下行，IP采用准入时定位绑定，未实现IP连接迁移，ID不是完整LISP。', '',
        '候选修订增加实际在途交付进度与全体当前移动观测，修复首次速度有效标志；比较32步Double QR-DQN采用/去掉势函数整形、采用/去掉新增观测的版本。完整版本整回合整形奖励严格等于原生成功数。最终采用哪些模块由验证集选择，不能宣称被撤除模块已证明有用。新增信息不改变逐包物理执行。状态仍是部分可观测汇总，未证明Markov充分性。', '',
        '只有研究点1完成。需求预测、CVaR和运行期主动模态调整均未实现；回报分位数不当作研究点2，协议被动切换不当作研究点3。', '',
        '## 常规留出比较', '',
        '| 方法 | SLA成功率 | 接纳率 | 有线KB/成功合同 | 预约秒/成功合同 | 平均决策ms |',
        '|---|---:|---:|---:|---:|---:|']
    for method, label in LABELS.items():
        s = stats[method]
        lines.append(f"| {label} | {fmt(s['rate'])} | {fmt(s['admitted'])} | {s['wire_per_success']/1000:.2f} | {s['reservation_per_success']:.3f} | {s['decision_ms']:.3f} |")
    modes = q['modes']
    mode_total = sum(modes.values())
    d = comparisons['fixed']
    lines += ['', f"主方法相对固定模态RL：{d['difference']*100:+.2f}个百分点，配对95%区间[{d['ci95'][0]*100:+.2f}, {d['ci95'][1]*100:+.2f}]；采用共同初始化比较。区间覆盖零时不能据此宣称模态选择带来稳定SLA提升。", '',
        '主方法在已接纳切片中的协议使用比例：'+', '.join(f"{m}: {100*v/max(1, mode_total):.2f}%" for m, v in modes.items())+'。协议使用率只说明选择行为，本身不证明其选择有效。', '',
        '| 分组 | 主RL成功率 | 验证集选定规则成功率 |', '|---|---:|---:|']
    for load in run.LOADS:
        group = study['by_load'][str(load)]
        lines.append(f"| {load}到达 | {fmt(group[main_method]['rate'])} | {fmt(group[best]['rate'])} |")
    for kind, name in [('telemetry', '态势图像流'), ('object', '共享地图'), ('command', '任务命令')]:
        lines.append(f"| {name} | {fmt(q['by_service'][kind]['rate'])} | {fmt(ref['by_service'][kind]['rate'])} |")
    if main_method in ('ppo', 'legacy'):
        lines += ['', '主方法为PPO或既有8步QR时，固定模态对照采用本轮增强观测/32步QR，因此还改变了学习配方，不能单独归因模态选择。受控的同配方QR模态消融见下表；本轮没有固定模态PPO。']
    lines += ['', 'S是训练目标；M是次要观察，V是固定留出分布上的零更新表现，不声称联合优化SMV。KB使用1000字节；预约秒为容量归一化代理成本，不能直接当作能耗或经济价值。决策时间在并行评测下测得，受CPU争用影响，不作为严格的部署延迟基准。', '',
        '## 消融与学习过程', '',
        '| 对照 | 完整RL−对照（百分点） | 配对95%区间 | 共同初始化数 |', '|---|---:|---:|---:|']
    for m in ('fixed', 'no_progress', 'no_shaping'):
        d = full_ablations[m]
        lines.append(f"| {LABELS[m]} | {d['difference']*100:+.2f} | [{d['ci95'][0]*100:+.2f}, {d['ci95'][1]*100:+.2f}] | {len(d['left_initializations'])} |")
    lines += ['', '旧观测消融移除整套新增信息；无整形保持32步窗口；固定模态采用与v1相同的业务母语指派。消融按共同初始化配对，不能用5对3的不配对均值代替。除主要规则比较外，各区间是探索性分析，未做多重比较校正。', '', '![验证学习曲线](figures/learning.png)', '',
        '| 版本 | 初始化 | 选中步数 | 验证成功率 | 常规测试成功率 |', '|---|---:|---:|---:|---:|']
    for m, i in all_specs():
        x = run.read(run.OUT/'training'/m/str(i)/'training.json')
        validation = x['fresh_validation'] if m == 'legacy' else x['selected']['validation']
        lines.append(f"| {LABELS[m]} | {i} | {x['selected']['step']} | {fmt(validation['rate'])} | {fmt(stats[m]['individual_rates'][str(i)])} |")
    lines += ['', '## 分布外结果（不更新模型）', '', '| 分布 | '+ ' | '.join(LABELS.values())+' |', '|---|'+ '---:|'*len(LABELS)]
    for name, values in ood.items():
        lines.append('| '+name+' | '+' | '.join(fmt(values[m]['rate']) for m in LABELS)+' |')
    lines += ['', 'object_heavy为地图占比70%；沿用场景名mobile_heavy实际为命令占比55%的组合，不改变移动端点比例，不能称作未见移动模型；balanced-112改变到达数；novel改变为6块/350ms地图合同，不是全新业务类别。', '', '## 复现与诚实边界', '',
        f"本轮19个模型各65536次训练决策，总计{verification['trained_steps']}次新训练交互；另复用5个既有联合模态模型，其历史训练共163840次、每模型32768次，不计入新训练数。完整与固定模态各5初始化；旧观测、无整形与PPO各3初始化。新模型在四个检查点验证，按验证成功数选择、平局优先较早；全部初始化保留。验证/规则调参和正式评测开销另计。新常规测试种子27000—27029，OOD28000—28014；旧23000系列测试已用于诊断，不再视为未见验收。", '',
        f"本轮{len(rows)}个正式回合全部在评测时逐包审计，交付时又独立重读全部报文轨迹重建期望数、按时数和SLA；{len(all_specs())}个已选模型各重放1个代表场景，完整决策/观测/结算/逐包事件日志完全一致。源码、原生二进制、协议和模型SHA256已核对。执行一致性不证明最优性或创新性。", '',
        '方法使用已有QR-DQN、Double-Q和势函数整形；论文贡献应限定于低空业务驱动的多模态切片联合建模、观测/回报设计和原生受控验证，不能把算法名称组合宣称成新算法理论。R05只提供架构语境。', '',
        '依据：[R05](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT250949)、[QR-DQN](https://arxiv.org/abs/1710.10044)、[Double DQN](https://arxiv.org/abs/1509.06461)、[势函数整形原始论文](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)、[PPO](https://arxiv.org/abs/1707.06347)。', '',
        '设计见[design.md](design.md)，机器可读统计见[analysis.json](analysis.json)，完整独立审计见[verification.json](verification.json)。']
    (run.OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(dict(main=main_method, rl=q['rate'], rule=best, rule_rate=ref['rate'], paired=delta, episodes=len(rows)))


if __name__ == '__main__':
    main()
