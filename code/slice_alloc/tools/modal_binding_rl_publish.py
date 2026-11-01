"""Publish local thesis artifacts after independent native verification."""
from pathlib import Path
import shutil

from slice_alloc.tools import modal_binding_rl_run as run
from slice_alloc.tools.export_document import to_docx


def final_design(out, method, config):
    original = out/'development_design.md'
    if not original.exists():
        shutil.copy2(out/'design.md', original)
    candidate = original.read_text()
    common = candidate.split('## 观测和学习修订')[0]
    common = common.replace('（RL修订版）', '（最终RL版本）').replace('主方法为验证集选定的单模型、带可行性掩码的Double QR-DQN', '主方法为验证集选定的单模型、带可行性掩码的强化学习策略')
    details = (
        '采用既有的单模型Masked Double QR-DQN，91维状态包含当前业务合同、17维资源在当前/1/3/6秒的已签预约、当前请求端点几何/关联及历史位移速度；候选45维，最多11动作。首帧速度置零，保留既定编码中的常量有效位，不将它视为真实速度已知的保证。32个回报分位数取均值选择合法动作，8步回报、gamma=1，原生到期SLA成功数作奖励。没有势函数整形、全体端点扩展或在途交付计数作为模型输入。\n\n学习率1e-4、batch128、回放20000、warmup1024、每4步更新、目标网络1000步、宽96，探索16384步降到0.05；每初始化历史训练32768步。本轮复用经SHA256锁定的5个模型，在全新留出种子前向验证，不重挑原检查点。'
        if method == 'legacy' else
        'Maskable PPO使用258维可观测状态和同样的候选/容量掩码；双128层actor/critic，学习率3e-4、512步rollout、batch128、10epochs、gamma=1、GAE lambda=1、clip0.2、entropy0.01、vf系数0.5、梯度上限0.5。奖励为原生到期SLA成功数，不用势函数；每初始化65536次训练，3个初始化全部报告。'
        if method == 'ppo' else
        f"Masked Double QR-DQN采用{'91维原观测' if method == 'no_progress' else '258维增强观测'}、45维候选和最多11动作。32个回报分位数取均值选合法动作；32步回报、gamma=1，{'势函数系数0.5，终止势0，回合总奖励严格等于原生成功数' if method != 'no_shaping' else '原生到期SLA成功数奖励，不用势函数'}。学习率1e-4、batch128、回放20000、warmup1024、每4步更新、目标网络1000步、宽96，探索16384步降到0.05；每初始化65536次训练。增强观测只有在实际采用时才包含当前全体位置/关联与在途进度汇总，不读取未来。")
    text = common+'## 最终采用的RL配置\n\n'+f'验证集选定 **{method}**：{config[method]}。初始化11在新测试前固定为主部署单模型，不使用模型集成、专家规则回退或测试选择。所有候选版本都将模态作为动作；legacy指既有联合模态RL，不是请求预先带模态标签的老问题。\n\n'+details+'\n\n'
    text += '## 选择、验收与三研究点接口\n\n在新测试开启前，按共同初始化11/23/37的新验证成功率选择RL配方；所有初始化及失败修订完整报告。五条强规则仍在相同验证集调阈值，主要规则对照也只按验证选择。新常规测试27000—27029、分布外28000—28014，全部逐包重算SLA；完整统计与置信区间见report.md和analysis.json。\n\n研究点1输出初始接纳、模态绑定、功能位置、资源预约和真实观测；研究点2以后预测资源需求，研究点3以后做运行期主动QoS调整。本轮不实现预测、CVaR或模态迁移，不套用R05自由能/Tikhonov收敛结论。模态到功能使用固定模板；资源数量由业务和协议包络计算，连续带宽与CPU份额优化不属于本轮。\n\n候选设计与本轮1245184次新增训练的试错证据保存在development_design.md、development_formulation.md和完整报告中；未采用的模块不进入主方法。\n'
    (out/'design.md').write_text(text)
    return text


def main():
    out, root = run.OUT, run.b.p.ROOT
    verified = run.read(out/'verification.json')
    study = run.read(out/'analysis.json')
    choice = run.read(out/'rl_selection.json')
    assert verified['passed'] and verified['evaluation_sha256'] == run.sha(out/'evaluation.json')
    report = (out/'report.md').read_text()
    method = choice['method']
    config = dict(
        full='258维增强观测，32步QR-DQN，势函数系数0.5，每初始化65536次训练',
        no_progress='91维原观测，32步QR-DQN，势函数系数0.5，每初始化65536次训练',
        no_shaping='258维增强观测，32步QR-DQN，原生稀疏奖励，每初始化65536次训练',
        ppo='258维增强观测，Maskable PPO，原生稀疏奖励，每初始化65536次训练',
        legacy='91维观测，8步单模型联合模态QR-DQN，原生稀疏奖励，复用每初始化32768次训练的不可改动模型')
    design = final_design(out, method, config)
    formulation = (out/'formulation.md').read_text()
    candidate_math = out/'development_formulation.md'
    if not candidate_math.exists():
        candidate_math.write_text(formulation)
    formulation = candidate_math.read_text()
    prefix = formulation.split('## 目标保持的整形与分位数学习')[0]
    tail = formulation.split('## 三研究点的接口')[1]
    if method == 'ppo':
        algorithm = '## 所选策略梯度方法\n\nPPO在可行性掩码限定的动作分布上优化clip目标 E[min(ratio·A,clip(ratio,1−0.2,1+0.2)·A)]，并加入熵正则和价值损失。优势由gamma=1、lambda=1的GAE计算，真实终止状态不bootstrap。原生回合总奖励为ΣY_i。PPO原始算法和动作掩码不是本论文新提出的理论；有限汇总观测下不声称最优性或收敛保证。\n\n'
    elif method == 'legacy':
        algorithm = '## 所选8步分位数学习\n\n原生奖励r_t在合同到期结算，gamma=1，因此完成回合Σr_t=ΣY_i。QR-DQN估计32个回报分位数，以其均值在合法动作中选择。8步目标为Σ_(j=0)^(n−1)r_(t+j)+1[未终止]·Z_target(o_(t+n),argmax_(a∈A)Q_online(o_(t+n),a))，n≤8；终止窗口正确冲刷。使用分位数Huber损失、Double-Q选择/评估分离。没有势函数奖励或CVaR。\n\n本轮没有把较长回报窗口或新增进度观测强行保留；它们的实测比较见完整报告。回报分位数不是资源需求预测。\n\n'
    else:
        algorithm = '## 所选分位数学习\n\n'+formulation.split('## 目标保持的整形与分位数学习')[1].split('## 三研究点的接口')[0]
        if method == 'no_shaping':
            algorithm = algorithm.replace('κ=0.5', 'κ=0（本方法不整形）')
    formulation = prefix+algorithm+'## 三研究点的接口'+tail
    (out/'formulation.md').write_text(formulation)
    # Keep the prior root manuscript available as historical evidence.
    archive = out/'previous_root_documents'
    archive.mkdir(exist_ok=True)
    for name in ('README.md', 'slice_alloc/design.md',
            '面向低空智联的多模态网络切片技术.md', '面向低空智联的多模态网络切片技术.docx'):
        source = root/name
        destination = archive/name
        if source.exists() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    contents = ['# 面向低空智联的多模态网络切片技术', '',
        '研究项点：（1）低空智联下的多模态网络切片智能分配；（2）低空智联下的多模态网络切片资源预测；（3）低空智联下的多模态网络切片QoS保障。', '',
        '本版完成研究点1的RL设计、原生实现、训练与新留出验证。研究点2、3只保留接口边界，不报告未实施算法的结果。', '',
        design.replace(design.splitlines()[0], '## 研究点1系统设计', 1), '',
        formulation.replace('# 研究点1的数学表述', '## 联合决策模型', 1), '',
        report.replace('# 研究点1：RL修订版最终结果', '## 新留出实验结果', 1), '',
        '## 复现材料', '',
        '[设计](slice_alloc/results/modal_binding_rl/design.md) · [正式报告](slice_alloc/results/modal_binding_rl/report.md) · [统计](slice_alloc/results/modal_binding_rl/analysis.json) · [独立验真](slice_alloc/results/modal_binding_rl/verification.json) · [复现步骤](slice_alloc/results/modal_binding_rl/REPRODUCE.md)', '',
        '旧版实验和开发试错保留为历史证据，不混入本轮新测试收益。QR-DQN、Double-Q与势函数整形是已有方法；本版不声称全局最优、理论收敛或任意网络泛化。']
    text = '\n'.join(contents)+'\n'
    text = text.replace('(figures/', '(slice_alloc/results/modal_binding_rl/figures/')
    for name in ('design.md', 'analysis.json', 'verification.json'):
        text = text.replace(f']({name})', f'](slice_alloc/results/modal_binding_rl/{name})')
    md = root/'面向低空智联的多模态网络切片技术.md'
    docx = root/'面向低空智联的多模态网络切片技术.docx'
    md.write_text(text)
    to_docx(text, destination=docx)
    to_docx(report.replace('(figures/', '(slice_alloc/results/modal_binding_rl/figures/'), destination=out/'report.docx')
    (root/'slice_alloc/design.md').write_text(design)
    q = study['test'][choice['method']]
    rule = study['best_validation_rule']
    delta = study['paired'][rule]
    verdict = ('支持本轮RL总体优于验证集选定规则' if delta['ci95'][0] > 0 else
        '本轮RL总体低于验证集选定规则' if delta['ci95'][1] < 0 else
        '本轮不能确认RL显著优于验证集选定规则')
    (root/'README.md').write_text('\n'.join([
        '# 面向低空智联的多模态网络切片技术', '',
        '当前版本只完成研究点1：以单模型RL联合决定业务接纳、IP/受限EID-RLOC/NDN绑定及切片功能部署。研究点2的资源预测、研究点3的运行期主动QoS保障尚未实现。', '',
        f"新留出测试：验证集选定RL版本为{choice['method']}，{len(q['individual_rates'])}次独立训练的单模型平均SLA成功率 **{q['rate']*100:.2f}%**；验证集选定规则为 **{study['test'][rule]['rate']*100:.2f}%**；差值 **{delta['difference']*100:+.2f}个百分点**，95%区间 **[{delta['ci95'][0]*100:+.2f}, {delta['ci95'][1]*100:+.2f}]**。{verdict}。", '',
        'RL为预先指定的研究主方法，具体版本按测试前补充协议在验证集选择；部署固定初始化11，不集成、不回退专家规则。研究方法选择与性能优越性结论分开说明。比较实际进度观测与势函数整形，物理合同和强基线保持一致；整回合训练奖励严格等于原生SLA成功数。', '',
        f"验真：{verified['independently_reaudited_episodes']}个正式回合全部重新逐包审计，24个模型各代表场景精确重放；新训练19个模型共{verified['trained_steps']}决策，每模型65536，另复用5个既有联合模态模型。", '',
        '[最终报告](slice_alloc/results/modal_binding_rl/report.md) · [设计](slice_alloc/design.md) · [独立验真](slice_alloc/results/modal_binding_rl/verification.json) · [复现](slice_alloc/results/modal_binding_rl/REPRODUCE.md)', '',
        '[论文正文](面向低空智联的多模态网络切片技术.md) · [Word正文](面向低空智联的多模态网络切片技术.docx)', '',
        '```bash',
        '/home/allen/miniconda3/envs/sdn/bin/python -m slice_alloc.tools.modal_binding_rl_allocate --seed 29000 --load 48 --output /tmp/binding-rl-demo',
        '```', '',
        '原始比较与事后诊断分别保留在slice_alloc/results/modal_binding与modal_binding_diagnosis；历史结果不作为本轮新留出结果。', '']))
    files = [md, docx, root/'README.md', root/'slice_alloc/design.md', out/'report.md', out/'report.docx',
        out/'analysis.json', out/'verification.json', out/'design.md', out/'formulation.md',
        root/'slice_alloc/tools/modal_binding_rl_report.py', root/'slice_alloc/tools/modal_binding_rl_verify.py',
        root/'slice_alloc/tools/modal_binding_rl_allocate.py', root/'slice_alloc/tools/modal_binding_rl_publish.py',
        root/'slice_alloc/tools/modal_binding_rl_select.py', root/'slice_alloc/tools/modal_binding_rl_ppo.py',
        root/'slice_alloc/tools/modal_binding_rl_registry.py', root/'slice_alloc/tools/modal_binding_rl_legacy.py',
        root/'slice_alloc/tests/test_modal_binding_rl_statistics.py', root/'slice_alloc/tests/test_modal_binding_rl_ppo.py',
        out/'rl_selection.json', out/'rl_selection_addendum.json', out/'ppo_protocol.json', out/'legacy_protocol.json']
    for source in files:
        if source.suffix == '.py':
            target = out/'frozen'/source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert run.sha(source) == run.sha(target)
            else:
                shutil.copy2(source, target)
    run.write(out/'delivery.json', dict(completed_research_points=[1], unimplemented_research_points=[2, 3],
        artifacts={str(p.relative_to(root)): run.sha(p) for p in files},
        training_steps=verified['trained_steps'], audited_episodes=verified['independently_reaudited_episodes'],
        main_method='single-model RL, initialization11; research choice does not assert superiority'))
    print('Published', md, docx)


if __name__ == '__main__':
    main()
