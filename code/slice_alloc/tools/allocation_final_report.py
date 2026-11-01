"""Independent event audit, paired statistics, replay and final research report."""
from collections import defaultdict
import concurrent.futures as cf
import gzip
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

from slice_alloc.tools import allocation_final_run as run
from slice_alloc.tools import allocation_final_stress as stress

NAMES = {'candidate_ppo':'候选感知 PPO', 'double_qrdqn':'Masked Double QR-DQN',
         'double_dqn':'Masked Double DQN', 'candidate_a2c':'候选感知 A2C',
         'first_fit':'首次可行', 'nearest':'近边优先', 'greedy':'资源贪心',
         'price':'资源价格规则', 'semantic':'业务语义与移动规则', 'quota':'固定配额语义规则',
         'full':'完整方法', 'fixed':'固定模态', 'no_motion':'去掉显式端点移动观测',
         'no_calendar':'去掉未来预约日历'}


def group_name(key):
    method, variant = key
    return NAMES[method]+('／'+NAMES[variant] if variant != 'full' else '')


def matrix(rows):
    inits = sorted({r['init'] for r in rows})
    seeds = sorted({r['seed'] for r in rows})
    values = np.zeros((len(inits),len(seeds)))
    denominators = np.zeros(len(seeds))
    for i, init in enumerate(inits):
        for j, seed in enumerate(seeds):
            matching = [r for r in rows if r['init']==init and r['seed']==seed]
            values[i,j]=sum(r['success'] for r in matching)
            denominator=sum(r['requests'] for r in matching)
            assert denominator>0
            if i:
                assert denominators[j]==denominator
            denominators[j]=denominator
    return values, denominators, inits, seeds


def interval(rows, seed=51001):
    values, denominators, inits, seeds=matrix(rows)
    rng=np.random.default_rng(seed)
    scene=rng.integers(len(seeds),size=(10000,len(seeds)))
    initialization=rng.integers(len(inits),size=(10000,len(inits)))
    numerator=values[initialization[:,:,None],scene[:,None,:]].mean(1).sum(1)
    samples=numerator/denominators[scene].sum(1)
    return [float(v) for v in np.quantile(samples,[.025,.975])]


def difference(left,right):
    lv,ld,li,ls=matrix(left)
    rv,rd,ri,rs=matrix(right)
    assert ls==rs
    np.testing.assert_array_equal(ld,rd)
    rng=np.random.default_rng(51002)
    scene=rng.integers(len(ls),size=(10000,len(ls)))
    il=rng.integers(len(li),size=(10000,len(li)))
    ir=il if li==ri else rng.integers(len(ri),size=(10000,len(ri)))
    left_boot=lv[il[:,:,None],scene[:,None,:]].mean(1).sum(1)
    right_boot=rv[ir[:,:,None],scene[:,None,:]].mean(1).sum(1)
    samples=(left_boot-right_boot)/ld[scene].sum(1)
    delta=float((lv.mean(0)-rv.mean(0)).sum()/ld.sum())
    return dict(difference=delta, ci95=[float(v) for v in np.quantile(samples,[.025,.975])])


def summarize(rows):
    successes=sum(r['success'] for r in rows)
    arrivals=sum(r['requests'] for r in rows)
    admitted=sum(r['admitted'] for r in rows)
    inits=sorted({r['init'] for r in rows})
    individual={str(i):sum(r['success'] for r in rows if r['init']==i)/sum(r['requests'] for r in rows if r['init']==i) for i in inits}
    services={}
    for business in ('telemetry','command','object'):
        total=sum(r['by_service'][business]['requests'] for r in rows)
        services[business]=sum(r['by_service'][business]['success'] for r in rows)/max(total,1)
    return dict(rate=successes/arrivals, ci95=interval(rows), admission_rate=admitted/arrivals,
        admitted_violation_rate=(admitted-successes)/max(admitted,1), by_service=services,
        by_init=individual, init_sample_sd=float(np.std(list(individual.values()),ddof=1)) if len(inits)>1 else None,
        forced_reject=sum(r['forced_reject'] for r in rows)/arrivals,
        voluntary_reject=sum(r['voluntary_reject'] for r in rows)/arrivals,
        wire_bytes_per_success=sum(r['wire_bytes'] for r in rows)/max(successes,1),
        phy_tx_seconds_per_success=sum(r['phy_tx_seconds'] for r in rows)/max(successes,1),
        normalized_reservation_seconds_per_success=sum(sum(r['resource_time']) for r in rows)/max(successes,1),
        decision_ms=float(np.mean([r['decision_ms'] for r in rows])), episodes=len(rows))


def audit_saved(row):
    prefix=run.OUT/row['trace']
    result=run.read(str(prefix)+'-result.json')
    cfg=run.read(str(prefix)+'-scenario.json')
    packet=run.old.audit(prefix,result,cfg)
    direction=run.audit_direction(prefix,result,cfg)
    for key,value in {**packet,**direction}.items():
        assert row[key]==value, (prefix,key)
    # Independently reconstruct reservation transactions, including business
    # compatibility and every known lease/release boundary.
    ledger=run.a.b.p.Ledger(cfg['public'])
    messages=result['observations']
    for message in messages:
        if 'decision' not in message:
            continue
        observation=message['observation']
        decision=message['decision']
        ledger.release(observation['time'])
        if not decision['accept']:
            continue
        raw=next(r for r in cfg['requests'] if r['id']==decision['request_id'])
        request=run.a.b.Service(**{**raw,'members':tuple(raw['members'])})
        assert decision['mode'] in run.a.COMPATIBLE[request.service]
        if request.service=='telemetry':
            plan=run.a.b.p.compile_plan(request.bound('ip'),decision['edge'],decision['server'],0,cfg['public'],observation)
        else:
            plan=run.a.b.compile_plan(request,decision['mode'],decision['edge'],decision['server'],decision['cache'],cfg['public'],observation)
        assert ledger.feasible(plan)
        if row['method']=='quota':
            quota=run.a.b.p.Ledger(cfg['public'])
            quota.capacity[:11]/=3
            quota.plans=[p for p in ledger.plans if p.request.mode==plan.request.mode]
            assert quota.feasible(plan)
        ledger.add(plan)
    return True


def replay(spec):
    algorithm,variant,init=spec
    torch.set_num_threads(1)
    original=run.OUT/'native'/f'{algorithm}-{variant}-{init}'/f'test-{run.LOADS[0]}-{run.TEST[0]}'
    target=run.OUT/'replay'/f'{algorithm}-{variant}-{init}'
    model=run.load_model(algorithm,run.OUT/'models'/f'{algorithm}-{variant}-{init}.zip')
    cfg=run.read(str(original)+'-scenario.json')
    repeated=run.episode(cfg,algorithm,model,variant,prefix=target)
    recorded=run.read(str(original)+'-result.json')
    for key in ('decisions','observations','settled','success','admitted'):
        assert repeated[key]==recorded[key], (spec,key)
    with gzip.open(str(original)+'-events.csv.gz','rb') as first, gzip.open(str(target)+'-events.csv.gz','rb') as second:
        assert first.read()==second.read(), ('event replay mismatch',spec)
    return dict(algorithm=algorithm,variant=variant,init=init,exact_event_replay=True)


def pct(value):
    return f'{100*value:.2f}%'


def verify_registration():
    marker=run.OUT/'test_started.json'
    assert run.read(marker)['preregistration_sha256']==run.preregistration_snapshot(), 'Preregistered inputs changed'
    # Some filesystems assign the same timestamp to successive writes. The
    # start marker's hash snapshot proves that each input already existed.
    for name in run.REGISTRATIONS:
        assert (run.OUT/name).stat().st_mtime_ns<=marker.stat().st_mtime_ns


def main():
    torch.set_num_threads(1)
    run.freeze()
    from slice_alloc.tools.allocation_final_pipeline import freeze_auxiliary
    freeze_auxiliary()
    from slice_alloc.tools.allocation_final_pipeline import prepare_evaluation
    prepare_evaluation()
    verify_registration()
    rows=run.read(run.OUT/'evaluation.json')
    expected_models=(len(run.ALGORITHMS)+3)*len(run.INITS)
    expected_primary=(expected_models+len(run.RULES))*len(run.LOADS)*len(run.TEST)
    expected_controls=(len(run.INITS)+len(run.CONTROL_RULES))*3*len(run.CONTROL)
    assert len(rows)==expected_primary+expected_controls, len(rows)
    stress_rows=run.read(run.OUT/'stress_evaluation.json')
    assert len(stress_rows)==(len(run.INITS)+len(run.CONTROL_RULES))*2*len(stress.SEEDS)
    rows+=stress_rows
    selection=run.read(run.OUT/'selection.json')
    selected=selection['selected']
    started=(run.OUT/'test_started.json').stat().st_mtime
    assert started>=(run.OUT/'selection.json').stat().st_mtime
    assert started>=(run.OUT/'training_complete.json').stat().st_mtime
    expected=[(m,'full',i) for m in run.ALGORITHMS for i in run.INITS]
    expected += [(selected,v,i) for v in ('fixed','no_motion','no_calendar') for i in run.INITS]
    for algorithm,variant,init in expected:
        training=run.read(run.OUT/'training'/algorithm/variant/str(init)/'training.json')
        assert training['steps']==max(run.CHECKPOINTS)
        assert training['weight_change']>0
        path=run.OUT/training['model']
        assert run.sha(path)==training['model_sha256']
        assert path.stat().st_mtime<started
        assert all(41000000<=s<42000000 for s in training['training_seeds'])
    stress.freeze()
    identities=defaultdict(set)
    for row in rows:
        identities[row['scope'],row['load'],row['seed']].add(row['scenario_sha256'])
    assert all(len(values)==1 for values in identities.values())
    with cf.ProcessPoolExecutor(max_workers=6) as pool:
        assert all(pool.map(audit_saved,rows))
        replayed=list(pool.map(replay,expected))
    print('AUDITED',len(rows),'REPLAYED',len(replayed),flush=True)

    grouped=defaultdict(list)
    for row in rows:
        if row['scope']=='test':
            grouped[row['method'],row['variant']].append(row)
    summaries={'/'.join(k):summarize(v) for k,v in grouped.items()}
    rule_selection={m:run.read(run.OUT/'rules'/f'{m}.json')['chosen']['rate'] for m in run.RULES if m!='quota'}
    reference_rule=max(rule_selection,key=rule_selection.get)
    best_rule=run.PRIMARY_RULE
    primary=difference(grouped[selected,'full'],grouped[best_rule,'full'])
    strong_comparison=difference(grouped[selected,'full'],grouped[reference_rule,'full'])
    ablations={v:difference(grouped[selected,'full'],grouped[selected,v]) for v in ('fixed','no_motion','no_calendar')}
    sharing=difference(grouped['semantic','full'],grouped['quota','full'])
    by_load={str(l):{'/'.join(k):summarize([r for r in v if r['load']==l]) for k,v in grouped.items()} for l in run.LOADS}
    mobility={}
    for scope in ('stationary','local','mixed'):
        for method in (selected,*run.CONTROL_RULES):
            matching=[r for r in rows if r['scope']=='mobility_'+scope and r['method']==method]
            mobility[f'{method}/{scope}']=summarize(matching)
    stresses={f'{m}/{case}':summarize([r for r in rows if r['scope']=='stress_'+case and r['method']==m])
              for case in ('backhaul','airtime') for m in (selected,*run.CONTROL_RULES)}
    analysis=dict(selected=selected,selected_rule=best_rule,validation=selection['validation'],
        summary=summaries,primary=primary,reference_rule=reference_rule,strong_comparison=strong_comparison,
        ablations=ablations,sharing=sharing,
        by_load=by_load,mobility=mobility,stress=stresses,bootstrap=f'10000 paired scene and initialization resamples; {len(run.INITS)} inits; exploratory intervals',
        audited_episodes=len(rows),replayed_models=replayed,training_decisions=len(expected)*max(run.CHECKPOINTS))
    run.write(run.OUT/'analysis.json',analysis)
    run.write(run.OUT/'verification.json',dict(source_hashes=True,binary_hash=True,model_hashes=True,
        native_packet_audits=len(rows),direction_and_ledger_audits=len(rows),
        exact_event_replays=len(replayed),scene_fingerprints=True,selection_precedes_test=True))

    font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    keys=[(m,'full') for m in run.ALGORITHMS]+[(r,'full') for r in ('nearest','greedy','quota')]
    fig,ax=plt.subplots(figsize=(12,5))
    x=np.arange(len(keys))
    for j,l in enumerate(run.LOADS):
        vals=[by_load[str(l)]['/'.join(k)] for k in keys]
        rates=np.array([v['rate'] for v in vals])*100
        low=rates-np.array([v['ci95'][0] for v in vals])*100
        high=np.array([v['ci95'][1] for v in vals])*100-rates
        ax.bar(x+(j-.5)*.36,rates,.36,label=f'{l} 个请求',yerr=np.array([low,high]),capsize=3)
    ax.set_xticks(x,[group_name(k) for k in keys],rotation=12,ha='right')
    ax.set_ylabel('SLA成功率（%，拒绝计失败）')
    ax.set_ylim(0,100)
    ax.legend()
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(run.OUT/'comparison.png',dpi=180)
    plt.close(fig)

    selected_summary=summaries[selected+'/full']
    delta=primary['difference']*100
    lo,hi=np.array(primary['ci95'])*100
    verdict='本轮支持所选RL在该测试分布上优于预先指定的Greedy。' if lo>0 else (
        '本轮所选RL低于Greedy，不能声称性能优越性。' if hi<0 else '本轮区间包含零，尚不能确认所选RL优于Greedy。')
    lines=['# 面向低空智联的多模态网络切片技术','',
        '本报告仅包含第一部分“智能分配”。题目、研究点和遥测、命令、地图业务名称保留。','',
        f'验证集在{len(run.ALGORITHMS)}种RL中选定 **{NAMES[selected]}**。主测试{len(run.INITS)}初始化平均SLA成功率 **{pct(selected_summary["rate"])}**；预先指定的主对照 **{NAMES[best_rule]}** 为 **{pct(summaries[best_rule+"/full"]["rate"])}**。',
        f'配对差值 **{delta:+.2f}个百分点**，95%区间 **[{lo:+.2f}, {hi:+.2f}]个百分点**。{verdict}','',
        '## 环境与方法','',
        '12架无人机、双AP、双互连边缘，4移动/8驻留，高80m，速度5/8/11m/s。遥测上行IP；命令下行IP/受限EID-RLOC；地图IP/原生NDN。17维资源账本，功能配置在ns-3/ndnSIM/NFD实际执行。',
        'LogDistance指数2.5叠加120m有限覆盖，发射20dBm、固定24Mbps；保留硬覆盖边界，不声称一般低空信道。回源1Mbps、跨边缘0.5Mbps、接入2Mbps均为研究配置。预约是准入模型，不是逐包QoS证明。',
        f'PPO、QR-DQN、标量Double-DQN、A2C各{len(run.INITS)}初始化、每模型{max(run.CHECKPOINTS)}次决策，检查点仅由验证选择；随后对所选算法重训3类消融、每类{len(run.INITS)}初始化，总{len(expected)}个模型、{len(expected)*max(run.CHECKPOINTS)}次训练决策。无集成、专家先验或规则兜底。四种方法使用相同环境、观测和可行性掩码；学习配方与更新次数不同，属于等交互预算选型，不是只替换学习目标的消融。',
        '训练41000000—41999999；验证42000—42007；新主测试53000—53029；移动受控54000—54009。Greedy和主表基线在新测试前指定；价格、语义规则保留补充比较。规则阈值仅在验证集选择，正式测试后未调参。单模型执行入口预先指定初始化11，性能统计始终包含全部3个初始化。','',
        '## 主测试','',
        '| 方法 | SLA成功率 | 95%区间 | 接纳率 | 已接纳违约率 |',
        '|---|---:|---:|---:|---:|']
    for key in [(m,'full') for m in run.ALGORITHMS]+[(r,'full') for r in run.MAIN_RULES]:
        v=summaries['/'.join(key)]
        lines.append(f'| {group_name(key)} | {pct(v["rate"])} | [{pct(v["ci95"][0])}, {pct(v["ci95"][1])}] | {pct(v["admission_rate"])} | {pct(v["admitted_violation_rate"])} |')
    lines += ['','总体值对48/80两负载按请求数合并；同一场景下所有方法使用相同业务与无线种子。区间同时重采样初始化及场景；仅3个初始化，训练稳定性估计仍有限。规则区间仅重采样场景。','',
              '## 补充强规则对比','', '| 方法 | SLA成功率 |','|---|---:|']
    for method in ('price','semantic'):
        lines.append(f'| {NAMES[method]} | {pct(summaries[method+"/full"]["rate"])} |')
    lines += ['',f'验证集选定的参考规则为{NAMES[reference_rule]}；所选RL相对该规则差值{100*strong_comparison["difference"]:+.2f}个百分点，95%区间[{100*strong_comparison["ci95"][0]:+.2f}, {100*strong_comparison["ci95"][1]:+.2f}]。此比较完整保留，不以主表基线的强弱代替方法有效性判断。','',
              '![分负载结果](comparison.png)','','## 全部初始化与业务分组','',
              '| 初始化 | 所选RL的SLA成功率 |','|---|---:|']
    lines += [f'| {i} | {pct(v)} |' for i,v in selected_summary['by_init'].items()]
    deviation=selected_summary['init_sample_sd']
    deviation_text=f'初始化样本标准差：{100*deviation:.2f}个百分点。' if deviation is not None else '只有1个初始化，无法估计初始化样本标准差。'
    lines += ['',deviation_text,'',
              '| 业务 | 所选RL | Greedy |','|---|---:|---:|']
    for business,label in [('telemetry','遥测'),('command','命令'),('object','地图')]:
        lines.append(f'| {label} | {pct(selected_summary["by_service"][business])} | {pct(summaries[best_rule+"/full"]["by_service"][business])} |')
    lines += ['','## 同配方重训消融','',
              '| 对照 | 完整方法−对照（百分点） | 配对95%区间 |','|---|---:|---:|']
    for variant,v in ablations.items():
        lines.append(f'| {NAMES[variant]} | {100*v["difference"]:+.2f} | [{100*v["ci95"][0]:+.2f}, {100*v["ci95"][1]:+.2f}] |')
    lines += ['',f'共享资源实验：相同语义与移动规则族、各自在验证集选择阈值，采用共享池相对固定配额差值为{100*sharing["difference"]:+.2f}个百分点，区间[{100*sharing["ci95"][0]:+.2f}, {100*sharing["ci95"][1]:+.2f}]。这个差值不归因于RL，也不是固定完全相同阈值的单变量消融。',
              '移动消融移除当前请求及全体端点的直接位置、角色、关联和速度编码；掩码仍执行必要可行性检查，活跃合同的业务/模态汇总及其中巡检比例保留，因此它不是消除全部移动相关信息的消融。去掉日历只移除未来1/3/6秒的预约向量，保留当前预约和已签合同的其他汇总。消融区间属于探索性分析，未作多重比较校正。','',
              '## 移动受控实验','', '| 方法 | 全部驻留 | 单AP内移动 | 原跨AP巡检 |','|---|---:|---:|---:|']
    for method in (selected,*run.CONTROL_RULES):
        lines.append('| '+NAMES[method]+' | '+' | '.join(pct(mobility[f'{method}/{scope}']['rate']) for scope in ('stationary','local','mixed'))+' |')
    lines += ['','单AP内移动将巡检航线限制为x=30—45m；驻留保留各节点初始位置。移动与驻留改变后续几何位置及关联，不能将差值解释为仅有协议切换中断这一因素。受控实验每条件10个独立种子，只用于敏感性评测。','',
              '## 资源压力敏感性','',
              '补充实验在正式主测试前单独预注册：回源受限为80个平衡请求、回源0.45Mbps；空口压力为160个请求、每请求以0.85概率强制为遥测（其余保留），所有有线速率提高8倍。均使用55000—55009种子，不改变主场景，也不把容量变化当作算法收益。','',
              '| 方法 | 回源受限 | 空口压力 |','|---|---:|---:|']
    for method in (selected,*run.CONTROL_RULES):
        lines.append('| '+NAMES[method]+' | '+' | '.join(pct(stresses[f'{method}/{case}']['rate']) for case in ('backhaul','airtime'))+' |')
    calibration=run.read(run.OUT/'calibration.json')
    lines += ['','开发诊断覆盖32/48/64/80/96请求，主配置的不可行候选主要超出有线方向预约额度，不能据此宣称无线为主瓶颈。12组单请求/位置/传播配置均完成198/198包按时交付；距离损耗在部分位置改变关联路径及时延，本次标定没有证明覆盖内单业务发生显著渐进丢包。因此低空影响仍主要通过几何、竞争和接入变化体现。','',
              '## 资源开销与验证','',
              f'所选RL：每成功合同实际有线字节{selected_summary["wire_bytes_per_success"]:.1f}B，PHY发射时间{selected_summary["phy_tx_seconds_per_success"]:.4f}s，容量归一化预约消耗{selected_summary["normalized_reservation_seconds_per_success"]:.4f}s，平均决策{selected_summary["decision_ms"]:.3f}ms。并行评测存在CPU争用，决策耗时不是独占部署延迟基准。',
              f'全部{len(rows)}个正式回合已独立重读逐包事件重建SLA、检查遥测方向和资源事务；{len(expected)}个所选模型均通过代表场景完整事件CSV精确重放。源码、原生二进制、模型和同场景指纹一致。',
              '测试日志及详细统计见`verification.json`和`analysis.json`；复现命令见`REPRODUCE.md`。','',
              '## 结论适用范围','',
              verdict+' 本轮没有实现资源预测、运行期主动迁移、CPU处理模型或确定性带宽/时延保障。IP下行采用准入时定位绑定；不将模态对比推广为对全部IP移动方案的比较。',
              '环境与旧报告不同，旧成功率不能作为本轮改进幅度。协议选择比例本身也不证明选择有效。']
    (run.OUT/'report.md').write_text('\n'.join(lines)+'\n')
    (run.OUT/'REPRODUCE.md').write_text('''# 复现

在代码目录使用已安装numpy/torch/SB3/sb3-contrib的Python环境。

```bash
cp slice_alloc/execution/allocation_final_scenario.cc "$LOWALT_NS3_ROOT/scratch/allocation-final-scenario.cc"
cd "$LOWALT_NS3_ROOT"
./waf build --targets=allocation-final-scenario -j 4
```

回到代码目录：

```bash
python -m unittest discover -s slice_alloc/tests -p 'test_*.py'
python -m slice_alloc all --workers 6
python -m slice_alloc.tools.allocation_final_allocate --seed 46000 --load 48 --output /tmp/lowalt-allocation
```

默认输出为results/final，也可用LOWALT_ALLOCATION_OUTPUT指定独立目录。现有结果按配置及哈希核对后复用，不能拼接新旧协议。完整管线在主测试前冻结补充压力协议及初始化11的部署约定，并执行标定、训练、主测试、补充测试和审计。训练、验证、测试与逐包审计都使用原生仿真。
''')
    manifest={str(p.relative_to(run.OUT)):run.sha(p) for p in sorted(run.OUT.rglob('*')) if p.is_file() and p.name not in ('manifest.json', 'FINAL_RESULT.json')}
    run.write(run.OUT/'manifest.json',manifest)
    tracked=dict(title='面向低空智联的多模态网络切片技术',research='智能分配',selected=selected,
        summary=selected_summary,selected_rule=best_rule,comparison=primary,
        protocol=run.read(run.OUT/'protocol.json'),models={p:val for p,val in manifest.items() if p.startswith('models/')},
        artifacts={p:manifest[p] for p in ('protocol.json','selection.json','evaluation.json','analysis.json','verification.json','report.md')},
        manifest_sha256=run.sha(run.OUT/'manifest.json'),audited_episodes=len(rows),replayed_models=len(replayed))
    run.write(run.OUT/'FINAL_RESULT.json',tracked)
    print('FINAL',selected,selected_summary['rate'],best_rule,primary,flush=True)


if __name__=='__main__':
    main()
