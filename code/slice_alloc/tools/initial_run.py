"""First research part: one PPO for native low-altitude multimodal slices.

Run tests, smoke, then all. Scientific code, plans, model selection and reports
share one immutable source snapshot. Historical studies are never inputs.
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from slice_alloc import patrol
from slice_alloc.initial_allocation import (AllocationPolicy, InitialAllocationEnv,
    RewardUnit, MODES, QuotaLedger, make_scenario)

ROOT = patrol.ROOT
OUT = ROOT / 'slice_alloc/results/lifecycle_ppo'
INITS = (11, 23, 37, 53, 71)
LOADS = (64, 80)
BUDGETS = (16384, 32768, 65536)
VALIDATION = tuple(range(10100, 10116))
TEST = tuple(range(10200, 10230))
MIX_SEEDS = tuple(range(10300, 10306))
MOBILITY_SEEDS = (10400, 10401, 10402)
RULES = ('near_edge', 'greedy', 'marginal_price', 'static_quota')
REWARD_UNIT = 80.
CONFIGS = dict(ppo=dict(width=64, hierarchical=True, lifecycle=True),
    no_hierarchy=dict(width=64, hierarchical=False, lifecycle=True),
    no_calendar=dict(width=64, hierarchical=True, lifecycle=False))
HYPER = dict(learning_rate=3e-4, n_steps=1024, batch_size=128, n_epochs=10,
    gamma=1., gae_lambda=1., ent_coef=.01, clip_range=.2, target_kl=.03)
SOURCES = ('slice_alloc/patrol.py', 'slice_alloc/patrol_contract.schema.json',
    'slice_alloc/execution/patrol_contract.hpp', 'slice_alloc/execution/patrol_scenario.cc',
    'slice_alloc/initial_allocation.py', 'slice_alloc/tools/initial_run.py',
    'slice_alloc/tools/patrol_verify.py', 'slice_alloc/tools/patrol_run.py',
    'slice_alloc/tools/export_document.py', 'slice_alloc/tests/test_initial_allocation.py',
    'slice_alloc/initial_design.md', 'slice_alloc/method_proposal.md')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def hashes():
    return {name: sha(ROOT / name) for name in SOURCES}


def choose(method, env, model, obs):
    if method in ('marginal_price', 'static_quota'):
        return min(range(1, len(env.options) + 1),
                   key=lambda i: float(obs['candidates'][i, -3])) if env.options else 0
    if method in ('near_edge', 'greedy'):
        return patrol.baseline(method, env)
    if method.startswith('fixed_'):
        edge = int(method[-1])
        return next((i for i, p in enumerate(env.options, 1)
                     if p.edge == edge and p.server == edge
                     and p.cache == (edge if p.request.mode == 'ndn' else 0)), 0)
    return int(model.predict(obs, deterministic=True)[0])


def episode(cfg, method, model=None, prefix=None):
    env = InitialAllocationEnv(partitioned=method == 'static_quota')
    timings = []
    try:
        obs, _ = env.reset(options=dict(scenario=cfg, prefix=prefix, trace=prefix is not None))
        done = False
        while not done:
            start = time.perf_counter()
            action = choose(method, env, model, obs)
            timings.append(time.perf_counter() - start)
            obs, _, done, _, _ = env.step(action)
        result = env.result()
        result['decision_ms'] = 1000 * float(np.mean(timings))
        if prefix:
            write(str(prefix) + '-result.json', result)
            raw = Path(str(prefix) + '-events.csv')
            with raw.open('rb') as src, gzip.open(str(raw) + '.gz', 'wb', compresslevel=3) as dst:
                shutil.copyfileobj(src, dst)
            raw.unlink()
        return result
    finally:
        env.close()


def row_for(cfg, method, result, tag, **metadata):
    prefix = OUT / 'native' / tag
    row = dict(method=method, requests=result['requests'], success=result['success'],
        admitted=result['admitted'], rejections=result['rejections'], by_mode=result['by_mode'],
        peak_usage=result['peak_usage'], pressure=result['pressure'], decision_ms=result['decision_ms'],
        path=tag, summary=json.loads(Path(str(prefix) + '-summary.json').read_text()), **metadata)
    for suffix, key in (('-scenario.json', 'scenario_sha256'), ('-result.json', 'result_sha256'),
                        ('-events.csv.gz', 'events_sha256'), ('-summary.json', 'summary_sha256')):
        row[key] = sha(str(prefix) + suffix)
    return row


def validate_model(model, method):
    rows = []
    for seed in VALIDATION:
        for load in LOADS:
            r = episode(make_scenario(seed, load), method, model)
            rows.append(dict(seed=seed, load=load, requests=r['requests'],
                             success=r['success'], admitted=r['admitted']))
    return dict(success_rate=sum(r['success'] for r in rows) /
                sum(r['requests'] for r in rows), rows=rows)


def train_one(spec):
    method, init = spec
    torch.set_num_threads(1)
    final_path = OUT / 'models' / f'{method}-{init}.zip'
    manifest = final_path.with_suffix('.json')
    if manifest.exists():
        info = json.loads(manifest.read_text())
        assert info['sources'] == hashes() and info['model_sha256'] == sha(final_path)
        return info
    env = RewardUnit(InitialAllocationEnv(seed=init + 100), unit=REWARD_UNIT)
    model = PPO(AllocationPolicy, env, **HYPER, policy_kwargs=CONFIGS[method],
                seed=init, device='cpu', verbose=0)
    initial_actor = model.policy.actor[-1].weight.detach().clone()
    stages = []
    start = time.perf_counter()
    try:
        for target in BUDGETS:
            model.learn(total_timesteps=target - model.num_timesteps,
                        reset_num_timesteps=model.num_timesteps == 0)
            assert model.num_timesteps == target
            checkpoint = OUT / 'checkpoints' / f'{method}-{init}-{target}.zip'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            model.save(checkpoint)
            validation = validate_model(model, method)
            stages.append(dict(step=target, checkpoint=str(checkpoint.relative_to(ROOT)),
                model_sha256=sha(checkpoint), validation=validation))
            print('VALIDATION', method, init, target, round(validation['success_rate'], 6), flush=True)
        # Pick by validation only. Restore the complete saved algorithm;
        # never modify a runtime knob or attach new weights to an old score.
        best = max(stages, key=lambda stage: (stage['validation']['success_rate'], -stage['step']))
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / best['checkpoint'], final_path)
        selected = PPO.load(final_path, device='cpu')
        assert selected.policy_kwargs == CONFIGS[method] and selected.num_timesteps == best['step']
        movement = float(torch.linalg.vector_norm(selected.policy.actor[-1].weight.detach() - initial_actor))
        assert movement > 0, 'The actor never learned'
        assert validate_model(selected, method) == best['validation'], 'Selected model is not the validated checkpoint'
        info = dict(method=method, initialization=init, steps=best['step'],
            trained_steps=BUDGETS[-1], policy_settings=CONFIGS[method], reward_unit=REWARD_UNIT,
            sources=hashes(), model_sha256=sha(final_path),
            checkpoint=str(final_path.relative_to(ROOT)), seconds=time.perf_counter() - start,
            actor_weight_change_norm=movement, training_scenarios=env.unwrapped.episode_specs,
            validation_stages=stages, selection='maximum validation SLA; earliest checkpoint on ties',
            demonstrations=0, shaping=False, rule_prior=False,
            reward='native settled SLA count divided by constant 80 during training only')
        write(manifest, info)
        print('TRAINED', method, init, 'selected', best['step'], 'actor_change', round(movement, 5), flush=True)
        return info
    finally:
        env.close()


def eval_group(spec):
    method, init, seeds, mix, scope = spec
    torch.set_num_threads(1)
    model = PPO.load(OUT / 'models' / f'{method}-{init}.zip', device='cpu') if method in CONFIGS else None
    rows = []
    for seed in seeds:
        for load in LOADS:
            cfg = make_scenario(seed, load, mix)
            tag = f'{scope}/{method}-{init or 0}/{mix}-{load}-{seed}'
            result = episode(cfg, method, model, OUT / 'native' / tag)
            rows.append(row_for(cfg, method, result, tag, initialization=init,
                                seed=seed, load=load, mix=mix, scope=scope))
    return rows


def mobility_case(spec):
    mode, condition, edge, seed = spec
    cfg = make_scenario(seed, 1)
    cfg['private']['initial_positions']['5'] = [30., 0., 80.]
    cfg['private']['motions'] = ([] if condition == 'stationary' else
        [dict(node=5, speed=5. if condition == 'moving_no_cross' else 11., right=True, start=0.)])
    if mode == 'ip':
        request = patrol.Request(1, 1, mode, (5,), 2., 9., .08, .2, 1000, 1, 1)
    elif mode == 'id':
        request = patrol.Request(1, 1, mode, (5,), 2., 9., .15, .6, 256, 1, 1)
    else:
        request = patrol.Request(1, 1, mode, (5, 6), 2., 9., 2., .8, 1000, 4, 1, minimum_ratio=1.)
    cfg['requests'] = [request.json()]
    tag = f'mobility/{mode}/{condition}-edge{edge}-{seed}'
    method = f'fixed_{edge}'
    result = episode(cfg, method, prefix=OUT / 'native' / tag)
    assert result['admitted'] == 1, (mode, condition, edge, result['rejections'])
    return row_for(cfg, method, result, tag, initialization=None, seed=seed,
                   load=1, mix='controlled', scope='mobility', modality=mode,
                   condition=condition, edge=edge)


def raw_check(row):
    from slice_alloc.tools import patrol_verify as auditor
    auditor.OUT = OUT
    raw = auditor.check_episode(row)
    prefix = OUT / 'native' / row['path']
    cfg = json.loads(Path(str(prefix) + '-scenario.json').read_text())
    result = json.loads(Path(str(prefix) + '-result.json').read_text())
    env = InitialAllocationEnv(partitioned=row['method'] == 'static_quota')
    env.cfg = cfg
    env.requests = {r['id']: patrol.request_from_json(r) for r in cfg['requests']}
    env.ledger = QuotaLedger(cfg['public']) if env.partitioned else patrol.Ledger(cfg['public'])
    env.pressure = collections.Counter()
    env.peak_usage = np.zeros(len(patrol.RESOURCES))
    env.records = []
    try:
        for record, pair in zip(result['decisions'], result['observations']):
            env.current = pair['observation']
            obs = env.encode()
            assert env.observation_space.contains(obs), row['path']
            r = env.requests[env.current['request_id']]
            if r.mode == 'ndn':
                env.seen[r.tenant, r.content, r.version] += 1
            d = record['decision']
            if d['accept']:
                env.ledger.add(next(p for p in env.options if p.decision() == d))
            env.records.append(record)
    finally:
        env.close()
    raw['encoded_observations_in_range'] = True
    raw['quota_checked'] = env.partitioned
    return raw


def aggregate(group):
    requests = sum(r['requests'] for r in group)
    modes = {}
    for mode in MODES:
        n = sum(r['by_mode'][mode]['requests'] for r in group)
        modes[mode] = dict(requests=n, success_rate=(sum(r['by_mode'][mode]['success']
                          for r in group) / n if n else None))
    return dict(success_rate=sum(r['success'] for r in group) / requests,
        admission_rate=sum(r['admitted'] for r in group) / requests, requests=requests,
        episodes=len(group), by_mode=modes,
        by_load={str(load):sum(r['success'] for r in group if r['load'] == load) /
                 sum(r['requests'] for r in group if r['load'] == load)
                 for load in LOADS if any(r['load'] == load for r in group)},
        rejections=dict(sum((collections.Counter(r['rejections']) for r in group), collections.Counter())),
        mean_decision_ms=float(np.mean([r['decision_ms'] for r in group])))


def analyze(rows):
    methods = tuple(CONFIGS) + RULES
    stats, matrices = {}, {}
    for method in methods:
        group = [r for r in rows if r['method'] == method]
        stats[method] = aggregate(group)
        inits = sorted({r['initialization'] for r in group}, key=lambda x: x or 0)
        matrices[method] = np.array([[sum(r['success'] for r in group if r['seed'] == seed
                                          and r['initialization'] == init) /
                                     sum(r['requests'] for r in group if r['seed'] == seed
                                         and r['initialization'] == init)
                                    for seed in TEST] for init in inits])
        rates = matrices[method].mean(1)
        stats[method]['initialization_rates'] = rates.tolist()
        stats[method]['initialization_std'] = float(np.std(rates, ddof=1)) if len(rates) > 1 else None
    rng = np.random.default_rng(317709)
    comparisons = {}
    for left, right in [('ppo', m) for m in methods if m != 'ppo']:
        a, b = matrices[left], matrices[right]
        boot = []
        for _ in range(4000):
            ss = rng.integers(0, len(TEST), len(TEST))
            aa = rng.integers(0, len(a), len(a))
            bb = rng.integers(0, len(b), len(b))
            boot.append(a[np.ix_(aa, ss)].mean() - b[np.ix_(bb, ss)].mean())
        comparisons[f'{left}-{right}'] = dict(difference=float(a.mean() - b.mean()),
            initialization_and_scenario_ci95=np.quantile(boot, [.025, .975]).tolist())
    return dict(stats=stats, comparisons=comparisons, episodes=len(rows),
                decisions=sum(r['requests'] for r in rows))


def summarize_mobility(rows):
    import csv
    summary = []
    for row in rows:
        prefix = OUT / 'native' / row['path']
        with gzip.open(str(prefix) + '-events.csv.gz', 'rt') as stream:
            events = list(csv.DictReader(stream))
        switches = [x for x in events if x['event'] == 'associate' and x['node'] == '5'
                    and 2. <= float(x['time']) < 9.]
        if row['condition'] == 'cross_ap':
            assert switches, row['path']
        else:
            assert not switches, row['path']
        completed = sorted(float(x['time']) for x in events if x['event'] == 'complete'
                           and x['node'] == ('0' if row['modality'] == 'ip' else '5'))
        complete_gaps = np.diff(completed)
        interval = [x for x in events if 2. <= float(x['time']) < 9.]
        operations = collections.Counter(x['event'] for x in interval)
        control_bytes = sum(int(x['bytes']) for x in interval
                            if x['event'] == 'udp_send' and int(float(x['detail'])) in (1, 2, 3, 9))
        summary.append(dict(path=row['path'], mode=row['modality'], condition=row['condition'],
            edge=row['edge'], seed=row['seed'], success=row['success'],
            on_time=row['by_mode'][row['modality']]['on_time'],
            expected=row['by_mode'][row['modality']]['expected'],
            service_window_ap_changes=len(switches),
            max_completion_gap=float(max(complete_gaps)) if len(complete_gaps) else None,
            control_network_bytes=control_bytes, operation_counts=dict(operations)))
    return dict(scope='Fixed-deployment controlled native mechanism cases; not main success-rate samples',
                service_window=[2., 9.], rows=summary)


def verify(protocol, main_rows, mix_rows, mobility_rows, jobs):
    assert protocol['sources'] == hashes()
    assert protocol['binary_sha256'] == sha(patrol.NS3 / 'build/scratch/patrol-scenario')
    all_rows = main_rows + mix_rows + mobility_rows
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        raw = list(pool.map(raw_check, all_rows))
    inputs = collections.defaultdict(set)
    for r in main_rows + mix_rows:
        inputs[r['scope'], r['mix'], r['seed'], r['load']].add(r['scenario_sha256'])
    assert all(len(x) == 1 for x in inputs.values())
    replays = []
    torch.set_num_threads(1)
    for checkpoint in protocol['models']:
        path = ROOT / checkpoint['checkpoint']
        assert sha(path) == checkpoint['model_sha256']
        original_row = next(r for r in main_rows if r['method'] == checkpoint['method']
                            and r['initialization'] == checkpoint['initialization'])
        prefix = OUT / 'native' / original_row['path']
        original = json.loads(Path(str(prefix) + '-result.json').read_text())
        cfg = json.loads(Path(str(prefix) + '-scenario.json').read_text())
        model = PPO.load(path, device='cpu')
        assert model.policy_kwargs == checkpoint['policy_settings'] == CONFIGS[checkpoint['method']]
        assert model.num_timesteps == checkpoint['steps']
        assert not hasattr(model.policy, 'delta')
        with tempfile.TemporaryDirectory(prefix='initial-allocation-replay-') as tmp:
            again_prefix = Path(tmp) / 'case'
            again = episode(cfg, checkpoint['method'], model, again_prefix)
            for key in ('observations', 'decisions', 'settled'):
                assert original[key] == again[key]
            with gzip.open(str(prefix) + '-events.csv.gz', 'rb') as a, gzip.open(str(again_prefix) + '-events.csv.gz', 'rb') as b:
                assert a.read() == b.read()
        replays.append(dict(method=checkpoint['method'], initialization=checkpoint['initialization'], exact=True))
    write(OUT / 'raw_audit.json', raw)
    audit = dict(passed=True, sources=hashes(), same_inputs=True, raw_sla_reconstructed=True,
        encoded_observations_in_range=True, fixed_quota_feasibility_checked=True,
        episodes=len(all_rows), main_episodes=len(main_rows),
        decisions=sum(r['requests'] for r in all_rows), checkpoint_replays=replays)
    audit['model_parameters_match_manifest'] = True
    audit['trained_actor_has_nonzero_updates'] = True
    write(OUT / 'audit.json', audit)
    return audit


def ensure_tests():
    import re
    import subprocess
    import sys
    record = OUT / 'tests.json'
    if record.exists():
        old = json.loads(record.read_text())
        if old['sources'] == hashes() and old['log_sha256'] == sha(OUT / 'tests.log'):
            assert old['passed']
            return old
    with (OUT / 'tests.log').open('w') as stream:
        subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
                        '-s', 'slice_alloc/tests', '-q'], cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    log = (OUT / 'tests.log').read_text()
    count = int(re.search(r'Ran (\d+) tests', log).group(1))
    assert log.strip().endswith('OK')
    info = dict(passed=True, count=count, sources=hashes(), log_sha256=sha(OUT / 'tests.log'))
    write(record, info)
    return info


def publish(analysis, audit, competition, mobility, *, promote=True):
    from slice_alloc.tools.export_document import to_docx
    labels = dict(ppo='PPO（分层动作＋生命周期）', no_hierarchy='PPO去掉动作分层', no_calendar='PPO去掉逐资源日历', near_edge='近边优先', greedy='资源贪心',
                  marginal_price='资源价格规则', static_quota='固定配额规则')
    protocol = json.loads((OUT / 'protocol.json').read_text())
    training = json.loads((OUT / 'training.json').read_text())
    lines = ['# 第一部分：低空智联下的多模态网络切片智能分配', '',
        '论文题目为《面向低空智联的多模态网络切片技术》，三项研究内容为智能分配、资源预测和QoS保障。本次仅实施智能分配。', '',
        '## 系统与方法', '',
        '地面任务平台、双互连边缘、双AP及12架无人机（4架巡检移动、8架驻留）共享基础设施。IP、受限EID/RLOC和NDN业务具有不同的服务切片合同，竞争通信资源、映射状态和缓存。AP关联由真实无线机制产生。PPO决定接纳／拒绝以及功能部署和缓存配置，不选择AP，不在运行期迁移已有切片。', '',
        'PPO学习全部动作评分，保留主动拒绝。网络不读取手写价格评分，不叠加规则分数、固定拒绝偏置或残差。训练奖励为真实SLA结算成功数除以常数80，仅改变数值单位；评测按原始SLA统计。无模仿预训练、无额外费用惩罚或奖励塑形。', '',
        '一个PPO策略先学习接纳概率g，再学习可行部署的条件分布q。联合概率为拒绝1−g、部署j为g×q_j；训练使用联合概率和联合熵。正式执行固定为g≥0.5时接纳并选条件概率最大的部署，否则拒绝。无可行部署时必须拒绝。', '',
        '网络以17种资源当前及未来1／3／6秒的已签合同占用、候选逐资源需求和逐端点实际移动观测为输入，采用宽度64的4头注意力。策略与价值编码器参数独立。资源日历是已知合同到期重算，不是预测；候选仍按公开允许AP范围保守预约，不读取未来真实轨迹或未到达请求。', '',
        '两个消融与主方法使用相同环境、数据和训练预算：去掉动作分层时改用拒绝与部署的联合softmax；去掉逐资源日历时移除全部显式未来资源占用采样，仍保留当前占用、当前请求持续时间和原有合同汇总特征。两者不是旧版本结果拼接。', '',
        '## 主实验', '',
        '| 方法 | 全部到达请求的SLA成功率 | 接纳率 | 初始化标准差（百分点） |',
        '|---|---:|---:|---:|']
    for method, data in analysis['stats'].items():
        std = data['initialization_std']
        suffix = f'{std*100:.3f}' if std is not None else '—'
        lines.append(f"| {labels[method]} | {data['success_rate']*100:.2f}% | {data['admission_rate']*100:.2f}% | {suffix} |")
    lines += ['', '分母是全部到达请求；拒绝和已接纳但未满足SLA均计失败。主实验的各方法执行相同物理容量、业务合同和无线场景。规则是外部对照，不参与PPO评分；资源价格规则按新增资源占用和当前拥挤程度选成本最低的可行部署。', '',
        '| 请求数 | PPO成功率 | 资源价格规则成功率 |', '|---|---:|---:|']
    for load in LOADS:
        lines.append(f"| {load} | {analysis['stats']['ppo']['by_load'][str(load)]*100:.2f}% | {analysis['stats']['marginal_price']['by_load'][str(load)]*100:.2f}% |")
    lines += ['', '| 模态 | PPO成功率 | 资源价格规则成功率 |', '|---|---:|---:|']
    for mode in MODES:
        lines.append(f"| {mode} | {analysis['stats']['ppo']['by_mode'][mode]['success_rate']*100:.2f}% | {analysis['stats']['marginal_price']['by_mode'][mode]['success_rate']*100:.2f}% |")
    lines += ['', '## 配对比较', '']
    for pair, values in analysis['comparisons'].items():
        lo, hi = np.array(values['initialization_and_scenario_ci95']) * 100
        label = labels[pair.removeprefix('ppo-')]
        lines.append(f"- PPO相对{label}：{values['difference']*100:+.2f}个百分点；场景与初始化联合95%区间[{lo:+.2f}, {hi:+.2f}]。")
    lo, hi = analysis['comparisons']['ppo-marginal_price']['initialization_and_scenario_ci95']
    if lo > 0:
        conclusion = '本轮PPO相对资源价格规则的差值区间高于零，支持本场景和预算下的提升；不外推为所有环境都更优。'
    elif hi < 0:
        conclusion = '本轮PPO低于资源价格规则，不能宣称RL领先。该差距作为实测限制保留。'
    else:
        conclusion = '本轮尚未证明PPO优于资源价格规则，也不把区间跨零称为已证明等效。'
    lines += ['', conclusion, '', '## 检查点与训练预算', '',
        f'各初始化从头训练{BUDGETS[-1]}次原生交互，在预定步数{list(BUDGETS)}保存完整模型，按独立验证集SLA选择检查点；相同时选较早者。全部5个初始化进入最终统计，不挑选最高初始化。保存后重新加载所选模型并重跑验证，必须逐项复现；正式评测也从磁盘加载同一模型。', '',
        '| 方法 | 初始化 | 总训练步数 | 所选检查点 | 所选验证成功率 | 策略权重变化范数 |', '|---|---:|---:|---:|---:|---:|']
    for info in training:
        selected = next(s for s in info['validation_stages'] if s['step'] == info['steps'])
        lines.append(f"| {labels[info['method']]} | {info['initialization']} | {info['trained_steps']} | {info['steps']} | {selected['validation']['success_rate']*100:.2f}% | {info['actor_weight_change_norm']:.5f} |")
    lines += ['', f'每个方法均保留全部5个初始化。训练场景种子6000000—6999999，验证{list(VALIDATION)}，主测试{list(TEST)}。这些测试种子在本轮模型选择完成并冻结前不用于训练或验证。本轮从头训练，不复用历史模型或成绩。9000—9029仍保留供三部分累计评测。', '',
        '## 真实移动与三模态机制', '',
        '固定部署对照驻留、服务窗口内移动但不切换、服务窗口内跨AP。其他业务与容量固定；速度分别为0、5、11m/s。该实验检验移动及切换过程对服务的影响，不把速度差异归因为单独切换效应。', '',
        '| 模态 | 移动条件 | 成功用例／总用例 | 按时交付／生成总数 | 服务期AP切换数 |', '|---|---|---:|---:|---:|']
    for mode in MODES:
        for condition in ('stationary', 'moving_no_cross', 'cross_ap'):
            group = [r for r in mobility['rows'] if r['mode'] == mode and r['condition'] == condition]
            lines.append(f"| {mode} | {condition} | {sum(r['success'] for r in group)}/{len(group)} | {sum(r['on_time'] for r in group)}/{sum(r['expected'] for r in group)} | {sum(r['service_window_ap_changes'] for r in group)} |")
    lines += ['', 'EID定位更新依赖底层覆盖，不承诺无中断；不同模态业务和SLA不同，不能据上述表格宣称某个协议全面更优。移动观测已进入网络，但不预设增加移动特征一定提高成功率。', '',
        '## 三模态共享竞争', '',
        '仅改变三类请求比例，物理容量相同。共享池和固定三等分通信配额采用同一资源价格规则；映射状态及内容缓存保持各自用途。该实验检验资源池管理，不是PPO学习贡献，也不声称实现无线硬调度隔离。', '',
        '| 请求构成 | 共享池成功率 | 固定配额成功率 |', '|---|---:|---:|']
    for mix, methods in competition.items():
        lines.append(f"| {mix} | {methods['marginal_price']['success_rate']*100:.2f}% | {methods['static_quota']['success_rate']*100:.2f}% |")
    tests = json.loads((OUT / 'tests.json').read_text())
    lines += ['', '## 验证与边界', '',
        f"通过{tests['count']}项测试。主实验{analysis['episodes']}次、{analysis['decisions']}次决策；移动机制{len(mobility['rows'])}次，共享竞争{sum(v['episodes'] for d in competition.values() for v in d.values())}次。全部{audit['episodes']}次原始逐包记录重建SLA，{len(audit['checkpoint_replays'])}个所选模型完成代表性原生仿真精确重放。训练、模型选择、加载参数、分析、报告使用同一源码快照及指纹。", '',
        '这是一套固定高度、有限功能候选和受限传播模型下的服务切片研究。EID/RLOC是自定义受限机制，平均资源预约不是无线逐包硬保证。没有“残差必不退化”或“RL必优于规则”的保证。资源预测与运行期QoS保障尚未实现，不预填后续成功率；后续引入真实未知需求变化时，三部分应在同一环境重新累计比较。', '',
        '算法依据：[PPO原始论文](https://arxiv.org/abs/1707.06347)、[集合注意力原始论文](https://proceedings.mlr.press/v97/lee19d.html)。本方法是PPO动作与状态结构的课题适配，不宣称提出全新RL算法。具体依赖版本记录在runtime.json。', '',
        '[设计](../../initial_design.md) · [冻结协议](protocol.json) · [分析](analysis.json) · [审计](audit.json)']
    report = '\n'.join(lines) + '\n'
    (OUT / 'report.md').write_text(report)
    to_docx(report).save(OUT / 'report.docx')
    documents = [str((OUT / name).relative_to(ROOT)) for name in ('report.md', 'report.docx')]
    if promote:
        title = '面向低空智联的多模态网络切片技术'
        archive = OUT / 'previous_documents'
        for name in ('README.md', title + '.md', title + '.docx', 'slice_alloc/design.md'):
            source = ROOT / name
            if source.exists():
                target = archive / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
        design = (ROOT / 'slice_alloc/initial_design.md').read_text()
        (ROOT / 'slice_alloc/design.md').write_text(design)
        manuscript = '# ' + title + '\n\n三项研究内容：智能分配、资源预测、QoS保障。当前详述第一部分，后两部分保留接口和研究边界。\n\n'
        manuscript += design.replace('# ' + title, '## 第一部分设计', 1) + '\n'
        manuscript += '\n'.join('#' + line if line.startswith('#') else line
                                for line in report.splitlines() if '](' not in line) + '\n'
        (ROOT / (title + '.md')).write_text(manuscript)
        to_docx(manuscript).save(ROOT / (title + '.docx'))
        base = str(OUT.relative_to(ROOT))
        readme = f'''# {title}

研究内容：智能分配、资源预测、QoS保障。本次完成第一部分。

双AP、双互连边缘、12架无人机（4移动＋8驻留）承载IP／身份位置分离／NDN三模态服务切片。PPO自主决定接纳、拒绝、功能部署与缓存配置，实际AP关联由无线机制决定。

主方法为分层动作与逐资源生命周期表示的PPO，不叠加价格规则或固定拒绝偏置。主方法与两个消融各5个初始化从头训练，独立验证集选检查点，新测试集主方法结果为 **{analysis['stats']['ppo']['success_rate']*100:.2f}%**。规则仅作外部对照，结论和限制以报告为准。

[设计](slice_alloc/design.md) · [完整实测报告]({base}/report.md) · [报告Word]({base}/report.docx) · [正文Word]({title}.docx) · [冻结协议]({base}/protocol.json) · [逐包审计]({base}/audit.json) · [交付记录]({base}/delivery.json)

复现入口：`python -B -m slice_alloc.tools.initial_run all --jobs 12`。自动执行测试、冒烟、训练、模型选择、正式评测、审计与文档导出。已完成实验禁止覆盖。

历史实验保留用于追溯，不作为当前模型或成绩来源。资源预测与运行期QoS尚未实现，不预设其成功率。
'''
        (ROOT / 'README.md').write_text(readme)
        documents += ['README.md', 'slice_alloc/design.md', title + '.md', title + '.docx']
    delivery = dict(passed=True, sources=hashes(), first_part_only=True,
        primary_success=analysis['stats']['ppo']['success_rate'],
        formal_episodes=analysis['episodes'], formal_decisions=analysis['decisions'],
        raw_audited_episodes=audit['episodes'], native_checkpoint_replays=len(audit['checkpoint_replays']),
        unit_tests=tests['count'], total_training_steps=sum(r['trained_steps'] for r in training),
        policy='PPO; hierarchical admission/deployment; per-resource known-contract calendar attention',
        scientific_artifacts={name: sha(OUT / name) for name in
            ('plan.json', 'training.json', 'protocol.json', 'evaluation.json', 'analysis.json',
             'mobility_evaluation.json', 'mobility.json', 'competition_evaluation.json',
             'competition.json', 'audit.json', 'raw_audit.json', 'tests.json', 'tests.log', 'runtime.json')},
        documents={name: sha(ROOT / name) for name in documents},
        completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write(OUT / 'delivery.json', delivery)
    if promote:
        write(ROOT / 'slice_alloc/results/document_verification.json', dict(passed=True,
            scope='lifecycle PPO first research part', sources=hashes(),
            analysis_sha256=sha(OUT / 'analysis.json'), audit_sha256=sha(OUT / 'audit.json'),
            documents=delivery['documents'], delivery_sha256=sha(OUT / 'delivery.json')))
    return delivery


def pipeline(jobs, *, promote=True):
    import platform
    import stable_baselines3
    plan = dict(sources=hashes(), hyperparameters=HYPER, policy_settings=CONFIGS,
        reward_unit=REWARD_UNIT, training_seed_range=[6000000, 6999999],
        initializations=INITS, budgets=BUDGETS, validation_seeds=VALIDATION,
        test_seeds=TEST, mix_seeds=MIX_SEEDS, mobility_seeds=MOBILITY_SEEDS,
        loads=LOADS, primary='ppo', rules=RULES, shaping=False, demonstrations=0,
        fixed_rule_action_score=False, selection='validation SLA, earliest on tie; all initializations retained',
        binary_sha256=sha(patrol.NS3 / 'build/scratch/patrol-scenario'))
    if (OUT / 'plan.json').exists():
        assert json.loads((OUT / 'plan.json').read_text()) == json.loads(json.dumps(plan)), 'Frozen plan changed'
    else:
        write(OUT / 'plan.json', plan)
    frozen = OUT / 'frozen_sources'
    for name, expected in hashes().items():
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(ROOT / name, target)
        assert sha(target) == expected
    ensure_tests()
    write(OUT / 'runtime.json', dict(python=platform.python_version(), numpy=np.__version__,
        torch=torch.__version__, stable_baselines3=stable_baselines3.__version__,
        cpu_threads_per_worker=1, device='cpu'))
    if not (OUT / 'mobility_evaluation.json').exists():
        specs = [(mode, condition, edge, seed) for mode in MODES
                 for condition in ('stationary', 'moving_no_cross', 'cross_ap')
                 for edge in (1, 2) for seed in MOBILITY_SEEDS]
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(mobility_case, specs))
        write(OUT / 'mobility_evaluation.json', rows)
    mobility_rows = json.loads((OUT / 'mobility_evaluation.json').read_text())
    mobility = summarize_mobility(mobility_rows)
    write(OUT / 'mobility.json', mobility)
    print('MOBILITY GATE', len(mobility_rows), flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        training = list(pool.map(train_one, [(m, i) for m in CONFIGS for i in INITS]))
    write(OUT / 'training.json', training)
    protocol = dict(plan, models=[{k: info[k] for k in
        ('method', 'initialization', 'steps', 'trained_steps', 'policy_settings', 'checkpoint', 'model_sha256')}
        for info in training], tests_sha256=sha(OUT / 'tests.log'),
        frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    if (OUT / 'protocol.json').exists():
        stored = json.loads((OUT / 'protocol.json').read_text())
        assert stored['sources'] == hashes() and stored['models'] == protocol['models']
        protocol = stored
    else:
        write(OUT / 'protocol.json', protocol)
    if not (OUT / 'evaluation.json').exists():
        specs = [(m, init, TEST, 'balanced', 'main') for m in CONFIGS for init in INITS]
        specs += [(method, None, TEST, 'balanced', 'main') for method in RULES]
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = [row for group in pool.map(eval_group, specs) for row in group]
        write(OUT / 'evaluation.json', dict(sources=hashes(), protocol_sha256=sha(OUT / 'protocol.json'), rows=rows))
    data = json.loads((OUT / 'evaluation.json').read_text())
    assert data['sources'] == hashes() and data['protocol_sha256'] == sha(OUT / 'protocol.json')
    analysis = analyze(data['rows'])
    write(OUT / 'analysis.json', analysis)
    print('MAIN RESULTS', {name:round(d['success_rate']*100, 4) for name,d in analysis['stats'].items()}, flush=True)
    if not (OUT / 'competition_evaluation.json').exists():
        specs = [(method, None, MIX_SEEDS, mix, 'competition') for mix in
                 ('balanced', 'ip_heavy', 'id_heavy', 'ndn_heavy')
                 for method in ('marginal_price', 'static_quota')]
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = [row for group in pool.map(eval_group, specs) for row in group]
        write(OUT / 'competition_evaluation.json', rows)
    mix_rows = json.loads((OUT / 'competition_evaluation.json').read_text())
    competition = {mix:{method:aggregate([r for r in mix_rows if r['mix']==mix and r['method']==method])
                       for method in ('marginal_price','static_quota')}
                   for mix in ('balanced','ip_heavy','id_heavy','ndn_heavy')}
    write(OUT / 'competition.json', competition)
    audit = verify(protocol, data['rows'], mix_rows, mobility_rows, jobs)
    delivery = publish(analysis, audit, competition, mobility, promote=promote)
    assert delivery['sources'] == protocol['sources'] == audit['sources'] == hashes()
    for name, expected in delivery['scientific_artifacts'].items():
        assert sha(OUT / name) == expected, name
    for name, expected in delivery['documents'].items():
        assert sha(ROOT / name) == expected, name
    write(OUT / 'completion.json', dict(passed=True, primary_success=delivery['primary_success'],
        delivery_sha256=sha(OUT / 'delivery.json'), same_frozen_sources=True,
        selected_models_roundtrip_verified=True, raw_audit_episodes=audit['episodes']))
    print('COMPLETE', delivery['primary_success'], flush=True)
    return delivery


def smoke(jobs):
    global OUT, INITS, BUDGETS, VALIDATION, TEST, MIX_SEEDS, MOBILITY_SEEDS
    original = OUT, INITS, BUDGETS, VALIDATION, TEST, MIX_SEEDS, MOBILITY_SEEDS
    with tempfile.TemporaryDirectory(prefix='smoke-', dir=OUT) as tmp:
        OUT, INITS, BUDGETS = Path(tmp), (999,), (1024,)
        VALIDATION, TEST, MIX_SEEDS, MOBILITY_SEEDS = (10090,), (10091,), (10092,), (10093,)
        try:
            delivery = pipeline(jobs, promote=False)
            info = dict(passed=True, sources=hashes(), native_training_steps=1024,
                main_episodes=delivery['formal_episodes'], raw_audited_episodes=delivery['raw_audited_episodes'],
                includes_model_selection_reload_evaluation_audit_and_word_export=True)
        finally:
            OUT, INITS, BUDGETS, VALIDATION, TEST, MIX_SEEDS, MOBILITY_SEEDS = original
    write(OUT / 'smoke.json', info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('all', 'smoke'))
    parser.add_argument('--jobs', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'delivery.json').exists(), 'Completed experiment is immutable'
    if args.command == 'smoke':
        smoke(args.jobs)
        return
    if not (OUT / 'smoke.json').exists():
        smoke(args.jobs)
    assert json.loads((OUT / 'smoke.json').read_text())['sources'] == hashes()
    pipeline(args.jobs)


if __name__ == '__main__':
    main()


