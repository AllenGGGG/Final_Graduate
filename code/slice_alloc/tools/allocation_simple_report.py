"""Independently verify the completed simple study and report paired results.

This tool never selects a checkpoint or changes an experiment parameter.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from slice_alloc import allocation_final as a
from slice_alloc import modal_binding as b
from slice_alloc.tools import modal_binding_run as native


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify_episode(out, row):
    prefix = out/'native'/f"{row['method']}-{row['init']}"/f"{row['load']}-{row['seed']}"
    cfg = read(str(prefix)+'-scenario.json')
    result = read(str(prefix)+'-result.json')
    metrics = read(str(prefix)+'-metrics.json')
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    assert digest == metrics['scenario_sha256'] == result['scenario_sha256']
    for key in ('requests', 'success', 'admitted', 'by_service'):
        assert row[key] == metrics[key] == result[key], (prefix, key)
    assert result['requests'] == len(cfg['requests']) == row['load']
    assert native.audit(prefix, result, cfg)['audited']
    requests = {r['id']: b.Service(**{**r, 'members': tuple(r['members'])}) for r in cfg['requests']}
    records = result['decisions']
    assert len(records) == len(requests)
    assert {r['request'] for r in records} == set(requests)
    accepted = {r['request']: r for r in records if r['decision']['accept']}
    assert len(accepted) == result['admitted']
    assert {s['id'] for s in result['settled']} == set(accepted)
    assert sum(s['success'] for s in result['settled']) == result['success']
    for item in result['settled']:
        assert item['expected'] == requests[item['id']].expected > 0
    # Reconstruct the committed reservations from saved causal observations.
    ledger = b.p.Ledger(cfg['public'])
    for message in result['observations']:
        observation = message['observation']
        if observation['kind'] == 'done':
            continue
        decision = message['decision']
        request = requests[observation['request_id']]
        assert decision['request_id'] == request.id
        assert 'rate_factor' not in decision
        ledger.release(observation['time'])
        if not decision['accept']:
            continue
        mode = decision['mode']
        assert mode in a.COMPATIBLE[request.service]
        if request.service == 'telemetry':
            plan = b.p.compile_plan(request.bound('ip'), decision['edge'], decision['server'], 0,
                                    ledger.pub, observation)
        else:
            plan = b.compile_plan(request, mode, decision['edge'], decision['server'], decision['cache'],
                                  ledger.pub, observation)
        assert ledger.feasible(plan), ('illegal reservation', prefix, request.id)
        ledger.add(plan)
    # Check packet direction independently of the aggregate SLA audit.
    with gzip.open(str(prefix)+'-events.csv.gz', 'rt') as stream:
        for event in csv.DictReader(stream):
            sid = int(event['slice'])
            if sid not in requests or requests[sid].service != 'telemetry':
                continue
            if event['event'] == 'emit':
                assert int(event['node']) in requests[sid].members
            if event['event'] == 'complete':
                assert int(event['node']) == 0
    forced = sum(not r['decision']['accept'] and not r['feasible'] for r in records)
    voluntary = sum(not r['decision']['accept'] and bool(r['feasible']) for r in records)
    violations = result['admitted'] - result['success']
    assert forced + voluntary + violations + result['success'] == result['requests']
    return dict(**row, forced_reject=forced, voluntary_reject=voluntary,
                admitted_violations=violations, modes=result['modes'], scenario_sha256=digest)


def summarize(rows):
    total = sum(r['requests'] for r in rows)
    counts = {key: sum(r[key] for r in rows) for key in
              ('requests', 'success', 'admitted', 'forced_reject', 'voluntary_reject', 'admitted_violations')}
    return dict(**counts, success_rate=counts['success']/total,
                admission_rate=counts['admitted']/total,
                forced_reject_rate=counts['forced_reject']/total,
                voluntary_reject_rate=counts['voluntary_reject']/total,
                violation_rate_all=counts['admitted_violations']/total,
                by_service={s: dict(requests=sum(r['by_service'][s]['requests'] for r in rows),
                                    success=sum(r['by_service'][s]['success'] for r in rows)) for s in b.SERVICES},
                modes=dict(sum((Counter(r['modes']) for r in rows), Counter())))


def analyze(rows, protocol):
    inits, seeds, loads = protocol['initializations'], protocol['test'], protocol['loads']
    lookup = {(r['method'],r['init'],r['seed'],r['load']):r for r in rows}
    def scenario_rate(method, init, seed):
        rr = [lookup[method,init,seed,load] for load in loads]
        return sum(r['success'] for r in rr)/sum(r['requests'] for r in rr)
    learned = np.array([[scenario_rate('double_dqn', i, s) for s in seeds] for i in inits])
    rng = np.random.default_rng(20261103)
    seed_draws = rng.integers(len(seeds), size=(20000,len(seeds)))
    init_draws = rng.integers(len(inits), size=(20000,len(inits)))
    rl_boot = learned[init_draws[:,:,None], seed_draws[:,None,:]].mean((1,2))
    output = dict(methods={}, paired_comparisons={}, initialization_results={},
                  interval_method='20000 paired bootstrap draws: scenario seed is the cluster (both loads together); independently resample RL initializations; no ensemble',
                  load_results={})
    for method in protocol['methods']:
        chosen = [r for r in rows if r['method'] == method]
        result = summarize(chosen)
        if method == 'double_dqn':
            boot = rl_boot
        else:
            rule = np.array([scenario_rate(method,0,s) for s in seeds])
            boot = rule[seed_draws].mean(1)
            delta = rl_boot-boot
            output['paired_comparisons'][method] = dict(
                difference=float(learned.mean()-rule.mean()), ci95=np.quantile(delta,[.025,.975]).tolist())
        result['ci95'] = np.quantile(boot,[.025,.975]).tolist()
        output['methods'][method] = result
        output['load_results'][method] = {str(load): summarize([r for r in chosen if r['load']==load]) for load in loads}
    output['initialization_results'] = {str(i):summarize([r for r in rows if r['method']=='double_dqn' and r['init']==i]) for i in inits}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    protocol = read(out/'protocol.json')
    registration = read(out/'analysis_registration.json')
    assert registration['source_sha256'] == sha(__file__)
    assert registration['protocol_sha256'] == sha(out/'protocol.json')
    test_log = (out/'full-tests.log').read_text()
    assert '\nOK\n' in test_log and '\nFAILED ' not in test_log
    test_count = int(re.search(r'Ran (\d+) tests',test_log)[1])
    for name,digest in protocol['sources'].items():
        assert sha(b.p.ROOT/'slice_alloc'/name) == digest, name
        assert sha(out/'frozen'/name) == digest, name
    assert sha(b.p.NS3/'build/scratch/allocation-final-scenario') == protocol['binary_sha256']
    for name,digest in read(out/'test_started.json').items():
        assert sha(out/name) == digest, name
    completion = read(out/'training_complete.json')
    selected_steps = {}
    for model in completion['models']:
        assert model['steps'] == max(protocol['budgets']) and model['updates'] > 0
        assert sha(out/model['model']) == model['model_sha256']
        assert model['protocol_sha256'] == sha(out/'protocol.json')
        assert model['selected'] == max(model['checkpoints'], key=lambda x:(x['validation']['rate'],-x['step']))
        for stage in model['checkpoints']:
            assert sha(out/stage['path']) == stage['sha256']
        assert all(protocol['training_seed_range'][0] <= s < protocol['training_seed_range'][1] for s in model['training_seeds'])
        selected_steps[str(model['init'])] = model['selected']['step']
    rows = read(out/'evaluation.json')
    expected = {(m,i,s,l) for m in protocol['methods']
                for i in (protocol['initializations'] if m=='double_dqn' else [0])
                for s in protocol['test'] for l in protocol['loads']}
    assert len(rows)==len(expected)
    assert {(r['method'],r['init'],r['seed'],r['load']) for r in rows} == expected
    verified = [verify_episode(out,row) for row in rows]
    for seed in protocol['test']:
        for load in protocol['loads']:
            assert len({r['scenario_sha256'] for r in verified if r['seed']==seed and r['load']==load}) == 1
    analysis = analyze(verified,protocol)
    analysis['selected_checkpoints'] = selected_steps
    write(out/'analysis.json',analysis)
    write(out/'verification.json',dict(
        formal_training_complete=True, formal_evaluation_complete=True,
        passed_tests=test_count,
        verified_native_episodes=len(verified), all_packet_audits_pass=True,
        all_direction_audits_pass=True, all_reservation_replays_pass=True,
        all_methods_share_scenarios=True, protocol_sha256=sha(out/'protocol.json'),
        full_test_log_sha256=sha(out/'full-tests.log')))
    report = ['# 无档位切片分配：实际验证结果', '',
              '主方法为Masked Double-DQN。5个初始化分别训练32768次决策；每个初始化仅按验证集选择检查点，全部纳入测试，无投票或择优初始化。', '',
              '| 方法 | SLA成功率 | 95%区间 | 接纳率 | 主动拒绝率 |',
              '|---|---:|---:|---:|---:|']
    for method,result in analysis['methods'].items():
        lo,hi=result['ci95']
        report.append(f"| {method} | {result['success_rate']:.2%} | [{lo:.2%}, {hi:.2%}] | {result['admission_rate']:.2%} | {result['voluntary_reject_rate']:.2%} |")
    comparison=analysis['paired_comparisons']['greedy']
    lo,hi=comparison['ci95']
    report += ['',f"RL − Greedy：{100*comparison['difference']:+.2f}个百分点，配对95%区间[{100*lo:+.2f}, {100*hi:+.2f}]。",'',
               '区间同时重采样场景与RL初始化，同一场景两种负载一起抽样。该区间未作多个基线比较校正，Greedy为预先指定主对照。', '',
               f"全部{len(verified)}个正式原生回合已重新核对逐包SLA、遥测方向、资源预约及场景一致性。完整测试日志见full-tests.log；模型、输入、逐包事件和决策保留在本目录。",'',
               '本轮去掉档位、超订与拉格朗日约束，结果属于此简化环境；不能与旧版成绩直接相减归因。资源账本不提供逐切片确定性QoS保证。', '',
               '详细分业务、分负载、初始化结果见analysis.json，独立核验见verification.json。']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
              if p.is_file() and p.name not in ('manifest.json','pipeline.log','process.json')}
    write(out/'manifest.json',manifest)
    print((out/'report.md').read_text(),flush=True)


if __name__=='__main__':
    main()
