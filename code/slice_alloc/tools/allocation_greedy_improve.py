"""Bounded Greedy tuning with separate development, validation and test scenes.

Network, contracts, mask and success metric are the frozen first-part model.
No RL model is trained or selected here. Test episodes retain native traces
and pass the original packet, direction and reservation audits.
"""
import argparse
import concurrent.futures as cf
from itertools import product
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from slice_alloc.tools import allocation_confirm_train as run
from slice_alloc.tools import allocation_simple_report as audit
from slice_alloc.tools import allocation_strong_baseline_supplement as rules

OUT=Path(os.environ.get('LOWALT_GREEDY_OUTPUT',str(run.PKG/'results/greedy-improvement'))).resolve()
DEV=tuple(range(548000,548012))
VALIDATION=tuple(range(549000,549030))
TEST=tuple(range(550000,550060))
LOADS=(32,64,96)
GRID=tuple(product((0.,1.,2.,4.),(0.,.5,1.),(0.,1.)))
THRESHOLDS=(.5,1.,2.,4.,None)


def configuration(name,rule='modal_greedy',parameters=(2.,.5,0.),threshold=None):
    return dict(name=name,rule=rule,parameters=list(parameters),threshold=threshold)


def rate(rows):
    return sum(r['success'] for r in rows)/sum(r['requests'] for r in rows)


def validate_rows(rows,names,seeds):
    expected={(name,s,l) for name in names for s in seeds for l in LOADS}
    actual={(r['method'],r['seed'],r['load']) for r in rows}
    if len(rows)!=len(actual) or actual!=expected:
        raise ValueError('Missing or duplicate scenario results')
    for r in rows:
        if not (r['requests']==r['load'] and
                r['requests']==r['success']+r['forced_reject']+r['voluntary_reject']+r['admitted_violations'] and
                r['admitted']==r['success']+r['admitted_violations']):
            raise ValueError('Inconsistent arrival, rejection or SLA counts')


def choose(configs,rows):
    """Only callers' development/validation rows enter the deterministic choice."""
    return max(configs,key=lambda c:rate([r for r in rows if r['method']==c['name']]))


def source_hashes():
    sources=[__file__,rules.__file__,audit.__file__,run.PKG/'supplement_tests/test_greedy_study.py']
    return {str(Path(p).relative_to(run.PKG)):run.sha(p) for p in sources}


def freeze():
    run.check()
    value=dict(sources=source_hashes(),primary_protocol_sha256=run.sha(run.OUT/'protocol.json'),
        binary_sha256=run.sha(run.environment.NS3/'build/scratch/allocation-final-scenario'),
        development=list(DEV),validation=list(VALIDATION),test=list(TEST),loads=list(LOADS),
        grid=[list(p) for p in GRID],thresholds=list(THRESHOLDS),
        selection='24 score formulas on development; five admission thresholds plus original modal rule on validation; tie first',
        primary='selected_greedy minus original_modal; paired scene-cluster bootstrap',
        secondary='selected_greedy minus resource_greedy and archived_modal_price; Holm corrected together with primary',
        bootstrap_seed=20261108,bootstrap_draws=20000,
        scope='Greedy-only experiment; no RL tuning, changed capacities, business weights or SLA',
        original_test_seeds_not_used_for_tuning=[447000,447059],
        audits='all held-out episodes: packet SLA, direction, reservation replay; development/validation use native aggregate outcomes')
    path=OUT/'protocol.json'
    if path.exists():
        if run.read(path)!=value:raise ValueError('Greedy study source/configuration changed after registration')
    else:run.write(path,value)
    return value


def task(spec):
    import torch
    torch.set_num_threads(1)
    cfg,seed,load,stage=spec
    return rules.episode(seed,load,cfg['name'],cfg['parameters'],cfg['threshold'],
        output=OUT/stage,trace=stage=='test',rule=cfg['rule'])


def stage(pool,name,configs,seeds):
    path=OUT/(name+'.json')
    if path.exists():
        prior=run.read(path)
        if prior['configurations']!=configs:raise ValueError('Cached stage uses different configurations')
        validate_rows(prior['rows'],[c['name'] for c in configs],seeds)
        return prior['rows']
    if name!='test' and (OUT/'test_started.json').exists():
        raise ValueError('Cannot rerun development or validation after opening test')
    specs=[(c,s,l,name) for c in configs for s in seeds for l in LOADS]
    rows=[]
    for row in pool.map(task,specs):
        rows.append(row)
        if len(rows)%90==0:print(name.upper(),len(rows),len(specs),flush=True)
    validate_rows(rows,[c['name'] for c in configs],seeds)
    run.write(path,dict(configurations=configs,rows=rows))
    return rows


def analyze(rows):
    names=('resource_greedy','original_modal','development_modal','selected_greedy','archived_modal_price')
    validate_rows(rows,names,TEST)
    lookup={(r['method'],r['seed'],r['load']):r for r in rows}
    rng=np.random.default_rng(20261108)
    scenes=rng.integers(len(TEST),size=(20000,len(TEST)))
    out=dict(methods={},comparisons={},per_load={},
        intervals='20000 paired scene bootstrap draws; all loads clustered within each scene',
        initialization='deterministic rules; no neural initialization resampling')
    boots={}
    for name in names:
        rr=[r for r in rows if r['method']==name]
        summary=audit.summarize(rr)
        summary['violation_rate_admitted']=summary['admitted_violations']/summary['admitted'] if summary['admitted'] else 0.
        summary['minimum_service_success']=min(v['success']/v['requests'] for v in summary['by_service'].values())
        values=np.array([sum(lookup[name,s,l]['success'] for l in LOADS)/sum(LOADS) for s in TEST])
        boots[name]=values[scenes].mean(1)
        summary['ci95']=np.quantile(boots[name],[.025,.975]).tolist()
        out['methods'][name]=summary
        out['per_load'][name]={str(l):audit.summarize([r for r in rr if r['load']==l]) for l in LOADS}
    for baseline in ('original_modal','resource_greedy','archived_modal_price'):
        delta=boots['selected_greedy']-boots[baseline]
        difference=out['methods']['selected_greedy']['success_rate']-out['methods'][baseline]['success_rate']
        out['comparisons']['selected-'+baseline]=dict(difference=difference,
            ci95=np.quantile(delta,[.025,.975]).tolist(),
            centered_bootstrap_p=float((1+np.count_nonzero(np.abs(delta-difference)>=abs(difference)))/(len(delta)+1)))
    running=0.
    for rank,(_,v) in enumerate(sorted(out['comparisons'].items(),key=lambda kv:kv[1]['centered_bootstrap_p'])):
        running=max(running,min(1.,(len(out['comparisons'])-rank)*v['centered_bootstrap_p']))
        v['holm_adjusted_p']=running
    return out


def render(analysis,selected):
    lines=['# Greedy 独立优化验证','',
        '沿用冻结的物理网络、三种模态、合同、动作掩码和成功率口径。仅调整现有规则评分及准入阈值，不训练或选择 RL。',
        '',f'开发集548000—548011，验证集549000—549029；24组评分公式后，在验证集选择5个阈值或原规则。最终选择：`{selected}`。',
        '', '测试集550000—550059，32/64/96请求。全部900个测试回合保存逐包轨迹，并通过SLA、方向、预约独立审计。',
        '', '| 方法 | 成功率 | 违约/已接纳 | 最低业务成功率 |','|---|---:|---:|---:|']
    for name,s in analysis['methods'].items():
        lines.append(f"| {name} | {s['success_rate']:.3%} | {s['violation_rate_admitted']:.3%} | {s['minimum_service_success']:.3%} |")
    lines+=['','| 配对比较 | 差值/百分点 | 未校正95%区间/百分点 | Holm近似p |','|---|---:|---:|---:|']
    for name,c in analysis['comparisons'].items():
        lo,hi=c['ci95'];lines.append(f"| {name} | {100*c['difference']:+.3f} | [{100*lo:+.3f},{100*hi:+.3f}] | {c['holm_adjusted_p']:.4f} |")
    lines+=['','三个比较共同进行Holm校正。测试集打开后不重新选参数；开发、验证、测试逐回合数据全部保留。',
        '本批与447000系列属于不同测试场景，跨批成功率不直接用于比较。参数变化只代表本次有限搜索结果，不主张全局最优。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();protocol=freeze()
    with cf.ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        configs=[configuration(f'formula-{i:02}',parameters=p) for i,p in enumerate(GRID)]
        dev=stage(pool,'development',configs,DEV);best=choose(configs,dev)
        candidates=[configuration('original_modal')]+[
            configuration(f'threshold-{i}',parameters=best['parameters'],threshold=t) for i,t in enumerate(THRESHOLDS)]
        val=stage(pool,'validation',candidates,VALIDATION);selected=choose(candidates,val)
        selection=dict(development=best,validation=selected,
            development_rates={c['name']:rate([r for r in dev if r['method']==c['name']]) for c in configs},
            validation_rates={c['name']:rate([r for r in val if r['method']==c['name']]) for c in candidates})
        path=OUT/'selection.json'
        if path.exists() and run.read(path)!=selection:raise ValueError('Frozen selection changed')
        run.write(path,selection)
        test_configs=[configuration('resource_greedy',rule='greedy'),configuration('original_modal'),
            configuration('development_modal',parameters=best['parameters']),
            {**selected,'name':'selected_greedy'},
            configuration('archived_modal_price',rule='modal_price_threshold',parameters=(2.,.5,1.),threshold=2.)]
        registration=dict(protocol_sha256=run.sha(OUT/'protocol.json'),selection_sha256=run.sha(path),configurations=test_configs)
        marker=OUT/'test_started.json'
        if marker.exists() and run.read(marker)!=registration:raise ValueError('Test registration changed')
        run.write(marker,registration)
        rows=stage(pool,'test',test_configs,TEST)
    # Always re-read native evidence, including when resuming from a completed stage.
    core=('method','init','seed','load','action_sha256','requests','success','admitted','by_service')
    for r in rows:
        if audit.verify_episode(OUT/'test',{k:r[k] for k in core})!=r:raise ValueError('Saved test differs from native audit')
    for seed in TEST:
        for load in LOADS:
            if len({r['scenario_sha256'] for r in rows if r['seed']==seed and r['load']==load})!=1:
                raise ValueError('Methods did not share the same scenario')
    analysis=analyze(rows);run.write(OUT/'analysis.json',analysis)
    verification=dict(native_episodes=len(rows),all_packet_audits=True,all_direction_audits=True,
        all_reservation_replays=True,all_methods_share_scenarios=True,
        protocol_sha256=run.sha(OUT/'protocol.json'),selection_sha256=run.sha(OUT/'selection.json'),
        test_sha256=run.sha(OUT/'test.json'),sources=source_hashes())
    run.write(OUT/'verification.json',verification)
    (OUT/'report.md').write_text(render(analysis,selected))
    print('GREEDY_COMPLETE', {k:v['success_rate'] for k,v in analysis['methods'].items()},flush=True)


if __name__=='__main__':main()
