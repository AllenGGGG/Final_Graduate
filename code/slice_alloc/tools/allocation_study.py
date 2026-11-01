"""Versioned initial-allocation study; native SLA labels, no imitation or forecast.

Invoked as python -m slice_alloc.tools.allocation_study COMMAND.  Historical
patrol results remain intact.  Development, training and final tests are disjoint.
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import copy
import gzip
import hashlib
import json
import time
import tempfile
from pathlib import Path
import numpy as np
from gymnasium import spaces
from slice_alloc import patrol
from slice_alloc.patrol import (PatrolEnv, RESOURCES, INDEX, Ledger, NS3, ROOT,
                                baseline, scenario, request_from_json)
from slice_alloc.tools.patrol_run import sha, write

OUT = ROOT / 'slice_alloc/results/allocation_v3'
LOADS = (64, 80)
DEV_SEEDS = tuple(range(7000, 7008))
TEST_SEEDS = tuple(range(8000, 8030))
INITS = (11, 23, 37, 53, 71)
STEPS = 32768
BASELINES = ('first_fit', 'near_edge', 'greedy', 'price', 'cache_aware', 'lifetime_price')
STATE_DIM = patrol.STATE_DIM + 2 * len(RESOURCES)
CANDIDATE_DIM = patrol.CANDIDATE_DIM + len(RESOURCES) + 3
SOURCES = ('slice_alloc/patrol.py', 'slice_alloc/patrol_contract.schema.json',
           'slice_alloc/execution/patrol_contract.hpp', 'slice_alloc/execution/patrol_scenario.cc',
           'slice_alloc/src/allocation_policy.py', 'slice_alloc/tools/allocation_study.py',
           'slice_alloc/tools/patrol_verify.py', 'slice_alloc/tools/patrol_run.py')


def hashes():
    return {p: sha(ROOT / p) for p in SOURCES}


class AllocationEnv(PatrolEnv):
    """Only current request, past content arrivals and admitted contracts are read."""
    def __init__(self, seed=1, loads=LOADS, profiles=('mixed',), potential=False):
        super().__init__(seed=seed, loads=loads, profiles=profiles, shaping=False)
        self.potential = potential
        self.seen_content = collections.Counter()
        self.observation_space = spaces.Dict({
            'state': spaces.Box(-10, 10, (STATE_DIM,), np.float32),
            'candidates': spaces.Box(-10, 10, (patrol.MAX_ACTIONS, CANDIDATE_DIM), np.float32),
            'mask': spaces.Box(0, 1, (patrol.MAX_ACTIONS,), np.float32)})

    def reset(self, seed=None, options=None):
        self.seen_content = collections.Counter()
        if options is None or 'scenario' not in options:
            # Train-only range excludes all development and final evaluation IDs.
            cfg = scenario(int(self.rng.integers(100000, 900000)),
                           int(self.rng.choice(self.loads)), str(self.rng.choice(self.profiles)))
            options = dict(scenario=cfg)
        return super().reset(seed=seed, options=options)

    def encode(self):
        original = super().encode()
        state = np.zeros(STATE_DIM, np.float32)
        rows = np.zeros((patrol.MAX_ACTIONS, CANDIDATE_DIM), np.float32)
        state[:patrol.STATE_DIM] = original['state']
        rows[:, :patrol.CANDIDATE_DIM] = original['candidates']
        if self.current['kind'] != 'done':
            now = self.current['time']
            r = self.requests[self.current['request_id']]
            # Resource release calendar contains only already accepted contracts.
            for i, horizon in enumerate((3., 6.)):
                at = patrol.STATE_DIM + i * len(RESOURCES)
                state[at:at+len(RESOURCES)] = self.ledger.usage(now+horizon) / self.ledger.capacity
            u = self.ledger.usage(now) / self.ledger.capacity
            for i, p in enumerate(self.options, 1):
                cost = float(np.dot(p.peak/self.ledger.capacity, 1+4*u*u))
                if p.cache:
                    cost += .1*r.size/self.ledger.capacity[INDEX[f'cache:{p.cache}']]
                rows[i, patrol.CANDIDATE_DIM:patrol.CANDIDATE_DIM+len(RESOURCES)] = (
                    p.peak/self.ledger.capacity * (r.drain-now)/8)
                rows[i, -3:] = [cost, min(10, self.seen_content[r.tenant, r.content, r.version])/10
                                if r.mode == 'ndn' else 0, (r.drain-now)/8]
        return dict(state=state, candidates=rows, mask=original['mask'])

    def phi(self):
        if self.current['kind'] == 'done':
            return 0.
        u = self.ledger.usage(self.current['time']) / self.ledger.capacity
        return -.1*float(np.dot(u, u))

    def step(self, action):
        before = self.phi() if self.potential else 0.
        r = self.requests[self.current['request_id']]
        if r.mode == 'ndn':
            self.seen_content[r.tenant, r.content, r.version] += 1
        obs, reward, done, truncated, info = super().step(action)
        # gamma=1; telescopes to a trajectory-independent constant, terminal phi=0.
        after = self.phi() if self.potential else 0.
        info['sla_reward'] = reward
        info['potential_reward'] = after-before
        return obs, reward+after-before, done, truncated, info


def choose(method, env, model=None, obs=None):
    if method == 'lifetime_price':
        if not env.options:
            return 0
        r = env.requests[env.current['request_id']]
        # Integrate congestion over known reservation-release intervals.  This
        # uses no future arrivals and is a stronger rule than instantaneous price.
        points = sorted({r.start,r.drain} | {t for p in env.ledger.plans
                                             for a,b,_ in p.segments for t in (a,b)
                                             if r.start < t < r.drain})
        weights = np.zeros(len(RESOURCES))
        for a,b in zip(points,points[1:]):
            u = env.ledger.usage((a+b)/2)/env.ledger.capacity
            weights += (b-a)*(1+4*u*u)
        return min(range(len(env.options)), key=lambda i: (
            np.dot(env.options[i].peak/env.ledger.capacity, weights),
            not bool(env.options[i].cache)))+1
    if method in BASELINES:
        return baseline(method, env)
    return int(model.predict(obs, deterministic=True)[0])


def run_episode(cfg, method, model=None, prefix=None):
    e = AllocationEnv()
    try:
        obs, _ = e.reset(options=dict(scenario=cfg, prefix=prefix, trace=prefix is not None))
        done = False
        while not done:
            action = choose(method, e, model, obs)
            obs, _, done, _, _ = e.step(action)
        result = e.result()
        if prefix is not None:
            write(str(prefix)+'-result.json', result)
            raw = Path(str(prefix)+'-events.csv')
            with raw.open('rb') as f, gzip.open(str(raw)+'.gz', 'wb', compresslevel=3) as g:
                g.write(f.read())
            raw.unlink()
        return result
    finally:
        e.close()


def train_one(spec):
    import torch
    from stable_baselines3 import PPO
    from slice_alloc.src.allocation_policy import ResourcePriorPolicy
    method, init, steps = spec
    torch.set_num_threads(1)
    assert method in ('prior_ppo', 'plain_ppo', 'potential_ppo')
    env = AllocationEnv(seed=init+100, potential=method=='potential_ppo')
    start = time.perf_counter()
    model = PPO(ResourcePriorPolicy, env, learning_rate=1e-4, n_steps=512,
                batch_size=128, n_epochs=5, gamma=1., gae_lambda=1., ent_coef=.005,
                clip_range=.1, seed=init, device='cpu', verbose=0,
                policy_kwargs=dict(prior_scale=0. if method=='plain_ppo' else 4.))
    model.learn(total_timesteps=steps)
    path = OUT/'models'/f'{method}-{init}-{steps}.zip'
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    info = dict(method=method, initialization=init, steps=steps, checkpoint=str(path.relative_to(OUT)),
                model_sha256=sha(path), sources=hashes(), seconds=time.perf_counter()-start,
                training_scenarios=env.episode_specs, demonstration_episodes=0,
                reward='native settled SLA successes' + (' + telescoping potential difference' if env.potential else ''),
                prior='analytic current-state price' if method!='plain_ppo' else 'none')
    env.close()
    write(path.with_suffix('.json'), info)
    print(json.dumps({k:v for k,v in info.items() if k not in ('training_scenarios', 'sources')}), flush=True)
    return info


def eval_group(spec):
    method, init, steps, seeds, profiles, loads, trace = spec
    model = None
    if method not in BASELINES:
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        model = PPO.load(OUT/'models'/f'{method}-{init}-{steps}.zip', device='cpu')
    rows = []
    for seed in seeds:
        for pindex, profile in enumerate(profiles):
            for load in loads:
                cfg = scenario(seed+100*pindex, load, profile)
                tag = f'{method}-{init or 0}-{steps}/{profile}-{load}-{seed}'
                prefix = OUT/'native'/tag if trace else None
                result = run_episode(cfg, method, model, prefix)
                row = dict(method=method, initialization=init, steps=steps, seed=seed,
                           profile=profile, load=load, requests=result['requests'],
                           success=result['success'], admitted=result['admitted'],
                           rejections=result['rejections'], by_mode=result['by_mode'],
                           native_seconds=result['native_seconds'])
                if trace:
                    row.update(path=tag, summary=json.loads(Path(str(prefix)+'-summary.json').read_text()))
                    for suffix, key in [('-scenario.json','scenario_sha256'), ('-result.json','result_sha256'),
                                        ('-events.csv.gz','events_sha256'), ('-summary.json','summary_sha256')]:
                        row[key] = sha(str(prefix)+suffix)
                rows.append(row)
    return rows


def parallel_eval(specs, jobs):
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        for group in pool.map(eval_group, specs):
            rows.extend(group)
    return rows


def analyze():
    p=json.loads((OUT/'protocol.json').read_text())
    data=json.loads((OUT/'evaluation.json').read_text())
    assert p['sources']==hashes() and data['protocol_sha256']==sha(OUT/'protocol.json')
    rows=data['rows'];stats={}
    methods=list(BASELINES)+p['methods']
    matrices={}
    for m in methods:
        group=[r for r in rows if r['method']==m]
        inits=sorted({r['initialization'] for r in group},key=lambda i:i or 0)
        matrices[m]=np.array([[sum(r['success'] for r in group if r['initialization']==i and r['seed']==s)/
                              sum(r['requests'] for r in group if r['initialization']==i and r['seed']==s)
                              for s in p['test_seeds']] for i in inits])
        per_init=matrices[m].mean(1)
        mode_stats={}
        for mode in ('ip','id','ndn'):
            n=sum(r['by_mode'][mode]['requests'] for r in group)
            mode_stats[mode]=sum(r['by_mode'][mode]['success'] for r in group)/n
        rejections=collections.Counter()
        for r in group:rejections.update(r['rejections'])
        count=sum(r['requests'] for r in group)
        stats[m]=dict(success_rate=sum(r['success'] for r in group)/count,
                      admission_rate=sum(r['admitted'] for r in group)/count,
                      by_mode=mode_stats,initialization_rates=per_init.tolist(),
                      initialization_std=float(np.std(per_init,ddof=1)) if len(per_init)>1 else None,
                      by_load={str(n):sum(r['success'] for r in group if r['load']==n)/
                               sum(r['requests'] for r in group if r['load']==n) for n in p['loads']},
                      rejections=dict(rejections),requests=count)
    comparisons={};rng=np.random.default_rng(91237)
    for m in p['methods']:
        comparisons[m]={}
        for other in methods:
            if other==m:continue
            a,b=matrices[m],matrices[other]
            differences=a.mean(0)-b.mean(0)
            boot=rng.choice(differences,(10000,len(differences)),replace=True).mean(1)
            nested=[]
            for _ in range(4000):
                ss=rng.integers(0,a.shape[1],a.shape[1])
                aa=rng.integers(0,len(a),len(a));bb=rng.integers(0,len(b),len(b))
                nested.append(a[np.ix_(aa,ss)].mean()-b[np.ix_(bb,ss)].mean())
            comparisons[m][other]=dict(difference=float(differences.mean()),
                paired_scenario_ci95=np.quantile(boot,[.025,.975]).tolist(),
                initialization_and_scenario_ci95=np.quantile(nested,[.025,.975]).tolist())
    write(OUT/'analysis.json',dict(episodes=len(rows),decisions=sum(r['requests'] for r in rows),
          stats=stats,comparisons=comparisons,evaluation_sha256=sha(OUT/'evaluation.json'),
          protocol_sha256=sha(OUT/'protocol.json'),scope='mixed primary profile only; stress and lower-load results are separate'))
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


def verify():
    from slice_alloc.tools import patrol_verify as auditor
    p=json.loads((OUT/'protocol.json').read_text());assert p['sources']==hashes()
    assert p['binary_sha256']==sha(NS3/'build/scratch/patrol-scenario')
    data=json.loads((OUT/'evaluation.json').read_text())
    assert data['protocol_sha256']==sha(OUT/'protocol.json') and data['sources']==hashes()
    assert json.loads((OUT/'analysis.json').read_text())['evaluation_sha256']==sha(OUT/'evaluation.json')
    auditor.OUT=OUT
    # Reuse the independent raw-event reconstruction, not policy reward values.
    reconstructed=[auditor.check_episode(r) for r in data['rows']]
    grouped=collections.defaultdict(set)
    for r in data['rows']:grouped[r['seed'],r['load'],r['profile']].add(r['scenario_sha256'])
    assert all(len(v)==1 for v in grouped.values())
    replays=[]
    from stable_baselines3 import PPO
    for c in p['checkpoints']:
        assert sha(OUT/c['checkpoint'])==c['model_sha256']
        row=next(r for r in data['rows'] if r['method']==c['method'] and r['initialization']==c['initialization'])
        prefix=OUT/'native'/row['path']
        original=json.loads(Path(str(prefix)+'-result.json').read_text())
        cfg=json.loads(Path(str(prefix)+'-scenario.json').read_text())
        model=PPO.load(OUT/c['checkpoint'],device='cpu')
        with tempfile.TemporaryDirectory(prefix='allocation-replay-') as tmp:
            new=Path(tmp)/'episode';again=run_episode(cfg,c['method'],model,new)
            assert again['decisions']==original['decisions']
            assert again['observations']==original['observations'] and again['settled']==original['settled']
            with gzip.open(str(prefix)+'-events.csv.gz','rb') as a,gzip.open(str(new)+'-events.csv.gz','rb') as b:
                assert a.read()==b.read()
        replays.append(dict(method=c['method'],initialization=c['initialization'],exact_events=True))
    write(OUT/'raw_reconstruction.json',reconstructed)
    result=dict(passed=True,episodes=len(reconstructed),decisions=sum(r['requests'] for r in data['rows']),
                sources=hashes(),same_inputs=True,raw_packet_sla=True,all_reservations_rechecked=True,
                exact_native_policy_replays=replays,
                artifacts={n:sha(OUT/n) for n in ('protocol.json','evaluation.json','analysis.json','raw_reconstruction.json')})
    write(OUT/'audit.json',result)
    print('raw SLA, contracts, fingerprints and exact native replays passed',flush=True)


def report():
    p=json.loads((OUT/'protocol.json').read_text());a=json.loads((OUT/'analysis.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text());assert audit['passed'] and audit['sources']==hashes()
    assert audit['artifacts']['analysis.json']==sha(OUT/'analysis.json')
    lines=['# 第一部分：低空巡检三模态联合初始分配（v3）','',
           '研究IP、受限EID/RLOC与原生NDN在共享空地基础设施上的接纳及功能映射。物理容量、协议、SLA未调整；本轮主场景为64/80个申请在2—18秒内到达。其他负载的开发扫描全部保留，不能与旧四场景合计成绩直接比较。','',
           '地面平台、双边缘、双AP和12架无人机保持不变。真实移动触发无线关联和定位更新。第一部分只决定初始部署；资源预测和运行期QoS尚未实现，未分配预期成绩。','',
           '## 算法与预算','',
           f'每个学习模型使用{p["train_steps"]}次原生交互，无示范预训练。资源先验PPO以公开的当前资源价格为初始动作偏好，学习可修正的残差；普通PPO去除先验，规则price就是无学习对照。已接纳合同的释放日历是已知预约，不是未来预测。',
           '潜势奖励消融使用Φ(s_next)−Φ(s)，γ=1且终止势为0，整局奖励总和保持真实成功数。评价完全使用原始包级SLA，拒绝计失败。','',
           f'冻结测试种子{p["test_seeds"][0]}—{p["test_seeds"][-1]}；正式执行{a["episodes"]}次、{a["decisions"]}次申请决策。配置仅在7000—7007开发集选择，最终测试只执行一次。','',
           '| 方法 | 总成功率 | 64请求 | 80请求 | 接纳率 | 初始化标准差/百分点 |',
           '|---|---:|---:|---:|---:|---:|']
    for m,s in a['stats'].items():
        std='—' if s['initialization_std'] is None else f'{100*s["initialization_std"]:.3f}'
        lines.append(f'| {m} | {100*s["success_rate"]:.2f}% | {100*s["by_load"]["64"]:.2f}% | {100*s["by_load"]["80"]:.2f}% | {100*s["admission_rate"]:.2f}% | {std} |')
    lines+=['','## 学习相对强规则的收益','',
            '同时报告配对场景区间，以及同时重采样训练初始化和场景的区间；不能把固定模型下的窄区间当作训练稳定性的证明。','',
            '| 学习方法−对照 | 差值/百分点 | 场景95%区间 | 初始化＋场景95%区间 |',
            '|---|---:|---|---|']
    for m,comparisons in a['comparisons'].items():
        for other in ('price','cache_aware','lifetime_price','plain_ppo'):
            if other not in comparisons:continue
            c=comparisons[other];lo,hi=c['paired_scenario_ci95'];l2,h2=c['initialization_and_scenario_ci95']
            lines.append(f'| {m} − {other} | {100*c["difference"]:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | [{100*l2:+.2f}, {100*h2:+.2f}] |')
    lines+=['','## 结论与限制','',
            '75%—82%是研发目标，不是筛选测试样本的规则。是否优于传统方法以本表和配对区间为准，不能仅凭进入目标区间宣称学习有收益。',
            '该实验采用固定高度、受限传播、有限边缘候选与共享队列，不证明一般低空信道下的硬QoS。后两项研究必须继续使用同一输入、资源和SLA做累计消融，不能靠修改分母或增加物理容量凑出递增曲线。','',
            '[冻结协议](protocol.json) · [实际结果](analysis.json) · [逐包核验](audit.json) · [负载扫描](load_calibration.json)']
    text='\n'.join(lines)+'\n';(OUT/'report.md').write_text(text)
    from slice_alloc.tools.export_document import to_docx
    to_docx(text).save(OUT/'report.docx')
    write(OUT/'delivery.json',dict(passed=True,sources=hashes(),artifacts={n:sha(OUT/n) for n in ('protocol.json','analysis.json','audit.json','report.md','report.docx')},scope='first research part only; no prediction or QoS success claims'))
    print(OUT/'report.md',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train','dev','freeze','evaluate','analyze','verify','report'])
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--methods', nargs='+', default=['prior_ppo','plain_ppo','potential_ppo'])
    parser.add_argument('--inits', nargs='+', type=int, default=list(INITS[:2]))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.command == 'train':
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            infos = list(pool.map(train_one, [(m,s,args.steps) for m in args.methods for s in args.inits]))
        write(OUT/f'training-{args.steps}.json', infos)
    elif args.command == 'dev':
        specs = [(m,i,args.steps,DEV_SEEDS,('mixed',),LOADS,False) for m in args.methods for i in args.inits]
        specs += [(m,None,0,DEV_SEEDS,('mixed',),LOADS,False) for m in BASELINES]
        rows = parallel_eval(specs, args.jobs)
        write(OUT/f'development-{args.steps}.json', rows)
        for m in BASELINES+tuple(args.methods):
            r=[x for x in rows if x['method']==m]
            print(m, sum(x['success'] for x in r)/sum(x['requests'] for x in r), flush=True)
    elif args.command == 'freeze':
        checkpoints=[]
        for m in args.methods:
            for i in args.inits:
                path=OUT/'models'/f'{m}-{i}-{args.steps}.json'
                d=json.loads(path.read_text());assert d['sources']==hashes()
                checkpoints.append(dict(method=m, initialization=i, steps=args.steps,
                                        checkpoint=d['checkpoint'], model_sha256=d['model_sha256']))
        protocol=dict(version='allocation-v3', primary_profile='mixed', loads=LOADS,
                      test_seeds=TEST_SEEDS, development_seeds=DEV_SEEDS, initializations=args.inits,
                      methods=args.methods, baselines=BASELINES, train_steps=args.steps,
                      budget='No demonstrations; each model gets the stated native interaction steps.',
                      reward='native actual SLA success count; potential variant separately labelled',
                      topology='unchanged patrol-v2, 12 UAVs, 2 APs, 2 edges, shared radio',
                      capacities='unchanged patrol-v2; all methods face identical inputs and quotas',
                      selection='configuration chosen on development only; final checkpoint at fixed budget',
                      primary_metric='successful SLA requests / all arriving requests, including rejected',
                      target='75-82% is an aspiration, never a reason to exclude a test case',
                      later_work='resource prediction and run-time QoS not implemented or assigned scores',
                      checkpoints=checkpoints, sources=hashes(),
                      binary_sha256=sha(NS3/'build/scratch/patrol-scenario'),
                      calibration_sha256=sha(OUT/'load_calibration.json'),
                      development_sha256=sha(OUT/f'development-{args.steps}.json'),
                      frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        assert not (OUT/'evaluation.json').exists(), 'Never overwrite a previously opened final test.'
        write(OUT/'protocol.json',protocol)
    elif args.command == 'evaluate':
        p=json.loads((OUT/'protocol.json').read_text());assert p['sources']==hashes()
        assert not (OUT/'evaluation.json').exists(), 'Final test already evaluated.'
        specs=[(m,None,0,tuple(p['test_seeds']),('mixed',),tuple(p['loads']),True) for m in BASELINES]
        specs += [(c['method'],c['initialization'],c['steps'],tuple(p['test_seeds']),('mixed',),tuple(p['loads']),True) for c in p['checkpoints']]
        for c in p['checkpoints']:assert c['model_sha256']==sha(OUT/c['checkpoint'])
        rows=parallel_eval(specs,args.jobs)
        write(OUT/'evaluation.json',dict(protocol_sha256=sha(OUT/'protocol.json'),sources=hashes(),rows=rows))
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
