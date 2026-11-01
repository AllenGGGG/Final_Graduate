"""Train and audit masked Double QR-DQN, without changing the native environment."""
import argparse
import concurrent.futures
import gzip
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import sb3_contrib

from slice_alloc import patrol
from slice_alloc.initial_allocation import InitialAllocationEnv, make_scenario
from slice_alloc.distributional_allocation import (RuleFreeObservation, NStepDictReplay,
    CandidateQRPolicy, MaskedDoubleQRDQN)
from slice_alloc.tools import initial_run as native

ROOT = patrol.ROOT
OUT = ROOT/'slice_alloc/results/qrdqn_allocation'
REFERENCE = ROOT/'slice_alloc/results/lifecycle_ppo'
INITS = (11, 23, 37, 53, 71)
PILOT_INITS = (107, 109)
PILOT_SEEDS = tuple(range(11000,11008))
VALIDATION = tuple(range(11100,11116))
TEST = tuple(range(11200,11230))
LOADS = (64,80)
RULES = native.RULES
CONFIGS = (dict(name='lr1e4_n3',learning_rate=1e-4,horizon=3),
           dict(name='lr3e4_n3',learning_rate=3e-4,horizon=3),
           dict(name='lr1e4_n8',learning_rate=1e-4,horizon=8))
HYPER = dict(buffer_size=20000,learning_starts=1024,batch_size=128,
    train_freq=4,gradient_steps=1,gamma=1.,target_update_interval=1000,tau=1.,
    exploration_initial_eps=1.,exploration_final_eps=.05,exploration_steps=16384,
    max_grad_norm=10.,policy_kwargs=dict(n_quantiles=64,net_arch=[128]))
SOURCES = native.SOURCES + ('slice_alloc/distributional_allocation.py',
    'slice_alloc/tools/distributional_run.py','slice_alloc/tests/test_distributional_allocation.py',
    'slice_alloc/qrdqn_design.md')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):native.write(path,data)
def read(path):return json.loads(Path(path).read_text())
def hashes():return {name:sha(ROOT/name) for name in SOURCES}


def dependencies():
    from sb3_contrib.qrdqn import qrdqn, policies
    from sb3_contrib.common import utils
    from stable_baselines3.common import buffers, off_policy_algorithm
    return {str(Path(module.__file__).resolve()):sha(module.__file__)
            for module in (qrdqn,policies,utils,buffers,off_policy_algorithm)}


class RangeEnv(InitialAllocationEnv):
    """Select independent training scenarios; physical contracts remain untouched."""
    def __init__(self,seed,seed_start):
        super().__init__(seed=seed)
        self.seed_start=seed_start

    def reset(self,seed=None,options=None):
        if options is None or 'scenario' not in options:
            cfg=make_scenario(int(self.rng.integers(self.seed_start,self.seed_start+1000000)),
                              int(self.rng.choice(LOADS)))
            options=dict(scenario=cfg)
        return super().reset(seed=seed,options=options)


def episode(cfg,method,model=None,prefix=None):
    if method!='qrdqn':return native.episode(cfg,method,model,prefix)
    env=RuleFreeObservation(InitialAllocationEnv());timings=[]
    try:
        obs,_=env.reset(options=dict(scenario=cfg,prefix=prefix,trace=prefix is not None))
        done=False
        while not done:
            start=time.perf_counter();action=int(model.predict(obs,deterministic=True)[0])
            timings.append(time.perf_counter()-start)
            assert obs['mask'][action]>.5
            obs,_,done,_,_=env.step(action)
        result=env.unwrapped.result();result['decision_ms']=1000*float(np.mean(timings))
        if prefix:
            write(str(prefix)+'-result.json',result)
            raw=Path(str(prefix)+'-events.csv')
            with raw.open('rb') as source,gzip.open(str(raw)+'.gz','wb',compresslevel=3) as target:
                shutil.copyfileobj(source,target)
            raw.unlink()
        return result
    finally:env.close()


def validate(model,seeds):
    rows=[]
    for seed in seeds:
        for load in LOADS:
            r=episode(make_scenario(seed,load),'qrdqn',model)
            rows.append(dict(seed=seed,load=load,success=r['success'],requests=r['requests'],admitted=r['admitted']))
    return dict(success_rate=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows),rows=rows)


class FixedCheckpoints(BaseCallback):
    """One uninterrupted learn call: checkpoint validation cannot reset exploration."""
    def __init__(self,folder,budgets,seeds):
        super().__init__();self.folder=folder;self.budgets=budgets;self.seeds=seeds;self.stages=[]

    def _on_step(self):
        if self.num_timesteps not in self.budgets:return True
        self.folder.mkdir(parents=True,exist_ok=True)
        path=self.folder/f'{self.num_timesteps}.zip'
        self.model.save(path)
        score=validate(self.model,self.seeds)
        self.stages.append(dict(step=self.num_timesteps,path=str(path.relative_to(ROOT)),
            sha256=sha(path),validation=score,exploration_rate=self.model.exploration_rate))
        print('CHECKPOINT',self.folder.name,self.num_timesteps,round(score['success_rate'],6),flush=True)
        return True


def train(spec):
    phase,config,init,budgets,seeds=spec
    torch.set_num_threads(1)
    folder=OUT/phase/config['name']/str(init)
    manifest=folder/'training.json'
    if manifest.exists():
        info=read(manifest);assert info['sources']==hashes()
        for stage in info['stages']:assert sha(ROOT/stage['path'])==stage['sha256']
        return info
    env=RuleFreeObservation(RangeEnv(init+100,7000000 if phase=='pilot' else 8000000))
    settings={**HYPER,'learning_starts':min(HYPER['learning_starts'],max(budgets)//4)}
    model=MaskedDoubleQRDQN(CandidateQRPolicy,env,**settings,
        learning_rate=config['learning_rate'],replay_buffer_class=NStepDictReplay,
        replay_buffer_kwargs=dict(horizon=config['horizon'],discount=1.),seed=init,device='cpu',verbose=0)
    before=model.quantile_net.deployment[-1].weight.detach().clone()
    callback=FixedCheckpoints(folder,budgets,seeds);started=time.perf_counter()
    try:
        model.learn(max(budgets),callback=callback)
        assert model.num_timesteps==max(budgets)
        assert len(callback.stages)==len(budgets)
        movement=float(torch.linalg.vector_norm(model.quantile_net.deployment[-1].weight.detach()-before))
        assert movement>0
        info=dict(sources=hashes(),phase=phase,config=config,initialization=init,
            trained_steps=model.num_timesteps,updates=model._n_updates,weight_change=movement,
            seconds=time.perf_counter()-started,stages=callback.stages,
            training_scenarios=env.unwrapped.episode_specs,hyperparameters=settings,
            algorithm='Masked Double QR-DQN',reward='raw native settled SLA successes',
            shaping=False,demonstrations=0,rule_prior=False)
        write(manifest,info)
        print('TRAINED',phase,config['name'],init,'steps',model.num_timesteps,flush=True)
        return info
    finally:env.close()


def select(training):
    records=[]
    for info in training:
        for method,limit in (('qrdqn',max(s['step'] for s in info['stages'])),('qrdqn_equal',65536)):
            eligible=[s for s in info['stages'] if s['step']<=limit]
            best=max(eligible,key=lambda x:(x['validation']['success_rate'],-x['step']))
            path=OUT/'models'/f"{method}-{info['initialization']}.zip"
            path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/best['path'],path)
            model=MaskedDoubleQRDQN.load(path,device='cpu')
            assert model.num_timesteps==best['step']
            assert all(model.policy_kwargs[k]==v for k,v in HYPER['policy_kwargs'].items())
            assert model.policy_kwargs['optimizer_class'] is torch.optim.Adam
            assert model.replay_buffer_kwargs==dict(horizon=info['config']['horizon'],discount=1.)
            assert validate(model,VALIDATION)==best['validation']
            records.append(dict(method=method,initialization=info['initialization'],path=str(path.relative_to(ROOT)),
                sha256=sha(path),selected_step=best['step'],budget_limit=limit,validation=best['validation']))
    return records


def evaluate(spec):
    method,init,seeds,mix,scope,model_record=spec
    torch.set_num_threads(1);native.OUT=OUT
    model=None
    if model_record:
        path=ROOT/model_record['path'];assert sha(path)==model_record['sha256']
        model=(MaskedDoubleQRDQN if method.startswith('qrdqn') else PPO).load(path,device='cpu')
    rows=[]
    for seed in seeds:
        for load in LOADS:
            cfg=make_scenario(seed,load,mix)
            tag=f'{scope}/{method}-{init or 0}/{mix}-{load}-{seed}'
            result=episode(cfg,'qrdqn' if method.startswith('qrdqn') else method,model,OUT/'native'/tag)
            rows.append(native.row_for(cfg,method,result,tag,initialization=init,
                seed=seed,load=load,mix=mix,scope=scope))
    return rows


def analyze(rows):
    methods=('qrdqn','qrdqn_equal','ppo_reference')+RULES
    stats={m:native.aggregate([r for r in rows if r['method']==m]) for m in methods}
    matrices={}
    for method in methods:
        group=[r for r in rows if r['method']==method]
        inits=sorted({r['initialization'] for r in group},key=lambda x:x or 0)
        a=np.array([[sum(r['success'] for r in group if r['seed']==s and r['initialization']==i)/
                     sum(r['requests'] for r in group if r['seed']==s and r['initialization']==i)
                     for s in TEST] for i in inits])
        matrices[method]=a
        stats[method]['initialization_rates']=a.mean(1).tolist()
        stats[method]['initialization_std']=float(a.mean(1).std(ddof=1)) if len(inits)>1 else None
    rng=np.random.default_rng(711904);comparisons={}
    for method in methods[1:]:
        a,b=matrices['qrdqn'],matrices[method];boot=[]
        paired = method == 'qrdqn_equal'
        if paired:
            assert sorted({r['initialization'] for r in rows if r['method']=='qrdqn'}) == sorted(
                {r['initialization'] for r in rows if r['method']==method})
        for _ in range(4000):
            scenarios=rng.integers(0,len(TEST),len(TEST))
            ai=rng.integers(0,len(a),len(a));bi=rng.integers(0,len(b),len(b))
            # Budget variants share training trajectories. Keep initialization
            # pairs as well as scenario pairs; consume the historical RNG draw
            # so statistics for independent comparisons remain unchanged.
            if paired:bi=ai
            boot.append(a[np.ix_(ai,scenarios)].mean()-b[np.ix_(bi,scenarios)].mean())
        comparisons[method]=dict(difference=float(a.mean()-b.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
    return dict(stats=stats,comparisons=comparisons,episodes=len(rows),decisions=sum(r['requests'] for r in rows))


def audit_one(row):
    native.OUT=OUT
    return native.raw_check(row)


def publish(analysis,audit,selection,training,smoke=False):
    from slice_alloc.tools.export_document import to_docx
    labels=dict(qrdqn='QR-DQN（131072步预算）',qrdqn_equal='QR-DQN（65536步预算）',
        ppo_reference='上一轮PPO（65536步预算）',near_edge='近边优先',greedy='资源贪心',
        marginal_price='资源价格规则',static_quota='固定配额规则')
    lines=['# 面向低空智联的多模态网络切片技术','',
        '研究内容为智能分配、资源预测、QoS保障。本轮仅测试第一部分的QR-DQN学习路线。','',
        '双AP、双互连边缘、12架无人机（4移动＋8驻留）与IP／受限EID-RLOC／NDN三模态切片合同全部沿用。物理容量、SLA和原生协议执行没有修改。','',
        '方法为动作掩码、Double-Q及多步回报的QR-DQN，使用共享候选编码与集合汇总。不注入价格规则、固定拒绝偏置、模仿数据或奖励塑形。上述组件为既有通用方法，不包装为原创算法。','',
        '## 新测试集结果','',
        '| 方法 | 全到达请求SLA成功率 | 接纳率 | 初始化标准差（百分点） |','|---|---:|---:|---:|']
    for method,s in analysis['stats'].items():
        sd=s['initialization_std'];std='—' if sd is None else f'{sd*100:.3f}'
        lines.append(f"| {labels[method]} | {s['success_rate']*100:.2f}% | {s['admission_rate']*100:.2f}% | {std} |")
    lines+=['','分母为全部到达请求，拒绝和已接纳但SLA失败都计失败。所有方法在相同新测试场景执行；旧PPO成绩重新测量，未复用旧测试结果。','',
        '## 配对差值与边界','']
    for method,c in analysis['comparisons'].items():
        lo,hi=np.array(c['ci95'])*100
        lines.append(f"- QR-DQN相对{labels[method]}：{c['difference']*100:+.2f}个百分点，场景和初始化联合95%区间[{lo:+.2f}, {hi:+.2f}]。")
    lo,hi=analysis['comparisons']['marginal_price']['ci95']
    lines+=['',('本轮支持QR-DQN相对资源价格规则提升。' if lo>0 else
                '本轮QR-DQN显著低于资源价格规则。' if hi<0 else '本轮未证明QR-DQN优于资源价格规则。'),'',
        '长预算与等预算QR-DQN共用训练前缀，不能当成两次独立训练。QR-DQN允许131072步，旧PPO训练65536步，长预算差异不能单独归因于算法；表中保留等步数检查点对照。筛选预算另计。QR-DQN奖励为原生成功数，旧PPO为成功数／80，两者目标相同、数值单位不同。','',
        f"选定配置：`{selection['config']}`。三个预定配置×两个筛选初始化，各32768步；主实验5初始化各131072步。探索按绝对步数固定，模型选择只读独立验证集。训练及选模细节见training.json与pilot_selection.json。",'',
        f"主实验{analysis['episodes']}次、{analysis['decisions']}次决策；共{audit['episodes']}次原始报文SLA审计，{len(audit['replays'])}个所选模型代表性精确重放。源码、依赖实现、原生二进制和模型指纹一致。",'',
        '第一部分只做初始分配。合同到期信息不算需求预测；资源预测和运行期QoS保障尚未实施，不预填其成功率。受限传播模型及平均预约不等于真实部署中的逐包硬保证。本轮仅检验该学习路线，不声称它是所有算法中的最优者。','',
        '[设计](../../qrdqn_design.md) · [协议](protocol.json) · [分析](analysis.json) · [审计](audit.json)']
    report='\n'.join(lines)+'\n';(OUT/'report.md').write_text(report)
    to_docx(report).save(OUT/'report.docx')
    documents={str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in ('report.md','report.docx')}
    if not smoke:
        title='面向低空智联的多模态网络切片技术'
        for n in ('README.md','slice_alloc/design.md',title+'.md',title+'.docx'):
            src=ROOT/n;dst=OUT/'previous_documents'/n
            if src.exists() and not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        design=(ROOT/'slice_alloc/qrdqn_design.md').read_text()
        (ROOT/'slice_alloc/design.md').write_text(design)
        manuscript=report+'\n'+design.replace('# '+title,'## 第一部分实验设计',1)
        (ROOT/(title+'.md')).write_text(manuscript);to_docx(manuscript).save(ROOT/(title+'.docx'))
        (ROOT/'README.md').write_text(f'# {title}\n\n研究内容：智能分配、资源预测、QoS保障。本轮仅测试智能分配的QR-DQN路线，环境及SLA未改。\n\n新测试集QR-DQN成功率：**{analysis["stats"]["qrdqn"]["success_rate"]*100:.2f}%**。是否优于规则及预算差异请见完整报告。\n\n[设计](slice_alloc/design.md) · [实测报告](slice_alloc/results/qrdqn_allocation/report.md) · [报告Word](slice_alloc/results/qrdqn_allocation/report.docx) · [审计](slice_alloc/results/qrdqn_allocation/audit.json)\n\n复现：`python -B -m slice_alloc.tools.distributional_run all --jobs 8`。已完成结果禁止覆盖；历史实验独立保留。\n')
        documents.update({n:sha(ROOT/n) for n in ('README.md','slice_alloc/design.md',title+'.md',title+'.docx')})
    names=('plan.json','protocol.json','pilot_training.json','pilot_selection.json','training.json',
           'evaluation.json','analysis.json','audit.json','raw_audit.json','tests.log',
           'mobility.json','mobility_evaluation.json','competition_evaluation.json','competition.json')
    delivery=dict(passed=True,sources=hashes(),dependencies=dependencies(),documents=documents,
        artifacts={n:sha(OUT/n) for n in names},primary_success=analysis['stats']['qrdqn']['success_rate'],
        formal_episodes=analysis['episodes'],audited_episodes=audit['episodes'],
        main_training_steps=sum(x['trained_steps'] for x in training),
        pilot_training_steps=selection['total_training_steps'],first_part_only=True)
    write(OUT/'delivery.json',delivery)
    if not smoke:write(ROOT/'slice_alloc/results/document_verification.json',dict(
        passed=True,scope='QR-DQN first-part study',delivery_sha256=sha(OUT/'delivery.json'),documents=documents))
    return delivery


def pipeline(jobs,smoke=False):
    torch.set_num_threads(1);native.OUT=OUT;native.LOADS=LOADS
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'delivery.json').exists(),'Completed experiment is immutable'
    pilot_budgets=(256,) if smoke else (16384,32768)
    final_budgets=(256,) if smoke else (32768,65536,131072)
    pilot_inits=(999,) if smoke else PILOT_INITS
    references=[]
    for init in INITS:
        path=REFERENCE/'models'/f'ppo-{init}.zip'
        references.append(dict(method='ppo_reference',initialization=init,
            path=str(path.relative_to(ROOT)),sha256=sha(path),trained_steps=65536))
    mobility_seeds=(10993,) if smoke else (11400,11401,11402)
    mix_seeds=(10994,) if smoke else tuple(range(11300,11306))
    plan=dict(sources=hashes(),dependencies=dependencies(),hyperparameters=HYPER,configs=CONFIGS,
        pilot_budgets=pilot_budgets,final_budgets=final_budgets,pilot_inits=pilot_inits,inits=INITS,
        pilot_seeds=PILOT_SEEDS,validation_seeds=VALIDATION,test_seeds=TEST,loads=LOADS,
        mobility_seeds=mobility_seeds,mix_seeds=mix_seeds,
        pilot_training_range=[7000000,7999999],main_training_range=[8000000,8999999],
        references=references,binary_sha256=sha(patrol.NS3/'build/scratch/patrol-scenario'),
        versions=dict(sb3_contrib=sb3_contrib.__version__,torch=torch.__version__),smoke=smoke)
    if (OUT/'plan.json').exists():assert read(OUT/'plan.json')==json.loads(json.dumps(plan))
    else:write(OUT/'plan.json',plan)
    for name,expected in hashes().items():
        dst=OUT/'frozen_sources'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(ROOT/name,dst)
        assert sha(dst)==expected
    with (OUT/'tests.log').open('w') as stream:
        subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','slice_alloc/tests','-q'],
                       cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    mobility_specs=[(mode,condition,edge,seed) for mode in ('ip','id','ndn')
        for condition in ('stationary','moving_no_cross','cross_ap') for edge in (1,2) for seed in mobility_seeds]
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        mobility_rows=list(pool.map(native.mobility_case,mobility_specs))
    write(OUT/'mobility_evaluation.json',mobility_rows)
    mobility=native.summarize_mobility(mobility_rows);write(OUT/'mobility.json',mobility)
    print('MOBILITY GATE',len(mobility_rows),flush=True)
    specs=[('pilot',cfg,init,pilot_budgets,PILOT_SEEDS) for cfg in CONFIGS for init in pilot_inits]
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:pilots=list(pool.map(train,specs))
    write(OUT/'pilot_training.json',pilots)
    scores={cfg['name']:float(np.mean([max(s['validation']['success_rate'] for s in r['stages'])
                  for r in pilots if r['config']==cfg])) for cfg in CONFIGS}
    chosen=max(CONFIGS,key=lambda cfg:scores[cfg['name']])
    selection=dict(config=chosen,scores=scores,total_training_steps=sum(x['trained_steps'] for x in pilots))
    write(OUT/'pilot_selection.json',selection);print('SELECTED',selection,flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        training=list(pool.map(train,[('training',chosen,init,final_budgets,VALIDATION) for init in INITS]))
    write(OUT/'training.json',training)
    selected=select(training)
    protocol=dict(plan,selection=selection,models=selected+references)
    write(OUT/'protocol.json',protocol)
    specs=[(r['method'],r['initialization'],TEST,'balanced','main',r) for r in selected+references]
    specs += [(method,None,TEST,'balanced','main',None) for method in RULES]
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        rows=[r for group in pool.map(evaluate,specs) for r in group]
    write(OUT/'evaluation.json',dict(sources=hashes(),protocol_sha256=sha(OUT/'protocol.json'),rows=rows))
    analysis=analyze(rows);write(OUT/'analysis.json',analysis)
    print('MAIN RESULTS',{m:round(s['success_rate']*100,4) for m,s in analysis['stats'].items()},flush=True)
    mix_specs=[(method,None,mix_seeds,mix,'competition',None) for mix in
        ('balanced','ip_heavy','id_heavy','ndn_heavy') for method in ('marginal_price','static_quota')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        mix_rows=[r for group in pool.map(evaluate,mix_specs) for r in group]
    write(OUT/'competition_evaluation.json',mix_rows)
    write(OUT/'competition.json',{mix:{method:native.aggregate([r for r in mix_rows if r['mix']==mix and r['method']==method])
        for method in ('marginal_price','static_quota')} for mix in ('balanced','ip_heavy','id_heavy','ndn_heavy')})
    all_rows=rows+mix_rows+mobility_rows
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:raw=list(pool.map(audit_one,all_rows))
    write(OUT/'raw_audit.json',raw)
    inputs={}
    for r in rows+mix_rows:inputs.setdefault((r['scope'],r['mix'],r['seed'],r['load']),set()).add(r['scenario_sha256'])
    assert all(len(x)==1 for x in inputs.values())
    replays=[]
    for record in selected+references:
        original_row=next(r for r in rows if r['method']==record['method'] and r['initialization']==record['initialization'])
        prefix=OUT/'native'/original_row['path'];original=read(str(prefix)+'-result.json')
        cfg=read(str(prefix)+'-scenario.json')
        model=(MaskedDoubleQRDQN if record['method'].startswith('qrdqn') else PPO).load(ROOT/record['path'],device='cpu')
        if record['method'].startswith('qrdqn'):
            assert model.num_timesteps==record['selected_step']
            assert model.replay_buffer_kwargs==dict(horizon=chosen['horizon'],discount=1.)
        with tempfile.TemporaryDirectory(prefix='qrdqn-replay-') as tmp:
            again_prefix=Path(tmp)/'case'
            again=episode(cfg,'qrdqn' if record['method'].startswith('qrdqn') else record['method'],model,again_prefix)
            for key in ('observations','decisions','settled'):assert original[key]==again[key]
            with gzip.open(str(prefix)+'-events.csv.gz','rb') as a,gzip.open(str(again_prefix)+'-events.csv.gz','rb') as b:
                assert a.read()==b.read()
        replays.append(dict(method=record['method'],initialization=record['initialization'],exact=True))
    assert hashes()==plan['sources'] and dependencies()==plan['dependencies']
    audit=dict(passed=True,sources=hashes(),dependencies=dependencies(),episodes=len(all_rows),
        decisions=sum(r['requests'] for r in all_rows),same_inputs=True,replays=replays,
        environment_unchanged=True,raw_sla_reconstructed=True)
    write(OUT/'audit.json',audit)
    delivery=publish(analysis,audit,selection,training,smoke)
    for name,h in delivery['artifacts'].items():assert sha(OUT/name)==h
    for name,h in delivery['documents'].items():assert sha(ROOT/name)==h
    write(OUT/'completion.json',dict(passed=True,delivery_sha256=sha(OUT/'delivery.json'),same_frozen_sources=True))
    print('COMPLETE',delivery['primary_success'],flush=True)


def main():
    global OUT,INITS,PILOT_SEEDS,VALIDATION,TEST
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('smoke','all'));parser.add_argument('--jobs',type=int,default=8)
    args=parser.parse_args()
    if args.command=='smoke':
        OUT.mkdir(parents=True,exist_ok=True)
        original=OUT
        with tempfile.TemporaryDirectory(prefix='smoke-',dir=OUT) as tmp:
            OUT=Path(tmp);INITS=(11,);PILOT_SEEDS=(10990,);VALIDATION=(10991,);TEST=(10992,)
            pipeline(args.jobs,True)
        write(original/'smoke.json',dict(passed=True,sources=hashes(),end_to_end=True))
    else:
        assert read(OUT/'smoke.json')['sources']==hashes(),'Run smoke with the same frozen sources first'
        pipeline(args.jobs)


if __name__=='__main__':main()
