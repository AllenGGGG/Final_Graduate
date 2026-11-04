"""Fresh testing and independent audits for the independent Double-DQN study."""
import argparse,concurrent.futures as cf,gzip,hashlib,json,multiprocessing as mp,re,shutil,os
from pathlib import Path
import numpy as np
import torch
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.tools import allocation_credit_run as run,allocation_simple_report as audit

# Fresh independent test namespace; the paired study has a separate gate.
run.OUT=Path(os.environ.get('LOWALT_MC_OUTPUT',str(run.PKG/'results/monte_carlo'))).resolve()
run.INITS=(11,23)
run.LOADS=(32,64,96)
run.TEST=tuple(range(445000,445060))

RULES=('greedy','nearest','first_fit','native')

def register():
    run.check()
    value=dict(source_sha256=run.sha(__file__),audit_sha256=run.sha(audit.__file__),
        protocol_sha256=run.sha(run.OUT/'protocol.json'),test=list(run.TEST),loads=list(run.LOADS),
        initializations=list(run.INITS),expected_episodes=1440,inference_probes=12,
        primary='independent Monte-Carlo-return Double-DQN versus Greedy',
        controls='nearest, first-fit, native modality rule; matched pretrained independent RL checkpoints',
        bootstrap='20000 paired scene and initialization draws, three loads clustered within scene',
        publication='paired95CI lower>0 versus Greedy, native rule, nearest and first-fit; arrival and admitted violation not worse than Greedy',
        no_rule_initialization=True,no_test_retuning=True)
    p=run.OUT/'evaluation_registration.json'
    if p.exists():assert run.read(p)==value
    else:run.write(p,value)

def model_path(method,init):
    return run.OUT/'models'/f'double_dqn-{init}.zip' if method=='double_dqn' else run.OUT/'reference_models'/f'double_dqn-{init}.zip'

def episode(seed,load,method,init=0,model=None,trace=True):
    cfg=run.envs.base.scenario(seed,load)
    prefix=run.OUT/'native'/f'{method}-{init}'/f'{load}-{seed}' if trace else None
    env=run.envs.CreditEnv(progress_weight=0.)
    try:
        obs,_=env.reset(options=dict(scenario=cfg,prefix=prefix,trace=trace));done=False;reward=0.
        while not done:
            action=int(model.predict(obs,deterministic=True)[0]) if model else (run.envs.native_action(env) if method=='native' else run.old.choose(method,env))
            assert obs['mask'][action]>.5
            obs,r,done,truncated,_=env.step(action);assert not truncated;reward+=r
        result=env.result();assert reward==result['success']
        scenario_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        action_hash=hashlib.sha256(json.dumps(result['decisions'],sort_keys=True).encode()).hexdigest()
        result.update(scenario_sha256=scenario_hash,action_sha256=action_hash)
        row=dict(method=method,init=init,seed=seed,load=load,action_sha256=action_hash,
            **{k:result[k] for k in ('requests','success','admitted','by_service')})
        if not trace:return row
        run.write(str(prefix)+'-result.json',result)
        raw=Path(str(prefix)+'-events.csv')
        with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
        raw.unlink();run.write(str(prefix)+'-metrics.json',dict(**row,scenario_sha256=scenario_hash))
        return audit.verify_episode(run.OUT,row)
    finally:env.close()

def group(spec):
    method,init=spec;torch.set_num_threads(1)
    model=MaskedDoubleDQN.load(model_path(method,init),device='cpu') if method not in RULES else None
    return [episode(s,l,method,init,model) for s in run.TEST for l in run.LOADS]

def evaluate(workers):
    register();models=run.read(run.OUT/'training_complete.json')
    models=models['models'] if isinstance(models,dict) else models
    assert {r['init'] for r in models}==set(run.INITS)
    for r in models:assert run.sha(run.OUT/r['model'])==r['model_sha256']
    log=(run.OUT/'full-tests.log').read_text();assert '\nOK\n' in log
    gate=run.OUT/'test_started.json';assert not gate.exists(),'test already opened; no post-test tuning'
    snapshot={n:run.sha(run.OUT/n) for n in ('protocol.json','training_complete.json','full-tests.log','evaluation_registration.json')}
    run.write(gate,snapshot)
    specs=[(m,i) for m in ('double_dqn','initial_rl') for i in run.INITS]+[(m,0) for m in RULES]
    rows=[]
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for group_rows in pool.map(group,specs):
            rows.extend(group_rows);print('AUDITED',len(rows),1440,flush=True)
    run.write(run.OUT/'evaluation.json',rows)

def audit_row(row):
    core={k:v for k,v in row.items() if k not in ('forced_reject','voluntary_reject','admitted_violations','modes','scenario_sha256')}
    verified=audit.verify_episode(run.OUT,core)
    assert verified==row
    prefix=run.OUT/'native'/f"{row['method']}-{row['init']}"/f"{row['load']}-{row['seed']}"
    r=run.read(str(prefix)+'-result.json')
    assert hashlib.sha256(json.dumps(r['decisions'],sort_keys=True).encode()).hexdigest()==row['action_sha256']
    return verified

def probe(init):
    torch.set_num_threads(1);model=MaskedDoubleDQN.load(model_path('double_dqn',init),device='cpu')
    return [episode(s,l,'double_dqn',init,model,False) for s in (run.TEST[0],run.TEST[-1]) for l in run.LOADS]

def analyze(rows):
    lookup={(r['method'],r['init'],r['seed'],r['load']):r for r in rows}
    rng=np.random.default_rng(20261106);sd=rng.integers(len(run.TEST),size=(20000,len(run.TEST)));ids=rng.integers(len(run.INITS),size=(20000,len(run.INITS)))
    a=dict(methods={},comparisons={},per_initialization={},per_load={});boots={}
    for m in ('double_dqn','initial_rl',*RULES):
        inits=(0,) if m in RULES else run.INITS
        matrix=np.array([[sum(lookup[m,i,s,l]['success'] for l in run.LOADS)/sum(run.LOADS) for s in run.TEST] for i in inits])
        boots[m]=matrix[0,sd].mean(1) if m in RULES else matrix[ids[:,:,None],sd[:,None,:]].mean((1,2))
        rr=[r for r in rows if r['method']==m];v=audit.summarize(rr)
        v['violation_rate_admitted']=v['admitted_violations']/v['admitted'] if v['admitted'] else 0.
        v['ci95']=np.quantile(boots[m],[.025,.975]).tolist()
        v['minimum_service_success']=min(x['success']/x['requests'] for x in v['by_service'].values())
        a['methods'][m]=v
        a['per_load'][m]={str(l):audit.summarize([r for r in rr if r['load']==l]) for l in run.LOADS}
        if m not in RULES:a['per_initialization'][m]={str(i):audit.summarize([r for r in rr if r['init']==i]) for i in run.INITS}
    for m in ('initial_rl',*RULES):
        point=a['methods']['double_dqn']['success_rate']-a['methods'][m]['success_rate'];d=boots['double_dqn']-boots[m]
        a['comparisons']['rl-'+m]=dict(difference=point,ci95=np.quantile(d,[.025,.975]).tolist(),centered_bootstrap_p=float((1+np.count_nonzero(np.abs(d-point)>=abs(point)))/(len(d)+1)))
    secondary=[(k,v) for k,v in a['comparisons'].items() if k!='rl-greedy'];running=0.
    for rank,(k,v) in enumerate(sorted(secondary,key=lambda kv:kv[1]['centered_bootstrap_p'])):
        running=max(running,min(1.,(len(secondary)-rank)*v['centered_bootstrap_p']));v['holm_adjusted_p']=running
    r=a['methods']['double_dqn'];g=a['methods']['greedy']
    a['publication_gates']={m+'_ci_positive':a['comparisons']['rl-'+m]['ci95'][0]>0 for m in RULES}
    a['publication_gates'].update(violation_all_not_worse=r['violation_rate_all']<=g['violation_rate_all'],violation_admitted_not_worse=r['violation_rate_admitted']<=g['violation_rate_admitted'])
    a['promoted']=all(a['publication_gates'].values())
    return a

def verify(workers):
    register()
    for n,h in run.read(run.OUT/'test_started.json').items():assert run.sha(run.OUT/n)==h
    training=run.read(run.OUT/'training_complete.json');training=training['models'] if isinstance(training,dict) else training
    assert {r['init'] for r in training}==set(run.INITS)
    actual=[];torch.set_num_threads(1)
    for r in training:
        assert r['selected']==max(r['checkpoints'],key=lambda x:(x['validation']['rate'],-x['step']))
        for c in r['checkpoints']:assert run.sha(run.OUT/c['path'])==c['sha256']
        assert run.sha(run.OUT/r['model'])==r['model_sha256']
        model=MaskedDoubleDQN.load(run.OUT/r['model'],device='cpu')
        assert model.n_quantiles==1 and model.gamma==1. and model.learning_rate==5e-5
        assert model.replay_buffer_kwargs['horizon']==8 and model.replay_buffer_kwargs['discount']==1.
        initial=MaskedDoubleDQN.load(run.OUT/'reference_models'/f"double_dqn-{r['init']}.zip",device='cpu')
        if r['selected']['step']==0:
            for n,w in model.policy.state_dict().items():torch.testing.assert_close(w,initial.policy.state_dict()[n],atol=0,rtol=0)
        else:assert model.num_timesteps==r['selected']['step']
        assert r['steps']==32768 and r['updates']>0
        assert r['returns_committed']>0
        actual.append(dict(init=r['init'],additional_selected_steps=r['selected']['step'],pretrained_steps=initial.num_timesteps,
            actual_model_steps=model.num_timesteps,horizon=model.replay_buffer_kwargs['horizon'],sha256=r['model_sha256']))
    rows=run.read(run.OUT/'evaluation.json')
    expected={(m,i,s,l) for m in ('double_dqn','initial_rl',*RULES) for i in ((0,) if m in RULES else run.INITS) for s in run.TEST for l in run.LOADS}
    assert len(rows)==len(expected)==1440 and {(r['method'],r['init'],r['seed'],r['load']) for r in rows}==expected
    with cf.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for n,_ in enumerate(pool.map(audit_row,rows),1):
            if n%180==0:print('VERIFIED',n,len(rows),flush=True)
        probes=[r for group in pool.map(probe,run.INITS) for r in group]
    lookup={(r['method'],r['init'],r['seed'],r['load']):r for r in rows}
    for p in probes:
        saved=lookup[p['method'],p['init'],p['seed'],p['load']]
        for k in ('action_sha256','success','admitted','by_service'):assert p[k]==saved[k]
    for s in run.TEST:
        for l in run.LOADS:assert len({r['scenario_sha256'] for r in rows if r['seed']==s and r['load']==l})==1
    a=analyze(rows);run.write(run.OUT/'analysis.json',a)
    v=dict(verified_native_episodes=1440,inference_probes=len(probes),actual_models=actual,
        all_packet_audits_pass=True,all_direction_audits_pass=True,all_reservation_replays_pass=True,
        all_methods_share_scenarios=True,all_initializations_reported=True,new_independent_test=True,
        passed_tests=int(re.search(r'Ran (\d+) tests',(run.OUT/'full-tests.log').read_text())[1]),publication_gates=a['publication_gates'])
    run.write(run.OUT/'verification.json',v)
    collection=run.read(run.OUT/'protocol.json').get('collection','ordinary masked epsilon-greedy exploration')
    lines=['# 第一部分：独立Double-DQN最终验证','',
        '物理场景、业务合同、三种模态和最多11个合法动作保持不变。32/64/96三负载，测试445000—445059，两个独立初始化全部保留。主方法仍为独立学习的Masked Double-DQN，没有价格、规则起点、专家轨迹、预测器、资源档位或对偶约束。','',
        '从已训练的独立RL检查点进行一次有界续训；唯一改动是把每个完整回合的真实终局成功数形成精确return-to-go，写入该回合历史转移。每个初始化按60个验证场景选择最高检查点，测试集只在冻结后打开。','',
        '实际训练交互方式：'+collection+'。Monte-Carlo回报只使用回合结束时已经观测到的真实结果，不把未来信息输入决策状态，也没有注入任何规则动作。','',
        '| 方法 | SLA成功率 | 接纳率 | 违约／全部到达 | 违约／已接纳 |','|---|---:|---:|---:|---:|']
    for name,r in a['methods'].items():lines.append(f"| {name} | {r['success_rate']:.3%} | {r['admission_rate']:.3%} | {r['violation_rate_all']:.3%} | {r['violation_rate_admitted']:.3%} |")
    for name,r in a['comparisons'].items():lines+=['',f"{name}：{r['difference']*100:+.3f}个百分点，配对95%区间[{r['ci95'][0]*100:+.3f},{r['ci95'][1]*100:+.3f}]。"]
    lines+=['',f"全部发布条件通过：{a['promoted']}。1440回合逐包、方向、预约审计，以及12个模型重载重放全部通过。",'',
        '成功率分母为全部到达，拒绝计失败。势函数只读取已观测的活跃业务按时交付计数，初末值为0，gamma=1，整局奖励恒等于实际SLA成功数。场景与初始化双重bootstrap，三负载一起重采样。次要比较的探索性近似p值与Holm校正、分业务／负载／初始化结果见analysis.json。','',
        '“切片”为准入预约与配置，不声称逐切片带宽保证。简化EID-RLOC不等同完整LISP；固定高度小规模平台不作空地信道创新。成功率优势仅限本协议列出的规则，不把未比较或曾观察过的其他规则说成已被超过。']
    (run.OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('FINAL',a['promoted'],a['comparisons'],a['publication_gates'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('register','evaluate','verify'));p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    if args.command=='register':register()
    elif args.command=='evaluate':evaluate(args.workers)
    else:verify(args.workers)
if __name__=='__main__':main()
