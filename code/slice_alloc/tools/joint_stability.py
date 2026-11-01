"""Expand unchanged DDQN/PPO protocols from three to eight initializations.

The original six checkpoints and 240 packet runs are immutable inputs.
Additional models use identical interaction budgets and validation selection.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import platform
import numpy as np
import torch
from stable_baselines3 import PPO
from ..joint import JointEnv,write_tasks
from ..src.qlearning import DoubleDQN
from ..src.policy import CandidatePolicy
from .joint_run import (OUT,PROTOCOL,profile,validation,ValidationCallback,episode,native_case,
                         write,source_hashes)
from .joint_analyze import failure_counts
from .platform_verify import sha

INITIALIZATIONS=(11,22,33,44,55,66,77,88)


def metadata():
    return dict(initializations=INITIALIZATIONS,source_sha256=sha(__file__),frozen_source_sha256=source_hashes(),
        calibration_sha256=sha(OUT/'calibration.json'),original_training_sha256=sha(OUT/'training.json'),
        comparison_sha256=sha(OUT/'comparison.json'),environment=dict(conda='sdn',python=platform.python_version(),torch=torch.__version__,device='cpu'),
        scope='unchanged algorithm and validation protocol; eight initializations on the same held-out request matrix, not new independent scenario seeds')


def train():
    torch.set_num_threads(1);prof=profile();original=json.loads((OUT/'training.json').read_text())
    path=OUT/'stability.json';expected=metadata()
    if path.exists():
        saved=json.loads(path.read_text())
        for key,value in expected.items():
            if key=='initializations':value=list(value)
            if saved[key]!=value:raise ValueError('stale stability run: '+key)
        models=saved['models']
    else:models=list(original['models'])
    for algorithm in ('ddqn','ppo'):
        for initialization in INITIALIZATIONS:
            existing=[r for r in models if r['algorithm']==algorithm and r['initialization']==initialization]
            if existing:
                if sha(OUT/existing[0]['file'])!=existing[0]['sha256']:raise ValueError('changed checkpoint')
                continue
            target=OUT/'models'/f'{algorithm}_{initialization}'
            env=JointEnv(prof,training=True,count=PROTOCOL['requests'],training_loads=PROTOCOL['loads'])
            if algorithm=='ddqn':
                agent=DoubleDQN(env.state_dim,env.candidate_dim,seed=initialization)
                obs,_=env.reset(seed=initialization);history=[];best=-np.inf
                for step in range(1,PROTOCOL['steps_per_model']+1):
                    epsilon=max(.05,1.-step/(PROTOCOL['steps_per_model']*.7))
                    action=agent.explore(obs,epsilon);nxt,reward,done,_,_=env.step(action)
                    agent.remember(obs,action,reward,nxt,done)
                    if step>=1024 and step%4==0:agent.update()
                    obs=env.reset()[0] if done else nxt
                    if step%PROTOCOL['checkpoint_interval']==0:
                        result=validation(agent);history.append(dict(steps=step,utility=result['utility']))
                        if result['utility']>best:best=result['utility'];agent.save(target.with_suffix('.pt'))
                        print(f'DDQN {initialization} steps={step} validation={result["utility"]:.3f}',flush=True)
                model_path=target.with_suffix('.pt')
            else:
                agent=PPO(CandidatePolicy,env,seed=initialization,device='cpu',verbose=0,n_steps=512,batch_size=128,
                    n_epochs=6,gamma=1.,learning_rate=3e-4,ent_coef=.01)
                callback=ValidationCallback(target);agent.learn(PROTOCOL['steps_per_model'],callback=callback)
                history=callback.history;best=callback.best;model_path=target.with_suffix('.zip')
            models.append(dict(algorithm=algorithm,initialization=initialization,file=str(model_path.relative_to(OUT)),
                sha256=sha(model_path),steps=PROTOCOL['steps_per_model'],validation_utility=best,history=history))
            write(path,dict(**expected,models=models,new_training_steps=sum(m['steps'] for m in models if m['initialization'] not in (11,22,33)),
                total_training_steps=sum(m['steps'] for m in models)))


def evaluate():
    torch.set_num_threads(1);prof=profile();training=json.loads((OUT/'stability.json').read_text())
    for key,value in metadata().items():
        if key=='initializations':value=list(value)
        if training[key]!=value:raise ValueError('stale extended training: '+key)
    if len(training['models'])!=16:raise ValueError('eight initializations per algorithm are required')
    data=json.loads((OUT/'comparison.json').read_text());rows=list(data['episodes']);packets=list(data['packet_cases']);tasks=[]
    for record in training['models']:
        if record['initialization'] in (11,22,33):continue
        path=OUT/record['file']
        if sha(path)!=record['sha256']:raise ValueError('changed extended checkpoint')
        agent=DoubleDQN.load(path) if record['algorithm']=='ddqn' else PPO.load(path,device='cpu')
        label=f'{record["algorithm"]}_{record["initialization"]}'
        for seed in PROTOCOL['test_seeds']:
            for load in PROTOCOL['loads']:
                env=JointEnv(prof,count=PROTOCOL['requests'],load=load)
                row,plans=episode(env,record['algorithm'],seed,model=agent,trace=True);row['method']=label
                name=f'test_{seed}_{load:g}_{label}';path=OUT/'inputs'/f'{name}.tsv';write_tasks(path,plans)
                row['case']=name;rows.append(row)
                tasks.append((name,path,seed+PROTOCOL['test_wifi_seed_offset'],0 if seed%2==0 else 5.))
    with ThreadPoolExecutor(4) as pool:new_packets=list(pool.map(native_case,tasks))
    packets.extend(new_packets);by_name={p['name']:p for p in new_packets}
    for row in rows:
        if row['case'] not in by_name:continue
        p=by_name[row['case']]
        row.update(packet_success=p['strict_slices']/row['requests'],admitted_packet_success=p['strict_slices']/max(row['accepted'],1),
            on_time_fraction=p['on_time']/max(p['expected'],1),packet_input_sha256=p['input_sha256'])
    summary={};rng=np.random.default_rng(91173);paired={}
    for algorithm in ('ddqn','ppo'):
        per_initialization=[]
        for init in INITIALIZATIONS:
            group=[r for r in rows if r['method']==f'{algorithm}_{init}']
            per_initialization.append(dict(initialization=init,**failure_counts(group)))
        rates=[r['success_rate'] for r in per_initialization]
        summary[algorithm]=dict(per_initialization=per_initialization,mean=float(np.mean(rates)),median=float(np.median(rates)),
            minimum=min(rates),maximum=max(rates),sample_std=float(np.std(rates,ddof=1)))
        for baseline in ('greedy','resource_price','adaptive_price'):
            differences=[]
            for seed in PROTOCOL['test_seeds']:
                treatment=[r['packet_success'] for r in rows if r['seed']==seed and r['method'].startswith(algorithm+'_')]
                base=[r['packet_success'] for r in rows if r['seed']==seed and r['method']==baseline]
                differences.append(float(np.mean(treatment)-np.mean(base)))
            boot=np.mean(rng.choice(differences,(10000,len(differences)),replace=True),axis=1)
            paired[algorithm+'-'+baseline]=dict(mean=float(np.mean(differences)),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),seeds=len(differences))
    proof=dict(**metadata(),training_sha256=sha(OUT/'stability.json'),episodes=rows,packet_cases=packets,
        new_native_runs=len(new_packets),total_native_comparison_runs=len(packets),summary=summary,paired=paired,
        statistical_scope='scenario-seed paired intervals; eight policy initializations averaged within seed, not independent extra scenarios')
    write(OUT/'stability_comparison.json',proof)
    print(json.dumps(dict(summary=summary,paired=paired,new_packet_runs=len(new_packets)),ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('train','evaluate','all'),default='all',nargs='?');phase=parser.parse_args().phase
    if phase=='all':train();evaluate()
    else:globals()[phase]()


if __name__=='__main__':main()
