"""Development ablation of admission exploration, on the same native SLA objective.

This entry point is recorded as an additional training source in every model
manifest and frozen protocol.  No expert demonstrations or alternative labels.
"""
import argparse
import concurrent.futures
import json
import time
from slice_alloc.tools.allocation_study import AllocationEnv, OUT, ROOT, hashes
from slice_alloc.tools.patrol_run import sha, write

CONFIGS={
    'admission_ppo':dict(learning_rate=1e-4,ent_coef=.005,reject_prior=-2.),
    'explore_ppo':dict(learning_rate=3e-4,ent_coef=.02,reject_prior=-2.),
    'normalized_prior_ppo':dict(learning_rate=1e-4,ent_coef=.005,reject_prior=-6.,prior_scale=4.,reward_scale=1/80),
    'normalized_plain_ppo':dict(learning_rate=1e-4,ent_coef=.005,reject_prior=-6.,prior_scale=0.,reward_scale=1/80),
    'normalized_admission_ppo':dict(learning_rate=1e-4,ent_coef=.005,reject_prior=-2.,prior_scale=4.,reward_scale=1/80),
    'paired_ppo':dict(learning_rate=1e-4,ent_coef=.005,reject_prior=-2.,prior_scale=4.,reward_scale=1/80,paired_reference=True),
    'paired_safe_ppo':dict(learning_rate=1e-4,ent_coef=.005,reject_prior=-6.,prior_scale=4.,reward_scale=1/80,paired_reference=True),
}


class TrainingEnv(AllocationEnv):
    def __init__(self,*args,reward_scale=1.,paired_reference=False,**kwargs):
        self.reward_scale=reward_scale
        self.paired_reference=paired_reference
        self.reference_decisions=0;self.reference_episodes=0;self.reference_success=0
        super().__init__(*args,**kwargs)

    def reset(self,*args,**kwargs):
        obs,info=super().reset(*args,**kwargs)
        if self.paired_reference:
            from slice_alloc.tools.allocation_study import run_episode
            result=run_episode(self.cfg,'price')
            self.reference_success=result['success']
            self.reference_decisions+=result['requests'];self.reference_episodes+=1
        return obs,info

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        # Exogenous-scenario reference is action independent.  Only a training
        # terminal label uses it; it is never part of observations or inference.
        if done and self.paired_reference:reward-=self.reference_success
        return obs,reward*self.reward_scale,done,truncated,info


def train(spec):
    import torch
    from stable_baselines3 import PPO
    from slice_alloc.src.allocation_policy import ResourcePriorPolicy
    method,init,steps=spec;cfg=CONFIGS[method];torch.set_num_threads(1)
    env=TrainingEnv(seed=init+100,reward_scale=cfg.get('reward_scale',1.),
                    paired_reference=cfg.get('paired_reference',False));start=time.perf_counter()
    model=PPO(ResourcePriorPolicy,env,learning_rate=cfg['learning_rate'],n_steps=512,
              batch_size=128,n_epochs=5,gamma=1.,gae_lambda=1.,ent_coef=cfg['ent_coef'],
              clip_range=.1,seed=init,device='cpu',verbose=0,
              policy_kwargs=dict(prior_scale=cfg.get('prior_scale',4.),reject_prior=cfg['reject_prior']))
    model.learn(total_timesteps=steps)
    path=OUT/'models'/f'{method}-{init}-{steps}.zip';model.save(path)
    record=dict(method=method,initialization=init,steps=steps,checkpoint=str(path.relative_to(OUT)),
                model_sha256=sha(path),sources=hashes(),
                additional_sources={'slice_alloc/tools/allocation_admission.py':sha(__file__)},
                config=cfg,n_steps=512,batch_size=128,n_epochs=5,gamma=1.,gae_lambda=1.,clip_range=.1,
                seconds=time.perf_counter()-start,training_scenarios=env.episode_specs,
                demonstration_episodes=0,reward='native settled SLA successes',
                reward_scale=cfg.get('reward_scale',1.),
                reference_decisions=env.reference_decisions,reference_episodes=env.reference_episodes,
                reward_reference='terminal minus independent price return on same exogenous scenario' if env.paired_reference else 'none')
    env.close();write(path.with_suffix('.json'),record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('sources','training_scenarios')}),flush=True)


def bind_training_sources():
    """Before final evaluation, bind all actual training entry points and manifests."""
    p=json.loads((OUT/'protocol.json').read_text())
    assert not (OUT/'evaluation.json').exists(), 'Cannot revise an opened final experiment.'
    manifests={};extra={};configs={}
    from stable_baselines3 import PPO
    for c in p['checkpoints']:
        f=(OUT/c['checkpoint']).with_suffix('.json');d=json.loads(f.read_text())
        manifests[str(f.relative_to(OUT))]=sha(f)
        for source,value in d.get('additional_sources',{}).items():
            assert sha(ROOT/source)==value
            extra[source]=value
        model=PPO.load(OUT/c['checkpoint'],device='cpu')
        configs[c['method']]=dict(learning_rate=model.learning_rate,n_steps=model.n_steps,
                batch_size=model.batch_size,n_epochs=model.n_epochs,gamma=model.gamma,
                gae_lambda=model.gae_lambda,ent_coef=model.ent_coef,clip_range=model.clip_range(1.),
                policy_kwargs=model.policy_kwargs,reward_scale=d.get('reward_scale',1.),
                reward_reference=d.get('reward_reference','none'),
                demonstration_episodes=0)
    p['training_manifest_sha256']=manifests;p['additional_training_sources']=extra
    p['design_sha256']=sha(ROOT/'slice_alloc/allocation_design_v3.md')
    p['training_audit_source_sha256']=sha(__file__)
    p['training_configurations']=configs
    p['reward']='Actual settled native SLA success count, optionally scaled by the disclosed fixed positive constant; evaluation always uses unscaled packet SLA.'
    all_records=[json.loads(f.read_text()) for f in (OUT/'models').glob('*32768.json')]
    p['development_training_budget']=dict(models=len(all_records),
            native_decisions=sum(d['steps'] for d in all_records),smoke_test_decisions=512,
            reference_decisions=sum(d.get('reference_decisions',0) for d in all_records),
            scope='All attempted v3 configurations, including those not selected; separate from per-model final budget.')
    p['runtime_versions']=dict(python=__import__('platform').python_version(),
            numpy=__import__('numpy').__version__,torch=__import__('torch').__version__,
            stable_baselines3=__import__('stable_baselines3').__version__,
            gymnasium=__import__('gymnasium').__version__)
    write(OUT/'protocol.json',p)


def check_training_sources():
    p=json.loads((OUT/'protocol.json').read_text())
    assert sha(__file__)==p['training_audit_source_sha256']
    for source,value in p.get('additional_training_sources',{}).items():assert sha(ROOT/source)==value
    for path,value in p['training_manifest_sha256'].items():assert sha(OUT/path)==value
    for c in p['checkpoints']:
        d=json.loads((OUT/c['checkpoint']).with_suffix('.json').read_text())
        assert d['sources']==p['sources'] and d['model_sha256']==c['model_sha256']
        assert d['steps']==p['train_steps'] and d['demonstration_episodes']==0
        assert all(100000<=s['seed']<900000 for s in d['training_scenarios'])
    assert sha(ROOT/'slice_alloc/allocation_design_v3.md')==p['design_sha256']
    write(OUT/'training_audit.json',dict(passed=True,protocol_sha256=sha(OUT/'protocol.json'),
          verifier_sha256=sha(__file__),additional_sources=p['additional_training_sources'],
          manifests=p['training_manifest_sha256'],development_budget=p['development_training_budget']))
    print('Full model budgets, manifests and all additional training sources verified.',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','bind','check'])
    p.add_argument('--methods',nargs='+',default=list(CONFIGS));p.add_argument('--inits',nargs='+',type=int,default=[11,23])
    p.add_argument('--steps',type=int,default=32768);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
    if a.command=='bind':bind_training_sources();return
    if a.command=='check':check_training_sources();return
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(train,[(m,s,a.steps) for m in a.methods for s in a.inits]))


if __name__=='__main__':main()
