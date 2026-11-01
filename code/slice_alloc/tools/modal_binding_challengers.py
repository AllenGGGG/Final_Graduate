"""Two additional strong semantic rules, preregistered before any formal test."""
import argparse
import concurrent.futures as cf
import math
from pathlib import Path
from slice_alloc import modal_binding as b
from slice_alloc.tools import modal_binding_run as run

RULES=('ndn_only','semantic_rule')

class Rule:
    def __init__(self,env,name,threshold):self.env=env;self.name=name;self.threshold=threshold
    def predict(self,obs,deterministic=True):
        env=self.env;r=env.requests[env.current['request_id']]
        if self.name=='ndn_only':modes=('ndn',)
        elif r.reusable:modes=('ndn',)
        elif any(env.cfg['public']['missions'][str(n)]['role']=='patrol' for n in r.members):modes=('id','ndn')
        else:modes=('ip',)
        indices=[i for i,p in enumerate(env.options) if p.request.mode in modes]
        if r.reusable:
            cached=[i for i in indices if env.options[i].cache]
            if cached:indices=cached
        if not indices:return 0,None
        costs={i:b.exponential_price(env.ledger,env.options[i]) for i in indices}
        best=min(indices,key=lambda i:costs[i]);limit=math.inf if self.threshold is None else self.threshold
        return (best+1 if costs[best]<=limit else 0),None


def episode(cfg,name,threshold,prefix=None):
    # Reuse the exact native evaluator and audit with a proxy that receives its env.
    original=b.BindingEnv
    holder={}
    class Captured(original):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);holder['env']=self
    class Proxy:
        def predict(self,obs,deterministic=True):return Rule(holder['env'],name,threshold).predict(obs)
    b.BindingEnv=Captured
    try:return run.episode(cfg,name,Proxy(),prefix)
    finally:b.BindingEnv=original


def tune(name):
    grid=[]
    for threshold in run.THRESHOLDS:
        results=[episode(b.scenario(s,l),name,threshold) for s in run.VALIDATION for l in run.LOADS]
        grid.append(dict(threshold=threshold,rate=sum(r['success'] for r in results)/sum(r['requests'] for r in results)))
    chosen=max(grid,key=lambda r:(r['rate'],math.inf if r['threshold'] is None else r['threshold']))
    value=dict(method=name,chosen=chosen,grid=grid);run.write(run.OUT/'rules'/f'{name}.json',value)
    print('CHALLENGER',name,chosen,flush=True)


def evaluate(name):
    threshold=run.read(run.OUT/'rules'/f'{name}.json')['chosen']['threshold'];rows=[]
    cases=[('test','balanced',load,run.TEST) for load in run.LOADS]+[
        ('ood',mix,load,run.OOD) for mix,load in [('object_heavy',80),('mobile_heavy',80),('balanced',112),('novel',80)]]
    for scope,mix,load,seeds in cases:
        for seed in seeds:
            prefix=run.OUT/'native'/f'{name}-0'/f'{scope}-{mix}-{load}-{seed}'
            path=Path(str(prefix)+'-metrics.json')
            r=run.read(path) if path.exists() else episode(b.scenario(seed,load,mix),name,threshold,prefix)
            rows.append({k:v for k,v in r.items() if k not in ('decisions','settled','observations')})
            rows[-1].update(method=name,init=0,scope=scope,mix=mix,load=load,seed=seed)
    run.write(run.OUT/f'{name}-evaluation.json',rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','tune','evaluate']);args=p.parse_args()
    if args.command=='freeze':
        assert not (run.OUT/'selection.json').exists(),'Formal selection/test phase already started'
        run.write(run.OUT/'challengers-protocol.json',dict(rules=RULES,thresholds=run.THRESHOLDS,
            validation=run.VALIDATION,test=run.TEST,ood=run.OOD,source_sha256=run.sha(__file__),
            reason='Check whether all-NDN or a simple semantic/mobile rule accounts for the gain; no changes to learned algorithms or scenarios.',
            policy='Prefer leased NDN for reusable objects; for nonreusable mobile endpoints choose ID/NDN by the same exponential cost, stationary endpoints use IP. All-NDN prefers leases for reusable objects. Same validation grid, no test tuning.'))
    if args.command=='tune':
        with cf.ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(tune,RULES))
    if args.command=='evaluate':
        assert run.read(run.OUT/'challengers-protocol.json')['source_sha256']==run.sha(__file__)
        run.write(run.OUT/'challengers-selection.json',[run.read(run.OUT/'rules'/f'{m}.json') for m in RULES])
        with cf.ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(evaluate,RULES))

if __name__=='__main__':main()
