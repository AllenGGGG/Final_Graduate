"""One-command allocation using a method chosen only on frozen validation data."""
import argparse
import numpy as np
from slice_alloc import modal_binding as b
from slice_alloc.distributional_allocation import MaskedDoubleQRDQN
from slice_alloc.tools import modal_binding_run as run
from slice_alloc.tools import modal_binding_challengers as extra


def select():
    rules=[run.read(run.OUT/'rules'/f'{m}.json') for m in (*run.RULES,*extra.RULES)]
    best=max(rules,key=lambda r:r['chosen']['rate'])
    records=[run.read(run.OUT/'training'/'joint_qrdqn'/str(i)/'training.json') for i in run.INITS]
    mean=float(np.mean([r['selected']['validation']['rate'] for r in records]))
    learned=mean>best['chosen']['rate']  # ties prefer the cheaper rule
    return dict(method='joint_qrdqn' if learned else best['method'],
        validation_rate=mean if learned else best['chosen']['rate'],
        learned_validation_mean=mean,best_rule_validation=best,
        model_initialization=11 if learned else None,
        selection='Validation only; all five learned initializations averaged to choose the method; initialization11 is predetermined for single-model deployment; ties prefer the rule.')


def execute(cfg,prefix,choice=None):
    choice=choice or select();method=choice['method']
    if method in extra.RULES:
        return extra.episode(cfg,method,choice['best_rule_validation']['chosen']['threshold'],prefix)
    if method=='joint_qrdqn':
        model=MaskedDoubleQRDQN.load(run.OUT/'models'/'joint_qrdqn-11.zip',device='cpu')
        return run.episode(cfg,method,model,prefix)
    return run.episode(cfg,method,prefix=prefix,threshold=choice['best_rule_validation']['chosen']['threshold'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario');p.add_argument('--seed',type=int,default=13350)
    p.add_argument('--load',type=int,default=48);p.add_argument('--output',required=True);args=p.parse_args()
    import torch
    torch.set_num_threads(1)
    cfg=run.read(args.scenario) if args.scenario else b.scenario(args.seed,args.load)
    choice=select();run.write(str(args.output)+'-selection.json',choice)
    r=execute(cfg,args.output,choice)
    print({'method':choice['method'],'requests':r['requests'],'success':r['success'],'audited':r['audited']})

if __name__=='__main__':main()
