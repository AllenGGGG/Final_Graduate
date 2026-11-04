"""One controlled sampler comparison; independent Double-DQN in native scenes."""
import argparse,concurrent.futures as cf,json,shutil
from pathlib import Path
from slice_alloc.tools import allocation_credit_run as common
from slice_alloc.allocation_pairing import PairedEnv,PairedReplay,PairedDoubleDQN
from slice_alloc.distributional_allocation import NStepDictReplay

OUT=common.PKG/'results/paired'
CONFIGS={'paired_uniform':(32,0.),'paired_contrast':(32,0.)}
ENV=None;BUFFER=None

def freeze():
    base=common.read(common.OUT/'protocol.json')
    sources=(*base['sources'],'allocation_pairing.py','tools/allocation_pairing_run.py','tests/test_allocation_pairing.py')
    p={**base,'sources':{n:common.sha(common.PKG/n) for n in sources},'configs':CONFIGS,
       'collection':'fresh native trajectory alternates with same-seed replay of its own recorded prefix and one random alternative legal action, then current policy continues; no expert or rule actions',
       'samplers':{'paired_uniform':'ordinary uniform n32 replay','paired_contrast':'half of batch from same-causal-observation distinct-action pairs, rest ordinary uniform replay; same scalar Double-Q TD targets'},
       'selection':'compare all2init full32k validation-selected checkpoints, expand maximum mean to5 iff improvement over matching pretrained independent RL >=.002; no heuristic initialization or test tuning',
       'base_protocol_sha256':common.sha(common.OUT/'protocol.json')}
    p=json.loads(json.dumps(p));path=OUT/'protocol.json'
    if path.exists():assert common.read(path)==p
    else:
        common.write(path,p)
        for n in sources:
            t=OUT/'frozen'/n;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(common.PKG/n,t)
        shutil.copytree(common.OUT/'initial_validation',OUT/'initial_validation',dirs_exist_ok=True)
    return p

def make_env(*args,**kwargs):
    global ENV
    e=PairedEnv(*args,**kwargs)
    if 'seed' in kwargs:ENV=e
    return e

def train(spec):
    global BUFFER
    config,init=spec
    cls=NStepDictReplay if config=='paired_uniform' else PairedReplay
    def buffer(*args,**kwargs):
        global BUFFER
        BUFFER=cls(*args,**kwargs);return BUFFER
    common.OUT=OUT;common.CONFIGS=CONFIGS
    common.envs.CreditEnv=make_env;common.MaskedDoubleDQN=PairedDoubleDQN;common.NStepDictReplay=buffer
    row=common.train(spec)
    row['collection_statistics']=dict(completed_native_episodes=len(ENV.episode_specs),
        verified_same_state_branches=len(ENV.pair_proofs),branch_proofs=ENV.pair_proofs,
        paired_state_count=len(getattr(BUFFER,'paired_keys',())),paired_minibatches=getattr(BUFFER,'paired_batches',0))
    common.write(OUT/'training'/config/str(init)/'training.json',row)
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--config',choices=CONFIGS);p.add_argument('--initializations',nargs='+',type=int,default=[11,23]);args=p.parse_args();freeze()
    configs=(args.config,) if args.config else tuple(CONFIGS)
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(train,[(c,i) for c in configs for i in args.initializations]))
    common.write(OUT/('pilot_complete.json' if set(args.initializations)=={11,23} else 'expansion_complete.json'),rows)
if __name__=='__main__':main()
