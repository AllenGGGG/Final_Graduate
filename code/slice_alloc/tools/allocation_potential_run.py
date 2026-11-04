"""One-factor development ablation: valid telescoping potential, gamma=1.

Reuses the exact training function, policy, splits and physical environment.
The only variant is Phi(s)=0.5*observed_active_contracts, Phi(terminal)=0.
"""
import argparse
import concurrent.futures as cf
import os
from pathlib import Path
import shutil
import torch
from slice_alloc.tools import allocation_improve_run as run

REFERENCE=run.OUT
OUT=Path(os.environ.get('LOWALT_POTENTIAL_OUTPUT',str(run.PACKAGE/'results/potential'))).resolve()


class PotentialEnv(run.TrainEnv):
    def __init__(self,*args,**kwargs):
        kwargs['shaping']=.5
        super().__init__(*args,**kwargs)


def configure():
    original=run.read(REFERENCE/'protocol.json')
    protocol={**original,'potential_coefficient':.5,
        'reward':'native_success + Phi(next)-Phi(current), gamma1, terminalPhi0; episode total exactly native_success',
        'ablation':'only reward potential differs from unshaped rule-prior DoubleDQN',
        'source_hashes':{**original['source_hashes'],
            'tools/allocation_potential_run.py':run.sha(__file__)},
        'prior_diagnosis_sha256':run.sha(REFERENCE/'diagnosis.json')}
    path=OUT/'protocol.json'
    if path.exists():assert run.read(path)==protocol
    else:
        run.write(path,protocol)
        shutil.copy2(REFERENCE/'diagnosis.json',OUT/'diagnosis.json')
        for name in protocol['source_hashes']:
            target=OUT/'frozen'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(run.PACKAGE/name,target)
    # Explicit dependency injection, recorded in this frozen variant source.
    run.OUT=OUT
    run.TrainEnv=PotentialEnv
    run.check()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--initializations',nargs='+',type=int,default=list(run.INITS));parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();assert set(args.initializations)<=set(run.INITS)
    configure()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(run.train,args.initializations))
    run.write(OUT/('training_complete.json' if set(args.initializations)==set(run.INITS) else 'pilot_complete.json'),rows)


if __name__=='__main__':main()
