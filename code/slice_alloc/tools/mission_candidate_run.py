"""One-factor actor-encoder ablation against the 3e-4 masked PPO setting."""
import argparse
import concurrent.futures as cf
import os
from pathlib import Path
import shutil
from slice_alloc.candidate_mission_policy import CandidateMissionPolicy
from slice_alloc.tools import mission_allocation_run as run

REFERENCE=run.OUT
OUT=Path(os.environ.get('LOWALT_MISSION_CANDIDATE_OUTPUT',str(run.PACKAGE/'results/mission-candidate'))).resolve()
original_parameters=run.parameters


def parameters():
    p=original_parameters();p['learning_rate']=3e-4;return p


def configure():
    protocol=run.read(REFERENCE/'protocol.json')
    protocol.update(hyperparameters=parameters(),ablation='shared candidate scorer instead of flattened fixed-index actor; compare with 3e-4 PPO',
        source_protocol_sha256=run.sha(REFERENCE/'protocol.json'))
    for name in ('candidate_mission_policy.py','tools/mission_candidate_run.py'):
        protocol['sources'][name]=run.sha(run.PACKAGE/name)
    path=OUT/'protocol.json'
    if path.exists():assert run.read(path)==protocol
    else:
        run.write(path,protocol)
        for name in ('diagnosis.json','prior_validation.json'):shutil.copy2(REFERENCE/name,OUT/name)
        for name in protocol['sources']:
            p=OUT/'frozen'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(run.PACKAGE/name,p)
    run.OUT=OUT;run.parameters=parameters;run.MissionPolicy=CandidateMissionPolicy;run.check()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);parser.add_argument('--initializations',type=int,nargs='+',default=[11,23])
    args=parser.parse_args();assert set(args.initializations)<=set(run.INITS)
    configure()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(run.train,args.initializations))
    run.write(OUT/('training_complete.json' if set(args.initializations)==set(run.INITS) else 'pilot_complete.json'),rows)


if __name__=='__main__':main()
