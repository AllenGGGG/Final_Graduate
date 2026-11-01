"""Descriptive validation of final fixed-budget checkpoints; never selects weights."""
import concurrent.futures
import json
from .patrol_run import OUT,PROTOCOL_SPEC,source_hashes,sha,eval_group,write


def main():
    protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['sources']==source_hashes()
    specs=[(m,i,s) for m in ('ddqn','ppo') for i in PROTOCOL_SPEC['initializations'] for s in PROTOCOL_SPEC['validation_seeds']]
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        for group in pool.map(eval_group,specs):rows.extend(group)
    write(OUT/'validation.json',dict(scope='all final fixed-budget checkpoints; descriptive validation only; no checkpoint or hyperparameter selection',protocol_sha256=sha(OUT/'protocol.json'),rows=rows))
    print(json.dumps(dict(episodes=len(rows),success=sum(r['success'] for r in rows)/sum(r['requests'] for r in rows))),flush=True)

if __name__=='__main__':main()
