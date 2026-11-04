"""Development-only feasible hindsight search; never an upper bound."""
import concurrent.futures as cf
import hashlib
from pathlib import Path
import numpy as np
from slice_alloc import guided_allocation as g
from slice_alloc import allocation_final as a
from slice_alloc.tools import allocation_improve_run as run


def task(seed):
    parameters=run.read(run.OUT/'diagnosis.json')['selected']['parameters']
    outcomes=[]
    for trial in range(17):
        rng=np.random.default_rng(seed*100+trial)
        total=0
        for load in run.LOADS:
            env=g.GuidedEnv(rule_parameters=parameters)
            try:
                obs,_=env.reset(options={'scenario':a.scenario(seed,load)})
                done=False;actions=[]
                while not done:
                    action=g.rule_action('modal_greedy',env,parameters)
                    if trial and rng.random()<(.05 if trial<=8 else .15):
                        action=int(rng.choice(np.flatnonzero(obs['mask']>.5)))
                    actions.append(action)
                    obs,_,done,_,_=env.step(action)
                total+=env.result()['success']
            finally:env.close()
        outcomes.append(dict(trial=trial,success=total))
    return dict(seed=seed,requests=sum(run.LOADS),rule_success=outcomes[0]['success'],
                feasible_hindsight_success=max(x['success'] for x in outcomes),trials=outcomes)


def main():
    run.check()
    with cf.ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(task,run.DEV))
    data=dict(rows=rows,rule_success=sum(r['rule_success'] for r in rows),
        feasible_hindsight_success=sum(r['feasible_hindsight_success'] for r in rows),
        requests=sum(r['requests'] for r in rows),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='17 feasible trajectories per scenario, selected with hindsight on development only. A lower bound on achievable optimum; a small gap cannot imply a small optimality gap.')
    run.write(run.OUT/'space_probe.json',data)
    print(data,flush=True)


if __name__=='__main__':main()
