"""Run the validation-selected single policy; no ensemble or rule fallback."""
import argparse
from pathlib import Path

import torch

from slice_alloc.tools import allocation_final_run as run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=46000)
    parser.add_argument('--load',type=int,default=48)
    parser.add_argument('--output',type=Path,default=run.OUT/'demo/allocation')
    args=parser.parse_args()
    torch.set_num_threads(1)
    selection=run.read(run.OUT/'selection.json')
    algorithm=selection['selected']
    model=run.load_model(algorithm,run.OUT/'models'/f'{algorithm}-full-11.zip')
    cfg=run.a.scenario(args.seed,args.load)
    result=run.episode(cfg,algorithm,model,prefix=args.output)
    run.write(str(args.output)+'-allocation.json',dict(
        algorithm=algorithm,initialization=11,model_sha256=run.sha(run.OUT/'models'/f'{algorithm}-full-11.zip'),
        success=result['success'],requests=result['requests'],admitted=result['admitted'],
        decisions=result['decisions'],scenario_sha256=result['scenario_sha256']))
    print(f"{algorithm}, init11: {result['success']}/{result['requests']} successful, "
          f"{result['admitted']} admitted. Output: {args.output}")


if __name__=='__main__':
    main()
