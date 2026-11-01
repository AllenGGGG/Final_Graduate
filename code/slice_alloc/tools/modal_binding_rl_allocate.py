"""Execute the predetermined single RL model; never fall back to an expert rule."""
import argparse
from pathlib import Path

import torch

from slice_alloc import modal_binding as b
from slice_alloc.distributional_allocation import MaskedDoubleQRDQN
from slice_alloc.tools import modal_binding_rl_run as run
from slice_alloc.tools.modal_binding_rl_registry import load, episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario')
    parser.add_argument('--seed', type=int, default=29000)
    parser.add_argument('--load', type=int, default=48)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    choice = run.read(run.OUT/'rl_selection.json')
    method, init = choice['method'], choice['initialization']
    protocol = run.read(run.OUT/'protocol.json')
    common = ('slice_alloc/modal_binding.py', 'slice_alloc/patrol.py',
        'slice_alloc/distributional_allocation.py', 'slice_alloc/initial_allocation.py')
    for name in common:
        assert run.sha(b.p.ROOT/name) == protocol['sources'][name]
    if method == 'legacy':
        assert run.sha(b.p.NS3/'build/scratch/binding-scenario') == run.read(run.OUT/'legacy_protocol.json')['binary_sha256']
    else:
        assert run.sha(b.p.NS3/'build/scratch/binding-rl-scenario') == protocol['binary_sha256']
        assert run.sha(b.p.ROOT/'slice_alloc/modal_binding_rl.py') == protocol['sources']['slice_alloc/modal_binding_rl.py']
    model_path = run.OUT/'models'/f'{method}-{init}.zip'
    manifest = run.read(run.OUT/'training'/method/str(init)/'training.json')
    assert run.sha(model_path) == manifest['model_sha256']
    model = load(method, model_path)
    cfg = run.read(args.scenario) if args.scenario else b.scenario(args.seed, args.load)
    b.validate(cfg)
    run.write(str(args.output)+'-selection.json', dict(method=method, initialization=init,
        model_sha256=run.sha(model_path), selected_step=manifest['selected']['step'],
        rule_fallback=False, ensemble=False, selection='initialization fixed before test'))
    result = episode(cfg, method, model, Path(args.output))
    print(dict(method=method, initialization=init, requests=result['requests'],
        success=result['success'], admitted=result['admitted'], audited=result['audited'],
        modes=result['modes']))


if __name__ == '__main__':
    main()
