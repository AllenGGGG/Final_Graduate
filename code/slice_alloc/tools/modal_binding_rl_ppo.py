"""Independent policy-gradient control, same native contracts and sample budget."""
import concurrent.futures as cf
import shutil
import time

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from slice_alloc.modal_binding_rl import BindingRLEnv
from slice_alloc.tools import modal_binding_rl_run as run

INITS = (11, 23, 37)
HYPER = dict(learning_rate=3e-4, n_steps=512, batch_size=128, n_epochs=10,
    gamma=1., gae_lambda=1., clip_range=.2, ent_coef=.01, vf_coef=.5,
    max_grad_norm=.5, policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))


class Adapter:
    def __init__(self, model):
        self.model = model

    def predict(self, observation, deterministic=True):
        return self.model.predict(observation, deterministic=deterministic,
            action_masks=observation['mask'] > .5)


def all_specs():
    return run.specs()+[('ppo', i) for i in INITS]


def load(method, path):
    if method == 'ppo':
        return Adapter(MaskablePPO.load(path, device='cpu'))
    return run.MaskedDoubleQRDQN.load(path, device='cpu')


class Checkpoints(BaseCallback):
    def __init__(self, folder):
        super().__init__()
        self.folder, self.records = folder, []

    def _on_step(self):
        if self.num_timesteps not in run.BUDGETS:
            return True
        path = self.folder/f'{self.num_timesteps}.zip'
        self.model.save(path)
        val = run.score(Adapter(self.model), 'ppo')
        self.records.append(dict(step=self.num_timesteps, path=str(path.relative_to(run.b.p.ROOT)),
            sha256=run.sha(path), validation=val))
        run.write(self.folder/'progress.json', self.records)
        print('CHECKPOINT ppo', self.folder.name, self.num_timesteps, round(val['rate'], 5), flush=True)
        return True


def train(init):
    torch.set_num_threads(1)
    folder = run.OUT/'training/ppo'/str(init)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'training.json').exists():
        return run.read(folder/'training.json')
    env = BindingRLEnv(seed=init+100, loads=run.LOADS, shaping=0.)
    wrapped = ActionMasker(env, lambda e: e.encode()['mask'] > .5)
    model = MaskablePPO('MultiInputPolicy', wrapped, **HYPER, seed=init, device='cpu', verbose=0)
    before = model.policy.action_net.weight.detach().clone()
    cb = Checkpoints(folder)
    started = time.perf_counter()
    try:
        model.learn(max(run.BUDGETS), callback=cb)
        change = float(torch.linalg.vector_norm(model.policy.action_net.weight.detach()-before))
        assert change > 0
        best = max(cb.records, key=lambda r: (r['validation']['rate'], -r['step']))
        selected = run.OUT/'models'/f'ppo-{init}.zip'
        selected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run.b.p.ROOT/best['path'], selected)
        assert run.score(load('ppo', selected), 'ppo') == best['validation']
        result = dict(method='ppo', init=init, steps=model.num_timesteps, updates=model._n_updates,
            weight_change=change, seconds=time.perf_counter()-started, stages=cb.records,
            selected=best, model=str(selected.relative_to(run.b.p.ROOT)), model_sha256=run.sha(selected),
            training_seeds=env.episode_specs, protocol_sha256=run.sha(run.OUT/'protocol.json'),
            ppo_protocol_sha256=run.sha(run.OUT/'ppo_protocol.json'))
        run.write(folder/'training.json', result)
        print('TRAINED ppo', init, flush=True)
    finally:
        env.close()


def evaluate_group(spec):
    method, init, scope, mix, count, seeds = spec
    torch.set_num_threads(1)
    model = load(method, run.OUT/'models'/f'{method}-{init}.zip')
    rows = []
    for seed in seeds:
        prefix = run.OUT/'native'/f'{method}-{init}'/f'{scope}-{mix}-{count}-{seed}'
        path = str(prefix)+'-metrics.json'
        from pathlib import Path
        result = run.read(path) if Path(path).exists() else run.episode(run.b.scenario(seed, count, mix), method, model, prefix)
        row = {k: v for k, v in result.items() if k not in ('decisions', 'observations', 'settled')}
        row.update(method=method, init=init, scope=scope, mix=mix, load=count, seed=seed)
        rows.append(row)
    return rows


def evaluate():
    cases = [('test', 'balanced', l, run.TEST) for l in run.LOADS]+[
        ('ood', mix, l, run.OOD) for mix, l in [('object_heavy', 80), ('mobile_heavy', 80), ('balanced', 112), ('novel', 80)]]
    groups = [('ppo', i, *case) for i in INITS for case in cases]
    rows = run.read(run.OUT/'evaluation.json')
    rows = [r for r in rows if r['method'] != 'ppo']
    with cf.ProcessPoolExecutor(max_workers=8) as pool:
        for batch in pool.map(evaluate_group, groups):
            rows.extend(batch)
    run.write(run.OUT/'evaluation.json', rows)
    print('PPO EVALUATED', len(rows), flush=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        evaluate()
    else:
        with cf.ProcessPoolExecutor(max_workers=3) as pool:
            list(pool.map(train, INITS))
