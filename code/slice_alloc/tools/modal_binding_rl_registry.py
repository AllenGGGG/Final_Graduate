"""Current trials plus the immutable, already-trained QR-DQN reference."""
from slice_alloc.tools import modal_binding_rl_run as run
from slice_alloc.tools import modal_binding_run as old
from slice_alloc.tools import modal_binding_rl_ppo as ppo


def all_specs():
    return ppo.all_specs()+[('legacy', i) for i in run.INITS]


def load(method, path):
    return ppo.load(method, path)


def episode(cfg, method, model, prefix=None):
    if method == 'legacy':
        return old.episode(cfg, 'learned', model, prefix)
    return run.episode(cfg, method, model, prefix)
