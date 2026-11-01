"""Research point 1: business-compatible, direction-correct native allocation.

The title, three research points and business names are unchanged. This module
implements intelligent initial allocation only, with no prediction or migration.
"""
from copy import deepcopy

from slice_alloc import modal_binding as b
from slice_alloc.modal_binding_rl import BindingRLEnv, Session

COMPATIBLE = {'telemetry': ('ip',), 'command': ('ip', 'id'), 'object': ('ip', 'ndn')}
NS3 = b.p.NS3


def scenario(seed, load=48, mix='balanced', mobility='mixed', distance_loss=True):
    cfg = deepcopy(b.scenario(seed, load, mix))
    cfg['public'].update(distance_loss=distance_loss, loss_exponent=2.5, tx_power_dbm=20.)
    # Resource formula includes a conservative constant MAC allowance. Physical
    # retries remain measurable; the ledger is not a hard airtime guarantee.
    if mobility == 'stationary':
        cfg['private']['motions'] = []
    elif mobility == 'local':
        # Native Motion takes its public turn points from the task config.
        cfg['public']['motion_left'] = 30.
        cfg['public']['motion_right'] = 45.
        for raw in cfg['private']['initial_positions'].values():
            if 30. <= raw[0] <= 55.:
                raw[0] = 35.
    elif mobility != 'mixed':
        raise ValueError(mobility)
    return cfg


class FinalSession(Session):
    binary_name = 'allocation-final-scenario'
    ns3_root = NS3


class AllocationEnv(BindingRLEnv):
    session_class = FinalSession

    def __init__(self, *args, **kwargs):
        self.no_calendar = kwargs.pop('no_calendar', False)
        self.no_motion = kwargs.pop('no_motion', False)
        self.quota = kwargs.pop('quota', False)
        kwargs.setdefault('shaping', 0.)
        super().__init__(*args, **kwargs)

    def encode(self):
        obs = super().encode()
        if self.no_calendar:
            obs['state'][len(b.p.RESOURCES):4*len(b.p.RESOURCES)] = 0.
        if self.no_motion:
            obs['state'][4*len(b.p.RESOURCES)+13:4*len(b.p.RESOURCES)+21] = 0.
            obs['state'][b.STATE_DIM:b.STATE_DIM+12*8] = 0.
        return obs

    def quota_feasible(self, plan):
        ledger = b.p.Ledger(self.ledger.pub)
        ledger.capacity[:11] /= 3.
        ledger.plans = [p for p in self.ledger.plans if p.request.mode == plan.request.mode]
        return ledger.feasible(plan)

    def reset(self, seed=None, options=None):
        opts = dict(options or {})
        if 'scenario' not in opts:
            if seed is not None:
                import numpy as np
                self.rng = np.random.default_rng(seed)
            opts['scenario'] = scenario(int(self.rng.integers(41000000, 42000000)),
                                        int(self.rng.choice(self.loads)))
        return super().reset(seed=seed, options=opts)

    def get_candidates(self, r):
        if any(self.current['associations'].get(str(n), 0) not in
               self.ledger.pub['allowed_aps'][str(n)] for n in r.members):
            return []
        modes = (b.native_mode(r),) if self.fixed else COMPATIBLE[r.service]
        plans = []
        for mode in modes:
            for edge in (1, 2):
                choices = ((s, 0) for s in (1, 2)) if mode == 'id' else (
                    ((edge, c) for c in (0, edge)) if mode == 'ndn' else ((edge, 0),))
                for server, cache in choices:
                    if r.service == 'telemetry':
                        plan = b.p.compile_plan(r.bound('ip'), edge, server, 0,
                                                self.ledger.pub, self.current)
                    else:
                        plan = b.compile_plan(r, mode, edge, server, cache,
                                              self.ledger.pub, self.current)
                    if self.ledger.feasible(plan) and (not self.quota or self.quota_feasible(plan)):
                        plans.append(plan)
        return plans
