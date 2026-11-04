"""RL revision of research point 1, without changing business or packet physics.

Extra inputs are observed counters, past motion and already admitted contracts.
Potential shaping telescopes to zero over a complete undiscounted episode.
No prediction, runtime reconfiguration, expert action or price rule is used.
"""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from slice_alloc import modal_binding as b

EXTRA_DIM = 12*8 + 9*7 + 8


class Session(b.Session):
    binary_name = 'binding-rl-scenario'
    ns3_root = b.p.NS3
    def __init__(self, cfg, prefix=None, trace=False):
        b.validate(cfg)
        self.temp = None
        if prefix is None:
            self.temp = tempfile.TemporaryDirectory(prefix='binding-rl-')
            prefix = Path(self.temp.name)/'episode'
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(str(self.prefix)+'-scenario.json')
        self.config_path.write_text(json.dumps(cfg, indent=2)+'\n')
        self.stderr = open(str(self.prefix)+'-stderr.log', 'w+')
        self.messages = []
        self.started = time.perf_counter()
        self.process = subprocess.Popen([
            str(self.ns3_root/'build/scratch'/self.binary_name),
            f'--config={self.config_path}', f'--output={self.prefix}', f'--trace={int(trace)}'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, bufsize=1,
            env=dict(os.environ, LD_LIBRARY_PATH=str(self.ns3_root/'build/lib')))
        self.observation = self.read()
        if 'active' not in self.observation:
            self.close()
            raise ValueError('Executor does not provide observed active-slice counters')


class BindingRLEnv(b.BindingEnv):
    session_class = Session
    def __init__(self, *args, enhanced=True, shaping=.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.enhanced = enhanced
        self.shaping = float(shaping)
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            'state': spaces.Box(-100, 100,
                (b.STATE_DIM + (EXTRA_DIM if enhanced else 0),), np.float32)})

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if self.session:
            self.session.close()
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = options or {}
        cfg = opts.get('scenario')
        if cfg is None:
            cfg = b.scenario(int(self.rng.integers(30000000, 31000000)),
                             int(self.rng.choice(self.loads)))
        self.episode_specs.append(cfg['radio_seed']-1)
        self.cfg = cfg
        self.requests = {r['id']: b.Service(**{**r, 'members': tuple(r['members'])})
                         for r in cfg['requests']}
        self.ledger = b.p.Ledger(cfg['public'])
        self.records, self.settled, self.previous = [], {}, {}
        self.previous_time = self.motion_frame = None
        self.velocities = {}
        self.resource_integral = np.zeros(len(b.p.RESOURCES))
        self.session = self.session_class(cfg, opts.get('prefix'), opts.get('trace', False))
        self.current = self.session.observation
        return self.encode(), {}

    def encode(self):
        # Keep the existing capacity calendar and exact same feasible actions.
        previous_time = self.previous_time
        fresh = self.current['kind'] != 'done' and (self.current['time'], self.current['request_id']) != self.motion_frame
        original = super().encode()
        if not self.enhanced:
            return original
        extra = np.zeros(EXTRA_DIM, np.float32)
        if self.current['kind'] == 'done':
            return {**original, 'state': np.r_[original['state'], extra]}
        now = self.current['time']
        request = self.requests[self.current['request_id']]
        # Motion validity is cached per frame, so repeated encoding is identical.
        if fresh:
            self.motion_known = previous_time is not None and now > previous_time+1e-9
        original['state'][4*len(b.p.RESOURCES)+16] = float(self.motion_known)
        endpoints = []
        for n in range(5, 17):
            key = str(n)
            x, y, z = self.current['positions'][key]
            ap = self.current['associations'][key]
            endpoints.extend([x/150, y/150, z/100,
                self.velocities[key][0]/11 if self.motion_enabled else 0.,
                float(self.cfg['public']['missions'][key]['role'] == 'patrol'),
                float(ap == 3), float(ap == 4), float(ap == 0)])
        # Group by mode and business. Totals retain load; ratios retain progress.
        groups = np.zeros((9, 7), np.float32)
        shared = np.zeros(8, np.float32)
        plans = {p.request.id: p for p in self.ledger.plans}
        for observed in self.current['active']:
            sid = observed['id']
            r = self.requests[sid]
            plan = plans[sid]
            generated = observed['generated']
            deficit = max(0, observed['matured']-observed['on_time'])
            group = b.MODES.index(plan.request.mode)*3 + b.SERVICES.index(r.service)
            mobile = np.mean([self.cfg['public']['missions'][str(n)]['role'] == 'patrol' for n in r.members])
            groups[group] += [1/20, max(0, r.drain-now)/160,
                generated/max(1, r.expected), observed['on_time']/max(1, generated),
                deficit/max(1, r.expected), r.size/r.period/100000, mobile]
            overlap = len(set(r.members) & set(request.members))
            same = r.reusable and request.reusable and (r.tenant, r.content, r.version) == (request.tenant, request.content, request.version)
            shared += [overlap/20, overlap*r.size/r.period/100000,
                overlap*deficit/max(1, r.expected), float(same)/10,
                float(same and plan.cache)/10, float(same and plan.warm)/10,
                float(r.tenant == request.tenant)/20, float(r.service == request.service)/20]
        counts = groups[:, 0]*20
        for col in (2, 3, 4, 6):
            groups[:, col] /= np.maximum(1, counts)
        extra[:] = np.r_[endpoints, groups.ravel(), shared]
        return {**original, 'state': np.r_[original['state'], extra]}

    def potential(self):
        if self.current['kind'] == 'done':
            return 0.
        return self.shaping*len(self.current['active'])

    def step(self, action):
        before = self.potential()
        obs, native_reward, done, truncated, info = super().step(action)
        reward = native_reward + self.potential() - before
        return obs, reward, done, truncated, {**info, 'native_reward': native_reward}
