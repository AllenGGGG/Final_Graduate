"""Behavioral invariants for the separate three-modality platform."""
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from slice_alloc.trimodal import (Request, Ledger, compile_plan, candidates, workload,
    RESOURCE_KEYS, CAPACITIES, TrimodalEnv, write_tasks, validate_metadata, IP_DATAGRAM)


class Contracts(unittest.TestCase):
    def request(self, **kw):
        return Request(**(dict(id=1, tenant=1, mode='id', uav=5, start=2., end=7.,
                               period=.5, deadline=.8) | kw))

    def test_control_service_and_tunnel_are_independent_real_choices(self):
        rows, stats = candidates(self.request(), Ledger())
        self.assertEqual({p.key for p in rows}, {(1,1,0),(1,2,0),(2,1,0),(2,2,0)})
        self.assertEqual(stats['enumerated'], 4)
        cross = compile_plan(self.request(), 1, 2, 0)
        u = dict(zip(RESOURCE_KEYS, cross.usage))
        self.assertEqual(u['map:2'], 1)
        self.assertEqual(u['mapcache:1'], 1)
        self.assertGreater(u['wire:1>2'], 0)
        self.assertGreater(u['wire:2>1'], 0)

    def test_cache_saves_only_upstream_not_wireless(self):
        r = self.request(mode='ndn')
        cold, warm = [compile_plan(r, 1, 1, c) for c in (0, 1)]
        a, b = [dict(zip(RESOURCE_KEYS, p.usage)) for p in (cold, warm)]
        self.assertGreater(a['wire:0>1'], b['wire:0>1'])
        self.assertEqual(a['airtime:shared'], b['airtime:shared'])
        self.assertEqual(b['cache:1'], 1000)

    def test_atomic_failure_and_interval_release(self):
        ledger = Ledger()
        r = self.request()
        p = compile_plan(r, 1, 1, 0)
        ledger.commit(p)
        before = list(ledger.plans)
        bad = dataclasses.replace(p, usage=tuple(2*CAPACITIES[k] for k in RESOURCE_KEYS))
        with self.assertRaises(ValueError):
            ledger.commit(bad)
        self.assertEqual(before, ledger.plans)
        self.assertTrue(np.any(ledger.usage(r.end) > 0))
        self.assertTrue(np.all(ledger.usage(r.drain) == 0))

    def test_same_endpoint_multislice_uses_separate_state(self):
        l = Ledger()
        for i in (1,2):
            l.commit(compile_plan(self.request(id=i), 1, 2, 0))
        u = dict(zip(RESOURCE_KEYS, l.usage(3)))
        self.assertEqual(u['map:2'], 2)
        self.assertEqual(u['mapcache:1'], 2)

    def test_one_shared_airtime_budget_for_both_aps(self):
        p = compile_plan(self.request(attachments=(3,)), 1, 1, 0)
        q = compile_plan(self.request(attachments=(3,4)), 1, 1, 0)
        k = RESOURCE_KEYS.index('airtime:shared')
        self.assertEqual(p.usage[k], q.usage[k])
        self.assertEqual(sum(k.startswith('airtime') for k in RESOURCE_KEYS), 1)

    def test_no_future_trajectory_argument_and_reproducible_workload(self):
        a = workload(4001)
        self.assertEqual(a, workload(4001))
        self.assertNotEqual(a, workload(4002))
        # Only declared traffic/attachments and ledger enter candidate generation.
        p, _ = candidates(a[0], Ledger())
        q, _ = candidates(a[0], Ledger())
        self.assertEqual(p, q)

    def test_mask_and_terminal_observation(self):
        env = TrimodalEnv(count=6)
        obs, _ = env.reset(seed=4)
        for _ in range(6):
            self.assertTrue(env.observation_space.contains(obs))
            with self.assertRaises(ValueError):
                env.step(16)
            action = int(np.flatnonzero(obs['mask'])[-1])
            obs, _, done, _, _ = env.step(action)
        self.assertTrue(done)
        self.assertEqual(obs['mask'].sum(), 1)

    def test_model_schema_mismatch_refused(self):
        env = TrimodalEnv()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'meta.json'
            p.write_text(json.dumps(env.metadata_contract))
            validate_metadata(p, env)
            bad = dict(env.metadata_contract, max_candidates=15)
            p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                validate_metadata(p, env)

    def test_task_only_contains_selected_requests_and_instance_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'tasks.tsv'
            plan = compile_plan(self.request(), 2, 1, 0)
            write_tasks(p, [plan])
            self.assertEqual(len(p.read_text().splitlines()), 2)
            self.assertIn('id 5 2.0 7.0', p.read_text())
            self.assertEqual(IP_DATAGRAM, 1037)

    def test_identity_encapsulation_is_not_free(self):
        ip = compile_plan(self.request(mode='ip',attachments=(3,)),1,1,0)
        ident = compile_plan(self.request(mode='id',attachments=(3,)),1,1,0)
        k = RESOURCE_KEYS.index('wire:1>3')
        self.assertGreater(ident.usage[k], ip.usage[k])

    def test_invalid_contracts_fail(self):
        for kw in ({'period':0}, {'mode':'unknown'}, {'attachments':(8,)}, {'end':1}):
            with self.assertRaises(ValueError):
                self.request(**kw)
