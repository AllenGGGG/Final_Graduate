import tempfile
import unittest
from pathlib import Path

import numpy as np

from slice_alloc import modal_binding as b
from slice_alloc.modal_binding_rl import BindingRLEnv


class BindingRLTests(unittest.TestCase):
    def test_features_use_only_arrived_contracts_and_observed_history(self):
        env = BindingRLEnv()
        try:
            obs, _ = env.reset(options={'scenario': b.scenario(25001, 16)})
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(obs['state'][4*len(b.p.RESOURCES)+16], 0.)
            env.cfg['private'] = {'motions': [], 'initial_positions': {}}
            for sid in list(env.requests):
                if sid != env.current['request_id']:
                    env.requests[sid] = None
            again = env.encode()
            for key in obs:
                np.testing.assert_array_equal(obs[key], again[key])
        finally:
            env.close()

    def test_shaping_telescopes_and_progress_matches_packet_log(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)/'rl'
            env = BindingRLEnv()
            try:
                cfg = b.scenario(25002, 48)
                obs, _ = env.reset(options={'scenario': cfg, 'prefix': prefix, 'trace': True})
                total = native = 0.
                done = False
                while not done:
                    action = b.baseline('joint_price', env)
                    self.assertEqual(obs['mask'][action], 1.)
                    obs, reward, done, _, info = env.step(action)
                    self.assertTrue(env.observation_space.contains(obs))
                    total += reward
                    native += info['native_reward']
                result = env.result()
                self.assertEqual(total, native)
                self.assertEqual(total, result['success'])
                with open(str(prefix)+'-events.csv') as stream:
                    events = list(csv.DictReader(stream))
                for message in result['observations']:
                    o = message['observation']
                    for progress in o['active']:
                        emitted = [r for r in events if r['event'] == 'emit' and int(r['slice']) == progress['id'] and float(r['time']) < o['time']-1e-9]
                        completed = [r for r in events if r['event'] == 'complete' and int(r['slice']) == progress['id'] and float(r['time']) < o['time']-1e-9]
                        self.assertEqual(progress['generated'], len(emitted))
                        self.assertEqual(progress['received'], len(completed))
            finally:
                env.close()

    def test_executor_extension_preserves_exact_packet_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = b.scenario(25003, 48)
            results = []
            for cls, name in [(b.BindingEnv, 'old'), (BindingRLEnv, 'new')]:
                env = cls()
                try:
                    prefix = Path(tmp)/name
                    obs, _ = env.reset(options={'scenario': cfg, 'prefix': prefix, 'trace': True})
                    done = False
                    while not done:
                        obs, _, done, _, _ = env.step(b.baseline('joint_price', env))
                    results.append((env.result(), Path(str(prefix)+'-events.csv').read_bytes()))
                finally:
                    env.close()
            self.assertEqual(results[0][0]['decisions'], results[1][0]['decisions'])
            self.assertEqual(results[0][0]['settled'], results[1][0]['settled'])
            self.assertEqual(results[0][1], results[1][1])


if __name__ == '__main__':
    unittest.main()
