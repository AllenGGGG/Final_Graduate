import tempfile
import unittest
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from slice_alloc.modal_binding_rl import BindingRLEnv
from slice_alloc.tools.modal_binding_rl_ppo import Adapter, load


class PPOContracts(unittest.TestCase):
    def test_learning_save_reload_and_execution_keep_capacity_masks(self):
        torch.set_num_threads(1)
        env = BindingRLEnv(seed=601, loads=(16,), shaping=0.)
        wrapped = ActionMasker(env, lambda e: e.encode()['mask'] > .5)
        try:
            model = MaskablePPO('MultiInputPolicy', wrapped, n_steps=64, batch_size=32,
                n_epochs=1, gamma=1., gae_lambda=1., seed=601, device='cpu',
                policy_kwargs=dict(net_arch=dict(pi=[32], vf=[32])))
            before = model.policy.action_net.weight.detach().clone()
            model.learn(128)
            self.assertGreater(float(torch.linalg.vector_norm(before-model.policy.action_net.weight.detach())), 0.)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'ppo.zip'
                model.save(path)
                again = load('ppo', path)
                obs, _ = env.reset()
                done = False
                total = 0.
                while not done:
                    a = int(Adapter(model).predict(obs)[0])
                    self.assertEqual(a, int(again.predict(obs)[0]))
                    self.assertEqual(obs['mask'][a], 1.)
                    obs, reward, done, _, _ = env.step(a)
                    total += reward
                self.assertEqual(total, env.result()['success'])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
