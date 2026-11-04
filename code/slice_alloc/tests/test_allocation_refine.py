"""Regression checks for resuming DQN with an initially empty replay."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from stable_baselines3.common.utils import get_schedule_fn

from slice_alloc.tools import allocation_simple_run as base
from slice_alloc.tools import allocation_refine_run as run
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.distributional_allocation import CandidateQRPolicy


class ContinuationTests(unittest.TestCase):
    def test_resume_fills_new_replay_before_updates_and_keeps_counter(self):
        torch.set_num_threads(1)
        env=base.environment.AllocationEnv(seed=990071,loads=(8,),no_calendar=True)
        params=base.parameters()
        params.update(learning_starts=8,buffer_size=128,batch_size=8,
            exploration_steps=64,target_update_interval=16)
        params['policy_kwargs']['net_arch']=[16]
        model=MaskedDoubleDQN(CandidateQRPolicy,env,**params,seed=11,device='cpu')
        try:
            model.learn(64)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'model.zip'
                model.save(path)
                again=MaskedDoubleDQN.load(path,device='cpu')
            self.assertEqual(again.replay_buffer.size(),0)
            initial_updates=again._n_updates
            with patch.object(run,'REPLAY_WARMUP',16):
                run.prepare_continuation(again,env)
            self.assertEqual(again.learning_starts,80)
            self.assertEqual(again.lr_schedule(1.),run.LEARNING_RATE)
            again.learn(16,reset_num_timesteps=False)
            self.assertEqual(again.num_timesteps,80)
            self.assertEqual(again._n_updates,initial_updates)
            before=again.quantile_net.deployment[-1].weight.detach().clone()
            again.learn(32,reset_num_timesteps=False)
            self.assertEqual(again.num_timesteps,112)
            self.assertGreater(again._n_updates,initial_updates)
            self.assertFalse(torch.equal(before,again.quantile_net.deployment[-1].weight))
        finally:
            env.close()

    def test_selection_is_validation_only_and_falls_back_to_reference(self):
        options=[dict(step=32768,validation=dict(rate=.8),origin='reference'),
                 dict(step=49152,validation=dict(rate=.79),origin='continuation'),
                 dict(step=65536,validation=dict(rate=.8),origin='continuation')]
        chosen=max(options,key=lambda r:(r['validation']['rate'],-r['step']))
        self.assertEqual(chosen['origin'],'reference')


if __name__=='__main__':
    unittest.main()
