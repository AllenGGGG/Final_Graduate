import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from stable_baselines3 import PPO

from slice_alloc import allocation_final as a
from slice_alloc.allocation_final_policy import CandidatePPOPolicy
from slice_alloc.tools import allocation_final_run as run


class FinalAllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_compatible_candidates_and_all_interval_capacity(self):
        env = a.AllocationEnv()
        try:
            obs, _ = env.reset(options={'scenario':a.scenario(39021,24)})
            done = False
            reward = 0
            while not done:
                r = env.requests[env.current['request_id']]
                self.assertTrue(all(p.request.mode in a.COMPATIBLE[r.service] for p in env.options))
                self.assertTrue(all(env.ledger.feasible(p) for p in env.options))
                action = run.rule_action('semantic',env)
                self.assertEqual(obs['mask'][action], 1.)
                obs, rew, done, _, _ = env.step(action)
                reward += rew
            self.assertEqual(reward,env.result()['success'])
        finally:
            env.close()

    def test_each_compatible_native_stack_has_same_task_and_correct_direction(self):
        cfg = a.scenario(39022,1,mobility='stationary')
        cfg['private']['initial_positions']['7'] = [20.,0.,80.]
        original = a.b.Service(**{**cfg['requests'][0], 'members':(7,)})
        for business, modes in a.COMPATIBLE.items():
            r = replace(original, id=1, service=business, start=2., stop=5.,
                period=.2 if business != 'object' else 2., deadline=.8,
                payload=1000, chunks=4 if business == 'object' else 1,
                content=1, reusable=business == 'object', minimum_ratio=.95)
            cfg['requests'] = [r.json()]
            for mode in modes:
                with self.subTest(business=business, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    env = a.AllocationEnv()
                    prefix = Path(tmp)/'one'
                    try:
                        env.reset(options={'scenario':cfg,'prefix':prefix,'trace':True})
                        index = next(i for i,p in enumerate(env.options,1) if p.request.mode == mode and p.edge == 1)
                        _,reward,done,_,_=env.step(index)
                        result=env.result()
                        self.assertTrue(done)
                        self.assertEqual(result['settled'][0]['expected'],r.expected)
                        self.assertEqual(reward,1.)
                        summary=run.read(str(prefix)+'-summary.json')
                        if business == 'telemetry':
                            self.assertEqual(summary['ip_uplink_payload_bytes'],r.expected*r.payload)
                        elif mode == 'id':
                            self.assertEqual(summary['id_decapsulated'],r.expected)
                        elif mode == 'ndn':
                            self.assertGreater(summary['ndn_source_payload_bytes'],0)
                    finally:
                        env.close()

    def test_telemetry_ledger_uses_uplink_wire_directions(self):
        env=a.AllocationEnv()
        cfg=a.scenario(39023,1,mobility='stationary')
        cfg['requests'][0].update(service='telemetry',members=[7],chunks=1,reusable=False)
        cfg['private']['initial_positions']['7']=[20.,0.,80.]
        try:
            env.reset(options={'scenario':cfg})
            plan=next(p for p in env.options if p.edge==1)
            self.assertGreater(plan.peak[a.b.p.INDEX['wire:3:1']],0.)
            self.assertGreater(plan.peak[a.b.p.INDEX['wire:1:0']],0.)
            self.assertEqual(plan.peak[a.b.p.INDEX['wire:0:1']],0.)
        finally:
            env.close()

    def test_repeated_encoding_and_unobserved_future_do_not_change_input(self):
        env=a.AllocationEnv()
        try:
            env.reset(options={'scenario':a.scenario(39024,12)})
            before=copy.deepcopy(env.encode())
            env.cfg['private']['motions']=[]
            env.cfg['private']['initial_positions']={}
            env.requests[max(env.requests)]=None
            for key,value in env.encode().items():
                np.testing.assert_array_equal(value,before[key])
        finally:
            env.close()

    def test_ppo_candidate_permutation_and_mask(self):
        env=a.AllocationEnv()
        try:
            obs,_=env.reset(options={'scenario':a.scenario(39025,8)})
            model=PPO(CandidatePPOPolicy,env,n_steps=64,batch_size=32,seed=11,device='cpu')
            with torch.no_grad():
                tensor,_=model.policy.obs_to_tensor(obs)
                dist,value=model.policy._outputs(tensor)
                permutation=np.r_[0,np.arange(1,11)[::-1]]
                permuted={k:v.clone() for k,v in tensor.items()}
                for key in ('candidates','mask'):
                    permuted[key]=permuted[key][:,permutation]
                other,other_value=model.policy._outputs(permuted)
                torch.testing.assert_close(other.distribution.probs,dist.distribution.probs[:,permutation])
                torch.testing.assert_close(value,other_value)
                self.assertEqual(float(dist.distribution.probs[0][tensor['mask'][0]<=.5].sum()),0.)
                tensor['mask'][:]=0
                tensor['mask'][:,0]=1
                only_reject,critic=model.policy._outputs(tensor)
                self.assertTrue(torch.isfinite(critic).all())
                self.assertEqual(int(only_reject.get_actions(True)[0]),0)
        finally:
            env.close()

    def test_quota_and_ablation_observations(self):
        cfg=a.scenario(39026,12)
        for variant in ('fixed','quota','no_motion','no_calendar'):
            env=run.environment(variant)
            try:
                obs,_=env.reset(options={'scenario':cfg})
                if variant=='no_calendar':
                    self.assertFalse(obs['state'][17:68].any())
                if variant=='no_motion':
                    self.assertFalse(obs['state'][81:89].any())
                    self.assertFalse(obs['state'][91:187].any())
                if variant=='fixed':
                    self.assertTrue(all(p.request.mode==a.b.native_mode(env.requests[env.current['request_id']]) for p in env.options))
                if variant=='quota':
                    self.assertTrue(all(env.quota_feasible(p) for p in env.options))
            finally:
                env.close()


if __name__=='__main__':
    unittest.main()
