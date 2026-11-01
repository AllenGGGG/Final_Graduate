import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from slice_alloc.initial_allocation import InitialAllocationEnv, PRICE_INDEX
from slice_alloc.distributional_allocation import (RuleFreeObservation, NStepDictReplay,
    CandidateQRPolicy, MaskedDoubleQRDQN)


class DistributionalPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def env(self):
        return RuleFreeObservation(InitialAllocationEnv())

    def model(self, env=None):
        return MaskedDoubleQRDQN(CandidateQRPolicy, env or self.env(),
            policy_kwargs=dict(n_quantiles=8, net_arch=[32]), learning_starts=16,
            buffer_size=256, batch_size=16, train_freq=4, gradient_steps=1,
            gamma=1., max_grad_norm=10., target_update_interval=32,
            replay_buffer_class=NStepDictReplay,
            replay_buffer_kwargs=dict(horizon=3, discount=1.), seed=41,
            device='cpu', verbose=0)

    def observation(self, env):
        obs={k:np.zeros(s.shape,np.float32) for k,s in env.observation_space.spaces.items()}
        obs['mask'][:4]=1
        obs['candidates'][1,0]=.2
        obs['candidates'][2,1]=.4
        return obs

    def test_rule_scalar_removed_before_replay(self):
        env=self.env()
        raw={k:np.zeros(s.shape,np.float32) for k,s in env.unwrapped.observation_space.spaces.items()}
        a=env.observation(raw)
        raw['candidates'][:,PRICE_INDEX]=1234
        b=env.observation(raw)
        np.testing.assert_array_equal(a['candidates'],b['candidates'])
        self.assertEqual(a['candidates'].shape[-1],raw['candidates'].shape[-1]-1)

    def test_library_defaults_do_not_mutate_declared_policy_settings(self):
        declared=dict(n_quantiles=8,net_arch=[32]);original=copy.deepcopy(declared)
        MaskedDoubleQRDQN(CandidateQRPolicy,self.env(),policy_kwargs=declared,buffer_size=32)
        self.assertEqual(declared,original)

    def test_masks_cover_warmup_exploration_and_greedy(self):
        model=self.model();obs=self.observation(self.env())
        obs['mask'][:]=0;obs['mask'][0]=1
        for explore in (0.,1.):
            model.exploration_rate=explore
            for _ in range(20):
                self.assertEqual(int(model.predict(obs,deterministic=False)[0]),0)
        model._last_obs={k:v[None] for k,v in obs.items()}
        self.assertEqual(int(model._sample_action(100)[0][0]),0)

    def test_candidate_permutation_equivariance_and_gradient(self):
        model=self.model();obs=self.observation(self.env())
        tensor,_=model.policy.obs_to_tensor(obs)
        changed=copy.deepcopy(tensor);order=torch.tensor([0,3,1,2,4])
        for key in ('candidates','needs','mask'):changed[key]=changed[key][:,order]
        torch.testing.assert_close(model.quantile_net(changed),model.quantile_net(tensor)[:,:,order],atol=1e-6,rtol=1e-5)
        model.quantile_net(tensor)[:,:,1].sum().backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.quantile_net.deployment.parameters()))

    def test_double_target_uses_online_selection_target_evaluation_and_mask(self):
        model=self.model()
        class Fixed(nn.Module):
            def __init__(self,values):super().__init__();self.values=values
            def forward(self,obs):return torch.tensor(self.values).float()[None,None].expand(2,8,-1)
        model.quantile_net=Fixed([0,8,5,999,7])
        model.quantile_net_target=Fixed([1,3,99,999,1])
        replay=SimpleNamespace(next_observations={'mask':torch.tensor([[1,1,1,0,1],[1,1,1,0,1]])},
            rewards=torch.tensor([[2.],[6.]]),dones=torch.tensor([[0.],[1.]]),discounts=torch.tensor([[.5],[.5]]))
        target=model.bootstrap_quantiles(replay)
        torch.testing.assert_close(target[0],torch.full((8,),3.5))
        torch.testing.assert_close(target[1],torch.full((8,),6.))

    def test_nstep_returns_and_terminal_flush_do_not_cross_episodes(self):
        env=self.env();model=self.model(env);buffer=model.replay_buffer
        obs=self.observation(env);obs={k:v[None] for k,v in obs.items()}
        for i in range(4):
            buffer.add(obs,obs,np.array([1]),np.array([i+1.]),np.array([i==3]),[{}])
        self.assertEqual(buffer.size(),4)
        np.testing.assert_array_equal(buffer.rewards[:4,0],[6,9,7,4])
        np.testing.assert_array_equal(buffer.dones[:4,0],[0,1,1,1])
        self.assertEqual(len(buffer.pending),0)
        buffer.add(obs,obs,np.array([0]),np.array([100.]),np.array([True]),[{}])
        self.assertEqual(buffer.rewards[4,0],100.)

    def test_real_update_and_full_roundtrip(self):
        env=self.env();model=self.model(env)
        try:
            before=model.quantile_net.deployment[-1].weight.detach().clone()
            model.learn(128)
            self.assertFalse(torch.equal(before,model.quantile_net.deployment[-1].weight))
            obs=self.observation(env)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model.zip';model.save(path)
                again=MaskedDoubleQRDQN.load(path,device='cpu')
            self.assertEqual(again.num_timesteps,128)
            self.assertEqual(again.exploration_steps,model.exploration_steps)
            a,_=model.policy.obs_to_tensor(obs);b,_=again.policy.obs_to_tensor(obs)
            torch.testing.assert_close(model.quantile_net(a),again.quantile_net(b),atol=0,rtol=0)
            self.assertEqual(int(model.predict(obs,deterministic=True)[0]),int(again.predict(obs,deterministic=True)[0]))
        finally:env.close()


class PairedBudgetStatisticsTests(unittest.TestCase):
    def test_identical_shared_trajectories_have_zero_difference_interval(self):
        from slice_alloc.tools import distributional_run as study
        rows=[]
        for method in ('qrdqn','qrdqn_equal','ppo_reference')+study.RULES:
            for init in (11,23,37):
                for seed in (1,2,3):
                    rows.append(dict(method=method,initialization=init,seed=seed,
                                     success=init+seed,requests=100))
        with patch.object(study,'TEST',(1,2,3)), patch.object(study.native,'aggregate',return_value={}):
            result=study.analyze(list(reversed(rows)))
        self.assertEqual(result['comparisons']['qrdqn_equal']['difference'],0.)
        self.assertEqual(result['comparisons']['qrdqn_equal']['ci95'],[0.,0.])
        # Unrelated training runs with the same marginal means still have
        # initialization uncertainty; they must not be paired by seed label.
        self.assertLess(result['comparisons']['ppo_reference']['ci95'][0],0.)
        self.assertGreater(result['comparisons']['ppo_reference']['ci95'][1],0.)


if __name__=='__main__':unittest.main()
