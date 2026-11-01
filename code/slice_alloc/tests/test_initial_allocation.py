import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from slice_alloc import patrol
from slice_alloc.initial_allocation import (AllocationPolicy, InitialAllocationEnv,
    PRICE_INDEX, HierarchicalCategorical, RewardUnit, QuotaLedger, make_scenario, marginal_cost)


class PPOPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env = InitialAllocationEnv()

    def observation(self):
        obs = {k: torch.zeros((1,) + space.shape) for k, space in
               self.env.observation_space.spaces.items()}
        obs['mask'][0, :4] = 1
        obs['member_mask'][0, 0] = 1
        for i in range(1, 4):
            obs['candidates'][0, i, i] = .2 * i
            obs['candidates'][0, i, PRICE_INDEX] = .3 * i
        return obs

    def policy(self, **kwargs):
        return AllocationPolicy(self.env.observation_space, self.env.action_space,
                                lambda _: 3e-4, **kwargs)

    def test_action_scores_do_not_contain_a_hidden_price_or_reject_rule(self):
        policy = self.policy()
        obs = self.observation()
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.zero_()
            probabilities = policy.get_distribution(obs).distribution.probs
        torch.testing.assert_close(probabilities[0, :4], torch.tensor([.5, 1/6, 1/6, 1/6]))
        changed = copy.deepcopy(obs)
        changed['candidates'][:, :, PRICE_INDEX] += 100.
        with torch.no_grad():
            policy = self.policy()
            torch.testing.assert_close(policy.get_distribution(obs).distribution.probs,
                                       policy.get_distribution(changed).distribution.probs,
                                       rtol=0, atol=0)

    def test_historical_rule_policy_cannot_be_loaded_silently(self):
        for kwargs in (dict(delta=0.), dict(price_weight=0.), dict(reject_bias=0.)):
            with self.assertRaisesRegex(ValueError, 'historical policy'):
                self.policy(**kwargs)

    def test_policy_actor_has_nonzero_gradient(self):
        policy = self.policy()
        loss = -policy.get_distribution(self.observation()).log_prob(torch.tensor([1])).mean()
        loss.backward()
        self.assertGreater(sum(float(x.grad.abs().sum()) for x in policy.actor.parameters()
                               if x.grad is not None), 0.)
        self.assertGreater(sum(float(x.grad.abs().sum()) for x in policy.reject.parameters()
                               if x.grad is not None), 0.)

    def test_critic_loss_does_not_update_actor_features(self):
        policy = self.policy()
        policy.predict_values(self.observation()).sum().backward()
        for module in (policy.encoder, policy.actor, policy.reject):
            self.assertTrue(all(x.grad is None for x in module.parameters()))
        self.assertTrue(any(x.grad is not None for x in policy.critic.parameters()))

    def test_rejection_is_a_learnable_legal_action(self):
        policy = self.policy()
        with torch.no_grad():
            policy.reject[-1].bias.fill_(10.)
            action = policy.get_distribution(self.observation()).get_actions(deterministic=True)
        self.assertEqual(int(action[0]), 0)

    def test_only_reject_is_available_when_all_deployments_masked(self):
        obs = self.observation()
        obs['mask'][:, 1:] = 0
        with torch.no_grad():
            probs = self.policy().get_distribution(obs).distribution.probs
        self.assertTrue(torch.isfinite(probs).all())
        self.assertEqual(float(probs[0, 0]), 1.)

    def test_policy_and_ppo_roundtrip_restore_architecture_and_distribution(self):
        from stable_baselines3 import PPO
        obs = self.observation()
        model = PPO(AllocationPolicy, self.env, policy_kwargs=dict(width=32),
                    n_steps=8, batch_size=4, device='cpu', verbose=0)
        with torch.no_grad():
            model.policy.actor[-1].weight.normal_()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'policy.pth'
            model.policy.save(path)
            policy = AllocationPolicy.load(path, device='cpu')
            model.save(Path(tmp) / 'ppo.zip')
            again = PPO.load(Path(tmp) / 'ppo.zip', device='cpu')
        for candidate in (policy, again.policy):
            self.assertEqual(candidate.width, 32)
            with torch.no_grad():
                torch.testing.assert_close(model.policy.get_distribution(obs).distribution.probs,
                                           candidate.get_distribution(obs).distribution.probs, rtol=0, atol=0)
        self.assertEqual(again.policy_kwargs, dict(width=32))

    def test_hierarchical_joint_probabilities_entropy_and_inference(self):
        reject = torch.tensor([[-.4054651]], requires_grad=True)
        scores = torch.zeros(1, 4, requires_grad=True)
        distribution = HierarchicalCategorical(reject, scores, torch.ones(1, 4, dtype=torch.bool))
        probs = distribution.distribution.probs
        torch.testing.assert_close(probs, torch.tensor([[.4, .15, .15, .15, .15]]), atol=1e-6, rtol=0)
        # Sequential inference accepts (g=.6), whereas global joint MAP rejects.
        self.assertEqual(int(distribution.mode()[0]), 1)
        self.assertEqual(int(probs.argmax(-1)[0]), 0)
        expected_entropy = -(probs * probs.log()).sum(-1)
        torch.testing.assert_close(distribution.entropy(), expected_entropy)
        (-distribution.log_prob(torch.tensor([1]))).backward()
        self.assertGreater(float(reject.grad.abs().sum()), 0.)
        self.assertGreater(float(scores.grad.abs().sum()), 0.)

    def test_hierarchical_admission_not_normalized_by_candidate_count(self):
        for count in (1, 2, 4):
            distribution = HierarchicalCategorical(torch.tensor([[.7]]), torch.zeros(1, count),
                torch.ones(1, count, dtype=torch.bool))
            self.assertAlmostEqual(float(distribution.distribution.probs[:, 1:].sum()),
                                   float(torch.sigmoid(torch.tensor(-.7))), places=6)

    def test_candidate_order_only_permutes_deployment_probabilities(self):
        obs = self.observation()
        policy = self.policy()
        order = torch.tensor([0, 3, 1, 2, 4])
        changed = copy.deepcopy(obs)
        for key in ('candidates', 'needs', 'mask'):
            changed[key] = changed[key][:, order]
        with torch.no_grad():
            a = policy.get_distribution(obs).distribution.probs
            b = policy.get_distribution(changed).distribution.probs
        torch.testing.assert_close(b, a[:, order], rtol=1e-5, atol=1e-6)

    def test_endpoint_order_does_not_change_policy(self):
        obs = self.observation()
        obs['member_mask'][:] = 1
        obs['members'][0, 0, 0] = .8
        obs['members'][0, 1, 1] = -.2
        changed = copy.deepcopy(obs)
        changed['members'] = changed['members'].flip(1)
        policy = self.policy()
        with torch.no_grad():
            torch.testing.assert_close(policy.get_distribution(obs).distribution.probs,
                policy.get_distribution(changed).distribution.probs, rtol=1e-5, atol=1e-6)

    def test_calendar_ablation_removes_all_explicit_calendar_inputs(self):
        obs = self.observation()
        changed = copy.deepcopy(obs)
        changed['state'][:, patrol.STATE_DIM:patrol.STATE_DIM + 2*len(patrol.RESOURCES)] = .7
        changed['resources'][:, :, 1:4] = .8
        changed['needs'][:, :, :, 2:5] = .6
        policy = self.policy(lifecycle=False)
        with torch.no_grad():
            torch.testing.assert_close(policy.get_distribution(obs).distribution.probs,
                policy.get_distribution(changed).distribution.probs, rtol=0, atol=0)

    def test_real_ppo_update_changes_actor_weights(self):
        from stable_baselines3 import PPO
        env = RewardUnit(InitialAllocationEnv(seed=9306))
        try:
            model = PPO(AllocationPolicy, env, learning_rate=3e-4, n_steps=64,
                        batch_size=32, n_epochs=2, seed=9306, device='cpu', verbose=0)
            before = model.policy.actor[-1].weight.detach().clone()
            model.learn(total_timesteps=128)
            self.assertFalse(torch.equal(before, model.policy.actor[-1].weight))
        finally:
            env.close()


class ResourceContractTests(unittest.TestCase):
    def test_mixture_changes_do_not_change_physical_capacity(self):
        configurations = [make_scenario(9300, 64, mix) for mix in
                          ('balanced', 'ip_heavy', 'id_heavy', 'ndn_heavy')]
        for cfg in configurations:
            self.assertEqual(cfg['public'], configurations[0]['public'])
        self.assertGreater(sum(r['mode'] == 'ip' for r in configurations[1]['requests']),
                           sum(r['mode'] == 'ip' for r in configurations[0]['requests']))

    def test_static_quota_shares_total_capacity_but_restricts_each_mode(self):
        public = patrol.public_config()
        shared = patrol.Ledger(public)
        quota = QuotaLedger(public)
        r = patrol.Request(1, 1, 'ip', (5,), 2., 5., .08, .2, 1000, 1, 1)
        vector = np.zeros(len(patrol.RESOURCES))
        vector[0] = shared.capacity[0] / 2
        plan = patrol.Plan(r, 1, 1, 0, [(2., r.drain, vector)], False)
        np.testing.assert_array_equal(shared.capacity, quota.capacity)
        self.assertTrue(shared.feasible(plan))
        self.assertFalse(quota.feasible(plan))
        with self.assertRaises(ValueError):
            QuotaLedger(public, dict(ip=.5, id=.5, ndn=.5))

    def test_shared_cache_cost_counts_only_extra_interval(self):
        public = patrol.public_config()
        ledger = patrol.Ledger(public)
        r1 = patrol.Request(1, 1, 'ndn', (5, 6), 2., 5., 2., .8, 1000, 4, 1, minimum_ratio=1.)
        r2 = patrol.Request(2, 1, 'ndn', (5, 6), 3., 6., 2., .8, 1000, 4, 1, minimum_ratio=1.)
        first = patrol.compile_plan(r1, 1, 1, 1, public, {'cached': []})
        second = patrol.compile_plan(r2, 1, 1, 1, public, {'cached': []})
        ledger.add(first)
        integral, _ = marginal_cost(ledger, second)
        self.assertAlmostEqual(integral[patrol.INDEX['cache:1']],
            r2.size / ledger.capacity[patrol.INDEX['cache:1']] * (r2.drain - r1.drain))


class NativeEnvironmentTests(unittest.TestCase):
    def config(self):
        cfg = patrol.scenario(9303, 3, 'mixed', 4)
        cfg['private']['motions'][0].update(start=10., speed=5.)
        cfg['requests'] = [patrol.Request(i, 1, 'ip', (5,), float(i + 1),
            float(i + 3), .08, .2, 1000, 1, i).json() for i in range(1, 4)]
        return cfg

    def test_future_trajectory_and_future_request_do_not_change_prefix(self):
        a = self.config()
        b = copy.deepcopy(a)
        b['private']['motions'][0]['speed'] = 11.
        b['requests'][-1].update(start=14., stop=17.)
        traces = []
        for cfg in (a, b):
            env = InitialAllocationEnv()
            try:
                obs, _ = env.reset(options=dict(scenario=cfg))
                prefix = []
                for _ in range(2):
                    prefix.append(copy.deepcopy(obs))
                    obs, _, _, _, _ = env.step(patrol.baseline('near_edge', env.unwrapped))
                traces.append(prefix)
            finally:
                env.close()
        for a, b in zip(*traces):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])

    def test_reward_is_native_success_and_encode_is_idempotent(self):
        env = RewardUnit(InitialAllocationEnv())
        try:
            obs, _ = env.reset(options=dict(scenario=self.config()))
            total, done = 0., False
            while not done:
                self.assertTrue(env.observation_space.contains(obs))
                for key, value in env.unwrapped.encode().items():
                    np.testing.assert_array_equal(obs[key], value)
                obs, reward, done, _, _ = env.step(patrol.baseline('near_edge', env.unwrapped))
                total += reward
            self.assertAlmostEqual(total * env.unit, env.unwrapped.result()['success'])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
