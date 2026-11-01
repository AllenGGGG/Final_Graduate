"""New adapter gates: cold-fill accounting, interval atomicity and RL semantics."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import unittest
import numpy as np
from slice_alloc.joint import (Compiler,Request,AtomicLedger,JointEnv,RESOURCES,
                               CAPACITIES,candidates,choose)

PROFILE=dict(passed=True,heldout_passed=True,airtime_seconds_per_pair_round=.004,
             ndn_data_header_bytes=61,ndn_interest_bytes=51)


class JointContractTests(unittest.TestCase):
    def setUp(self):
        self.compiler=Compiler(PROFILE)
        self.request=Request(0,1.,4,('ip','ndn_cold','ndn_pin'))

    def test_uncalibrated_profile_is_rejected(self):
        with self.assertRaises(ValueError):Compiler(dict(PROFILE,heldout_passed=False))

    def test_pin_charges_first_cold_fill_and_reduces_only_later_upstream(self):
        cold=self.compiler.compile(self.request,'ndn_cold')
        pin=self.compiler.compile(self.request,'ndn_pin')
        self.assertEqual(pin.reservation['bytes:origin>relay'],1061)
        self.assertEqual(cold.reservation['bytes:origin>relay'],4*1061)
        for link in ('relay>a','relay>b','a>relay','b>relay'):
            self.assertEqual(pin.reservation['bytes:'+link],cold.reservation['bytes:'+link])
        self.assertEqual(pin.segments[0].usage[0],cold.segments[0].usage[0])
        self.assertEqual(pin.segments[1].usage[0],0.)

    def test_protocols_have_bidirectional_resource_tradeoffs_not_a_global_discount(self):
        ip=self.compiler.compile(self.request,'ip');ndn=self.compiler.compile(self.request,'ndn_cold')
        self.assertGreater(ip.peak[0],ndn.peak[0])
        self.assertLess(ip.peak[2],ndn.peak[2])
        self.assertEqual(ip.peak[1],ndn.peak[1])
        pin=self.compiler.compile(self.request,'ndn_pin')
        self.assertLess(pin.area[0],ndn.area[0])
        self.assertTrue(pin.reservation.new_leases)
        self.assertFalse(ndn.reservation.new_leases)

    def test_each_protocol_remains_feasible_under_its_own_resource_advantage(self):
        state_limited=AtomicLedger(capacities={**CAPACITIES,'forwarding_state':2.})
        bandwidth_limited=AtomicLedger(capacities={**CAPACITIES,'backhaul':100000.})
        ip=self.compiler.compile(self.request,'ip');ndn=self.compiler.compile(self.request,'ndn_cold')
        self.assertTrue(state_limited.feasible(ip));self.assertFalse(state_limited.feasible(ndn))
        self.assertFalse(bandwidth_limited.feasible(ip));self.assertTrue(bandwidth_limited.feasible(ndn))

    def test_ip_request_and_response_network_bytes_are_both_charged(self):
        p=self.compiler.compile(self.request,'ip')
        self.assertEqual(p.reservation['bytes:origin>relay'],4*2*1032)
        self.assertEqual(p.reservation['bytes:relay>origin'],4*2*(len(self.request.name)+32))

    def test_fixed_interface_cannot_be_substituted(self):
        with self.assertRaises(ValueError):self.compiler.compile(replace(self.request,allowed=('ip',)),'ndn_pin')

    def test_failed_resource_commit_has_no_cache_side_effect(self):
        ledger=AtomicLedger(capacities={**CAPACITIES,'backhaul':1.})
        p=self.compiler.compile(self.request,'ndn_pin')
        with self.assertRaises(ValueError):ledger.commit(p)
        self.assertEqual(ledger.version,0);self.assertFalse(ledger.active);self.assertFalse(ledger.cache.leases)

    def test_failed_cache_commit_has_no_resource_side_effect(self):
        ledger=AtomicLedger(cache_capacities={'relay':999})
        with self.assertRaises(ValueError):ledger.commit(self.compiler.compile(self.request,'ndn_pin'))
        self.assertFalse(ledger.active);self.assertFalse(ledger.cache.leases);self.assertEqual(ledger.version,0)

    def test_stale_snapshot_is_rejected_and_release_is_complete(self):
        ledger=AtomicLedger();p=self.compiler.compile(self.request,'ndn_pin')
        ledger.commit(p,expected_version=0)
        other=self.compiler.compile(replace(self.request,id=1),'ndn_pin')
        with self.assertRaises(ValueError):ledger.commit(other,expected_version=0)
        self.assertEqual(len(ledger.active),1);self.assertEqual(ledger.cache.usage(1.1),{'relay':1000})
        ledger.release('0');self.assertFalse(ledger.active);self.assertFalse(ledger.cache.leases)

    def test_concurrent_same_version_commits_cannot_both_succeed(self):
        ledger=AtomicLedger();barrier=Barrier(2)
        def commit(i):
            p=self.compiler.compile(replace(self.request,id=i),'ndn_pin');barrier.wait()
            try:ledger.commit(p,expected_version=0);return True
            except ValueError:return False
        with ThreadPoolExecutor(2) as pool:out=list(pool.map(commit,(0,1)))
        self.assertEqual(sum(out),1);self.assertEqual(len(ledger.active),1)
        self.assertEqual(sum(ledger.cache.usage(1.1).values()),1000)

    def test_interval_check_catches_later_conflict_not_only_current_occupancy(self):
        r=replace(self.request,arrival=2.,allowed=('ip',))
        ledger=AtomicLedger(capacities={**CAPACITIES,'backhaul':180000.})
        ledger.commit(self.compiler.compile(r,'ip'))
        self.assertTrue(np.all(ledger.used(1.)==0))
        crossing=replace(r,id=1,arrival=1.9)
        self.assertFalse(ledger.feasible(self.compiler.compile(crossing,'ip')))

    def test_no_future_request_information_in_observation(self):
        a=JointEnv(PROFILE);b=JointEnv(PROFILE)
        future=replace(self.request,id=1,arrival=2.)
        oa,_=a.reset(options={'requests':(self.request,future)})
        ob,_=b.reset(options={'requests':(self.request,replace(future,rounds=1,allowed=('ip',)))})
        for key in oa:np.testing.assert_array_equal(oa[key],ob[key])

    def test_reward_is_delayed_then_settled_once_and_ledger_drains(self):
        env=JointEnv(PROFILE,count=12);obs,_=env.reset(seed=71000);total=0.
        while not env.done:
            action=choose(env);obs,reward,done,_,_=env.step(action);total+=reward
            self.assertTrue(env.observation_space.contains(obs))
        accepted=sum(p is not None for _,p in env.decisions)
        self.assertEqual(total,accepted);self.assertEqual(env.settled,accepted)
        self.assertFalse(env.ledger.active);self.assertFalse(env.ledger.cache.leases)
        np.testing.assert_array_equal(obs['mask'],[1,0,0,0])

    def test_padded_actions_are_rejected(self):
        env=JointEnv(PROFILE);env.reset(options={'requests':(replace(self.request,allowed=('ip',)),)})
        with self.assertRaises(ValueError):env.step(2)


class JointLearningInterfaceTests(unittest.TestCase):
    def test_candidate_permutation_preserves_q_and_mask(self):
        import torch
        from slice_alloc.src.qlearning import DoubleDQN,tensor_observation
        torch.set_num_threads(1)
        env=JointEnv(PROFILE);obs,_=env.reset(seed=1010)
        agent=DoubleDQN(env.state_dim,env.candidate_dim,11)
        original=tensor_observation(obs);perm=np.array([2,0,3,1])
        reordered={k:v.clone() for k,v in original.items()}
        for key in ('candidates','mask'):reordered[key]=reordered[key][:,perm]
        torch.testing.assert_close(agent.network(reordered),agent.network(original)[:,perm])
        self.assertTrue(torch.all(agent.network(original)[original['mask']==0]==-1e9))

    def test_ddqn_update_changes_weights_and_restores_identical_decision(self):
        import torch,tempfile
        from pathlib import Path
        from slice_alloc.src.qlearning import DoubleDQN
        torch.set_num_threads(1)
        env=JointEnv(PROFILE,count=8);obs,_=env.reset(seed=1001)
        agent=DoubleDQN(env.state_dim,env.candidate_dim,11)
        before=[p.detach().clone() for p in agent.network.parameters()]
        for _ in range(32):
            action=agent.explore(obs,1.);nxt,r,done,_,_=env.step(action)
            agent.remember(obs,action,r,nxt,done);obs=env.reset()[0] if done else nxt
        self.assertTrue(np.isfinite(agent.update(16)))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,agent.network.parameters())))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';agent.save(path);restored=DoubleDQN.load(path)
            self.assertEqual(agent.predict(obs)[0],restored.predict(obs)[0])

    def test_ppo_policy_permutation_and_mask_on_new_contracts(self):
        import torch
        from stable_baselines3 import PPO
        from slice_alloc.src.policy import CandidatePolicy
        from slice_alloc.src.qlearning import tensor_observation
        torch.set_num_threads(1)
        env=JointEnv(PROFILE,count=8);obs,_=env.reset(seed=1002)
        agent=PPO(CandidatePolicy,env,n_steps=32,batch_size=16,n_epochs=1,gamma=1.,seed=11,device='cpu')
        original=tensor_observation(obs);perm=np.array([3,1,0,2]);changed={k:v.clone() for k,v in original.items()}
        for key in ('candidates','mask'):changed[key]=changed[key][:,perm]
        a=agent.policy.get_distribution(original).distribution.probs
        b=agent.policy.get_distribution(changed).distribution.probs
        torch.testing.assert_close(b,a[:,perm]);self.assertTrue(torch.all(a[original['mask']==0]==0))
        agent.learn(64)
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.policy.parameters()))


if __name__=='__main__':unittest.main()
