"""Read-only instrumentation, exact catalogue, counterfactual and rollout gates."""
from dataclasses import replace
from itertools import product
import unittest
from slice_alloc.joint import AtomicLedger,Compiler,JointEnv,Request,CAPACITIES,candidates
from slice_alloc.tools.joint_diagnose import clone_ledger,decision_diagnostic,plan_slack,solve_reference
from slice_alloc.tools.joint_rollout import sampled_action
from slice_alloc.tools.joint_reject_timing import summarize_agreement

PROFILE=dict(passed=True,heldout_passed=True,airtime_seconds_per_pair_round=.004,
             ndn_data_header_bytes=61,ndn_interest_bytes=51)


class AllocationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.compiler=Compiler(PROFILE)
        self.request=Request(0,1.,4,('ip','ndn_cold','ndn_pin'))

    def test_rejection_agreement_pools_actual_decision_denominators(self):
        rows=[dict(case='a',id=0,method='ddqn_11',rollout_agrees_with_rejection=True)]
        rows += [dict(case='b',id=i,method='ddqn_22',rollout_agrees_with_rejection=False) for i in range(3)]
        summary,total=summarize_agreement(rows,{'ddqn_11':.8,'ddqn_22':.6})
        self.assertEqual(total['policy_rejections'],4)
        self.assertEqual(total['rollout_agreement_rate'],.25)
        self.assertEqual(total['rollout_disagreement_rate'],.75)
        self.assertAlmostEqual(total['pearson_r_across_initializations'],1.)
        self.assertEqual(summary['ddqn_22']['policy_rejections'],3)
        with self.assertRaises(ValueError):summarize_agreement(rows+rows[:1],{'ddqn_11':.8,'ddqn_22':.6})

    def test_rejection_agreement_leaves_undefined_statistics_unset(self):
        summary,total=summarize_agreement([],{'ddqn_11':.8})
        self.assertIsNone(summary['ddqn_11']['rollout_agreement_rate'])
        self.assertIsNone(total['rollout_agreement_rate'])
        self.assertIsNone(total['pearson_r_across_initializations'])

    def test_counterfactual_slack_identifies_resource_and_cache_failures(self):
        p=self.compiler.compile(self.request,'ndn_pin')
        d=plan_slack(AtomicLedger(capacities={**CAPACITIES,'forwarding_state':2.},cache_capacities={'relay':999}),p)
        self.assertFalse(d['feasible'])
        self.assertIn('forwarding_state',d['blocking_dimensions']);self.assertIn('cache_payload_bytes',d['blocking_dimensions'])

    def test_instrumentation_distinguishes_policy_reject_from_missing_candidates(self):
        env=JointEnv(PROFILE);env.reset(options={'requests':(self.request,)})
        d=decision_diagnostic(env,0)
        self.assertEqual(d['feasible_candidates'],3);self.assertEqual(d['enumeration_status'],'exhaustive')
        self.assertEqual(d['rejection_reason'],'policy')
        env.ledger=AtomicLedger(capacities={**CAPACITIES,'forwarding_state':1.})
        env.plans=candidates(env.compiler,env.ledger,self.request)
        d=decision_diagnostic(env,0)
        self.assertEqual(d['enumeration_status'],'infeasible');self.assertEqual(d['rejection_reason'],'no_candidate')

    def test_catalog_omission_is_detected(self):
        env=JointEnv(PROFILE);env.reset(options={'requests':(self.request,)})
        env.plans.pop()
        with self.assertRaises(ValueError):decision_diagnostic(env,0)

    def test_cloned_ledger_is_independent_and_preserves_immutable_contracts(self):
        ledger=AtomicLedger();ledger.commit(self.compiler.compile(self.request,'ndn_pin'))
        other=clone_ledger(ledger);other.release('0')
        self.assertFalse(other.active);self.assertTrue(ledger.active)
        self.assertEqual(ledger.cache.usage(1.1),{'relay':1000})

    def test_exact_reference_matches_exhaustive_small_instance(self):
        requests=tuple(replace(self.request,id=i,arrival=1.+i*.01,allowed=('ndn_cold','ndn_pin')) for i in range(3))
        best=0
        for actions in product((None,'ndn_cold','ndn_pin'),repeat=3):
            ledger=AtomicLedger();count=0
            try:
                for r,k in zip(requests,actions):
                    if k:ledger.commit(self.compiler.compile(r,k));count+=1
            except ValueError:continue
            best=max(best,count)
        result=solve_reference(requests,self.compiler)
        self.assertEqual(result['objective'],best);self.assertTrue(result['witness_interval_and_cache_check'])

    def test_sampled_rollout_does_not_read_real_future_requests_or_modify_current_state(self):
        a=JointEnv(PROFILE,count=2);b=JointEnv(PROFILE,count=2)
        tail=replace(self.request,id=1,arrival=2.)
        a.reset(options={'requests':(self.request,tail)})
        b.reset(options={'requests':(self.request,replace(tail,arrival=10.,rounds=2,allowed=('ip',)))})
        self.assertEqual(sampled_action(a,horizon=2,samples=2),sampled_action(b,horizon=2,samples=2))
        self.assertFalse(a.ledger.active);self.assertFalse(a.ledger.cache.leases);self.assertEqual(a.ledger.version,0)


if __name__=='__main__':unittest.main()
