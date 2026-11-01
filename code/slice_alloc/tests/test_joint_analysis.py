"""Check denominators and paired scenario identity in post-evaluation analysis."""
from copy import deepcopy
import unittest
from slice_alloc.tools.joint_analyze import failure_counts,reference_comparison


class OutcomeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(seed=1,load=1.,method='ddqn_11',input_sha256='same',requests=4,
                      accepted=3,packet_success=.5,rejected_no_candidate=0,policy_rejected=1)
        self.ref=dict(seed=1,load=1.,input_sha256='same',requests=4,objective=4,optimal=True)

    def test_failure_causes_partition_with_a_common_request_denominator(self):
        r=failure_counts([self.row])
        self.assertEqual(r['failed'],2)
        self.assertEqual(r['admitted_violation_request_rate'],.25)
        self.assertEqual(r['admitted_success_rate'],2/3)
        self.assertEqual(r['no_candidate_rate']+r['policy_rejected_rate']+
                         r['admitted_violation_request_rate']+r['success_rate'],1.)

    def test_invalid_counters_and_noninteger_success_are_rejected(self):
        for changes in ({'policy_rejected':2},{'packet_success':.49},{'packet_success':1.}):
            with self.assertRaises(ValueError):failure_counts([{**self.row,**changes}])

    def test_initializations_repeat_reference_weight_without_extra_scenarios(self):
        other={**self.row,'method':'ddqn_22'}
        r=reference_comparison([self.row,other],[self.ref])
        self.assertEqual(r['scenarios'],1);self.assertEqual(r['policy_runs'],2)
        self.assertEqual(r['reference_model_count_with_policy_repetition'],8)
        self.assertEqual(r['packet_fraction_of_offline_model'],.5)

    def test_input_mismatch_and_incomplete_coverage_are_rejected(self):
        with self.assertRaises(ValueError):reference_comparison([{**self.row,'input_sha256':'other'}],[self.ref])
        with self.assertRaises(ValueError):reference_comparison([self.row],[self.ref,{**self.ref,'seed':2}])

    def test_unproven_or_duplicate_or_inconsistent_oracles_are_rejected(self):
        with self.assertRaises(ValueError):reference_comparison([self.row],[{**self.ref,'optimal':False}])
        with self.assertRaises(ValueError):reference_comparison([self.row],[self.ref,self.ref])
        with self.assertRaises(ValueError):reference_comparison([self.row],[{**self.ref,'objective':2}])
        with self.assertRaises(ValueError):reference_comparison([self.row,deepcopy(self.row)],[self.ref])


if __name__=='__main__':unittest.main()
