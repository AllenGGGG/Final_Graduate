"""Regression checks for the supplementary study; frozen primary tests stay intact."""
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from slice_alloc.tools import allocation_greedy_improve as study
from slice_alloc.tools import allocation_strong_baseline_supplement as rules


class GreedyStudyTests(unittest.TestCase):
    def test_modal_threshold_can_reject_the_selected_feasible_plan(self):
        env=SimpleNamespace(options=[object(),object()],ledger=object())
        with mock.patch.object(rules,'modal_greedy_costs',return_value=np.array([2.,1.])), \
             mock.patch.object(rules.b,'exponential_price',return_value=3.):
            self.assertEqual(rules.rule_action('modal_price_threshold',env,threshold=2.),0)
            self.assertEqual(rules.rule_action('modal_greedy',env),2)

    def test_exponential_price_remains_distinct_from_modal_admission_rule(self):
        env=SimpleNamespace(options=[object()])
        with mock.patch.object(rules.b,'baseline',return_value=1) as baseline:
            self.assertEqual(rules.rule_action('exponential_price',env),1)
            self.assertEqual(baseline.call_args.args[0],'joint_price')
            self.assertTrue(np.isinf(baseline.call_args.kwargs['threshold']))

    def test_selection_uses_all_arrivals_not_admission_rate(self):
        configs=[study.configuration('a'),study.configuration('b')]
        rows=[dict(method='a',success=6,requests=10,admitted=6),
              dict(method='b',success=7,requests=10,admitted=10)]
        self.assertEqual(study.choose(configs,rows)['name'],'b')

    def test_complete_coverage_rejects_missing_and_duplicate_cases(self):
        with mock.patch.object(study,'LOADS',(10,)):
            row=dict(method='a',seed=1,load=10,requests=10,success=6,
                     admitted=7,forced_reject=2,voluntary_reject=1,admitted_violations=1)
            study.validate_rows([row],['a'],[1])
            for invalid in ([],[row,row],[{**row,'success':5}]):
                with self.assertRaises(ValueError):study.validate_rows(invalid,['a'],[1])

    def test_splits_do_not_overlap_each_other_or_previous_tests(self):
        splits=[set(study.DEV),set(study.VALIDATION),set(study.TEST)]
        for i,left in enumerate(splits):
            self.assertFalse(left.intersection(range(447000,447060)))
            for right in splits[i+1:]:self.assertFalse(left.intersection(right))


if __name__=='__main__':unittest.main()
