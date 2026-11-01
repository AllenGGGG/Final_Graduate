"""Statistical contracts: paired scenes are clusters, not repeated model rows."""
import unittest

from slice_alloc.tools.modal_binding_rl_report import paired


def rows(inits, increment=0):
    return [dict(init=i, seed=s, load=l, requests=l, success=l//2+increment,
        admitted=l, resource_time=[1.]*17, wire_bytes=100, origin_wire_bytes=10,
        phy_tx_seconds=.1, decision_ms=1., modes={'ip': l},
        by_service={k: dict(success=0, requests=0) for k in ('telemetry', 'object', 'command')})
        for i in inits for s in (1, 2, 3) for l in (48, 80)]


class PairedStatisticsTests(unittest.TestCase):
    def test_identical_paired_policies_have_zero_difference_and_interval(self):
        result = paired(rows([11, 23, 37, 53, 71]), rows([11, 23, 37]), samples=100)
        self.assertEqual(result['difference'], 0.)
        self.assertEqual(result['ci95'], [0., 0.])
        self.assertEqual(result['left_initializations'], [11, 23, 37])
        self.assertEqual(result['clusters'], 3)

    def test_two_loads_keep_the_correct_denominator_and_known_gain(self):
        result = paired(rows([11, 23, 37], increment=1), rows([0]), samples=100)
        # Two extra successful requests per seed out of 48+80 arrivals.
        self.assertAlmostEqual(result['difference'], 2/128)
        for bound in result['ci95']:
            self.assertAlmostEqual(bound, 2/128)
        self.assertFalse(result['matched_initializations'])


if __name__ == '__main__':
    unittest.main()
