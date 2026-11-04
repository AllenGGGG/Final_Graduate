"""A repeated verification must retain historical-replication disclosure."""
import json
from pathlib import Path
import tempfile
import unittest
from slice_alloc.__main__ import finalize_output


class FinalProvenanceTests(unittest.TestCase):
    def test_verify_retains_recovery_and_denominators(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            def write(name,value):
                (out/name).write_text(json.dumps(value))
            write('verification.json',{'fresh_test_complete':True})
            write('recovery_verification.json',dict(source_commit='original',all_actual_models_loaded=True,
                verified_original_episodes=480,original_statistics_reproduced_exactly=True))
            row=dict(success_rate=.8,violation_rate_all=.1,admitted_violations=1,admitted=9)
            write('analysis.json',dict(comparisons={'double_dqn-reference_dqn':{'difference':.002}},
                methods={m:row for m in ('double_dqn','greedy','nearest','first_fit')}))
            (out/'report.md').write_text('60个新测试种子\n')
            finalize_output(out)
            first=(out/'report.md').read_text()
            finalize_output(out)
            self.assertEqual(first,(out/'report.md').read_text())
            self.assertIn('违约／已接纳',first)
            self.assertIn('不是新增独立测试',first)
            v=json.loads((out/'verification.json').read_text())
            self.assertFalse(v['fresh_test_complete'])
            self.assertTrue(v['historical_reproduction'])
            self.assertEqual(v['separately_verified_original_episodes'],480)


if __name__=='__main__':unittest.main()
