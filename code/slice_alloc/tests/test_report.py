"""Prevent stale prose from overriding current results and decision denominators."""
import unittest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch
from slice_alloc.tools.report import current_summary, decision_count


class ReportEvidenceTests(unittest.TestCase):
    def test_summary_uses_supplied_means_gains_and_sample_deviations(self):
        evidence = dict(summary={
            'ddqn': dict(mean=.8123, sample_std=.01234),
            'ppo': dict(mean=.8456, sample_std=.00987)},
            paired={'ddqn-adaptive_price': dict(mean=.0212),
                    'ppo-adaptive_price': dict(mean=.0545)})
        text = current_summary(evidence)
        for token in ('81.23%/84.56%', '2.12/5.45', '1.234/0.987'):
            self.assertIn(token, text)
        self.assertNotIn('68.71', text)

    def test_decision_denominator_counts_all_decisions_of_the_requested_family(self):
        rows = [dict(method='ddqn_11', requests=3, decisions=[{}, {}, {}]),
                dict(method='ddqn_22', requests=2, decisions=[{}, {}]),
                dict(method='ppo_11', requests=7, decisions=[{}] * 7)]
        self.assertEqual(decision_count(rows, 'ddqn_'), 5)

    def test_missing_decision_records_cannot_be_reported_as_a_full_replay(self):
        with self.assertRaises(ValueError):
            decision_count([dict(method='ddqn_11', requests=3, decisions=[{}])], 'ddqn_')

    def test_standalone_word_build_does_not_overwrite_manuscript(self):
        from slice_alloc.tools import export_document as exporter
        with tempfile.TemporaryDirectory() as tmp:
            thesis=Path(tmp)/'thesis.md';thesis.with_suffix('.docx').write_bytes(b'existing thesis')
            with patch.object(exporter,'MD',thesis):
                document=exporter.to_docx('# 独立实验报告\n\n成功率77.00%。')
            self.assertEqual(thesis.with_suffix('.docx').read_bytes(),b'existing thesis')
            target=Path(tmp)/'report.docx';document.save(target)
            with zipfile.ZipFile(target) as archive:
                self.assertIn('77.00%',archive.read('word/document.xml').decode())

    def test_explicit_manuscript_export_retains_required_content_check(self):
        from slice_alloc.tools.export_document import to_docx
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'Word missing required content'):
                to_docx('# Incomplete manuscript',destination=Path(tmp)/'thesis.docx',validate_manuscript=True)


class PpoPresentationTests(unittest.TestCase):
    def evidence(self):
        from slice_alloc.tools.allocation_publish import LABELS
        stats={m:dict(success_rate=.76,by_load={'64':.79,'80':.74},admission_rate=.83,
                      initialization_std=.003,initialization_rates=[.75,.77]) for m in LABELS}
        comparison=dict(difference=.06,paired_scenario_ci95=[.04,.08],
                        initialization_and_scenario_ci95=[.03,.09])
        analysis=dict(stats=stats,comparisons={'prior_ppo':{m:comparison for m in LABELS if m!='prior_ppo'}},
                      episodes=1260,decisions=90720)
        protocol=dict(primary_method='prior_ppo',baselines=['price','lifetime_price'],
                      development_training_budget=dict(models=44,native_decisions=1441792,reference_decisions=327904))
        evaluation=dict(rows=[dict(method='prior_ppo',requests=3,admitted=3,success=3,
                                  by_mode={m:dict(requests=1,admitted=1,success=1) for m in ('ip','id','ndn')})])
        return analysis,protocol,evaluation

    def test_one_proposed_ppo_and_plain_ablation_in_main_report(self):
        from slice_alloc.tools.allocation_publish import report_text
        text=report_text(*self.evidence());main=text.split('## 附录：')[0]
        self.assertEqual(main.count('| 本方法：资源先验PPO |'),1)
        self.assertIn('| 普通PPO（消融） |',main)
        self.assertNotIn('normalized_plain_ppo',main)
        self.assertIn('原始',main)
        self.assertIn('尚未证明优于最强规则',main)

    def test_full_comparisons_and_costs_are_retained_in_appendix(self):
        from slice_alloc.tools.allocation_publish import report_text
        appendix=report_text(*self.evidence()).split('## 附录：')[1]
        for token in ('normalized_plain_ppo','near_edge','cache_aware','1441792','327904'):
            self.assertIn(token,appendix)

    def test_relative_links_are_rebased_for_root_manuscript(self):
        from slice_alloc.tools.allocation_publish import root_links,ROOT
        text='[设计](../../design.md) · [结果](analysis.json) · [外部](https://example.org)'
        converted=root_links(text,ROOT/'slice_alloc/results/allocation_v3')
        self.assertIn('](slice_alloc/design.md)',converted)
        self.assertIn('](slice_alloc/results/allocation_v3/analysis.json)',converted)
        self.assertIn('](https://example.org)',converted)


if __name__ == '__main__':
    unittest.main()
