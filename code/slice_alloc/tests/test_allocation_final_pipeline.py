"""Regression checks for failed validation, frozen config and formal entry points."""
import copy
from contextlib import ExitStack
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from slice_alloc.tools import allocation_final_run as run
from slice_alloc.tools import allocation_final_pipeline as pipeline
from slice_alloc.tools import allocation_final_stress as stress


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(run, 'OUT', self.out))
        run.freeze()

    def test_failed_tests_are_retried_and_failure_log_survives(self):
        failure = subprocess.CalledProcessError(1, ['unittest'])
        with patch.object(pipeline.subprocess, 'run', side_effect=[failure, None]) as execute:
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.ensure_tests()
            self.assertFalse((self.out/'tests_complete.json').exists())
            failed_log = self.out/'tests-attempt-0001.log'
            self.assertTrue(failed_log.exists())
            pipeline.ensure_tests()
            pipeline.ensure_tests()
            self.assertEqual(execute.call_count, 2)
            self.assertTrue(failed_log.exists())
            self.assertTrue((self.out/'tests-attempt-0002.log').exists())
            self.assertTrue((self.out/'tests_complete.json').exists())

    def test_interruption_does_not_mark_tests_complete(self):
        with patch.object(pipeline.subprocess, 'run', side_effect=[KeyboardInterrupt, None]) as execute:
            with self.assertRaises(KeyboardInterrupt):
                pipeline.ensure_tests()
            self.assertFalse((self.out/'tests_complete.json').exists())
            pipeline.ensure_tests()
            self.assertEqual(execute.call_count, 2)

    def test_existing_log_without_success_marker_cannot_skip_tests(self):
        (self.out/'tests.log').write_text('FAILED')
        with patch.object(pipeline.subprocess, 'run') as execute:
            pipeline.ensure_tests()
            execute.assert_called_once()

    def test_modified_validation_log_requires_revalidation(self):
        with patch.object(pipeline.subprocess, 'run') as execute:
            pipeline.ensure_tests()
            (self.out/'tests.log').write_text('changed')
            pipeline.ensure_tests()
            self.assertEqual(execute.call_count, 2)

    def test_actual_qrdqn_learning_rate_change_is_rejected(self):
        with patch.dict(run.old.HYPER, learning_rate=.123):
            with self.assertRaisesRegex(AssertionError, 'Frozen sources/protocol changed'):
                run.freeze()

    def test_nested_training_parameters_are_frozen(self):
        modified = copy.deepcopy(run.old.HYPER)
        modified['replay_buffer_kwargs']['horizon'] = 99
        with patch.object(run.old, 'HYPER', modified):
            with self.assertRaises(AssertionError):
                run.freeze()

    def test_training_dependency_source_is_frozen(self):
        self.assertIn('tools/modal_binding_run.py', run.SOURCES)
        actual_sha = run.sha
        def changed_source(path):
            return 'changed' if Path(path).name == 'modal_binding_run.py' else actual_sha(path)
        with patch.object(run, 'sha', side_effect=changed_source):
            with self.assertRaises(AssertionError):
                run.freeze()

    def test_fresh_protocol_records_actual_config_and_class_names(self):
        new_out = self.out/'new'
        with patch.object(run, 'OUT', new_out), patch.dict(run.old.HYPER, learning_rate=.123):
            protocol = run.freeze()
            self.assertEqual(protocol['qrdqn']['learning_rate'], .123)
            self.assertEqual(protocol['qrdqn']['replay_buffer_kwargs']['horizon'],
                             run.training_parameters('double_qrdqn')['replay_buffer_kwargs']['horizon'])
            self.assertTrue(protocol['qrdqn']['replay_buffer_class'].endswith('.NStepDictReplay'))

    def test_changed_config_cannot_reuse_cached_training(self):
        folder = self.out/'training/double_qrdqn/full/11'
        folder.mkdir(parents=True)
        run.write(folder/'training.json', {'sentinel': 'stale'})
        with patch.dict(run.old.HYPER, learning_rate=.123), patch.object(run, 'environment') as env:
            with self.assertRaises(AssertionError):
                run.train(('double_qrdqn', 'full', 11))
            env.assert_not_called()

    def training_result(self, spec):
        algorithm, variant, init = spec
        result = dict(algorithm=algorithm, variant=variant, init=init,
            selected={'validation': {'rate': .8 if algorithm=='candidate_ppo' else .7}})
        if variant != 'full':
            run.write(self.out/'training'/algorithm/variant/str(init)/'training.json', result)
        return result

    def fake_pool(self, function, specs, workers):
        if function is run.train:
            return [self.training_result(spec) for spec in specs]
        if function is run.tune_rule:
            for method in specs:
                run.write(self.out/'rules'/f'{method}.json', {'chosen': {'threshold': None}})
            return []
        if function is run.evaluate_group:
            started = (self.out/'test_started.json').stat().st_mtime_ns
            for name in ('stress_protocol.json', 'auxiliary_sources.json', 'deployment.json', 'tests_complete.json'):
                self.assertTrue((self.out/name).exists())
                self.assertLessEqual((self.out/name).stat().st_mtime_ns, started)
            return [[] for _ in specs]
        raise AssertionError(function)

    def invoke(self, command):
        with patch.object(sys, 'argv', ['allocation_final_run', command, '--workers', '1']), \
             patch.object(run, 'pool_map', side_effect=self.fake_pool), \
             patch.object(pipeline.subprocess, 'run'):
            run.main()

    def assert_full_pipeline_can_resume(self):
        # Main evaluation is cached; the full pipeline must still reach stress
        # evaluation and the final report without late-registration failures.
        run.write(self.out/'environment.json', {})
        run.write(self.out/'calibration.json', {})
        commands = []
        with patch.object(sys, 'argv', ['allocation_final_pipeline']), \
             patch.object(pipeline, 'command', side_effect=lambda module,*args: commands.append(module)):
            pipeline.main()
        self.assertEqual(commands, ['slice_alloc.tools.allocation_final_stress',
                                    'slice_alloc.tools.allocation_final_report'])

    def test_all_registers_before_testing_and_full_pipeline_resumes(self):
        self.invoke('all')
        self.assert_full_pipeline_can_resume()

    def test_train_then_independent_evaluate_allows_pipeline_resume(self):
        self.invoke('train')
        self.assertFalse((self.out/'test_started.json').exists())
        self.invoke('evaluate')
        self.assert_full_pipeline_can_resume()

    def test_missing_registration_after_test_is_not_backdated(self):
        run.write(self.out/'test_started.json', {})
        with self.assertRaisesRegex(AssertionError, 'preregister'):
            stress.freeze()

    def test_prepare_evaluation_is_idempotent_after_test_start(self):
        run.write(self.out/'selection.json', {'selected': 'candidate_ppo'})
        with patch.object(pipeline.subprocess, 'run'):
            pipeline.prepare_evaluation()
            paths = ['stress_protocol.json', 'auxiliary_sources.json', 'deployment.json']
            stamps = {p:(self.out/p).stat().st_mtime_ns for p in paths}
            run.write(self.out/'test_started.json', {})
            pipeline.prepare_evaluation()
            self.assertEqual(stamps, {p:(self.out/p).stat().st_mtime_ns for p in paths})


if __name__ == '__main__':
    unittest.main()
