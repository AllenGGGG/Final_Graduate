"""The public entry point must include supplemental evaluation and reporting."""
import sys
import unittest
from unittest.mock import patch

from slice_alloc import __main__ as entry
from slice_alloc.tools import allocation_final_pipeline as pipeline
from slice_alloc.tools import allocation_final_run as run


class EntrypointTests(unittest.TestCase):
    def test_all_runs_complete_pipeline_and_preserves_options(self):
        with patch.object(sys, 'argv', ['slice_alloc', 'all', '--workers', '2']), \
             patch.object(pipeline, 'main') as complete, patch.object(run, 'main') as partial:
            entry.main()
            complete.assert_called_once_with()
            partial.assert_not_called()
            self.assertEqual(sys.argv, ['slice_alloc', '--workers', '2'])

    def test_default_runs_complete_pipeline(self):
        with patch.object(sys, 'argv', ['slice_alloc']), patch.object(pipeline, 'main') as complete:
            entry.main()
            complete.assert_called_once_with()

    def test_independent_evaluation_remains_available(self):
        with patch.object(sys, 'argv', ['slice_alloc', 'evaluate', '--workers', '1']), \
             patch.object(run, 'main') as partial, patch.object(pipeline, 'main') as complete:
            entry.main()
            partial.assert_called_once_with()
            complete.assert_not_called()
            self.assertEqual(sys.argv[1], 'evaluate')

    def test_help_does_not_start_a_partial_experiment(self):
        with patch.object(sys, 'argv', ['slice_alloc', '--help']), \
             patch.object(pipeline, 'main') as complete, patch.object(run, 'main') as partial:
            entry.main()
            complete.assert_called_once_with()
            partial.assert_not_called()


if __name__ == '__main__':
    unittest.main()
