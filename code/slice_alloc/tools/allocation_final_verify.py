"""Real subprocess and native integration verification, never a performance test."""
import argparse
import copy
from contextlib import ExitStack
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

from slice_alloc.tools import allocation_final_pipeline as pipeline
from slice_alloc.tools import allocation_final_report as report
from slice_alloc.tools import allocation_final_run as run
from slice_alloc.tools import allocation_final_stress as stress


def retry_probe(destination):
    with tempfile.TemporaryDirectory(prefix='lowalt-retry-') as directory:
        root = Path(directory)
        tests = root/'slice_alloc/tests'
        tests.mkdir(parents=True)
        (tests/'test_probe.py').write_text(
            'import unittest\nfrom pathlib import Path\n'
            'class Probe(unittest.TestCase):\n'
            '    def test_gate(self):\n'
            "        self.assertTrue(Path('allow_pass').exists())\n")
        out = root/'output'
        run.write(out/'protocol.json', {'probe': 'real subprocess retry'})
        with patch.object(run, 'ROOT', root), patch.object(run, 'OUT', out):
            try:
                pipeline.ensure_tests()
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError('The first real test process must fail')
            assert not (out/'tests_complete.json').exists()
            failed = (out/'tests-attempt-0001.log').read_bytes()
            (root/'allow_pass').touch()
            pipeline.ensure_tests()
            assert (out/'tests-attempt-0001.log').read_bytes() == failed
            assert (out/'tests-attempt-0002.log').exists()
            with patch.object(pipeline.subprocess, 'run', side_effect=AssertionError('cache was not reused')):
                pipeline.ensure_tests()
            shutil.copytree(out, destination)
    return dict(failure_retried=True, failed_log_preserved=True, successful_validation_cached=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=run.ROOT/'slice_alloc/results/final_validation')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('Use a new verification output directory')
    output.mkdir(parents=True)
    retry = retry_probe(output/'retry_probe')
    hyper = copy.deepcopy(run.old.HYPER)
    hyper.update(buffer_size=128, learning_starts=16, batch_size=16, exploration_steps=64)
    hyper['replay_buffer_kwargs']['horizon'] = 3
    ppo = dict(run.PPO_HYPER, n_steps=64, batch_size=32, n_epochs=1)
    with ExitStack() as stack:
        for name, value in dict(OUT=output/'experiment', INITS=(11,), LOADS=(4, 8),
                                CHECKPOINTS=(64, 128), VALIDATION=(49000,),
                                TEST=(49001,), CONTROL=(49002,), PPO_HYPER=ppo).items():
            stack.enter_context(patch.object(run, name, value))
        stack.enter_context(patch.object(run.old, 'HYPER', hyper))
        stack.enter_context(patch.object(stress, 'SEEDS', (49100,)))
        stack.enter_context(patch.object(sys, 'argv', ['allocation_final_run', 'all', '--workers', '1']))
        run.main()
        names = ('stress_protocol.json', 'auxiliary_sources.json', 'deployment.json')
        before = {name: (run.OUT/name).stat().st_mtime_ns for name in names}
        pipeline.prepare_evaluation()
        assert before == {name: (run.OUT/name).stat().st_mtime_ns for name in names}
        sys.argv = ['allocation_final_stress']
        stress.main()
        rows = run.read(run.OUT/'evaluation.json') + run.read(run.OUT/'stress_evaluation.json')
        assert len(rows) == 37
        assert all(report.audit_saved(row) for row in rows)
        records = [run.read(path) for path in sorted((run.OUT/'training').rglob('training.json'))]
        assert len(records) == 5
        replays = []
        for record in records:
            algorithm = record['algorithm']
            model = run.load_model(algorithm, run.OUT/record['model'])
            expected = run.training_parameters(algorithm)
            assert model.learning_rate == expected['learning_rate']
            assert model.batch_size == expected['batch_size']
            assert record['protocol_sha256'] == run.sha(run.OUT/'protocol.json')
            replays.append(report.replay((algorithm, record['variant'], record['init'])))
        result = dict(purpose='engineering verification only, not algorithm performance',
            retry_probe=retry, trained_and_reloaded_models=len(records),
            training_steps_per_model=128, audited_episodes=len(rows), exact_event_replays=len(replays),
            preregistration_unchanged_after_test=True, actual_parameters_match_protocol=True,
            protocol_sha256=run.sha(run.OUT/'protocol.json'))
        run.write(output/'verification.json', result)
        print('VERIFICATION_PASSED', result, flush=True)


if __name__ == '__main__':
    main()
