"""Regressions for retry, actual-configuration freezing and paired statistics."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from slice_alloc.tools import allocation_confirm_train as train
from slice_alloc.tools import allocation_confirm_evaluate as evaluate
from slice_alloc.allocation_credit import CreditEnv
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.distributional_allocation import CandidateQRPolicy
from slice_alloc.allocation_mc import MonteCarloDictReplay
import torch


class ConfirmationTests(unittest.TestCase):
    def test_serialization_repair_preserves_every_policy_tensor(self):
        with TemporaryDirectory() as folder,mock.patch.object(train,'OUT',Path(folder)),mock.patch.object(train,'PARAMETERS',{**train.PARAMETERS,'buffer_size':12}):
            env=CreditEnv(loads=(8,))
            try:
                model=MaskedDoubleDQN(CandidateQRPolicy,env,buffer_size=4,learning_starts=1,
                                     replay_buffer_class=MonteCarloDictReplay,
                                     policy_kwargs=dict(n_quantiles=1,net_arch=[8]),device='cpu')
                path=train.OUT/'training/11/0.zip';path.parent.mkdir(parents=True,exist_ok=True);model.save(path)
                before={n:t.clone() for n,t in model.policy.state_dict().items()}
                chosen=dict(step=0,path=str(path.relative_to(train.OUT)),sha256=train.sha(path),validation={'rate':.5})
                final=train.OUT/'models/double_dqn-11.zip';final.parent.mkdir(parents=True,exist_ok=True);model.save(final)
                row=dict(init=11,checkpoints=[chosen],selected=chosen,model=str(final.relative_to(train.OUT)),model_sha256=train.sha(final))
                train.write(train.OUT/'training_complete.json',dict(models=[row]))
                evaluate.prepare_artifacts()
                loaded=MaskedDoubleDQN.load(final,device='cpu');self.assertEqual(loaded.buffer_size,12)
                for n,t in before.items():torch.testing.assert_close(t,loaded.policy.state_dict()[n],atol=0,rtol=0)
                log=train.read(train.OUT/'artifact_metadata_repair.json')
                self.assertEqual(len(log['changes']),1);self.assertTrue(log['changes'][0]['all_policy_tensors_identical'])
                self.assertTrue((train.OUT/log['changes'][0]['original']).exists())
            finally:env.close()

    def test_failed_test_log_never_marks_completion_or_skips_retry(self):
        with TemporaryDirectory() as folder,mock.patch.object(train,'OUT',Path(folder)):
            with mock.patch.object(evaluate.subprocess,'run',return_value=SimpleNamespace(returncode=1)) as process:
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError,'tests failed'):evaluate.tests()
                self.assertEqual(process.call_count,2)
                self.assertTrue((train.OUT/'full-tests.log').exists())
                self.assertFalse((train.OUT/'tests_passed.json').exists())
            def success(*args,**kwargs):
                kwargs['stdout'].write('Ran 3 tests in 0.1s\n\nOK\n')
                return SimpleNamespace(returncode=0)
            with mock.patch.object(evaluate.subprocess,'run',side_effect=success) as process:
                first=evaluate.tests();self.assertEqual(first['passed_tests'],3)
                self.assertEqual(first,evaluate.tests());self.assertEqual(process.call_count,1)

    def test_freeze_detects_actual_training_parameters_and_dependency_changes(self):
        with TemporaryDirectory() as folder,mock.patch.object(train,'OUT',Path(folder)),mock.patch.object(train,'SOURCES',('allocation_mc.py',)):
            for i in train.INITS:
                target=train.OUT/'reference_models'/f'double_dqn-{i}.zip'
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(str(i).encode())
            train.freeze();train.check()
            changed={**deepcopy(train.PARAMETERS),'learning_rate':.123}
            with mock.patch.object(train,'PARAMETERS',changed):
                with self.assertRaises(AssertionError):train.freeze()
                with self.assertRaises(AssertionError):train.check()
            (train.OUT/'frozen/allocation_mc.py').write_text('changed dependency')
            with self.assertRaises(AssertionError):train.check()

    def test_paired_statistic_uses_all_arrivals_and_checks_complete_coverage(self):
        with mock.patch.object(train,'INITS',(11,23)),mock.patch.object(train,'TEST',(700,701)),mock.patch.object(train,'LOADS',(10,)):
            rows=[]
            for method,init in evaluate.specifications():
                for seed in train.TEST:
                    success=8 if method=='rl' else 7
                    rows.append(dict(method=method,init=init,seed=seed,load=10,requests=10,
                                     success=success,admitted=9,forced_reject=1,voluntary_reject=0,
                                     admitted_violations=9-success,modes={'ip':9},
                                     by_service={s:dict(requests=10,success=success) for s in ('telemetry','command','object')}))
            result=evaluate.analyze(rows)
            self.assertAlmostEqual(result['comparisons']['rl-greedy']['difference'],.1)
            self.assertAlmostEqual(result['methods']['rl']['success_rate'],.8)
            self.assertAlmostEqual(result['methods']['rl']['violation_rate_admitted'],1/9)
            self.assertTrue(result['primary_comparison_passed'])
            with self.assertRaises(AssertionError):evaluate.analyze(rows[:-1])
            with self.assertRaises(AssertionError):evaluate.analyze([*rows,rows[0]])


if __name__=='__main__':unittest.main()
