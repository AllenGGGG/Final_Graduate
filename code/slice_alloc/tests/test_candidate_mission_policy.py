"""Reordering available plans must not change their learned policy scores."""
from pathlib import Path
import tempfile
import unittest
import torch
from sb3_contrib import MaskablePPO
from slice_alloc import mission_allocation as m
from slice_alloc.candidate_mission_policy import CandidateMissionPolicy


class CandidatePolicyTests(unittest.TestCase):
    def test_trained_actor_equivariance_and_exact_roundtrip(self):
        torch.set_num_threads(1)
        env=m.MissionEnv(loads=(8,),rule_parameters=(2.,.5,1.),admission_threshold=2.)
        try:
            model=MaskablePPO(CandidateMissionPolicy,env,n_steps=32,batch_size=16,n_epochs=2,
                policy_kwargs=dict(net_arch=dict(pi=[],vf=[96,96]),ortho_init=False),seed=11,device='cpu')
            model.learn(128)
            self.assertTrue(bool(torch.any(model.policy.action_net.scorer.deployment[-1].weight!=0)))
            obs,_=env.reset(options={'scenario':m.scenario(329007,24)})
            tensor,_=model.policy.obs_to_tensor(obs)
            order=torch.tensor([0,3,1,2,4,5,6,7,8,9,10])
            changed={**tensor,'candidates':tensor['candidates'][:,order],
                'mask':tensor['mask'][:,order],'prior':tensor['prior'][:,order]}
            before=model.policy.action_net(model.policy.extract_features(tensor))
            after=model.policy.action_net(model.policy.extract_features(changed))
            torch.testing.assert_close(after,before[:,order],atol=1e-6,rtol=1e-5)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'model.zip';model.save(path)
                again=MaskablePPO.load(path,device='cpu')
            torch.testing.assert_close(before,again.policy.action_net(again.policy.extract_features(tensor)),atol=0,rtol=0)
            self.assertEqual(int(model.predict(obs,action_masks=obs['mask']>.5,deterministic=True)[0]),
                int(again.predict(obs,action_masks=obs['mask']>.5,deterministic=True)[0]))
        finally:env.close()


if __name__=='__main__':unittest.main()
