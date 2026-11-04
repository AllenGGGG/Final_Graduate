import tempfile
from pathlib import Path
import unittest
import torch
from sb3_contrib import MaskablePPO
from slice_alloc import mission_allocation as m, allocation_final as base
from slice_alloc.mission_policy import MissionPolicy


class MissionTests(unittest.TestCase):
    def test_only_arrivals_change_and_rule_start_is_exact(self):
        torch.set_num_threads(1)
        cfg=m.scenario(329001,24);original=base.scenario(329001,24)
        self.assertEqual(cfg['public'],original['public'])
        before={r['id']:r for r in original['requests']}
        for r in cfg['requests']:
            old=before[r['id']]
            self.assertAlmostEqual(r['stop']-r['start'],old['stop']-old['start'])
            for k in r:
                if k not in ('start','stop'):self.assertEqual(r[k],old[k])
        env=m.MissionEnv(rule_parameters=(2.,.5,1.))
        try:
            model=MaskablePPO(MissionPolicy,env,n_steps=32,batch_size=16,n_epochs=2,
                policy_kwargs=dict(net_arch=dict(pi=[],vf=[96,96]),ortho_init=False),seed=11,device='cpu')
            obs,_=env.reset(options={'scenario':cfg});done=False;reward=0.
            while not done:
                action=int(model.predict(obs,action_masks=obs['mask']>.5,deterministic=True)[0])
                self.assertEqual(action,m.rule_action('modal_greedy',env))
                self.assertTrue(env.observation_space.contains(obs))
                obs,r,done,_,_=env.step(action);reward+=r
            self.assertEqual(reward,env.result()['success'])
        finally:env.close()

    def test_native_ppo_updates_and_roundtrips(self):
        torch.set_num_threads(1)
        env=m.MissionEnv(loads=(8,),rule_parameters=(2.,.5,1.))
        try:
            model=MaskablePPO(MissionPolicy,env,n_steps=32,batch_size=16,n_epochs=2,gamma=1.,
                policy_kwargs=dict(net_arch=dict(pi=[],vf=[96,96]),ortho_init=False),seed=23,device='cpu')
            before=model.policy.action_net.residual[-1].weight.detach().clone()
            model.learn(128)
            self.assertFalse(torch.equal(before,model.policy.action_net.residual[-1].weight))
            obs,_=env.reset(options={'scenario':m.scenario(329002,8)})
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'model.zip';model.save(path)
                again=MaskablePPO.load(path,device='cpu')
            self.assertEqual(int(model.predict(obs,action_masks=obs['mask']>.5,deterministic=True)[0]),
                int(again.predict(obs,action_masks=obs['mask']>.5,deterministic=True)[0]))
            for name,t in model.policy.state_dict().items():torch.testing.assert_close(t,again.policy.state_dict()[name],atol=0,rtol=0)
        finally:env.close()


if __name__=='__main__':unittest.main()
