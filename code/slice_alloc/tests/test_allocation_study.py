import copy
import unittest
import numpy as np
import torch
from slice_alloc.patrol import scenario, Request, baseline
from slice_alloc.tools.allocation_study import AllocationEnv, STATE_DIM, CANDIDATE_DIM
from slice_alloc.src.allocation_policy import ResourcePriorPolicy


class AllocationStudyTests(unittest.TestCase):
    def config(self):
        c=scenario(6200,1,'mixed',4)
        c['private']['motions'][0].update(speed=5.,start=12.)
        c['requests']=[Request(i,1,'ip',(7,),2.+i,4.+i,.08,.2,1000,1,i).json() for i in range(1,5)]
        return c

    def test_future_arrival_count_and_trajectory_do_not_change_encoded_prefix(self):
        c=self.config();other=copy.deepcopy(c)
        extra=copy.deepcopy(other['requests'][-1]);extra.update(id=5,start=15.,stop=18.)
        other['requests'].append(extra)
        other['private']['motions'][0].update(speed=11.,start=12.)
        prefixes=[]
        for cfg in (c,other):
            e=AllocationEnv()
            try:
                obs,_=e.reset(options=dict(scenario=cfg));past=[]
                for _ in range(3):
                    past.append(copy.deepcopy(obs));obs,_,_,_,_=e.step(baseline('price',e))
                prefixes.append(past)
            finally:e.close()
        for a,b in zip(*prefixes):
            for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_potential_preserves_episode_success_objective(self):
        e=AllocationEnv(potential=True);total=0.;raw=0.
        try:
            obs,_=e.reset(options=dict(scenario=self.config()));done=False
            while not done:
                self.assertTrue(e.observation_space.contains(obs))
                obs,r,done,_,info=e.step(baseline('price',e));total+=r;raw+=info['sla_reward']
            self.assertAlmostEqual(total,raw,places=10)
            self.assertEqual(raw,e.result()['success'])
        finally:e.close()

    def test_zero_residual_equals_price_and_is_candidate_permutation_equivariant(self):
        e=AllocationEnv()
        try:
            obs,_=e.reset(options=dict(scenario=self.config()))
            policy=ResourcePriorPolicy(e.observation_space,e.action_space,lambda _:1e-4)
            self.assertEqual(int(policy.predict(obs,deterministic=True)[0]),baseline('price',e))
            tensor={k:torch.as_tensor(v).unsqueeze(0) for k,v in obs.items()}
            first=policy.get_distribution(tensor).distribution.probs.detach().numpy()[0]
            perm=np.array([0,2,1,4,3])
            tensor['candidates']=tensor['candidates'][:,perm];tensor['mask']=tensor['mask'][:,perm]
            second=policy.get_distribution(tensor).distribution.probs.detach().numpy()[0]
            np.testing.assert_allclose(first[perm],second,atol=1e-7)
        finally:e.close()

    def test_reject_only_observation_is_valid(self):
        e=AllocationEnv()
        policy=ResourcePriorPolicy(e.observation_space,e.action_space,lambda _:1e-4)
        obs=dict(state=np.zeros(STATE_DIM,np.float32),
                 candidates=np.zeros((5,CANDIDATE_DIM),np.float32),
                 mask=np.array([1,0,0,0,0],np.float32))
        self.assertEqual(int(policy.predict(obs,deterministic=True)[0]),0)

    def test_reward_unit_scaling_changes_no_observation_or_sla(self):
        from slice_alloc.tools.allocation_admission import TrainingEnv
        trajectories=[]
        for scale in (1.,1/80):
            e=TrainingEnv(reward_scale=scale);past=[];total=0.
            try:
                obs,_=e.reset(options=dict(scenario=self.config()));done=False
                while not done:
                    past.append(copy.deepcopy(obs))
                    obs,r,done,_,_=e.step(baseline('price',e));total+=r
                self.assertAlmostEqual(total/scale,e.result()['success'])
                trajectories.append(past)
            finally:e.close()
        for a,b in zip(*trajectories):
            for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_paired_reference_is_terminal_training_label_only(self):
        from slice_alloc.tools.allocation_admission import TrainingEnv
        trajectories=[]
        for paired in (False,True):
            e=TrainingEnv(reward_scale=1/80,paired_reference=paired);past=[];total=0.
            try:
                obs,_=e.reset(options=dict(scenario=self.config()));done=False
                while not done:
                    past.append(copy.deepcopy(obs))
                    obs,r,done,_,_=e.step(baseline('price',e));total+=r
                reference=e.reference_success if paired else 0
                self.assertAlmostEqual(total*80,e.result()['success']-reference)
                if paired:self.assertEqual(e.reference_decisions,len(self.config()['requests']))
                trajectories.append(past)
            finally:e.close()
        for a,b in zip(*trajectories):
            for key in a:np.testing.assert_array_equal(a[key],b[key])


if __name__=='__main__':unittest.main()
