import copy
import unittest
import numpy as np
from slice_alloc import patrol
from slice_alloc.tools.lowalt_allocation import LowaltEnv,marginal_calendar


class LowaltAllocationTests(unittest.TestCase):
    def config(self):
        c=patrol.scenario(7521,4,'mixed',4)
        c['private']['motions'][0].update(start=10.,speed=5.)
        c['requests']=[patrol.Request(i,1,'ip',(5,),float(i+1),float(i+2),.08,.2,1000,1,i).json() for i in range(1,5)]
        return c

    def test_future_changes_do_not_change_past_motion_or_candidates(self):
        a=self.config();b=copy.deepcopy(a)
        b['private']['motions'][0]['speed']=11.
        extra=copy.deepcopy(b['requests'][-1]);extra.update(id=5,start=14.,stop=17.);b['requests'].append(extra)
        prefixes=[]
        for cfg in (a,b):
            e=LowaltEnv();past=[]
            try:
                obs,_=e.reset(options=dict(scenario=cfg))
                for _ in range(3):
                    past.append(copy.deepcopy(obs));obs,_,_,_,_=e.step(patrol.baseline('price',e))
                prefixes.append(past)
            finally:e.close()
        for a,b in zip(*prefixes):
            for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_repeated_encoding_is_idempotent_and_in_range(self):
        e=LowaltEnv()
        try:
            obs,_=e.reset(options=dict(scenario=self.config()))
            for _ in range(3):
                self.assertTrue(e.observation_space.contains(obs))
                repeated=e.encode()
                for k in obs:np.testing.assert_array_equal(obs[k],repeated[k])
                obs,_,_,_,_=e.step(patrol.baseline('price',e))
        finally:e.close()

    def test_marginal_cache_charges_only_extension_of_shared_lease(self):
        pub=patrol.public_config();ledger=patrol.Ledger(pub)
        r=patrol.Request(1,1,'ndn',(5,6),2.,5.,2.,.8,1000,4,1,minimum_ratio=1.)
        observation={'cached':[]};p=patrol.compile_plan(r,1,1,1,pub,observation)
        ledger.add(p)
        r2=patrol.Request(2,1,'ndn',(5,6),3.,6.,2.,.8,1000,4,1,minimum_ratio=1.)
        new=patrol.compile_plan(r2,1,1,1,pub,observation)
        integral,_=marginal_calendar(ledger,new)
        expected=r2.size/ledger.capacity[patrol.INDEX['cache:1']]*(r2.drain-r.drain)
        self.assertAlmostEqual(integral[patrol.INDEX['cache:1']],expected)

    def test_motion_ablation_changes_only_state_tail(self):
        trajectories=[]
        for enabled in (False,True):
            e=LowaltEnv(motion=enabled)
            try:
                obs,_=e.reset(options=dict(scenario=self.config()));trajectories.append(obs)
            finally:e.close()
        a,b=trajectories
        np.testing.assert_array_equal(a['candidates'],b['candidates'])
        np.testing.assert_array_equal(a['mask'],b['mask'])
        np.testing.assert_array_equal(a['state'][:-10],b['state'][:-10])
        self.assertTrue(np.all(a['state'][-10:]==0));self.assertTrue(np.any(b['state'][-10:]!=0))

    def test_reward_sums_to_unmodified_native_success_count(self):
        e=LowaltEnv();total=0.
        try:
            obs,_=e.reset(options=dict(scenario=self.config()));done=False
            while not done:
                obs,r,done,_,_=e.step(patrol.baseline('price',e));total+=r
            self.assertEqual(total,e.result()['success'])
        finally:e.close()

    def test_velocity_is_estimated_from_two_observed_positions(self):
        cfg=self.config();cfg['private']['motions'][0].update(start=0.,speed=5.)
        e=LowaltEnv()
        try:
            first,_=e.reset(options=dict(scenario=cfg))
            self.assertEqual(first['state'][-6],0.)
            second,_,_,_,_=e.step(patrol.baseline('price',e))
            self.assertEqual(second['state'][-6],1.)
            self.assertAlmostEqual(float(second['state'][-8]),5/11,places=6)
        finally:e.close()


if __name__=='__main__':unittest.main()
