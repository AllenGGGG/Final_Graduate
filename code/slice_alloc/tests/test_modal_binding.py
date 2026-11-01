import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from slice_alloc import modal_binding as b

class BindingTests(unittest.TestCase):
    def cfg(self):return b.scenario(13000,8)
    def test_contract_has_no_assigned_modality(self):
        cfg=self.cfg();self.assertTrue(all('mode' not in r for r in cfg['requests']))
        cfg['requests'][0]['mode']='ip'
        with self.assertRaises(ValueError):b.validate(cfg)
    def test_reusable_name_and_version_have_one_immutable_object_size(self):
        sizes={}
        for seed in range(13320,13330):
            for raw in b.scenario(seed,80)['requests']:
                if not raw['reusable']:continue
                key=(raw['tenant'],raw['content'],raw['version'])
                size=(raw['payload'],raw['chunks'])
                if key in sizes:self.assertEqual(sizes[key],size)
                sizes[key]=size
    def test_same_business_payload_and_sla_in_every_native_stack(self):
        from dataclasses import replace
        cfg=self.cfg();r=b.Service(**{**cfg['requests'][0],'members':(7,)})
        r=replace(r,id=1,service='object',start=2.,stop=7.,period=2.,deadline=.8,
            chunks=4,content=1,reusable=True,minimum_ratio=1.)
        cfg['requests']=[r.json()];cfg['private']['motions']=[]
        cfg['private']['initial_positions']['7']=[20.,0.,80.]
        expected=[]
        for mode in b.MODES:
            env=b.BindingEnv()
            try:
                env.reset(options={'scenario':cfg});plan=next(i for i,p in enumerate(env.options,1) if p.request.mode==mode and p.edge==1)
                _,reward,done,_,_=env.step(plan);result=env.result()
                self.assertTrue(done);self.assertEqual(reward,1.)
                expected.append(result['settled'][0]['expected'])
                self.assertEqual(result['settled'][0]['on_time'],r.expected)
            finally:env.close()
        self.assertEqual(expected,[r.expected]*3)
    def test_encoding_ignores_private_future_and_unarrived_contracts(self):
        env=b.BindingEnv()
        try:
            env.reset(options={'scenario':self.cfg()});a=env.encode()
            env.cfg['private']['motions']=[];env.cfg['private']['initial_positions']={}
            last=max(env.requests);env.requests[last]=None
            c=env.encode()
            for key in a:np.testing.assert_array_equal(a[key],c[key])
        finally:env.close()
    def test_reservation_mask_and_reward_reconstruct(self):
        env=b.BindingEnv()
        try:
            obs,_=env.reset(options={'scenario':b.scenario(13001,48)});reward=0.;done=False
            while not done:
                for plan in env.options:self.assertTrue(env.ledger.feasible(plan))
                action=b.baseline('joint_price',env)
                self.assertEqual(obs['mask'][action],1.)
                obs,r,done,_,_=env.step(action);reward+=r
            result=env.result();self.assertEqual(reward,result['success'])
            for s in result['settled']:
                self.assertEqual(s['expected'],env.requests[s['id']].expected)
        finally:env.close()
    def test_two_consumers_cannot_select_single_endpoint_id(self):
        from dataclasses import replace
        cfg=self.cfg();r=b.Service(**{**cfg['requests'][0],'members':(7,9)})
        r=replace(r,service='object',period=2.,chunks=4,reusable=True)
        obs={'associations':{str(n):3 for n in range(5,17)},'cached':[]}
        plans=b.candidates(r,b.p.Ledger(cfg['public']),obs)
        self.assertEqual({p.request.mode for p in plans},{'ip','ndn'})

if __name__=='__main__':unittest.main()
