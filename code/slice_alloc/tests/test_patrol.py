import copy
import unittest
from pathlib import Path
from slice_alloc.patrol import *

class PatrolContracts(unittest.TestCase):
    def obs(self,cfg):
        return dict(schema=PROTOCOL,kind='observation',time=2,request_id=1,associations={n:aps[0] for n,aps in cfg['public']['allowed_aps'].items()},positions=cfg['private']['initial_positions'],cached=[],settled=[])
    def test_generated_native_boundary(self):
        self.assertEqual(generate_header(),(ROOT/'slice_alloc/execution/patrol_contract.hpp').read_text())
    def test_actual_association_required(self):
        cfg=scenario();r=request_from_json(cfg['requests'][0]);o=self.obs(cfg);ledger=Ledger(cfg['public'])
        self.assertTrue(candidates(r,ledger,o));o['associations'][str(r.members[0])]=0
        self.assertEqual([],candidates(r,ledger,o))
    def test_private_future_does_not_change_candidates(self):
        c=scenario();r=request_from_json(c['requests'][0]);o=self.obs(c)
        before=[p.decision() for p in candidates(r,Ledger(c['public']),o)]
        c['private']['motions'][0]['speed']=1000
        self.assertEqual(before,[p.decision() for p in candidates(r,Ledger(c['public']),o)])
    def test_edge_path_is_real_cost(self):
        c=scenario();o=self.obs(c);r=Request(1,1,'ip',(7,),2,8,.04,.2,1000,1,1)
        a=compile_plan(r,1,1,0,c['public'],o);b=compile_plan(r,2,2,0,c['public'],o)
        self.assertEqual(a.peak[INDEX['wire:1:2']],0)
        self.assertGreater(b.peak[INDEX['wire:1:2']],0)
    def test_shared_cache_accounting_and_release(self):
        c=scenario();o=self.obs(c);l=Ledger(c['public'])
        r=Request(1,1,'ndn',(7,),2,8,2,.8,1000,4,1)
        p=compile_plan(r,1,1,1,c['public'],o);l.add(p)
        r2=Request(2,1,'ndn',(7,),4,10,2,.8,1000,4,1)
        l.add(compile_plan(r2,1,1,1,c['public'],o))
        self.assertEqual(l.usage(5)[INDEX['cache:1']],4000)
        l.release(9);self.assertEqual(l.usage(9)[INDEX['cache:1']],4000)
        l.release(11);self.assertEqual(l.usage(11)[INDEX['cache:1']],0)
    def test_no_future_cache_discount(self):
        c=scenario();o=self.obs(c);r=Request(1,1,'ndn',(7,),2,8,2,.8,1000,4,1)
        cold=compile_plan(r,1,1,1,c['public'],o)
        o['cached']=[dict(node=1,tenant=1,content=1,version=1,bytes=4000)]
        warm=compile_plan(r,1,1,1,c['public'],o)
        self.assertGreater(cold.peak[INDEX['wire:0:1']],warm.peak[INDEX['wire:0:1']])
        self.assertTrue(all(not p.warm for p in candidates(r,Ledger(c['public']),o) if not p.cache))
    def test_mapping_quota_is_enforced(self):
        c=scenario(profile='mapping');o=self.obs(c);l=Ledger(c['public'])
        for i in (1,2):l.add(compile_plan(Request(i,1,'id',(7,),2,8,.15,.6,256,1,i),1,1,0,c['public'],o))
        p=compile_plan(Request(3,1,'id',(7,),2,8,.15,.6,256,1,3),1,1,0,c['public'],o)
        self.assertFalse(l.feasible(p))
    def test_modality_semantics_and_name_shared(self):
        c=scenario(seed=8,load=80,profile='cache');requests=[request_from_json(x) for x in c['requests']]
        groups=collections.defaultdict(list)
        for r in requests:
            r.json()
            if r.mode=='ndn':groups[r.prefix].append(r.id)
        self.assertTrue(any(len(v)>1 for v in groups.values()))
        self.assertEqual(len(set(r.id for r in requests)),len(requests))
        self.assertTrue(all(len(r.members)==1 for r in requests if r.mode!='ndn'))
    def test_load_changes_count_not_horizon(self):
        for n in (24,40):
            c=scenario(2,n);self.assertEqual(n,len(c['requests']))
            self.assertTrue(all(2<=r['start']<=18 for r in c['requests']))
    def test_bad_observation_fails_boundary(self):
        c=scenario();o=self.obs(c);validate('observation',o)
        o['future_trajectory']=c['private']['motions']
        with self.assertRaises(Exception):validate('observation',o)

if __name__=='__main__':unittest.main()
