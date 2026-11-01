"""Controlled protocol sanity checks; not algorithm tuning or official test data."""
from dataclasses import replace
import copy
import numpy as np
from slice_alloc import modal_binding as b
from slice_alloc.tools.modal_binding_run import OUT,episode,write

class Forced:
    def __init__(self,mode,cache=0):self.mode=mode;self.cache=cache
    def predict(self,obs,deterministic=True):
        rows=obs['candidates'];offset=2*len(b.p.RESOURCES)
        valid=np.flatnonzero((obs['mask']>.5)&(rows[:,offset+b.MODES.index(self.mode)]>.5))
        for i in valid:
            if rows[i,offset+3]>.5 and (self.mode!='ndn' or bool(rows[i,offset+7])==bool(self.cache)):return int(i),None
        return 0,None

def main():
    cfg=b.scenario(13100,1);cfg['private']['initial_positions']['5']=[30.,0.,80.]
    cfg['private']['motions']=[dict(node=5,speed=8.,right=True,start=0.)]
    r=b.Service(1,1,'command',(5,),2.,17.,.3,.6,512,1,1001,False)
    cfg['requests']=[r.json()];rows=[]
    for moving in (False,True):
        case=copy.deepcopy(cfg)
        if not moving:case['private']['motions']=[]
        for mode in b.MODES:
            result=episode(case,'mechanism',Forced(mode),OUT/'controlled'/f'mobility-{moving}-{mode}')
            rows.append(dict(case='mobility',moving=moving,mode=mode,**{k:v for k,v in result.items() if k not in ('decisions','observations','settled')},
                delivery_ratio=result['settled'][0]['on_time']/result['settled'][0]['expected']))
    cfg['private']['motions']=[]
    for n in (7,9):cfg['private']['initial_positions'][str(n)]=[20.,0.,80.]
    r=replace(r,service='object',members=(7,9),start=2.,stop=11.,period=2.,deadline=.8,payload=1000,chunks=4,content=1,reusable=True,minimum_ratio=1.)
    cfg['requests']=[r.json()]
    for mode,cache in [('ip',0),('ndn',0),('ndn',1)]:
        result=episode(cfg,'mechanism',Forced(mode,cache),OUT/'controlled'/f'cache-{mode}-{cache}')
        rows.append(dict(case='reuse',mode=mode,cache=cache,**{k:v for k,v in result.items() if k not in ('decisions','observations','settled')}))
    write(OUT/'controlled.json',rows)
    print([(r['case'],r['mode'],r.get('moving',r.get('cache')),r['success'],r.get('delivery_ratio'),r['origin_wire_bytes']) for r in rows])

if __name__=='__main__':main()
