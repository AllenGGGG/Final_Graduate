"""Post-study provenance, common-input and exact replay checks."""
import collections
import importlib.metadata
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from slice_alloc.tools.modal_binding_run import OUT,ROOT,read,write,sha,hashes,episode,RULES,INITS
from slice_alloc.distributional_allocation import MaskedDoubleQRDQN
from slice_alloc import modal_binding as b


def main():
    torch.set_num_threads(1);protocol=read(OUT/'protocol.json')
    assert protocol['sources']==hashes()
    assert protocol['binary_sha256']==sha(b.p.NS3/'build/scratch/binding-scenario')
    assert read(OUT/'deployment_selection_protocol.json')['sha256']==sha(ROOT/'slice_alloc/tools/modal_binding_allocate.py')
    for name,digest in read(OUT/'platform.json')['files'].items():assert sha(ROOT/name)==digest
    data=read(OUT/'evaluation.json')
    for name in ('ndn_only','semantic_rule'):data+=read(OUT/f'{name}-evaluation.json')
    assert read(OUT/'challengers-protocol.json')['source_sha256']==sha(ROOT/'slice_alloc/tools/modal_binding_challengers.py')
    inputs=collections.defaultdict(set)
    for row in data:
        assert row['audited'];inputs[row['scope'],row['mix'],row['load'],row['seed']].add(row['scenario_sha256'])
        prefix=OUT/'native'/f"{row['method']}-{row['init']}"/f"{row['scope']}-{row['mix']}-{row['load']}-{row['seed']}"
        cfg=read(str(prefix)+'-scenario.json');result=read(str(prefix)+'-result.json')
        objects={}
        for raw in cfg['requests']:
            if not raw['reusable']:continue
            key=(raw['tenant'],raw['content'],raw['version'])
            size=(raw['payload'],raw['chunks'])
            assert key not in objects or objects[key]==size
            objects[key]=size
        contracts={raw['id']:b.Service(**{**raw,'members':tuple(raw['members'])}) for raw in cfg['requests']}
        for settled in result['settled']:
            assert settled['expected']==contracts[settled['id']].expected
        assert row['success']==sum(s['success'] for s in result['settled'])
    assert all(len(x)==1 for x in inputs.values())
    replay=[];models=read(OUT/'selection.json')['models']
    for record in models:
        path=ROOT/record['model'];assert sha(path)==record['model_sha256']
        assert record['steps']==32768 and record['weight_change']>0 and record['updates']>0
        assert set(record['training_seeds']).isdisjoint(set(protocol['validation']+protocol['test']+protocol['ood']))
        model=MaskedDoubleQRDQN.load(path,device='cpu')
        assert model.num_timesteps==record['selected']['step']
        assert model.n_quantiles==32 and model.gamma==1
        assert model.policy_kwargs['net_arch']==[96]
        assert model.learning_rate==1e-4 and model.batch_size==128
        assert model.buffer_size==20000 and model.learning_starts==1024
        assert model.target_update_interval==1000 and model.exploration_steps==16384
        assert model.replay_buffer_kwargs==dict(horizon=8,discount=1.)
        method=record['method'];init=record['init'];seed=protocol['test'][0];load=protocol['loads'][0]
        prefix=OUT/'native'/f'{method}-{init}'/f'test-balanced-{load}-{seed}'
        original=read(str(prefix)+'-result.json')
        cfg=read(str(prefix)+'-scenario.json')
        with tempfile.TemporaryDirectory(prefix='binding-replay-') as temporary:
            target=Path(temporary)/'episode'
            again=episode(cfg,method,model,target,fixed=method=='fixed_qrdqn',motion=method!='no_motion')
            assert again['decisions']==original['decisions']
            assert again['settled']==original['settled']
            assert again['observations']==original['observations']
            import gzip
            with gzip.open(str(prefix)+'-events.csv.gz','rb') as a,gzip.open(str(target)+'-events.csv.gz','rb') as c:
                assert a.read()==c.read()
        replay.append(dict(method=method,init=init,model_sha256=sha(path),exact_packet_replay=True))
    result=dict(passed=True,source_fingerprints=True,binary_fingerprint=True,application_expected_counts=True,
        audited_episodes=len(data),identical_input_scenarios=len(inputs),exact_replays=replay,
        packages={name:importlib.metadata.version(name) for name in ('numpy','torch','gymnasium','stable-baselines3','sb3-contrib','jsonschema')},
        caveat='First episode of each selected model replayed exactly; every formal episode audited during evaluation. CPU timing is not bitwise deterministic.')
    write(OUT/'verification.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
