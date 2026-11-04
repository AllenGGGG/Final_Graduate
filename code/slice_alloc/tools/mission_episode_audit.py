"""Original packet/direction/reservation auditor adapted only for arrival-reordered IDs.

The raw scenario hash is checked before using an ID-ordered view in the native
packet counter. Contract deadlines and every acceptance assertion are retained.
"""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from slice_alloc import allocation_final as a
from slice_alloc import modal_binding as b
from slice_alloc.tools import modal_binding_run as native


def read(path):
    return json.loads(Path(path).read_text())


def verify_episode(out, row):
    prefix = out/'native'/f"{row['method']}-{row['init']}"/f"{row['load']}-{row['seed']}"
    cfg = read(str(prefix)+'-scenario.json')
    result = read(str(prefix)+'-result.json')
    metrics = read(str(prefix)+'-metrics.json')
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    assert digest == metrics['scenario_sha256'] == result['scenario_sha256']
    for key in ('requests', 'success', 'admitted', 'by_service'):
        assert row[key] == metrics[key] == result[key], (prefix, key)
    assert result['requests'] == len(cfg['requests']) == row['load']
    # The native counter auditor expects contiguous IDs, independently of arrival order.
    ordered=sorted(cfg['requests'],key=lambda r:r['id'])
    assert [r['id'] for r in ordered]==list(range(1,len(ordered)+1))
    assert native.audit(prefix,result,dict(cfg,requests=ordered))['audited']
    requests = {r['id']: b.Service(**{**r, 'members': tuple(r['members'])}) for r in cfg['requests']}
    records = result['decisions']
    assert len(records) == len(requests)
    assert {r['request'] for r in records} == set(requests)
    accepted = {r['request']: r for r in records if r['decision']['accept']}
    assert len(accepted) == result['admitted']
    assert {s['id'] for s in result['settled']} == set(accepted)
    assert sum(s['success'] for s in result['settled']) == result['success']
    for item in result['settled']:
        assert item['expected'] == requests[item['id']].expected > 0
    # Reconstruct the committed reservations from saved causal observations.
    ledger = b.p.Ledger(cfg['public'])
    for message in result['observations']:
        observation = message['observation']
        if observation['kind'] == 'done':
            continue
        decision = message['decision']
        request = requests[observation['request_id']]
        assert decision['request_id'] == request.id
        assert 'rate_factor' not in decision
        ledger.release(observation['time'])
        if not decision['accept']:
            continue
        mode = decision['mode']
        assert mode in a.COMPATIBLE[request.service]
        if request.service == 'telemetry':
            plan = b.p.compile_plan(request.bound('ip'), decision['edge'], decision['server'], 0,
                                    ledger.pub, observation)
        else:
            plan = b.compile_plan(request, mode, decision['edge'], decision['server'], decision['cache'],
                                  ledger.pub, observation)
        assert ledger.feasible(plan), ('illegal reservation', prefix, request.id)
        ledger.add(plan)
    # Check packet direction independently of the aggregate SLA audit.
    with gzip.open(str(prefix)+'-events.csv.gz', 'rt') as stream:
        for event in csv.DictReader(stream):
            sid = int(event['slice'])
            if sid not in requests or requests[sid].service != 'telemetry':
                continue
            if event['event'] == 'emit':
                assert int(event['node']) in requests[sid].members
            if event['event'] == 'complete':
                assert int(event['node']) == 0
    forced = sum(not r['decision']['accept'] and not r['feasible'] for r in records)
    voluntary = sum(not r['decision']['accept'] and bool(r['feasible']) for r in records)
    violations = result['admitted'] - result['success']
    assert forced + voluntary + violations + result['success'] == result['requests']
    return dict(**row, forced_reject=forced, voluntary_reject=voluntary,
                admitted_violations=violations, modes=result['modes'], scenario_sha256=digest)

