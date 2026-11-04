"""Complete the declared family selection, then exactly one fresh test."""
from pathlib import Path
import json,time,subprocess,sys,os,hashlib,shutil
ROOT=Path('/home/allen/My_Tasks/Final_Graduate')
BASE=ROOT/'code/slice_alloc/results/mission'
PYTHON='/home/allen/miniconda3/envs/sdn/bin/python'
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
paths=[BASE/'training_complete.json',BASE.parent/'mission-fast/pilot_complete.json',BASE.parent/'mission-candidate/pilot_complete.json']
while not all(p.exists() for p in paths):time.sleep(10)
base=read(paths[0]);baseline={r['init']:r['selected']['validation']['rate'] for r in base}
pilots=[]
for path in paths[1:]:
 rows=read(path)
 gain=sum(r['selected']['validation']['rate']-baseline[r['init']] for r in rows)/len(rows)
 qualifies=gain>=.005 and all(r['selected']['validation']['rate']>=r['checkpoints'][0]['validation']['rate'] for r in rows)
 pilots.append(dict(family=path.parent.name,gain_over_matching_base=gain,qualifies_for_expansion=qualifies,results=[dict(init=r['init'],step=r['selected']['step'],validation=r['selected']['validation']['rate']) for r in rows]))
if any(p['qualifies_for_expansion'] for p in pilots):
 write(BASE/'family_continuation_required.json',pilots)
 print('NEEDS FULL8 FAMILY EXPANSION',pilots,flush=True)
 sys.exit(2)
write(BASE/'family_selection.json',dict(selected_family='mission',initializations=[r['init'] for r in base],
 mean_validation=sum(baseline.values())/len(base),prior_validation=read(BASE/'prior_validation.json')['rate'],
 pilot_decisions=pilots,selected_models=[dict(init=r['init'],step=r['selected']['step'],validation=r['selected']['validation']['rate'],sha256=r['sha256']) for r in base],
 controller_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 decision='base family retained after both bounded pilots fail declared0.5pp expansion threshold; final test determines publication, not validation gain'))
shutil.copy2(__file__,BASE/'family_selection_audit.py')
log=Path('/tmp/lowalt-final-study-tests.log').read_text();assert '\nOK\n' in log
shutil.copy2('/tmp/lowalt-final-study-tests.log',BASE/'full-tests.log')
assert not(BASE/'test_started.json').exists()
env=dict(os.environ,PYTHONPATH=str(ROOT/'code'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for command in ('evaluate','verify'):
 with open('/tmp/lowalt-mission-'+command+'.log','w') as stream:
  subprocess.run([PYTHON,'-u','-m','slice_alloc.tools.mission_allocation_evaluate',command,'--workers','8'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
 print(command.upper(),'COMPLETE',flush=True)
a=read(BASE/'analysis.json')
print('FINAL REAL RESULT',a['promoted'],a['primary_rule'],a['comparisons']['ppo-'+a['primary_rule']],a['promotion'],flush=True)
