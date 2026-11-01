"""Complete reproducible workflow, including supplemental tests and audits."""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import importlib.metadata
import shutil

from slice_alloc.tools import allocation_final_run as run
from slice_alloc.tools import allocation_final_stress as stress

AUXILIARY=('allocation_final_pipeline.py','allocation_final_report.py',
           'allocation_final_calibrate.py','allocation_final_stress.py','allocation_final_allocate.py')


def freeze_auxiliary():
    value={name:run.sha(Path(__file__).parent/name) for name in AUXILIARY}
    path=run.OUT/'auxiliary_sources.json'
    if path.exists():
        assert run.read(path)==value,'Supplemental/report scripts changed'
    else:
        assert not (run.OUT/'test_started.json').exists()
        run.write(path,value)


def provenance():
    packages={p:importlib.metadata.version(p) for p in
        ('numpy','torch','stable-baselines3','sb3-contrib','gymnasium','matplotlib')}
    listing=subprocess.check_output(['ldd',str(run.a.NS3/'build/scratch/allocation-final-scenario')],text=True,
        env=dict(os.environ,LD_LIBRARY_PATH=str(run.a.NS3/'build/lib')))
    libraries={}
    for line in listing.splitlines():
        if 'libns3' in line and '=>' in line:
            path=Path(line.split('=>',1)[1].strip().split()[0])
            libraries[str(path)]=run.sha(path)
    run.write(run.OUT/'environment.json',dict(python=platform.python_version(),packages=packages,native_libraries=libraries))


def command(module,*args):
    subprocess.run([sys.executable,'-m',module,*args],check=True,cwd=run.ROOT,
                   env=dict(os.environ, LOWALT_ALLOCATION_OUTPUT=str(run.OUT)))


def ensure_tests():
    """Only a successful, matching test run authorizes reuse of validation."""
    marker = run.OUT/'tests_complete.json'
    inputs = dict(protocol_sha256=run.sha(run.OUT/'protocol.json'),
        sources={str(p.relative_to(run.ROOT)): run.sha(p)
                 for p in sorted((run.ROOT/'slice_alloc/tests').rglob('test_*.py'))})
    canonical_log = run.OUT/'tests.log'
    if marker.exists():
        completed = run.read(marker)
        if (completed.get('inputs') == inputs and canonical_log.exists()
                and completed.get('log_sha256') == run.sha(canonical_log)):
            return
        marker.unlink()
    index = 1
    while (run.OUT/f'tests-attempt-{index:04d}.log').exists():
        index += 1
    log = run.OUT/f'tests-attempt-{index:04d}.log'
    with log.open('x') as stream:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','slice_alloc/tests','-p','test_*.py'],
            check=True,cwd=run.ROOT,stdout=stream,stderr=subprocess.STDOUT)
    shutil.copy2(log, canonical_log)
    pending = marker.with_suffix('.tmp')
    run.write(pending, dict(inputs=inputs, log_sha256=run.sha(canonical_log)))
    pending.replace(marker)


def prepare_evaluation():
    """All formal entry points register the same prerequisites before testing."""
    run.freeze()
    ensure_tests()
    stress.freeze()
    freeze_auxiliary()
    deployment = dict(algorithm=run.read(run.OUT/'selection.json')['selected'],initialization=11,
        selection='lowest preregistered initialization chosen before formal test, never best-test initialization',
        policy='single learned model, no ensemble or rule fallback')
    path = run.OUT/'deployment.json'
    if path.exists():
        assert run.read(path)==deployment
    else:
        assert not (run.OUT/'test_started.json').exists()
        run.write(path,deployment)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    run.freeze()
    stress.freeze()
    freeze_auxiliary()
    if not (run.OUT/'environment.json').exists():
        provenance()
    ensure_tests()
    if not (run.OUT/'calibration.json').exists():
        command('slice_alloc.tools.allocation_final_calibrate')
    if not (run.OUT/'training_complete.json').exists():
        command('slice_alloc.tools.allocation_final_run','train','--workers',str(args.workers))
    prepare_evaluation()
    if not (run.OUT/'evaluation.json').exists():
        command('slice_alloc.tools.allocation_final_run','evaluate','--workers',str(args.workers))
    if not (run.OUT/'stress_evaluation.json').exists():
        command('slice_alloc.tools.allocation_final_stress')
    command('slice_alloc.tools.allocation_final_report')


if __name__=='__main__':
    main()
