"""Export the audited first-part study and thesis figures, without model selection."""
import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from slice_alloc.tools import allocation_confirm_train as run

NAMES={'rl':'RL (full return)','initial_rl':'Initial RL','greedy':'Greedy',
       'nearest':'Nearest','first_fit':'First-Fit','native':'Native modality'}
COLORS={'rl':'#1565c0','initial_rl':'#74a9cf','greedy':'#ef6c00',
        'nearest':'#6a1b9a','first_fit':'#616161','native':'#00897b'}


def figures(out,analysis,training):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':200,'pdf.fonttype':42})
    methods=('rl','nearest','greedy','native','first_fit')
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('png','pdf'):fig.savefig(out/f'{name}.{suffix}',bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3.7))
    values=np.array([analysis['methods'][m]['success_rate']*100 for m in methods])
    ci=np.array([analysis['methods'][m]['ci95'] for m in methods]).T*100
    ax.bar(np.arange(len(methods)),values,color=[COLORS[m] for m in methods],width=.65)
    ax.errorbar(np.arange(len(methods)),values,yerr=np.maximum(0,np.stack([values-ci[0],ci[1]-values])),
                fmt='none',ecolor='#333333',capsize=4)
    ax.set(xticks=np.arange(len(methods)),xticklabels=[NAMES[m] for m in methods],
           ylabel='SLA success / all arrivals (%)',ylim=(0,100))
    for x,y in enumerate(values):ax.text(x,y+2,f'{y:.2f}',ha='center',fontsize=9)
    save(fig,'comparison')

    fig,ax=plt.subplots(figsize=(7.2,3.7))
    for m in methods:
        values=[analysis['per_load'][m][str(l)]['success_rate']*100 for l in run.LOADS]
        ax.plot(run.LOADS,values,'o-',label=NAMES[m],color=COLORS[m])
    ax.set(xlabel='Arriving requests per episode',ylabel='SLA success / all arrivals (%)',xticks=run.LOADS)
    ax.legend(ncol=2,frameon=False);ax.grid(axis='y',alpha=.2)
    save(fig,'load-comparison')

    fig,ax=plt.subplots(figsize=(7.2,3.7));services=('telemetry','command','object');width=.16
    for j,m in enumerate(methods):
        values=[100*analysis['methods'][m]['by_service'][s]['success']/analysis['methods'][m]['by_service'][s]['requests'] for s in services]
        ax.bar(np.arange(3)+(j-2)*width,values,width=width,color=COLORS[m],label=NAMES[m])
    ax.set(xticks=np.arange(3),xticklabels=('Telemetry','Command','Map object'),
           ylabel='SLA success / service arrivals (%)',ylim=(0,100))
    ax.legend(ncol=3,frameon=False,fontsize=8);save(fig,'service-comparison')

    fig,ax=plt.subplots(figsize=(7.2,3.7))
    for r in training['models']:
        x=[c['step']/1000 for c in r['checkpoints']];y=[100*c['validation']['rate'] for c in r['checkpoints']]
        line,=ax.plot(x,y,'o-',label=f"Init {r['init']}")
        chosen=r['selected'];ax.scatter([chosen['step']/1000],[100*chosen['validation']['rate']],
                                      s=100,facecolors='none',edgecolors=line.get_color(),linewidths=1.5)
    ax.set(xlabel='Additional training decisions (thousands)',ylabel='Validation SLA success (%)',
           title='Validation checkpoints; open circle = selected model')
    ax.legend(ncol=3,frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2)
    save(fig,'validation-curves')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=run.PKG/'final');args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    a=run.read(run.OUT/'analysis.json');v=run.read(run.OUT/'verification.json')
    assert v['verified_native_episodes']==2520 and v['all_packet_audits_pass']
    for name in ('analysis.json','verification.json','evaluation.json','protocol.json','evaluation_registration.json','test_started.json',
                 'tests_passed.json','full-tests.log','artifact_metadata_repair.json','training_complete.json',
                 'independent_counter_check.json','reference_training_provenance.json','report.md'):
        shutil.copy2(run.OUT/name,out/name)
    figures(out,a,run.read(run.OUT/'training_complete.json'))
    # Keep raw bytes and checkpoints outside Git, with a complete inventory so
    # copying a bundle can be checked independently of the summary tables.
    run.write(out/'artifact_manifest.json',{str(p.relative_to(run.OUT)):run.sha(p)
        for p in sorted(run.OUT.rglob('*')) if p.is_file() and not p.name.endswith('.tmp')})
    run.write(out/'code_manifest.json',{str(p.relative_to(run.PKG)):run.sha(p)
        for p in sorted(run.PKG.rglob('*')) if p.is_file() and p.relative_to(run.PKG).parts[0] not in ('results','final')
        and '__pycache__' not in p.parts})
    run.write(out/'export_manifest.json',{str(p.relative_to(out)):run.sha(p)
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='export_manifest.json'})
    print('EXPORTED',out,'primary',a['primary_comparison_passed'],'full',a['full_publication_passed'])


if __name__=='__main__':main()
