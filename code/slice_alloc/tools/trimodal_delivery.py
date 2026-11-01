"""Bind tested source, independent evidence and generated manuscript together."""
import json
import re
from pathlib import Path
from urllib.parse import unquote
from .trimodal_run import OUT, ROOT, NS3, sources, write
from .platform_verify import sha


def main():
    independent=json.loads((OUT/'independent_audit.json').read_text())
    diagnosis=json.loads((OUT/'diagnosis.json').read_text())
    document=json.loads((ROOT/'slice_alloc/results/document_verification.json').read_text())
    old=json.loads((ROOT/'slice_alloc/results/joint/final_audit.json').read_text())
    if not all(d['passed'] for d in (independent,document,old)):raise AssertionError('upstream acceptance failed')
    if independent['sources']!=sources():raise AssertionError('new source changed after native execution')
    if independent['verifier_sha256']!=sha(Path(__file__).with_name('trimodal_verify.py')):raise AssertionError('independent verifier changed')
    if diagnosis['source_sha256']!=sha(Path(__file__).with_name('trimodal_diagnose.py')):raise AssertionError('diagnosis changed')
    if diagnosis['analysis_sha256']!=sha(OUT/'analysis.json') or diagnosis['independent_audit_sha256']!=sha(OUT/'independent_audit.json'):raise AssertionError('diagnosis inputs stale')
    for name,digest in independent['artifacts'].items():
        if sha(OUT/name)!=digest:raise AssertionError('independent artifact changed: '+name)
    if document['trimodal_independent_audit_sha256']!=sha(OUT/'independent_audit.json'):raise AssertionError('manuscript uses stale new audit')
    if document['trimodal_report_sha256']!=sha(OUT/'report.md'):raise AssertionError('manuscript uses stale scores')
    if document['design_source_sha256']!=sha(ROOT/'slice_alloc/design.md'):raise AssertionError('design not exported')
    for suffix in ('md','docx'):
        p=ROOT/f'面向低空智联的多模态网络切片技术.{suffix}'
        if document[suffix+'_sha256']!=sha(p):raise AssertionError('generated document changed')
    for name,digest in old['sources'].items():
        if sha(ROOT/name)!=digest:raise AssertionError('old regression source changed')
    for name in ('comparison','analysis','diagnostic_verification','allocation_diagnostic_verification'):
        if old[name+'_sha256']!=sha(ROOT/'slice_alloc/results/joint'/f'{name}.json'):raise AssertionError('old upstream fingerprint mismatch')
    md=(ROOT/'面向低空智联的多模态网络切片技术.md').read_text()
    for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',md):
        if '://' not in link and not (ROOT/unquote(link)).exists():raise AssertionError('broken manuscript link: '+link)
    testlog=(OUT/'delivery-tests.log').read_text()
    if '\nOK' not in testlog:raise AssertionError('delivery tests failed')
    count=int(re.search(r'Ran (\d+) tests',testlog).group(1))
    mechanisms=json.loads((OUT/'mechanisms.json').read_text())
    names=[ROOT/'README.md',ROOT/'.gitignore',ROOT/'slice_alloc/design.md',
           ROOT/'slice_alloc/tools/trimodal_verify.py',ROOT/'slice_alloc/tools/trimodal_diagnose.py',Path(__file__),
           ROOT/'slice_alloc/tools/report.py',ROOT/'slice_alloc/tools/export_document.py',
           ROOT/'面向低空智联的多模态网络切片技术.md',ROOT/'面向低空智联的多模态网络切片技术.docx',
           ROOT/'slice_alloc/results/document_verification.json',ROOT/'slice_alloc/results/joint/final_audit.json']
    names+=list((ROOT/'图').glob('*.png'))+list((ROOT/'图').glob('*.svg'))+list((ROOT/'图/源文件').glob('*.dot'))
    names+=[OUT/n for n in ('independent_audit.json','diagnosis.json','validation.md','report.md','cleanup.json','delivery-tests.log')]
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in names}
    result=dict(passed=True,tests=count,new_protocol_cases=len(mechanisms['results']),new_mechanism_assertions=len(mechanisms['checks']),new_native_test_cases=independent['raw_cases'],
        independent_policy_replays=independent['policy_checkpoint_replays'],old_native_cases_reconstructed=old['native_joint_cases_reconstructed'],
        document_and_evidence_consistent=True,all_local_manuscript_links_exist=True,sources=sources(),artifacts=artifacts,
        native_binary_sha256=sha(NS3/'build/scratch/trimodal-scenario'),
        scope='tested restricted initial-allocation prototype; no demonstrated RL or joint-mapping performance advantage; not full dynamic thesis platform')
    write(OUT/'delivery.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','artifacts')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
