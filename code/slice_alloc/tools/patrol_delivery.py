"""Final binding of frozen experiments, source, manuscript and regression evidence."""
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote
from .patrol_run import OUT,ROOT,NS3,source_hashes,sha,write


def main():
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['passed'] and audit['sources']==source_hashes()
    assert audit['verifier_sha256']==sha(Path(__file__).with_name('patrol_verify.py'))
    for n,h in audit['artifacts'].items():assert sha(OUT/n)==h,n
    protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['binary_sha256']==sha(NS3/'build/scratch/patrol-scenario')
    report=json.loads((OUT/'report_verification.json').read_text());assert report['passed']
    for name,key in [('report.md','report_sha256'),('primary_analysis.json','primary_analysis_sha256'),('analysis.json','analysis_sha256'),('independent_audit.json','independent_audit_sha256')]:assert sha(OUT/name)==report[key]
    assert report['generator_sha256']==sha(Path(__file__).with_name('patrol_report.py'))
    document=json.loads((ROOT/'slice_alloc/results/document_verification.json').read_text());assert document['passed']
    assert document['patrol_report_sha256']==sha(OUT/'report.md') and document['patrol_independent_audit_sha256']==sha(OUT/'independent_audit.json')
    assert document['design_source_sha256']==sha(ROOT/'slice_alloc/design.md')
    assert document['generator_sha256']==sha(Path(__file__).with_name('report.py'))
    for ext in ('md','docx'):assert document[ext+'_sha256']==sha(ROOT/f'面向低空智联的多模态网络切片技术.{ext}')
    old=json.loads((ROOT/'slice_alloc/results/joint/final_audit.json').read_text());assert old['passed']
    for n,h in old['sources'].items():assert sha(ROOT/n)==h,n
    for n in ('comparison','analysis','diagnostic_verification','allocation_diagnostic_verification'):assert old[n+'_sha256']==sha(ROOT/'slice_alloc/results/joint'/f'{n}.json')
    assert old['document_verification_sha256']==sha(ROOT/'slice_alloc/results/document_verification.json')
    text=(ROOT/'面向低空智联的多模态网络切片技术.md').read_text();links=[]
    for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        if '://' not in link:assert (ROOT/unquote(link)).exists(),link;links.append(link)
    with zipfile.ZipFile(ROOT/'面向低空智联的多模态网络切片技术.docx') as z:
        xml=z.read('word/document.xml').decode();assert all(s in xml for s in ('85.21%','75.75%','81.94%','115200','3600','640'))
    tests=(OUT/'delivery-tests.log').read_text();assert '\nOK' in tests
    count=int(re.search(r'Ran (\d+) tests',tests).group(1));assert count>=90
    extras=['README.md','.gitignore','slice_alloc/design.md','slice_alloc/tools/patrol_verify.py','slice_alloc/tools/patrol_validation.py','slice_alloc/tools/patrol_report.py','slice_alloc/tools/patrol_delivery.py','slice_alloc/tools/report.py','slice_alloc/tools/export_document.py','slice_alloc/results/document_verification.json','slice_alloc/results/joint/final_audit.json','面向低空智联的多模态网络切片技术.md','面向低空智联的多模态网络切片技术.docx']
    extras += [str(p.relative_to(ROOT)) for p in (ROOT/'图').glob('*.png')]+[str(p.relative_to(ROOT)) for p in (ROOT/'图').glob('*.svg')]+[str(p.relative_to(ROOT)) for p in (ROOT/'图/源文件').glob('*.dot')]
    extras += ['slice_alloc/results/patrol/'+n for n in ('independent_audit.json','report.md','report_verification.json','primary_analysis.json','cleanup.json','delivery-tests.log')]
    result=dict(passed=True,unit_tests=count,native_mechanism_cases=8,validation_episodes=640,formal_episodes=audit['episodes'],formal_request_decisions=audit['decisions'],exact_policy_native_replays=len(audit['policy_native_replays']),all_local_links_exist=True,source_and_experiment_consistent=True,manuscript_and_word_consistent=True,sources=source_hashes(),artifacts={p:sha(ROOT/p) for p in extras},scope='restricted three-modality patrol initial allocation; RL does not beat the strongest rule in this frozen study; later prediction/QoS research not claimed complete')
    write(OUT/'delivery.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('sources','artifacts')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
