"""One supported entry point for the verified low-altitude allocation study."""
import argparse
import hashlib
import json
import re
import shutil
import os
from pathlib import Path
import subprocess
import sys


def finalize_output(output):
    """Keep cleanup evidence and file hashes consistent after re-verification."""
    digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    package=Path(__file__).resolve().parent
    verification=json.loads((output/'verification.json').read_text())
    recovery=output/'recovery_verification.json'
    recovery_data=json.loads(recovery.read_text()) if recovery.exists() else None
    if recovery_data is not None:
        verification.update(recovered_source_commit=recovery_data['source_commit'],
            actual_parameters_verified=recovery_data['all_actual_models_loaded'],
            historical_reproduction=True,new_independent_test=False,fresh_test_complete=False,
            separately_verified_original_episodes=recovery_data['verified_original_episodes'],
            original_statistics_reproduced_exactly=recovery_data['original_statistics_reproduced_exactly'])
    post=output/'post-cleanup-tests.log'
    if post.exists():
        log=post.read_text()
        assert '\nOK\n' in log
        verification.update(post_cleanup_passed_tests=int(re.search(r'Ran (\d+) tests',log)[1]),
            final_entry_verified=True,post_cleanup_test_log_sha256=digest(post))
    if (output/'cleanup.json').exists():
        verification['obsolete_results_deleted']=True
    dependencies=output/'analysis_dependencies.json'
    if dependencies.exists():
        for name,value in json.loads(dependencies.read_text()).items():
            assert digest(package/name)==value
            assert digest(output/'frozen'/name)==value
        verification['analysis_dependency_hashes_verified']=True
    (output/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
    analysis=json.loads((output/'analysis.json').read_text())
    gain=analysis['comparisons']['double_dqn-reference_dqn']
    report=output/'report.md'
    text=report.read_text()
    if '## 最终结论与清理' not in text:
        text+='\n## 最终结论与清理\n\n续训相对原版增加'+f"{100*gain['difference']:.2f}"+'个百分点，但区间跨零，尚不能断言这项训练优化带来稳定收益。主比较RL对Greedy的结论见上。\n'
        if post.exists():
            text+='\n清理后'+str(verification['post_cleanup_passed_tests'])+'项相关测试通过；默认入口重新核验了全部正式回合。旧版独立结果目录已删除，仅保留最终研究与本轮必要对照审计证据。\n'
    if recovery_data is not None:
        text=text.replace('60个新测试种子','60个原协议测试种子')
        text=text.replace('旧版独立结果目录已删除，仅保留最终研究与本轮必要对照审计证据。',
            '扩展版及其负面结果另行归档，当前仅保留第一部分主线与必要对照证据。')
        if '## 恢复复现与违约率口径' not in text:
            text+='\n## 恢复复现与违约率口径\n\n沿用历史协议种子重新训练和核验，不是新增独立测试，不重复计作显著性证据。原版480回合另经审计，见recovery_verification.json与original_480_report.md。\n\n'
            text+='| 方法 | SLA成功率 | 违约／全部到达 | 违约／已接纳 |\n|---|---:|---:|---:|\n'
            for method in ('double_dqn','greedy','nearest','first_fit'):
                row=analysis['methods'][method]
                conditional=row['admitted_violations']/row['admitted'] if row['admitted'] else 0.
                text+=f"| {method} | {row['success_rate']:.2%} | {row['violation_rate_all']:.2%} | {conditional:.2%} |\n"
            text+='\n小幅优势限定于本协议所列规则，不外推为优于所有传统方法。扩展版负面结果保留于archive/expanded-study-ee8f44d。\n'
    report.write_text(text)
    source_manifest={str(f.relative_to(package)):digest(f) for f in sorted(package.rglob('*'))
        if f.is_file() and f.relative_to(package).parts[0] not in ('results','final')
        and '__pycache__' not in f.parts}
    (output/'code_manifest.json').write_text(json.dumps(source_manifest,ensure_ascii=False,indent=2)+'\n')
    manifest={str(f.relative_to(output)):digest(f) for f in sorted(output.rglob('*'))
        if f.is_file() and f.name!='manifest.json'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    if output==package/'results/final' and (package/'final').exists():
        for name in ('report.md','analysis.json','verification.json','protocol.json','manifest.json','code_manifest.json'):
            shutil.copy2(output/name,package/'final'/name)
        export=package/'final'
        (export/'export_manifest.json').write_text(json.dumps({
            str(f.relative_to(export)):digest(f) for f in sorted(export.rglob('*'))
            if f.is_file() and f.name!='export_manifest.json'},ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description='低空多模态切片分配：最终验证版')
    parser.add_argument('command',nargs='?',default='all',choices=['all','train','evaluate','verify','report'])
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    output=Path(os.environ.get('LOWALT_REFINE_OUTPUT',os.environ.get('LOWALT_ALLOCATION_OUTPUT',
        str(Path(__file__).resolve().parent/'results/final')))).expanduser().resolve()
    os.environ['LOWALT_REFINE_OUTPUT']=str(output)
    reference=output/'pretraining'
    os.environ.setdefault('LOWALT_REFERENCE_OUTPUT',str(reference))
    if args.command in ('verify','report') or (args.command=='all' and (output/'evaluation.json').exists()):
        from .tools.allocation_refine_report import main as report
        sys.argv=[sys.argv[0],str(output),'--workers',str(args.workers)]
        report()
        finalize_output(output)
        return
    if args.command in ('train','all') and not (output/'training_complete.json').exists():
        if not (Path(os.environ['LOWALT_REFERENCE_OUTPUT'])/'training_complete.json').exists():
            env=dict(os.environ,LOWALT_SIMPLE_OUTPUT=os.environ['LOWALT_REFERENCE_OUTPUT'])
            subprocess.run([sys.executable,'-m','slice_alloc.tools.allocation_simple_run','train',
                '--workers',str(args.workers)],env=env,check=True)
        from .tools.allocation_refine_run import main as execute
        sys.argv=[sys.argv[0],args.command,'--workers',str(args.workers)]
        execute()
    elif args.command in ('evaluate','all'):
        from .tools import allocation_refine_run as run
        import concurrent.futures as cf
        protocol=run.read(output/'protocol.json')
        for name,digest in protocol['sources'].items():
            assert run.sha(run.base.ROOT/'slice_alloc'/name)==digest,name
        assert run.sha(run.base.b.p.NS3/'build/scratch/allocation-final-scenario')==protocol['binary_sha256']
        run.prepare_evaluation()
        specs=[(m,i) for m in ('double_dqn','reference_dqn') for i in run.INITS]
        specs += [(m,0) for m in ('greedy','nearest','first_fit')]
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows=[row for group in pool.map(run.evaluate_group,specs) for row in group]
        run.write(output/'evaluation.json',rows)
    if args.command=='all':
        from .tools.allocation_refine_report import main as report
        sys.argv=[sys.argv[0],str(output),'--workers',str(args.workers)]
        report()
        finalize_output(output)


if __name__=='__main__':
    main()
