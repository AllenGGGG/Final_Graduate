"""Delete superseded studies only after the complete new study was audited."""
import shutil
from slice_alloc import study as s

def finalize():
 verification=s.read(s.OUT/'verification.json')
 assert verification['verified_native_episodes']>20000 and verification['queue_conservation_pass'] and verification['physical_serialization_pass']
 assert verification['training_models']==32
 from slice_alloc.tools.study_run import preregister
 preregister()
 removed=[]
 results=s.PACKAGE/'results'
 for child in results.iterdir():
  if child.resolve()==s.OUT:continue
  removed.append(str(child.relative_to(s.PACKAGE)))
  if child.is_dir():shutil.rmtree(child)
  else:child.unlink()
 old_files=['SIMPLE_DESIGN.md','REFINE_PROTOCOL.md','tools/allocation_simple_run.py','tools/allocation_simple_report.py','tools/allocation_refine_run.py','tools/allocation_refine_report.py','tools/modal_binding_run.py','tests/test_allocation_simple.py','tests/test_allocation_refine.py']
 for name in old_files:
  p=s.PACKAGE/name
  if p.exists():removed.append(name);p.unlink()
 for p in s.PACKAGE.rglob('__pycache__'):shutil.rmtree(p)
 destination=results/'final'
 if s.OUT!=destination:s.OUT.rename(destination)
 s.write(destination/'cleanup.json',{'removed':removed,'only_result_directory':'results/final','completed_audit_before_deletion':True})
 export=s.PACKAGE/'final'
 if export.exists():shutil.rmtree(export)
 export.mkdir()
 for name in ['report.md','analysis.json','verification.json','protocol.json','forecast_evaluation.json','forecast_complete.json','capacity.json','baseline_selection.json','cleanup.json','comparison.png','comparison.pdf']:
  shutil.copy2(destination/name,export/name)
 manifest={str(p.relative_to(destination)):s.sha(p) for p in sorted(destination.rglob('*')) if p.is_file() and p.name not in ('manifest.json','pipeline.log')}
 s.write(destination/'manifest.json',manifest);shutil.copy2(destination/'manifest.json',export/'manifest.json')
 (export/'README.md').write_text('# 面向低空智联的多模态网络切片技术\n\n唯一最终结果：[完整报告](report.md)、[统计](analysis.json)、[独立验证](verification.json)、[预测精度](forecast_evaluation.json)。\n\n三十二个RL模型均已训练完成，正式测试与逐包/队列审计完整运行。模型和逐包原始数据保留于本地results/final，不上传Git。\n\n复核：code目录运行 `python -m slice_alloc verify --workers 8`。从头复现：先设置LOWALT_NS3_ROOT，运行 `python -m slice_alloc build`，随后设置LOWALT_OUTPUT为新的空目录并运行 `python -m slice_alloc all --workers 8`。需要requirements.txt所列依赖。\n')
 (s.PACKAGE.parent.parent/'README.md').write_text('# 面向低空智联的多模态网络切片技术\n\n低空巡检环境 → IP/EID-RLOC/NDN多模态网络 → 智能分配 → 实际负载预测 → QoS队列控制。\n\n[唯一最终设计与验证结果](code/slice_alloc/final/README.md)\n\n[实施设计](code/slice_alloc/DESIGN.md)\n')
 print('FINALIZED',destination,'removed',len(removed),flush=True)

if __name__=='__main__':finalize()
