"""Independent recomputation, representative replay and report for ensemble study."""
import gzip,tempfile
from pathlib import Path
import numpy as np
import torch
from slice_alloc.tools import distributional_run as d
from slice_alloc.tools.qrdqn_ensemble_probe import Ensemble
from slice_alloc.distributional_allocation import MaskedDoubleQRDQN
from slice_alloc.tools.export_document import to_docx

OUT=d.ROOT/'slice_alloc/results/qrdqn_ensemble'

def verify():
 torch.set_num_threads(1)
 if (OUT/'delivery.json').exists() and d.read(OUT/'delivery.json').get('documents'):
  raise RuntimeError('Published delivery is immutable. Use python -m slice_alloc.tools.qrdqn_followup_verify for read-only verification.')
 plan=d.read(OUT/'holdout_plan.json');analysis=d.read(OUT/'analysis.json');rows=d.read(OUT/'evaluation.json');raw=d.read(OUT/'raw_audit.json')
 assert analysis['passed'] and plan['sources']==d.hashes()
 assert d.sha(d.ROOT/'slice_alloc/tools/qrdqn_ensemble_evaluate.py')==plan['script_sha256']
 assert d.sha(d.ROOT/'slice_alloc/tools/qrdqn_ensemble_probe.py')==plan['ensemble_script_sha256']
 assert analysis['plan_sha256']==d.sha(OUT/'holdout_plan.json')
 assert len(rows)==len(raw)==560
 assert sum(r['requests'] for r in rows)==40320
 for row,check in zip(rows,raw):assert row['path']==check['path'] and row['success']==check['request_success'] and check['encoded_observations_in_range']
 extra=d.read(OUT/'additional_rules_evaluation.json');extra_raw=d.read(OUT/'additional_rules_raw_audit.json')
 assert len(extra)==len(extra_raw)==240
 inputs={}
 for row in rows+extra:
  h=d.sha(OUT/'native'/(row['path']+'-scenario.json'))
  assert inputs.setdefault((row['seed'],row['load']),h)==h
 for row,check in zip(extra,extra_raw):
  assert row['path']==check['path'] and row['success']==check['request_success']
 extra_stats=d.read(OUT/'additional_rules.json')['stats']
 for method,stats in extra_stats.items():
  group=[r for r in extra if r['method']==method]
  assert sum(r['success'] for r in group)/sum(r['requests'] for r in group)==stats['success_rate']
 vectors={}
 for method,stats in analysis['stats'].items():
  group=[r for r in rows if r['method']==method];total=sum(r['requests'] for r in group)
  assert total==stats['requests'] and sum(r['success'] for r in group)/total==stats['success_rate']
  vectors[method]=np.array([sum(r['success'] for r in group if r['seed']==s)/sum(r['requests'] for r in group if r['seed']==s) for s in plan['test_seeds']])
 rng=np.random.default_rng(1170040)
 for method,c in analysis['comparisons'].items():
  delta=vectors['ensemble']-vectors[method]
  assert abs(float(delta.mean())-c['difference'])<1e-12
  samples=[float(delta[rng.integers(0,len(delta),len(delta))].mean()) for _ in range(10000)]
  np.testing.assert_allclose(np.quantile(samples,[.025,.975]),c['ci95'],rtol=0,atol=1e-12)
 models=[]
 for r in plan['models']:
  assert d.sha(d.ROOT/r['path'])==r['sha256'];models.append(MaskedDoubleQRDQN.load(d.ROOT/r['path'],device='cpu'))
 model=Ensemble(models,'mean');example=next(r for r in rows if r['method']=='ensemble')
 prefix=OUT/'native'/example['path'];cfg=d.read(str(prefix)+'-scenario.json');result=d.read(str(prefix)+'-result.json')
 with tempfile.TemporaryDirectory(prefix='ensemble-replay-') as tmp:
  output=Path(tmp)/'case';repeat=d.episode(cfg,'qrdqn',model,output)
  for field in ('observations','decisions','settled'):assert repeat[field]==result[field]
  with gzip.open(str(prefix)+'-events.csv.gz','rb') as a,gzip.open(str(output)+'-events.csv.gz','rb') as b:assert a.read()==b.read()
 testtext=(OUT/'tests.log').read_text();assert 'Ran 136 tests' in testtext and testtext.rstrip().endswith('OK')
 verification=dict(passed=True,statistics_independently_recomputed=True,models_unchanged=True,exact_ensemble_model_replay=True,
  raw_audited_episodes=800,decisions=57600,tests=136,conditional_on_fixed_five_models=True)
 d.write(OUT/'verification.json',verification)
 lines=['# 面向低空智联的多模态网络切片技术','',
 '第一部分智能分配的补充优化测试。题目和三项研究内容保持不变；环境、容量、SLA、业务语义及原生协议执行未修改。','',
 '本轮先修复同一训练轨迹两档预算的配对重采样。原有QR-DQN 78.54%及其与强规则均值相同的结论不变。修订后的长／短预算差值区间为[-0.09,+0.45]个百分点，仍跨零。','',
 '## 优化方法与预先选择','',
 '以已有5个独立初始化的QR-DQN模型，对合法候选的学习价值做等权平均后选动作；候选价格评分仍被删除。它是通用集成估计，未引入资源规则、固定拒绝偏置或奖励塑形，不宣称原创算法。没有新增训练；但部署该策略要依赖5条训练轨迹，其训练预算上限合计655360步，相当于单个131072步模型的5倍。存储和推理成本也高于单模型，不能将收益描述成同单模型计算预算下的优势。','',
 '在11600—11615新验证种子上预先比较均值与投票，仅选均值进入11700—11739共40个新留出种子的正式验收。每种场景64／80请求均运行。新测试不参与方案或模型选择。比较对象为相同的5个单模型平均表现和资源价格规则；各自完整执行独立资源历史，输入场景相同。','',
 '| 方法 | SLA成功率 | 接纳率 | 平均决策耗时（毫秒） |','|---|---:|---:|---:|']
 labels=dict(ensemble='5模型学习价值平均',single='原QR-DQN单模型平均',marginal_price='资源价格规则')
 for m,s in analysis['stats'].items():lines.append(f"| {labels[m]} | {100*s['success_rate']:.2f}% | {100*s['admission_rate']:.2f}% | {s['mean_decision_ms']:.3f} |")
 for m,label in [('near_edge','近边优先'),('greedy','资源贪心'),('static_quota','固定配额规则')]:
  s=extra_stats[m];lines.append(f"| {label} | {100*s['success_rate']:.2f}% | {100*s['admission_rate']:.2f}% | {s['mean_decision_ms']:.3f} |")
 lines+=['','主指标分母为全部到达请求；拒绝和接纳后未满足SLA均计失败。新旧测试集不同，不把这里与旧78.54%的直接差值视为优化收益。','']
 for m,c in analysis['comparisons'].items():
  lo,hi=np.array(c['ci95'])*100
  lines.append(f"- 平均策略相对{labels[m]}：{100*c['difference']:+.2f}个百分点，场景配对95%区间[{lo:+.2f},{hi:+.2f}]。")
 ci=analysis['comparisons']['marginal_price']['ci95']
 lines+=['',('本批新留出场景支持平均策略优于价格规则。' if ci[0]>0 else '本批新留出场景未证明平均策略优于价格规则。'),
 '区间针对这组固定的5个已训练模型，仅对40个场景种子重采样；不能声称已有5次独立集成训练，也不能据此证明跨训练种子组、跨拓扑或实际部署普遍领先。','',
 '## 核验与边界','',
 '136项测试通过，主对照560次及补充规则240次，合计800次场景／57600次决策全部逐包核验；所有方法场景指纹相同，模型和源码指纹一致，平均策略代表性场景精确回放通过。原5初始化从头重训的复现记录另见qrdqn_allocation/statistical_correction/retraining。','',
 '分模态成功率（平均学习策略／价格规则）：IP为60.29%／62.67%，ID为86.88%／85.63%，NDN为90.97%／85.56%。总体改善没有覆盖每一种模态，IP仍低于规则。平均推理耗时约2.61毫秒，单模型约0.91毫秒，价格规则约0.0045毫秒；这是本机观测，不能当成硬实时上界。','',
 '第一部分仍只决定初始接纳和功能／缓存／映射配置，AP关联由无线机制决定；资源预测和运行期QoS保障未在本轮实现。该结果不意味着身份模态或其他每类业务都分别获益。','',
 '[分析](analysis.json) · [冻结验收协议](holdout_plan.json) · [核验](verification.json) · [原修订结果](../qrdqn_allocation/final_result.md)']
 text='\n'.join(lines)+'\n';(OUT/'report.md').write_text(text);to_docx(text).save(OUT/'report.docx')
 d.write(OUT/'delivery.json',dict(passed=True,scope='Frozen-model ensemble held-out evaluation; first research part only',
  sources=plan['sources'],models=plan['models'],artifacts={n:d.sha(OUT/n) for n in ('plan.json','validation.json','holdout_plan.json','evaluation.json','raw_audit.json','additional_rules_evaluation.json','additional_rules_raw_audit.json','additional_rules.json','analysis.json','verification.json','tests.log','report.md','report.docx')},
  implementation={n:d.sha(d.ROOT/n) for n in ('slice_alloc/tools/qrdqn_ensemble_probe.py','slice_alloc/tools/qrdqn_ensemble_evaluate.py','slice_alloc/tools/qrdqn_ensemble_verify.py','slice_alloc/tests/test_qrdqn_ensemble.py')}))
 print(verification,flush=True)
 print(analysis['comparisons'],flush=True)

if __name__=='__main__':verify()
