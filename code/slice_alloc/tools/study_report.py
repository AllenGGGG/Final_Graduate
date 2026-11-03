"""Independent packet, queue, reservation and statistical audit of one study."""
from collections import Counter,defaultdict
import concurrent.futures as cf
import csv
import gzip
import json
from pathlib import Path
import re
import shutil
import numpy as np
import torch
from slice_alloc import study as s
from slice_alloc.forecast import Predictor,windows,baseline_predictions,metrics

def verify_episode(row):
 prefix=s.OUT/row['prefix'];cfg=s.read(str(prefix)+'-scenario.json');result=s.read(str(prefix)+'-result.json');summary=s.read(str(prefix)+'-summary.json')
 import hashlib
 assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==result['scenario_sha256']
 for key in ('requests','success','admitted','by_service'):assert result[key]==row[key]
 requests={r['id']:s.b.Service(**{**r,'members':tuple(r['members'])}) for r in cfg['requests']}
 decisions=result['decisions'];accepted={r['request']:r for r in decisions if r['decision']['accept']}
 assert len(decisions)==len(requests)==row['load'];assert len(accepted)==result['admitted']
 assert {v['id'] for v in result['settled']}==set(accepted)
 specs=queue_specs(cfg);capacities={(q['node'],q['peer']):q['capacity'] for q in specs}
 emitted={};emit_count=Counter();completed=set();ontime=Counter();queue_arrival=Counter();queue_departure=Counter();queue_drop=Counter();delays=[];class_bytes=Counter();last_end={};weight_changes=0
 with gzip.open(str(prefix)+'-events.csv.gz','rt') as stream:
  for e in csv.DictReader(stream):
   sid,seq,node=int(e['slice']),int(e['seq']),int(e['node']);event=e['event'];now=float(e['time']);size=int(e['bytes'])
   if event=='emit':
    assert (sid,seq) not in emitted;emitted[sid,seq]=now;emit_count[sid]+=1
    if requests[sid].service=='telemetry':assert node in requests[sid].members
   elif event=='complete':
    assert (sid,seq) in emitted and (sid,seq) not in completed;completed.add((sid,seq));r=requests[sid];delay=now-emitted[sid,seq];delays.append(delay)
    if delay<=r.deadline+1e-8:ontime[sid]+=1
    if r.service=='telemetry':assert node==0
   elif event=='queue_arrive':queue_arrival[node,seq]+=size
   elif event=='queue_depart':
    key=(node,seq);queue_departure[key]+=size;class_bytes[sid]+=size
    capacity=capacities[key]
    assert now+1e-8>=last_end.get(key,0),('physical link overserved',prefix,key)
    last_end[key]=now+size/capacity
    assert float(e['detail'])>=-1e-9
   elif event=='queue_drop':queue_drop[node,seq]+=size
   elif event=='queue_weight':
    assert int(e['bytes']) in (1,4);weight_changes+=1
 for settled in result['settled']:
  sid=settled['id'];r=requests[sid];expected=emit_count[sid]
  assert expected==settled['expected']==r.expected
  assert ontime[sid]==settled['on_time'];assert settled['success']==(ontime[sid]+1e-9>=expected*r.minimum_ratio)
 assert sum(x['success'] for x in result['settled'])==row['success']
 assert not any(summary[k] for k in ('remaining_mapping','remaining_mapcache','remaining_pending','remaining_application','remaining_cache_entries'))
 specs=queue_specs(cfg);loads=np.zeros(10)
 with gzip.open(str(prefix)+'-telemetry.jsonl.gz','rt') as stream:
  previous=0.
  for line in stream:
   tick=json.loads(line);assert abs(tick['time']-previous-.1)<1e-7;previous=tick['time']
   assert len(tick['load'])==len(tick['queue'])==10
   assert min(tick['load']+tick['queue'])>=0;loads+=tick['load']
 for i,q in enumerate(summary['queues']):
  key=q['node'],q['peer'];assert key==(specs[i]['node'],specs[i]['peer'])
  assert q['input']==q['served']+q['dropped']+q['remaining']
  assert queue_arrival[key]==q['input'];assert queue_departure[key]==q['served'];assert queue_drop[key]==q['dropped']
  assert abs(loads[i]*specs[i]['capacity']*.1-q['input'])<1e-4
 assert weight_changes==sum(q['weight_changes'] for q in summary['queues'])
 if cfg['public']['qos']=='none':assert weight_changes==0
 ledger=s.b.p.Ledger(cfg['public'])
 for message in result['observations']:
  obs=message['observation'];assert all(h['time']<=obs['time']+1e-8 for h in obs['load_history'])
  if obs['kind']=='done':continue
  d=message['decision'];r=requests[d['request_id']];ledger.release(obs['time']);assert 'rate_factor' not in d
  if not d['accept']:continue
  assert d['mode'] in s.base.COMPATIBLE[r.service]
  plan=s.b.p.compile_plan(r.bound('ip'),d['edge'],d['server'],0,ledger.pub,obs) if r.service=='telemetry' else s.b.compile_plan(r,d['mode'],d['edge'],d['server'],d['cache'],ledger.pub,obs)
  assert ledger.feasible(plan);ledger.add(plan)
 forced=sum(not r['decision']['accept'] and not r['feasible'] for r in decisions)
 voluntary=sum(not r['decision']['accept'] and bool(r['feasible']) for r in decisions)
 return {**row,'forced':forced,'voluntary':voluntary,'violations':row['admitted']-row['success'],
  'modes':result['modes'],'weight_changes':weight_changes,'decision_ms':result['decision_ms'],
  'queue_bytes':sum(q['served'] for q in summary['queues']),'queue_unclassified':sum(q['unclassified'] for q in summary['queues']),
  'input_sha256':hashlib.sha256(json.dumps({**cfg,'public':{k:v for k,v in cfg['public'].items() if k!='qos'}},sort_keys=True).encode()).hexdigest(),
  'mean_delivered_delay':float(np.mean(delays)) if delays else 0.}

def queue_specs(cfg):
 result=[]
 for a,b in s.WIRES:
  mbps=cfg['public']['backhaul_mbps'] if 0 in (a,b) else cfg['public']['cross_mbps'] if {a,b}=={1,2} else cfg['public']['access_mbps']
  result.append({'node':a,'peer':b,'capacity':mbps*1e6/8})
 return result

def summarize(rows):
 counts={k:sum(r[k] for r in rows) for k in ('requests','success','admitted','forced','voluntary','violations','weight_changes','queue_bytes','queue_unclassified')}
 counts.update(success_rate=counts['success']/counts['requests'],admission_rate=counts['admitted']/counts['requests'],
  decision_ms=float(np.mean([r['decision_ms'] for r in rows])),modes=dict(sum((Counter(r['modes']) for r in rows),Counter())))
 counts['by_service']={service:dict(requests=sum(r['by_service'][service]['requests'] for r in rows),success=sum(r['by_service'][service]['success'] for r in rows)) for service in s.b.SERVICES}
 counts['minimum_service_success_rate']=min(v['success']/v['requests'] for v in counts['by_service'].values())
 return counts

def paired(rows,left,right,mobile_only=False):
 chosen=[r for r in rows if not mobile_only or r['mobility']!='stationary'];seeds=list(s.TEST)
 def array(method):
  values=[r for r in chosen if r['method']==method];inits=s.INITS if method not in s.RULES else (0,)
  result=[]
  for init in inits:
   per=[]
   for seed in seeds:
    subset=[r for r in values if r['init']==init and r['seed']==seed]
    per.append(sum(r['success'] for r in subset)/sum(r['requests'] for r in subset))
   result.append(per)
  return np.asarray(result)
 a,b=array(left),array(right);rng=np.random.default_rng(20261103);draws=20000
 seed_draw=rng.integers(len(seeds),size=(draws,len(seeds)));init_draw=rng.integers(len(s.INITS),size=(draws,len(s.INITS)))
 aa=a[init_draw[:,:,None],seed_draw[:,None,:]].mean((1,2)) if len(a)>1 else a[0,seed_draw].mean(1)
 bb=b[init_draw[:,:,None],seed_draw[:,None,:]].mean((1,2)) if len(b)>1 else b[0,seed_draw].mean(1)
 delta=aa-bb
 p=min(1.,2*min((np.count_nonzero(delta<=0)+1)/(draws+1),(np.count_nonzero(delta>=0)+1)/(draws+1)))
 return dict(left=left,right=right,mobile_only=mobile_only,difference=float(a.mean()-b.mean()),ci95=np.quantile(delta,[.025,.975]).tolist(),bootstrap_two_sided_p=p)

def forecast_evaluation(rows):
 from slice_alloc.forecast import GRU
 model=Predictor(s.OUT/'forecast.pt').model;cal=s.read(s.OUT/'forecast_complete.json')['baseline_p90_residual']
 records=[]
 # One fixed initialization of A supplies an independent frozen-policy test
 # distribution; no model or baseline is selected using these future labels.
 for row in rows:
  if row['method']!='A' or row['init']!=s.INITS[0]:continue
  x,y,_=windows(str(s.OUT/row['prefix'])+'-telemetry.jsonl.gz')
  with torch.inference_mode():pred=torch.cat([model(torch.from_numpy(x[k:k+1024])) for k in range(0,len(x),1024)]).numpy()
  outputs={'gru':pred}
  for name,point in baseline_predictions(x).items():outputs[name]=np.stack([point,point+np.asarray(cal[name])],-1)
  records.append(dict(seed=row['seed'],load=row['load'],mobility=row['mobility'],windows=len(x),methods={name:metrics(p,y) for name,p in outputs.items()}))
 total=sum(r['windows'] for r in records)
 return dict(test_windows=total,test_episodes=len(records),methods={name:{metric:sum(r['windows']*r['methods'][name][metric] for r in records)/total for metric in records[0]['methods'][name]} for name in records[0]['methods']},episodes=records)

def report(workers=8):
 torch.set_num_threads(1);protocol=s.read(s.OUT/'protocol.json')
 for name,digest in protocol['sources'].items():assert s.sha(s.PACKAGE/name)==digest,name
 assert s.sha(s.b.p.NS3/'build/scratch/allocation-final-scenario')==protocol['binary_sha256']
 for name,digest in protocol['native_libraries'].items():assert s.sha(s.b.p.NS3/'build/lib'/name)==digest,name
 for name,digest in s.read(s.OUT/'test_started.json').items():assert s.sha(s.OUT/name)==digest,name
 training=s.read(s.OUT/'training_complete.json');actual=[]
 from slice_alloc.allocation_value import MaskedDoubleDQN
 assert {(r['variant'],r['init']) for r in training}==set(__import__('itertools').product(s.VARIANTS,s.INITS))
 for model in training:
  assert model['steps']==max(s.BUDGETS) and model['updates']>0
  assert s.sha(s.OUT/model['model'])==model['sha256']
  loaded=MaskedDoubleDQN.load(s.OUT/model['model'],device='cpu')
  assert loaded.num_timesteps==model['selected']['step'] and loaded.gamma==1 and loaded.quantile_net.n_quantiles==1
  assert loaded.learning_rate==protocol['hyperparameters']['learning_rate']
  actual.append({'variant':model['variant'],'init':model['init'],'saved_steps':loaded.num_timesteps,'learning_rate':loaded.learning_rate,'gamma':loaded.gamma,'outputs':loaded.quantile_net.n_quantiles})
  assert model['selected']==max(model['checkpoints'],key=lambda r:(r['validation']['rate'],-r['step']))
  assert all(51000000<=v<52000000 for v in model['training_seeds'])
 rows=s.read(s.OUT/'evaluation.json')
 expected={(method,init,seed,mobility,load) for method in (*s.VARIANTS,'AP_passive','AP_predictive',*s.RULES)
  for init in (s.INITS if method not in s.RULES else (0,)) for seed in s.TEST for mobility in s.MOBILITY for load in s.LOADS
  if not(method=='no_motion' and mobility=='stationary')}
 assert len(rows)==len(expected) and {(r['method'],r['init'],r['seed'],r['mobility'],r['load']) for r in rows}==expected
 with cf.ProcessPoolExecutor(max_workers=workers) as pool:verified=list(pool.map(verify_episode,rows,chunksize=8))
 s.write(s.OUT/'actual_parameters.json',actual)
 shared=defaultdict(set)
 for row in verified:shared[row['seed'],row['mobility'],row['load']].add(row['input_sha256'])
 assert all(len(v)==1 for v in shared.values()),'methods did not share scenarios'
 methods=(*s.VARIANTS,'AP_passive','AP_predictive',*s.RULES)
 analysis={'methods':{m:summarize([r for r in verified if r['method']==m]) for m in methods},'primary':paired(verified,'A','modal_greedy')}
 secondary=[paired(verified,'A','fixed'),paired(verified,'A','no_motion',True),paired(verified,'AP','A'),paired(verified,'AP_predictive','AP_passive')]
 order=sorted(range(len(secondary)),key=lambda i:secondary[i]['bootstrap_two_sided_p']);last=0.
 for rank,i in enumerate(order):
  last=max(last,min(1.,(len(order)-rank)*secondary[i]['bootstrap_two_sided_p']));secondary[i]['holm_adjusted_p']=last
 analysis['secondary']=secondary
 analysis['by_load']={str(l):{m:summarize([r for r in verified if r['method']==m and r['load']==l]) for m in methods} for l in s.LOADS}
 analysis['by_mobility']={mob:{m:summarize([r for r in verified if r['method']==m and r['mobility']==mob]) for m in methods if not(m=='no_motion' and mob=='stationary')} for mob in s.MOBILITY}
 analysis['initializations']={m:{str(i):summarize([r for r in verified if r['method']==m and r['init']==i]) for i in s.INITS} for m in (*s.VARIANTS,'AP_passive','AP_predictive')}
 analysis['initialization_summary']={m:{'median':float(np.median([v['success_rate'] for v in values.values()])), 'minimum':min(v['success_rate'] for v in values.values()),'maximum':max(v['success_rate'] for v in values.values())} for m,values in analysis['initializations'].items()}
 s.write(s.OUT/'analysis.json',analysis);s.write(s.OUT/'verified_episodes.json',verified)
 forecast=forecast_evaluation(rows);s.write(s.OUT/'forecast_evaluation.json',forecast)
 log=(s.OUT/'tests.log').read_text();assert '\nOK\n' in log
 verification=dict(passed_tests=int(re.search(r'Ran (\d+) tests',log)[1]),verified_native_episodes=len(verified),all_packet_audits_pass=True,
  all_direction_audits_pass=True,all_reservation_replays_pass=True,queue_conservation_pass=True,physical_serialization_pass=True,
  all_initializations_reported=True,all_methods_share_scenarios=True,actual_parameters_verified=True,training_models=len(training),training_steps_per_model=max(s.BUDGETS),protocol_sha256=s.sha(s.OUT/'protocol.json'))
 s.write(s.OUT/'verification.json',verification)
 text=['# 面向低空智联的多模态网络切片技术：最终验证结果','',
  '本报告仅使用本轮重新训练和新测试种子的结果。旧版79.29%不作为本轮成绩。低空巡检场景采用12架无人机、双AP、双边缘、固定80米高度与二维巡检航线；不主张低空信道模型创新。','',
  '## 实现与实验协议','',
  '第一部分：单个Masked Double-DQN选择准入、模态及映射/缓存位置，无档位、超订或拉格朗日惩罚。奖励为原生逐包交付重建的SLA成功数。遥测只支持IP，命令支持IP/ID，地图支持IP/NDN。','',
  '所有方法共用设备级DRR队列，IP、ID与原生NDN统一调度；默认权重1、增强权重4、最小量子512字节、共享100包缓冲。预约账本不是确定性SLA保证。缓存共享报文只归账一次，归于最早活跃的匹配切片；这不是对每个受益切片单独保证服务。','',
  '第二部分：每100ms采样实际到达设备队列的字节量，过去3秒预测未来1秒的10方向负载率P50/P90，使用GRU(隐藏32)。A+P在每次请求到达时使用预测重新训练；不是把未来真实值放进状态。','',
  '第三部分：每100ms检查队首等待时间。被动控制在等待超过0.25倍截止时间时提高权重；预测控制额外在P90>0.8且等待超过0.1倍截止时间时提高权重。冷却0.5秒，权重最低1。预测在准入事件更新，超过1秒失效后回退被动控制。控制对象仅为有线队列，无法修复无线失联。','',
  '32/64/96请求，驻留/局部/跨AP移动混合训练、分条件测试。四类RL各8初始化，每个65536决策；32768/65536检查点只按24个验证种子选择。60个新测试种子，每种9个负载/移动组合。无移动观测消融只测试移动条件。模态感知Greedy预注册24组参数，在同一验证集选择。','',
  '## 正式测试','', '| 方法 | SLA成功率 | 接纳率 | 最低分业务成功率 |','|---|---:|---:|---:|']
 for m,v in analysis['methods'].items():text.append(f"| {m} | {v['success_rate']:.2%} | {v['admission_rate']:.2%} | {v['minimum_service_success_rate']:.2%} |")
 def comparison_line(c):
  low,high=c['ci95'];return f"{c['left']} − {c['right']}：{100*c['difference']:+.2f}个百分点，95%区间[{100*low:+.2f}, {100*high:+.2f}]。"
 text+=['',comparison_line(analysis['primary']),'','主比较同时重采样场景种子和训练初始化。次要比较采用相同场景/初始化配对，bootstrap双侧概率作Holm校正；详细数值见analysis.json。','']
 text.extend(comparison_line(c)+f" Holm校正p={c['holm_adjusted_p']:.4f}。" for c in secondary)
 text+=['','## 预测精度','','| 方法 | MAE（负载率） | 分位数损失 | P90覆盖率 |','|---|---:|---:|---:|']
 for name,v in forecast['methods'].items():text.append(f"| {name} | {v['mae']:.4f} | {v['pinball']:.4f} | {v['p90_coverage']:.2%} |")
 c=analysis['primary'];claim='本轮支持RL优于所比较的模态感知规则。' if c['ci95'][0]>0 else '本轮尚不能证明RL优于模态感知规则。' if c['ci95'][1]>=0 else '本轮RL低于模态感知规则。'
 text+=['','## 结论与范围','',claim,'','预测和QoS的收益分别依据上述配对比较判断；区间跨零时不宣称稳定提升。记录各业务最小成功率是诊断，不代表加入了公平性保证。初始化区间、逐业务、逐负载、逐移动条件以及调整开销全部保留在analysis.json。','',
  f"已独立核验{len(verified)}个正式原生回合的逐包SLA、遥测方向、预约可行性、队列字节守恒及物理序列化。{verification['passed_tests']}项自动化测试通过。模型和原始轨迹本地保存，仓库发布配置、统计、报告和哈希；复现见README。"]
 (s.OUT/'report.md').write_text('\n'.join(text)+'\n')
 plot(analysis)
 manifest={str(p.relative_to(s.OUT)):s.sha(p) for p in sorted(s.OUT.rglob('*')) if p.is_file() and p.name not in ('manifest.json','pipeline.log')}
 s.write(s.OUT/'manifest.json',manifest)
 print('\n'.join(text),flush=True)

def plot(analysis):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 names=['A','AP','AP_passive','AP_predictive','modal_greedy','native','greedy','first_fit']
 fig,axes=plt.subplots(1,2,figsize=(12,4.5))
 axes[0].barh(names,[100*analysis['methods'][m]['success_rate'] for m in names]);axes[0].set_xlabel('SLA success (%)');axes[0].invert_yaxis()
 for m in names:axes[1].plot(s.LOADS,[100*analysis['by_load'][str(l)][m]['success_rate'] for l in s.LOADS],marker='o',label=m)
 axes[1].set_xlabel('Requests');axes[1].set_ylabel('SLA success (%)');axes[1].legend(fontsize=8);fig.tight_layout()
 for ext in ('png','pdf'):fig.savefig(s.OUT/f'comparison.{ext}',dpi=180)
 plt.close(fig)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args();report(args.workers)
