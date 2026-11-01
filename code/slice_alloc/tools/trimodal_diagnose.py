"""Explain the observed new-platform result without tuning fees or selecting seeds."""
from collections import Counter
import json
import re
import numpy as np
from .trimodal_run import OUT, PROTOCOL, profile, write
from .platform_verify import sha
from ..trimodal import Request, Ledger, compile_plan, RESOURCE_KEYS


def main():
    analysis=json.loads((OUT/'analysis.json').read_text())
    independent=json.loads((OUT/'independent_audit.json').read_text())
    decisions=json.loads((OUT/'decisions.json').read_text())['rows']
    rejected=0;candidate_violations=Counter();rejection_violations=Counter();active_rejections=0
    selections={m:Counter() for m in ('ip','id','ndn')}
    # Wireless repetitions have identical policy decisions. Count each allocation
    # once, retaining all methods and training initializations.
    for d in decisions:
        if d['radio_repeat']!=0:continue
        ledger=Ledger()
        for x in d['decisions']:
            q=dict(x['request']);q['attachments']=tuple(q['attachments']);r=Request(**q)
            if x['selected'] is not None:
                p=compile_plan(r,*x['selected'],profile=profile());ledger.commit(p)
                selections[r.mode][str(tuple(x['selected']))]+=1
            else:
                rejected+=1
                choices=[(e,s,0) for e in (1,2) for s in (1,2)] if r.mode=='id' else [
                    (e,e,c) for e in (1,2) for c in ((0,e) if r.mode=='ndn' else (0,))]
                union=set();any_feasible=False
                for config in choices:
                    p=compile_plan(r,*config,profile=profile())
                    times={r.start}|{t for other in ledger.plans for t in (other.request.start,other.request.drain) if r.start<=t<r.drain}
                    bad={RESOURCE_KEYS[i] for t in times for i,v in enumerate(ledger.usage(t)+p.usage-ledger.capacity) if v>1e-9}
                    candidate_violations.update(bad);union|=bad
                    if not bad:any_feasible=True
                rejection_violations.update(union)
                active_rejections+=any_feasible
    only_airtime=set(rejection_violations)=={'airtime:shared'}
    methods=analysis['scores']
    max_admission_spread=max(v['admission'] for v in methods.values())-min(v['admission'] for v in methods.values())
    result=dict(source_sha256=sha(__file__),analysis_sha256=sha(OUT/'analysis.json'),independent_audit_sha256=sha(OUT/'independent_audit.json'),
        distinct_allocation_runs=len(decisions)//len(PROTOCOL['radio_repeats']),rejected=rejected,policy_rejected_with_feasible_candidate=active_rejections,
        rejected_candidate_resource_violations=dict(candidate_violations),rejection_resource_violations=dict(rejection_violations),
        only_airtime_limits_rejected_requests=only_airtime,model_admission_spread=max_admission_spread,
        selections={m:dict(c) for m,c in selections.items()},
        conclusion='The execution prototype is verified; current parameters do not establish a reinforcement-learning or joint-mapping performance advantage.')
    write(OUT/'diagnosis.json',result)
    mechanics=json.loads((OUT/'mechanisms.json').read_text())
    test_count=int(re.search(r'Ran (\d+) tests', (OUT/'tests.log').read_text()).group(1))
    lines=['## 独立核验与结果限制','',
        f'完整回归通过{test_count}项测试；{len(mechanics["results"])}个机制/规模用例通过{len(mechanics["checks"])}条机制验收项。主实验2700次重放全部完成；独立检查从原始事件重构{independent["raw_cases"]}个场景结果，并从保存权重复现{independent["policy_checkpoint_replays"]}份决策记录。',
        f'另有{independent["stable_eid_handover_cases"]}个独立ID机制用例证明内层EID在真实AP切换前后不变，且两侧均完成交付。',
        f'主实验中实际观测到双AP切换的重放为{independent["scenarios_with_actual_handovers"]}次；这是重复执行次数，不是独立场景数。',
        f'链路发送工作量与PHY积分检查发现{independent["measured_resource_integral_violation_cases"]}个超过模型积分预算的重放。此检查不能推出瞬时或普适无线硬保障。','',
        '必须保留的研究结论：本批实验中RL没有超过强规则。所有方法的模型接纳率相同，包级差别很小；不能把机制闭环通过写成强化学习性能优势。']
    if only_airtime:
        lines+=['',f'对{rejected}次拒绝逐候选重查，实际限制只有共享空口，未出现有线带宽、映射表或缓存容量成为拒绝瓶颈。不同部署虽改变真实路径和费用，本批目标对这些取舍不敏感，尚不能用这组参数证明联合映射收益。']
    lines+=['','后续如研究多资源瓶颈，应在测试前声明独立的带宽/状态/缓存压力轴及业务合同，重新冻结协议；不能为了让DDQN或ID获胜调整费用或筛选测试种子。','',
            '完整数值：[analysis.json](analysis.json)；瓶颈诊断：[diagnosis.json](diagnosis.json)；独立核验：[independent_audit.json](independent_audit.json)。','']
    # Report generator rewrites these local links when inserting at the root.
    (OUT/'validation.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k not in ('selections',)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
