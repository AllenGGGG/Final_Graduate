"""Complete and publish one frozen low-altitude allocation experiment."""
import collections
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import unquote
from slice_alloc.tools import lowalt_allocation as study
from slice_alloc.tools import allocation_study as old
from slice_alloc.tools.patrol_run import sha,write
from slice_alloc.tools.export_document import to_docx

ROOT=study.ROOT
OUT=study.OUT
NAMES={'proposed':'本方法：低空多模态联合分配',
       'without_motion':'去掉移动观测（消融）','plain':'去掉先验（消融）',
       'previous_ppo':'上版冻结PPO','first_fit':'首次可行','near_edge':'近边优先',
       'greedy':'资源贪心','price':'价格规则','cache_aware':'缓存优先',
       'lifetime_price':'生命周期价格规则','marginal_price':'边际价格规则'}


def command(*args):
    print('RUN',*args,flush=True)
    subprocess.run([sys.executable,'-B',*args],check=True,cwd=ROOT)


def assert_frozen():
    p=json.loads((OUT/'protocol.json').read_text())
    assert p['sources']==study.hashes()
    assert p['finalizer_sha256']==sha(__file__)
    assert p['design_sha256']==sha(ROOT/'slice_alloc/lowalt_design_v4.md')
    assert p['development_sha256']==sha(OUT/'development.json')
    assert p['tests_sha256']==sha(ROOT/'slice_alloc/tests/test_lowalt_allocation.py')
    assert p['native_mechanism_regression_sha256']==sha(OUT/'current_mechanisms.json')
    regression=json.loads((OUT/'current_mechanisms.json').read_text())
    assert regression['passed'] and regression['binary_sha256']==p['binary_sha256']
    for c in p['checkpoints']:
        assert sha(ROOT/c['checkpoint'])==c['model_sha256']
        assert sha(ROOT/c['manifest'])==c['manifest_sha256']
        d=json.loads((ROOT/c['manifest']).read_text())
        assert d['steps']==32768
        if c['method']!='previous_ppo':
            assert d['sources']==study.hashes()
            assert d['demonstration_episodes']==d['reference_episodes']==0
            assert all(2000000<=s['seed']<3000000 for s in d['training_scenarios'])
    return p


def result_table(analysis,methods):
    rows=['| 方法 | 合计成功率 | 64请求 | 80请求 | 初始化标准差/百分点 |',
          '|---|---:|---:|---:|---:|']
    for method in methods:
        d=analysis['stats'][method]
        std='—' if d['initialization_std'] is None else f'{100*d["initialization_std"]:.3f}'
        rows.append(f'| {NAMES[method]} | {100*d["success_rate"]:.2f}% | {100*d["by_load"]["64"]:.2f}% | {100*d["by_load"]["80"]:.2f}% | {std} |')
    return '\n'.join(rows)


def make_report(analysis,protocol,evaluation):
    d=analysis['stats']['proposed'];best=max(protocol['baselines'],key=lambda m:analysis['stats'][m]['success_rate'])
    c=analysis['comparisons']['proposed'][best];lo,hi=c['initialization_and_scenario_ci95']
    if lo>0:judgement='相对最强规则的提升在本轮场景下得到区间支持。'
    elif c['difference']>0:judgement='均值高于最强规则，但区间包含0，不能宣称稳健领先。'
    else:judgement='均值仍低于最强规则，不能声称已证明学习领先。'
    rows=['# 低空智联三模态联合分配：最终实验结果','',
          f'本轮主方法真实SLA成功率为{100*d["success_rate"]:.2f}%。采用一个PPO策略，在相同原生IP／EID-RLOC／NDN基础设施中联合决定接纳和初始功能部署；算法名称不是研究目标，移动与共享资源的执行关系才是研究对象。','',
          '## 设计与实际修改','',
          '低空方面：当前位置和相邻历史观测估计的移动状态进入策略，原生无线层决定实际关联。多模态方面：三种固定语义竞争同一账本；资源价格按已签订合同的全部释放区间和物理共享缓存并集计算新增占用。已知到期日历不属于未来预测。',
          '双边缘、双AP、12架无人机、物理容量、模态比例、业务和SLA均与v3相同。主负载仍为64/80个申请在2—18秒到达，拒绝计失败。没有增加容量、放宽期限或改变分母。','',
          '## 主要结果','',result_table(analysis,('proposed','without_motion','plain','previous_ppo','first_fit','price','lifetime_price','marginal_price')),'',
          '## 相对提升与不确定性','',
          '| 本方法−对照 | 差值/百分点 | 初始化＋场景95%区间 |',
          '|---|---:|---|']
    for method in ('without_motion','plain','previous_ppo','price','lifetime_price','marginal_price'):
        x=analysis['comparisons']['proposed'][method];a,b=x['initialization_and_scenario_ci95']
        rows.append(f'| {NAMES[method]} | {100*x["difference"]:+.2f} | [{100*a:+.2f}, {100*b:+.2f}] |')
    rows+=['',f'最强规则为{NAMES[best]}，成功率{100*analysis["stats"][best]["success_rate"]:.2f}%。'
           f'本方法相对它的差值{100*c["difference"]:+.2f}个百分点，区间[{100*lo:+.2f}, {100*hi:+.2f}]。'+judgement,
           '达到75%—82%的合计目标、超过上版策略、超过传统强规则是不同判断。所有样本和初始化保留。','',
           '主方法五个初始化：'+' / '.join(f'{100*x:.2f}%' for x in d['initialization_rates'])+'。','',
           '## 三种模态与失败构成','',
           '| 模态 | 申请数 | 接纳数 | SLA成功数 | 成功率 |','|---|---:|---:|---:|---:|']
    group=[r for r in evaluation['rows'] if r['method']=='proposed']
    for mode in ('ip','id','ndn'):
        n=sum(r['by_mode'][mode]['requests'] for r in group)
        admitted=sum(r['by_mode'][mode]['admitted'] for r in group)
        success=sum(r['by_mode'][mode]['success'] for r in group)
        rows.append(f'| {mode} | {n} | {admitted} | {success} | {100*success/n:.2f}% |')
    n=sum(r['requests'] for r in group);admitted=sum(r['admitted'] for r in group);success=sum(r['success'] for r in group)
    rows+=['',f'主方法共{n}次申请，拒绝{n-admitted}次，接纳后SLA失败{admitted-success}次。'
           '三模态业务合同不同，分模态成绩不能作为协议优劣排名。','',
           '## 训练、留出与核验','',
           '主方法和两个消融均5个初始化、每模型32768次真实策略交互，原始SLA奖励，无示范预训练或额外参照执行。参数固定：学习率1e-4、rollout512、batch128、5轮更新、gamma=1、GAE lambda=1、entropy0.005、clip0.1。',
           f'训练种子2000000—2999999、开发7510—7517、最终8500—8529；最终测试冻结后仅执行一次。共{analysis["episodes"]}次正式仿真、{analysis["decisions"]}次申请决策。上版模型只作为冻结对照，读取相同当前原生场景。',
           '核验从原始产生/完成事件重建SLA，重验全部已接纳预约，检查所有方法输入一致及全部指纹，并重新加载20个检查点逐字节复现代表性原生事件。场景和初始化共同重采样生成95%区间；多方法比较为限定场景下探索性统计。','',
           '## 研究边界与后续接口','',
           '本轮完成初始分配。资源预测以后使用历史实际负载，QoS以后据预测与监测进行运行期调整；两者未实现，不预先赋予83%—89%或90%—93%成绩。累计消融必须保持相同业务、资源和SLA，使用保留的9000—9029种子。',
           '当前为固定高度、受限传播和有限边缘配置，无CPU处理硬保障或运行中主动迁移。平均发送工作量预约不等于无线逐包硬保障；控制器计算时仿真暂停，本轮不把墙钟计算延迟算成网络服务时延。','',
           '## 附录：全部对照','',result_table(analysis,list(analysis['stats'])),'',
           '[设计](../../lowalt_design_v4.md) · [协议](protocol.json) · [完整分析](analysis.json) · [逐包审计](audit.json) · [开发记录](development.json)']
    return '\n'.join(rows)+'\n'


def rebase(text,base):
    def replace(match):
        target=match.group(2)
        if '://' in target or target.startswith('#'):return match.group(0)
        name,sep,fragment=target.partition('#')
        new=str((base/unquote(name)).resolve().relative_to(ROOT.resolve()))
        return match.group(1)+new+(sep+fragment if sep else '')+')'
    return re.sub(r'(!?\[[^\]]*\]\()([^)]+)\)',replace,text)


def publish():
    p=assert_frozen();a=json.loads((OUT/'analysis.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
    assert audit['passed'] and audit['sources']==study.hashes()
    for name,value in audit['artifacts'].items():assert sha(OUT/name)==value
    assert a['evaluation_sha256']==sha(OUT/'evaluation.json') and a['protocol_sha256']==sha(OUT/'protocol.json')
    evaluation=json.loads((OUT/'evaluation.json').read_text())
    text=make_report(a,p,evaluation);(OUT/'report.md').write_text(text);to_docx(text).save(OUT/'report.docx')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=('proposed','without_motion','plain','previous_ppo','price','lifetime_price','marginal_price')
    labels=('Proposed','No motion','Plain PPO','Previous PPO','Price','Lifetime price','Marginal price')
    fig,ax=plt.subplots(figsize=(10,5));x=__import__('numpy').arange(len(methods))
    for j,load in enumerate(study.LOADS):
        ax.bar(x+(j-.5)*.36,[100*a['stats'][m]['by_load'][str(load)] for m in methods],width=.36,label=f'{load} arrivals')
    ax.set_ylim(0,100);ax.set_ylabel('SLA successes / all arrivals (%)');ax.set_xticks(x,labels,rotation=20,ha='right');ax.legend()
    ax.set_title('Low-altitude three-modality allocation, unchanged resources and SLA');fig.tight_layout()
    figure='图/08_低空三模态联合分配结果'
    for ext in ('.png','.svg'):fig.savefig(ROOT/(figure+ext),dpi=170)
    plt.close(fig)
    archive=OUT/'previous_document';archive.mkdir(exist_ok=True)
    title='面向低空智联的多模态网络切片技术'
    files=('README.md','slice_alloc/design.md',title+'.md',title+'.docx','slice_alloc/results/document_verification.json')
    for name in files:
        destination=archive/name;destination.parent.mkdir(exist_ok=True,parents=True)
        if not destination.exists():shutil.copyfile(ROOT/name,destination)
    (ROOT/'slice_alloc/design.md').write_text((ROOT/'slice_alloc/lowalt_design_v4.md').read_text())
    design=rebase((ROOT/'slice_alloc/design.md').read_text(),ROOT/'slice_alloc')
    manuscript=design+'\n'+rebase(text,OUT)+'\n![冻结实验结果]('+figure+'.png)\n'
    (ROOT/(title+'.md')).write_text(manuscript);to_docx(manuscript).save(ROOT/(title+'.docx'))
    primary=a['stats']['proposed']['success_rate'];best=max(study.BASELINES,key=lambda m:a['stats'][m]['success_rate'])
    readme=f'''# 面向低空智联的多模态网络切片技术

当前第一部分为低空巡检IP／EID-RLOC／NDN三模态共享资源的联合初始分配。双边缘、双AP、12架无人机；真实移动影响接入与定位，三模态竞争同一账本。主方法用一个PPO策略，加入已发生移动观测和精确边际预约成本。物理容量、业务、SLA及64/80请求负载沿用v3。

当前主方法成功率 **{100*primary:.2f}%**；上版冻结PPO {100*a['stats']['previous_ppo']['success_rate']:.2f}%；最强规则{NAMES[best]} {100*a['stats'][best]['success_rate']:.2f}%。具体差值及统计判断见报告。

[设计](slice_alloc/design.md) · [最终报告](slice_alloc/results/allocation_v4/report.md) · [Word正文]({title}.docx) · [冻结协议](slice_alloc/results/allocation_v4/protocol.json) · [逐包审计](slice_alloc/results/allocation_v4/audit.json) · [交付指纹](slice_alloc/results/allocation_v4/delivery.json)

正式执行{a['episodes']}次仿真、{a['decisions']}次请求决策。主方法与两个消融各5个初始化、每模型32768次真实交互，无模仿或参照预训练。全部旧版、消融和强规则均保留，主指标含拒绝失败。

## 复现与范围

使用sdn环境运行 `python -B -m unittest discover -s slice_alloc/tests -q`。入口为 `python -B -m slice_alloc.tools.lowalt_allocation`，完整流程为 `python -B -m slice_alloc.tools.lowalt_delivery`。冻结实验不允许在已打开留出集上重新调参；新研究须隔离版本和新留出集。

第一部分输出初始部署、预约和实际运行记录。资源预测和运行期QoS尚未实现，不赋予预期成绩；三部分最终累计消融保留9000—9029种子。当前范围为固定高度和受限信道，无CPU硬保障或运行期主动迁移。v3及更早结果保留为历史，不能混用不同测试集数值。
'''
    (ROOT/'README.md').write_text(readme)
    links=0
    for file in (ROOT/'README.md',ROOT/(title+'.md'),OUT/'report.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',file.read_text()):
            if '://' in target or target.startswith('#'):continue
            assert (file.parent/unquote(target.partition('#')[0])).exists(),(str(file),target)
            links+=1
    for filename in (OUT/'report.docx',ROOT/(title+'.docx')):
        with zipfile.ZipFile(filename) as z:
            xml=ET.fromstring(z.read('word/document.xml'));word=''.join(t.text or '' for t in xml.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'))
        for method,d in a['stats'].items():assert NAMES[method] in word and f'{100*d["success_rate"]:.2f}%' in word
    manuscript_files=['.gitignore','README.md','slice_alloc/design.md','slice_alloc/lowalt_design_v4.md',title+'.md',title+'.docx',figure+'.png',figure+'.svg','slice_alloc/tools/lowalt_delivery.py','slice_alloc/tools/export_document.py','slice_alloc/tests/test_lowalt_allocation.py']
    verification=dict(passed=True,scope='allocation-v4 current document',analysis_sha256=sha(OUT/'analysis.json'),
                      audit_sha256=sha(OUT/'audit.json'),protocol_sha256=sha(OUT/'protocol.json'),
                      md_sha256=sha(ROOT/(title+'.md')),docx_sha256=sha(ROOT/(title+'.docx')),local_links=links)
    write(ROOT/'slice_alloc/results/document_verification.json',verification)
    manuscript_files.append('slice_alloc/results/document_verification.json')
    write(OUT/'delivery.json',dict(passed=True,sources=study.hashes(),formal_episodes=a['episodes'],decisions=a['decisions'],
          primary_success=primary,source_and_artifact_hashes_match=True,word_scores_match=True,local_links=links,
          finalizer_sha256=sha(__file__),artifacts={name:sha(OUT/name) for name in ('protocol.json','analysis.json','audit.json','report.md','report.docx')},
          manuscript_artifacts={name:sha(ROOT/name) for name in manuscript_files},scope='First part only; prediction and run-time QoS not claimed'))
    print('Published all current documents with matching scores and fingerprints.',flush=True)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    if not (OUT/'protocol.json').exists():
        expected=[OUT/'models'/f'{m}-{i}.json' for m in study.METHODS for i in study.INITS]
        while True:
            ready=[]
            for file in expected:
                try:d=json.loads(file.read_text())
                except (FileNotFoundError,json.JSONDecodeError):continue
                assert d['sources']==study.hashes() and d['steps']==study.STEPS
                ready.append(file)
            if len(ready)==len(expected):break
            print('Fixed-budget model training complete',len(ready),'/',len(expected),flush=True);time.sleep(30)
        command('-m','slice_alloc.tools.lowalt_allocation','dev','--jobs','10')
        command('-m','slice_alloc.tools.lowalt_allocation','freeze')
        p=json.loads((OUT/'protocol.json').read_text());p['finalizer_sha256']=sha(__file__)
        p['versions']=dict(python=platform.python_version(),numpy=__import__('numpy').__version__,torch=__import__('torch').__version__,stable_baselines3=__import__('stable_baselines3').__version__)
        p['tests_sha256']=sha(ROOT/'slice_alloc/tests/test_lowalt_allocation.py')
        p['native_mechanism_regression_sha256']=sha(OUT/'current_mechanisms.json')
        write(OUT/'protocol.json',p)
    assert_frozen()
    if not (OUT/'evaluation.json').exists():command('-m','slice_alloc.tools.lowalt_allocation','evaluate','--jobs','10')
    if not (OUT/'analysis.json').exists():command('-m','slice_alloc.tools.lowalt_allocation','analyze')
    if not (OUT/'audit.json').exists():command('-m','slice_alloc.tools.lowalt_allocation','verify','--jobs','10')
    if not (OUT/'delivery.json').exists():publish()
    delivery=json.loads((OUT/'delivery.json').read_text());assert delivery['sources']==study.hashes()
    for group,base in ((delivery['artifacts'],OUT),(delivery['manuscript_artifacts'],ROOT)):
        for name,value in group.items():assert sha(base/name)==value,name
    print('COMPLETE',delivery['primary_success'],delivery['formal_episodes'],flush=True)


if __name__=='__main__':main()
