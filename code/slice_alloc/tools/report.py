"""Render the maintained design and separately verified four-node evidence."""
from pathlib import Path
from urllib.parse import unquote
import hashlib
import json
import re
import zipfile
from .export_document import to_docx
ROOT=Path(__file__).resolve().parents[2]

DESIGN_PATH = Path(__file__).resolve().parents[1] / 'design.md'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_evidence():
    path=ROOT/'slice_alloc/results/platform/verification.json'
    if not path.exists():return '原生协议集成记录尚不存在，不能宣称已通过执行验证。'
    proof=json.loads(path.read_text())
    platform=proof['platform']
    if not proof['passed'] or not proof['link_layer_comparison']['passed']:
        raise ValueError('native or controlled comparison failed; cannot issue a passing report')
    for relative,key in [('slice_alloc/tools/platform_verify.py','verification_driver_sha256'),
                         ('slice_alloc/protocol_contracts.py','contract_kernel_sha256')]:
        if sha(ROOT/relative)!=proof[key]:raise ValueError('stale native driver/contract: '+relative)
    source=ROOT/'slice_alloc/execution/protocol_probe.cc'
    if sha(source)!=platform['source_sha256']:raise ValueError('stale native protocol source evidence')
    for row in proof['results']:
        for name,digest in row['files'].items():
            if sha(path.parent/name)!=digest:raise ValueError('changed native protocol evidence: '+name)
    rows=['### 10.1 已实际运行的原生协议集成验证','',
          f"当前确认集共{proof['cases']}次ns-3包级运行、{proof['assertions']}条断言（包括跨实验配对校验）：原有60次回归用例使用11/12/13；新增3工况×3组×2协议×3种子共54次链路层对照，另3次错误邻居注入测试，新增确认种子41/42/43，开发种子21。总状态为{'通过' if proof['passed'] else '存在失败，须检查记录'}。下面先列原有集成场景；等价任务和pin对照另列，不能混称算法性能评测。",'',
          '| 场景 | 次数 | 有效NDN对象交付/次 | IP报文交付/次 | EID数据交付/次 |',
          '|---|---|---|---|---|']
    names={'wired_sync_cache':'有线同步、有缓存','wired_sync_cold':'有线同步、无缓存','wired_async_cold':'有线非同步、无缓存',
           'wifi_sync_cache':'共享802.11n、有缓存','wifi_sync_cold':'共享802.11n、无缓存','wifi_async_cold':'共享802.11n非同步、无缓存',
           'wifi_moving_cache':'共享802.11n、5m/s移动','wifi_outage_cache':'80m/s离开覆盖的负面对照'}
    def span(values):
        return str(min(values)) if min(values)==max(values) else f'{min(values)}—{max(values)}'
    for case,name in names.items():
        group=[r for r in proof['results'] if r['case']==case]
        rows.append(f"| {name} | {len(group)} | {span([r['ndn_received'] for r in group])}/6 | {span([r['ip_received'] for r in group])}/502 | {span([r['id_received'] for r in group])}/41 |")
    rows.extend(['',
      '每次用例另发送1次无路由名称请求，不能把它混入6次有效对象交付而声称零损失；有效对象大小为1000字节、单块，截止期5秒。该用例规模不等于计划中的双AP三模态主实验。覆盖内仍有真实无线重传，不能由最终交付推断无碰撞、无开销。',
      '',
      '有线同步用例在两个消费者上只产生1次上游Interest和1份上游Data，再由中继向两条下游分支发送；关闭缓存后重复请求重新回源，改为非同步冷请求时上游不再合并。数据版本变化和另一租户名称空间均触发对应回源，未知前缀不交付。有线首个同步对象的逐链路NDN网络层字节和逐节点逻辑发送次数均与独立合同核心一致。无线用例另外记录PhyTxBegin/End和PHY State TX，两条独立时钟的发送时长在1e-7秒容差内相等。聚合MPDU按同一设备同一PPDU区间去重；同时报告发送工作时长之和与信道区间并集，两者不混用。实测包含链路层控制、重传和前导开销，但不包含空闲/退避时延，也不构成可调度容量保证。此前PCAP没有radiotap速率元数据；本轮新PCAP启用radiotap，并以原生MonitorSnifferTx逐帧TxVector记录真实模式，以独立PHY时钟核对发送时长，不能只靠包长反推PHY速率。名称空间不同的验证不等于已实现独立PIT/FIB/CS容量隔离。',
      '',
      'EID/RLOC用例中，位置更新后新定位实际接收数据；旧定位缓存导致每次2个数据报失效，旧版本注册被拒绝。原位置信息仍在缓存期间出现中断是被保留的结果，不授予身份模态零中断。该定位切换是逻辑接入事件，物理移动由另一个无线轨迹用例验证，二者不混为完整移动管理实现。',
      '',
      '实际构建修复记录：ndn-cxx补入缺失的optional头；NFD补回sendData非void函数缺失的返回值；共享Wi-Fi明确创建AD_HOC Face，避免误用默认点对点Face导致其他消费者发回不合适的NACK。补丁、具体提交、编译器及库哈希均在验证记录中。',
      '',
      '开发种子1/2/3的首次24次测试保留了一个移动场景5/6交付的失败：用例原先约0.48秒就结束重试，未覆盖既定5秒合同。修正为按截止期重传并排空后，用未参与调试的11/12/13确认；原失败及源码快照保留在开发验证数据中，不删除或改写为通过。',
      '',
      '复跑命令：`/home/allen/miniconda3/envs/sdn/bin/python -B -m slice_alloc.tools.platform_verify`。首次需联网获取固定提交并在用户目录构建，之后复用同一平台；C++使用系统工具链，验证驱动使用sdn。',
      '',
      '[确认集与源码/二进制/日志哈希](slice_alloc/results/platform/verification.json)；[开发失败记录](slice_alloc/results/platform/development-verification.json)。',
      '',
      '旧四节点探针本身不覆盖的能力（新平台范围以前文为准）：一般分切片FIB/PIT/CS与CPU/空口配额强制执行；覆盖主规模条件的独立空口标定；缓存元数据/物理内存画像；完整初始映射与移动包络安装；动态低空场景的强化学习与独立对比；IP应用缓存/组播增强对照。最小适配器的局部进展见10.6；本节协议探针和数学排序检查不证明RL优势。'])
    rows.extend(['', '### 10.2 同任务协议对照与真实负面结果','',
        '每对实验使用相同内容名、版本、消费者、请求时刻、1000字节载荷和5秒截止期；去除IP背景流和EID背景流，分别运行IP UDP单播与NDN。所有15对均逐任务核对输入，双方各完成6/6，且全部交付早于最早任务截止期。这是IP无应用缓存的单播基线，不代表IP不能使用应用缓存/组播。', '',
        '| 场景（各3对） | IP生产者载荷/B | NDN生产者载荷/B | IP无线TX/ms | NDN无线TX/ms |',
        '|---|---|---|---|---|'])
    for case,title in [('wired_sync_cache','有线同步/缓存'),('wired_sync_cold','有线同步/冷缓存'),('wired_async_cold','有线非同步/冷缓存'),('wifi_sync_cache','无线同步/缓存'),('wifi_sync_cold','无线同步/冷缓存')]:
        pairs=[p for p in proof['equivalent_tasks']['pairs'] if p['case']==case]
        def radio(mode):
            values=[round(p[mode+'_radio']['tx_seconds_from_state']*1000,3) for p in pairs if p[mode+'_radio']]
            return span(values) if values else '不适用'
        rows.append(f"| {title} | {span([p['ip_producer_payload_bytes'] for p in pairs])} | {span([p['ndn_producer_payload_bytes'] for p in pairs])} | {radio('ip')} | {radio('ndn')} |")
    rows.extend(['',
        '有线同步共享可减少生产者回源载荷；非同步冷请求没有该收益。当前Wi-Fi下NDN实际TX时长高于IP单播，即使回源载荷减少也不能据此推断空口节省。当前NDN Face使用广播承载，IP使用单播，帧发送方式/实际速率、链路层控制和重试开销不同，表中保留该实现条件，不将结果泛化为协议本身必然高效或低效。空口时长为事后实测，不能直接充作新切片的确定性预约或RL收益。', '',
        '### 10.3 缓存责任归属与原生pin验证','',
        '合同检查覆盖重复读取去重、引用释放、不同版本/租户/切片计费、租约区间及容量冲突和过期快照。原生NFD另设一表项CS，在对象已加载后声明[1.05,1.5)租约、载荷预算1000字节，1.15秒插入另一对象造成真实淘汰压力；pin策略保留原对象，普通LRU对照淘汰原对象。原生快照的载荷占用与CacheLedger逐项相等，并记录编码Data字节；租约结束后1.7秒再插入对象，原对象可淘汰。6次pin/对照全部通过，但这不是一般多租户FIB/PIT/CS隔离，也不是1000字节物理内存的实测。', '',
        '### 10.4 本地版本与复核边界','',
        '本目录已建立独立Git仓库，基线源码、原始协议日志及PCAP、设计检查、MD/DOCX均受版本控制。首次PHY核对把同一PPDU内的MPDU时长重复相加而失败，该失败证据保留于提交e3c2205；随后按同无线设备聚合区间去重修复。Git记录本地回滚点，不向外部仓库发布；旧的大型模型及失效历史运行数据未纳入当前新方案证据。'])
    control=proof['link_layer_comparison']
    from statistics import mean
    rows.extend(['', '### 10.5 链路层受控对照与第一项设计判断（已实际执行）', '',
        '已实现逐邻居PeerUnicastTransport、接收过滤、逐帧TxVector与链路层分类，并运行54次正式受控比较（27对）及3次错误邻居注入测试。开发使用种子21；确认使用预先指定的41/42/43。它们属于协议资源画像验证，不是新分配算法或RL成绩。', '',
        '**实际实现。**官方NDN NetDeviceTransport向GetBroadcast()发送，不能只修改Face标签或URI就变成单播，见[R34](参考文献/仿真方法与平台/R34_ndnSIM_fixed_commit_NetDeviceTransport.cpp)。本课题新增适配器每邻居建立POINT_TO_POINT Face，显式向邻居MAC发送，接收时同时核对设备、协议号、源/目的MAC并拒绝OTHERHOST；多个Face仍共享同一个实际无线设备和信道。3次从非邻居发送的有效Interest均被过滤，未进入NFD；合法帧无重复注入。NFD的真实FIB/PIT/CS机制仍运行。', '',
        '三组实验分别是：①原生部署（IP单播、NDN广播，保留原PHY默认差异）；②共同PHY但NDN仍广播；③共同PHY且双方均单播。第②组控制DataMode/ControlMode/NonUnicastMode为ERP OFDM6 Mbps、关闭聚合并预置邻居，第③组沿用同一条件再启用NDN邻居单播。第②组与第①组不只改变一个参数，不能从两组差值算出“纯速率效应占比”；第③组结果也只适用于该适配器和固定PHY条件。非单播模式的默认回落逻辑见[R35](参考文献/仿真方法与平台/R35_ns3_companion_WifiRemoteStationManager.cc)。', '',
        '当前使用ConstantRateWifiManager，不能写成IP速率自适应。共同PHY组所有业务数据帧实测均为ERP6、NSS=1、GI=800ns、20MHz和相同前导、QoS TID，均无A-MPDU/A-MSDU。记录Data/Control/NonUnicastMode和聚合属性的运行时读回；该ns-3版本的MaxSsrc/MaxSlrc只有setter，元数据诚实标为requested而非伪造读回，两侧共同配置为7。原生组保留默认属性。实际MAC重试另计，不要求两个协议重试次数相同。', '',
        '每个协议仍执行同名/版本/消费者/请求时间/1000字节载荷/5秒期限的六个任务。新IP请求/响应基线在线路上携带4字节序号，提供与NDN相同的截止期和按节点119/122ms周期重请求预算，并拒绝重复响应；这些头字节纳入真实帧开销。全部27对各自6/6完成。原生IP一次发送的既有15对保持原口径，不与新可靠性基线混算。', '',
        '逐帧证据包括实际目的MAC类型、EtherType、PHY mode/bps、NSS、GI、带宽、前导、retry位、聚合信息和QoS TID。业务数据字节按单播/广播/组播分别统计，ARP和ACK等控制另列。MonitorSnifferTx聚合子帧包含分隔符/填充，本轮修正为先剥离后解码并分别计量；MPDU字节与PhyTxBegin逐项总量相等，聚合PPDU发送时长不重复累计。首次错误解码导致的失败保留在Git提交9f16133，不通过放宽字节对账规避。', '',
        '证据读取接口保留radio.link_layer，并在每对ip_radio/ndn_radio直接提供l2_delivery_mode、actual_phy_mode_histogram，避免按错层级读成空值。controls始终为一条完整字符串，并在导出前检查类型。新增implied_rate_mbps仅表示“实际发送MPDU字节×8÷全部发送者PHY累计TX秒数÷10^6”，包括已发送控制/重试MPDU；它不是应用吞吐、实测PHY速率或准入容量，也不使用信道区间并集作分母。真实速率以逐帧TxVector为准。同步缓存原生组的累计TX为133.064ms、信道区间并集为130.209ms；共同PHY广播组分别为22.158ms、21.685ms，不能混用。', '',
        '| 对象工况 | 组别 | IP PHY TX/ms | NDN PHY TX/ms | NDN/IP |',
        '|---|---|---|---|---|'])
    titles={'async_cold':'非同步冷缓存','sync_cold':'同步冷缓存','sync_cache':'同步重复/允许缓存'}
    groups={'native':'原生部署','common_phy_broadcast':'共同PHY、NDN广播','aligned_unicast':'共同PHY、双方单播'}
    for profile,title in titles.items():
        for group,label in groups.items():
            pairs=[r for r in control['pairs'] if r['profile']==profile and r['group']==group]
            ip=mean(r['ip_radio']['tx_seconds_from_state']*1000 for r in pairs)
            ndn=mean(r['ndn_radio']['tx_seconds_from_state']*1000 for r in pairs)
            rows.append(f"| {title} | {label} | {ip:.3f} | {ndn:.3f} | {ndn/ip:.4f} |")
    rows.extend(['',
        '表中是3个确认种子的均值，PHY时长为全体发送者的发送工作总量，包含帧头、控制、前导和重传，未把同时发送的工作合并掉；信道区间并集另在JSON保留。它不含退避/空闲等待，也不等同于时延或可预约容量。3种子的小规模固定场景只支持机制确认，不作普遍显著性结论。', '',
        '**如何归因。**原生组与共同PHY广播组均标记link_layer_conditions_aligned=false；只有双方单播的9对满足实际模式、目的类型、QoS、配置和任务对齐检查。原有6对无线等价任务也已显式标为不可直接识别协议效应。未使用“空口差异全部可归因于链路层”这种预设断言；两种模式对齐后剩余的是限定实现下的协议栈及报文机制综合差异，不是所有IP/NDN实现的固有代价。', '',
        '| 双方单播工况 | IP回源载荷/B | NDN回源载荷/B | IP平均交付/ms | NDN平均交付/ms |',
        '|---|---|---|---|---|'])
    for profile,title in titles.items():
        pairs=[r for r in control['pairs'] if r['profile']==profile and r['group']=='aligned_unicast']
        rows.append(f"| {title} | {mean(r['producer_payload_bytes']['ip'] for r in pairs):.0f} | {mean(r['producer_payload_bytes']['ndn'] for r in pairs):.0f} | {mean(r['mean_delivery_delay_ms']['ip'] for r in pairs):.3f} | {mean(r['mean_delivery_delay_ms']['ndn'] for r in pairs):.3f} |")
    rows.extend(['',
        '**设计判断。**受控条件下原先的数倍空口差距不再存在，但NDN仍没有证明更省消费者侧空口：同样六次对象交付，IP为10.320ms、NDN为10.608ms（约高2.79%）。本拓扑的共同段是生产者至接入点的有线回源，缓存和聚合节省该段载荷，不能省掉每个消费者的最后一跳；同步缓存工况回源6000B降至4000B，同时平均交付时间下降。这支持按资源位置区分协议画像，不支持给名称模态所有链路统一乘一个缓存折扣。', '',
        '邻居准备和Face/路由安装在测量前完成，共同PHY组预置ARP条目以消除发现过程混淆；当前表未计量安装CPU/控制总成本，不能称为完整生命周期节省。IP应用缓存/组播增强对照、独立保守空口服务包络、元数据内存画像、多块大对象、移动和拥塞扫描仍待完成。原始部署负面结果全部保留，不能将调整适配器后的结果包装成分配算法的改进。', '',
        '**第一项采用的方向。**继续研究共享低空空地网络上多协议实例切片的联合初始分配。切片仍包含租户/端点、协议实例、功能图、虚拟连接、资源配额、SLA和生命周期。以实测及保守验证后的逐链路、状态和缓存画像生成完整可行候选，学习策略只选择候选或拒绝，强基线使用同一合同与候选。固定协议请求不被算法擅自改协议；只有明确允许替代的请求才比较实现。名称模态的回源/存储权衡、身份机制的映射/移动开销都应进入画像，而非预设某模态必胜。', '',
        '当前判断为：协议资源画像的验证方法和修正后的分层设计可以继续使用；完整第一项尚不能验收。本节链路层实验本身不含RL训练或分配算法成绩；后续最小合同适配器的训练与包级重放单列于10.6，避免混用证据范围。'])
    return '\n'.join(rows)


def current_summary(stability):
    ddqn, ppo = (stability['summary'][key] for key in ('ddqn', 'ppo'))
    gains = [stability['paired'][key + '-adaptive_price']['mean'] for key in ('ddqn', 'ppo')]
    return (f"**摘要（当前证据范围）。**设计在统一低空基础设施上共存的IP、身份/位置分离ID与NDN切片，"
            f"以4架无人机、双AP、双边缘节点起步，阶段0/1测量后冻结规模及候选预算。"
            f"已完成的四节点IP/NDN原型中，八初始化DDQN/PPO全申请包级严格成功率为"
            f"{100*ddqn['mean']:.2f}%/{100*ppo['mean']:.2f}%，相对自适应价格提高"
            f"{100*gains[0]:.2f}/{100*gains[1]:.2f}个百分点。初始化间样本标准差分别为"
            f"{100*ddqn['sample_std']:.3f}/{100*ppo['sample_std']:.3f}个百分点。"
            "这些是旧四节点原型的限定结果，不能替代新三模态平台的独立评测。")


def decision_count(episodes, prefix):
    rows = [row for row in episodes if row['method'].startswith(prefix)]
    count = sum(len(row['decisions']) for row in rows)
    if count != sum(row['requests'] for row in rows):
        raise ValueError('incomplete decision records: ' + prefix)
    return count


def joint_evidence():
    directory = ROOT / 'slice_alloc/results/joint'
    proof_path = directory / 'verification.json'
    if not proof_path.exists():
        return '', None
    proof = json.loads(proof_path.read_text())
    if not proof['passed']:
        raise ValueError('joint adapter verification failed')
    for path, digest in proof['source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('stale joint adapter source: ' + path)
    comparison = json.loads((directory / 'comparison.json').read_text())
    if sha(directory / 'comparison.json') != proof['comparison_sha256']:
        raise ValueError('changed joint results')
    for name, key in [('training.json', 'training_sha256'), ('calibration.json', 'calibration_sha256')]:
        if sha(directory / name) != comparison[key]:
            raise ValueError('changed joint input: ' + name)
    training = json.loads((directory / 'training.json').read_text())
    cal = json.loads((directory / 'calibration.json').read_text())
    analysis = json.loads((directory / 'analysis.json').read_text())
    if (analysis['comparison_sha256'] != sha(directory / 'comparison.json')
            or analysis['source_sha256'] != sha(Path(__file__).with_name('joint_analyze.py'))):
        raise ValueError('stale joint diagnostics')
    for model in training['models']:
        if sha(directory / model['file']) != model['sha256']:
            raise ValueError('changed frozen policy')
    for record in cal['cases'] + comparison['packet_cases']:
        for path, digest in record['files'].items():
            if sha(directory / path) != digest:
                raise ValueError('changed joint packet record: ' + path)
        if sha(directory / record['input_file']) != record['input_sha256']:
            raise ValueError('changed joint task input')

    from statistics import mean
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels = {'first_fit': '首次适配', 'greedy': '当前请求贪心', 'resource_price': '资源价格',
              'adaptive_price': '自适应价格', 'ddqn': 'DDQN（三初始化）', 'ppo': 'PPO（三初始化）'}
    def family(method):
        return [r for r in comparison['episodes'] if
                (r['method'].startswith(method + '_') if method in ('ddqn', 'ppo') else r['method'] == method)]
    methods = list(labels)
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    axes[0].bar(x - .18, [100 * mean(r['model_acceptance'] for r in family(m)) for m in methods],
                width=.36, label='Model admission')
    axes[0].bar(x + .18, [100 * mean(r['packet_success'] for r in family(m)) for m in methods],
                width=.36, label='Native whole-slice SLA success')
    axes[0].axhline(100 * analysis['matched_references']['ddqn']['reference_model_rate'],
                    color='black', linestyle='--', label='Offline reservation-model optimum')
    axes[0].set_xticks(x, ['First fit', 'Greedy', 'Price', 'Adaptive', 'DDQN (3)', 'PPO (3)'], rotation=20)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('All requested slices (%)')
    axes[0].legend(fontsize=8)
    axes[0].set_title('Four-node adapter: three initializations')
    for record in training['models']:
        history = record['history']
        axes[1].plot([h['steps'] for h in history], [h['utility'] for h in history],
                     label=f"{record['algorithm'].upper()} {record['initialization']}", marker='.', linewidth=1)
    axes[1].set_ylim(0, comparison['protocol']['requests'])
    axes[1].set_xlabel('Training steps per model')
    axes[1].set_ylabel('Validation model admitted count')
    axes[1].set_title('Proxy validation, not packet reward')
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(ROOT / f'图/04_最小合同适配器训练与包级结果.{suffix}', dpi=160)
    svg = ROOT / '图/04_最小合同适配器训练与包级结果.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)

    protocol = comparison['protocol']
    loads = '/'.join(f'{load:g}' for load in protocol['loads'])
    rows = ['### 10.6 四节点IP/NDN合同适配与三初始化证据', '',
        '本节为已有原型：源、接入/中继、两个消费者，固定路径，1000字节单块对象，双向有线1Mbps，共享ERP6单播。40%请求指定IP、40%指定NDN、20%允许等价对象接口；该旧工作负载不等于新设计的三类固定需求。NDN冷转发和锁缓存是同一模态的两种配置。', '',
        f"本轮三初始化{protocol['initializations']}，每模型{protocol['steps_per_model']}步，合计{proof['training_steps']}步；训练负载在reset时从{loads}采样，测试也分层报告这三档。每场景{protocol['requests']}个请求、{len(protocol['test_seeds'])}个独立请求种子，共{proof['test_packet_runs']}次原生测试重放。检查点只按验证集模型效用选择。", '',
        '训练负载修复与评估负载调整发生在同一轮；新旧均值和方差差异不能全部归因于修复，需要同负载对照才能隔离修复效应。', '',
        f"独立空口标定共{proof['calibration_packet_runs']}次，拟合最大每轮双消费者TX为{1000*cal['profile']['fitted_max_seconds']:.4f}ms，加声明余量后为{1000*cal['profile']['airtime_seconds_per_pair_round']:.4f}ms。该经验额度不是一般无线服务曲线。", '',
        '合同资源与缓存租约原子提交，锁定缓存先计首轮冷填充，仅减少后续回源；共享设备继续真实竞争。模型奖励为到期结算的接纳权重，包级严格成功独立统计。每切片全部消费者、全部轮次均在150ms内交付才成功，拒绝计为全申请失败。CPU、每切片FIB/PIT和无线份额的完整强制隔离尚未实现。', '',
        '| 方法 | 模型接纳率 | 全申请严格成功率 | 已接纳严格成功率 | 分配流水线/ms |',
        '|---|---|---|---|---|']
    for method, label in labels.items():
        group = family(method)
        accepted = sum(r['accepted'] for r in group)
        success = sum(r['packet_success'] * r['requests'] for r in group)
        rows.append(f"| {label} | {100*mean(r['model_acceptance'] for r in group):.2f}% | {100*mean(r['packet_success'] for r in group):.2f}% | {100*success/max(accepted,1):.2f}% | {mean(r['decision_ms'] for r in group):.3f} |")
    ref = analysis['matched_references']['ddqn']
    references = analysis['offline_references']
    total = sum(r['requests'] for r in references)
    optimal = sum(r['objective'] for r in references)
    rows.extend(['',
        f"同请求合同的{len(references)}组离线MILP最优总接纳为{optimal:.0f}/{total}，即{100*ref['reference_model_rate']:.2f}%。这是知道全部未来请求的预约模型参照，不是物理包级SLA上界。", '',
        f"三初始化矩阵中，{analysis['cases_with_sla_failure']}次重放出现至少一个已接纳切片期限失败，{analysis['cases_exceeding_empirical_tx_allowance']}次超过累计经验TX额度。模型失配保留，不删除失败。", '',
        '![三初始化原型训练与包级结果](图/04_最小合同适配器训练与包级结果.png)', '',
        '[合同与标定](slice_alloc/results/joint/calibration.json)；[三初始化对比](slice_alloc/results/joint/comparison.json)；[失败分解与同模型参照](slice_alloc/results/joint/analysis.json)。'])
    return '\n'.join(rows), proof


def allocation_diagnostic_evidence():
    directory = ROOT / 'slice_alloc/results/joint'
    path = directory / 'allocation_diagnostic_verification.json'
    if not path.exists():
        return '', None
    proof = json.loads(path.read_text())
    if not proof['passed']:
        raise ValueError('allocation diagnostic gates failed')
    for relative, digest in proof['artifact_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('stale diagnostic evidence: ' + relative)
    diagnostic = json.loads((directory / 'allocation_diagnostics.json').read_text())
    stability = json.loads((directory / 'stability_comparison.json').read_text())
    rollout = json.loads((directory / 'rollout_comparison.json').read_text())
    timing = json.loads((directory / 'reject_timing_agreement.json').read_text())
    rows = ['### 10.7 决策核对与预约敏感性', '',
        f"逐一重建{diagnostic['counts']['decisions']}次冻结策略决策。当前目录只含三个固定路径合同且全部枚举，不能据此证明新平台一般联合候选搜索充分。", '',
        '| 预约敏感性设置 | 模型最优接纳数/申请 | 模型最优接纳率 |', '|---|---|---|']
    for name, metric in diagnostic['envelope_sensitivity'].items():
        rows.append(f"| {name} | {metric['optimal_model_count']:.0f}/{metric['requests']} | {100*metric['optimal_model_rate']:.2f}% |")
    rows.extend(['', '这些变体仅诊断预约模型，不替换在线合同，也不构成真实物理容量或因果损失分解。', '',
        '### 10.8 八初始化结果与负载分层', '',
        '三初始化与八初始化分别报告，均为本轮冻结实验。相同请求种子内先平均负载及初始化，再作场景配对；初始化重复不增加独立场景数。', '',
        '| 算法 | 均值 | 最小 | 中位数 | 最大 | 初始化间样本标准差/百分点 |', '|---|---|---|---|---|---|'])
    for algorithm in ('ddqn', 'ppo'):
        m = stability['summary'][algorithm]
        rows.append(f"| {algorithm.upper()} | {100*m['mean']:.2f}% | {100*m['minimum']:.2f}% | {100*m['median']:.2f}% | {100*m['maximum']:.2f}% | {100*m['sample_std']:.3f} |")
    rows.extend(['', '| load | 模型最优接纳/申请 | 贪心 | 资源价格 | 自适应价格 | DDQN（8初始化） | PPO（8初始化） |',
                 '|---|---|---|---|---|---|---|'])
    for load, group in proof['load_stratification'].items():
        rates = ' | '.join(f"{100*group['methods'][m]['packet_success']:.2f}%" for m in ('greedy', 'resource_price', 'adaptive_price', 'ddqn', 'ppo'))
        rows.append(f"| {load} | {group['offline_model_count']:.0f}/{group['requests']} | {rates} |")
    rows.extend(['', '| 八初始化族—基线 | 均值差/百分点 | 95%场景配对描述性区间 |', '|---|---|---|'])
    for key, metric in stability['paired'].items():
        rows.append(f"| {key} | {100*metric['mean']:+.2f} | [{100*metric['low']:+.2f}, {100*metric['high']:+.2f}] |")
    rows.extend(['', '| 初始化 | DDQN严格成功率 | PPO严格成功率 |', '|---|---|---|'])
    for initialization in stability['initializations']:
        rates = [next(r['success_rate'] for r in stability['summary'][a]['per_initialization'] if r['initialization'] == initialization) for a in ('ddqn', 'ppo')]
        rows.append(f"| {initialization} | {100*rates[0]:.2f}% | {100*rates[1]:.2f}% |")
    rows.extend(['', '区间只描述本批请求场景，未作多重比较校正。样本标准差统一使用ddof=1，不与总体标准差混用；不按测试成绩挑初始化。', '',
        '### 10.9 采样rollout对照', '',
        f"rollout使用当前合同、已有预约及合成未来，不读取真实未到达请求。选中horizon={rollout['selected_config']['horizon']}，每动作{rollout['selected_config']['samples']}条采样；同批场景严格成功率{100*rollout['summary']['success_rate']:.2f}%，平均分配耗时{rollout['mean_decision_ms']:.2f}ms。该有限预算基线不是在线最优界。", '',
        '### 10.10 主动拒绝检查', ''])
    ddqn_rows = [row for row in stability['episodes'] if row['method'].startswith('ddqn_')]
    decisions = decision_count(stability['episodes'], 'ddqn_')
    aggregate = timing['aggregate']
    rejections = sum(row['policy_rejected'] for row in ddqn_rows)
    if rejections != aggregate['policy_rejections']:
        raise ValueError('rejection count mismatch')
    rows.append(f"八初始化DDQN共重放{decisions}次决策，有可行候选却主动拒绝{rejections}次。")
    if not rejections:
        rows.append('没有主动拒绝样本，rollout拒绝认同率未定义；不能据此断言已学会最优拒绝时机，或将结果归因为未来机会成本。')
    else:
        rows.append(f"同状态rollout也拒绝{aggregate['rollout_also_rejects']}次；认同率{100*aggregate['rollout_agreement_rate']:.2f}%。决策共享轨迹，不能当作独立样本；分歧不证明任何一方正确。")
    rows.extend(['', '### 10.11 当前证据的最终判定', '', current_summary(stability), '',
        '八初始化方法均值及配对差值以本节记录为准，不混用三初始化统计。均值差不等于等效性检验，初始化离散度也不等于场景泛化误差。模型接纳与包级SLA之间的归一化比值仅作描述，不宣称理论逼近保证。', '',
        '已有四节点证据支持该限定原型的合同选择收益，尚不证明双AP三模态联合映射、完整隔离或一般低空动态优势；收益机制仍需受控对照。新平台按前文独立结果及核验记录评价，不沿用本节成绩。', '',
        '[八初始化对比](slice_alloc/results/joint/stability_comparison.json)；[决策与预约诊断](slice_alloc/results/joint/allocation_diagnostics.json)；[rollout](slice_alloc/results/joint/rollout_comparison.json)；[主动拒绝核对](slice_alloc/results/joint/reject_timing_agreement.json)；[证据核验](slice_alloc/results/joint/allocation_diagnostic_verification.json)。'])
    return '\n'.join(rows), proof


def main():
    current = ROOT/'slice_alloc/results/qrdqn_ensemble/delivery.json'
    if current.exists():
        # The current delivery is immutable; verify its published documents
        # rather than regenerate them from the superseded patrol experiment.
        from .qrdqn_followup_verify import main as verify_current
        verify_current()
        print('当前QR-DQN交付已核验，正文和报告见 slice_alloc/results/qrdqn_ensemble/report.md')
        return
    joint_text,joint_proof=joint_evidence()
    diagnostic_text,diagnostic_proof=allocation_diagnostic_evidence()
    design=DESIGN_PATH.read_text().replace('](../', '](')
    stability=json.loads((ROOT/'slice_alloc/results/joint/stability_comparison.json').read_text())
    title,body=design.split('\n',1)
    trimodal_dir=ROOT/'slice_alloc/results/trimodal'
    trimodal_audit=trimodal_dir/'independent_audit.json'
    trimodal_text=''
    patrol_dir=ROOT/'slice_alloc/results/patrol'
    patrol_audit=patrol_dir/'independent_audit.json'
    introduction=current_summary(stability)
    if trimodal_audit.exists() and not patrol_audit.exists():
        proof3=json.loads(trimodal_audit.read_text())
        if not proof3['passed']:raise ValueError('three-modality independent audit failed')
        for name,digest in proof3['artifacts'].items():
            if sha(trimodal_dir/name)!=digest:raise ValueError('stale three-modality artifact: '+name)
        trimodal_text=(trimodal_dir/'report.md').read_text().replace('# 三模态平台：实际独立评测结果','## 新平台独立评测结果',1)
        if (trimodal_dir/'validation.md').exists():trimodal_text+='\n\n'+(trimodal_dir/'validation.md').read_text().replace('](analysis.json)', '](slice_alloc/results/trimodal/analysis.json)').replace('](diagnosis.json)', '](slice_alloc/results/trimodal/diagnosis.json)').replace('](independent_audit.json)', '](slice_alloc/results/trimodal/independent_audit.json)')
        introduction='本轮完成双AP、双边缘三模态初始分配原型及独立包级评测。新结果与旧四节点证据分开；代理接纳率不等于实际SLA成功率。'
    if patrol_audit.exists():
        patrol_proof=json.loads(patrol_audit.read_text())
        if not patrol_proof['passed']:raise ValueError('patrol independent audit failed')
        for name,digest in patrol_proof['artifacts'].items():
            if sha(patrol_dir/name)!=digest:raise ValueError('stale patrol evidence: '+name)
        from .patrol_run import source_hashes
        if patrol_proof['sources']!=source_hashes():raise ValueError('patrol source changed after experiment')
        report_proof=json.loads((patrol_dir/'report_verification.json').read_text())
        if report_proof['report_sha256']!=sha(patrol_dir/'report.md') or report_proof['independent_audit_sha256']!=sha(patrol_audit):raise ValueError('stale patrol report')
        trimodal_text=(patrol_dir/'report.md').read_text().replace('# 低空巡检三模态平台：最终独立实验结果','## 当前低空巡检平台独立结果',1)
        introduction='本轮已完成真实交互的IP上行、受限EID/RLOC移动通信、NDN同名地图共享及联合初始分配。正式测试3600次、验证640次，结果和适用范围见当前独立报告；旧成绩仅作历史回归。'
    rendered=title+'\n\n'+introduction+'\n'+body+'\n\n'+trimodal_text+'\n\n## 旧四节点回归与历史实验\n\n以下成绩仅属旧接口，不作为新平台成绩。\n\n'+execution_evidence()+'\n\n'+joint_text+'\n\n'+diagnostic_text+'\n'
    md=ROOT/'面向低空智联的多模态网络切片技术.md';md.write_text(rendered)
    doc=to_docx(rendered,destination=md.with_suffix('.docx'),validate_manuscript=True)
    refs=ROOT/'参考文献';inventory_path=refs/'文献清单.json';inventory=json.loads(inventory_path.read_text())
    cited=sorted({int(v) for v in re.findall(r'\[R(\d+)\]',rendered)});checked=0
    for record in inventory['references']:
        for local in record['localFiles']:
            p=refs/local['file']
            if sha(p)!=local['sha256']:raise ValueError('reference changed: '+str(p))
            checked+=1
    known={r['id'] for r in inventory['references']}
    if not set(cited)<=known:raise ValueError('missing citation original')
    inventory.update(scope='Primary sources cited in the retained manuscript',currentDocumentReferences=cited,citedReferences=len(cited),allCitationsHaveLocalOriginals=True)
    inventory_path.write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
    groups=[('研究背景','00_研究背景'),('第一部分：低空智联下的多模态网络切片智能分配','01_第一部分_智能分配'),('第二部分：低空智联下的多模态网络切片资源预测','02_第二部分_资源预测_背景引用'),('数学求解与仿真平台','仿真方法与平台')]
    index=['# 参考文献分类索引','','只保留当前正文实际引用的原文，每份一份。DDQN与PPO为并列验证的候选感知强化学习方法；平台资料对应固定提交的ns-3/ndnSIM及协议依赖。','']
    for heading,group in groups:
        if not any(r['group']==group for r in inventory['references']):
            index.extend(['## '+heading,'','本轮尚无该项正式引用；相关研究接口与后续范围见唯一正文。',''])
            continue
        index.extend(['## '+heading,'','| 编号 | 本地原文 | 本文使用 |','|---|---|---|'])
        for r in inventory['references']:
            if r['group']!=group:continue
            links='、'.join(f"[{Path(v['file']).stem}]({v['file']})" for v in r['localFiles'])
            index.append(f"| R{r['id']:02} | {links} | {'已引用' if r['id'] in cited else '补充参考'} |")
        index.append('')
    index.extend(['## 第三部分：低空智联下的多模态网络切片QoS保障','','当前只确定接口，尚未选定和实现保障算法，不将其他方向的材料冒充本项算法证据。'])
    index.extend(['','来源URL、文件大小和SHA256见[文献清单](文献清单.json)。',''])
    (refs/'文献索引.md').write_text('\n'.join(index).replace('\n− ','\n- '))
    links=0
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',rendered):
        if '://' not in target:
            if not (ROOT/unquote(target)).exists():raise ValueError('broken local link: '+target)
            links+=1
    with zipfile.ZipFile(doc) as z:
        xml=z.read('word/document.xml').decode();images=[n for n in z.namelist() if n.startswith('word/media/')]
        for token in ('身份/位置分离','候选感知','完整动态切片平台尚未完成','ns-3＋ndnSIM','移动包络','低空智联下的多模态网络切片智能分配'):
            if token not in xml:raise ValueError('Word missing design content: '+token)
        if len(images)!=len(re.findall(r'^!\[',rendered,re.M)):raise ValueError('architecture/results figure count mismatch')
    contract_path=ROOT/'slice_alloc/results/design_verification.json'
    contract=json.loads(contract_path.read_text())
    if not contract['passed']:raise ValueError('design contract checks failed')
    for name,digest in contract['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('stale design contract checks: '+name)
    native_path=ROOT/'slice_alloc/results/platform/verification.json'
    native=json.loads(native_path.read_text()) if native_path.exists() else None
    proof=dict(design_source_sha256=sha(DESIGN_PATH),passed=True,kind='design, native protocol evidence and bounded joint-adapter experiment; not full thesis acceptance',md_sha256=sha(md),docx_sha256=sha(doc),generator_sha256=sha(__file__),renderer_sha256=sha(Path(__file__).with_name('export_document.py')),local_links=links,reference_files_checked=checked,citations=cited,embedded_images=len(images),historical_algorithm_results_used=False,training_performed=bool(joint_proof),simulation_performed=bool(native),native_verification_sha256=sha(native_path) if native else None,native_packet_cases=native['cases'] if native else 0,contract_tests=contract['tests'],contract_verification_sha256=sha(contract_path),joint_verification_sha256=sha(ROOT/'slice_alloc/results/joint/verification.json') if joint_proof else None,joint_analysis_sha256=sha(ROOT/'slice_alloc/results/joint/analysis.json') if joint_proof else None,joint_scope=joint_proof['scope'] if joint_proof else None,entry_sha256=sha(ROOT/'slice_alloc/__main__.py'),figures={p.name:sha(p) for p in (ROOT/'图').glob('*.png')})
    if patrol_audit.exists():
        proof['patrol_independent_audit_sha256']=sha(patrol_audit)
        proof['patrol_report_sha256']=sha(patrol_dir/'report.md')
        proof['patrol_packet_cases']=patrol_proof['episodes']
        proof['old_four_node_evidence_separately_labelled']=True
    elif trimodal_audit.exists():
        proof['trimodal_independent_audit_sha256']=sha(trimodal_audit)
        proof['trimodal_report_sha256']=sha(trimodal_dir/'report.md')
        proof['trimodal_packet_cases']=proof3['raw_cases']
        proof['old_four_node_evidence_separately_labelled']=True
    if diagnostic_proof:proof['allocation_diagnostic_verification_sha256']=sha(ROOT/'slice_alloc/results/joint/allocation_diagnostic_verification.json')
    (ROOT/'slice_alloc/results/document_verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if not k.endswith('sha256') and k!='figures'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
