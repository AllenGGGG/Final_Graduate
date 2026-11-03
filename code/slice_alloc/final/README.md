# 面向低空智联的多模态网络切片技术

唯一最终版本：低空巡检环境 → IP/EID-RLOC/NDN → 智能分配A → 实际负载预测A+P → 队列控制A+P+Q。

[实施设计](../DESIGN.md) · [分章节结果与消融](THESIS_RESULTS.md) · [完整报告](report.md) · [原始统计](analysis.json) · [独立审计](verification.json) · [复现步骤](reproduction/README.md)

| 方法 | SLA成功率 |
|---|---:|
| Double-DQN（A） | 85.14% |
| Double-DQN＋预测（A+P） | 85.09% |
| A+P＋被动队列控制 | 85.19% |
| A+P＋预测队列控制 | 85.17% |
| 模态感知Greedy | 87.03% |
| 业务原生模态规则 | 85.38% |
| 资源Greedy | 86.15% |
| First-Fit | 84.42% |

主比较A−模态感知Greedy：-1.90个百分点，95%区间[-2.28, -1.50]。本轮RL低于模态感知Greedy。

四项次要比较及Holm校正见完整报告；区间跨零不宣称稳定收益。全部8个初始化均报告，不挑最优种子。

32个RL模型各训练65536个决策，检查点只由验证集选择；正式测试26640个原生回合，全部完成逐包SLA、遥测方向、预约重放、队列守恒和物理序列化审计。24项自动化测试在清理前后均通过。

旧结果、旧入口和旧方案已清理，清单见[cleanup.json](cleanup.json)。原始模型、逐包轨迹、训练日志和冻结源码保存在本地`code/slice_alloc/results/final`，未上传Git；仓库提供汇总、实际参数、依赖来源、补丁和哈希。没有本地原始产物时须按复现步骤重新训练，不能仅凭汇总文件运行verify。

仅有线队列实行DRR服务区分；不宣称无线硬隔离或确定性SLA保证。场景为固定高度的小规模低空巡检仿真，不主张信道模型创新。

本地复核：在code目录运行`python -m slice_alloc verify --workers 8`。从头复现：设置LOWALT_NS3_ROOT与新的LOWALT_OUTPUT目录，执行`python -m slice_alloc build`及`python -m slice_alloc all --workers 8`。
