# 第一部分补充验证：更强规则与独立测试

本次实际完成了新的设计与训练，而非重复历史79.29%的成绩。物理网络、三种模态及业务合同保持不变；阶段工作负载只改变到达时间。它与历史均匀工作负载不是同一个评测协议，不能直接相减归因于算法。

主方法为规则引导Masked PPO：拒绝或选择模态、边缘及映射/缓存位置，最多11个动作；单个策略网络，无资源档位、超订、对偶约束、预测器或QoS控制器。起点为开发集选定的模态与准入价格规则，保留零步检查点作为明确消融。

24组模态公式及7个准入价格阈值在60个验证场景比较。主模型8个初始化各训练65536步，全部保留；两个额外模型族各完成2个初始化的同预算开发对照，均未达到已记录的扩展条件。统一选定模型族之后才打开323000—323059独立测试种子，三档请求量为48/80/112。

| 方法 | SLA成功率 | 违约／全部到达 | 违约／已接纳 |
|---|---:|---:|---:|
| ppo | 72.6936% | 7.3030% | 9.1291% |
| greedy | 68.9583% | 13.2986% | 16.1672% |
| nearest | 68.9236% | 13.2431% | 16.1173% |
| first_fit | 67.9931% | 13.0833% | 16.1370% |
| native | 69.5347% | 7.3194% | 9.5238% |
| modal_greedy | 72.4028% | 8.7847% | 10.8203% |
| price | 72.6389% | 7.3472% | 9.1856% |
| prior | 72.6389% | 7.3472% | 9.1856% |

主比较PPO−price：+0.0547个百分点；场景与初始化双重配对bootstrap95%区间[-0.0530, +0.1849]。

8个初始化中选择零步规则起点的有4个：[37, 53, 101, 127]。因此不能将这些初始化的成绩归因于学习。其他初始化的具体检查点、每业务、负载、阶段和初始化成绩见analysis.json与training_complete.json。

全部发布条件通过：**False**。条件包括主比较和相对零步消融的区间下限为正，以及两种违约率均不变差。

2700个正式回合全部重新逐包、遥测方向及预约重放核验，另完成48个真实模型重载推理回合；研究期间31项测试通过。首次测试因审计器以数组位置代替请求ID而阻断，原始失败记录、原注册及审计修复补充记录全部保留；修复仅给计数审计提供按ID排序的合同视图，未改变模型、请求、动作、种子或原始场景哈希。

均匀负载上的规则引导Double-DQN8初始化与势函数对照2初始化均未通过开发关卡，没有打开它们的正式测试集。这些负面结果保留，不能只报告任务阶段方案。

资源账本仅表示准入预约，不等同于逐切片带宽保证；EID-RLOC是简化实现，遥测仅可选IP；固定高度小规模平台不作为低空信道创新。

归档分支：archive/firstpart-strong-baselines-20261104。完整模型与逐包产物单独存放在本机$HOME/.local/share/lowalt-experiments/firstpart-strong-baselines-20261104，Git内保留源码、各检查点验证记录、正式评测数据、冻结协议、审计结果和全部原始文件哈希。只拿报告与哈希不能代替逐包重放。

**结论：本轮未证实学习超过验证选定的强规则，不替换历史主线，不声称已实现“RL可靠领先传统算法”。历史主线的小幅优势仍仅限其列出的三条普通规则。**

在本机重放这项归档验证（需先检出归档分支，保留完整原生平台）：

```bash
export PYTHONPATH=code
export LOWALT_MISSION_OUTPUT="$HOME/.local/share/lowalt-experiments/firstpart-strong-baselines-20261104/mission"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m slice_alloc.tools.mission_allocation_evaluate verify --workers 8
```

各方案完整本地产物的逐文件SHA256清单为对应目录的raw_artifacts_sha256.json。版本和执行环境见mission/environment_run.json。模型族选择记录、原始预注册、首次失败以及审计修复补充说明分别见mission/family_selection*.json、mission/evaluation_registration.json、mission/attempts及mission/evaluation_audit_amendment.json。原测试启动清单未被覆盖。

方法依据包括[策略梯度PPO原论文](https://arxiv.org/abs/1707.06347)和[势函数奖励塑形](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)。规则起点不冒充从零学习，也不声称实现了DQfD或RUDDER。没有把合同结算奖励任意搬回单个历史动作。
