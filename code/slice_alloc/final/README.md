# 面向低空智联的多模态网络切片技术

当前主线恢复为第一部分：低空巡检 → IP/EID-RLOC/NDN → Masked Double-DQN切片初始分配。无档位、超订、对偶惩罚、预测或运行期QoS控制。

[第一部分最终设计](DESIGN.md) · [完整结果及违约率](report.md) · [恢复说明](RECOVERY.md) · [验收](acceptance.json) · [原版480回合复现](original_480_report.md) · [验证记录](verification.json)

| 方法 | SLA成功率 | 违约／全部到达 | 违约／已接纳 |
|---|---:|---:|---:|
| double_dqn | 79.29% | 8.40% | 9.58% |
| greedy | 78.07% | 11.74% | 13.08% |
| nearest | 78.06% | 11.59% | 12.93% |
| first_fit | 76.68% | 12.16% | 13.69% |

RL−Greedy：+1.22个百分点，配对95%区间[+0.20, +2.31]。

5个初始化全部重训及重载，1560回合完整对照与原版480回合均重新逐包、方向、预约审计；清理后23项测试通过。沿用历史种子做复现，不是新增独立证据。

模型与逐包轨迹保存在本地results/final，未上传Git。历史扩展版及其负面结果保留于archive/expanded-study-ee8f44d分支；本次不按测试表现调参。小幅优势仅针对本协议三条规则，不宣称超过全部传统算法。

新增的[强规则补充验证](ADDITIONAL_VALIDATION.md)已实际完成：阶段到达场景中，规则引导PPO为72.69%，普通Greedy为68.96%，模态感知Greedy为72.40%，价格规则为72.64%。PPO相对价格规则仅+0.055个百分点，配对95%区间[-0.053,+0.185]，未证实学习增益，不替换主线。这套新负载与上表历史负载不同，数值不能直接比较。

补充研究的2700回合及48次模型重载重放均通过逐包、方向与预约审计，31项研究测试通过；当前清理后的主线另有23项测试。实验代码和完整开发记录见archive/firstpart-strong-baselines-20261104（9ff23de）；模型及逐包产物在本机$HOME/.local/share/lowalt-experiments/firstpart-strong-baselines-20261104。主线仅保留原分配器与补充验证报告。

从头复现见[REPRODUCE.md](REPRODUCE.md)；已有本地完整产物时在code目录运行`python -m slice_alloc verify --workers 5`。
