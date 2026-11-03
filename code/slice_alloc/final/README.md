# 面向低空智联的多模态网络切片技术

当前唯一最终版本：低空背景 → IP / EID-RLOC / NDN多模态网络 → Masked Double-DQN切片初始分配。无资源档位、超订或拉格朗日惩罚。

| 方法 | SLA成功率 |
|---|---:|
| 最终RL | **79.29%** |
| Greedy | 78.07% |
| Nearest | 78.06% |
| First-Fit | 76.68% |

RL比Greedy高1.22个百分点，配对95%区间[+0.20,+2.31]个百分点。相对原版增加0.23个百分点，但该增益区间跨零，不能断言优化收益稳定。

- [最终设计与完整结果](report.md)
- [验证记录](verification.json)：1560回合独立审计，清理后22项测试通过。
- [全部统计](analysis.json)
- [复现说明](REPRODUCE.md)
- [结果对比图](comparison.png)、[训练验证曲线](learning-curves.png)

默认入口：code目录执行 `python -m slice_alloc verify --workers 5`。旧版独立结果目录和无用脚本已经删除；本地只有results/final这一份当前结果，保留必要的本轮对照审计记录。模型及逐包数据存放在本地results/final；从头复现说明见上。
