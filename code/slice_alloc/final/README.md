# 面向低空智联的多模态网络切片技术

最终研究点：**搭建低空背景 → 实现 IP / EID-RLOC / NDN 多模态网络 → 完成网络切片智能分配**。

本轮最终实现采用分层 PPO，联合完成切片准入、模态绑定、边缘部署和资源档位选择。原生入口整形、逐包 SLA 审计和受控超订已纳入实验。

## 最终结果

新测试集上：

- 分层 PPO：**73.34%** SLA 成功率；
- Greedy：**77.32%**；
- PPO 相对 Greedy：**−3.98 个百分点**，95% 区间 **[−6.44, −1.85]**。

因此本轮不声称 RL 优于 Greedy。结果说明当前场景中规则分配仍更稳健。

- [最终实验报告](experiment/report.md)
- [最终设计](DESIGN.md)
- [主测试原始数据](experiment/evaluation.json)
- [压力测试原始数据](experiment/stress_evaluation.json)
- [验证记录](experiment/verification.json)
- [决策空间诊断](experiment/diagnosis.json)
- [复现说明](experiment/REPRODUCE.md)
