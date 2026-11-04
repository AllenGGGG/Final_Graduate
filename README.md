# 面向低空智联的多模态网络切片技术

当前只保留第一部分：低空巡检场景 → IP/EID-RLOC/NDN 多模态网络 → 独立 RL 切片初始分配。

[最终验证结果](code/slice_alloc/final/README.md) · [设计](code/slice_alloc/final/DESIGN.md) · [复现](code/slice_alloc/final/REPRODUCE.md) · [论文结果文字](code/slice_alloc/final/THESIS_TEXT.md)

五个初始化、全新 60 个场景、32/64/96 三负载：RL 成功率 **79.078%**，Greedy **78.073%**；主配对差 **+1.005 个百分点，95% 区间 [+0.299,+1.740]**。接纳后违约率分别为 9.04% 和 12.77%。2520 个原生回合、30 次模型重载重放、28 项代码测试通过。

结论限定于总体成功率和所列对照。遥测仍弱于 Greedy；Nearest 的补充比较经 Holm 校正后未达到 0.05；完整回报续训相对初始 RL 的增益尚未被可靠证实。旧设计及负面结果保存在归档分支，主目录没有并列的旧结果入口。
