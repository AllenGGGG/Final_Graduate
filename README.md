# 面向低空智联的多模态网络切片技术

低空背景 → IP / EID-RLOC / NDN多模态网络 → Masked Double-DQN切片智能分配。

唯一最终入口：[最终设计、结果及复现说明](code/slice_alloc/final/README.md)。RL成功率79.29%，Greedy为78.07%；1560回合审计通过。

在code目录执行 `python -m slice_alloc verify --workers 5` 复核本地结果。旧版结果及未使用脚本已清理。
