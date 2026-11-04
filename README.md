# 面向低空智联的多模态网络切片技术

当前恢复第一部分：低空巡检场景 → IP/EID-RLOC/NDN → Masked Double-DQN智能分配。

[最终结果与违约率](code/slice_alloc/final/README.md) · [设计](code/slice_alloc/final/DESIGN.md) · [恢复说明](code/slice_alloc/RECOVERY.md)

[新增独立测试与强规则核验](code/slice_alloc/final/ADDITIONAL_VALIDATION.md)：2700回合与48次模型重放已完成。规则引导PPO优于普通Greedy，但未证实超过价格规则或自身规则起点；未替换主线，不能将历史优势外推为RL普遍领先。试验源码及全部开发记录保存在archive/firstpart-strong-baselines-20261104分支。
