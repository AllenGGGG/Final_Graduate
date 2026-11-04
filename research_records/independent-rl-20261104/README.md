# 独立RL开发记录归档

包含短/长n-step、按时进度势函数、同状态不同动作采样和两个初始化的Monte-Carlo回报先导测试。没有价格、专家起点或模仿学习。

paired的2520个回合未证明可靠超过Greedy；Monte-Carlo先导1440个回合的主比较为正，但Nearest区间跨零、初始化不足，未通过完整发布条件。不能仅挑这一批当作最终全面领先。

本分支保存实际源码、训练/选型记录、冻结文件和负面结果。原始逐包数据及权重在本机用户目录.local/share/lowalt-experiments/independent-rl-development-20261104中，未上传Git；完整文件哈希见full_local_artifact_manifest.json。新五初始化确认测试使用另一批种子，不能将先导回合并入其显著性计算。
