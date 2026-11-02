# 复现最终智能分配实验

本实验只覆盖第一部分：低空背景、多模态网络和切片智能分配。需要 ns-3/ndnSIM、Python、numpy、torch 和 Stable-Baselines3 环境。

## 原生场景

```bash
export LOWALT_NS3_ROOT=/path/to/lowalt-ndnsim
cp slice_alloc/execution/allocation_cmdp_scenario.cc "$LOWALT_NS3_ROOT/scratch/allocation-cmdp-scenario.cc"
cd "$LOWALT_NS3_ROOT"
./waf build --targets=allocation-cmdp-scenario -j 4
```

## 测试与完整管线

```bash
cd /path/to/Final_Graduate/code
PYTHONPATH=. python -m unittest discover -s slice_alloc/tests -p 'test_*.py'
PYTHONPATH=. python -m slice_alloc all --workers 8
```

完整管线固定使用：分层 PPO、5 个训练初始化（11、23、37、53、71）、每个模型 8192 步、30 个验证种子、30 个主测试种子和独立压力测试种子。运行前先执行决策空间诊断；正式结果必须通过逐包 SLA、方向、资源账本和事件重放审计。

最终导出的结果在 `slice_alloc/final/experiment/`。模型权重和完整逐包事件轨迹属于生成物，不随源码仓库提交；结果文件保留模型哈希、协议哈希和审计计数。
