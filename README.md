# 面向低空智联的多模态网络切片技术

当前实现范围：第一部分“智能分配”。保留论文题目以及遥测、命令、地图业务名称。

**唯一交付说明：[最终结果与验证](code/slice_alloc/final/README.md)。** 三项流程问题修复后，已从空目录完成15模型、491520次训练决策、1510回合审计及15模型精确重放。QR-DQN为78.39%，价格规则78.49%，尚不能确认RL优于强规则。完整结果与边界见交付说明。

- [已实现的场景与分配设计](code/slice_alloc/ALLOCATION_FINAL.md)
- [历史证据归档](code/slice_alloc/archive/README.md)

在 `code/` 目录、安装了 `slice_alloc/requirements.txt` 依赖的 Python 环境中运行：

```bash
export LOWALT_NS3_ROOT="$HOME/.local/share/lowalt-ndnsim/ns-3"
cp slice_alloc/execution/allocation_final_scenario.cc "$LOWALT_NS3_ROOT/scratch/allocation-final-scenario.cc"
(cd "$LOWALT_NS3_ROOT" && ./waf build --targets=allocation-final-scenario -j 4)
python -m slice_alloc all --workers 6
```

完整入口依次执行配置冻结、测试、标定、训练、验证集选择、主测试、补充压力实验、独立审计及报告。新实验默认输出 `code/slice_alloc/results/final/`；新协议请用 `LOWALT_ALLOCATION_OUTPUT` 指定空目录。`freeze`、`train`、`evaluate` 提供单步入口，正式测试共用预注册流程。

验证工程流程：

```bash
python -m unittest discover -s slice_alloc/tests -p 'test_*.py'
python -m slice_alloc.tools.allocation_final_verify
```

原生验证默认输出 `results/final_validation/`，已有目录时拒绝覆盖；可用 `--output` 指定新目录。训练数据、权重、逐包轨迹和本地产物不进入 Git；交付验证日志、历史结果摘要和哈希清单保留在仓库中。
