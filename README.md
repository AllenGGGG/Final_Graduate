# 面向低空智联的多模态网络切片技术

项目主线：**搭建低空背景 → 实现多模态网络 → 网络切片智能分配**。当前只实现第一部分“智能分配”，题目和遥测、命令、地图业务名称保持不变。

**唯一最终入口：[最终方案、结果与验证](code/slice_alloc/final/README.md)。**

最终分配器经验证集选型为Masked Double-DQN。完整训练688128次决策、1920个回合审计及21个模型重放已完成；新测试SLA成功率79.38%，Greedy为79.35%，尚不能确认显著优势。

在 `code/` 目录、安装了 `slice_alloc/requirements.txt` 依赖的Python环境中运行：

```bash
export LOWALT_NS3_ROOT="$HOME/.local/share/lowalt-ndnsim/ns-3"
cp slice_alloc/execution/allocation_final_scenario.cc "$LOWALT_NS3_ROOT/scratch/allocation-final-scenario.cc"
(cd "$LOWALT_NS3_ROOT" && ./waf build --targets=allocation-final-scenario -j 4)
python -m slice_alloc all --workers 12
```

完整管线执行冻结、测试、标定、训练、验证集选型、主测试、补充压力实验、独立审计及报告。默认输出 `code/slice_alloc/results/final/`；不同协议的新实验用 `LOWALT_ALLOCATION_OUTPUT` 指定空目录。

使用最终模型查看实际分配：

```bash
python -m slice_alloc.tools.allocation_final_allocate --seed 56000 --load 12 --output /tmp/lowalt-allocation
```

源码、报告、逐回合汇总、配置、哈希和一个实际执行示例纳入Git。大批训练权重与原始轨迹保存在本地正式输出目录，历史实验移到仓库外归档；必要依赖保留，重复入口和临时模型、缓存已清理。
