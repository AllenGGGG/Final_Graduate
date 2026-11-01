# 复现

在代码目录使用已安装numpy/torch/SB3/sb3-contrib的Python环境。

```bash
cp slice_alloc/execution/allocation_final_scenario.cc "$LOWALT_NS3_ROOT/scratch/allocation-final-scenario.cc"
cd "$LOWALT_NS3_ROOT"
./waf build --targets=allocation-final-scenario -j 4
```

回到代码目录：

```bash
python -m unittest discover -s slice_alloc/tests -p 'test_*.py'
python -m slice_alloc all --workers 6
python -m slice_alloc.tools.allocation_final_allocate --seed 46000 --load 48 --output /tmp/lowalt-allocation
```

默认输出为results/final，也可用LOWALT_ALLOCATION_OUTPUT指定独立目录。现有结果按配置及哈希核对后复用，不能拼接新旧协议。完整管线在主测试前冻结补充压力协议及初始化11的部署约定，并执行标定、训练、主测试、补充测试和审计。训练、验证、测试与逐包审计都使用原生仿真。
