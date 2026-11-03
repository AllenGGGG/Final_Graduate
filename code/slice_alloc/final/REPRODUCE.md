# 最终验证版复现

当前唯一入口为code目录下的 `python -m slice_alloc`。已完成的本地结果位于 `slice_alloc/results/final`；运行以下命令重新核验，不会重新训练：

```bash
python -m slice_alloc verify --workers 5
```

从头训练请使用全新的输出目录，默认入口自动完成32768步初始训练、最多65536步的有界续训、验证集选检查点、新测试集对照和独立审计：

```bash
export LOWALT_REFINE_OUTPUT=/absolute/path/to/new-output
export LOWALT_REFERENCE_OUTPUT="$LOWALT_REFINE_OUTPUT/pretraining"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m slice_alloc all --workers 5
```

先在安装requirements.txt依赖的Python环境中运行。本机实际包版本见environment.json。原生平台为带ndnSIM的ns-3，通过LOWALT_NS3_ROOT指定；默认是用户目录下.local/share/lowalt-ndnsim/ns-3。将execution/allocation_final_scenario.cc复制为ns-3的scratch/allocation-final-scenario.cc，在ns-3目录运行：

```bash
./waf build --targets=allocation-final-scenario -j 4
```

protocol.json记录模型、源码、参数和种子；frozen保存冻结源码；training/models保存检查点及模型；native保存输入、逐包事件、决策和审计结果。全部初始化均保留。pretraining仅在从头复现时生成，历史版本的独立结果目录已经删除。

论文主指标和原版对照都来自同一批新场景。原版对照在本轮重新运行，是当前审计证据的一部分；不是另一套待选最终结果。续训收益区间跨零，应照实写作小幅观察值，不应写作已证实的算法改进。

源码和报告可以入Git；本地完整模型及逐包数据目前保存在results/final，没有上传外部存储。只拿报告和哈希不能逐包复核，需取得完整输出目录或从头复现。
