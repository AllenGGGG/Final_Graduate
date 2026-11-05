# 验证与完整复现

## 已有完整本地产物

唯一入口是在 `code` 目录运行：

```bash
python -m slice_alloc verify --workers 8
```

默认读取 `slice_alloc/results/final`；可用 `LOWALT_ALLOCATION_OUTPUT` 指定其他完整输出目录。该命令核对冻结参数、所有模型、2520 个原生回合的逐包 SLA/方向/预约和 30 次模型重载重放，不重新选择检查点或修改原始评测数据。

代码测试：

```bash
python -m unittest discover -s slice_alloc/tests -p 'test_*.py' -v
```

测试只有在进程成功退出后才写 `tests_passed.json`。失败日志不会被当作完成状态；独立 `evaluate` 也会先完成同一套训练、代码测试与分析注册检查，再打开测试集。

## Git 中的数据边界

Git 保存源码、协议、训练/选型记录、原始逐回合汇总、审计结果和论文图表，不包含模型权重、逐包轨迹或 ns-3 二进制。因此仅克隆仓库不能直接执行逐包核验：需要复制完整 `results/final`，或按下述训练链重新产生它。完整模型获取目前没有公开下载地址，不声称只靠哈希就能独立验证。

`results/final` 包含 `reference_models/`、`training/`、`models/`、`native/`、冻结源码、测试日志与所有注册。`artifact_metadata_repair.json` 记录先导模型在测试前进行的回放容量属性修正，原文件位于 `provenance/`，每一个策略张量都验证为完全相同。该修正不改变训练所得策略或其验证结果。

## 从头生成训练输入

本轮方法以独立训练的 Double-DQN 为起点，不能把额外 32768 步当作从零总预算。重做这一链条需要：

1. 按 `reproduction/README.md` 配置 ns-3/ndnSIM，设置 `LOWALT_NS3_ROOT`。执行源码为 `execution/allocation_final_scenario.cc`，构建 `allocation-final-scenario`。包版本见 `environment.json`。
2. 在历史提交 `e4e330a` 的独立 worktree 中，按该提交的复现说明运行 `python -m slice_alloc train --workers 5`，产生五个独立 RL 参考模型。旧正式测试种子已经公开；重跑只算复现。
3. 在归档分支 `archive/firstpart-independent-rl-20261105`（`e524fc5`）的独立 worktree 中，把参考模型复制到 `code/slice_alloc/results/final/models/`。在其 `code` 目录执行 `python -m slice_alloc.tools.allocation_credit_run prepare --workers 5`，再执行 `python -m slice_alloc.tools.allocation_mc_run --workers 2`。将五个参考模型另存到所生成 `results/monte_carlo/reference_models/`。这里产生两个初始化的真实先导训练和验证记录，不必重跑先导正式测试。
4. 回到当前主线，在 `code` 目录指定先导产物和一个空的新输出目录：

```bash
export LOWALT_MC_PILOT=/absolute/path/to/archive-worktree/code/slice_alloc/results/monte_carlo
export LOWALT_ALLOCATION_OUTPUT=/absolute/path/to/new-confirmation-output
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m slice_alloc all --workers 5
```

它保留两个先导初始化的策略，补齐另三个初始化，冻结实际配置，按验证集选检查点，然后执行已公开协议的评测和独立审计。重新生成的模型可能因软件/硬件版本而不同，必须重新记录自己的哈希。已有测试种子的重跑是复现，不应冒称新增独立证据。

## 归档

之前的 paired 回放、势函数、短/长窗口、两初始化先导和负面结果保存在 `archive/firstpart-independent-rl-20261105`，逐包/权重另存于用户目录 `.local/share/lowalt-experiments/independent-rl-development-20261104`。更早的强规则研究在 `archive/firstpart-strong-baselines-20261104`。它们不与当前批次合并计算置信区间，也不作为并列的最终入口。

## 事后补充与 Greedy 优化

当前 `report.md` 是包含事后补充的发布报告。原冻结 `verify` 只重生成工作目录 `results/final/report.md` 的预注册主实验部分；它不生成事后补充段落，也不覆盖 Git 中的发布报告。补充分析独立运行，不能把原2520回合的审计范围自动扩展到新增回合。

同协议强规则补充在 `code` 目录执行：

```bash
python -m slice_alloc.tools.allocation_strong_baseline_supplement --workers 8
python -m slice_alloc.tools.allocation_strong_baseline_analysis
python -m slice_alloc.tools.allocation_per_load_comparison
```

360个新增回合保留逐包轨迹，按原审计器检查SLA、方向和预约。`exponential_price`是无准入阈值的指数资源价格规则，不同于归档中的模态感知准入价格规则。扩展的六项次要比较共同进行Holm校正，原冻结主分析保持不变；12项分负载结果另作探索性比较并共同校正。

Greedy优化另用独立输出目录和全新场景：

```bash
python -m slice_alloc.tools.allocation_greedy_improve --workers 8
python -m slice_alloc.tools.allocation_strong_baseline_analysis --audit-traces
python -m unittest discover -s slice_alloc/supplement_tests -p 'test_*.py' -v
```

默认输出 `results/greedy-improvement`，可用 `LOWALT_GREEDY_OUTPUT`覆盖。先注册24组评分、5个阈值、开发/验证/测试种子与比较方式；开发用548000—548011，验证用549000—549029，测试用550000—550059。测试集打开前保存选择和源码哈希；测试后不重新选参数。五个规则在相同的60场景、32/64/96负载上运行，900个正式回合全部逐包审计。开发、验证、测试数据及本地轨迹分别保留，不能与447000系列直接拼接比较。

新增5项回归检查放在`supplement_tests`，避免改变已冻结主实验的28项测试来源和完成标记。

Git中的`greedy-improvement/`保留协议、完整开发/验证/测试逐回合记录、选型、统计、核验和逐包文件哈希。实际轨迹位于本地`results/greedy-improvement/test/native`，强规则补充轨迹位于`results/final/native`；这两批的按请求ID计数和轨迹清单分别保存，不能只凭哈希宣称已下载并独立验证轨迹。
