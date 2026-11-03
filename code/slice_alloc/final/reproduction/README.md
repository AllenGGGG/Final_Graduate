# 复现环境与执行

原始模型、逐包轨迹和完整回合观测保存在本地 `code/slice_alloc/results/final`，不随Git上传。仓库中的统计和哈希可供复核；没有原始文件时需要重新训练与评测，不能只运行verify得到结果。

`native_sources.json`记录实际使用的四个仓库提交、编译器和本机配置。NFD补丁修复bool函数缺少返回值，ndn-cxx补丁补充optional头文件，均为当前环境原有补丁；不改变本轮不同方法之间的实验条件。

1. 准备Python依赖：使用兼容的Python3.10环境安装 `code/slice_alloc/requirements.txt`。另需C++17编译器、Boost、OpenSSL、SQLite和nlohmann-json开发依赖。完整实验使用CPU；本轮并行8个任务。
2. 从 `native_sources.json` 中的URL获取ns-3和ndnSIM，分别检出记录的提交。ndnSIM放在ns-3的 `src/ndnSIM`。初始化子模块，检出记录的NFD和ndn-cxx提交，并在对应子模块中应用 `nfd.patch` 和 `ndncxx.patch`。
3. 在ns-3根目录用本机记录的配置编译：

```bash
python waf configure --build-profile=optimized --disable-python --enable-modules=ndnSIM,wifi,internet,applications,mobility,point-to-point,flow-monitor --cxx-standard=-std=c++17
```

4. 在本项目的 `code` 目录设置路径、复制并构建实际执行器。输出目录应使用一个新的空目录：

```bash
export LOWALT_NS3_ROOT=/absolute/path/to/ns-3
export LOWALT_OUTPUT=/absolute/path/to/new-study-output
python -m slice_alloc build
python -m slice_alloc all --workers 8
```

管线顺序为自动化测试、容量诊断、预测数据/模型、24组规则选型、32个RL模型训练、正式测试、独立审计与报告。失败不会生成成功标记；已完成产物只在冻结配置和哈希一致时复用。正式测试打开后不允许重新训练选型。

已有完整本地产物时运行：

```bash
export LOWALT_OUTPUT=/absolute/path/to/existing-study-output
python -m slice_alloc verify --workers 8
```

不同处理器或依赖环境不要求生成逐位相同的二进制/权重。每轮都会重新记录实际配置、源码、二进制与模型哈希；应在同一轮内比较方法，不混用不同轮次结果。
