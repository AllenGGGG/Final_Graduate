# 原生平台复现

`native_sources.json` 记录本轮实际使用的 ns-3、ndnSIM、NFD、ndn-cxx 提交及编译配置，已与本机逐项核对。两个补丁分别补充 optional 头文件及 bool 返回值，已与实际工作区补丁核对；它们不是资源分配策略。

从文件中的地址获取 ns-3 和 ndnSIM，按记录检出提交。将 ndnSIM 放在 ns-3 的 `src/ndnSIM`，初始化子模块，并将 NFD 和 ndn-cxx 检出为记录的提交。在对应子模块根目录用 `git apply` 应用本目录对应补丁。

安装 C++17 编译器、Boost、OpenSSL、SQLite、nlohmann-json 开发依赖，在 ns-3 根目录配置：

```bash
python waf configure --build-profile=optimized --disable-python --enable-modules=ndnSIM,wifi,internet,applications,mobility,point-to-point,flow-monitor --cxx-standard=-std=c++17
```

随后按 [第一部分复现步骤](../REPRODUCE.md) 复制、编译执行器，设置 `LOWALT_NS3_ROOT`，使用空输出目录训练并评测。当前第一部分入口没有 build 子命令。不同机器不要求产生逐位相同的二进制，但同一轮的全部方法必须共用该轮冻结的执行器与配置。
