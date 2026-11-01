# 历史归档

此目录中的文件属于历史设计或历史实验，不是当前交付的新成绩。当前交付入口只有 [最终结果与验证](../final/README.md)。

- `designs/`：已退出当前入口的早期方案。
- `previous_experiment/`：原正式实验的结果摘要、报告、模型与产物哈希。归档前已核对15个模型及主要产物哈希，未修改历史统计。

完整历史数据、模型、冻结源码和轨迹保留在仓库外的 `../Final_Graduate_experiment_archive/2026-11-01/`（相对于仓库根目录）。这部分不上传 Git，仓库中的清单可用于核对。

`slice_alloc/` 根目录下的 `design.md`、`allocation_design_v3.md`、`lowalt_design_v4.md`、`qrdqn_design.md` 仍被历史复现脚本引用，因此保留兼容路径；它们不属于当前设计入口。旧模块和测试中被当前实现直接或间接导入的代码一并保留，避免清理破坏依赖。
