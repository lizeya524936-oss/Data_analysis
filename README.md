# Data_analysis

数据分析工具集合

## 项目列表

### sensor_hill_fit — 传感器 Hill 方程拟合分析工具 v1.4.1

多传感器压力实验数据分析工具，支持 Hill 方程和双曲线拟合，提供残差分析和跨批次一致性验证。

**核心功能：**
- 多次实验 CSV 文件批量导入
- 4 传感器逐行求和 → 多次实验均值曲线 → Hill/双曲线拟合
- 拟合参数导出（JSON/CSV）与再导入，支持跨批次残差对比
- 离群点剪除、平滑窗口等参数可调

详细说明请查看 [sensor_hill_fit/README.md](sensor_hill_fit/README.md)
