# 传感器 Hill 方程拟合分析 — 技术规格文档

**版本：** v1.7  
**作者：** Manus AI  
**日期：** 2026-04-07

---

## 1. 概述

本文档详细描述传感器 Hill 方程拟合分析工具的核心算法、数据处理流程、函数接口和功能需求规格，旨在为将该工具的功能集成到其他平台（如 Web 页面）提供完整的技术参考。

工具的核心目标是：对压力传感器阵列的多次实验数据进行标准化处理，通过 Hill 方程和双曲线方程拟合传感器的压力-ADC 响应特性，并支持从 ADC 值反推压力的逆向计算。

---

## 2. 数据处理流程

整体数据处理分为 **6 个阶段**，每个阶段对应一个独立的纯函数，无副作用，便于在任何环境中复用。

![数据处理流程](pipeline_flow.png)

### 2.1 流程概览

| 阶段 | 函数名 | 输入 | 输出 | 说明 |
|------|--------|------|------|------|
| 1 | `parse_csv` | CSV 文本 | 结构化数据 | 解析压力列和传感器列 |
| 2 | `compute_sum_curve` | 结构化数据 | 单传感器均值曲线 | 同一文件内 4 传感器求和 ÷ 数量 |
| 3 | `compute_average_from_sum_curves` | 多条均值曲线 | 平均曲线 | 多次实验分箱取均值 + 平滑 |
| 4a | `fit_hill` | 平均曲线 | Hill 参数 (a, b, n) | 非线性最小二乘拟合 |
| 4b | `fit_hyperbolic` | 平均曲线 | 双曲线参数 (a, b) | 非线性最小二乘拟合 |
| 5 | `back_project_to_sum_curves` | 拟合参数 + 各次实验曲线 | 残差分析结果 | 回代计算 RMSE / R² |
| 6 | `inverse_hill` | ADC 值 + Hill 参数 | 压力值 | 反向公式计算 |

---

## 3. 数学模型

### 3.1 Hill 方程（正向）

Hill 方程源自生物化学中的 Hill-Langmuir 方程 [1]，用于描述具有饱和特性的 S 型响应曲线。在压力传感器场景中，它描述了传感器 ADC 输出值随施加压力变化的非线性关系：

$$y = \frac{a \cdot x^n}{b^n + x^n}$$

![Hill 方程正向与反向曲线](hill_curves.png)

**参数含义：**

| 参数 | 符号 | 物理含义 | 典型范围 | 单位 |
|------|------|---------|---------|------|
| 饱和值 | a | ADC 输出的理论最大值，当压力趋向无穷大时 y → a | 100 ~ 5000 | ADC |
| 半饱和压力 | b | 使 ADC 输出达到 a/2 时对应的压力值 | 0.1 ~ 500 | N |
| Hill 系数 | n | 曲线的陡峭程度，n > 1 为正协同效应，n < 1 为负协同效应 | 0.1 ~ 5.0 | 无量纲 |

> **关键特性：** 当 x = b 时，y = a/2。Hill 系数 n 决定了曲线从低响应到高响应的过渡速度。n = 1 时退化为标准 Michaelis-Menten 方程（即双曲线方程）。

### 3.2 双曲线方程

双曲线方程是 Hill 方程在 n = 1 时的特例，也称为 Michaelis-Menten 方程 [2]：

$$y = \frac{a \cdot x}{b + x}$$

| 参数 | 符号 | 物理含义 |
|------|------|---------|
| 饱和值 | a | ADC 输出的理论最大值 |
| 半饱和常数 | b | y = a/2 时的压力值 |

### 3.3 Hill 方程反向公式

已知 ADC 均值 y，反推对应压力 x。通过代数变换可得解析解：

$$x = b \cdot \left(\frac{y}{a - y}\right)^{1/n}$$

**推导过程：**

$$y = \frac{a \cdot x^n}{b^n + x^n}$$

$$y \cdot (b^n + x^n) = a \cdot x^n$$

$$y \cdot b^n = x^n \cdot (a - y)$$

$$x^n = \frac{y \cdot b^n}{a - y}$$

$$x = b \cdot \left(\frac{y}{a - y}\right)^{1/n}$$

**有效域约束：**

| 条件 | 结果 | 说明 |
|------|------|------|
| 0 < y < a | 有效 | 正常计算范围 |
| y ≤ 0 | 无效 | 传感器无响应 |
| y ≥ a | 无效 | ADC 已饱和，压力趋向无穷 |
| 计算结果 x > 100 N | 警告 | 超出传感器标定范围 |

### 3.4 评估指标

**均方根误差 (RMSE)：**

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

**决定系数 (R²)：**

$$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$

其中 $y_i$ 为实测值，$\hat{y}_i$ 为拟合预测值，$\bar{y}$ 为实测值均值。R² 越接近 1 表示拟合效果越好。

---

## 4. 函数接口规格

以下为每个核心函数的详细接口定义，可直接在 Python、JavaScript/TypeScript 或任何语言中实现。

### 4.1 parse_csv — CSV 数据解析

**功能：** 将 CSV 文本解析为结构化数据，自动识别压力列和传感器列。

**输入参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| content | string | 是 | CSV 文件的完整文本内容 |
| filename | string | 否 | 文件名，用于错误提示 |

**输出结构：**

```typescript
interface ParsedCSV {
  filename: string;           // 文件名
  sensor_ids: string[];       // 传感器列名列表，如 ["Sensor#1", "Sensor#2", "Sensor#3", "Sensor#4"]
  rows: Array<{
    pressure: number;         // 压力值 (N)
    sensor_values: number[];  // 各传感器的 ADC 值，顺序与 sensor_ids 对应
  }>;
}
```

**列识别规则：**

| 列类型 | 匹配关键词（不区分大小写） |
|--------|--------------------------|
| 压力列 | `pressure`, `压力`, `force`, `load` |
| 传感器列 | `sensor`, `传感器`, `#数字`, `ch数字`, `s数字` |

若未匹配到传感器列，则将第 3 列及之后的所有列视为传感器列。若未匹配到压力列，默认使用第 2 列。

### 4.2 compute_sum_curve — 单传感器均值曲线

**功能：** 对每行数据中的所有传感器值求和后除以传感器数量，得到单传感器平均响应曲线。

**输入参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| parsed | ParsedCSV | — | parse_csv 的输出 |
| p_min | number | 0.0 | 压力下限 (N)，低于此值的数据点被过滤 |
| p_max | number | 100.0 | 压力上限 (N)，高于此值的数据点被过滤 |
| val_thresh | number | 0.0 | 均值最小阈值，低于此值的数据点被过滤（用于去除静息期） |

**输出结构：**

```typescript
interface SumCurvePoint {
  pressure: number;    // 压力值 (N)
  sum_value: number;   // 单传感器均值 = Σ(sensor_values) / sensor_count
}
type SumCurve = SumCurvePoint[];
```

**核心计算逻辑（伪代码）：**

```
n_sensors = len(parsed.sensor_ids)
for each row in parsed.rows:
    if row.pressure <= p_min or row.pressure > p_max: skip
    avg = sum(row.sensor_values) / n_sensors
    if avg <= val_thresh: skip
    output.append({pressure: row.pressure, sum_value: avg})
```

### 4.3 compute_average_from_sum_curves — 多次实验平均曲线

**功能：** 将多次实验的单传感器均值曲线合并，按压力分箱取均值，并进行滑动窗口平滑。

**输入参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| sum_curves | SumCurve[] | — | 多次实验的均值曲线数组 |
| p_min | number | 0.0 | 压力下限 |
| p_max | number | 100.0 | 压力上限 |
| smooth_window | number | 5 | 滑动平均窗口大小 |
| remove_outliers | boolean | false | 是否去除异常值 |
| outlier_threshold | number | 15.0 | 异常值阈值（偏离均值的百分比） |

**输出：** `(pressures: number[], values: number[])` — 平均曲线的压力数组和对应的 ADC 均值数组。

**算法步骤：**

1. **合并所有曲线数据点**，按压力值排序。
2. **自适应分箱**：根据数据密度计算分箱宽度（0.5 ~ 1.0 N），将压力轴均匀分为若干 bin。
3. **箱内均值**：对每个 bin 内的所有数据点取压力均值和 ADC 均值。
4. **异常值过滤**（可选）：若某 bin 内数据点 ≥ 3 个，去除偏离 bin 均值超过 `outlier_threshold%` 的点。
5. **滑动窗口平滑**：使用宽度为 `smooth_window` 的移动平均，边界处使用对称缩窗。

### 4.4 fit_hill — Hill 方程拟合

**功能：** 对平均曲线进行 Hill 方程非线性最小二乘拟合。

**输入参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| pressures | number[] | 压力数组 |
| values | number[] | ADC 均值数组 |
| p0 | [number, number, number] 或 null | 初始猜测 [a, b, n]，为 null 时自动估计 |

**输出结构：**

```typescript
interface HillFitResult {
  a: number;          // 饱和值
  b: number;          // 半饱和压力
  n: number;          // Hill 系数
  rmse: number;       // 均方根误差
  r2: number;         // 决定系数
  residuals: number[]; // 残差数组 (y_true - y_pred)
}
```

**拟合策略：**

| 步骤 | 方法 | 参数边界 |
|------|------|---------|
| 主拟合 | scipy.optimize.curve_fit，3 参数 (a, b, n) | a: [0, 5000], b: [0.1, 500], n: [0.1, 5.0] |
| 降级拟合 | 若主拟合失败，固定 n=1.0，拟合 2 参数 (a, b) | a: [0, 5000], b: [0.1, 500] |
| 兜底 | 若均失败，使用初始估计值 | a = max(values) × 1.1, b = median(pressures), n = 1.0 |

**初始值自动估计：**

```
a0 = max(values) × 1.1    // 略高于最大观测值
b0 = median(pressures)     // 压力中位数
n0 = 0.9                   // 接近 1 的初始猜测
```

### 4.5 fit_hyperbolic — 双曲线拟合

**功能：** 对平均曲线进行双曲线方程拟合（Hill 方程 n=1 的特例）。

**输入/输出：** 与 `fit_hill` 类似，但只有 2 个参数 (a, b)，无 n 参数。

### 4.6 back_project_to_sum_curves — 回代残差分析

**功能：** 将拟合参数回代到每次实验的均值曲线，计算各次实验与拟合曲线的偏差。

**输入参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| sum_curves | SumCurve[] | 各次实验的均值曲线 |
| filenames | string[] | 对应的文件名 |
| hill_a, hill_b, hill_n | number | Hill 拟合参数 |
| hyp_a, hyp_b | number 或 null | 双曲线拟合参数（可选） |
| p_min, p_max | number | 压力范围 |
| smooth_window | number | 平滑窗口 |

**输出结构（每次实验一条记录）：**

```typescript
interface BackProjection {
  filename: string;
  pressures: number[];       // 分箱后的压力数组
  sum_values: number[];      // 分箱平滑后的实测值
  hill_pred: number[];       // Hill 预测值
  hill_rmse: number;
  hill_r2: number;
  hill_residuals: number[];  // 残差 = 实测 - 预测
  // 双曲线部分（若有）
  hyp_pred?: number[];
  hyp_rmse?: number;
  hyp_r2?: number;
  hyp_residuals?: number[];
}
```

### 4.7 inverse_hill — ADC 反推压力

**功能：** 已知 ADC 均值，用 Hill 反向公式计算对应压力。

**输入参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| y | number | ADC 均值 |
| a | number | Hill 饱和值参数 |
| b | number | Hill 半饱和压力参数 |
| n | number | Hill 系数参数 |

**输出：**

| 返回值 | 类型 | 说明 |
|--------|------|------|
| x | number 或 null | 反推压力值 (N)，无效时返回 null |
| status | string | "valid" / "saturated" / "zero" / "out_of_range" |

**实现逻辑（伪代码）：**

```
function inverse_hill(y, a, b, n):
    if y <= 0:
        return {x: null, status: "zero"}
    if y >= a:
        return {x: null, status: "saturated"}
    
    x = b * (y / (a - y)) ^ (1 / n)
    
    if x > 100:
        return {x: x, status: "out_of_range"}
    
    return {x: x, status: "valid"}
```

---

## 5. 功能模块需求规格

以下为工具的完整功能模块定义，适用于在 Web 或其他平台重新实现。

### 5.1 模块一：数据导入

| 需求项 | 规格 |
|--------|------|
| 输入格式 | CSV 文件，UTF-8 或 GBK 编码 |
| 文件数量 | 支持 1~N 个文件同时导入（每个文件 = 一次实验） |
| 列要求 | 至少 1 列压力 + 1 列传感器数据 |
| 自动识别 | 自动识别压力列和传感器列（见 4.1 节识别规则） |
| 压力范围 | 默认 0~100 N，用户可自定义 |
| 数据过滤 | 支持设置最小 ADC 阈值，过滤静息期数据 |

### 5.2 模块二：拟合分析

| 需求项 | 规格 |
|--------|------|
| 预处理 | 每个文件内 4 传感器求和 ÷ 传感器数量 → 单传感器均值 |
| 平均曲线 | 多次实验数据合并 → 分箱取均值 → 滑动窗口平滑 |
| 拟合模型 | Hill 方程（3 参数）+ 双曲线方程（2 参数） |
| 拟合方法 | 非线性最小二乘法（Levenberg-Marquardt） |
| 评估指标 | RMSE、R² |
| 异常值处理 | 可选开启，按偏离均值百分比过滤（默认 15%） |
| 平滑窗口 | 可调，默认 5 |

### 5.3 模块三：残差分析

| 需求项 | 规格 |
|--------|------|
| 均值曲线残差 | 拟合曲线 vs 平均曲线的逐点残差 |
| 各次实验残差 | 拟合参数回代到每次实验的均值曲线，计算各自的 RMSE / R² |
| 可视化 | 残差散点图（按压力分布）、残差分布直方图、各实验 RMSE 柱状图 |

### 5.4 模块四：加载参数模式

| 需求项 | 规格 |
|--------|------|
| 参数输入 | 手动输入 Hill 参数 (a, b, n) 和/或双曲线参数 (a, b) |
| JSON 导入 | 从之前导出的 JSON 文件自动解析参数 |
| 计算逻辑 | 不重新拟合，直接用已有参数对当前数据计算残差 |
| 适用场景 | 跨批次一致性验证、传感器老化追踪 |

**JSON 格式兼容性（导入时需支持以下 4 种格式）：**

```json
// 格式 1：本工具标准导出
{"fit_mode": {"hill": {"a": 468.87, "b": 11.66, "n": 0.758}}}

// 格式 2：加载参数模式导出
{"loaded_param_mode": {"hill": {"a": 468.87, "b": 11.66, "n": 0.758}}}

// 格式 3：简单嵌套
{"hill": {"a": 468.87, "b": 11.66, "n": 0.758}}

// 格式 4：扁平格式
{"hill_a": 468.87, "hill_b": 11.66, "hill_n": 0.758}
```

### 5.5 模块五：ADC → 压力反推

| 需求项 | 规格 |
|--------|------|
| 输入 | 一个或多个 ADC 均值（逗号或换行分隔） |
| 参数来源 | 可选"拟合结果"或"加载参数" |
| 公式 | x = b · (y / (a − y))^(1/n) |
| 有效域 | 0 < y < a，且结果 x ≤ 100 N |
| 可视化 | 正向曲线（压力→ADC）+ 反向曲线（ADC→压力），查询点标注 |

### 5.6 模块六：数据导出

| 导出格式 | 内容 |
|----------|------|
| CSV | 每行一条记录：模式、文件名、Hill 参数、双曲线参数、RMSE、R² |
| JSON | 结构化导出，包含 fit_mode 和 loaded_param_mode 两个子对象 |

**JSON 导出结构：**

```json
{
  "fit_mode": {
    "fileCount": 5,
    "filenames": ["exp1.csv", "exp2.csv", ...],
    "hill": {"a": 468.87, "b": 11.66, "n": 0.758, "rmse": 1.78, "r2": 0.9993},
    "hyperbolic": {"a": 510.23, "b": 15.42, "rmse": 3.21, "r2": 0.9971},
    "backProjections": [
      {"filename": "exp1.csv", "hill_rmse": 2.1, "hill_r2": 0.996, ...},
      ...
    ]
  },
  "loaded_param_mode": { ... }  // 若使用了加载参数模式
}
```

---

## 6. 配置参数汇总

以下为所有用户可调参数及其默认值，在 Web 实现中应作为表单控件暴露给用户。

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| 压力下限 p_min | 0.0 N | ≥ 0 | 低于此值的数据点被过滤 |
| 压力上限 p_max | 100.0 N | > p_min | 高于此值的数据点被过滤 |
| ADC 最小阈值 val_thresh | 0.0 | ≥ 0 | 单传感器均值低于此值的数据点被过滤 |
| 平滑窗口 smooth_window | 5 | 1 ~ 20 | 滑动平均窗口大小，1 表示不平滑 |
| 异常值去除 remove_outliers | false | — | 是否启用分箱内异常值过滤 |
| 异常值阈值 outlier_threshold | 15.0% | 1 ~ 50 | 偏离箱内均值超过此百分比的点被去除 |
| 反推压力上限 | 100.0 N | — | 反推结果超过此值时标记警告 |

---

## 7. 可视化规格

### 7.1 拟合曲线 Tab

| 图层 | 内容 | 颜色 |
|------|------|------|
| 各次实验均值曲线 | 散点图，每次实验一种颜色 | 蓝/绿/橙/粉/紫 |
| 总平均曲线 | 粗线 | 白色 |
| Hill 拟合曲线 | 平滑线 | 红色 |
| 双曲线拟合曲线 | 平滑线 | 青色 |

### 7.2 残差分析 Tab

| 子图 | 内容 |
|------|------|
| 上部 | Hill vs 双曲线残差对比（按压力分布的散点图） |
| 中部 | 各次实验回代残差散点（按压力分布），含 ±2σ 参考带 |
| 下部 | 残差分布直方图 + 各次实验 RMSE 柱状图 |

### 7.3 各次实验对比 Tab

每次实验一个子图（2 列网格），每个子图包含：左轴为实测值 vs 拟合预测值的散点/线图，右轴为残差柱状图。

### 7.4 反向推算 Tab

| 子图 | 内容 |
|------|------|
| 左图 | 正向曲线（压力→ADC），X 轴 0-100N |
| 右图 | 反向曲线（ADC→压力），Y 轴 0-100N |
| 标注 | 查询点用橙色圆点标注，十字参考线辅助读数 |

---

## 8. Web 实现建议

### 8.1 前端技术选型

| 功能 | 推荐方案 |
|------|---------|
| 图表渲染 | Chart.js（简单场景）或 Plotly.js（交互式图表） |
| CSV 解析 | Papa Parse 库 |
| 数值计算 | math.js 或原生 JavaScript |
| 非线性拟合 | levenberg-marquardt npm 包，或后端 Python scipy |

### 8.2 前后端分工建议

| 功能 | 前端 | 后端 |
|------|------|------|
| CSV 解析 | 可 | 可 |
| 传感器求和/均值 | 可 | 可 |
| 分箱取均值 | 可 | 推荐 |
| 非线性拟合 | 可（levenberg-marquardt） | 推荐（scipy） |
| 反向公式计算 | 推荐（实时响应） | 可 |
| 图表渲染 | 推荐 | — |

### 8.3 核心算法的 JavaScript 实现参考

**Hill 方程：**

```javascript
function hillFunc(x, a, b, n) {
  const xn = Math.pow(Math.abs(x), n);
  const bn = Math.pow(Math.abs(b), n);
  return a * xn / (bn + xn + 1e-12);
}
```

**反向公式：**

```javascript
function inverseHill(y, a, b, n) {
  if (y <= 0) return { x: null, status: 'zero' };
  if (y >= a) return { x: null, status: 'saturated' };
  const x = b * Math.pow(y / (a - y), 1.0 / n);
  if (x > 100) return { x, status: 'out_of_range' };
  return { x, status: 'valid' };
}
```

**RMSE 和 R²：**

```javascript
function computeMetrics(yTrue, yPred) {
  const n = yTrue.length;
  const mean = yTrue.reduce((s, v) => s + v, 0) / n;
  let ssRes = 0, ssTot = 0;
  const residuals = [];
  for (let i = 0; i < n; i++) {
    const r = yTrue[i] - yPred[i];
    residuals.push(r);
    ssRes += r * r;
    ssTot += (yTrue[i] - mean) ** 2;
  }
  return {
    rmse: Math.sqrt(ssRes / n),
    r2: ssTot > 0 ? 1 - ssRes / ssTot : 0,
    residuals
  };
}
```

---

## 9. 版本历史

| 版本 | 主要变更 |
|------|---------|
| v1.0 | 初始版本，独立传感器拟合分析 |
| v1.3 | 核心逻辑重构：同文件 4 传感器求和 → 多次实验均值 → 拟合 → 残差 |
| v1.4 | 新增加载参数模式（不重新拟合，直接计算残差） |
| v1.4.1 | 修复 JSON 导入参数解析的兼容性问题 |
| v1.5 | 求和后除以传感器数量，输出单传感器均值 |
| v1.6 | 新增 ADC → 压力反向推算功能 |
| v1.6.1 | 反推压力范围限制为 0-100N |
| v1.7 | 默认压力范围从 0-200N 改为 0-100N |

---

## References

[1]: [Hill equation (biochemistry) - Wikipedia](https://en.wikipedia.org/wiki/Hill_equation_(biochemistry))

[2]: [Michaelis-Menten kinetics - Wikipedia](https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics)
