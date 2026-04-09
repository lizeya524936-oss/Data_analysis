# hill_core — 传感器 Hill 方程拟合分析核心库

将 `sensor_hill_fit.py` 的全部核心算法拆解为 **6 个独立模块**，无 GUI 依赖，可直接集成到任何 Python 后端、Web 服务或数据分析脚本中。

## 模块架构

```
hill_core/
├── __init__.py          # 统一导出（推荐从这里导入）
├── models.py            # 数学模型：Hill 方程、双曲线、反向公式、评估指标
├── parser.py            # CSV 数据解析：自动识别压力列和传感器列
├── preprocessing.py     # 数据预处理：传感器求和均值、分箱、平滑
├── fitting.py           # 拟合引擎：Hill 非线性拟合、双曲线拟合
├── residuals.py         # 残差分析：回代残差计算
├── io_utils.py          # 导入导出：JSON/CSV 序列化
└── pipeline.py          # 完整流水线：一键分析入口
```

## 依赖

```
numpy>=1.20
scipy>=1.7
```

无 GUI 依赖（不需要 tkinter、customtkinter、matplotlib）。

---

## 快速开始

### 方式一：一键分析（推荐）

```python
from hill_core import analyze_experiments, AnalysisConfig

# 读取多次实验的 CSV 文件
csv_texts = [open(f).read() for f in ["exp1.csv", "exp2.csv", "exp3.csv"]]

# 配置参数（全部可选，有合理默认值）
config = AnalysisConfig(
    p_min=0.0,              # 压力下限 (N)
    p_max=100.0,            # 压力上限 (N)
    smooth_window=5,        # 滑动平均窗口
    remove_outliers=False,  # 是否去除异常值
    outlier_threshold=15.0, # 异常值阈值 (%)
)

# 一键分析
result = analyze_experiments(csv_texts, config=config)

# 查看摘要
print(result.summary())

# 获取 Hill 参数
print(f"a={result.hill_fit.a}, b={result.hill_fit.b}, n={result.hill_fit.n}")
print(f"R²={result.hill_fit.r2}, RMSE={result.hill_fit.rmse}")

# 导出结果
result.export_json("output.json")
result.export_csv("output.csv")
```

### 方式二：逐步调用

```python
from hill_core import (
    parse_csv,
    compute_sensor_average,
    compute_mean_curve,
    fit_hill,
    fit_hyperbolic,
    back_project,
    CurvePoint,
)

# 步骤 1: 解析 CSV
parsed = parse_csv(open("exp1.csv").read(), filename="exp1.csv")
print(f"传感器: {parsed.sensor_ids}")
print(f"数据行: {parsed.row_count}")

# 步骤 2: 计算单传感器均值曲线
curve = compute_sensor_average(parsed.to_legacy_dict(), p_max=100.0)

# 步骤 3: 多次实验取平均
# （假设有多条 curve）
from hill_core.preprocessing import curve_to_legacy_list
mean = compute_mean_curve([curve1, curve2, curve3])

# 步骤 4: 拟合
hill_result = fit_hill(mean.pressures, mean.values)
hyp_result = fit_hyperbolic(mean.pressures, mean.values)

# 步骤 5: 回代残差
legacy_curves = [curve_to_legacy_list(c) for c in [curve1, curve2, curve3]]
bp_results = back_project(
    legacy_curves, filenames=["exp1", "exp2", "exp3"],
    hill_a=hill_result.a, hill_b=hill_result.b, hill_n=hill_result.n,
)
```

---

## 模块详细说明

### 1. models.py — 数学模型

**核心公式：**

| 函数 | 公式 | 说明 |
|------|------|------|
| `hill_func(x, a, b, n)` | y = a·x^n / (b^n + x^n) | Hill 方程正向 |
| `hyperbolic_func(x, a, b)` | y = a·x / (b + x) | 双曲线（Hill n=1 特例） |
| `inverse_hill(y, a, b, n)` | x = b·(y/(a-y))^(1/n) | Hill 反向推算 |
| `compute_metrics(y_true, y_pred)` | RMSE, R² | 评估指标 |

**数据结构：**

```python
FitResult(a, b, n, rmse, r2, residuals)   # 拟合结果
InverseResult(adc_value, pressure, status) # 反推结果
MetricsResult(rmse, r2, residuals)         # 评估指标
```

**反推状态码：**

| status | 含义 |
|--------|------|
| `"valid"` | 有效结果 |
| `"zero"` | ADC ≤ 0，无意义 |
| `"saturated"` | ADC ≥ a，已饱和 |
| `"out_of_range"` | 压力超出 p_max |

### 2. parser.py — CSV 解析

```python
parsed = parse_csv(csv_text, filename="exp1.csv")
# 返回 ParsedCSV 对象
# .sensor_ids  → ["Sensor#1", "Sensor#2", "Sensor#3", "Sensor#4"]
# .rows        → [DataRow(pressure=1.0, sensor_values=[10, 12, 11, 13]), ...]
# .sensor_count → 4
# .row_count    → 500
```

**列识别规则：**
- 压力列：包含 `pressure`、`压力`、`force`、`load`
- 传感器列：包含 `sensor`、`传感器`、`#数字`、`ch数字`、`s数字`
- 兜底：第2列为压力，第3列起为传感器

### 3. preprocessing.py — 数据预处理

```python
# 单传感器均值（4传感器求和 ÷ 4）
curve = compute_sensor_average(parsed.to_legacy_dict(), p_max=100.0)

# 多次实验平均曲线
mean = compute_mean_curve(
    [curve1, curve2, curve3],
    smooth_window=5,
    remove_outliers=True,
    outlier_threshold=15.0,
)

# 单独使用平滑
smoothed = smooth_values(np.array(values), window=5)
```

**处理流程：**
1. 每行 4 传感器值求和 ÷ 传感器数量 → 单传感器均值
2. 多次实验数据合并 → 按压力排序
3. 自适应分箱（0.5~1.0 N 宽度）→ 取均值
4. 滑动窗口平滑（边界对称缩窗）

### 4. fitting.py — 拟合引擎

```python
hill_result = fit_hill(pressures, values)
# FitResult(a=468.87, b=11.66, n=0.758, rmse=1.78, r2=0.9993)

hyp_result = fit_hyperbolic(pressures, values)
# FitResult(a=430.12, b=9.81, n=1.0, rmse=5.23, r2=0.9917)
```

**拟合策略（自动降级）：**
1. 完整 3 参数拟合 (a, b, n)
2. 若失败 → 固定 n=1.0，拟合 2 参数
3. 若仍失败 → 使用初始估计值

**可调参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bounds_a` | (0, 5000) | a 的搜索范围 |
| `bounds_b` | (0.1, 500) | b 的搜索范围 |
| `bounds_n` | (0.1, 5.0) | n 的搜索范围 |
| `max_iter` | 10000 | 最大迭代次数 |

### 5. residuals.py — 残差分析

```python
bp_results = back_project(
    sum_curves=legacy_curves,
    filenames=["exp1.csv", "exp2.csv"],
    hill_a=468.87, hill_b=11.66, hill_n=0.758,
    hyp_a=430.12, hyp_b=9.81,  # 可选
)

for bp in bp_results:
    print(f"{bp.filename}: RMSE={bp.hill_rmse:.4f}, R²={bp.hill_r2:.4f}")
    # bp.pressures       → 压力数组
    # bp.actual_values    → 实测值
    # bp.hill_predicted   → Hill 预测值
    # bp.hill_residuals   → 残差数组
```

### 6. io_utils.py — 导入导出

```python
# 导出 JSON
export_results_json(hill_result, hyp_result, bp_results, "output.json")

# 导出 CSV
export_results_csv(hill_result, hyp_result, bp_results, "output.csv")

# 从 JSON 导入参数（兼容 4 种格式）
params = import_params_from_json("output.json")
# {"hill": {"a": 468.87, "b": 11.66, "n": 0.758}, "hyperbolic": {"a": 430.12, "b": 9.81}}
```

---

## 加载参数模式

用已有参数对新数据计算残差，不重新拟合：

```python
from hill_core import analyze_with_loaded_params, import_params_from_json, AnalysisConfig

# 导入旧参数
params = import_params_from_json("previous_result.json")

# 对新数据计算残差
new_csv_texts = [open(f).read() for f in new_files]
bp_results = analyze_with_loaded_params(
    new_csv_texts,
    hill_a=params["hill"]["a"],
    hill_b=params["hill"]["b"],
    hill_n=params["hill"]["n"],
    config=AnalysisConfig(p_max=100.0),
)

for bp in bp_results:
    print(f"{bp.filename}: RMSE={bp.hill_rmse:.4f}, R²={bp.hill_r2:.4f}")
```

---

## 反向推算（ADC → 压力）

```python
from hill_core import inverse_hill, inverse_hill_batch

# 单个值
result = inverse_hill(y=200, a=468.87, b=11.66, n=0.758, p_max=100.0)
if result.status == "valid":
    print(f"ADC=200 → 压力={result.pressure:.2f} N")

# 批量
results = inverse_hill_batch(
    y_values=[50, 100, 200, 300],
    a=468.87, b=11.66, n=0.758, p_max=100.0,
)
for r in results:
    print(f"ADC={r.adc_value} → {r.pressure or r.status}")
```

---

## 集成到 Web 后端示例

### Flask

```python
from flask import Flask, request, jsonify
from hill_core import analyze_experiments, AnalysisConfig, inverse_hill

app = Flask(__name__)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    files = request.files.getlist("files")
    csv_texts = [f.read().decode("utf-8") for f in files]
    filenames = [f.filename for f in files]
    
    config = AnalysisConfig(
        p_max=float(request.form.get("p_max", 100.0)),
    )
    result = analyze_experiments(csv_texts, filenames, config)
    
    return jsonify({
        "hill": result.hill_fit.to_dict(),
        "hyp": result.hyp_fit.to_dict() if result.hyp_fit else None,
        "back_projections": [bp.to_dict() for bp in result.back_projections],
    })

@app.route("/api/inverse", methods=["POST"])
def inverse():
    data = request.json
    result = inverse_hill(
        y=data["adc"], a=data["a"], b=data["b"], n=data["n"],
        p_max=data.get("p_max", 100.0),
    )
    return jsonify(result.to_dict())
```

### JavaScript/TypeScript 转译参考

核心公式可直接转译为 JS：

```javascript
// Hill 正向
function hillFunc(x, a, b, n) {
    const xn = Math.pow(Math.abs(x), n);
    const bn = Math.pow(Math.abs(b), n);
    return a * xn / (bn + xn + 1e-12);
}

// Hill 反向
function inverseHill(y, a, b, n, pMax = 100) {
    if (y <= 0) return { pressure: null, status: "zero" };
    if (y >= a) return { pressure: null, status: "saturated" };
    const x = b * Math.pow(y / (a - y), 1 / n);
    if (x > pMax + 0.01) return { pressure: x, status: "out_of_range" };
    return { pressure: x, status: "valid" };
}

// 评估指标
function computeMetrics(yTrue, yPred) {
    const n = yTrue.length;
    const residuals = yTrue.map((y, i) => y - yPred[i]);
    const ssRes = residuals.reduce((s, r) => s + r * r, 0);
    const mean = yTrue.reduce((s, y) => s + y, 0) / n;
    const ssTot = yTrue.reduce((s, y) => s + (y - mean) ** 2, 0);
    return {
        rmse: Math.sqrt(ssRes / n),
        r2: ssTot > 0 ? 1 - ssRes / ssTot : 0,
        residuals,
    };
}
```

---

## 版本

v1.7.0 — 与 sensor_hill_fit.py v1.7 算法完全一致
