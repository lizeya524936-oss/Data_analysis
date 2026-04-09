"""
hill_core — 传感器 Hill 方程拟合分析核心库
============================================
将 sensor_hill_fit.py 的核心算法拆解为 6 个独立模块，
无 UI 依赖，可直接导入到任何 Python 项目或 Web 后端。

模块结构:
    hill_core/
    ├── __init__.py          ← 本文件（统一导出）
    ├── models.py            ← 数学模型：Hill 方程、双曲线、反向公式、评估指标
    ├── parser.py            ← CSV 数据解析
    ├── preprocessing.py     ← 数据预处理：求和均值、分箱、平滑
    ├── fitting.py           ← 拟合引擎：Hill 拟合、双曲线拟合
    ├── residuals.py         ← 残差分析：回代残差计算
    ├── io_utils.py          ← 导入导出：JSON/CSV
    └── pipeline.py          ← 完整流水线：一键分析入口

快速使用:
    from hill_core import analyze_experiments, AnalysisConfig

    csv_texts = [open(f).read() for f in csv_files]
    result = analyze_experiments(csv_texts, filenames=csv_files)
    print(result.summary())
"""

# ── 数学模型 ──
from .models import (
    hill_func,
    hyperbolic_func,
    inverse_hill,
    inverse_hill_batch,
    compute_metrics,
    FitResult,
    InverseResult,
    MetricsResult,
)

# ── CSV 解析 ──
from .parser import parse_csv, ParsedCSV, DataRow

# ── 数据预处理 ──
from .preprocessing import (
    compute_sensor_average,
    compute_mean_curve,
    compute_mean_curve_from_legacy,
    smooth_values,
    CurvePoint,
    MeanCurve,
)

# ── 拟合引擎 ──
from .fitting import fit_hill, fit_hyperbolic

# ── 残差分析 ──
from .residuals import back_project, BackProjectionResult

# ── 导入导出 ──
from .io_utils import (
    export_results_json,
    export_results_csv,
    import_params_from_json,
)

# ── 流水线 ──
from .pipeline import (
    analyze_experiments,
    analyze_with_loaded_params,
    AnalysisConfig,
    AnalysisResult,
)

__version__ = "1.7.0"
__all__ = [
    # 模型
    "hill_func", "hyperbolic_func", "inverse_hill", "inverse_hill_batch",
    "compute_metrics", "FitResult", "InverseResult", "MetricsResult",
    # 解析
    "parse_csv", "ParsedCSV", "DataRow",
    # 预处理
    "compute_sensor_average", "compute_mean_curve", "compute_mean_curve_from_legacy",
    "smooth_values", "CurvePoint", "MeanCurve",
    # 拟合
    "fit_hill", "fit_hyperbolic",
    # 残差
    "back_project", "BackProjectionResult",
    # 导入导出
    "export_results_json", "export_results_csv", "import_params_from_json",
    # 流水线
    "analyze_experiments", "analyze_with_loaded_params",
    "AnalysisConfig", "AnalysisResult",
]
