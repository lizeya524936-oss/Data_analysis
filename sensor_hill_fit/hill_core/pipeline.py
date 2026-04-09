"""
hill_core.pipeline — 完整分析流水线
=====================================
将所有模块串联为一键调用的高层 API。
这是集成到其他项目时最推荐使用的入口。

使用示例:
    from hill_core.pipeline import analyze_experiments, AnalysisConfig, AnalysisResult

    # 读取 CSV 文件
    csv_texts = [open(f).read() for f in csv_files]

    # 一键分析
    result = analyze_experiments(csv_texts, filenames=csv_files)

    # 获取结果
    print(result.hill_fit)           # FitResult
    print(result.mean_curve)         # MeanCurve
    print(result.back_projections)   # [BackProjectionResult, ...]

    # 反推压力
    inv = result.inverse(adc_value=200.0)
    print(f"ADC=200 → 压力={inv.pressure:.2f} N")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .models import FitResult, InverseResult, inverse_hill, inverse_hill_batch
from .parser import ParsedCSV, parse_csv
from .preprocessing import (
    CurvePoint,
    MeanCurve,
    compute_mean_curve,
    compute_sensor_average,
    curve_to_legacy_list,
)
from .fitting import fit_hill, fit_hyperbolic
from .residuals import BackProjectionResult, back_project
from .io_utils import export_results_json, export_results_csv, import_params_from_json


# ─── 配置 ────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    """分析配置参数"""
    p_min: float = 0.0              # 压力下限 (N)
    p_max: float = 100.0            # 压力上限 (N)
    val_thresh: float = 0.0         # ADC 最小阈值
    smooth_window: int = 5          # 滑动平均窗口
    remove_outliers: bool = False   # 是否去除异常值
    outlier_threshold: float = 15.0 # 异常值阈值 (%)
    fit_hyperbolic: bool = True     # 是否同时进行双曲线拟合


# ─── 分析结果 ─────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """完整分析结果"""
    config: AnalysisConfig
    parsed_files: List[ParsedCSV]
    sensor_curves: List[List[CurvePoint]]  # 各次实验的单传感器均值曲线
    mean_curve: MeanCurve                  # 总平均曲线
    hill_fit: FitResult                    # Hill 拟合结果
    hyp_fit: Optional[FitResult]           # 双曲线拟合结果
    back_projections: List[BackProjectionResult]  # 各次实验回代残差

    def inverse(self, adc_value: float) -> InverseResult:
        """用 Hill 参数反推单个 ADC 值对应的压力"""
        return inverse_hill(
            adc_value,
            self.hill_fit.a,
            self.hill_fit.b,
            self.hill_fit.n,
            p_max=self.config.p_max,
        )

    def inverse_batch(self, adc_values: List[float]) -> List[InverseResult]:
        """批量反推"""
        return inverse_hill_batch(
            adc_values,
            self.hill_fit.a,
            self.hill_fit.b,
            self.hill_fit.n,
            p_max=self.config.p_max,
        )

    def export_json(self, output_path: str) -> str:
        """导出结果为 JSON"""
        filenames = [p.filename for p in self.parsed_files]
        return export_results_json(
            self.hill_fit, self.hyp_fit,
            self.back_projections, output_path,
            filenames=filenames,
        )

    def export_csv(self, output_path: str) -> str:
        """导出结果为 CSV"""
        return export_results_csv(
            self.hill_fit, self.hyp_fit,
            self.back_projections, output_path,
        )

    def summary(self) -> str:
        """生成文本摘要"""
        lines = [
            f"═══ 分析摘要 ═══",
            f"文件数量: {len(self.parsed_files)}",
            f"压力范围: {self.config.p_min} ~ {self.config.p_max} N",
            f"平均曲线数据点: {self.mean_curve.length}",
            f"",
            f"── Hill 拟合 ──",
            f"  a (饱和值)    = {self.hill_fit.a:.4f}",
            f"  b (半饱和压力) = {self.hill_fit.b:.4f} N",
            f"  n (Hill系数)  = {self.hill_fit.n:.4f}",
            f"  RMSE          = {self.hill_fit.rmse:.4f}",
            f"  R²            = {self.hill_fit.r2:.6f}",
        ]

        if self.hyp_fit:
            lines += [
                f"",
                f"── 双曲线拟合 ──",
                f"  a (饱和值)    = {self.hyp_fit.a:.4f}",
                f"  b (半饱和常数) = {self.hyp_fit.b:.4f} N",
                f"  RMSE          = {self.hyp_fit.rmse:.4f}",
                f"  R²            = {self.hyp_fit.r2:.6f}",
            ]

        if self.back_projections:
            lines += [f"", f"── 各次实验回代 ──"]
            for bp in self.back_projections:
                lines.append(
                    f"  {bp.filename}: "
                    f"Hill RMSE={bp.hill_rmse:.4f}, R²={bp.hill_r2:.4f}"
                )

        return "\n".join(lines)


# ─── 主流水线 ─────────────────────────────────────────────────────────────────

def analyze_experiments(
    csv_texts: List[str],
    filenames: Optional[List[str]] = None,
    config: Optional[AnalysisConfig] = None,
) -> AnalysisResult:
    """
    完整分析流水线：一键从 CSV 文本到拟合结果。

    参数:
        csv_texts: CSV 文件文本内容列表（每个元素 = 一次实验）
        filenames: 对应的文件名列表（可选，默认自动编号）
        config:    分析配置（可选，使用默认值）

    返回:
        AnalysisResult 包含所有分析结果

    流程:
        1. 解析 CSV → ParsedCSV
        2. 计算各实验的单传感器均值曲线
        3. 合并计算总平均曲线
        4. Hill 方程拟合 + 双曲线拟合
        5. 回代残差分析
    """
    if config is None:
        config = AnalysisConfig()

    if filenames is None:
        filenames = [f"experiment_{i+1}.csv" for i in range(len(csv_texts))]

    # 步骤 1: 解析 CSV
    parsed_files = []
    for text, fname in zip(csv_texts, filenames):
        parsed = parse_csv(text, filename=fname)
        parsed_files.append(parsed)

    # 步骤 2: 计算各实验的单传感器均值曲线
    sensor_curves = []
    legacy_curves = []  # 兼容格式，用于回代
    for pf in parsed_files:
        curve = compute_sensor_average(
            pf.to_legacy_dict(),
            p_min=config.p_min,
            p_max=config.p_max,
            val_thresh=config.val_thresh,
        )
        sensor_curves.append(curve)
        legacy_curves.append(curve_to_legacy_list(curve))

    # 步骤 3: 计算总平均曲线
    mean_curve = compute_mean_curve(
        sensor_curves,
        p_min=config.p_min,
        p_max=config.p_max,
        smooth_window=config.smooth_window,
        remove_outliers=config.remove_outliers,
        outlier_threshold=config.outlier_threshold,
    )

    if mean_curve.length < 4:
        raise ValueError("平均曲线数据点不足（< 4），无法拟合。请检查数据或调整参数。")

    # 步骤 4: 拟合
    hill_result = fit_hill(mean_curve.pressures, mean_curve.values)

    hyp_result = None
    if config.fit_hyperbolic:
        hyp_result = fit_hyperbolic(mean_curve.pressures, mean_curve.values)

    # 步骤 5: 回代残差
    bp_results = back_project(
        sum_curves=legacy_curves,
        filenames=[pf.filename for pf in parsed_files],
        hill_a=hill_result.a,
        hill_b=hill_result.b,
        hill_n=hill_result.n,
        hyp_a=hyp_result.a if hyp_result else None,
        hyp_b=hyp_result.b if hyp_result else None,
        p_min=config.p_min,
        p_max=config.p_max,
        smooth_window=config.smooth_window,
    )

    return AnalysisResult(
        config=config,
        parsed_files=parsed_files,
        sensor_curves=sensor_curves,
        mean_curve=mean_curve,
        hill_fit=hill_result,
        hyp_fit=hyp_result,
        back_projections=bp_results,
    )


# ─── 加载参数模式 ─────────────────────────────────────────────────────────────

def analyze_with_loaded_params(
    csv_texts: List[str],
    hill_a: float,
    hill_b: float,
    hill_n: float,
    hyp_a: Optional[float] = None,
    hyp_b: Optional[float] = None,
    filenames: Optional[List[str]] = None,
    config: Optional[AnalysisConfig] = None,
) -> List[BackProjectionResult]:
    """
    加载参数模式：用已有参数对新数据计算残差，不重新拟合。

    参数:
        csv_texts:         CSV 文件文本内容列表
        hill_a, hill_b, hill_n: 已有的 Hill 参数
        hyp_a, hyp_b:      已有的双曲线参数（可选）
        filenames:          文件名列表
        config:             分析配置

    返回:
        BackProjectionResult 列表

    适用场景:
        - 跨批次一致性验证
        - 传感器老化追踪
        - 用标准参数评估新传感器
    """
    if config is None:
        config = AnalysisConfig()

    if filenames is None:
        filenames = [f"experiment_{i+1}.csv" for i in range(len(csv_texts))]

    # 解析 + 均值曲线
    legacy_curves = []
    for text, fname in zip(csv_texts, filenames):
        parsed = parse_csv(text, filename=fname)
        curve = compute_sensor_average(
            parsed.to_legacy_dict(),
            p_min=config.p_min,
            p_max=config.p_max,
            val_thresh=config.val_thresh,
        )
        legacy_curves.append(curve_to_legacy_list(curve))

    # 回代残差
    return back_project(
        sum_curves=legacy_curves,
        filenames=filenames,
        hill_a=hill_a, hill_b=hill_b, hill_n=hill_n,
        hyp_a=hyp_a, hyp_b=hyp_b,
        p_min=config.p_min,
        p_max=config.p_max,
        smooth_window=config.smooth_window,
    )
