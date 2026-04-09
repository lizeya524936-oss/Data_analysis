"""
hill_core.preprocessing — 数据预处理
======================================
包含传感器求和均值、多次实验分箱平均、滑动窗口平滑等预处理函数。
仅依赖 numpy。

使用示例:
    from hill_core.parser import parse_csv
    from hill_core.preprocessing import compute_sensor_average, compute_mean_curve

    parsed = parse_csv(csv_text, "exp1.csv")
    curve = compute_sensor_average(parsed.to_legacy_dict(), p_max=100.0)
    avg_p, avg_v = compute_mean_curve([curve1, curve2, curve3])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class CurvePoint:
    """曲线上的一个数据点"""
    pressure: float    # 压力 (N)
    value: float       # ADC 均值

    def to_dict(self) -> dict:
        return {"pressure": self.pressure, "sum_value": self.value}


@dataclass
class MeanCurve:
    """平均曲线"""
    pressures: List[float]  # 压力数组
    values: List[float]     # ADC 均值数组

    @property
    def length(self) -> int:
        return len(self.pressures)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """转为 numpy 数组对"""
        return np.array(self.pressures), np.array(self.values)


# ─── 单传感器均值曲线 ─────────────────────────────────────────────────────────

def compute_sensor_average(
    parsed: dict,
    p_min: float = 0.0,
    p_max: float = 100.0,
    val_thresh: float = 0.0,
) -> List[CurvePoint]:
    """
    对每行数据中的所有传感器值求和后除以传感器数量，
    得到单传感器平均响应曲线。

    参数:
        parsed:     parse_csv 的输出（字典格式，含 sensor_ids 和 rows）
        p_min:      压力下限 (N)，低于此值的数据点被过滤
        p_max:      压力上限 (N)，高于此值的数据点被过滤
        val_thresh: ADC 最小阈值，低于此值的数据点被过滤

    返回:
        CurvePoint 列表，每个点包含 pressure 和 value

    计算公式:
        value = Σ(sensor_values) / sensor_count
    """
    n_sensors = max(len(parsed.get("sensor_ids", [])), 1)
    result = []

    for row in parsed.get("rows", []):
        p = row["pressure"]
        if p <= p_min or p > p_max:
            continue

        sensor_vals = row.get("sensor_values", [])
        avg = sum(sensor_vals) / n_sensors
        if avg <= val_thresh:
            continue

        result.append(CurvePoint(pressure=p, value=avg))

    return result


def curve_to_legacy_list(curve: List[CurvePoint]) -> List[dict]:
    """将 CurvePoint 列表转为与原始代码兼容的字典列表格式"""
    return [{"pressure": pt.pressure, "sum_value": pt.value} for pt in curve]


# ─── 自适应分箱 ──────────────────────────────────────────────────────────────

def _adaptive_binning(
    pressures: np.ndarray,
    values: np.ndarray,
    remove_outliers: bool = False,
    outlier_threshold: float = 15.0,
) -> Tuple[List[float], List[float]]:
    """
    自适应分箱取均值。

    根据数据密度自动计算分箱宽度（0.5 ~ 1.0 N），
    对每个 bin 内的数据点取压力均值和 ADC 均值。

    参数:
        pressures:         压力数组（已排序）
        values:            ADC 值数组
        remove_outliers:   是否去除异常值
        outlier_threshold: 异常值阈值（偏离均值的百分比）

    返回:
        (avg_pressures, avg_values) 分箱后的均值数组
    """
    p_range = pressures[-1] - pressures[0]
    bin_width = max(0.5, min(1.0, p_range / (len(pressures) / 5)))
    bin_count = int(np.ceil(p_range / bin_width))
    bin_edges = np.linspace(pressures[0], pressures[-1], bin_count + 1)

    avg_p, avg_v = [], []

    for i in range(bin_count):
        mask = (pressures >= bin_edges[i]) & (pressures < bin_edges[i + 1])
        if mask.sum() == 0:
            continue

        bin_vals = values[mask]
        bin_pres = pressures[mask]

        # 异常值过滤
        if remove_outliers and len(bin_vals) >= 3:
            bin_mean = np.mean(bin_vals)
            if bin_mean > 0:
                ratio = outlier_threshold / 100.0
                keep = np.abs(bin_vals - bin_mean) / bin_mean <= ratio
                if keep.sum() >= 2:
                    bin_vals = bin_vals[keep]
                    bin_pres = bin_pres[keep]

        avg_p.append(float(np.mean(bin_pres)))
        avg_v.append(float(np.mean(bin_vals)))

    return avg_p, avg_v


# ─── 滑动窗口平滑 ────────────────────────────────────────────────────────────

def smooth_values(values: np.ndarray, window: int = 5) -> np.ndarray:
    """
    滑动窗口平均平滑，边界处使用对称缩窗。

    参数:
        values: 待平滑的值数组
        window: 窗口大小（奇数效果最佳）

    返回:
        平滑后的数组，长度与输入相同
    """
    if window <= 1 or len(values) < 3:
        return values.copy()

    half = window // 2
    smoothed = np.convolve(values, np.ones(window) / window, mode="same")

    # 边界修正：使用对称缩窗
    for i in range(half):
        w = i + 1
        smoothed[i] = np.mean(values[:w * 2 + 1])
        smoothed[-(i + 1)] = np.mean(values[-(w * 2 + 1):])

    return smoothed


# ─── 多次实验平均曲线 ─────────────────────────────────────────────────────────

def compute_mean_curve(
    curves: List[List[CurvePoint]],
    p_min: float = 0.0,
    p_max: float = 100.0,
    smooth_window: int = 5,
    remove_outliers: bool = False,
    outlier_threshold: float = 15.0,
) -> MeanCurve:
    """
    将多次实验的单传感器均值曲线合并，按压力分箱取均值，并平滑。

    参数:
        curves:            多次实验的 CurvePoint 列表
        p_min:             压力下限
        p_max:             压力上限
        smooth_window:     滑动平均窗口大小
        remove_outliers:   是否去除异常值
        outlier_threshold: 异常值阈值（百分比）

    返回:
        MeanCurve 包含平均曲线的 pressures 和 values

    算法步骤:
        1. 合并所有曲线数据点
        2. 按压力排序
        3. 自适应分箱取均值
        4. 滑动窗口平滑
    """
    # 合并
    merged = []
    for curve in curves:
        for pt in curve:
            merged.append((pt.pressure, pt.value))

    if len(merged) < 4:
        return MeanCurve(pressures=[], values=[])

    merged.sort(key=lambda x: x[0])
    pressures = np.array([m[0] for m in merged])
    values = np.array([m[1] for m in merged])

    # 分箱
    avg_p, avg_v = _adaptive_binning(
        pressures, values,
        remove_outliers=remove_outliers,
        outlier_threshold=outlier_threshold,
    )

    if not avg_p:
        return MeanCurve(pressures=[], values=[])

    # 平滑
    avg_v_arr = np.array(avg_v)
    smoothed = smooth_values(avg_v_arr, smooth_window)

    return MeanCurve(pressures=avg_p, values=smoothed.tolist())


def compute_mean_curve_from_legacy(
    sum_curves: List[List[dict]],
    p_min: float = 0.0,
    p_max: float = 100.0,
    smooth_window: int = 5,
    remove_outliers: bool = False,
    outlier_threshold: float = 15.0,
) -> Tuple[List[float], List[float]]:
    """
    兼容原始代码格式的平均曲线计算。

    参数:
        sum_curves: 原始格式的曲线列表 [{"pressure": ..., "sum_value": ...}, ...]

    返回:
        (pressures, values) 元组
    """
    curves = []
    for sc in sum_curves:
        curve = [CurvePoint(pressure=pt["pressure"], value=pt["sum_value"]) for pt in sc]
        curves.append(curve)

    result = compute_mean_curve(
        curves, p_min, p_max, smooth_window, remove_outliers, outlier_threshold
    )
    return result.pressures, result.values
