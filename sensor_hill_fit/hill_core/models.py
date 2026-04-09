"""
hill_core.models — 数学模型
============================
包含 Hill 方程、双曲线方程、反向公式和评估指标。
所有函数均为纯函数，无副作用，仅依赖 numpy。

使用示例:
    from hill_core.models import hill_func, inverse_hill, compute_metrics

    y = hill_func(50.0, a=468.87, b=11.66, n=0.758)
    x = inverse_hill(y, a=468.87, b=11.66, n=0.758)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class FitResult:
    """拟合结果的标准数据结构"""
    a: float                   # 饱和值
    b: float                   # 半饱和压力
    n: float = 1.0             # Hill 系数（双曲线时固定为 1.0）
    rmse: float = 0.0          # 均方根误差
    r2: float = 0.0            # 决定系数
    residuals: List[float] = None  # 残差数组

    def __post_init__(self):
        if self.residuals is None:
            self.residuals = []

    def to_dict(self) -> dict:
        """转为字典，便于 JSON 序列化"""
        return {
            "a": round(self.a, 6),
            "b": round(self.b, 6),
            "n": round(self.n, 6),
            "rmse": round(self.rmse, 6),
            "r2": round(self.r2, 6),
        }


@dataclass
class InverseResult:
    """反向推算结果"""
    adc_value: float           # 输入的 ADC 值
    pressure: Optional[float]  # 反推的压力值 (N)，无效时为 None
    status: str                # "valid" | "saturated" | "zero" | "out_of_range"

    def to_dict(self) -> dict:
        return {
            "adc_value": self.adc_value,
            "pressure": round(self.pressure, 6) if self.pressure is not None else None,
            "status": self.status,
        }


@dataclass
class MetricsResult:
    """评估指标结果"""
    rmse: float
    r2: float
    residuals: np.ndarray


# ─── Hill 方程（正向）─────────────────────────────────────────────────────────

def hill_func(x: ArrayLike, a: float, b: float, n: float) -> np.ndarray:
    """
    Hill 方程: y = a * x^n / (b^n + x^n)

    参数:
        x: 压力值，标量或数组 (N)
        a: 饱和值 (ADC 最大值)
        b: 半饱和压力 (N)，当 x=b 时 y=a/2
        n: Hill 系数，控制曲线陡峭程度

    返回:
        y: ADC 预测值，与 x 同形状
    """
    x = np.asarray(x, dtype=float)
    xn = np.power(np.abs(x), n)
    bn = np.power(np.abs(b), n)
    return a * xn / (bn + xn + 1e-12)


# ─── 双曲线方程 ──────────────────────────────────────────────────────────────

def hyperbolic_func(x: ArrayLike, a: float, b: float) -> np.ndarray:
    """
    双曲线方程: y = a * x / (b + x)
    等价于 Hill 方程 n=1 的特例 (Michaelis-Menten)。

    参数:
        x: 压力值，标量或数组 (N)
        a: 饱和值 (ADC 最大值)
        b: 半饱和常数 (N)

    返回:
        y: ADC 预测值
    """
    x = np.asarray(x, dtype=float)
    return a * x / (b + x + 1e-12)


# ─── Hill 方程反向公式 ────────────────────────────────────────────────────────

def inverse_hill(
    y: float,
    a: float,
    b: float,
    n: float,
    p_max: float = 100.0,
) -> InverseResult:
    """
    Hill 方程反向公式: x = b * (y / (a - y))^(1/n)

    已知 ADC 均值 y，反推对应压力 x。

    参数:
        y:     ADC 均值
        a:     Hill 饱和值参数
        b:     Hill 半饱和压力参数
        n:     Hill 系数参数
        p_max: 压力上限 (N)，超出则标记 out_of_range

    返回:
        InverseResult 包含 pressure, status

    有效域:
        0 < y < a，且结果 x <= p_max
    """
    if y <= 0:
        return InverseResult(adc_value=y, pressure=None, status="zero")
    if y >= a:
        return InverseResult(adc_value=y, pressure=None, status="saturated")

    x = b * (y / (a - y)) ** (1.0 / n)

    if x > p_max + 0.01:  # 允许微小浮点误差
        return InverseResult(adc_value=y, pressure=float(x), status="out_of_range")

    return InverseResult(adc_value=y, pressure=float(x), status="valid")


def inverse_hill_batch(
    y_values: List[float],
    a: float,
    b: float,
    n: float,
    p_max: float = 100.0,
) -> List[InverseResult]:
    """
    批量反向推算。

    参数:
        y_values: ADC 均值列表
        a, b, n:  Hill 参数
        p_max:    压力上限

    返回:
        InverseResult 列表
    """
    return [inverse_hill(y, a, b, n, p_max) for y in y_values]


# ─── 评估指标 ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> MetricsResult:
    """
    计算拟合评估指标。

    参数:
        y_true: 实测值数组
        y_pred: 预测值数组

    返回:
        MetricsResult 包含 rmse, r2, residuals

    公式:
        RMSE = sqrt(Σ(y-ŷ)² / N)
        R²   = 1 - Σ(y-ŷ)² / Σ(y-ȳ)²
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    rmse = float(np.sqrt(ss_res / len(y_true)))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return MetricsResult(rmse=rmse, r2=r2, residuals=residuals)
