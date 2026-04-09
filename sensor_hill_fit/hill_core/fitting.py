"""
hill_core.fitting — 拟合引擎
==============================
非线性最小二乘拟合：Hill 方程和双曲线方程。
依赖 scipy.optimize.curve_fit 和 numpy。

使用示例:
    from hill_core.fitting import fit_hill, fit_hyperbolic

    result = fit_hill(pressures, values)
    print(f"a={result.a:.2f}, b={result.b:.2f}, n={result.n:.3f}")
    print(f"R²={result.r2:.4f}, RMSE={result.rmse:.4f}")
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

from .models import FitResult, MetricsResult, hill_func, hyperbolic_func, compute_metrics

warnings.filterwarnings("ignore", category=OptimizeWarning)


# ─── Hill 方程拟合 ────────────────────────────────────────────────────────────

def fit_hill(
    pressures: List[float],
    values: List[float],
    p0: Optional[List[float]] = None,
    bounds_a: Tuple[float, float] = (0, 5000),
    bounds_b: Tuple[float, float] = (0.1, 500),
    bounds_n: Tuple[float, float] = (0.1, 5.0),
    max_iter: int = 10000,
) -> FitResult:
    """
    对压力-ADC 数据进行 Hill 方程非线性最小二乘拟合。

    拟合公式: y = a * x^n / (b^n + x^n)

    参数:
        pressures: 压力值数组 (N)
        values:    ADC 均值数组
        p0:        初始猜测 [a, b, n]，为 None 时自动估计
        bounds_a:  参数 a 的边界 (min, max)
        bounds_b:  参数 b 的边界 (min, max)
        bounds_n:  参数 n 的边界 (min, max)
        max_iter:  最大迭代次数

    返回:
        FitResult 包含 a, b, n, rmse, r2, residuals

    拟合策略:
        1. 主拟合: 3 参数 (a, b, n)
        2. 降级拟合: 若主拟合失败，固定 n=1.0，拟合 2 参数
        3. 兜底: 若均失败，使用初始估计值
    """
    p = np.array(pressures, dtype=float)
    v = np.array(values, dtype=float)

    # 自动初始估计
    a0 = float(np.max(v) * 1.1)
    b0 = float(np.median(p))
    n0 = 0.9
    init = p0 if p0 else [a0, b0, n0]

    # 策略 1: 完整 3 参数拟合
    try:
        popt, _ = curve_fit(
            hill_func, p, v, p0=init,
            bounds=(
                [bounds_a[0], bounds_b[0], bounds_n[0]],
                [bounds_a[1], bounds_b[1], bounds_n[1]],
            ),
            maxfev=max_iter,
        )
        a, b, n = float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        # 策略 2: 固定 n=1.0 的 2 参数拟合
        try:
            popt2, _ = curve_fit(
                lambda x, a, b: hill_func(x, a, b, 1.0),
                p, v, p0=[a0, b0],
                bounds=(
                    [bounds_a[0], bounds_b[0]],
                    [bounds_a[1], bounds_b[1]],
                ),
                maxfev=max_iter // 2,
            )
            a, b, n = float(popt2[0]), float(popt2[1]), 1.0
        except Exception:
            # 策略 3: 兜底
            a, b, n = a0, b0, 1.0

    # 计算拟合指标
    y_pred = hill_func(p, a, b, n)
    metrics = compute_metrics(v, y_pred)

    return FitResult(
        a=a, b=b, n=n,
        rmse=metrics.rmse,
        r2=metrics.r2,
        residuals=metrics.residuals.tolist(),
    )


# ─── 双曲线拟合 ──────────────────────────────────────────────────────────────

def fit_hyperbolic(
    pressures: List[float],
    values: List[float],
    bounds_a: Tuple[float, float] = (0, 5000),
    bounds_b: Tuple[float, float] = (0.1, 500),
    max_iter: int = 5000,
) -> FitResult:
    """
    对压力-ADC 数据进行双曲线方程拟合。

    拟合公式: y = a * x / (b + x)  (等价于 Hill 方程 n=1)

    参数:
        pressures: 压力值数组 (N)
        values:    ADC 均值数组
        bounds_a:  参数 a 的边界
        bounds_b:  参数 b 的边界
        max_iter:  最大迭代次数

    返回:
        FitResult 包含 a, b (n 固定为 1.0), rmse, r2, residuals
    """
    p = np.array(pressures, dtype=float)
    v = np.array(values, dtype=float)

    a0 = float(np.max(v) * 1.1)
    b0 = float(np.median(p))

    try:
        popt, _ = curve_fit(
            hyperbolic_func, p, v, p0=[a0, b0],
            bounds=(
                [bounds_a[0], bounds_b[0]],
                [bounds_a[1], bounds_b[1]],
            ),
            maxfev=max_iter,
        )
        a, b = float(popt[0]), float(popt[1])
    except Exception:
        a, b = a0, b0

    y_pred = hyperbolic_func(p, a, b)
    metrics = compute_metrics(v, y_pred)

    return FitResult(
        a=a, b=b, n=1.0,
        rmse=metrics.rmse,
        r2=metrics.r2,
        residuals=metrics.residuals.tolist(),
    )
