"""
hill_core.residuals — 残差分析
================================
将拟合参数回代到各次实验数据，计算残差和评估指标。
依赖 numpy 和 hill_core.models。

使用示例:
    from hill_core.residuals import back_project

    results = back_project(
        sum_curves=curves,
        filenames=["exp1.csv", "exp2.csv"],
        hill_a=468.87, hill_b=11.66, hill_n=0.758,
    )
    for r in results:
        print(f"{r.filename}: RMSE={r.hill_rmse:.4f}, R²={r.hill_r2:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import hill_func, hyperbolic_func, compute_metrics
from .preprocessing import smooth_values


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class BackProjectionResult:
    """单次实验的回代残差结果"""
    filename: str
    pressures: List[float]
    actual_values: List[float]      # 实测值（分箱平滑后）
    hill_predicted: List[float]     # Hill 预测值
    hill_rmse: float
    hill_r2: float
    hill_residuals: List[float]     # 残差 = 实测 - 预测
    hyp_predicted: Optional[List[float]] = None
    hyp_rmse: Optional[float] = None
    hyp_r2: Optional[float] = None
    hyp_residuals: Optional[List[float]] = None

    def to_dict(self) -> dict:
        d = {
            "filename": self.filename,
            "pressures": self.pressures,
            "actual_values": self.actual_values,
            "hill_predicted": self.hill_predicted,
            "hill_rmse": round(self.hill_rmse, 6),
            "hill_r2": round(self.hill_r2, 6),
            "hill_residuals": self.hill_residuals,
        }
        if self.hyp_predicted is not None:
            d.update({
                "hyp_predicted": self.hyp_predicted,
                "hyp_rmse": round(self.hyp_rmse, 6),
                "hyp_r2": round(self.hyp_r2, 6),
                "hyp_residuals": self.hyp_residuals,
            })
        return d


# ─── 回代残差计算 ─────────────────────────────────────────────────────────────

def back_project(
    sum_curves: List[List[dict]],
    filenames: List[str],
    hill_a: float,
    hill_b: float,
    hill_n: float,
    hyp_a: Optional[float] = None,
    hyp_b: Optional[float] = None,
    p_min: float = 0.0,
    p_max: float = 100.0,
    smooth_window: int = 5,
) -> List[BackProjectionResult]:
    """
    将拟合参数回代到每次实验的均值曲线，计算残差。

    对每次实验的数据进行分箱平滑后，用 Hill 参数（和可选的双曲线参数）
    预测 ADC 值，并计算 RMSE 和 R²。

    参数:
        sum_curves:    各次实验的均值曲线列表
                       每条曲线为 [{"pressure": float, "sum_value": float}, ...]
        filenames:     对应的文件名列表
        hill_a, hill_b, hill_n: Hill 拟合参数
        hyp_a, hyp_b:  双曲线拟合参数（可选）
        p_min, p_max:  压力范围
        smooth_window: 平滑窗口

    返回:
        BackProjectionResult 列表，每次实验一条

    算法步骤:
        1. 对每次实验数据按压力排序
        2. 自适应分箱取均值
        3. 滑动窗口平滑
        4. 用 Hill 参数计算预测值
        5. 计算残差和评估指标
    """
    results = []

    for curve, fname in zip(sum_curves, filenames):
        if len(curve) < 4:
            continue

        # 排序
        pts = sorted(curve, key=lambda x: x["pressure"])
        pressures = np.array([pt["pressure"] for pt in pts])
        values = np.array([pt["sum_value"] for pt in pts])

        # 分箱
        p_range = pressures[-1] - pressures[0]
        if p_range <= 0:
            continue
        bin_width = max(0.5, min(1.0, p_range / (len(pts) / 5)))
        bin_count = int(np.ceil(p_range / bin_width))
        bin_edges = np.linspace(pressures[0], pressures[-1], bin_count + 1)

        bp_list, bv_list = [], []
        for i in range(bin_count):
            mask = (pressures >= bin_edges[i]) & (pressures < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bp_list.append(float(np.mean(pressures[mask])))
            bv_list.append(float(np.mean(values[mask])))

        if not bp_list:
            continue

        bp_arr = np.array(bp_list)
        bv_arr = np.array(bv_list)

        # 平滑
        sv = smooth_values(bv_arr, smooth_window)

        # Hill 回代
        hill_pred = hill_func(bp_arr, hill_a, hill_b, hill_n)
        h_metrics = compute_metrics(sv, hill_pred)

        entry = BackProjectionResult(
            filename=fname,
            pressures=bp_arr.tolist(),
            actual_values=sv.tolist(),
            hill_predicted=hill_pred.tolist(),
            hill_rmse=h_metrics.rmse,
            hill_r2=h_metrics.r2,
            hill_residuals=h_metrics.residuals.tolist(),
        )

        # 双曲线回代（可选）
        if hyp_a is not None and hyp_b is not None:
            hyp_pred = hyperbolic_func(bp_arr, hyp_a, hyp_b)
            y_metrics = compute_metrics(sv, hyp_pred)
            entry.hyp_predicted = hyp_pred.tolist()
            entry.hyp_rmse = y_metrics.rmse
            entry.hyp_r2 = y_metrics.r2
            entry.hyp_residuals = y_metrics.residuals.tolist()

        results.append(entry)

    return results
