"""
hill_core.io_utils — 导入导出工具
===================================
参数的 JSON 导入/导出，分析结果的 CSV/JSON 序列化。
仅依赖 Python 标准库。

使用示例:
    from hill_core.io_utils import export_results_json, import_params_from_json

    # 导出
    export_results_json(hill_result, hyp_result, back_projections, "output.json")

    # 导入
    params = import_params_from_json("output.json")
    print(params["hill"])  # {"a": ..., "b": ..., "n": ...}
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import FitResult
from .residuals import BackProjectionResult


# ─── JSON 导出 ────────────────────────────────────────────────────────────────

def export_results_json(
    hill_result: Optional[FitResult],
    hyp_result: Optional[FitResult],
    back_projections: Optional[List[BackProjectionResult]],
    output_path: str,
    filenames: Optional[List[str]] = None,
    extra_data: Optional[dict] = None,
) -> str:
    """
    将分析结果导出为 JSON 文件。

    参数:
        hill_result:      Hill 拟合结果
        hyp_result:       双曲线拟合结果
        back_projections: 回代残差结果列表
        output_path:      输出文件路径
        filenames:        原始文件名列表
        extra_data:       额外数据（会合并到顶层）

    返回:
        输出文件的绝对路径
    """
    data: Dict[str, Any] = {}

    if extra_data:
        data.update(extra_data)

    # 拟合模式结果
    fit_mode: Dict[str, Any] = {}

    if filenames:
        fit_mode["fileCount"] = len(filenames)
        fit_mode["filenames"] = filenames

    if hill_result:
        fit_mode["hill"] = hill_result.to_dict()

    if hyp_result:
        fit_mode["hyperbolic"] = hyp_result.to_dict()

    if back_projections:
        fit_mode["backProjections"] = [bp.to_dict() for bp in back_projections]

    if fit_mode:
        data["fit_mode"] = fit_mode

    path = Path(output_path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path.resolve())


# ─── JSON 导入 ────────────────────────────────────────────────────────────────

def import_params_from_json(filepath: str) -> Dict[str, Dict[str, float]]:
    """
    从 JSON 文件导入 Hill 和双曲线参数。

    兼容以下 4 种格式:
        1. {"fit_mode": {"hill": {"a": ..., "b": ..., "n": ...}}}
        2. {"loaded_param_mode": {"hill": {"a": ..., "b": ..., "n": ...}}}
        3. {"hill": {"a": ..., "b": ..., "n": ...}}
        4. {"hill_a": ..., "hill_b": ..., "hill_n": ...}

    参数:
        filepath: JSON 文件路径

    返回:
        {"hill": {"a": float, "b": float, "n": float},
         "hyperbolic": {"a": float, "b": float} 或 None}

    异常:
        FileNotFoundError: 文件不存在
        ValueError: 无法解析出 Hill 参数
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    data = json.loads(path.read_text(encoding="utf-8"))
    hill_params = None
    hyp_params = None

    # 格式 1: fit_mode 嵌套
    if "fit_mode" in data and isinstance(data["fit_mode"], dict):
        fm = data["fit_mode"]
        if "hill" in fm:
            hill_params = _extract_hill(fm["hill"])
        if "hyperbolic" in fm:
            hyp_params = _extract_hyp(fm["hyperbolic"])

    # 格式 2: loaded_param_mode 嵌套
    if hill_params is None and "loaded_param_mode" in data:
        lm = data["loaded_param_mode"]
        if isinstance(lm, dict) and "hill" in lm:
            hill_params = _extract_hill(lm["hill"])
        if isinstance(lm, dict) and "hyperbolic" in lm:
            hyp_params = _extract_hyp(lm["hyperbolic"])

    # 格式 3: 顶层 hill 键
    if hill_params is None and "hill" in data:
        hill_params = _extract_hill(data["hill"])
    if hyp_params is None and "hyperbolic" in data:
        hyp_params = _extract_hyp(data["hyperbolic"])

    # 格式 4: 扁平格式
    if hill_params is None and "hill_a" in data:
        hill_params = {
            "a": float(data["hill_a"]),
            "b": float(data["hill_b"]),
            "n": float(data.get("hill_n", 1.0)),
        }
    if hyp_params is None and "hyp_a" in data:
        hyp_params = {
            "a": float(data["hyp_a"]),
            "b": float(data["hyp_b"]),
        }

    if hill_params is None:
        top_keys = list(data.keys())
        raise ValueError(
            f"无法从 JSON 中解析 Hill 参数。"
            f"文件顶层键: {top_keys}"
        )

    result = {"hill": hill_params}
    if hyp_params:
        result["hyperbolic"] = hyp_params

    return result


def _extract_hill(obj: dict) -> Optional[Dict[str, float]]:
    """从字典中提取 Hill 参数 a, b, n"""
    if not isinstance(obj, dict):
        return None
    if "a" in obj and "b" in obj:
        return {
            "a": float(obj["a"]),
            "b": float(obj["b"]),
            "n": float(obj.get("n", 1.0)),
        }
    return None


def _extract_hyp(obj: dict) -> Optional[Dict[str, float]]:
    """从字典中提取双曲线参数 a, b"""
    if not isinstance(obj, dict):
        return None
    if "a" in obj and "b" in obj:
        return {"a": float(obj["a"]), "b": float(obj["b"])}
    return None


# ─── CSV 导出 ─────────────────────────────────────────────────────────────────

def export_results_csv(
    hill_result: Optional[FitResult],
    hyp_result: Optional[FitResult],
    back_projections: Optional[List[BackProjectionResult]],
    output_path: str,
) -> str:
    """
    将分析结果导出为 CSV 文件。

    每行一条记录：模式、文件名、参数、RMSE、R²。

    参数:
        hill_result:      Hill 拟合结果
        hyp_result:       双曲线拟合结果
        back_projections: 回代残差结果列表
        output_path:      输出文件路径

    返回:
        输出文件的绝对路径
    """
    path = Path(output_path)
    headers = [
        "类型", "文件名",
        "Hill_a", "Hill_b", "Hill_n", "Hill_RMSE", "Hill_R2",
        "Hyp_a", "Hyp_b", "Hyp_RMSE", "Hyp_R2",
    ]

    rows = []

    # 均值曲线拟合结果
    if hill_result:
        row = ["均值曲线拟合", "—"]
        row += [hill_result.a, hill_result.b, hill_result.n,
                hill_result.rmse, hill_result.r2]
        if hyp_result:
            row += [hyp_result.a, hyp_result.b, hyp_result.rmse, hyp_result.r2]
        else:
            row += ["", "", "", ""]
        rows.append(row)

    # 各次实验回代
    if back_projections:
        for bp in back_projections:
            row = ["回代残差", bp.filename]
            row += [
                hill_result.a if hill_result else "",
                hill_result.b if hill_result else "",
                hill_result.n if hill_result else "",
                bp.hill_rmse, bp.hill_r2,
            ]
            if bp.hyp_rmse is not None:
                row += [
                    hyp_result.a if hyp_result else "",
                    hyp_result.b if hyp_result else "",
                    bp.hyp_rmse, bp.hyp_r2,
                ]
            else:
                row += ["", "", "", ""]
            rows.append(row)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    return str(path.resolve())
