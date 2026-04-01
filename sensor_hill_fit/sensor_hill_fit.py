"""
传感器 Hill 方程拟合分析工具  v1.5
=======================================
核心分析逻辑（v1.3 重构，v1.4 新增加载参数模式，v1.5 改为单传感器均值）：
  1. 每个 CSV 文件包含同一次实验的 4 个传感器列
  2. 对每行按压力分箱，将 4 个传感器值求和后除以传感器数量 → 得到该次实验的"单传感器均值曲线"
  3. 将多个文件（多次实验）的均值曲线按压力分箱取均值 → 得到"总平均曲线"
  4. 对平均曲线进行 Hill 方程 & 双曲线拟合  [拟合模式]
     OR 直接使用已有参数，不重新拟合       [加载参数模式 v1.4]
  5. 将拟合曲线回代到每次实验的均值曲线，计算残差

v1.4 新增功能：
  - 左侧面板新增"加载拟合参数"区域
    * 可手动输入 Hill 参数 (a, b, n) 和双曲线参数 (a, b)
    * 可从之前导出的 JSON 文件一键导入参数
    * 点击"▶ 加载参数计算残差"按钮，跳过拟合直接计算残差
  - 新增 Tab "加载参数残差" 展示加载参数模式的结果
    * 顶部：各次实验求和曲线 + 加载的 Hill/双曲线预测曲线
    * 中部：各次实验回代残差散点（按压力）
    * 底部：残差分布直方图

依赖：
  pip install scipy customtkinter matplotlib pandas numpy
"""

import os
import json
import re
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit, OptimizeWarning
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

warnings.filterwarnings("ignore", category=OptimizeWarning)

# ─── 全局样式 ──────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT_TITLE = ("Microsoft YaHei", 14, "bold")
FONT_LABEL = ("Microsoft YaHei", 11)
FONT_SMALL = ("Microsoft YaHei", 10)
FONT_MONO  = ("Consolas", 10)

try:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei",
                                        "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

COLORS_EXP  = ["#4FC3F7", "#81C784", "#FFB74D", "#F06292", "#CE93D8"]
COLOR_AVG   = "#FFFFFF"
COLOR_HILL  = "#FF6B6B"
COLOR_HYP   = "#4ECDC4"
COLOR_RESID = "#FFB74D"
COLOR_LOAD  = "#A5D6A7"   # 加载参数模式专用颜色

# ─── 拟合算法 ──────────────────────────────────────────────────────────────────

def hill_func(x, a, b, n):
    """Hill 方程: y = a * x^n / (b^n + x^n)"""
    xn = np.power(np.abs(x), n)
    bn = np.power(np.abs(b), n)
    return a * xn / (bn + xn + 1e-12)

def hyperbolic_func(x, a, b):
    """双曲线: y = a * x / (b + x)"""
    return a * x / (b + x + 1e-12)

def compute_metrics(y_true, y_pred):
    """计算 RMSE 和 R²"""
    residuals = np.array(y_true) - np.array(y_pred)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
    rmse = np.sqrt(ss_res / len(y_true))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(rmse), float(r2), residuals

def fit_hill(pressures, values, p0=None):
    p = np.array(pressures, dtype=float)
    v = np.array(values, dtype=float)
    a0 = float(np.max(v) * 1.1)
    b0 = float(np.median(p))
    n0 = 0.9
    init = p0 if p0 else [a0, b0, n0]
    try:
        popt, _ = curve_fit(
            hill_func, p, v, p0=init,
            bounds=([0, 0.1, 0.1], [5000, 500, 5.0]),
            maxfev=10000,
        )
        a, b, n = popt
    except Exception:
        try:
            popt2, _ = curve_fit(
                lambda x, a, b: hill_func(x, a, b, 1.0),
                p, v, p0=[a0, b0],
                bounds=([0, 0.1], [5000, 500]),
                maxfev=5000,
            )
            a, b, n = popt2[0], popt2[1], 1.0
        except Exception:
            a, b, n = a0, b0, 1.0
    y_pred = hill_func(p, a, b, n)
    rmse, r2, resid = compute_metrics(v, y_pred)
    return a, b, n, rmse, r2, resid.tolist()

def fit_hyperbolic(pressures, values):
    p = np.array(pressures, dtype=float)
    v = np.array(values, dtype=float)
    a0 = float(np.max(v) * 1.1)
    b0 = float(np.median(p))
    try:
        popt, _ = curve_fit(
            hyperbolic_func, p, v, p0=[a0, b0],
            bounds=([0, 0.1], [5000, 500]),
            maxfev=5000,
        )
        a, b = popt
    except Exception:
        a, b = a0, b0
    y_pred = hyperbolic_func(p, a, b)
    rmse, r2, resid = compute_metrics(v, y_pred)
    return a, b, rmse, r2, resid.tolist()

# ─── CSV 解析 ──────────────────────────────────────────────────────────────────

def parse_csv(content: str, filename: str = "") -> dict:
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"文件 {filename} 为空")
    header_idx = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if "pressure" in low or "传感器" in low or "sensor" in low or "压力" in low:
            header_idx = i
            break
    header = [h.strip().strip('"').strip("'").lstrip('\ufeff')
              for h in lines[header_idx].split(",")]
    data_lines = lines[header_idx + 1:]
    pressure_col = None
    sensor_cols = []
    for i, h in enumerate(header):
        hl = h.lower()
        if "pressure" in hl or "压力" in hl or hl == "force" or hl == "load":
            pressure_col = i
        elif re.search(r"sensor|传感器|#\d+|ch\d+|s\d+", h, re.I):
            sensor_cols.append((i, h))
    if pressure_col is None:
        pressure_col = 1
    if not sensor_cols:
        for i in range(2, len(header)):
            sensor_cols.append((i, header[i]))
    sensor_ids = [h for _, h in sensor_cols]
    rows = []
    for line in data_lines:
        parts = line.split(",")
        max_col = max(pressure_col, max((i for i, _ in sensor_cols), default=0))
        if len(parts) <= max_col:
            continue
        try:
            pressure = float(parts[pressure_col])
        except (ValueError, IndexError):
            continue
        vals = []
        for col_idx, _ in sensor_cols:
            try:
                vals.append(float(parts[col_idx]))
            except (ValueError, IndexError):
                vals.append(0.0)
        rows.append({"pressure": pressure, "sensor_values": vals})
    return {"filename": filename, "sensor_ids": sensor_ids, "rows": rows}

# ─── 核心：单传感器均值曲线（求和后除以传感器数量）────────────────────────────────
def compute_sum_curve(parsed: dict, p_min: float, p_max: float,
                      val_thresh: float) -> list:
    """对每行的传感器值求和后除以传感器数量，得到单传感器平均值曲线。
    返回列表中的 sum_value 字段实际为单传感器均值（保持字段名兼容性）。
    """
    n_sensors = max(len(parsed["sensor_ids"]), 1)  # 防止除以零
    result = []
    for row in parsed["rows"]:
        p = row["pressure"]
        if p <= p_min or p > p_max:
            continue
        s = sum(row["sensor_values"]) / n_sensors  # 除以传感器数量，得到单传感器均值
        if s <= val_thresh:
            continue
        result.append({"pressure": p, "sum_value": s})
    return result

# ─── 核心：均值曲线 ────────────────────────────────────────────────────────────

def compute_average_from_sum_curves(
    sum_curves: list,
    p_min: float, p_max: float,
    smooth_window: int = 5,
    remove_outliers: bool = False,
    outlier_threshold: float = 15.0,
) -> tuple:
    merged = []
    for curve in sum_curves:
        for pt in curve:
            merged.append((pt["pressure"], pt["sum_value"]))
    if len(merged) < 4:
        return [], []
    merged.sort(key=lambda x: x[0])
    pressures = np.array([m[0] for m in merged])
    values    = np.array([m[1] for m in merged])
    p_range   = pressures[-1] - pressures[0]
    bin_width = max(0.5, min(1.0, p_range / (len(merged) / 5)))
    bin_count = int(np.ceil(p_range / bin_width))
    bin_edges = np.linspace(pressures[0], pressures[-1], bin_count + 1)
    avg_p, avg_v = [], []
    for i in range(bin_count):
        mask = (pressures >= bin_edges[i]) & (pressures < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_vals = values[mask]
        bin_pres = pressures[mask]
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
    avg_p = np.array(avg_p)
    avg_v = np.array(avg_v)
    half  = smooth_window // 2
    smoothed_v = np.convolve(avg_v, np.ones(smooth_window) / smooth_window, mode="same")
    for i in range(half):
        w = i + 1
        smoothed_v[i]      = np.mean(avg_v[:w * 2 + 1])
        smoothed_v[-(i+1)] = np.mean(avg_v[-(w * 2 + 1):])
    return avg_p.tolist(), smoothed_v.tolist()

# ─── 核心：回代残差 ────────────────────────────────────────────────────────────

def back_project_to_sum_curves(
    sum_curves: list,
    filenames: list,
    hill_a, hill_b, hill_n,
    hyp_a=None, hyp_b=None,
    p_min=0.0, p_max=200.0,
    smooth_window=5,
) -> list:
    results = []
    for curve, fname in zip(sum_curves, filenames):
        if len(curve) < 4:
            continue
        pts = sorted(curve, key=lambda x: x["pressure"])
        pressures = np.array([pt["pressure"] for pt in pts])
        values    = np.array([pt["sum_value"]  for pt in pts])
        p_range   = pressures[-1] - pressures[0]
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
        bp_arr = np.array(bp_list)
        bv_arr = np.array(bv_list)
        half   = smooth_window // 2
        sv = np.convolve(bv_arr, np.ones(smooth_window) / smooth_window, mode="same")
        for i in range(half):
            w = i + 1
            sv[i]      = np.mean(bv_arr[:w * 2 + 1])
            sv[-(i+1)] = np.mean(bv_arr[-(w * 2 + 1):])
        hill_pred = hill_func(bp_arr, hill_a, hill_b, hill_n)
        h_rmse, h_r2, h_resid = compute_metrics(sv, hill_pred)
        entry = {
            "filename": fname,
            "pressures": bp_arr.tolist(),
            "sum_values": sv.tolist(),
            "hill_pred": hill_pred.tolist(),
            "hill_rmse": h_rmse,
            "hill_r2": h_r2,
            "hill_residuals": h_resid.tolist(),
        }
        if hyp_a is not None and hyp_b is not None:
            hyp_pred = hyperbolic_func(bp_arr, hyp_a, hyp_b)
            y_rmse, y_r2, y_resid = compute_metrics(sv, hyp_pred)
            entry.update({
                "hyp_pred": hyp_pred.tolist(),
                "hyp_rmse": y_rmse,
                "hyp_r2": y_r2,
                "hyp_residuals": y_resid.tolist(),
            })
        results.append(entry)
    return results

# ─── 主应用 ────────────────────────────────────────────────────────────────────

class SensorAnalyzerApp(ctk.CTk):
    """传感器 Hill 方程拟合分析工具 v1.5"""

    VERSION = "v1.5"

    def __init__(self):
        super().__init__()
        self.title(f"传感器 Hill 方程拟合分析工具  {self.VERSION}")
        self.outlier_removal_var   = tk.BooleanVar(value=False)
        self.outlier_threshold_var = tk.DoubleVar(value=15.0)
        self.geometry("1400x900")
        self.minsize(1100, 700)

        self.loaded_files: list[dict] = []
        self.analysis_result: Optional[dict] = None
        self.loaded_param_result: Optional[dict] = None  # 加载参数模式结果

        self._build_ui()

    # ── UI 构建 ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        header = ctk.CTkFrame(self, height=50, corner_radius=0, fg_color="#1a1a2e")
        header.pack(fill="x", side="top")
        ctk.CTkLabel(
            header,
            text=f"⚗  传感器 Hill 方程拟合分析工具  {self.VERSION}",
            font=("Microsoft YaHei", 14, "bold"),
            text_color="#4FC3F7",
        ).pack(side="left", padx=20, pady=10)
        ctk.CTkLabel(
            header,
            text="上传多次实验 CSV → 4传感器求和÷数量 → 单传感器均值曲线 → Hill 拟合 → 残差分析",
            font=("Microsoft YaHei", 11),
            text_color="#888",
        ).pack(side="left", padx=10)

        body = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        body.pack(fill="both", expand=True)

        self._build_left_panel(body)
        self._build_right_panel(body)

    def _build_left_panel(self, parent):
        left = ctk.CTkFrame(parent, width=340, corner_radius=0, fg_color="#16213e")
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        scroll = ctk.CTkScrollableFrame(left, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # ── 文件上传区 ──
        self._section(scroll, "📁  数据文件（每个文件 = 一次实验）")

        self.file_listbox = tk.Listbox(
            scroll, bg="#0f3460", fg="#ccc", selectbackground="#4FC3F7",
            selectforeground="#000", font=FONT_SMALL, height=7,
            relief="flat", bd=0, highlightthickness=1, highlightcolor="#4FC3F7",
        )
        self.file_listbox.pack(fill="x", pady=(0, 6))

        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0, 4))
        ctk.CTkButton(btn_row, text="+ 添加文件", command=self._add_files,
                      width=120, height=30).pack(side="left", padx=(0, 4))
        ctk.CTkButton(btn_row, text="清空", command=self._clear_files,
                      width=70, height=30, fg_color="#c0392b",
                      hover_color="#e74c3c").pack(side="left")

        ctk.CTkLabel(
            scroll,
            text="每个 CSV 文件对应一次实验，包含同一组传感器的多列数据。\n"
                 "工具将对每行的所有传感器列求和后除以传感器数量，得到单传感器均值曲线。",
            font=FONT_SMALL, text_color="#888", wraplength=300,
        ).pack(anchor="w", pady=(0, 8))

        # ── 分析配置 ──
        self._section(scroll, "⚙  分析配置")

        self._label(scroll, "压力范围 (N)")
        prow = ctk.CTkFrame(scroll, fg_color="transparent")
        prow.pack(fill="x", pady=(0, 6))
        self.p_min_var = tk.DoubleVar(value=0.0)
        self.p_max_var = tk.DoubleVar(value=200.0)
        ctk.CTkEntry(prow, textvariable=self.p_min_var, width=80,
                     placeholder_text="最小").pack(side="left", padx=(0, 4))
        ctk.CTkLabel(prow, text="~", font=FONT_LABEL).pack(side="left", padx=2)
        ctk.CTkEntry(prow, textvariable=self.p_max_var, width=80,
                     placeholder_text="最大").pack(side="left", padx=(4, 0))

        self._label(scroll, "单传感器均值最小阈值（过滤静息期）")
        self.val_threshold_var = tk.DoubleVar(value=0.0)
        ctk.CTkEntry(scroll, textvariable=self.val_threshold_var,
                     placeholder_text="默认 0，单传感器均值大于此值才参与分析").pack(fill="x", pady=(0, 6))

        self._label(scroll, "平滑窗口大小（均值曲线）")
        self.smooth_var = tk.IntVar(value=5)
        smooth_slider = ctk.CTkSlider(scroll, from_=1, to=15, number_of_steps=14,
                                       variable=self.smooth_var)
        smooth_slider.pack(fill="x", pady=(0, 2))
        self.smooth_label = ctk.CTkLabel(scroll, text="窗口: 5", font=FONT_SMALL)
        self.smooth_label.pack(anchor="w")
        smooth_slider.configure(command=lambda v: self.smooth_label.configure(
            text=f"窗口: {int(float(v))}"))

        # ── 离群点剪除 ──
        self._section(scroll, "🔍  离群点剪除")

        outlier_row = ctk.CTkFrame(scroll, fg_color="transparent")
        outlier_row.pack(fill="x", pady=(0, 4))
        self.outlier_chk = ctk.CTkCheckBox(
            outlier_row, text="启用离群点剪除",
            variable=self.outlier_removal_var,
            font=FONT_LABEL, command=self._on_outlier_toggle,
        )
        self.outlier_chk.pack(side="left")

        self._label(scroll, "剪除阈值：偏离均值 ± X%")
        outlier_thresh_row = ctk.CTkFrame(scroll, fg_color="transparent")
        outlier_thresh_row.pack(fill="x", pady=(0, 2))
        self.outlier_entry = ctk.CTkEntry(
            outlier_thresh_row, textvariable=self.outlier_threshold_var,
            width=80, placeholder_text="15",
        )
        self.outlier_entry.pack(side="left", padx=(0, 6))
        ctk.CTkLabel(outlier_thresh_row, text="%", font=FONT_LABEL).pack(side="left")
        self.outlier_entry.configure(state="disabled")

        self.outlier_info = ctk.CTkLabel(
            scroll, text="关闭状态：不剪除任何点",
            font=FONT_SMALL, text_color="#888", wraplength=280,
        )
        self.outlier_info.pack(anchor="w", pady=(0, 6))

        # ── Hill 初始值 ──
        self._label(scroll, "Hill 拟合初始值 (可选)")
        init_row = ctk.CTkFrame(scroll, fg_color="transparent")
        init_row.pack(fill="x", pady=(0, 6))
        self.init_a = ctk.CTkEntry(init_row, width=70, placeholder_text="a")
        self.init_b = ctk.CTkEntry(init_row, width=70, placeholder_text="b")
        self.init_n = ctk.CTkEntry(init_row, width=70, placeholder_text="n")
        self.init_a.pack(side="left", padx=(0, 3))
        self.init_b.pack(side="left", padx=(0, 3))
        self.init_n.pack(side="left")

        # ── 拟合分析按钮 ──
        ctk.CTkButton(
            scroll, text="▶  开始拟合分析", command=self._run_analysis,
            height=40, font=("Microsoft YaHei", 12, "bold"),
            fg_color="#1565C0", hover_color="#1976D2",
        ).pack(fill="x", pady=(10, 4))

        # ════════════════════════════════════════════
        # ── 加载拟合参数区域（v1.4 新增）──
        # ════════════════════════════════════════════
        self._section(scroll, "📥  加载拟合参数（不重新拟合）")

        ctk.CTkLabel(
            scroll,
            text="输入或从 JSON 导入已有参数，直接对当前数据计算残差，不重新拟合。",
            font=FONT_SMALL, text_color="#A5D6A7", wraplength=300,
        ).pack(anchor="w", pady=(0, 6))

        # Hill 参数输入
        self._label(scroll, "Hill 参数  y = a·xⁿ / (bⁿ + xⁿ)")
        hill_row = ctk.CTkFrame(scroll, fg_color="transparent")
        hill_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(hill_row, text="a=", font=FONT_SMALL, width=20).pack(side="left")
        self.load_hill_a = ctk.CTkEntry(hill_row, width=72, placeholder_text="e.g. 468.87")
        self.load_hill_a.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(hill_row, text="b=", font=FONT_SMALL, width=20).pack(side="left")
        self.load_hill_b = ctk.CTkEntry(hill_row, width=72, placeholder_text="e.g. 11.66")
        self.load_hill_b.pack(side="left", padx=(0, 4))

        hill_row2 = ctk.CTkFrame(scroll, fg_color="transparent")
        hill_row2.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(hill_row2, text="n=", font=FONT_SMALL, width=20).pack(side="left")
        self.load_hill_n = ctk.CTkEntry(hill_row2, width=72, placeholder_text="e.g. 0.758")
        self.load_hill_n.pack(side="left")

        # 双曲线参数输入（可选）
        self._label(scroll, "双曲线参数（可选）  y = a·x / (b + x)")
        hyp_row = ctk.CTkFrame(scroll, fg_color="transparent")
        hyp_row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(hyp_row, text="a=", font=FONT_SMALL, width=20).pack(side="left")
        self.load_hyp_a = ctk.CTkEntry(hyp_row, width=72, placeholder_text="可选")
        self.load_hyp_a.pack(side="left", padx=(0, 4))
        ctk.CTkLabel(hyp_row, text="b=", font=FONT_SMALL, width=20).pack(side="left")
        self.load_hyp_b = ctk.CTkEntry(hyp_row, width=72, placeholder_text="可选")
        self.load_hyp_b.pack(side="left")

        # 从 JSON 导入按钮
        ctk.CTkButton(
            scroll, text="📂  从 JSON 导入参数", command=self._import_params_from_json,
            height=30, font=FONT_LABEL,
            fg_color="#2e7d32", hover_color="#388e3c",
        ).pack(fill="x", pady=(0, 4))

        # 当前已加载参数显示
        self.loaded_param_label = ctk.CTkLabel(
            scroll, text="尚未加载参数", font=FONT_SMALL,
            text_color="#888", wraplength=300,
        )
        self.loaded_param_label.pack(anchor="w", pady=(0, 4))

        # 执行按钮
        ctk.CTkButton(
            scroll, text="▶  加载参数计算残差", command=self._run_loaded_params,
            height=40, font=("Microsoft YaHei", 12, "bold"),
            fg_color="#1b5e20", hover_color="#2e7d32",
        ).pack(fill="x", pady=(0, 8))

        # ── 拟合参数展示 ──
        self._section(scroll, "📐  拟合参数")
        self.param_text = tk.Text(
            scroll, height=16, bg="#0f3460", fg="#e0e0e0",
            font=FONT_MONO, relief="flat", bd=0,
            state="disabled", wrap="none",
        )
        self.param_text.pack(fill="x", pady=(0, 6))

        # ── 导出 ──
        self._section(scroll, "💾  导出结果")
        exp_row = ctk.CTkFrame(scroll, fg_color="transparent")
        exp_row.pack(fill="x")
        ctk.CTkButton(exp_row, text="导出 CSV", command=lambda: self._export("csv"),
                      width=110, height=30).pack(side="left", padx=(0, 4))
        ctk.CTkButton(exp_row, text="导出 JSON", command=lambda: self._export("json"),
                      width=110, height=30).pack(side="left")

        # ── 状态栏 ──
        self.status_var = tk.StringVar(value="就绪 — 请添加 CSV 文件")
        ctk.CTkLabel(scroll, textvariable=self.status_var,
                     font=FONT_SMALL, text_color="#aaa",
                     wraplength=300).pack(anchor="w", pady=(10, 0))

    def _build_right_panel(self, parent):
        right = ctk.CTkFrame(parent, corner_radius=0, fg_color="#0d1b2a")
        right.pack(side="left", fill="both", expand=True)

        self.tab_view = ctk.CTkTabview(right, fg_color="#0d1b2a",
                                        segmented_button_fg_color="#16213e",
                                        segmented_button_selected_color="#1565C0")
        self.tab_view.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_fit    = self.tab_view.add("拟合曲线")
        self.tab_resid  = self.tab_view.add("残差分析")
        self.tab_each   = self.tab_view.add("各次实验对比")
        self.tab_loaded = self.tab_view.add("加载参数残差")   # v1.4 新增

        self.fig_fit,    self.canvas_fit    = self._make_canvas(self.tab_fit)
        self.fig_resid,  self.canvas_resid  = self._make_canvas(self.tab_resid)
        self.fig_each,   self.canvas_each   = self._make_canvas(self.tab_each)
        self.fig_loaded, self.canvas_loaded = self._make_canvas(self.tab_loaded)

    def _make_canvas(self, parent):
        fig = Figure(figsize=(10, 7), dpi=96, facecolor="#0d1b2a")
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar_frame = tk.Frame(parent, bg="#0d1b2a")
        toolbar_frame.pack(fill="x")
        NavigationToolbar2Tk(canvas, toolbar_frame)
        return fig, canvas

    def _section(self, parent, title):
        ctk.CTkLabel(parent, text=title,
                     font=("Microsoft YaHei", 12, "bold"),
                     text_color="#4FC3F7").pack(anchor="w", pady=(12, 2))

    def _label(self, parent, text):
        ctk.CTkLabel(parent, text=text, font=FONT_SMALL,
                     text_color="#aaa").pack(anchor="w", pady=(4, 1))

    def _on_outlier_toggle(self):
        enabled = self.outlier_removal_var.get()
        if enabled:
            self.outlier_entry.configure(state="normal")
            thresh = self.outlier_threshold_var.get()
            self.outlier_info.configure(
                text=f"已开启：剪除分箱内偏离均值 ±{thresh:.0f}% 的数据点",
                text_color="#4FC3F7",
            )
        else:
            self.outlier_entry.configure(state="disabled")
            self.outlier_info.configure(
                text="关闭状态：不剪除任何点", text_color="#888",
            )

    # ── 文件操作 ───────────────────────────────────────────────────────────────

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="选择 CSV 文件（每个文件 = 一次实验）",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        )
        added = 0
        for path in paths:
            fname = Path(path).name
            if any(f["filename"] == fname for f in self.loaded_files):
                continue
            try:
                content = Path(path).read_text(encoding="utf-8", errors="replace")
                parsed = parse_csv(content, fname)
                self.loaded_files.append({
                    "filename": fname, "path": path,
                    "content": content, "parsed": parsed,
                })
                n_sensors = len(parsed["sensor_ids"])
                self.file_listbox.insert(
                    tk.END, f"{fname}  [{n_sensors} 传感器]")
                added += 1
            except Exception as e:
                messagebox.showerror("解析错误", f"文件 {fname} 解析失败：{e}")
        if added:
            self._set_status(f"已加载 {len(self.loaded_files)} 个文件")

    def _clear_files(self):
        self.loaded_files.clear()
        self.file_listbox.delete(0, tk.END)
        self.analysis_result = None
        self.loaded_param_result = None
        self._set_status("已清空文件列表")

    # ── 加载参数：从 JSON 导入 ─────────────────────────────────────────────────

    def _import_params_from_json(self):
        """从之前导出的 JSON 文件中读取 Hill/双曲线参数，填入输入框"""
        path = filedialog.askopenfilename(
            title="选择参数 JSON 文件",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 兼容多种 JSON 格式：
            # 格式1（本工具 v1.4 导出）：{"fit_mode": {"hill": {"a":..., "b":..., "n":...}, ...}}
            # 格式2（本工具 v1.4 加载参数模式导出）：{"loaded_param_mode": {"hill": {...}, ...}}
            # 格式3（简单嵌套）：{"hill": {"a":..., "b":..., "n":...}}
            # 格式4（扁平）：{"hill_a":..., "hill_b":..., "hill_n":...}
            hill_a = hill_b = hill_n = None
            hyp_a = hyp_b = None

            # 优先从 fit_mode 或 loaded_param_mode 中提取（本工具导出格式）
            source = data
            if "fit_mode" in data:
                source = data["fit_mode"]
            elif "loaded_param_mode" in data:
                source = data["loaded_param_mode"]

            if "hill" in source:
                h = source["hill"]
                hill_a = h.get("a")
                hill_b = h.get("b")
                hill_n = h.get("n")
            else:
                hill_a = source.get("hill_a")
                hill_b = source.get("hill_b")
                hill_n = source.get("hill_n")

            if "hyperbolic" in source:
                y = source["hyperbolic"]
                hyp_a = y.get("a")
                hyp_b = y.get("b")
            elif "hyp" in source:
                y = source["hyp"]
                hyp_a = y.get("a")
                hyp_b = y.get("b")
            else:
                hyp_a = source.get("hyp_a")
                hyp_b = source.get("hyp_b")

            if hill_a is None or hill_b is None or hill_n is None:
                # 给出详细错误提示，列出实际 JSON 结构
                top_keys = list(data.keys())
                messagebox.showerror(
                    "导入失败",
                    f"JSON 文件中未找到有效的 Hill 参数 (a, b, n)。\n"
                    f"文件顶层键名：{top_keys}\n\n"
                    f"支持的格式：\n"
                    f"  本工具导出格式：{{\"fit_mode\": {{\"hill\": {{\"a\":..., \"b\":..., \"n\":...}}}}}}\n"
                    f"  简单格式：{{\"hill\": {{\"a\":..., \"b\":..., \"n\":...}}}}"
                )
                return

            # 填入输入框
            self.load_hill_a.delete(0, tk.END)
            self.load_hill_a.insert(0, f"{hill_a:.6g}")
            self.load_hill_b.delete(0, tk.END)
            self.load_hill_b.insert(0, f"{hill_b:.6g}")
            self.load_hill_n.delete(0, tk.END)
            self.load_hill_n.insert(0, f"{hill_n:.6g}")

            if hyp_a is not None and hyp_b is not None:
                self.load_hyp_a.delete(0, tk.END)
                self.load_hyp_a.insert(0, f"{hyp_a:.6g}")
                self.load_hyp_b.delete(0, tk.END)
                self.load_hyp_b.insert(0, f"{hyp_b:.6g}")

            hyp_tag = f"  双曲线 a={hyp_a:.4g} b={hyp_b:.4g}" if hyp_a else ""
            self.loaded_param_label.configure(
                text=f"已导入：Hill a={hill_a:.4g} b={hill_b:.4g} n={hill_n:.4g}{hyp_tag}",
                text_color="#A5D6A7",
            )
            self._set_status(f"参数已从 {Path(path).name} 导入")

        except Exception as e:
            messagebox.showerror("导入失败", f"读取 JSON 失败：{e}")

    # ── 加载参数：直接计算残差 ─────────────────────────────────────────────────

    def _run_loaded_params(self):
        """使用已输入/导入的参数，对当前数据直接计算残差，不重新拟合"""
        if not self.loaded_files:
            messagebox.showwarning("提示", "请先添加 CSV 文件")
            return

        # 读取参数
        try:
            hill_a = float(self.load_hill_a.get())
            hill_b = float(self.load_hill_b.get())
            hill_n = float(self.load_hill_n.get())
        except (ValueError, TypeError):
            messagebox.showerror("参数错误", "请输入有效的 Hill 参数 a、b、n（均为数字）")
            return

        hyp_a = hyp_b = None
        try:
            ha = self.load_hyp_a.get().strip()
            hb = self.load_hyp_b.get().strip()
            if ha and hb:
                hyp_a = float(ha)
                hyp_b = float(hb)
        except (ValueError, TypeError):
            pass

        p_min      = self.p_min_var.get()
        p_max      = self.p_max_var.get()
        val_thresh = self.val_threshold_var.get()
        smooth_w   = self.smooth_var.get()

        self._set_status("正在用加载参数计算残差...")
        self.update_idletasks()

        # 构建求和曲线
        sum_curves, filenames = [], []
        for fi in self.loaded_files:
            sc = compute_sum_curve(fi["parsed"], p_min, p_max, val_thresh)
            if len(sc) < 4:
                continue
            sum_curves.append(sc)
            filenames.append(fi["filename"])

        if not sum_curves:
            messagebox.showerror("错误", "所有文件有效数据不足，请检查压力范围和阈值设置")
            self._set_status("计算失败")
            return

        # 回代残差（不拟合）
        back_projections = back_project_to_sum_curves(
            sum_curves, filenames,
            hill_a, hill_b, hill_n,
            hyp_a, hyp_b,
            p_min=p_min, p_max=p_max,
            smooth_window=smooth_w,
        )

        # 生成拟合曲线（用于图示）
        all_p = [pt["pressure"] for sc in sum_curves for pt in sc]
        p_curve = np.linspace(0, max(all_p) * 1.05, 300)
        hill_curve_y = hill_func(p_curve, hill_a, hill_b, hill_n)
        hyp_curve_y  = hyperbolic_func(p_curve, hyp_a, hyp_b) if hyp_a else None

        self.loaded_param_result = {
            "fileCount": len(sum_curves),
            "filenames": filenames,
            "hill_a": hill_a, "hill_b": hill_b, "hill_n": hill_n,
            "hyp_a": hyp_a, "hyp_b": hyp_b,
            "curveP": p_curve.tolist(),
            "hill_curveV": hill_curve_y.tolist(),
            "hyp_curveV": hyp_curve_y.tolist() if hyp_curve_y is not None else None,
            "backProjections": back_projections,
        }

        self._update_params_text()
        self._plot_loaded_params()
        self.tab_view.set("加载参数残差")

        hyp_tag = f"  双曲线 a={hyp_a:.3f} b={hyp_b:.3f}" if hyp_a else ""
        self._set_status(
            f"加载参数完成：Hill a={hill_a:.3f} b={hill_b:.3f} n={hill_n:.3f}{hyp_tag}，"
            f"共 {len(back_projections)} 次实验"
        )

    # ── 拟合分析流程 ───────────────────────────────────────────────────────────

    def _run_analysis(self):
        if not self.loaded_files:
            messagebox.showwarning("提示", "请先添加 CSV 文件")
            return

        p_min           = self.p_min_var.get()
        p_max           = self.p_max_var.get()
        val_thresh      = self.val_threshold_var.get()
        smooth_w        = self.smooth_var.get()
        remove_outliers = self.outlier_removal_var.get()
        outlier_thresh  = self.outlier_threshold_var.get()

        init_p0 = None
        try:
            ia  = float(self.init_a.get())
            ib  = float(self.init_b.get())
            in_ = float(self.init_n.get())
            init_p0 = [ia, ib, in_]
        except (ValueError, TypeError):
            pass

        self._set_status("正在分析...")
        self.update_idletasks()

        sum_curves, filenames, sensor_ids_per_file = [], [], []
        for fi in self.loaded_files:
            sc = compute_sum_curve(fi["parsed"], p_min, p_max, val_thresh)
            if len(sc) < 4:
                self._set_status(f"警告：{fi['filename']} 有效数据不足，已跳过")
                continue
            sum_curves.append(sc)
            filenames.append(fi["filename"])
            sensor_ids_per_file.append(fi["parsed"]["sensor_ids"])

        if not sum_curves:
            messagebox.showerror("错误", "所有文件有效数据不足，请检查压力范围和阈值设置")
            self._set_status("分析失败")
            return

        avg_p, avg_v = compute_average_from_sum_curves(
            sum_curves, p_min, p_max,
            smooth_window=smooth_w,
            remove_outliers=remove_outliers,
            outlier_threshold=outlier_thresh,
        )
        if len(avg_p) < 4:
            messagebox.showerror("错误", "均值曲线数据点不足，请调整参数")
            self._set_status("分析失败")
            return

        hill_a, hill_b, hill_n, hill_rmse, hill_r2, hill_resid = fit_hill(
            avg_p, avg_v, p0=init_p0)
        hyp_a, hyp_b, hyp_rmse, hyp_r2, hyp_resid = fit_hyperbolic(avg_p, avg_v)

        p_curve      = np.linspace(0, max(avg_p) * 1.05, 300)
        hill_curve_y = hill_func(p_curve, hill_a, hill_b, hill_n)
        hyp_curve_y  = hyperbolic_func(p_curve, hyp_a, hyp_b)

        back_projections = back_project_to_sum_curves(
            sum_curves, filenames,
            hill_a, hill_b, hill_n,
            hyp_a, hyp_b,
            p_min=p_min, p_max=p_max,
            smooth_window=smooth_w,
        )

        self.analysis_result = {
            "fileCount": len(sum_curves),
            "filenames": filenames,
            "sensor_ids_per_file": sensor_ids_per_file,
            "sumCurves": sum_curves,
            "avgP": avg_p,
            "avgV": avg_v,
            "hill": {
                "a": hill_a, "b": hill_b, "n": hill_n,
                "rmse": hill_rmse, "r2": hill_r2,
                "residuals": hill_resid,
                "curveP": p_curve.tolist(),
                "curveV": hill_curve_y.tolist(),
            },
            "hyp": {
                "a": hyp_a, "b": hyp_b,
                "rmse": hyp_rmse, "r2": hyp_r2,
                "residuals": hyp_resid,
                "curveP": p_curve.tolist(),
                "curveV": hyp_curve_y.tolist(),
            },
            "backProjections": back_projections,
        }

        self._update_params_text()
        self._plot_fit_curve()
        self._plot_residuals()
        self._plot_each_experiment()

        outlier_tag = f"（已剪除±{outlier_thresh:.0f}%离群点）" if remove_outliers else ""
        self._set_status(
            f"分析完成：{len(sum_curves)} 次实验{outlier_tag}，"
            f"Hill R²={hill_r2:.4f}，双曲线 R²={hyp_r2:.4f}"
        )

    # ── 参数文本 ───────────────────────────────────────────────────────────────

    def _update_params_text(self):
        lines = []

        if self.analysis_result:
            r   = self.analysis_result
            h   = r["hill"]
            y   = r["hyp"]
            bps = r["backProjections"]
            outlier_tag = ""
            if self.outlier_removal_var.get():
                thresh = self.outlier_threshold_var.get()
                outlier_tag = f"  [已剪除±{thresh:.0f}%]"
            lines += [
                "═══ 拟合模式 ═══",
                f"实验次数  : {r['fileCount']} 次",
                f"均值点数  : {len(r['avgP'])} 个{outlier_tag}",
                "─" * 34,
                "Hill  y = a·xⁿ / (bⁿ + xⁿ)",
                f"  a={h['a']:.4f}  b={h['b']:.4f}  n={h['n']:.4f}",
                f"  均值 RMSE={h['rmse']:.4f}  R²={h['r2']:.6f}",
                "─" * 34,
                "双曲线  y = a·x / (b + x)",
                f"  a={y['a']:.4f}  b={y['b']:.4f}",
                f"  均值 RMSE={y['rmse']:.4f}  R²={y['r2']:.6f}",
                "─" * 34,
                "各次实验回代（Hill）:",
            ]
            for bp in bps:
                lines.append(
                    f"  {Path(bp['filename']).stem[:18]}"
                    f"  RMSE={bp['hill_rmse']:.3f}  R²={bp['hill_r2']:.4f}"
                )

        if self.loaded_param_result:
            lp  = self.loaded_param_result
            bps = lp["backProjections"]
            lines += [
                "",
                "═══ 加载参数模式 ═══",
                f"Hill  a={lp['hill_a']:.4f}  b={lp['hill_b']:.4f}  n={lp['hill_n']:.4f}",
            ]
            if lp["hyp_a"]:
                lines.append(f"双曲线  a={lp['hyp_a']:.4f}  b={lp['hyp_b']:.4f}")
            lines.append("─" * 34)
            lines.append("各次实验回代（Hill）:")
            for bp in bps:
                lines.append(
                    f"  {Path(bp['filename']).stem[:18]}"
                    f"  RMSE={bp['hill_rmse']:.3f}  R²={bp['hill_r2']:.4f}"
                )

        if not lines:
            return

        self.param_text.configure(state="normal")
        self.param_text.delete("1.0", tk.END)
        self.param_text.insert(tk.END, "\n".join(lines))
        self.param_text.configure(state="disabled")

    # ── 图表绘制 ───────────────────────────────────────────────────────────────

    def _style_ax(self, ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor("#0f3460")
        ax.tick_params(colors="#aaa", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.xaxis.label.set_color("#aaa")
        ax.yaxis.label.set_color("#aaa")
        ax.title.set_color("#e0e0e0")
        if title:  ax.set_title(title, fontsize=11, pad=8)
        if xlabel: ax.set_xlabel(xlabel, fontsize=9)
        if ylabel: ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, color="#1a3a5c", linewidth=0.5, alpha=0.7)

    def _plot_fit_curve(self):
        self.fig_fit.clear()
        self.fig_fit.patch.set_facecolor("#0d1b2a")
        r = self.analysis_result
        ax = self.fig_fit.add_subplot(111)
        for i, bp in enumerate(r["backProjections"]):
            color = COLORS_EXP[i % len(COLORS_EXP)]
            ax.plot(bp["pressures"], bp["sum_values"],
                    color=color, linewidth=1.2, alpha=0.6,
                    label=f"实验{i+1}: {Path(bp['filename']).stem}")
        ax.plot(r["avgP"], r["avgV"],
                "o-", color=COLOR_AVG, markersize=3, linewidth=2, alpha=0.9,
                label="均值曲线", zorder=4)
        ax.plot(r["hill"]["curveP"], r["hill"]["curveV"],
                color=COLOR_HILL, linewidth=2.5,
                label=f"Hill 拟合  a={r['hill']['a']:.2f}  b={r['hill']['b']:.2f}"
                      f"  n={r['hill']['n']:.2f}  R²={r['hill']['r2']:.4f}",
                zorder=5)
        ax.plot(r["hyp"]["curveP"], r["hyp"]["curveV"],
                color=COLOR_HYP, linewidth=2, linestyle="--",
                label=f"双曲线拟合  a={r['hyp']['a']:.2f}  b={r['hyp']['b']:.2f}"
                      f"  R²={r['hyp']['r2']:.4f}",
                zorder=5)
        outlier_tag = ""
        if self.outlier_removal_var.get():
            outlier_tag = f"  [已剪除±{self.outlier_threshold_var.get():.0f}%离群点]"
        self._style_ax(ax,
            title=f"4传感器求和曲线 + 均值 + 拟合{outlier_tag}",
            xlabel="压力 (N)", ylabel="4传感器求和值")
        ax.legend(fontsize=8, facecolor="#16213e",
                  edgecolor="#333", labelcolor="#ddd", loc="upper left")
        self.fig_fit.tight_layout()
        self.canvas_fit.draw()

    def _plot_residuals(self):
        self.fig_resid.clear()
        self.fig_resid.patch.set_facecolor("#0d1b2a")
        r   = self.analysis_result
        bps = r["backProjections"]
        n   = len(bps)
        gs  = gridspec.GridSpec(3, 2, figure=self.fig_resid, hspace=0.55, wspace=0.35)

        ax_avg = self.fig_resid.add_subplot(gs[0, :])
        resid_h = np.array(r["hill"]["residuals"])
        resid_y = np.array(r["hyp"]["residuals"])
        x_idx   = np.arange(len(resid_h))
        ax_avg.axhline(0, color="#555", linewidth=1)
        ax_avg.fill_between(x_idx, resid_h, alpha=0.4, color=COLOR_HILL)
        ax_avg.plot(x_idx, resid_h, color=COLOR_HILL, linewidth=1,
                    label=f"Hill 残差  RMSE={r['hill']['rmse']:.3f}")
        ax_avg.fill_between(x_idx, resid_y, alpha=0.3, color=COLOR_HYP)
        ax_avg.plot(x_idx, resid_y, color=COLOR_HYP, linewidth=1, linestyle="--",
                    label=f"双曲线残差  RMSE={r['hyp']['rmse']:.3f}")
        self._style_ax(ax_avg, title="均值曲线残差（Hill vs 双曲线）",
                       xlabel="数据点索引", ylabel="残差")
        ax_avg.legend(fontsize=8, facecolor="#16213e",
                      edgecolor="#333", labelcolor="#ddd")

        ax_bp = self.fig_resid.add_subplot(gs[1, :])
        ax_bp.axhline(0, color="#555", linewidth=1)
        for i, bp in enumerate(bps):
            color = COLORS_EXP[i % len(COLORS_EXP)]
            ax_bp.scatter(bp["pressures"], bp["hill_residuals"],
                          s=10, alpha=0.7, color=color,
                          label=f"实验{i+1}  RMSE={bp['hill_rmse']:.3f}  R²={bp['hill_r2']:.4f}")
        self._style_ax(ax_bp,
            title="各次实验求和曲线 → Hill 拟合回代残差",
            xlabel="压力 (N)", ylabel="残差（求和值 − Hill预测）")
        ax_bp.legend(fontsize=8, facecolor="#16213e",
                     edgecolor="#333", labelcolor="#ddd", ncol=min(3, n))

        ax_h1 = self.fig_resid.add_subplot(gs[2, 0])
        ax_h2 = self.fig_resid.add_subplot(gs[2, 1])
        all_hill_resid = []
        for bp in bps:
            all_hill_resid.extend(bp["hill_residuals"])
        ax_h1.hist(all_hill_resid, bins=25, color=COLOR_HILL, alpha=0.75, edgecolor="#333")
        self._style_ax(ax_h1, title="Hill 回代残差分布（所有实验）",
                       xlabel="残差", ylabel="频次")
        all_hyp_resid = []
        for bp in bps:
            if "hyp_residuals" in bp:
                all_hyp_resid.extend(bp["hyp_residuals"])
        if all_hyp_resid:
            ax_h2.hist(all_hyp_resid, bins=25, color=COLOR_HYP, alpha=0.75, edgecolor="#333")
        self._style_ax(ax_h2, title="双曲线 回代残差分布（所有实验）",
                       xlabel="残差", ylabel="频次")
        self.fig_resid.tight_layout()
        self.canvas_resid.draw()

    def _plot_each_experiment(self):
        self.fig_each.clear()
        self.fig_each.patch.set_facecolor("#0d1b2a")
        r   = self.analysis_result
        bps = r["backProjections"]
        n   = len(bps)
        if n == 0:
            return
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        gs   = gridspec.GridSpec(rows, cols, figure=self.fig_each,
                                 hspace=0.55, wspace=0.35)
        for i, bp in enumerate(bps):
            row = i // cols
            col = i % cols
            ax  = self.fig_each.add_subplot(gs[row, col])
            color = COLORS_EXP[i % len(COLORS_EXP)]
            ax.plot(bp["pressures"], bp["sum_values"],
                    "o-", color=color, markersize=4, linewidth=1.5,
                    alpha=0.8, label="求和曲线")
            ax.plot(bp["pressures"], bp["hill_pred"],
                    color=COLOR_HILL, linewidth=2, label="Hill 预测")
            if "hyp_pred" in bp:
                ax.plot(bp["pressures"], bp["hyp_pred"],
                        color=COLOR_HYP, linewidth=1.5, linestyle="--",
                        label="双曲线预测")
            ax2 = ax.twinx()
            ax2.bar(bp["pressures"], bp["hill_residuals"],
                    width=(bp["pressures"][-1] - bp["pressures"][0]) / len(bp["pressures"]) * 0.8,
                    alpha=0.3, color=COLOR_RESID)
            ax2.axhline(0, color="#555", linewidth=0.8)
            ax2.tick_params(colors="#aaa", labelsize=7)
            ax2.set_ylabel("残差", fontsize=7, color=COLOR_RESID)
            self._style_ax(ax,
                title=f"实验 {i+1}: {Path(bp['filename']).stem}\n"
                      f"RMSE={bp['hill_rmse']:.3f}  R²={bp['hill_r2']:.4f}",
                xlabel="压力(N)", ylabel="求和值")
            ax.legend(fontsize=7, facecolor="#16213e",
                      edgecolor="#333", labelcolor="#ddd")
        self.fig_each.suptitle(
            f"各次实验求和曲线 vs 拟合曲线（共 {n} 次）",
            color="#e0e0e0", fontsize=13, y=1.01,
        )
        self.fig_each.tight_layout()
        self.canvas_each.draw()

    def _plot_loaded_params(self):
        """Tab4（v1.4）：加载参数模式的可视化"""
        self.fig_loaded.clear()
        self.fig_loaded.patch.set_facecolor("#0d1b2a")
        lp  = self.loaded_param_result
        bps = lp["backProjections"]
        n   = len(bps)
        if n == 0:
            return

        gs = gridspec.GridSpec(3, 2, figure=self.fig_loaded,
                               hspace=0.55, wspace=0.35)

        # ── 顶部：各次实验求和曲线 + 加载的拟合曲线 ──
        ax_top = self.fig_loaded.add_subplot(gs[0, :])
        for i, bp in enumerate(bps):
            color = COLORS_EXP[i % len(COLORS_EXP)]
            ax_top.plot(bp["pressures"], bp["sum_values"],
                        color=color, linewidth=1.2, alpha=0.65,
                        label=f"实验{i+1}: {Path(bp['filename']).stem}")

        # 加载的 Hill 曲线
        ax_top.plot(lp["curveP"], lp["hill_curveV"],
                    color=COLOR_LOAD, linewidth=2.5,
                    label=f"加载 Hill  a={lp['hill_a']:.2f}  b={lp['hill_b']:.2f}"
                          f"  n={lp['hill_n']:.2f}",
                    zorder=5)
        # 加载的双曲线（可选）
        if lp["hyp_curveV"] is not None:
            ax_top.plot(lp["curveP"], lp["hyp_curveV"],
                        color=COLOR_HYP, linewidth=2, linestyle="--",
                        label=f"加载双曲线  a={lp['hyp_a']:.2f}  b={lp['hyp_b']:.2f}",
                        zorder=5)
        self._style_ax(ax_top,
            title="各次实验求和曲线 + 加载的拟合曲线",
            xlabel="压力 (N)", ylabel="4传感器求和值")
        ax_top.legend(fontsize=8, facecolor="#16213e",
                      edgecolor="#333", labelcolor="#ddd", loc="upper left")

        # ── 中部：各次实验回代残差（按压力散点）──
        ax_mid = self.fig_loaded.add_subplot(gs[1, :])
        ax_mid.axhline(0, color="#555", linewidth=1)
        for i, bp in enumerate(bps):
            color = COLORS_EXP[i % len(COLORS_EXP)]
            ax_mid.scatter(bp["pressures"], bp["hill_residuals"],
                           s=12, alpha=0.75, color=color,
                           label=f"实验{i+1}  RMSE={bp['hill_rmse']:.3f}  R²={bp['hill_r2']:.4f}")
        self._style_ax(ax_mid,
            title="各次实验 → 加载 Hill 参数回代残差",
            xlabel="压力 (N)", ylabel="残差（求和值 − Hill预测）")
        ax_mid.legend(fontsize=8, facecolor="#16213e",
                      edgecolor="#333", labelcolor="#ddd", ncol=min(3, n))

        # ── 底部左：Hill 残差分布直方图 ──
        ax_h1 = self.fig_loaded.add_subplot(gs[2, 0])
        all_hill_resid = []
        for bp in bps:
            all_hill_resid.extend(bp["hill_residuals"])
        ax_h1.hist(all_hill_resid, bins=25, color=COLOR_LOAD, alpha=0.75, edgecolor="#333")
        self._style_ax(ax_h1, title="加载 Hill 回代残差分布",
                       xlabel="残差", ylabel="频次")

        # ── 底部右：各次实验 RMSE 柱状图 ──
        ax_h2 = self.fig_loaded.add_subplot(gs[2, 1])
        exp_names = [f"实验{i+1}" for i in range(n)]
        rmse_vals  = [bp["hill_rmse"] for bp in bps]
        r2_vals    = [bp["hill_r2"]   for bp in bps]
        bars = ax_h2.bar(exp_names, rmse_vals,
                         color=[COLORS_EXP[i % len(COLORS_EXP)] for i in range(n)],
                         alpha=0.8, edgecolor="#333")
        # 在柱上标注 R²
        for bar, r2 in zip(bars, r2_vals):
            ax_h2.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + max(rmse_vals) * 0.02,
                       f"R²={r2:.4f}", ha="center", va="bottom",
                       fontsize=7, color="#ddd")
        self._style_ax(ax_h2, title="各次实验 RMSE（加载 Hill 参数）",
                       xlabel="实验", ylabel="RMSE")

        self.fig_loaded.suptitle(
            f"加载参数模式残差分析  "
            f"Hill: a={lp['hill_a']:.3f}  b={lp['hill_b']:.3f}  n={lp['hill_n']:.3f}",
            color="#A5D6A7", fontsize=12, y=1.01,
        )
        self.fig_loaded.tight_layout()
        self.canvas_loaded.draw()

    # ── 导出 ───────────────────────────────────────────────────────────────────

    def _export(self, fmt: str):
        if not self.analysis_result and not self.loaded_param_result:
            messagebox.showwarning("提示", "请先运行分析或加载参数计算")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if fmt == "csv":
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile=f"hill_fit_sum_{ts}.csv",
                filetypes=[("CSV", "*.csv")],
            )
            if not path:
                return
            rows = []
            if self.analysis_result:
                r = self.analysis_result
                rows.append({
                    "模式": "拟合模式-均值曲线",
                    "文件名": f"（{r['fileCount']} 次实验均值）",
                    "hill_a": r["hill"]["a"], "hill_b": r["hill"]["b"],
                    "hill_n": r["hill"]["n"], "hill_rmse": r["hill"]["rmse"],
                    "hill_r2": r["hill"]["r2"], "hyp_a": r["hyp"]["a"],
                    "hyp_b": r["hyp"]["b"], "hyp_rmse": r["hyp"]["rmse"],
                    "hyp_r2": r["hyp"]["r2"],
                })
                for bp in r["backProjections"]:
                    rows.append({
                        "模式": "拟合模式-实验回代",
                        "文件名": bp["filename"],
                        "hill_a": r["hill"]["a"], "hill_b": r["hill"]["b"],
                        "hill_n": r["hill"]["n"], "hill_rmse": bp["hill_rmse"],
                        "hill_r2": bp["hill_r2"], "hyp_a": r["hyp"]["a"],
                        "hyp_b": r["hyp"]["b"],
                        "hyp_rmse": bp.get("hyp_rmse", ""),
                        "hyp_r2": bp.get("hyp_r2", ""),
                    })
            if self.loaded_param_result:
                lp = self.loaded_param_result
                for bp in lp["backProjections"]:
                    rows.append({
                        "模式": "加载参数模式",
                        "文件名": bp["filename"],
                        "hill_a": lp["hill_a"], "hill_b": lp["hill_b"],
                        "hill_n": lp["hill_n"], "hill_rmse": bp["hill_rmse"],
                        "hill_r2": bp["hill_r2"],
                        "hyp_a": lp["hyp_a"] or "",
                        "hyp_b": lp["hyp_b"] or "",
                        "hyp_rmse": bp.get("hyp_rmse", ""),
                        "hyp_r2": bp.get("hyp_r2", ""),
                    })
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("导出成功", f"CSV 已保存至：\n{path}")

        elif fmt == "json":
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile=f"hill_fit_sum_{ts}.json",
                filetypes=[("JSON", "*.json")],
            )
            if not path:
                return
            export = {}
            if self.analysis_result:
                r = self.analysis_result
                export["fit_mode"] = {
                    "fileCount": r["fileCount"],
                    "filenames": r["filenames"],
                    "hill": {k: v for k, v in r["hill"].items()
                             if k not in ("curveP", "curveV", "residuals")},
                    "hyperbolic": {k: v for k, v in r["hyp"].items()
                                   if k not in ("curveP", "curveV", "residuals")},
                    "backProjections": [
                        {k: v for k, v in bp.items()
                         if k not in ("pressures", "sum_values", "hill_pred",
                                      "hyp_pred", "hill_residuals", "hyp_residuals")}
                        for bp in r["backProjections"]
                    ],
                }
            if self.loaded_param_result:
                lp = self.loaded_param_result
                export["loaded_param_mode"] = {
                    "fileCount": lp["fileCount"],
                    "filenames": lp["filenames"],
                    "hill": {"a": lp["hill_a"], "b": lp["hill_b"], "n": lp["hill_n"]},
                    "hyperbolic": {"a": lp["hyp_a"], "b": lp["hyp_b"]}
                                  if lp["hyp_a"] else None,
                    "backProjections": [
                        {k: v for k, v in bp.items()
                         if k not in ("pressures", "sum_values", "hill_pred",
                                      "hyp_pred", "hill_residuals", "hyp_residuals")}
                        for bp in lp["backProjections"]
                    ],
                }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(export, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("导出成功", f"JSON 已保存至：\n{path}")

    # ── 工具 ───────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()


# ─── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = SensorAnalyzerApp()
    app.mainloop()
