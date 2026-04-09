"""
hill_core.parser — CSV 数据解析
================================
将传感器实验 CSV 文件解析为标准化的结构化数据。
仅依赖 Python 标准库（re），无第三方依赖。

使用示例:
    from hill_core.parser import parse_csv, ParsedCSV

    with open("experiment_1.csv", "r") as f:
        parsed = parse_csv(f.read(), filename="experiment_1.csv")

    print(parsed.sensor_ids)   # ["Sensor#1", "Sensor#2", ...]
    print(len(parsed.rows))    # 数据行数
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# ─── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class DataRow:
    """单行数据"""
    pressure: float            # 压力值 (N)
    sensor_values: List[float] # 各传感器的 ADC 值

    def to_dict(self) -> dict:
        return {"pressure": self.pressure, "sensor_values": self.sensor_values}


@dataclass
class ParsedCSV:
    """CSV 解析结果"""
    filename: str                       # 文件名
    sensor_ids: List[str]               # 传感器列名
    rows: List[DataRow] = field(default_factory=list)  # 数据行

    @property
    def sensor_count(self) -> int:
        """传感器数量"""
        return len(self.sensor_ids)

    @property
    def row_count(self) -> int:
        """数据行数"""
        return len(self.rows)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "sensor_ids": self.sensor_ids,
            "rows": [r.to_dict() for r in self.rows],
        }

    def to_legacy_dict(self) -> dict:
        """转为与原始 parse_csv 兼容的字典格式"""
        return {
            "filename": self.filename,
            "sensor_ids": self.sensor_ids,
            "rows": [{"pressure": r.pressure, "sensor_values": r.sensor_values}
                     for r in self.rows],
        }


# ─── 列识别规则 ──────────────────────────────────────────────────────────────

# 压力列关键词（不区分大小写）
PRESSURE_KEYWORDS = {"pressure", "压力", "force", "load"}

# 传感器列正则（不区分大小写）
SENSOR_PATTERN = re.compile(r"sensor|传感器|#\d+|ch\d+|s\d+", re.IGNORECASE)


def _is_pressure_col(header: str) -> bool:
    """判断列名是否为压力列"""
    h = header.lower().strip()
    for kw in PRESSURE_KEYWORDS:
        if kw in h:
            return True
    return False


def _is_sensor_col(header: str) -> bool:
    """判断列名是否为传感器列"""
    return bool(SENSOR_PATTERN.search(header))


# ─── 主解析函数 ──────────────────────────────────────────────────────────────

def parse_csv(content: str, filename: str = "") -> ParsedCSV:
    """
    解析传感器实验 CSV 文件。

    自动识别压力列和传感器列，支持多种列名格式。

    参数:
        content:  CSV 文件的完整文本内容
        filename: 文件名（用于错误提示和标识）

    返回:
        ParsedCSV 结构化数据

    列识别规则:
        压力列: 包含 "pressure", "压力", "force", "load" 的列
        传感器列: 包含 "sensor", "传感器", "#数字", "ch数字", "s数字" 的列
        若未匹配到传感器列，则第3列及之后的所有列视为传感器列
        若未匹配到压力列，默认使用第2列

    异常:
        ValueError: 文件为空
    """
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"文件 {filename} 为空")

    # ── 定位表头行 ──
    header_idx = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if any(kw in low for kw in ("pressure", "传感器", "sensor", "压力")):
            header_idx = i
            break

    # ── 解析表头 ──
    header = [
        h.strip().strip('"').strip("'").lstrip('\ufeff')
        for h in lines[header_idx].split(",")
    ]
    data_lines = lines[header_idx + 1:]

    # ── 识别列 ──
    pressure_col = None
    sensor_cols: List[tuple] = []  # [(col_index, col_name), ...]

    for i, h in enumerate(header):
        if _is_pressure_col(h):
            pressure_col = i
        elif _is_sensor_col(h):
            sensor_cols.append((i, h))

    # 兜底策略
    if pressure_col is None:
        pressure_col = 1 if len(header) > 1 else 0

    if not sensor_cols:
        start = 2 if len(header) > 2 else (1 if pressure_col == 0 else 0)
        for i in range(start, len(header)):
            if i != pressure_col:
                sensor_cols.append((i, header[i]))

    sensor_ids = [name for _, name in sensor_cols]

    # ── 解析数据行 ──
    rows: List[DataRow] = []
    max_col = max(pressure_col, max((i for i, _ in sensor_cols), default=0))

    for line in data_lines:
        parts = line.split(",")
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

        rows.append(DataRow(pressure=pressure, sensor_values=vals))

    return ParsedCSV(filename=filename, sensor_ids=sensor_ids, rows=rows)
