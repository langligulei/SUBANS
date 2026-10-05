# -*- coding: utf-8 -*-
"""
绘制SUBOFF六个速度工况下的表面压力系数和误差分布。

读取文件：
    results/predictions/*_main_prediction.csv

每个工况绘制四类结果：
    1. cp_static_truth
    2. cp_static_prediction
    3. cp_error = prediction - truth
    4. abs(cp_error)

输出：
    results/figures/pressure_distribution_all_cases.png
    results/figures/pressure_distribution_all_cases.pdf
    results/figures/<case_id>_pressure_error_distribution.png
    results/figures/<case_id>_pressure_error_distribution.pdf
    results/tables/pressure_error_summary.csv

注意：
    当前程序使用面心坐标进行三维散点绘图，颜色表示Cp或误差。
    由于预测CSV中没有三角面连接关系，因此不能直接绘制连续三角面云图。
"""

from __future__ import annotations

from pathlib import Path
import importlib
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

# =============================================================================
# 1. 导入实验配置
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

cfg = importlib.import_module("00_experiment_config")

# =============================================================================
# 2. 全局绘图参数
# =============================================================================

CASE_ORDER = [
    "suboff_0000_v0p75",
    "suboff_0000_v2p5",
    "suboff_0000_v4p5",
    "suboff_0000_v6p5",
    "suboff_0000_v8p5",
    "suboff_0000_v10p5",
]

SPEED_MAP = {
    "suboff_0000_v0p75": 0.75,
    "suboff_0000_v2p5": 2.5,
    "suboff_0000_v4p5": 4.5,
    "suboff_0000_v6p5": 6.5,
    "suboff_0000_v8p5": 8.5,
    "suboff_0000_v10p5": 10.5,
}

# 三维散点大小。面数为121772时，不宜设置过大。
POINT_SIZE = 0.35

# 统一图片分辨率
DPI = int(getattr(cfg, "FIGURE_DPI", 300))

# =============================================================================
# 3. 辅助函数
# =============================================================================

def find_prediction_file(case_id: str) -> Path:
    """根据工况ID查找03程序生成的主模型预测文件。"""
    prediction_root = Path(cfg.PREDICTION_ROOT)

    exact_name = (
        prediction_root
        / f"{case_id}_{cfg.PART_NAME}_"
        f"{cfg.MAIN_SENSOR_METHOD}_main_prediction.csv"
    )

    if exact_name.exists():
        return exact_name

    candidates = sorted(
        prediction_root.glob(
            f"{case_id}_*_main_prediction.csv"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"未找到工况 {case_id} 的预测文件。\n"
            f"预期文件：{exact_name}"
        )

    raise RuntimeError(
        f"工况 {case_id} 找到多个预测文件：\n"
        + "\n".join(str(path) for path in candidates)
    )

def load_prediction(case_id: str) -> pd.DataFrame:
    """读取并检查单个预测文件。"""
    path = find_prediction_file(case_id)

    table = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required_columns = {
        "global_face_id",
        "x_m",
        "y_m",
        "z_m",
        "cp_static_truth",
        "cp_static_prediction",
        "cp_error",
    }

    missing = required_columns - set(table.columns)
    if missing:
        raise KeyError(
            f"文件 {path.name} 缺少字段：{sorted(missing)}"
        )

    numeric_columns = [
        "x_m",
        "y_m",
        "z_m",
        "cp_static_truth",
        "cp_static_prediction",
        "cp_error",
    ]

    for column in numeric_columns:
        table[column] = pd.to_numeric(
            table[column],
            errors="coerce",
        )

    if table[numeric_columns].isna().any().any():
        raise ValueError(
            f"文件 {path.name} 包含NaN或无法转换的数值。"
        )

    # 重新计算误差，防止CSV中的误差列与真值、预测值不一致。
    table["cp_error"] = (
        table["cp_static_prediction"]
        - table["cp_static_truth"]
    )

    table["abs_cp_error"] = np.abs(table["cp_error"])
    table["case_id"] = case_id
    table["speed_m_s"] = SPEED_MAP[case_id]
    table["source_file"] = str(path)

    return table

def set_equal_3d_axes(ax, xyz: np.ndarray) -> None:
    """设置等比例三维坐标轴，避免艇体形状发生视觉变形。"""
    xyz_min = np.min(xyz, axis=0)
    xyz_max = np.max(xyz, axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    radius = 0.5 * np.max(xyz_max - xyz_min)

    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    # 兼容不同版本Matplotlib。
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass

def configure_3d_axis(ax, title: str, xyz: np.ndarray) -> None:
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.set_zlabel("z (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=18.0, azim=-120.0)
    set_equal_3d_axes(ax, xyz)

def add_colorbar(fig, ax, values, cmap, norm, label):
    """为指定三维坐标轴添加颜色条。"""
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array(values)
    colorbar = fig.colorbar(
        mapper,
        ax=ax,
        shrink=0.72,
        pad=0.02,
        aspect=24,
    )
    colorbar.set_label(label, fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    return colorbar

def plot_scatter(ax, table, column, cmap, norm, title, colorbar_label):
    xyz = table[["x_m", "y_m", "z_m"]].to_numpy()
    values = table[column].to_numpy(dtype=float)

    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=values,
        cmap=cmap,
        norm=norm,
        s=POINT_SIZE,
        linewidths=0.0,
        alpha=0.95,
        rasterized=True,
    )

    configure_3d_axis(ax, title, xyz)
    add_colorbar(
        ax.figure,
        ax,
        values,
        cmap,
        norm,
        colorbar_label,
    )

def save_single_case_figure(table: pd.DataFrame, output_root: Path) -> None:
    """保存单个速度工况的四联图。"""
    case_id = str(table["case_id"].iloc[0])
    speed = float(table["speed_m_s"].iloc[0])
    xyz = table[["x_m", "y_m", "z_m"]].to_numpy()

    cp_values = np.concatenate([
        table["cp_static_truth"].to_numpy(),
        table["cp_static_prediction"].to_numpy(),
    ])

    cp_min = float(np.nanpercentile(cp_values, 1.0))
    cp_max = float(np.nanpercentile(cp_values, 99.0))

    # 保证色标范围有效，并使用对称范围便于不同工况比较。
    cp_abs = max(abs(cp_min), abs(cp_max), 1.0e-12)
    cp_norm = Normalize(vmin=-cp_abs, vmax=cp_abs)

    error_values = table["cp_error"].to_numpy()
    abs_error_values = table["abs_cp_error"].to_numpy()

    error_abs = max(
        float(np.nanpercentile(np.abs(error_values), 99.0)),
        1.0e-12,
    )

    error_norm = TwoSlopeNorm(
        vmin=-error_abs,
        vcenter=0.0,
        vmax=error_abs,
    )

    abs_error_max = max(
        float(np.nanpercentile(abs_error_values, 99.0)),
        1.0e-12,
    )
    abs_error_norm = Normalize(
        vmin=0.0,
        vmax=abs_error_max,
    )

    fig = plt.figure(figsize=(18, 13))

    plot_scatter(
        fig.add_subplot(2, 2, 1, projection="3d"),
        table,
        "cp_static_truth",
        "turbo",
        cp_norm,
        f"Truth pressure, U = {speed:.2f} m/s",
        "Cp",
    )

    plot_scatter(
        fig.add_subplot(2, 2, 2, projection="3d"),
        table,
        "cp_static_prediction",
        "turbo",
        cp_norm,
        f"Sparse POD reconstruction, U = {speed:.2f} m/s",
        "Cp",
    )

    plot_scatter(
        fig.add_subplot(2, 2, 3, projection="3d"),
        table,
        "cp_error",
        "RdBu_r",
        error_norm,
        f"Signed error: prediction - truth, U = {speed:.2f} m/s",
        "Cp error",
    )

    plot_scatter(
        fig.add_subplot(2, 2, 4, projection="3d"),
        table,
        "abs_cp_error",
        "magma",
        abs_error_norm,
        f"Absolute error, U = {speed:.2f} m/s",
        "|Cp error|",
    )

    fig.suptitle(
        f"SUBOFF surface pressure and error distribution\n"
        f"{case_id} | part = {cfg.PART_NAME}",
        fontsize=15,
        y=0.98,
    )

    fig.subplots_adjust(
        left=0.01,
        right=0.97,
        bottom=0.02,
        top=0.92,
        wspace=0.02,
        hspace=0.04,
    )

    png_path = output_root / (
        f"{case_id}_pressure_error_distribution.png"
    )
    pdf_path = output_root / (
        f"{case_id}_pressure_error_distribution.pdf"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close(fig)

def save_summary(tables, table_root: Path) -> None:
    """保存六个工况的统计指标。"""
    rows = []

    for case_id, table in tables.items():
        truth = table["cp_static_truth"].to_numpy()
        prediction = table["cp_static_prediction"].to_numpy()
        error = table["cp_error"].to_numpy()

        truth_variation = np.sum((truth - np.mean(truth)) ** 2)
        r2 = (
            1.0 - np.sum(error ** 2) / truth_variation
            if truth_variation > np.finfo(float).eps
            else np.nan
        )

        rows.append({
            "case_id": case_id,
            "speed_m_s": SPEED_MAP[case_id],
            "face_count": len(table),
            "cp_truth_min": np.min(truth),
            "cp_truth_max": np.max(truth),
            "cp_prediction_min": np.min(prediction),
            "cp_prediction_max": np.max(prediction),
            "cp_error_mean": np.mean(error),
            "cp_error_rmse": np.sqrt(np.mean(error ** 2)),
            "cp_error_mae": np.mean(np.abs(error)),
            "cp_error_max_abs": np.max(np.abs(error)),
            "cp_error_p95": np.percentile(np.abs(error), 95.0),
            "cp_error_p99": np.percentile(np.abs(error), 99.0),
            "cp_r2": r2,
        })

    pd.DataFrame(rows).sort_values("speed_m_s").to_csv(
        table_root / "pressure_error_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10e",
    )

def save_combined_figure(tables, output_root: Path) -> None:
    """保存六个速度工况的总览图。"""
    all_cp = np.concatenate([
        np.concatenate([
            table["cp_static_truth"].to_numpy(),
            table["cp_static_prediction"].to_numpy(),
        ])
        for table in tables.values()
    ])

    all_error = np.concatenate([
        table["cp_error"].to_numpy()
        for table in tables.values()
    ])

    cp_abs = max(
        abs(float(np.nanpercentile(all_cp, 1.0))),
        abs(float(np.nanpercentile(all_cp, 99.0))),
        1.0e-12,
    )
    cp_norm = Normalize(vmin=-cp_abs, vmax=cp_abs)

    error_abs = max(
        float(np.nanpercentile(np.abs(all_error), 99.0)),
        1.0e-12,
    )
    error_norm = TwoSlopeNorm(
        vmin=-error_abs,
        vcenter=0.0,
        vmax=error_abs,
    )

    # 总览图：每个速度一行，三列分别为真值、预测、误差。
    fig = plt.figure(figsize=(18, 4.8 * len(tables)))

    for row_index, (case_id, table) in enumerate(tables.items()):
        speed = SPEED_MAP[case_id]

        plot_scatter(
            fig.add_subplot(len(tables), 3, row_index * 3 + 1, projection="3d"),
            table,
            "cp_static_truth",
            "turbo",
            cp_norm,
            f"Truth, U = {speed:.2f} m/s",
            "Cp",
        )

        plot_scatter(
            fig.add_subplot(len(tables), 3, row_index * 3 + 2, projection="3d"),
            table,
            "cp_static_prediction",
            "turbo",
            cp_norm,
            f"Prediction, U = {speed:.2f} m/s",
            "Cp",
        )

        plot_scatter(
            fig.add_subplot(len(tables), 3, row_index * 3 + 3, projection="3d"),
            table,
            "cp_error",
            "RdBu_r",
            error_norm,
            f"Signed error, U = {speed:.2f} m/s",
            "Cp error",
        )

    fig.suptitle(
        f"SUBOFF pressure reconstruction over six speeds\n"
        f"part = {cfg.PART_NAME}",
        fontsize=16,
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.01,
        right=0.97,
        bottom=0.01,
        top=0.985,
        wspace=0.01,
        hspace=0.03,
    )

    fig.savefig(
        output_root / "pressure_distribution_all_cases.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        output_root / "pressure_distribution_all_cases.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

def main():
    figure_root = Path(cfg.FIGURE_ROOT)
    table_root = Path(cfg.TABLE_ROOT)

    figure_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)

    tables = {}

    print("=" * 80)
    print("SUBOFF压力分布与误差分布绘图")
    print("=" * 80)
    print(f"目标部件：{cfg.PART_NAME}")
    print(f"预测目录：{cfg.PREDICTION_ROOT}")
    print(f"图片目录：{figure_root}")

    for case_id in CASE_ORDER:
        table = load_prediction(case_id)
        tables[case_id] = table

        print(
            f"读取：{case_id}, "
            f"U={SPEED_MAP[case_id]:.2f} m/s, "
            f"面数={len(table):,}, "
            f"RMSE={np.sqrt(np.mean(table['cp_error'] ** 2)):.6e}"
        )

    save_summary(tables, table_root)

    for table in tables.values():
        save_single_case_figure(table, figure_root)

    save_combined_figure(tables, figure_root)

    print("\n" + "=" * 80)
    print("压力分布与误差分布图片生成完成")
    print(f"图片目录：{figure_root}")
    print(f"统计表：{table_root / 'pressure_error_summary.csv'}")
    print("=" * 80)

if __name__ == "__main__":
    main()