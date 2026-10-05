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
def load_case_order():
    """按照工况分组和速度模长排序读取未见工况清单。"""
    unseen = pd.read_csv(
        cfg.UNSEEN_CASES_CSV_PATH,
        encoding="utf-8-sig",
    )

    # 【修改】容错处理：补充group_name
    if "group_name" not in unseen.columns:
        unseen["group_name"] = unseen[
            "split_type"
        ].map({
            "interpolation": "Group_A",
            "extrapolation": "Group_C",
        }).fillna("Group_A")

    # 计算速度模长用于排序
    if "speed_inf_m_s" not in unseen.columns:
        unseen["speed_inf_m_s"] = np.sqrt(
            unseen["u_inf_m_s"] ** 2
            + unseen["v_inf_m_s"] ** 2
            + unseen["w_inf_m_s"] ** 2
        )

    return unseen.sort_values(
        ["group_name", "speed_inf_m_s"]
    ).reset_index(drop=True)

def format_velocity_label(row):
    """生成速度标签，格式：u=−0.5, v=0.5, w=−0.5 m/s"""
    try:
        u = float(row["u_inf_m_s"])
        v = float(row["v_inf_m_s"])
        w = float(row["w_inf_m_s"])
        return (
            f"u={u:+.2f}, "
            f"v={v:+.2f}, "
            f"w={w:+.2f} m/s"
        )
    except (KeyError, TypeError):
        speed = float(row.get("speed_m_s", 0))
        return f"|U|={speed:.3f} m/s"


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

def load_prediction(case_id):
    """
    读取单个工况的预测结果。

    预测CSV保存面级数据；
    case_id、速度、误差指标等实验元数据
    从results/tables/all_metrics.csv中读取。
    """
    prediction_path = (
        cfg.PREDICTION_ROOT
        / (
            f"{case_id}_"
            f"{cfg.PART_NAME}_"
            f"{cfg.MAIN_SENSOR_METHOD}_"
            "main_prediction.csv"
        )
    )

    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"预测文件不存在：{prediction_path}"
        )

    table = pd.read_csv(
        prediction_path,
        encoding="utf-8-sig",
    )

    required_prediction_columns = {
        "global_face_id",
        "x_m",
        "y_m",
        "z_m",
        "face_area_m2",
        "cp_static_truth",
        "cp_static_prediction",
        "cp_error",
    }

    missing_prediction = (
        required_prediction_columns
        - set(table.columns)
    )

    if missing_prediction:
        raise KeyError(
            f"预测文件{prediction_path.name}缺少面级字段："
            f"{sorted(missing_prediction)}"
        )

    # ------------------------------------------------------------
    # 从面级字段计算绘图所需的基本误差量
    # ------------------------------------------------------------
    table["cp_truth"] = pd.to_numeric(
        table["cp_static_truth"],
        errors="raise",
    )

    table["cp_prediction"] = pd.to_numeric(
        table["cp_static_prediction"],
        errors="raise",
    )

    table["cp_error"] = (
        table["cp_prediction"]
        - table["cp_truth"]
    )

    # ------------------------------------------------------------
    # 从all_metrics.csv读取工况元数据和整体误差指标
    # ------------------------------------------------------------
    metrics_path = (
        cfg.TABLE_ROOT
        / "all_metrics.csv"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"实验指标文件不存在：{metrics_path}"
        )

    metrics = pd.read_csv(
        metrics_path,
        encoding="utf-8-sig",
    )

    required_metrics_columns = {
        "case_id",
        "u_inf_m_s",
        "v_inf_m_s",
        "w_inf_m_s",
        "speed_m_s",
        "method",
        "sensor_count",
        "mode_count",
        "cp_rmse",
        "cp_r2",
    }

    missing_metrics = (
        required_metrics_columns
        - set(metrics.columns)
    )

    if missing_metrics:
        raise KeyError(
            f"all_metrics.csv缺少字段："
            f"{sorted(missing_metrics)}"
        )

    metrics["case_id"] = (
        metrics["case_id"]
        .astype(str)
        .str.strip()
    )

    metrics["method"] = (
        metrics["method"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    target_case_id = str(case_id).strip()
    target_method = str(
        cfg.MAIN_SENSOR_METHOD
    ).strip().lower()

    # 优先读取main_model中的主模型记录
    metadata_rows = metrics.loc[
        (metrics["case_id"] == target_case_id)
        & (metrics["method"] == target_method)
        & (
            metrics["experiment"]
            == "main_model"
        )
    ].copy()

    # 如果旧结果中没有experiment字段或没有main_model记录，
    # 则退化为按照case_id和method筛选。
    if len(metadata_rows) == 0:
        metadata_rows = metrics.loc[
            (metrics["case_id"] == target_case_id)
            & (metrics["method"] == target_method)
        ].copy()

    if len(metadata_rows) == 0:
        raise KeyError(
            f"all_metrics.csv中没有找到工况"
            f"{target_case_id}、方法{target_method}的元数据"
        )

    # 理论上main_model每个工况和方法只有一行。
    # 如果存在多行，优先使用最后一行。
    metadata = metadata_rows.iloc[-1]

    metadata_columns = [
        "case_id",
        "u_inf_m_s",
        "v_inf_m_s",
        "w_inf_m_s",
        "speed_m_s",
        "method",
        "sensor_count",
        "mode_count",
        "cp_rmse",
        "cp_r2",
    ]

    for column in metadata_columns:
        table[column] = metadata[column]

    # 保证case_id使用当前文件名对应的case_id
    table["case_id"] = target_case_id

    # 重新计算一遍面级误差，避免CSV中的旧误差列不一致
    table["cp_error"] = (
            table["cp_static_prediction"]
            - table["cp_static_truth"]
    )

    table["abs_cp_error"] = np.abs(
        table["cp_error"]
    )

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
    velocity_text = format_velocity_label(table.iloc[0])
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
        f"Truth pressure, {velocity_text}",
        "Cp",
    )

    plot_scatter(
        fig.add_subplot(2, 2, 2, projection="3d"),
        table,
        "cp_static_prediction",
        "turbo",
        cp_norm,
        f"Sparse POD reconstruction, {velocity_text}",
        "Cp",
    )

    plot_scatter(
        fig.add_subplot(2, 2, 3, projection="3d"),
        table,
        "cp_error",
        "RdBu_r",
        error_norm,
        f"Signed error: prediction - truth, {velocity_text}",
        "Cp error",
    )

    plot_scatter(
        fig.add_subplot(2, 2, 4, projection="3d"),
        table,
        "abs_cp_error",
        "magma",
        abs_error_norm,
        f"Absolute error, {velocity_text}",
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
            "u_inf_m_s": float(table["u_inf_m_s"].iloc[0]),
            "v_inf_m_s": float(table["v_inf_m_s"].iloc[0]),
            "w_inf_m_s": float(table["w_inf_m_s"].iloc[0]),
            "speed_m_s": float(table["speed_m_s"].iloc[0]),
            "group_name": str(table["group_name"].iloc[0])
            if "group_name" in table.columns
            else "",
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
            "cp_relative_l2_variance_normalized": float(
                table[
                    "cp_relative_l2_variance_normalized"
                ].iloc[0]
            ) if "cp_relative_l2_variance_normalized" in table.columns else float("nan"),
        })

    pd.DataFrame(rows).sort_values("speed_m_s").to_csv(
        table_root / "pressure_error_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10e",
    )

def save_combined_figure(tables, output_root: Path) -> None:
    """将多个工况的总览图分页保存，避免图像高度超过Matplotlib限制。"""
    if not tables:
        raise ValueError("没有可用于绘图的工况数据")

    # 每页最多显示6个工况。
    # 当前52个工况将生成9页总览图。
    cases_per_page = 6

    case_items = list(tables.items())
    total_pages = int(
        np.ceil(len(case_items) / cases_per_page)
    )

    # 统一所有工况的Cp色标。
    all_cp = np.concatenate([
        np.concatenate([
            table["cp_static_truth"].to_numpy(dtype=float),
            table["cp_static_prediction"].to_numpy(dtype=float),
        ])
        for _, table in case_items
    ])

    all_error = np.concatenate([
        table["cp_error"].to_numpy(dtype=float)
        for _, table in case_items
    ])

    cp_abs = max(
        abs(float(np.nanpercentile(all_cp, 1.0))),
        abs(float(np.nanpercentile(all_cp, 99.0))),
        1.0e-12,
    )
    cp_norm = Normalize(
        vmin=-cp_abs,
        vmax=cp_abs,
    )

    error_abs = max(
        float(np.nanpercentile(np.abs(all_error), 99.0)),
        1.0e-12,
    )
    error_norm = TwoSlopeNorm(
        vmin=-error_abs,
        vcenter=0.0,
        vmax=error_abs,
    )

    for page_index in range(total_pages):
        start = page_index * cases_per_page
        end = min(
            start + cases_per_page,
            len(case_items),
        )

        page_items = case_items[start:end]
        page_rows = len(page_items)

        # 每页最多6行，避免输出图片高度过大。
        fig = plt.figure(
            figsize=(18, 4.8 * page_rows)
        )

        for row_index, (case_id, table) in enumerate(
            page_items
        ):
            velocity_text = format_velocity_label(
                table.iloc[0]
            )

            plot_scatter(
                fig.add_subplot(
                    page_rows,
                    3,
                    row_index * 3 + 1,
                    projection="3d",
                ),
                table,
                "cp_static_truth",
                "turbo",
                cp_norm,
                f"Truth, {velocity_text}",
                "Cp",
            )

            plot_scatter(
                fig.add_subplot(
                    page_rows,
                    3,
                    row_index * 3 + 2,
                    projection="3d",
                ),
                table,
                "cp_static_prediction",
                "turbo",
                cp_norm,
                f"Prediction, {velocity_text}",
                "Cp",
            )

            plot_scatter(
                fig.add_subplot(
                    page_rows,
                    3,
                    row_index * 3 + 3,
                    projection="3d",
                ),
                table,
                "cp_error",
                "RdBu_r",
                error_norm,
                f"Signed error, {velocity_text}",
                "Cp error",
            )

        fig.suptitle(
            (
                "SUBOFF pressure reconstruction over "
                "three-dimensional velocity conditions\n"
                f"part = {cfg.PART_NAME} | "
                f"page {page_index + 1}/{total_pages}"
            ),
            fontsize=16,
            y=0.995,
        )

        fig.subplots_adjust(
            left=0.01,
            right=0.97,
            bottom=0.01,
            top=0.975,
            wspace=0.01,
            hspace=0.03,
        )

        png_path = output_root / (
            "pressure_distribution_all_cases_"
            f"page_{page_index + 1:02d}.png"
        )

        pdf_path = output_root / (
            "pressure_distribution_all_cases_"
            f"page_{page_index + 1:02d}.pdf"
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

        print(
            f"已保存总览图第{page_index + 1}/{total_pages}页："
            f"{png_path.name}"
        )

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

    case_order = load_case_order()

    for _, case_row in case_order.iterrows():
        case_id = str(
            case_row["case_id"]
        )

        table = load_prediction(case_id)

        table["group_name"] = case_row.get(
            "group_name",
            "",
        )

        tables[case_id] = table

        print(
            f"读取：{case_id}, "
            f"u={float(case_row['u_inf_m_s']):.2f} m/s, "
            f"v={float(case_row['v_inf_m_s']):.2f} m/s, "
            f"w={float(case_row['w_inf_m_s']):.2f} m/s, "
            f"|V|={float(table['speed_m_s'].iloc[0]):.3f} m/s, "
            f"group={table['group_name'].iloc[0]}, "
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