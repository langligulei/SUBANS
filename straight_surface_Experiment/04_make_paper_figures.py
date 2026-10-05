# -*- coding: utf-8 -*-
"""
根据02_run_sparse_rom_experiments.py的结果生成SCI论文图片。

输出
----
Figure 1:
    传感器数量敏感性

Figure 2:
    POD模态数量敏感性

Figure 3:
    QR、DEIM、均匀和随机传感器比较

Figure 4:
    均值场、线性速度基线和稀疏ROM比较

Figure 5:
    噪声鲁棒性

Figure 6:
    插值与外推工况泛化

Figure 7:
    压力积分误差（如果存在法向量）

格式
----
每幅图同时保存：
    PNG，300 dpi
    PDF，矢量格式
"""

from pathlib import Path
import importlib
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

cfg = importlib.import_module(
    "00_experiment_config"
)

# =============================================================================
# 1. 绘图设置
# =============================================================================

METHOD_COLORS = {
    "qr": "#d62728",
    "deim": "#1f77b4",
    "uniform": "#2ca02c",
    "random": "#7f7f7f",
    "mean_field": "#9467bd",
    "linear_speed_interpolation": "#ff7f0e",
    "linear_speed_extrapolation": "#8c564b",
}

METHOD_MARKERS = {
    "qr": "o",
    "deim": "s",
    "uniform": "^",
    "random": "D",
    "mean_field": "P",
    "linear_speed_interpolation": "X",
    "linear_speed_extrapolation": "X",
}

def configure_style():
    """设置统一SCI绘图风格。"""
    plt.rcParams.update({
        "font.family": cfg.PLOT_FONT_FAMILY,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    })

def save_figure(figure, stem):
    """同时保存PNG和PDF。"""
    for extension in cfg.FIGURE_FORMATS:
        path = (
            cfg.FIGURE_ROOT
            / f"{stem}.{extension}"
        )

        figure.savefig(
            path,
            dpi=cfg.FIGURE_DPI,
            bbox_inches="tight",
            pad_inches=0.05,
        )

    plt.close(figure)

def read_table(name):
    """读取结果表。"""
    path = cfg.TABLE_ROOT / name

    if not path.is_file():
        return None

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

def plot_grouped_curves(
    table,
    x_column,
    y_column,
    title,
    x_label,
    y_label,
    stem,
):
    """
    对每种方法绘制跨工况平均曲线。

    阴影为不同未见工况之间的最小值和最大值，
    曲线为工况平均值。
    """
    if table is None or len(table) == 0:
        return

    figure, axis = plt.subplots(
        figsize=(7.2, 4.8),
        constrained_layout=True,
    )

    for method, group in table.groupby(
        "method"
    ):
        statistics = (
            group.groupby(
                x_column,
                as_index=False,
            )[y_column]
            .agg([
                "mean",
                "min",
                "max",
            ])
            .reset_index()
        )

        statistics = statistics.sort_values(
            x_column
        )

        x = statistics[
            x_column
        ].to_numpy(dtype=float)

        mean = statistics[
            "mean"
        ].to_numpy(dtype=float)

        lower = statistics[
            "min"
        ].to_numpy(dtype=float)

        upper = statistics[
            "max"
        ].to_numpy(dtype=float)

        color = METHOD_COLORS.get(
            method,
            None,
        )

        axis.plot(
            x,
            100.0 * mean,
            marker=METHOD_MARKERS.get(
                method,
                "o",
            ),
            color=color,
            label=method.upper(),
        )

        axis.fill_between(
            x,
            100.0 * lower,
            100.0 * upper,
            color=color,
            alpha=0.12,
            linewidth=0.0,
        )

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(
        linestyle=":",
        alpha=0.35,
    )
    axis.legend(
        frameon=True,
    )

    save_figure(
        figure,
        stem,
    )

# =============================================================================
# 2. 传感器数量敏感性
# =============================================================================

def make_sensor_count_figure():
    table = read_table(
        "sensor_count_sensitivity.csv"
    )

    plot_grouped_curves(
        table=table,
        x_column="sensor_count",
        y_column=(
            "cp_area_weighted_relative_l2"
        ),
        title=(
            "Sensitivity to the number "
            "of pressure sensors"
        ),
        x_label=r"Number of sensors, $m$",
        y_label=(
            "Area-weighted relative "
            r"$L_2$ error (%)"
        ),
        stem="figure_sensor_count_sensitivity",
    )

# =============================================================================
# 3. POD模态数量敏感性
# =============================================================================

def make_mode_count_figure():
    table = read_table(
        "pod_mode_sensitivity.csv"
    )

    plot_grouped_curves(
        table=table,
        x_column="mode_count",
        y_column=(
            "cp_area_weighted_relative_l2"
        ),
        title=(
            "Sensitivity to the number "
            "of POD modes"
        ),
        x_label=r"Number of POD modes, $r$",
        y_label=(
            "Area-weighted relative "
            r"$L_2$ error (%)"
        ),
        stem="figure_pod_mode_sensitivity",
    )

# =============================================================================
# 4. QR、DEIM、均匀和随机比较
# =============================================================================

def make_sensor_method_figure():
    deterministic = read_table(
        "sensor_count_sensitivity.csv"
    )

    random_statistics = read_table(
        "random_sensor_statistics.csv"
    )

    if deterministic is None:
        return

    figure, axis = plt.subplots(
        figsize=(7.4, 4.9),
        constrained_layout=True,
    )

    for method, group in deterministic.groupby(
        "method"
    ):
        summary = (
            group.groupby(
                "sensor_count",
                as_index=False,
            )[
                "cp_area_weighted_relative_l2"
            ]
            .mean()
            .sort_values(
                "sensor_count"
            )
        )

        axis.plot(
            summary["sensor_count"],
            100.0
            * summary[
                "cp_area_weighted_relative_l2"
            ],
            marker=METHOD_MARKERS.get(
                method,
                "o",
            ),
            color=METHOD_COLORS.get(
                method,
                None,
            ),
            label=method.upper(),
        )

    if (
        random_statistics is not None
        and len(random_statistics) > 0
    ):
        summary = (
            random_statistics.groupby(
                "sensor_count",
                as_index=False,
            )
            .agg({
                "cp_area_weighted_relative_l2_mean": "mean",
                "cp_area_weighted_relative_l2_ci95_lower": "mean",
                "cp_area_weighted_relative_l2_ci95_upper": "mean",
            })
            .sort_values(
                "sensor_count"
            )
        )

        x = summary[
            "sensor_count"
        ].to_numpy(dtype=float)

        mean = summary[
            "cp_area_weighted_relative_l2_mean"
        ].to_numpy(dtype=float)

        lower = summary[
            "cp_area_weighted_relative_l2_ci95_lower"
        ].to_numpy(dtype=float)

        upper = summary[
            "cp_area_weighted_relative_l2_ci95_upper"
        ].to_numpy(dtype=float)

        axis.plot(
            x,
            100.0 * mean,
            color=METHOD_COLORS["random"],
            marker=METHOD_MARKERS["random"],
            linestyle="--",
            label="Random mean",
        )

        axis.fill_between(
            x,
            100.0 * lower,
            100.0 * upper,
            color=METHOD_COLORS["random"],
            alpha=0.20,
            label="Random 95% interval",
        )

    axis.set_xlabel(
        r"Number of sensors, $m$"
    )
    axis.set_ylabel(
        "Area-weighted relative "
        r"$L_2$ error (%)"
    )
    axis.set_title(
        "Comparison of sensor placement methods"
    )
    axis.grid(
        linestyle=":",
        alpha=0.35,
    )
    axis.legend(
        ncol=2,
        frameon=True,
    )

    save_figure(
        figure,
        "figure_sensor_method_comparison",
    )

# =============================================================================
# 5. 基线比较
# =============================================================================

def make_baseline_figure():
    baseline = read_table(
        "baseline_comparison.csv"
    )

    main = read_table(
        "multi_condition_generalization.csv"
    )

    if baseline is None or main is None:
        return

    main = main.copy()
    main["method"] = (
        "sparse_rom_"
        + main["method"].astype(str)
    )

    combined = pd.concat(
        [
            baseline,
            main,
        ],
        ignore_index=True,
        sort=False,
    )

    combined = combined.sort_values(
        "speed_m_s"
    )

    figure, axis = plt.subplots(
        figsize=(7.8, 5.0),
        constrained_layout=True,
    )

    for method, group in combined.groupby(
        "method"
    ):
        group = group.sort_values(
            "speed_m_s"
        )

        label = method.replace(
            "_",
            " ",
        ).title()

        axis.plot(
            group["speed_m_s"],
            100.0
            * group[
                "cp_area_weighted_relative_l2"
            ],
            marker="o",
            label=label,
        )

    # 标识训练速度区间之外的外推区域。
    axis.axvspan(
        0.0,
        1.0,
        color="0.8",
        alpha=0.25,
        label="Extrapolation region",
    )

    axis.axvspan(
        10.0,
        11.0,
        color="0.8",
        alpha=0.25,
    )

    axis.set_xlabel(
        r"Unseen inflow speed, $U_\infty$ (m/s)"
    )
    axis.set_ylabel(
        "Area-weighted relative "
        r"$L_2$ error (%)"
    )
    axis.set_title(
        "Sparse ROM versus simple baselines"
    )
    axis.grid(
        linestyle=":",
        alpha=0.35,
    )
    axis.legend(
        frameon=True,
        fontsize=8,
    )

    save_figure(
        figure,
        "figure_baseline_comparison",
    )

# =============================================================================
# 6. 噪声鲁棒性
# =============================================================================

def make_noise_figure():
    table = read_table(
        "noise_robustness_statistics.csv"
    )

    if table is None or len(table) == 0:
        return

    summary = (
        table.groupby(
            [
                "method",
                "noise_percent",
            ],
            as_index=False,
        )
        .agg({
            "relative_l2_mean": "mean",
            "relative_l2_ci95_lower": "mean",
            "relative_l2_ci95_upper": "mean",
        })
    )

    figure, axis = plt.subplots(
        figsize=(7.3, 4.8),
        constrained_layout=True,
    )

    for method, group in summary.groupby(
        "method"
    ):
        group = group.sort_values(
            "noise_percent"
        )

        x = group[
            "noise_percent"
        ].to_numpy(dtype=float)

        mean = group[
            "relative_l2_mean"
        ].to_numpy(dtype=float)

        lower = group[
            "relative_l2_ci95_lower"
        ].to_numpy(dtype=float)

        upper = group[
            "relative_l2_ci95_upper"
        ].to_numpy(dtype=float)

        color = METHOD_COLORS.get(
            method,
            None,
        )

        axis.plot(
            x,
            100.0 * mean,
            marker=METHOD_MARKERS.get(
                method,
                "o",
            ),
            color=color,
            label=method.upper(),
        )

        axis.fill_between(
            x,
            100.0 * lower,
            100.0 * upper,
            color=color,
            alpha=0.15,
        )

    axis.set_xlabel(
        r"Pressure-noise level, "
        r"$\sigma_p/q_\infty$ (%)"
    )
    axis.set_ylabel(
        "Area-weighted relative "
        r"$L_2$ error (%)"
    )
    axis.set_title(
        "Robustness to pressure-measurement noise"
    )
    axis.grid(
        linestyle=":",
        alpha=0.35,
    )
    axis.legend(
        frameon=True,
    )

    save_figure(
        figure,
        "figure_noise_robustness",
    )

# =============================================================================
# 7. 插值和外推能力
# =============================================================================

def make_generalization_figure():
    table = read_table(
        "multi_condition_generalization.csv"
    )

    if table is None or len(table) == 0:
        return

    table = table.sort_values(
        "speed_m_s"
    )

    colors = [
        (
            "#c44e52"
            if split == "extrapolation"
            else "#4c72b0"
        )
        for split in table[
            "split_type"
        ].astype(str).str.lower()
    ]

    figure, axis = plt.subplots(
        figsize=(7.3, 4.8),
        constrained_layout=True,
    )

    axis.plot(
        table["speed_m_s"],
        100.0
        * table[
            "cp_area_weighted_relative_l2"
        ],
        color="0.45",
        linewidth=1.2,
        zorder=1,
    )

    axis.scatter(
        table["speed_m_s"],
        100.0
        * table[
            "cp_area_weighted_relative_l2"
        ],
        c=colors,
        s=60,
        edgecolors="black",
        linewidths=0.5,
        zorder=2,
    )

    for _, row in table.iterrows():
        axis.annotate(
            str(row["split_type"]),
            (
                row["speed_m_s"],
                100.0
                * row[
                    "cp_area_weighted_relative_l2"
                ],
            ),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    axis.set_xlabel(
        r"Unseen inflow speed, $U_\infty$ (m/s)"
    )
    axis.set_ylabel(
        "Area-weighted relative "
        r"$L_2$ error (%)"
    )
    axis.set_title(
        "Generalization to unseen SUBOFF conditions"
    )
    axis.grid(
        linestyle=":",
        alpha=0.35,
    )

    save_figure(
        figure,
        "figure_unseen_condition_generalization",
    )

# =============================================================================
# 8. 压力积分
# =============================================================================

def make_integral_figure():
    table = read_table(
        "pressure_integral_errors.csv"
    )

    if table is None or len(table) == 0:
        return

    if "status" in table.columns:
        if str(table.iloc[0]["status"]) == "skipped":
            print(
                "未找到法向量，跳过压力积分图。"
            )
            return

    main_method = cfg.MAIN_SENSOR_METHOD

    selected = table.loc[
        table["method"].astype(str)
        == main_method
    ].copy()

    if len(selected) == 0:
        return

    selected = selected.sort_values(
        "speed_m_s"
    )

    quantities = [
        (
            "coefficient_force_x_relative_error",
            r"Axial-force coefficient $C_{F_x}$",
        ),
        (
            "coefficient_force_z_relative_error",
            r"Lift coefficient $C_{F_z}$",
        ),
        (
            "coefficient_moment_y_relative_error",
            r"Pitching-moment coefficient $C_{M_y}$",
        ),
    ]

    figure, axis = plt.subplots(
        figsize=(7.8, 5.0),
        constrained_layout=True,
    )

    for column, label in quantities:
        if column not in selected.columns:
            continue

        values = pd.to_numeric(
            selected[column],
            errors="coerce",
        )

        axis.plot(
            selected["speed_m_s"],
            100.0 * values,
            marker="o",
            label=label,
        )

    axis.set_xlabel(
        r"Unseen inflow speed, $U_\infty$ (m/s)"
    )
    axis.set_ylabel(
        "Relative pressure-integral error (%)"
    )
    axis.set_title(
        "Errors of pressure-integrated loads"
    )
    axis.grid(
        linestyle=":",
        alpha=0.35,
    )
    axis.legend(
        frameon=True,
    )

    save_figure(
        figure,
        "figure_pressure_integral_errors",
    )

# =============================================================================
# 9. 主入口
# =============================================================================

def run():
    configure_style()

    make_sensor_count_figure()
    make_mode_count_figure()
    make_sensor_method_figure()
    make_baseline_figure()
    make_noise_figure()
    make_generalization_figure()
    make_integral_figure()

    print("=" * 80)
    print("SCI论文图片生成完成")
    print(f"图片目录：{cfg.FIGURE_ROOT}")
    print("=" * 80)

if __name__ == "__main__":
    run()