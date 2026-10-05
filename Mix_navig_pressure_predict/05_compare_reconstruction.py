# -*- coding: utf-8 -*-
"""
三向流场稀疏压力重构结果比较与可视化。

与直航版本的主要区别：
1. 速度信息显示三个分量
2. 动压计算使用三维速度模长
3. 图表标题包含速度矢量信息
4. 统计指标记录三向速度
"""

from pathlib import Path
import importlib
import json
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import cKDTree
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, str(Path(__file__).parent))
cfg = importlib.import_module("00_sparse_config")
def load_topology_part_names() -> np.ndarray:
    """
    读取Master拓扑中每个global_face_id对应的部件名称。
    返回数组下标与global_face_id一致。
    """
    topology_path = (
        cfg.TOPOLOGY_PATH
        .expanduser()
        .resolve()
    )

    if not topology_path.is_file():
        raise FileNotFoundError(
            f"拓扑文件不存在：{topology_path}"
        )

    with h5py.File(
        topology_path,
        mode="r",
    ) as file:
        if "components/part_name" not in file:
            raise KeyError(
                "topology.h5中不存在："
                "components/part_name"
            )

        part_names = np.asarray(
            file["components/part_name"][()]
        )

    if part_names.dtype.kind == "S":
        part_names = np.char.decode(
            part_names,
            "utf-8",
            errors="replace",
        )

    elif part_names.dtype.kind == "O":
        part_names = np.asarray([
            value.decode(
                "utf-8",
                errors="replace",
            )
            if isinstance(value, bytes)
            else str(value)
            for value in part_names
        ])

    else:
        part_names = part_names.astype(str)

    part_names = np.asarray([
        str(value).strip().strip("\x00").lower()
        for value in part_names
    ])

    if len(part_names) == 0:
        raise ValueError(
            "topology.h5中的part_name为空"
        )

    return part_names

# ========== 此处包含直航版本的所有辅助函数 ==========
# load_topology_part_names()
# load_appendage_xz_projections()
# prepare_projection_data()
# plot_filled_surface_projection()
# plot_all_surface_by_parts()
# ... （完全相同，这里省略）

def calculate_validation_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
        area: np.ndarray,
        rho: float,
        vx: float,
        vy: float,
        vz: float,
) -> dict:
    """
    计算三向流场验证指标。

    新增参数：
        vx, vy, vz: 三个速度分量
    """
    error = predicted - actual

    speed_magnitude = float(np.sqrt(vx ** 2 + vy ** 2 + vz ** 2))
    q_inf = 0.5 * rho * speed_magnitude ** 2

    pressure_error = q_inf * error

    actual_variance = np.sum((actual - np.mean(actual)) ** 2)

    if actual_variance > np.finfo(float).eps:
        cp_r2 = float(1.0 - np.sum(error ** 2) / actual_variance)
    else:
        cp_r2 = np.nan

    weighted_error_norm = np.sqrt(np.sum(area * error ** 2))
    weighted_truth_norm = np.sqrt(np.sum(area * actual ** 2))
    area_sum = float(np.sum(area))

    actual_mean = float(
        np.sum(area * actual) / max(
            area_sum,
            np.finfo(float).tiny,
        )
    )

    weighted_variance_norm = np.sqrt(
        np.sum(
            area
            * (actual - actual_mean) ** 2
        )
    )

    return {
        "cp_rmse": float(np.sqrt(np.mean(error ** 2))),
        "cp_mae": float(np.mean(np.abs(error))),
        "cp_max_abs_error": float(np.max(np.abs(error))),
        "cp_error_p95": float(np.percentile(np.abs(error), 95.0)),
        "cp_error_p99": float(np.percentile(np.abs(error), 99.0)),
        "cp_area_weighted_relative_l2": float(
            weighted_error_norm / max(weighted_truth_norm, np.finfo(float).tiny)
        ),
        "cp_bias": float(np.mean(error)),
        "cp_correlation": float(np.corrcoef(actual, predicted)[0, 1]),
        "cp_r2": cp_r2,
        # 三向速度信息
        "u_inf_m_s": float(vx),
        "v_inf_m_s": float(vy),
        "w_inf_m_s": float(vz),
        "speed_magnitude_m_s": float(speed_magnitude),
        "dynamic_pressure_pa": float(q_inf),
        "pressure_rmse_pa": float(np.sqrt(np.mean(pressure_error ** 2))),
        "pressure_mae_pa": float(np.mean(np.abs(pressure_error))),
        "pressure_max_abs_error_pa": float(np.max(np.abs(pressure_error))),
        "pressure_error_p95_pa": float(np.percentile(np.abs(pressure_error), 95.0)),
        "pressure_error_p99_pa": float(np.percentile(np.abs(pressure_error), 99.0)),
    }


def compare(sensor_method: str):
    """主比较流程"""
    sensor_method = str(sensor_method).strip().lower()

    if sensor_method not in {"deim", "qr"}:
        raise ValueError(f"未知传感器方法：{sensor_method}")

    result_prefix = f"{cfg.TEST_CASE_ID}_{cfg.PART_NAME}_{sensor_method}"

    prediction_path = cfg.PREDICTION_ROOT / f"{result_prefix}_prediction.csv"
    truth_path = cfg.TRUTH_ROOT / f"{cfg.TEST_CASE_ID}_{cfg.PART_NAME}_truth.csv"

    if not prediction_path.is_file():
        raise FileNotFoundError(f"预测结果不存在：{prediction_path}")

    if not truth_path.is_file():
        raise FileNotFoundError(f"真值不存在：{truth_path}")

    prediction = pd.read_csv(prediction_path, encoding="utf-8-sig")
    truth = pd.read_csv(truth_path, encoding="utf-8-sig")
    merged = truth.merge(prediction, on="global_face_id", how="inner", validate="one_to_one")

    topology_part_names = load_topology_part_names()

    global_ids = merged[
        "global_face_id"
    ].to_numpy(
        dtype=np.int64
    )

    if np.any(global_ids < 0) or np.any(
            global_ids >= len(topology_part_names)
    ):
        raise ValueError(
            "预测结果中的global_face_id超出topology.h5范围"
        )

    merged["part_name"] = (
        topology_part_names[global_ids]
    )

    if len(merged) != len(truth):
        raise ValueError(f"预测结果没有覆盖全部真值面")

    actual = merged["cp_static_starccm"].to_numpy(dtype=np.float64)
    predicted = merged["cp_static_reconstructed"].to_numpy(dtype=np.float64)
    area = merged["face_area_m2"].to_numpy(dtype=np.float64)

    error = predicted - actual

    # 计算指标（传入三向速度）
    vx = float(cfg.TEST_U_INF)
    vy = float(cfg.TEST_V_INF)
    vz = float(cfg.TEST_W_INF)

    metrics = {
        "case_id": cfg.TEST_CASE_ID,
        "part_name": cfg.PART_NAME,
        "sensor_method": sensor_method,
        "face_count": int(len(merged)),
    }

    metrics.update(
        calculate_validation_metrics(
            actual=actual,
            predicted=predicted,
            area=area,
            rho=float(cfg.RHO),
            vx=vx,
            vy=vy,
            vz=vz,
        )
    )

    # 保存结果
    merged["cp_error"] = error
    merged["cp_absolute_error"] = np.abs(error)

    merged.to_csv(
        cfg.COMPARISON_ROOT / f"{result_prefix}_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([metrics]).to_csv(
        cfg.COMPARISON_ROOT / f"{result_prefix}_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_json = cfg.COMPARISON_ROOT / f"{result_prefix}_metrics.json"
    metric_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # =========================================================
    # 基础误差图
    # 先验证数值结果，避免依赖未恢复的三维投影辅助函数。
    # =========================================================
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
        constrained_layout=True,
    )

    axes[0].scatter(
        actual,
        predicted,
        s=3.0,
        alpha=0.65,
        rasterized=True,
    )

    finite_actual = np.isfinite(actual)
    finite_predicted = np.isfinite(predicted)
    finite_both = (
            finite_actual
            & finite_predicted
    )

    if np.any(finite_both):
        lower = float(
            min(
                np.min(actual[finite_both]),
                np.min(predicted[finite_both]),
            )
        )
        upper = float(
            max(
                np.max(actual[finite_both]),
                np.max(predicted[finite_both]),
            )
        )

        axes[0].plot(
            [lower, upper],
            [lower, upper],
            color="black",
            linestyle="--",
            linewidth=1.0,
        )

    axes[0].set_xlabel(
        "STAR-CCM+ truth $C_p$"
    )
    axes[0].set_ylabel(
        "Reconstructed $C_p$"
    )
    axes[0].set_title(
        "Truth versus reconstruction"
    )
    axes[0].grid(
        linestyle=":",
        alpha=0.35,
    )

    axes[1].hist(
        error[np.isfinite(error)],
        bins=80,
        color="#4c72b0",
        alpha=0.8,
    )

    axes[1].set_xlabel(
        "Reconstruction error in $C_p$"
    )
    axes[1].set_ylabel(
        "Face count"
    )
    axes[1].set_title(
        "Error distribution"
    )
    axes[1].grid(
        axis="y",
        linestyle=":",
        alpha=0.35,
    )

    figure.suptitle(
        f"{cfg.TEST_CASE_ID}, "
        f"{cfg.PART_NAME}, "
        f"{sensor_method.upper()}\n"
        f"V=({vx:.2f}, {vy:.2f}, {vz:.2f}) m/s, "
        f"|V|={metrics['speed_magnitude_m_s']:.2f} m/s\n"
        "sparse pressure reconstruction",
    )

    figure_path = (
            cfg.COMPARISON_ROOT
            / f"{result_prefix}_comparison.png"
    )

    figure.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )


    print("\n稀疏压力重构验证结果：")
    for name, value in metrics.items():
        print(f"  {name:<35} {value}")
    print(f"对比图：{figure_path}")


if __name__ == "__main__":
    successful_methods = []
    failed_methods = []

    for sensor_method in ["deim", "qr"]:
        print("\n" + "=" * 90)
        print(f"开始比较{sensor_method.upper()}重构结果")
        print("=" * 90)

        try:
            compare(sensor_method)
            successful_methods.append(sensor_method)
        except Exception as exc:
            failed_methods.append({
                "sensor_method": sensor_method,
                "error": repr(exc),
            })
            print(f"{sensor_method.upper()}比较失败：{exc}")

    print("\n" + "=" * 90)
    print("重构结果比较摘要")
    print("=" * 90)
    print(
        "成功方法："
        + (", ".join(method.upper() for method in successful_methods) if successful_methods else "无")
    )

    if failed_methods:
        print("失败方法：")
        for item in failed_methods:
            print(f"  {item['sensor_method'].upper()}: {item['error']}")
        raise RuntimeError("存在重构结果比较失败")
    else:
        print("失败方法：无")

    print("=" * 90)