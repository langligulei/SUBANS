# -*- coding: utf-8 -*-

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

def load_topology_part_names():
    """读取topology.h5中每个全局面对应的部件名称。"""
    with h5py.File(
        cfg.TOPOLOGY_PATH,
        mode="r",
    ) as file:
        values = file[
            "components/part_name"
        ][()]

    return np.asarray([
        value.decode("utf-8").strip().lower()
        if isinstance(value, bytes)
        else str(value).strip().lower()
        for value in values
    ])

def load_appendage_xz_projections():
    """读取五个附体的面心，并统一投影到hull使用的XZ平面。"""
    with h5py.File(cfg.TOPOLOGY_PATH, "r") as file:
        centroids = file["geometry/centroid"][()]
        raw_names = file["components/part_name"][()]

    part_names = np.asarray([
        value.decode("utf-8").strip().lower()
        if isinstance(value, bytes)
        else str(value).strip().lower()
        for value in raw_names
    ])

    projections = {}

    for name in ("sail", "fin1", "fin2", "fin3", "fin4"):
        mask = part_names == name
        points_xz = centroids[mask][:, [0, 2]]
        points_xz = points_xz[np.all(np.isfinite(points_xz), axis=1)]

        # 删除两侧表面投影后产生的重复坐标。
        points_xz = np.unique(
            np.round(points_xz, decimals=10),
            axis=0,
        )

        if len(points_xz) >= 3:
            projections[name] = points_xz

    return projections

def prepare_projection_data(
    table: pd.DataFrame,
    values: np.ndarray,
    part_name: str,
):
    """
    根据附体方向选择合适的投影平面，并只显示一个可见侧面。

    sail、fin1、fin3：XZ投影，沿Y方向选取一个侧面；
    fin2、fin4：XY投影，沿Z方向选取一个侧面；
    hull、all_surface：XZ投影，不执行单侧筛选。
    """
    part_name = str(part_name).strip().lower()
    values = np.asarray(values, dtype=np.float64)

    x = table["x_m"].to_numpy(dtype=np.float64)
    y = table["y_m"].to_numpy(dtype=np.float64)
    z = table["z_m"].to_numpy(dtype=np.float64)

    if part_name in {"sail", "fin1", "fin3"}:
        # 竖直附体：显示XZ平面，并选择Y正侧面。
        side_coordinate = y
        side_middle = 0.5 * (
            np.min(side_coordinate)
            + np.max(side_coordinate)
        )
        mask = side_coordinate >= side_middle

        u = x[mask]
        v = z[mask]
        color_value = values[mask]
        x_label = "X (m)"
        y_label = "Z (m)"
        projection_name = "XZ projection"

    elif part_name in {"fin2", "fin4"}:
        # 水平附体：显示XY平面，并选择Z正侧面。
        side_coordinate = z
        side_middle = 0.5 * (
            np.min(side_coordinate)
            + np.max(side_coordinate)
        )
        mask = side_coordinate >= side_middle

        u = x[mask]
        v = y[mask]
        color_value = values[mask]
        x_label = "X (m)"
        y_label = "Y (m)"
        projection_name = "XY projection"

    else:
        # 艇体及全艇仍使用XZ投影。
        u = x
        v = z
        color_value = values
        x_label = "X (m)"
        y_label = "Z (m)"
        projection_name = "XZ projection"

    finite = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(color_value)
    )

    u = u[finite]
    v = v[finite]
    color_value = color_value[finite]

    # 删除投影后重合的坐标点，避免Delaunay三角剖分失败。
    projection_table = pd.DataFrame({
        "u": u,
        "v": v,
        "value": color_value,
    })

    projection_table["u_key"] = np.round(
        projection_table["u"],
        decimals=10,
    )
    projection_table["v_key"] = np.round(
        projection_table["v"],
        decimals=10,
    )

    projection_table = (
        projection_table
        .groupby(
            ["u_key", "v_key"],
            as_index=False,
        )
        .agg({
            "u": "mean",
            "v": "mean",
            "value": "mean",
        })
    )

    return (
        projection_table["u"].to_numpy(dtype=np.float64),
        projection_table["v"].to_numpy(dtype=np.float64),
        projection_table["value"].to_numpy(dtype=np.float64),
        x_label,
        y_label,
        projection_name,
    )

def plot_filled_surface_projection(
    axis,
    table: pd.DataFrame,
    values: np.ndarray,
    part_name: str,
    title_prefix: str,
    cmap: str = "coolwarm",
    vmin=None,
    vmax=None,
):
    """
    根据面心坐标生成三角剖分连续填色云图。

    注意：这是投影三角剖分云图，不是原始CFD多边形面的严格渲染。
    """
    (
        u,
        v,
        color_value,
        x_label,
        y_label,
        projection_name,
    ) = prepare_projection_data(
        table=table,
        values=values,
        part_name=part_name,
    )

    if len(u) < 3:
        raise ValueError(
            f"{part_name}投影后的有效点少于3个，无法三角剖分"
        )

    triangulation = mtri.Triangulation(u, v)

    # ---------------------------------------------------------
    # 屏蔽跨越附体外部空白区域的过长三角形
    # ---------------------------------------------------------
    points = np.column_stack([u, v])
    tree = cKDTree(points)

    nearest_distance, _ = tree.query(
        points,
        k=2,
    )

    characteristic_spacing = float(
        np.median(nearest_distance[:, 1])
    )

    triangles = triangulation.triangles
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]

    edge_01 = np.linalg.norm(p0 - p1, axis=1)
    edge_12 = np.linalg.norm(p1 - p2, axis=1)
    edge_20 = np.linalg.norm(p2 - p0, axis=1)

    maximum_edge = np.maximum.reduce([
        edge_01,
        edge_12,
        edge_20,
    ])

    # 标准化部件名称。必须在后续判断前定义。
    part_name_normalized = str(
        part_name
    ).strip().lower()

    # all_surface由多个相互分离的部件构成，只对它进行跨部件
    # 长边屏蔽。hull的三维表面投影会产生非常小的最近邻距离，
    # 不适合使用全局距离阈值，否则可能屏蔽全部正常三角形。
    if part_name_normalized == "all_surface":
        maximum_allowed_edge = (
                8.0 * characteristic_spacing
        )

        invalid_triangle = (
                maximum_edge
                > maximum_allowed_edge
        )

        masked_fraction = float(
            np.mean(invalid_triangle)
        )

        # 防止阈值异常导致云图被全部删除。
        if masked_fraction < 0.95:
            triangulation.set_mask(
                invalid_triangle
            )

    # ---------------------------------------------------------
    # 绘制hull时，用sail和四个fin的XZ投影切除安装区域
    # ---------------------------------------------------------
    if part_name_normalized == "hull":
        appendage_projections = (
            load_appendage_xz_projections()
        )

        triangle_centers = (
                                   p0 + p1 + p2
                           ) / 3.0

        midpoint_01 = (p0 + p1) / 2.0
        midpoint_12 = (p1 + p2) / 2.0
        midpoint_20 = (p2 + p0) / 2.0

        # 每个hull三角形使用中心和三个边中点进行判定。
        triangle_samples = np.stack(
            [
                triangle_centers,
                midpoint_01,
                midpoint_12,
                midpoint_20,
            ],
            axis=1,
        )

        flat_samples = triangle_samples.reshape(-1, 2)

        hull_cutout_mask = np.zeros(
            len(triangles),
            dtype=bool,
        )

        for appendage_name, appendage_points in (
                appendage_projections.items()
        ):
            appendage_tree = cKDTree(
                appendage_points
            )

            # 根据附体投影点云自身的局部间距确定切除半径。
            neighbor_count = min(
                7,
                len(appendage_points),
            )

            neighbor_distances, _ = (
                appendage_tree.query(
                    appendage_points,
                    k=neighbor_count,
                )
            )

            if neighbor_count <= 1:
                continue
            #   如果孔洞切除偏小，0.8 调到 1.0；如果切除范围过宽，则调到 0.6
            coverage_radius = 0 * float(
                np.percentile(
                    neighbor_distances[:, -1],
                    90.0,
                )
            )

            sample_distances, _ = (
                appendage_tree.query(
                    flat_samples,
                    k=1,
                )
            )

            sample_hits = (
                    sample_distances.reshape(-1, 4)
                    <= coverage_radius
            )

            # 三角形中心命中，或者至少两个边中点命中时切除。
            appendage_cutout = (
                    sample_hits[:, 0]
                    | (
                            np.count_nonzero(
                                sample_hits[:, 1:],
                                axis=1,
                            ) >= 2
                    )
            )

            hull_cutout_mask |= appendage_cutout

            print(
                f"Hull cutout {appendage_name}: "
                f"points={len(appendage_points):,}, "
                f"radius={coverage_radius:.6e}, "
                f"triangles="
                f"{np.count_nonzero(appendage_cutout):,}"
            )

        hull_cutout_fraction = float(
            np.mean(hull_cutout_mask)
        )

        print(
            "Hull appendage cutout: "
            f"masked={np.count_nonzero(hull_cutout_mask):,}/"
            f"{len(hull_cutout_mask):,}, "
            f"ratio={hull_cutout_fraction:.2%}"
        )

        # 防止坐标或半径异常导致大面积hull被删除。
        if 0.0 < hull_cutout_fraction < 0.50:
            triangulation.set_mask(
                hull_cutout_mask
            )
        elif hull_cutout_fraction >= 0.50:
            print(
                "警告：hull切除比例超过50%，"
                "本次不应用附体切除。"
            )

    if vmin is None:
        vmin = float(np.min(color_value))

    if vmax is None:
        vmax = float(np.max(color_value))

    if np.isclose(vmin, vmax):
        delta = max(abs(vmin), 1.0) * 1.0e-12
        vmin -= delta
        vmax += delta

    levels = np.linspace(
        float(vmin),
        float(vmax),
        101,
    )

    surface = axis.tricontourf(
        triangulation,
        color_value,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend="both",
    )

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(
        f"{title_prefix}: {projection_name}"
    )
    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    return surface

def plot_all_surface_by_parts(
    axis,
    table: pd.DataFrame,
    values: np.ndarray,
    title_prefix: str,
    cmap: str = "coolwarm",
    vmin=None,
    vmax=None,
):
    """
    将六个部件绘制在同一个XZ投影坐标轴中。

    注意：
    1. 六个部件使用同一个色标；
    2. 每个部件独立进行二维Delaunay三角剖分；
    3. 不对hull执行附体区域切除；
    4. sail和fin绘制在hull上层；
    5. 避免hull、sail和fin之间出现虚假的跨部件三角形。
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(table) != len(values):
        raise ValueError(
            "all_surface绘图数据长度不一致："
            f"table={len(table):,}, "
            f"values={len(values):,}"
        )

    if "part_name" not in table.columns:
        raise KeyError(
            "all_surface绘图数据缺少part_name列"
        )

    normalized_part_names = (
        table["part_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .to_numpy()
    )

    # hull最先绘制，附体随后绘制在hull上方。
    part_order = [
        "hull",
        "sail",
        "fin1",
        "fin2",
        "fin3",
        "fin4",
    ]

    surfaces = []

    for current_part_name in part_order:
        part_mask = (
            normalized_part_names
            == current_part_name
        )

        part_face_count = int(
            np.count_nonzero(part_mask)
        )

        if part_face_count < 3:
            print(
                f"跳过{current_part_name}："
                f"有效面数={part_face_count:,}"
            )
            continue

        part_table = (
            table.loc[part_mask]
            .reset_index(drop=True)
        )

        part_values = values[part_mask]

        # 这里故意传入all_surface，而不是current_part_name。
        #
        # 原因：
        # 1. 强制所有部件统一使用XZ投影；
        # 2. 不触发part_name == "hull"的孔洞切除；
        # 3. fin2、fin4在整体侧视图中也使用XZ投影；
        # 4. 当前传入的数据仍然只有一个部件，因此不会发生
        #    不同部件之间的Delaunay错误连接。
        surface = plot_filled_surface_projection(
            axis=axis,
            table=part_table,
            values=part_values,
            part_name="all_surface",
            title_prefix="",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        surfaces.append(surface)

        print(
            f"all_surface绘制部件："
            f"{current_part_name}, "
            f"faces={part_face_count:,}"
        )

    if not surfaces:
        raise ValueError(
            "all_surface中没有足够的有效面用于绘图"
        )

    axis.set_xlabel("X (m)")
    axis.set_ylabel("Z (m)")
    axis.set_title(title_prefix)
    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    # 所有部件使用相同cmap、vmin和vmax，
    # 因而任意一个surface都可以用于建立公共颜色条。
    return surfaces[0]


def calculate_validation_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    area: np.ndarray,
    rho: float,
    speed_inf: float,
) -> dict:
    error = predicted - actual

    q_inf = (
        0.5
        * rho
        * speed_inf ** 2
    )

    pressure_error = q_inf * error

    actual_variance = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if actual_variance > np.finfo(float).eps:
        cp_r2 = float(
            1.0
            - np.sum(error ** 2)
            / actual_variance
        )
    else:
        cp_r2 = np.nan

    weighted_error_norm = np.sqrt(
        np.sum(area * error ** 2)
    )

    weighted_truth_norm = np.sqrt(
        np.sum(area * actual ** 2)
    )

    return {
        "cp_rmse": float(
            np.sqrt(np.mean(error ** 2))
        ),
        "cp_mae": float(
            np.mean(np.abs(error))
        ),
        "cp_max_abs_error": float(
            np.max(np.abs(error))
        ),
        "cp_error_p95": float(
            np.percentile(
                np.abs(error),
                95.0,
            )
        ),
        "cp_error_p99": float(
            np.percentile(
                np.abs(error),
                99.0,
            )
        ),
        "cp_area_weighted_relative_l2": float(
            weighted_error_norm
            / max(
                weighted_truth_norm,
                np.finfo(float).tiny,
            )
        ),
        "cp_bias": float(
            np.mean(error)
        ),
        "cp_correlation": float(
            np.corrcoef(
                actual,
                predicted,
            )[0, 1]
        ),
        "cp_r2": cp_r2,
        "speed_inf_m_s": float(
            speed_inf
        ),
        "dynamic_pressure_pa": float(
            q_inf
        ),
        "pressure_rmse_pa": float(
            np.sqrt(
                np.mean(
                    pressure_error ** 2
                )
            )
        ),
        "pressure_mae_pa": float(
            np.mean(
                np.abs(pressure_error)
            )
        ),
        "pressure_max_abs_error_pa": float(
            np.max(
                np.abs(pressure_error)
            )
        ),
        "pressure_error_p95_pa": float(
            np.percentile(
                np.abs(pressure_error),
                95.0,
            )
        ),
        "pressure_error_p99_pa": float(
            np.percentile(
                np.abs(pressure_error),
                99.0,
            )
        ),
    }

def compare(sensor_method: str):
    sensor_method = str(
        sensor_method
    ).strip().lower()

    if sensor_method not in {"deim", "qr"}:
        raise ValueError(
            f"未知传感器方法：{sensor_method}"
        )

    result_prefix = (
        f"{cfg.TEST_CASE_ID}_"
        f"{cfg.PART_NAME}_"
        f"{sensor_method}"
    )

    prediction_path = (
        cfg.PREDICTION_ROOT
        / f"{result_prefix}_prediction.csv"
    )
    truth_path = (
        cfg.TRUTH_ROOT
        / f"{cfg.TEST_CASE_ID}_{cfg.PART_NAME}_truth.csv"
    )

    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"预测结果不存在：{prediction_path}"
        )

    if not truth_path.is_file():
        raise FileNotFoundError(
            f"STAR-CCM+真值不存在：{truth_path}"
        )

    prediction = pd.read_csv(prediction_path, encoding="utf-8-sig")
    truth = pd.read_csv(truth_path, encoding="utf-8-sig")
    merged = truth.merge(
        prediction,
        on="global_face_id",
        how="inner",
        validate="one_to_one",
    )

    topology_part_names = (
        load_topology_part_names()
    )

    global_ids = merged[
        "global_face_id"
    ].to_numpy(dtype=np.int64)

    if np.any(global_ids < 0) or np.any(
            global_ids >= len(topology_part_names)
    ):
        raise ValueError(
            "comparison结果中的global_face_id超出topology.h5范围"
        )

    merged["part_name"] = topology_part_names[
        global_ids
    ]

    if len(merged) != len(truth):
        raise ValueError(
            "预测结果没有覆盖全部真值面："
            f"truth={len(truth):,}, matched={len(merged):,}"
        )

    actual = merged[
        "cp_static_starccm"
    ].to_numpy(dtype=np.float64)

    predicted = merged[
        "cp_static_reconstructed"
    ].to_numpy(dtype=np.float64)

    area = merged[
        "face_area_m2"
    ].to_numpy(dtype=np.float64)

    error = predicted - actual

    # ---------------------------------------------------------
    # 逐面相对误差
    # ---------------------------------------------------------
    # Cp可能经过零点，直接除以abs(actual)会产生极大值或Inf。
    # 使用全场Cp幅值的1e-6作为最小分母，仅用于绘图统计。
    cp_reference_scale = max(
        float(np.max(np.abs(actual))),
        np.finfo(float).eps,
    )

    relative_error_denominator = np.maximum(
        np.abs(actual),
        1.0e-6 * cp_reference_scale,
    )

    relative_absolute_error = (
            np.abs(error)
            / relative_error_denominator
    )

    relative_absolute_error_percent = (
            100.0 * relative_absolute_error
    )

    # ---------------------------------------------------------
    # 补充验证指标
    # ---------------------------------------------------------

    # 1. C_p决定系数 R2
    actual_variance = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if actual_variance > np.finfo(float).eps:
        cp_r2 = float(
            1.0
            - np.sum(error ** 2)
            / actual_variance
        )
    else:
        cp_r2 = np.nan

    # 2. C_p误差分位数
    cp_error_p95 = float(
        np.percentile(
            np.abs(error),
            95.0,
        )
    )

    cp_error_p99 = float(
        np.percentile(
            np.abs(error),
            99.0,
        )
    )

    # 3. 来流速度和动压
    speed_inf = float(
        np.sqrt(
            float(cfg.TEST_U_INF) ** 2
            + float(cfg.TEST_V_INF) ** 2
            + float(cfg.TEST_W_INF) ** 2
        )
    )

    q_inf = float(
        0.5
        * float(cfg.RHO)
        * speed_inf ** 2
    )

    # 4. 将Cp误差转换为绝对压力误差
    pressure_error = q_inf * error

    pressure_rmse_pa = float(
        np.sqrt(
            np.mean(
                pressure_error ** 2
            )
        )
    )

    pressure_mae_pa = float(
        np.mean(
            np.abs(pressure_error)
        )
    )

    pressure_max_abs_error_pa = float(
        np.max(
            np.abs(pressure_error)
        )
    )

    pressure_error_p95_pa = float(
        np.percentile(
            np.abs(pressure_error),
            95.0,
        )
    )

    pressure_error_p99_pa = float(
        np.percentile(
            np.abs(pressure_error),
            99.0,
        )
    )

    speed_inf = float(
        np.sqrt(
            float(cfg.TEST_U_INF) ** 2
            + float(cfg.TEST_V_INF) ** 2
            + float(cfg.TEST_W_INF) ** 2
        )
    )

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
            speed_inf=speed_inf,
        )
    )
    metrics.update({
        "cp_relative_absolute_error_median_percent": float(
            np.median(
                relative_absolute_error_percent
            )
        ),
        "cp_relative_absolute_error_p90_percent": float(
            np.percentile(
                relative_absolute_error_percent,
                90.0,
            )
        ),
        "cp_relative_absolute_error_p95_percent": float(
            np.percentile(
                relative_absolute_error_percent,
                95.0,
            )
        ),
        "cp_relative_absolute_error_p99_percent": float(
            np.percentile(
                relative_absolute_error_percent,
                99.0,
            )
        ),
    })

    merged["cp_error"] = error
    merged["cp_absolute_error"] = np.abs(error)
    merged[
        "cp_relative_absolute_error"
    ] = relative_absolute_error

    merged[
        "cp_relative_absolute_error_percent"
    ] = relative_absolute_error_percent
    merged.to_csv(
        cfg.COMPARISON_ROOT
        / f"{result_prefix}_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([metrics]).to_csv(
        cfg.COMPARISON_ROOT
        / f"{result_prefix}_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_json = (
            cfg.COMPARISON_ROOT
            / f"{result_prefix}_metrics.json"
    )
    metric_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # =========================================================
    # 2行3列重构验证图
    # =========================================================
    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(20.0, 12.0),
        constrained_layout=True,
    )

    # ---------------------------------------------------------
    # 第1行第1列：逐面预测与真值比较
    # ---------------------------------------------------------
    lower = float(
        min(
            np.min(actual),
            np.min(predicted),
        )
    )

    upper = float(
        max(
            np.max(actual),
            np.max(predicted),
        )
    )

    axes[0, 0].scatter(
        actual,
        predicted,
        s=2,
        alpha=0.35,
        rasterized=True,
    )

    axes[0, 0].plot(
        [lower, upper],
        [lower, upper],
        "k--",
        linewidth=1.0,
    )

    axes[0, 0].set_xlim(lower, upper)
    axes[0, 0].set_ylim(lower, upper)
    axes[0, 0].set_xlabel(r"STAR-CCM+ $C_p$")
    axes[0, 0].set_ylabel(r"Reconstructed $C_p$")
    axes[0, 0].set_title("Pointwise comparison")
    axes[0, 0].set_aspect(
        "equal",
        adjustable="box",
    )

    # ---------------------------------------------------------
    # 第1行第2列：带符号Cp误差面元数量分布
    # ---------------------------------------------------------
    axes[0, 1].hist(
        error,
        bins=100,
        color="#4c72b0",
        alpha=0.85,
    )

    axes[0, 1].axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )

    axes[0, 1].set_xlabel(
        r"$C_{p,recon}-C_{p,STAR}$"
    )
    axes[0, 1].set_ylabel("Face count")
    axes[0, 1].set_title("Error distribution")

    # ---------------------------------------------------------
    # 第1行第3列：逐面绝对相对误差分布
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 第1行第3列：逐面绝对相对误差分布及P90、P95位置
    # ---------------------------------------------------------
    relative_error_p90 = float(
        np.percentile(
            relative_absolute_error_percent,
            90.0,
        )
    )

    relative_error_p95 = float(
        np.percentile(
            relative_absolute_error_percent,
            95.0,
        )
    )

    # 图中显示到P99，以保留相对误差分布的尾部信息；
    # 参考线改为P90和P95。
    relative_error_p99 = float(
        np.percentile(
            relative_absolute_error_percent,
            99.0,
        )
    )

    # 在P99右侧保留少量空白。
    relative_error_plot_limit = max(
        1.05 * relative_error_p99,
        np.finfo(float).eps,
    )

    # 仅隐藏超过P99的极端相对误差，防止横轴被拉长。
    relative_error_for_plot = (
        relative_absolute_error_percent[
            relative_absolute_error_percent
            <= relative_error_p99
            ]
    )

    axes[0, 2].hist(
        relative_error_for_plot,
        bins=100,
        range=(0.0, relative_error_p99),
        color="#dd8452",
        alpha=0.85,
    )

    # P90相对误差线
    axes[0, 2].axvline(
        relative_error_p90,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"P90 = {relative_error_p90:.2f}%"
        ),
    )

    # P95相对误差线
    axes[0, 2].axvline(
        relative_error_p95,
        color="#d62728",
        linestyle="-.",
        linewidth=1.5,
        label=(
            f"P95 = {relative_error_p95:.2f}%"
        ),
    )

    axes[0, 2].set_xlim(
        0.0,
        relative_error_plot_limit,
    )

    axes[0, 2].set_xlabel(
        r"Absolute relative error "
        r"$|\Delta C_p|/|C_{p,STAR}|$ (%)"
    )
    axes[0, 2].set_ylabel("Face count")
    axes[0, 2].set_title(
        "Relative error distribution (up to P99)"
    )

    axes[0, 2].legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
    )

    # ---------------------------------------------------------
    # STAR真值与重构结果使用相同Cp色标
    # ---------------------------------------------------------
    cp_color_min = float(
        min(
            np.min(actual),
            np.min(predicted),
        )
    )

    cp_color_max = float(
        max(
            np.max(actual),
            np.max(predicted),
        )
    )

    # ---------------------------------------------------------
    # 第2行第1列：STAR-CCM+压力系数空间分布
    # ---------------------------------------------------------
    if str(cfg.PART_NAME).strip().lower() == "all_surface":
        surface_truth = plot_all_surface_by_parts(
            axis=axes[1, 0],
            table=merged,
            values=actual,
            title_prefix=r"STAR-CCM+ $C_p$: XZ projection",
            cmap="coolwarm",
            vmin=cp_color_min,
            vmax=cp_color_max,
        )
    else:
        surface_truth = plot_filled_surface_projection(
            axis=axes[1, 0],
            table=merged,
            values=actual,
            part_name=cfg.PART_NAME,
            title_prefix=r"STAR-CCM+ $C_p$",
            cmap="coolwarm",
            vmin=cp_color_min,
            vmax=cp_color_max,
        )

    colorbar_truth = figure.colorbar(
        surface_truth,
        ax=axes[1, 0],
        orientation="horizontal",
        pad=0.20,
        fraction=0.08,
        aspect=30,
    )
    colorbar_truth.set_label(r"$C_p$")

    # ---------------------------------------------------------
    # 第2行第2列：重构后压力系数空间分布
    # ---------------------------------------------------------
    if str(cfg.PART_NAME).strip().lower() == "all_surface":
        surface_prediction = plot_all_surface_by_parts(
            axis=axes[1, 1],
            table=merged,
            values=predicted,
            title_prefix=r"Reconstructed $C_p$: XZ projection",
            cmap="coolwarm",
            vmin=cp_color_min,
            vmax=cp_color_max,
        )
    else:
        surface_prediction = plot_filled_surface_projection(
            axis=axes[1, 1],
            table=merged,
            values=predicted,
            part_name=cfg.PART_NAME,
            title_prefix=r"Reconstructed $C_p$",
            cmap="coolwarm",
            vmin=cp_color_min,
            vmax=cp_color_max,
        )

    colorbar_prediction = figure.colorbar(
        surface_prediction,
        ax=axes[1, 1],
        orientation="horizontal",
        pad=0.20,
        fraction=0.08,
        aspect=30,
    )
    colorbar_prediction.set_label(r"$C_p$")

    # ---------------------------------------------------------
    # 第2行第3列：重构误差空间分布
    # ---------------------------------------------------------
    error_limit = float(
        np.percentile(
            np.abs(error),
            99.0,
        )
    )

    error_limit = max(
        error_limit,
        np.finfo(float).eps,
    )

    if str(cfg.PART_NAME).strip().lower() == "all_surface":
        surface_error = plot_all_surface_by_parts(
            axis=axes[1, 2],
            table=merged,
            values=error,
            title_prefix=(
                "Reconstruction error: XZ projection"
            ),
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
        )
    else:
        surface_error = plot_filled_surface_projection(
            axis=axes[1, 2],
            table=merged,
            values=error,
            part_name=cfg.PART_NAME,
            title_prefix="Reconstruction error",
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
        )

    # 注意：以下颜色条代码必须位于if/else外部，
    # 从而保证all_surface和单独部件均生成误差色带。
    error_colorbar_ticks = np.linspace(
        -error_limit,
        error_limit,
        7,
    )

    colorbar_error = figure.colorbar(
        surface_error,
        ax=axes[1, 2],
        orientation="horizontal",
        pad=0.20,
        fraction=0.08,
        aspect=30,
        ticks=error_colorbar_ticks,
    )

    colorbar_error.set_label(
        r"$\Delta C_p$",
        fontsize=10,
    )

    colorbar_error.ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.4f")
    )

    colorbar_error.ax.tick_params(
        axis="x",
        labelsize=8,
        pad=2,
    )

    colorbar_error.update_ticks()

    # ---------------------------------------------------------
    # 统一坐标轴样式
    # ---------------------------------------------------------
    for axis in axes.ravel():
        axis.grid(
            linestyle=":",
            alpha=0.25,
        )

    figure.suptitle(
        f"{cfg.TEST_CASE_ID}, {cfg.PART_NAME}, "
        f"{sensor_method.upper()}: "
        "sparse pressure reconstruction"
    )
    figure_path = (
            cfg.COMPARISON_ROOT
            / f"{result_prefix}_comparison.png"
    )
    figure.savefig(figure_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)

    print("\n稀疏压力重构验证结果：")
    for name, value in metrics.items():
        print(f"  {name:<32} {value}")
    print(f"对比图：{figure_path}")

if __name__ == "__main__":
    successful_methods = []
    failed_methods = []

    for sensor_method in ["deim", "qr"]:
        print("\n" + "=" * 90)
        print(
            f"开始比较{sensor_method.upper()}重构结果"
        )
        print("=" * 90)

        try:
            compare(sensor_method)
            successful_methods.append(sensor_method)
        except Exception as exc:
            failed_methods.append({
                "sensor_method": sensor_method,
                "error": repr(exc),
            })
            print(
                f"{sensor_method.upper()}比较失败：{exc}"
            )

    print("\n" + "=" * 90)
    print("DEIM和QR全场重构比较摘要")
    print("=" * 90)
    print(
        "成功方法："
        + (
            ", ".join(
                method.upper()
                for method in successful_methods
            )
            if successful_methods
            else "无"
        )
    )

    if failed_methods:
        print("失败方法：")
        for item in failed_methods:
            print(
                f"  {item['sensor_method'].upper()}: "
                f"{item['error']}"
            )
        raise RuntimeError(
            "存在重构结果比较失败，请检查上述错误"
        )

    print("失败方法：无")