# -*- coding: utf-8 -*-
"""
SUBOFF表面压力稀疏POD重构SCI对照实验。

包含实验
--------
1. 传感器数量敏感性：
       m = 3, 5, 7, 9, 12, 15, 20

2. POD模态数量敏感性：
       r = 1, ..., 9

3. 随机传感器基线：
       每个m随机重复200次
       输出均值、2.5%和97.5%经验分位数

4. 均值场基线

5. 速度线性插值/外推基线

6. 均匀布点、DEIM、QR比较

7. 噪声鲁棒性：
       0.1%、0.5%、1%、2%、5%

8. 多未见工况：
       0.75、2.5、4.5、6.5、8.5、10.5 m/s

9. 插值与外推能力比较

10. 压力积分验证：
       若topology.h5中存在表面法向量，
       比较压力积分力和力矩误差。

数据泄漏控制
------------
对每个目标工况单独构造训练集。
如果训练cases.csv中存在相同速度，则自动剔除。

重要定义
--------
噪声水平定义为相对于来流动压：

    sigma_p = epsilon * q_inf

因此Cp噪声标准差为：

    sigma_Cp = epsilon
"""

from __future__ import annotations

from pathlib import Path
import importlib
import json
import sys
import time
import warnings

import h5py
import numpy as np
import pandas as pd
from scipy.linalg import qr as scipy_qr
from scipy.spatial.distance import cdist

# =============================================================================
# 1. 配置
# =============================================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

cfg = importlib.import_module(
    "00_experiment_config"
)

EPS = np.finfo(np.float64).eps

# =============================================================================
# 2. 基础读取函数
# =============================================================================

def decode_strings(values):
    """将HDF5字符串数组转换为小写字符串。"""
    return np.asarray([
        (
            value.decode(
                "utf-8",
                errors="replace",
            )
            if isinstance(value, bytes)
            else str(value)
        ).strip().lower()
        for value in values
    ], dtype=str)

def resolve_path(path_value, base_directory):
    """解析绝对或相对路径。"""
    path = Path(str(path_value)).expanduser()

    if not path.is_absolute():
        path = Path(base_directory) / path

    return path.resolve()

def calculate_speed_from_row(row):
    """计算单个工况的来流速度模长。"""
    return float(np.sqrt(
        float(row["u_inf_m_s"]) ** 2
        + float(row["v_inf_m_s"]) ** 2
        + float(row["w_inf_m_s"]) ** 2
    ))

def load_topology():
    """
    读取Master表面拓扑并确定分析部件。

    返回的local数组只包含PART_NAME对应的面。
    """
    with h5py.File(
        cfg.TOPOLOGY_PATH,
        "r",
    ) as file:
        centroid_global = np.asarray(
            file["geometry/centroid"][()],
            dtype=np.float64,
        )

        area_global = np.asarray(
            file["geometry/face_area"][()],
            dtype=np.float64,
        )

        part_name_global = decode_strings(
            file["components/part_name"][()]
        )

        topology_hash = file.attrs.get(
            "topology_hash_sha256",
            "",
        )

        if isinstance(topology_hash, bytes):
            topology_hash = topology_hash.decode(
                "utf-8",
                errors="replace",
            )

    face_count_global = len(
        centroid_global
    )

    if cfg.PART_NAME == "all_surface":
        global_face_ids = np.arange(
            face_count_global,
            dtype=np.int64,
        )
    else:
        global_face_ids = np.flatnonzero(
            part_name_global
            == str(cfg.PART_NAME).lower()
        ).astype(np.int64)

    if len(global_face_ids) == 0:
        raise ValueError(
            f"拓扑中不存在部件：{cfg.PART_NAME}"
        )

    return {
        "centroid_global": centroid_global,
        "area_global": area_global,
        "part_name_global": part_name_global,
        "global_face_ids": global_face_ids,
        "centroid": centroid_global[
            global_face_ids
        ],
        "area": area_global[
            global_face_ids
        ],
        "part_name": part_name_global[
            global_face_ids
        ],
        "face_count_global": face_count_global,
        "face_count_part": len(
            global_face_ids
        ),
        "topology_hash": str(
            topology_hash
        ),
    }

def load_complete_field(
    path,
    topology,
):
    """
    读取一个完整压力场，并提取目标部件。

    支持
    ----
    1. HDF5:
         fields/cp_static

    2. CSV:
         global_face_id + cp_static
         global_face_id + cp_static_starccm
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as file:
            field_path = (
                f"fields/{cfg.TARGET_FIELD}"
            )

            if field_path not in file:
                raise KeyError(
                    f"{path}中不存在{field_path}"
                )

            values_global = np.asarray(
                file[field_path][()],
                dtype=np.float64,
            ).ravel()

        if len(values_global) != (
            topology["face_count_global"]
        ):
            raise ValueError(
                f"{path}场长度"
                f"{len(values_global):,}与拓扑面数"
                f"{topology['face_count_global']:,}不一致"
            )

        values = values_global[
            topology["global_face_ids"]
        ]

    elif suffix == ".csv":
        table = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        candidates = [
            cfg.TARGET_FIELD,
            "cp_static_starccm",
            "cp_static_truth",
            "cp_static_reconstructed",
        ]

        field_column = next(
            (
                name
                for name in candidates
                if name in table.columns
            ),
            None,
        )

        if field_column is None:
            raise KeyError(
                f"{path}缺少Cp列，允许列："
                f"{candidates}"
            )

        if "global_face_id" in table.columns:
            lookup = pd.Series(
                table[field_column].to_numpy(
                    dtype=np.float64
                ),
                index=table[
                    "global_face_id"
                ].to_numpy(dtype=np.int64),
            )

            try:
                values = lookup.loc[
                    topology["global_face_ids"]
                ].to_numpy(dtype=np.float64)

            except KeyError as exc:
                raise ValueError(
                    f"{path}没有覆盖部件"
                    f"{cfg.PART_NAME}的全部面"
                ) from exc
        else:
            raw = table[
                field_column
            ].to_numpy(dtype=np.float64)

            if len(raw) == topology[
                "face_count_global"
            ]:
                values = raw[
                    topology["global_face_ids"]
                ]
            elif len(raw) == topology[
                "face_count_part"
            ]:
                values = raw
            else:
                raise ValueError(
                    f"{path}的行数无法与拓扑匹配"
                )

    else:
        raise ValueError(
            f"不支持文件格式：{path.suffix}"
        )

    values = np.asarray(
        values,
        dtype=np.float64,
    ).ravel()

    if np.any(~np.isfinite(values)):
        raise ValueError(
            f"{path}目标压力场包含NaN或Inf"
        )

    return values

def load_training_data(topology):
    """读取训练cases.csv和全部训练压力快照。"""
    case_df = pd.read_csv(
        cfg.TRAIN_CASES_CSV_PATH,
        encoding="utf-8-sig",
    )

    required = {
        "case_id",
        "case_h5_path",
        *cfg.VELOCITY_COLUMNS,
    }

    missing = required - set(
        case_df.columns
    )

    if missing:
        raise KeyError(
            f"训练cases.csv缺少：{sorted(missing)}"
        )

    snapshots = []
    speeds = []

    for _, row in case_df.iterrows():
        case_path = resolve_path(
            row["case_h5_path"],
            cfg.TRAIN_CASES_CSV_PATH.parent,
        )

        snapshots.append(
            load_complete_field(
                case_path,
                topology,
            )
        )

        speeds.append(
            calculate_speed_from_row(row)
        )

    snapshots = np.stack(
        snapshots,
        axis=0,
    )

    case_df = case_df.copy()
    case_df["speed_inf_m_s"] = np.asarray(
        speeds,
        dtype=np.float64,
    )

    return case_df, snapshots

def load_unseen_data(topology):
    """读取六个未见工况。"""
    unseen_df = pd.read_csv(
        cfg.UNSEEN_CASES_CSV_PATH,
        encoding="utf-8-sig",
    )

    required = {
        "case_id",
        "truth_path",
        "split_type",
        *cfg.VELOCITY_COLUMNS,
    }

    missing = required - set(
        unseen_df.columns
    )

    if missing:
        raise KeyError(
            f"unseen_cases.csv缺少：{sorted(missing)}"
        )

    truths = []

    unseen_df = unseen_df.copy()

    speeds = []

    for _, row in unseen_df.iterrows():
        truth_path = resolve_path(
            row["truth_path"],
            cfg.UNSEEN_CASES_CSV_PATH.parent,
        )

        truths.append(
            load_complete_field(
                truth_path,
                topology,
            )
        )

        speeds.append(
            calculate_speed_from_row(row)
        )

    unseen_df[
        "speed_inf_m_s"
    ] = np.asarray(
        speeds,
        dtype=np.float64,
    )

    truths = np.stack(
        truths,
        axis=0,
    )

    return unseen_df, truths

# =============================================================================
# 3. 面积加权POD
# =============================================================================

def fit_weighted_pod(
    snapshots,
    area,
    max_modes,
):
    """
    对训练压力快照执行中心化面积加权POD。

    参数
    ----
    snapshots:
        shape=(n_cases,n_faces)

    area:
        shape=(n_faces,)

    返回
    ----
    mean_field:
        训练平均Cp场。

    basis:
        物理空间面积正交POD基，满足：

            basis.T @ diag(area) @ basis = I

    singular_values:
        POD奇异值。

    cumulative_energy:
        累计能量比例。
    """
    snapshots = np.asarray(
        snapshots,
        dtype=np.float64,
    )

    area = np.asarray(
        area,
        dtype=np.float64,
    )

    mean_field = np.mean(
        snapshots,
        axis=0,
    )

    fluctuations = (
        snapshots
        - mean_field[None, :]
    )

    sqrt_area = np.sqrt(
        np.maximum(area, EPS)
    )

    weighted_matrix = (
        fluctuations
        * sqrt_area[None, :]
    )

    # weighted_matrix.T:
    #   shape=(n_faces,n_cases)
    U, singular_values, _Vt = np.linalg.svd(
        weighted_matrix.T,
        full_matrices=False,
    )

    tolerance = (
        EPS
        * max(weighted_matrix.shape)
        * max(
            singular_values[0],
            EPS,
        )
    )

    numerical_rank = int(
        np.count_nonzero(
            singular_values > tolerance
        )
    )

    keep = min(
        int(max_modes),
        numerical_rank,
        snapshots.shape[0] - 1,
        snapshots.shape[1],
    )

    if keep <= 0:
        raise ValueError(
            "训练快照无法形成有效POD模态"
        )

    singular_values = singular_values[
        :keep
    ]

    basis = (
        U[:, :keep]
        / sqrt_area[:, None]
    )

    energy = singular_values ** 2

    cumulative_energy = np.cumsum(
        energy
    ) / max(
        np.sum(energy),
        EPS,
    )

    return {
        "mean_field": mean_field,
        "basis": basis,
        "singular_values": singular_values,
        "cumulative_energy": (
            cumulative_energy
        ),
        "numerical_rank": numerical_rank,
    }

def build_target_specific_training_set(
    train_df,
    train_snapshots,
    target_speed,
):
    """
    为一个测试速度构造无泄漏训练集。

    如果训练集中存在与测试速度相同的工况，
    则自动删除。
    """
    train_speed = train_df[
        "speed_inf_m_s"
    ].to_numpy(dtype=np.float64)

    if cfg.EXCLUDE_TEST_SPEED_FROM_TRAINING:
        keep = ~np.isclose(
            train_speed,
            float(target_speed),
            atol=cfg.SPEED_MATCH_TOLERANCE,
            rtol=0.0,
        )
    else:
        keep = np.ones(
            len(train_df),
            dtype=bool,
        )

    if np.count_nonzero(keep) < 3:
        raise ValueError(
            f"目标速度{target_speed} m/s删除同速工况后"
            "训练样本少于3个"
        )

    return (
        train_df.loc[keep]
        .reset_index(drop=True),
        train_snapshots[keep],
        int(np.count_nonzero(~keep)),
    )

# =============================================================================
# 4. 传感器选择
# =============================================================================

def select_qr_sensors(
    basis,
    sensor_count,
    xyz,
):
    """
    QR传感器选择。

    前min(m,r)个点由带列主元QR确定。
    当m>r时，QR的剩余主元已缺少明确模态意义，
    因此使用空间最远点采样补足，形成oversampled QR。
    """
    n_faces, n_modes = basis.shape

    initial_count = min(
        int(sensor_count),
        n_modes,
        n_faces,
    )

    _Q, _R, pivots = scipy_qr(
        basis.T,
        mode="economic",
        pivoting=True,
    )

    selected = list(
        np.asarray(
            pivots[:initial_count],
            dtype=np.int64,
        )
    )

    return fill_sensors_by_farthest_points(
        selected,
        sensor_count,
        xyz,
    )

def select_deim_sensors(
    basis,
    sensor_count,
    xyz,
):
    """
    标准DEIM贪心选点。

    当m>r时，DEIM最多提供r个模态驱动点，
    剩余点使用空间最远点采样补足。
    """
    n_faces, n_modes = basis.shape
    target = min(
        int(sensor_count),
        n_faces,
    )

    deim_count = min(
        target,
        n_modes,
    )

    selected = [
        int(np.argmax(
            np.abs(basis[:, 0])
        ))
    ]

    for mode_index in range(
        1,
        deim_count,
    ):
        previous = np.asarray(
            selected,
            dtype=np.int64,
        )

        active_basis = basis[
            :, :mode_index
        ]

        sampled_basis = active_basis[
            previous, :
        ]

        sampled_mode = basis[
            previous,
            mode_index,
        ]

        coefficients = np.linalg.lstsq(
            sampled_basis,
            sampled_mode,
            rcond=None,
        )[0]

        residual = (
            basis[:, mode_index]
            - active_basis @ coefficients
        )

        residual[previous] = 0.0

        selected.append(
            int(np.argmax(
                np.abs(residual)
            ))
        )

    return fill_sensors_by_farthest_points(
        selected,
        target,
        xyz,
    )

def normalize_coordinates(xyz):
    """
    对坐标进行各方向尺度标准化。

    防止SUBOFF轴向长度远大于径向尺寸，
    导致均匀布点只关注X方向。
    """
    xyz = np.asarray(
        xyz,
        dtype=np.float64,
    )

    minimum = np.min(
        xyz,
        axis=0,
    )

    extent = np.ptp(
        xyz,
        axis=0,
    )

    extent = np.where(
        extent > EPS,
        extent,
        1.0,
    )

    return (
        xyz - minimum[None, :]
    ) / extent[None, :]

def fill_sensors_by_farthest_points(
    initial_indices,
    sensor_count,
    xyz,
):
    """
    使用最远点采样补足传感器。

    每次选择距离已有传感器集合最远的面心。
    """
    xyz_normalized = normalize_coordinates(
        xyz
    )

    n_faces = len(xyz_normalized)
    target = min(
        int(sensor_count),
        n_faces,
    )

    selected = list(
        dict.fromkeys(
            int(value)
            for value in initial_indices
        )
    )

    if not selected:
        center = np.mean(
            xyz_normalized,
            axis=0,
        )

        first = int(np.argmax(
            np.linalg.norm(
                xyz_normalized
                - center[None, :],
                axis=1,
            )
        ))

        selected.append(first)

    minimum_distance = np.min(
        cdist(
            xyz_normalized,
            xyz_normalized[
                np.asarray(selected)
            ],
        ),
        axis=1,
    )

    minimum_distance[
        np.asarray(selected)
    ] = -np.inf

    while len(selected) < target:
        next_index = int(
            np.argmax(minimum_distance)
        )

        selected.append(
            next_index
        )

        new_distance = np.linalg.norm(
            xyz_normalized
            - xyz_normalized[
                next_index
            ][None, :],
            axis=1,
        )

        minimum_distance = np.minimum(
            minimum_distance,
            new_distance,
        )

        minimum_distance[
            np.asarray(selected)
        ] = -np.inf

    return np.asarray(
        selected,
        dtype=np.int64,
    )

def select_uniform_sensors(
    sensor_count,
    xyz,
):
    """纯空间最远点均匀布置基线。"""
    xyz_normalized = normalize_coordinates(
        xyz
    )

    center = np.mean(
        xyz_normalized,
        axis=0,
    )

    # 首点选距离几何中心最近的面。
    first = int(np.argmin(
        np.linalg.norm(
            xyz_normalized
            - center[None, :],
            axis=1,
        )
    ))

    return fill_sensors_by_farthest_points(
        [first],
        sensor_count,
        xyz,
    )

def select_sensors(
    method,
    basis,
    sensor_count,
    xyz,
    rng=None,
):
    """统一的传感器选择接口。"""
    method = str(method).lower()

    if method == "qr":
        return select_qr_sensors(
            basis,
            sensor_count,
            xyz,
        )

    if method == "deim":
        return select_deim_sensors(
            basis,
            sensor_count,
            xyz,
        )

    if method == "uniform":
        return select_uniform_sensors(
            sensor_count,
            xyz,
        )

    if method == "random":
        if rng is None:
            raise ValueError(
                "随机传感器需要rng"
            )

        return np.sort(
            rng.choice(
                basis.shape[0],
                size=min(
                    int(sensor_count),
                    basis.shape[0],
                ),
                replace=False,
            )
        ).astype(np.int64)

    raise ValueError(
        f"未知传感器方法：{method}"
    )

# =============================================================================
# 5. 稀疏重构
# =============================================================================

def solve_coefficients(
    sampled_basis,
    sampled_fluctuation,
):
    """
    求解POD系数。

    Ridge采用尺度自适应正则化，避免不同部件上的
    POD基数量级不同导致固定lambda不公平。
    """
    A = np.asarray(
        sampled_basis,
        dtype=np.float64,
    )

    y = np.asarray(
        sampled_fluctuation,
        dtype=np.float64,
    )

    solver = str(
        cfg.COEFFICIENT_SOLVER
    ).lower()

    if solver == "lstsq":
        coefficient = np.linalg.lstsq(
            A,
            y,
            rcond=cfg.LSTSQ_RCOND,
        )[0]

        return coefficient

    if solver == "ridge":
        gram = A.T @ A

        scale = float(
            np.trace(gram)
            / max(gram.shape[0], 1)
        )

        scale = max(scale, EPS)

        lambda_effective = (
            float(cfg.RIDGE_LAMBDA)
            * scale
        )

        return np.linalg.solve(
            gram
            + lambda_effective
            * np.eye(
                gram.shape[0],
                dtype=np.float64,
            ),
            A.T @ y,
        )

    raise ValueError(
        f"未知系数求解器："
        f"{cfg.COEFFICIENT_SOLVER}"
    )

def reconstruct_sparse_field(
    truth,
    mean_field,
    basis,
    sensor_indices,
    noise_level,
    rng,
):
    """
    使用真值场在传感器位置的Cp模拟测量，再重构全场。

    参数
    ----
    noise_level:
        Cp噪声标准差。
        因为sigma_p=noise_level*q_inf，
        所以sigma_Cp=noise_level。
    """
    sensor_indices = np.asarray(
        sensor_indices,
        dtype=np.int64,
    )

    observed_cp = truth[
        sensor_indices
    ].copy()

    if noise_level > 0.0:
        observed_cp += rng.normal(
            loc=0.0,
            scale=float(noise_level),
            size=len(sensor_indices),
        )

    sampled_mean = mean_field[
        sensor_indices
    ]

    sampled_basis = basis[
        sensor_indices,
        :,
    ]

    observed_fluctuation = (
        observed_cp - sampled_mean
    )

    coefficient = solve_coefficients(
        sampled_basis,
        observed_fluctuation,
    )

    prediction = (
        mean_field
        + basis @ coefficient
    )

    condition_number = float(
        np.linalg.cond(sampled_basis)
    )

    rank = int(
        np.linalg.matrix_rank(
            sampled_basis
        )
    )

    return {
        "prediction": prediction,
        "coefficient": coefficient,
        "condition_number": (
            condition_number
        ),
        "sampling_rank": rank,
    }

# =============================================================================
# 6. 误差指标
# =============================================================================

def calculate_metrics(
    truth,
    prediction,
    area,
):
    """计算SCI论文常用的压力场误差。"""
    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    area = np.asarray(
        area,
        dtype=np.float64,
    )

    error = prediction - truth

    truth_variation = np.sum(
        (
            truth - np.mean(truth)
        ) ** 2
    )

    if truth_variation > EPS:
        r2 = float(
            1.0
            - np.sum(error ** 2)
            / truth_variation
        )
    else:
        r2 = np.nan

    weighted_numerator = np.sum(
        area * error ** 2
    )

    weighted_denominator = np.sum(
        area * truth ** 2
    )

    correlation = np.nan

    if (
        np.std(truth) > EPS
        and np.std(prediction) > EPS
    ):
        correlation = float(
            np.corrcoef(
                truth,
                prediction,
            )[0, 1]
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
        "cp_bias": float(
            np.mean(error)
        ),
        "cp_r2": r2,
        "cp_correlation": correlation,
        "cp_area_weighted_relative_l2": float(
            np.sqrt(
                weighted_numerator
                / max(
                    weighted_denominator,
                    EPS,
                )
            )
        ),
    }

# =============================================================================
# 7. 均值场和速度插值基线
# =============================================================================

def linear_speed_prediction(
    target_speed,
    train_df,
    train_snapshots,
):
    """
    完整压力场随速度的线性插值或外推。

    插值：
        使用目标速度两侧最近的训练速度。

    外推：
        使用训练区间端部最近的两个速度。

    如果同一速度有多个训练工况，则先对同速压力场求平均。
    """
    speeds = train_df[
        "speed_inf_m_s"
    ].to_numpy(dtype=np.float64)

    unique_speeds = np.unique(
        speeds
    )

    if len(unique_speeds) < 2:
        raise ValueError(
            "线性速度基线至少需要两个训练速度"
        )

    mean_fields = {}

    for speed in unique_speeds:
        mask = np.isclose(
            speeds,
            speed,
            atol=cfg.SPEED_MATCH_TOLERANCE,
            rtol=0.0,
        )

        mean_fields[float(speed)] = np.mean(
            train_snapshots[mask],
            axis=0,
        )

    target_speed = float(
        target_speed
    )

    if target_speed < unique_speeds[0]:
        lower_speed = float(
            unique_speeds[0]
        )
        upper_speed = float(
            unique_speeds[1]
        )
        interpolation_type = "extrapolation"

    elif target_speed > unique_speeds[-1]:
        lower_speed = float(
            unique_speeds[-2]
        )
        upper_speed = float(
            unique_speeds[-1]
        )
        interpolation_type = "extrapolation"

    else:
        lower_candidates = unique_speeds[
            unique_speeds < target_speed
        ]

        upper_candidates = unique_speeds[
            unique_speeds > target_speed
        ]

        if (
            len(lower_candidates) == 0
            or len(upper_candidates) == 0
        ):
            raise ValueError(
                "无法找到目标速度两侧的训练工况"
            )

        lower_speed = float(
            lower_candidates[-1]
        )

        upper_speed = float(
            upper_candidates[0]
        )

        interpolation_type = "interpolation"

    weight = (
        target_speed - lower_speed
    ) / (
        upper_speed - lower_speed
    )

    prediction = (
        (1.0 - weight)
        * mean_fields[lower_speed]
        + weight
        * mean_fields[upper_speed]
    )

    return {
        "prediction": prediction,
        "lower_speed": lower_speed,
        "upper_speed": upper_speed,
        "upper_weight": float(weight),
        "interpolation_type": (
            interpolation_type
        ),
    }

# =============================================================================
# 8. 法向量及压力积分
# =============================================================================

def detect_normals(topology):
    """
    自动查找并归一化表面法向量。

    返回目标部件的单位法向量。
    """
    candidates = []

    if cfg.FACE_NORMAL_DATASET:
        candidates.append(
            cfg.FACE_NORMAL_DATASET
        )

    candidates.extend(
        cfg.FACE_NORMAL_CANDIDATES
    )

    candidates = list(
        dict.fromkeys(candidates)
    )

    with h5py.File(
        cfg.TOPOLOGY_PATH,
        "r",
    ) as file:
        for name in candidates:
            if name not in file:
                continue

            raw = np.asarray(
                file[name][()],
                dtype=np.float64,
            )

            if raw.shape != (
                topology["face_count_global"],
                3,
            ):
                continue

            magnitude = np.linalg.norm(
                raw,
                axis=1,
            )

            if (
                np.any(~np.isfinite(raw))
                or np.any(magnitude <= EPS)
            ):
                continue

            unit = (
                raw / magnitude[:, None]
            )

            return {
                "dataset": name,
                "normal": unit[
                    topology["global_face_ids"]
                ],
                "raw_magnitude_min": float(
                    np.min(magnitude)
                ),
                "raw_magnitude_max": float(
                    np.max(magnitude)
                ),
                "raw_magnitude_mean": float(
                    np.mean(magnitude)
                ),
            }

    return None

def calculate_pressure_integrals(
    cp,
    speed,
    topology,
    normal,
):
    """
    根据Cp积分压力力和压力力矩。

    压力表面力：
        dF = -p_gauge * n dA

    因为：
        p_gauge = q_inf Cp

    所以：
        F = -q_inf sum(Cp n A)

    力矩：
        M = sum((r-r0) x dF)

    输出
    ----
    Fx:
        轴向压力力，X方向。

    Fz:
        Z方向升力。

    My:
        绕Y轴的俯仰力矩。
    """
    cp = np.asarray(
        cp,
        dtype=np.float64,
    )

    normal = np.asarray(
        normal,
        dtype=np.float64,
    )

    q_inf = (
        0.5
        * float(cfg.RHO)
        * float(speed) ** 2
    )

    area = topology["area"]
    xyz = topology["centroid"]

    differential_force = (
        -q_inf
        * cp[:, None]
        * normal
        * area[:, None]
    )

    force = np.sum(
        differential_force,
        axis=0,
    )

    reference_point = np.asarray(
        cfg.MOMENT_REFERENCE_POINT_M,
        dtype=np.float64,
    )

    arm = (
        xyz - reference_point[None, :]
    )

    differential_moment = np.cross(
        arm,
        differential_force,
    )

    moment = np.sum(
        differential_moment,
        axis=0,
    )

    force_denominator = (
        q_inf
        * float(cfg.REFERENCE_AREA_M2)
    )

    moment_denominator = (
        force_denominator
        * float(cfg.REFERENCE_LENGTH_M)
    )

    force_coefficient = (
        force
        / max(force_denominator, EPS)
    )

    moment_coefficient = (
        moment
        / max(moment_denominator, EPS)
    )

    return {
        "dynamic_pressure_pa": q_inf,
        "force_x_n": float(force[0]),
        "force_y_n": float(force[1]),
        "force_z_n": float(force[2]),
        "moment_x_nm": float(moment[0]),
        "moment_y_nm": float(moment[1]),
        "moment_z_nm": float(moment[2]),
        "coefficient_force_x": float(
            force_coefficient[0]
        ),
        "coefficient_force_y": float(
            force_coefficient[1]
        ),
        "coefficient_force_z": float(
            force_coefficient[2]
        ),
        "coefficient_moment_x": float(
            moment_coefficient[0]
        ),
        "coefficient_moment_y": float(
            moment_coefficient[1]
        ),
        "coefficient_moment_z": float(
            moment_coefficient[2]
        ),
    }

def build_integral_error_row(
    truth_integral,
    prediction_integral,
):
    """计算压力积分量误差。"""
    result = {}

    quantity_names = [
        "force_x_n",
        "force_y_n",
        "force_z_n",
        "moment_x_nm",
        "moment_y_nm",
        "moment_z_nm",
        "coefficient_force_x",
        "coefficient_force_y",
        "coefficient_force_z",
        "coefficient_moment_x",
        "coefficient_moment_y",
        "coefficient_moment_z",
    ]

    for name in quantity_names:
        truth_value = float(
            truth_integral[name]
        )

        prediction_value = float(
            prediction_integral[name]
        )

        absolute_error = (
            prediction_value
            - truth_value
        )

        relative_error = (
            abs(absolute_error)
            / max(
                abs(truth_value),
                1.0e-12,
            )
        )

        result[
            f"{name}_truth"
        ] = truth_value

        result[
            f"{name}_prediction"
        ] = prediction_value

        result[
            f"{name}_signed_error"
        ] = absolute_error

        result[
            f"{name}_relative_error"
        ] = relative_error

    return result

# =============================================================================
# 9. 单次实验
# =============================================================================

def evaluate_sparse_method(
    *,
    case_id,
    target_speed,
    split_type,
    truth,
    pod,
    topology,
    sensor_method,
    sensor_count,
    mode_count,
    noise_level,
    repeat,
    rng,
    experiment_name,
):
    """运行并记录一次稀疏重构实验。"""
    available_modes = pod[
        "basis"
    ].shape[1]

    actual_mode_count = min(
        int(mode_count),
        available_modes,
    )

    basis = pod["basis"][
        :, :actual_mode_count
    ]

    sensor_indices = select_sensors(
        method=sensor_method,
        basis=basis,
        sensor_count=sensor_count,
        xyz=topology["centroid"],
        rng=rng,
    )

    start = time.perf_counter()

    result = reconstruct_sparse_field(
        truth=truth,
        mean_field=pod["mean_field"],
        basis=basis,
        sensor_indices=sensor_indices,
        noise_level=noise_level,
        rng=rng,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    metrics = calculate_metrics(
        truth,
        result["prediction"],
        topology["area"],
    )

    row = {
        "experiment": experiment_name,
        "case_id": case_id,
        "speed_m_s": target_speed,
        "split_type": split_type,
        "part_name": cfg.PART_NAME,
        "method": sensor_method,
        "sensor_count": len(
            sensor_indices
        ),
        "requested_sensor_count": (
            sensor_count
        ),
        "mode_count": actual_mode_count,
        "requested_mode_count": (
            mode_count
        ),
        "noise_level": noise_level,
        "noise_percent": (
            100.0 * noise_level
        ),
        "repeat": repeat,
        "sampling_rank": result[
            "sampling_rank"
        ],
        "condition_number": result[
            "condition_number"
        ],
        "runtime_ms": elapsed_ms,
    }

    row.update(metrics)

    return (
        row,
        result["prediction"],
        sensor_indices,
    )

# =============================================================================
# 10. 主实验流程
# =============================================================================

def run():
    print("=" * 100)
    print("SUBOFF稀疏POD压力重构SCI对照实验")
    print("=" * 100)

    topology = load_topology()

    train_df, train_snapshots = (
        load_training_data(topology)
    )

    unseen_df, unseen_truths = (
        load_unseen_data(topology)
    )

    normal_information = detect_normals(
        topology
    )

    print(
        f"目标部件：{cfg.PART_NAME}"
    )
    print(
        f"目标部件面数："
        f"{topology['face_count_part']:,}"
    )
    print(
        f"训练工况数：{len(train_df)}"
    )
    print(
        f"未见工况数：{len(unseen_df)}"
    )

    if normal_information is None:
        print(
            "未找到表面法向量："
            "将跳过压力积分实验。"
        )
    else:
        print(
            "法向量字段："
            f"{normal_information['dataset']}"
        )

    master_rng = np.random.default_rng(
        cfg.RANDOM_SEED
    )

    all_rows = []
    random_rows = []
    integral_rows = []
    baseline_rows = []
    pod_rows = []

    # =========================================================================
    # 每一个未见工况独立运行
    # =========================================================================
    for case_index, unseen_row in (
        unseen_df.iterrows()
    ):
        case_id = str(
            unseen_row["case_id"]
        )

        target_speed = float(
            unseen_row["speed_inf_m_s"]
        )

        split_type = str(
            unseen_row["split_type"]
        ).strip().lower()

        truth = unseen_truths[
            case_index
        ]

        print("\n" + "-" * 100)
        print(
            f"测试工况：{case_id}, "
            f"U={target_speed:.6f} m/s, "
            f"type={split_type}"
        )

        (
            local_train_df,
            local_train_snapshots,
            excluded_count,
        ) = build_target_specific_training_set(
            train_df,
            train_snapshots,
            target_speed,
        )

        print(
            f"训练工况：{len(local_train_df)}，"
            f"同速剔除：{excluded_count}"
        )

        pod = fit_weighted_pod(
            local_train_snapshots,
            topology["area"],
            cfg.MAX_POD_MODES,
        )

        print(
            f"POD数值秩："
            f"{pod['numerical_rank']}，"
            f"保留模态：{pod['basis'].shape[1]}"
        )

        # 保存POD能量信息。
        for mode_index, cumulative in enumerate(
            pod["cumulative_energy"],
            start=1,
        ):
            pod_rows.append({
                "case_id": case_id,
                "speed_m_s": target_speed,
                "part_name": cfg.PART_NAME,
                "mode_index": mode_index,
                "singular_value": pod[
                    "singular_values"
                ][mode_index - 1],
                "cumulative_energy": (
                    cumulative
                ),
                "excluded_same_speed_count": (
                    excluded_count
                ),
            })

        # ---------------------------------------------------------------------
        # 10.1 均值场基线
        # ---------------------------------------------------------------------
        if cfg.RUN_MEAN_FIELD_BASELINE:
            prediction = pod[
                "mean_field"
            ].copy()

            metrics = calculate_metrics(
                truth,
                prediction,
                topology["area"],
            )

            row = {
                "experiment": "baseline",
                "case_id": case_id,
                "speed_m_s": target_speed,
                "split_type": split_type,
                "part_name": cfg.PART_NAME,
                "method": "mean_field",
                "sensor_count": 0,
                "requested_sensor_count": 0,
                "mode_count": 0,
                "requested_mode_count": 0,
                "noise_level": 0.0,
                "noise_percent": 0.0,
                "repeat": 0,
                "sampling_rank": 0,
                "condition_number": np.nan,
                "runtime_ms": 0.0,
            }

            row.update(metrics)
            all_rows.append(row)
            baseline_rows.append(row.copy())

            if normal_information is not None:
                truth_integral = (
                    calculate_pressure_integrals(
                        truth,
                        target_speed,
                        topology,
                        normal_information[
                            "normal"
                        ],
                    )
                )

                pred_integral = (
                    calculate_pressure_integrals(
                        prediction,
                        target_speed,
                        topology,
                        normal_information[
                            "normal"
                        ],
                    )
                )

                integral_row = {
                    "case_id": case_id,
                    "speed_m_s": target_speed,
                    "split_type": split_type,
                    "part_name": cfg.PART_NAME,
                    "method": "mean_field",
                    "sensor_count": 0,
                    "mode_count": 0,
                    "noise_level": 0.0,
                }

                integral_row.update(
                    build_integral_error_row(
                        truth_integral,
                        pred_integral,
                    )
                )

                integral_rows.append(
                    integral_row
                )

        # ---------------------------------------------------------------------
        # 10.2 速度线性插值/外推基线
        # ---------------------------------------------------------------------
        if cfg.RUN_LINEAR_SPEED_BASELINE:
            interpolation = (
                linear_speed_prediction(
                    target_speed,
                    local_train_df,
                    local_train_snapshots,
                )
            )

            prediction = interpolation[
                "prediction"
            ]

            metrics = calculate_metrics(
                truth,
                prediction,
                topology["area"],
            )

            row = {
                "experiment": "baseline",
                "case_id": case_id,
                "speed_m_s": target_speed,
                "split_type": split_type,
                "part_name": cfg.PART_NAME,
                "method": (
                    "linear_speed_"
                    + interpolation[
                        "interpolation_type"
                    ]
                ),
                "sensor_count": 0,
                "requested_sensor_count": 0,
                "mode_count": 0,
                "requested_mode_count": 0,
                "noise_level": 0.0,
                "noise_percent": 0.0,
                "repeat": 0,
                "sampling_rank": 0,
                "condition_number": np.nan,
                "runtime_ms": 0.0,
                "lower_training_speed_m_s": (
                    interpolation[
                        "lower_speed"
                    ]
                ),
                "upper_training_speed_m_s": (
                    interpolation[
                        "upper_speed"
                    ]
                ),
                "upper_speed_weight": (
                    interpolation[
                        "upper_weight"
                    ]
                ),
            }

            row.update(metrics)
            all_rows.append(row)
            baseline_rows.append(row.copy())

            if normal_information is not None:
                truth_integral = (
                    calculate_pressure_integrals(
                        truth,
                        target_speed,
                        topology,
                        normal_information[
                            "normal"
                        ],
                    )
                )

                pred_integral = (
                    calculate_pressure_integrals(
                        prediction,
                        target_speed,
                        topology,
                        normal_information[
                            "normal"
                        ],
                    )
                )

                integral_row = {
                    "case_id": case_id,
                    "speed_m_s": target_speed,
                    "split_type": split_type,
                    "part_name": cfg.PART_NAME,
                    "method": row["method"],
                    "sensor_count": 0,
                    "mode_count": 0,
                    "noise_level": 0.0,
                }

                integral_row.update(
                    build_integral_error_row(
                        truth_integral,
                        pred_integral,
                    )
                )

                integral_rows.append(
                    integral_row
                )

        # ---------------------------------------------------------------------
        # 10.3 传感器数量敏感性
        #
        # 固定请求模态数为9。
        # 对m<r的情况使用Ridge求解。
        # ---------------------------------------------------------------------
        for sensor_count in cfg.SENSOR_COUNTS:
            for method in (
                cfg.DETERMINISTIC_SENSOR_METHODS
            ):
                row, prediction, _sensor_ids = (
                    evaluate_sparse_method(
                        case_id=case_id,
                        target_speed=target_speed,
                        split_type=split_type,
                        truth=truth,
                        pod=pod,
                        topology=topology,
                        sensor_method=method,
                        sensor_count=sensor_count,
                        mode_count=(
                            cfg.MAX_POD_MODES
                        ),
                        noise_level=0.0,
                        repeat=0,
                        rng=master_rng,
                        experiment_name=(
                            "sensor_count_sensitivity"
                        ),
                    )
                )

                all_rows.append(row)

        # ---------------------------------------------------------------------
        # 10.4 POD模态数量敏感性
        # ---------------------------------------------------------------------
        for mode_count in cfg.POD_MODE_COUNTS:
            if mode_count > pod[
                "basis"
            ].shape[1]:
                continue

            for method in (
                cfg.DETERMINISTIC_SENSOR_METHODS
            ):
                row, _prediction, _sensor_ids = (
                    evaluate_sparse_method(
                        case_id=case_id,
                        target_speed=target_speed,
                        split_type=split_type,
                        truth=truth,
                        pod=pod,
                        topology=topology,
                        sensor_method=method,
                        sensor_count=(
                            cfg.MODE_SENSITIVITY_SENSOR_COUNT
                        ),
                        mode_count=mode_count,
                        noise_level=0.0,
                        repeat=0,
                        rng=master_rng,
                        experiment_name=(
                            "pod_mode_sensitivity"
                        ),
                    )
                )

                all_rows.append(row)

        # ---------------------------------------------------------------------
        # 10.5 随机传感器基线
        # ---------------------------------------------------------------------
        if cfg.RUN_RANDOM_SENSOR_BASELINE:
            for sensor_count in (
                cfg.SENSOR_COUNTS
            ):
                for repeat in range(
                    cfg.RANDOM_SENSOR_REPEATS
                ):
                    # 为每一次试验生成独立但可复现的随机数流。
                    child_seed = int(
                        master_rng.integers(
                            0,
                            np.iinfo(
                                np.int32
                            ).max,
                        )
                    )

                    child_rng = (
                        np.random.default_rng(
                            child_seed
                        )
                    )

                    row, _prediction, _ids = (
                        evaluate_sparse_method(
                            case_id=case_id,
                            target_speed=target_speed,
                            split_type=split_type,
                            truth=truth,
                            pod=pod,
                            topology=topology,
                            sensor_method="random",
                            sensor_count=sensor_count,
                            mode_count=(
                                cfg.MAX_POD_MODES
                            ),
                            noise_level=0.0,
                            repeat=repeat + 1,
                            rng=child_rng,
                            experiment_name=(
                                "random_sensor_baseline"
                            ),
                        )
                    )

                    row["random_seed"] = (
                        child_seed
                    )

                    random_rows.append(row)
                    all_rows.append(row)

        # ---------------------------------------------------------------------
        # 10.6 噪声鲁棒性
        # ---------------------------------------------------------------------
        for method in (
            cfg.DETERMINISTIC_SENSOR_METHODS
        ):
            for noise_level in (
                cfg.NOISE_LEVELS
            ):
                for repeat in range(
                    cfg.NOISE_REPEATS
                ):
                    child_seed = int(
                        master_rng.integers(
                            0,
                            np.iinfo(
                                np.int32
                            ).max,
                        )
                    )

                    child_rng = (
                        np.random.default_rng(
                            child_seed
                        )
                    )

                    row, _prediction, _ids = (
                        evaluate_sparse_method(
                            case_id=case_id,
                            target_speed=target_speed,
                            split_type=split_type,
                            truth=truth,
                            pod=pod,
                            topology=topology,
                            sensor_method=method,
                            sensor_count=(
                                cfg.NOISE_SENSOR_COUNT
                            ),
                            mode_count=(
                                cfg.NOISE_MODE_COUNT
                            ),
                            noise_level=(
                                noise_level
                            ),
                            repeat=repeat + 1,
                            rng=child_rng,
                            experiment_name=(
                                "noise_robustness"
                            ),
                        )
                    )

                    row["random_seed"] = (
                        child_seed
                    )

                    all_rows.append(row)

        # ---------------------------------------------------------------------
        # 10.7 主设置重构、压力积分和完整预测保存
        # ---------------------------------------------------------------------
        main_row, main_prediction, (
            main_sensor_indices
        ) = evaluate_sparse_method(
            case_id=case_id,
            target_speed=target_speed,
            split_type=split_type,
            truth=truth,
            pod=pod,
            topology=topology,
            sensor_method=(
                cfg.MAIN_SENSOR_METHOD
            ),
            sensor_count=(
                cfg.MAIN_SENSOR_COUNT
            ),
            mode_count=(
                cfg.MAIN_MODE_COUNT
            ),
            noise_level=0.0,
            repeat=0,
            rng=master_rng,
            experiment_name="main_model",
        )

        all_rows.append(main_row)

        if cfg.SAVE_MAIN_PREDICTIONS:
            prediction_table = pd.DataFrame({
                "global_face_id": topology[
                    "global_face_ids"
                ],
                "x_m": topology[
                    "centroid"
                ][:, 0],
                "y_m": topology[
                    "centroid"
                ][:, 1],
                "z_m": topology[
                    "centroid"
                ][:, 2],
                "face_area_m2": topology[
                    "area"
                ],
                "cp_static_truth": truth,
                "cp_static_prediction": (
                    main_prediction
                ),
                "cp_error": (
                    main_prediction - truth
                ),
            })

            prediction_table.to_csv(
                cfg.PREDICTION_ROOT
                / (
                    f"{case_id}_"
                    f"{cfg.PART_NAME}_"
                    f"{cfg.MAIN_SENSOR_METHOD}_"
                    "main_prediction.csv"
                ),
                index=False,
                encoding="utf-8-sig",
                float_format=(
                    cfg.CSV_FLOAT_FORMAT
                ),
            )

            sensor_table = pd.DataFrame({
                "sensor_rank": np.arange(
                    len(main_sensor_indices)
                ) + 1,
                "local_face_index": (
                    main_sensor_indices
                ),
                "global_face_id": topology[
                    "global_face_ids"
                ][main_sensor_indices],
                "x_m": topology[
                    "centroid"
                ][main_sensor_indices, 0],
                "y_m": topology[
                    "centroid"
                ][main_sensor_indices, 1],
                "z_m": topology[
                    "centroid"
                ][main_sensor_indices, 2],
                "cp_static_truth": truth[
                    main_sensor_indices
                ],
            })

            sensor_table.to_csv(
                cfg.PREDICTION_ROOT
                / (
                    f"{case_id}_"
                    f"{cfg.PART_NAME}_"
                    f"{cfg.MAIN_SENSOR_METHOD}_"
                    "main_sensors.csv"
                ),
                index=False,
                encoding="utf-8-sig",
                float_format=(
                    cfg.CSV_FLOAT_FORMAT
                ),
            )

        if normal_information is not None:
            truth_integral = (
                calculate_pressure_integrals(
                    truth,
                    target_speed,
                    topology,
                    normal_information[
                        "normal"
                    ],
                )
            )

            prediction_integral = (
                calculate_pressure_integrals(
                    main_prediction,
                    target_speed,
                    topology,
                    normal_information[
                        "normal"
                    ],
                )
            )

            integral_row = {
                "case_id": case_id,
                "speed_m_s": target_speed,
                "split_type": split_type,
                "part_name": cfg.PART_NAME,
                "method": (
                    cfg.MAIN_SENSOR_METHOD
                ),
                "sensor_count": (
                    cfg.MAIN_SENSOR_COUNT
                ),
                "mode_count": (
                    cfg.MAIN_MODE_COUNT
                ),
                "noise_level": 0.0,
            }

            integral_row.update(
                build_integral_error_row(
                    truth_integral,
                    prediction_integral,
                )
            )

            integral_rows.append(
                integral_row
            )
        # =====================================================================
        # 【新增】10.7.1 补充DEIM方法的压力积分计算
        # =====================================================================
        if normal_information is not None:
            # 用DEIM方法重构
            deim_row, deim_prediction, (
                deim_sensor_indices
            ) = evaluate_sparse_method(
                case_id=case_id,
                target_speed=target_speed,
                split_type=split_type,
                truth=truth,
                pod=pod,
                topology=topology,
                sensor_method='deim',  # 使用DEIM
                sensor_count=(
                    cfg.MAIN_SENSOR_COUNT
                ),
                mode_count=(
                    cfg.MAIN_MODE_COUNT
                ),
                noise_level=0.0,
                repeat=0,
                rng=master_rng,
                experiment_name="main_model_deim",
            )

            # 计算DEIM的压力积分
            deim_prediction_integral = (
                calculate_pressure_integrals(
                    deim_prediction,
                    target_speed,
                    topology,
                    normal_information[
                        "normal"
                    ],
                )
            )

            deim_integral_row = {
                "case_id": case_id,
                "speed_m_s": target_speed,
                "split_type": split_type,
                "part_name": cfg.PART_NAME,
                "method": 'deim',
                "sensor_count": (
                    cfg.MAIN_SENSOR_COUNT
                ),
                "mode_count": (
                    cfg.MAIN_MODE_COUNT
                ),
                "noise_level": 0.0,
            }

            deim_integral_row.update(
                build_integral_error_row(
                    truth_integral,  # 真值积分已经在上面计算过了
                    deim_prediction_integral,
                )
            )

            integral_rows.append(
                deim_integral_row
            )

            print(f"  ✓ 已补充DEIM方法的压力积分")

    # =========================================================================
    # 11. 保存原始长表
    # =========================================================================
    all_metrics = pd.DataFrame(
        all_rows
    )

    all_metrics_path = (
        cfg.TABLE_ROOT
        / "all_metrics.csv"
    )

    all_metrics.to_csv(
        all_metrics_path,
        index=False,
        encoding="utf-8-sig",
        float_format=cfg.CSV_FLOAT_FORMAT,
    )

    pd.DataFrame(
        pod_rows
    ).to_csv(
        cfg.TABLE_ROOT
        / "pod_energy_by_test_case.csv",
        index=False,
        encoding="utf-8-sig",
        float_format=cfg.CSV_FLOAT_FORMAT,
    )

    pd.DataFrame(
        baseline_rows
    ).to_csv(
        cfg.TABLE_ROOT
        / "baseline_comparison.csv",
        index=False,
        encoding="utf-8-sig",
        float_format=cfg.CSV_FLOAT_FORMAT,
    )

    # =========================================================================
    # 12. 拆分各实验表
    # =========================================================================
    for experiment_name, file_name in [
        (
            "sensor_count_sensitivity",
            "sensor_count_sensitivity.csv",
        ),
        (
            "pod_mode_sensitivity",
            "pod_mode_sensitivity.csv",
        ),
        (
            "noise_robustness",
            "noise_robustness_raw.csv",
        ),
        (
            "main_model",
            "multi_condition_generalization.csv",
        ),
    ]:
        selected = all_metrics.loc[
            all_metrics["experiment"]
            == experiment_name
        ].copy()

        selected.to_csv(
            cfg.TABLE_ROOT / file_name,
            index=False,
            encoding="utf-8-sig",
            float_format=cfg.CSV_FLOAT_FORMAT,
        )

    # =========================================================================
    # 13. 随机传感器95%区间
    # =========================================================================
    if random_rows:
        random_df = pd.DataFrame(
            random_rows
        )

        if cfg.SAVE_RANDOM_RAW_RESULTS:
            random_df.to_csv(
                cfg.TABLE_ROOT
                / "random_sensor_raw.csv",
                index=False,
                encoding="utf-8-sig",
                float_format=(
                    cfg.CSV_FLOAT_FORMAT
                ),
            )

        metric_columns = [
            "cp_rmse",
            "cp_mae",
            "cp_r2",
            "cp_area_weighted_relative_l2",
            "condition_number",
        ]

        grouped_rows = []

        group_columns = [
            "case_id",
            "speed_m_s",
            "split_type",
            "part_name",
            "sensor_count",
            "mode_count",
        ]

        for keys, group in random_df.groupby(
            group_columns,
            dropna=False,
        ):
            result = dict(
                zip(group_columns, keys)
            )

            result[
                "repeat_count"
            ] = len(group)

            for metric in metric_columns:
                values = pd.to_numeric(
                    group[metric],
                    errors="coerce",
                ).to_numpy(dtype=np.float64)

                values = values[
                    np.isfinite(values)
                ]

                if len(values) == 0:
                    result[
                        f"{metric}_mean"
                    ] = np.nan
                    result[
                        f"{metric}_ci95_lower"
                    ] = np.nan
                    result[
                        f"{metric}_ci95_upper"
                    ] = np.nan
                    continue

                result[
                    f"{metric}_mean"
                ] = float(np.mean(values))

                result[
                    f"{metric}_ci95_lower"
                ] = float(
                    np.percentile(
                        values,
                        2.5,
                    )
                )

                result[
                    f"{metric}_ci95_upper"
                ] = float(
                    np.percentile(
                        values,
                        97.5,
                    )
                )

            grouped_rows.append(result)

        pd.DataFrame(
            grouped_rows
        ).to_csv(
            cfg.TABLE_ROOT
            / "random_sensor_statistics.csv",
            index=False,
            encoding="utf-8-sig",
            float_format=cfg.CSV_FLOAT_FORMAT,
        )

    # =========================================================================
    # 14. 噪声统计
    # =========================================================================
    noise_df = all_metrics.loc[
        all_metrics["experiment"]
        == "noise_robustness"
    ].copy()

    if len(noise_df) > 0:
        noise_group_columns = [
            "case_id",
            "speed_m_s",
            "split_type",
            "part_name",
            "method",
            "sensor_count",
            "mode_count",
            "noise_level",
            "noise_percent",
        ]

        noise_summary = (
            noise_df.groupby(
                noise_group_columns,
                as_index=False,
                dropna=False,
            )
            .agg(
                repeat_count=(
                    "repeat",
                    "count",
                ),
                cp_rmse_mean=(
                    "cp_rmse",
                    "mean",
                ),
                cp_rmse_std=(
                    "cp_rmse",
                    "std",
                ),
                cp_rmse_ci95_lower=(
                    "cp_rmse",
                    lambda x: np.percentile(
                        x,
                        2.5,
                    ),
                ),
                cp_rmse_ci95_upper=(
                    "cp_rmse",
                    lambda x: np.percentile(
                        x,
                        97.5,
                    ),
                ),
                relative_l2_mean=(
                    "cp_area_weighted_relative_l2",
                    "mean",
                ),
                relative_l2_ci95_lower=(
                    "cp_area_weighted_relative_l2",
                    lambda x: np.percentile(
                        x,
                        2.5,
                    ),
                ),
                relative_l2_ci95_upper=(
                    "cp_area_weighted_relative_l2",
                    lambda x: np.percentile(
                        x,
                        97.5,
                    ),
                ),
            )
        )

        noise_summary.to_csv(
            cfg.TABLE_ROOT
            / "noise_robustness_statistics.csv",
            index=False,
            encoding="utf-8-sig",
            float_format=cfg.CSV_FLOAT_FORMAT,
        )

    # =========================================================================
    # 15. 压力积分结果
    # =========================================================================
    if normal_information is not None:
        pd.DataFrame(
            integral_rows
        ).to_csv(
            cfg.TABLE_ROOT
            / "pressure_integral_errors.csv",
            index=False,
            encoding="utf-8-sig",
            float_format=cfg.CSV_FLOAT_FORMAT,
        )

        integral_status = {
            "status": "success",
            "normal_dataset": (
                normal_information[
                    "dataset"
                ]
            ),
            "reference_area_m2": (
                cfg.REFERENCE_AREA_M2
            ),
            "reference_length_m": (
                cfg.REFERENCE_LENGTH_M
            ),
            "moment_reference_point_m": (
                list(
                    cfg.MOMENT_REFERENCE_POINT_M
                )
            ),
            "note": (
                "法向量被归一化后用于压力积分；"
                "必须人工确认其方向为流体域指向艇体还是艇体指向流体域。"
                "若方向相反，所有力和力矩符号会反转，"
                "但重构相对误差不受整体符号反转影响。"
            ),
        }
    else:
        pd.DataFrame([{
            "status": "skipped",
            "reason": (
                "topology.h5中没有检测到"
                "shape=(n_faces,3)的表面法向量"
            ),
        }]).to_csv(
            cfg.TABLE_ROOT
            / "pressure_integral_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

        integral_status = {
            "status": "skipped",
            "reason": (
                "face normal not found"
            ),
        }

    # =========================================================================
    # 16. 保存实验元数据
    # =========================================================================
    metadata = {
        "part_name": cfg.PART_NAME,
        "target_field": cfg.TARGET_FIELD,
        "training_case_count": len(
            train_df
        ),
        "unseen_case_count": len(
            unseen_df
        ),
        "part_face_count": topology[
            "face_count_part"
        ],
        "topology_hash": topology[
            "topology_hash"
        ],
        "sensor_counts": (
            cfg.SENSOR_COUNTS
        ),
        "pod_mode_counts": (
            cfg.POD_MODE_COUNTS
        ),
        "random_sensor_repeats": (
            cfg.RANDOM_SENSOR_REPEATS
        ),
        "noise_levels": (
            cfg.NOISE_LEVELS
        ),
        "noise_repeats": (
            cfg.NOISE_REPEATS
        ),
        "coefficient_solver": (
            cfg.COEFFICIENT_SOLVER
        ),
        "ridge_lambda": (
            cfg.RIDGE_LAMBDA
        ),
        "random_seed": (
            cfg.RANDOM_SEED
        ),
        "pressure_integral": (
            integral_status
        ),
    }

    (
        cfg.RESULT_ROOT
        / "experiment_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("全部数值实验完成")
    print("=" * 100)
    print(f"总指标表：{all_metrics_path}")
    print(f"表格目录：{cfg.TABLE_ROOT}")
    print(f"预测目录：{cfg.PREDICTION_ROOT}")
    print("=" * 100)

if __name__ == "__main__":
    run()