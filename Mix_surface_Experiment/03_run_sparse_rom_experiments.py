# -*- coding: utf-8 -*-
"""
SUBOFF表面压力工程测压孔稀疏POD重构实验。

当前主模型
----------
1. 工程测压孔：
       all_surface_practical_sensors.csv
       共126个测点

2. POD模态数：
       70阶

3. 主重构：
       126个STAR测压孔 + 70阶POD

4. 参数预测：
       (u,v,w) -> POD系数
       RBF核岭模型

5. 基线：
       均值场
       速度线性插值/外推
       随机测点数值基线

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
# 链路B：CGNS提取的工程传感器矩阵
# =============================================================================

EXTRACTED_SENSOR_MATRIX_PATH = (
    Path(
        getattr(
            cfg,
            "EXTRACTED_SENSOR_MATRIX_PATH",
            cfg.DATASET_ROOT
            / "sensor_check"
            / "sensor_measurements_matrix.csv",
        )
    )
    .expanduser()
    .resolve()
)

# True：优先使用01_extract_sensor_values_from_cfd.py生成的矩阵
# False：使用原有逐工况PracticalSensorFromStar_*.csv流程
USE_EXTRACTED_SENSOR_MATRIX = bool(
    getattr(
        cfg,
        "USE_EXTRACTED_SENSOR_MATRIX",
        True,
    )
)
# 当前测试工况的STAR传感器数据。
# 在run()的每个测试工况开始时更新。
CURRENT_STAR_SENSOR_DATA = {}


# =============================================================================
# 2. 基础读取函数
# =============================================================================

# =============================================================================
# STAR-CCM+传感器读取与坐标匹配
# =============================================================================

STAR_SENSOR_COLUMNS = [
    "Absolute Pressure (Pa)",
    "Absolute Total Pressure (Pa)",
    "Velocity: Magnitude (m/s)",
    "Velocity[i] (m/s)",
    "Velocity[j] (m/s)",
    "Velocity[k] (m/s)",
    "Vorticity: Magnitude (/s)",
    "Vorticity[i] (/s)",
    "Vorticity[j] (/s)",
    "Vorticity[k] (/s)",
    "X (m)",
    "Y (m)",
    "Z (m)",
]

def read_star_sensor_csv(path):
    """读取STAR-CCM+传感器CSV。"""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"传感器文件不存在：{path}"
        )

    table = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    table.columns = [
        str(column).strip()
        for column in table.columns
    ]

    missing = set(STAR_SENSOR_COLUMNS) - set(table.columns)
    if missing:
        raise KeyError(
            f"传感器文件{path.name}缺少字段：{sorted(missing)}"
        )

    for column in STAR_SENSOR_COLUMNS:
        table[column] = pd.to_numeric(
            table[column],
            errors="coerce",
        )

    if table[STAR_SENSOR_COLUMNS].isna().any().any():
        raise ValueError(
            f"传感器文件{path.name}存在NaN或非数值数据"
        )

    return table

def format_extracted_case_tag(u, v, w):
    """
    与01_extract_sensor_values_from_cfd.py中的
    format_velocity_tag保持一致。

    例如：
        (1.5, 0, 0) -> u1p5v0w0
        (-0.5, 1, 2) -> um0p5v1w2
    """
    def fmt(value):
        text = f"{float(value):.10g}"
        return text.replace("-", "m").replace(".", "p")

    return (
        f"u{fmt(u)}"
        f"v{fmt(v)}"
        f"w{fmt(w)}"
    )

def load_extracted_sensor_matrix(path):
    """
    读取01_extract_sensor_values_from_cfd.py生成的
    sensor_measurements_matrix.csv。

    文件格式：
        tap_index,tap_id,global_face_id,
        u1p5v0w0,u-1p5v0w0,...

    返回：
        {
            "table": 原始DataFrame,
            "case_columns": 工况列名列表,
            "tap_ids": tap_id数组,
            "global_face_ids": global_face_id数组,
        }
    """
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            "未找到CGNS传感器矩阵："
            f"{path}\n"
            "请先运行：python 01_extract_sensor_values_from_cfd.py"
        )

    table = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    table.columns = [
        str(column).strip()
        for column in table.columns
    ]

    required = {
        "tap_index",
        "tap_id",
        "global_face_id",
    }

    missing = required - set(table.columns)

    if missing:
        raise KeyError(
            f"{path.name}缺少传感器索引字段：{sorted(missing)}"
        )

    index_columns = [
        "tap_index",
        "tap_id",
        "global_face_id",
    ]

    case_columns = [
        column
        for column in table.columns
        if column not in index_columns
    ]

    if not case_columns:
        raise ValueError(
            f"{path.name}中没有任何工况传感器列"
        )

    for column in index_columns:
        table[column] = pd.to_numeric(
            table[column],
            errors="raise",
        )

    table["tap_index"] = table[
        "tap_index"
    ].astype(np.int64)

    table["tap_id"] = table[
        "tap_id"
    ].astype(np.int64)

    table["global_face_id"] = table[
        "global_face_id"
    ].astype(np.int64)

    if table["tap_id"].duplicated().any():
        raise ValueError(
            f"{path.name}中tap_id重复"
        )

    if table["global_face_id"].duplicated().any():
        raise ValueError(
            f"{path.name}中global_face_id重复"
        )

    for column in case_columns:
        table[column] = pd.to_numeric(
            table[column],
            errors="coerce",
        )

    values = table[case_columns].to_numpy(
        dtype=np.float64
    )

    if np.any(~np.isfinite(values)):
        raise ValueError(
            f"{path.name}中存在NaN或Inf"
        )

    table = table.sort_values(
        "tap_id"
    ).reset_index(drop=True)

    print(
        f"已加载CGNS传感器矩阵：{path}"
    )
    print(
        f"传感器数量：{len(table)}；"
        f"工况数量：{len(case_columns)}"
    )

    return {
        "table": table,
        "case_columns": case_columns,
        "tap_ids": table["tap_id"].to_numpy(
            dtype=np.int64
        ),
        "global_face_ids": table[
            "global_face_id"
        ].to_numpy(dtype=np.int64),
        "path": str(path),
    }

def load_case_sensor_data_from_matrix(
    unseen_row,
    method,
    topology,
    sensor_matrix,
):
    """
    从sensor_measurements_matrix.csv读取单个工况的传感器Cp。

    当前链路B主要支持practical工程测点。
    矩阵中的传感器顺序按照tap_id排序，与
    all_surface_practical_sensors.csv保持一致。
    """
    method = str(method).lower()

    if method != "practical":
        raise ValueError(
            "CGNS传感器矩阵当前只支持practical工程测点；"
            f"收到：{method}"
        )

    velocity = np.asarray([
        float(unseen_row["u_inf_m_s"]),
        float(unseen_row["v_inf_m_s"]),
        float(unseen_row["w_inf_m_s"]),
    ], dtype=np.float64)

    case_tag = format_extracted_case_tag(
        velocity[0],
        velocity[1],
        velocity[2],
    )

    matrix_table = sensor_matrix["table"]
    case_columns = sensor_matrix["case_columns"]

    if case_tag not in case_columns:
        raise KeyError(
            f"传感器矩阵中没有工况列：{case_tag}\n"
            f"当前矩阵前20个工况列："
            f"{case_columns[:20]}"
        )

    sensor_definition = load_sensor_definition(
        "practical"
    )

    definition_tap_ids = sensor_definition[
        "tap_id"
    ].to_numpy(dtype=np.int64)

    matrix_tap_ids = matrix_table[
        "tap_id"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        matrix_tap_ids,
        definition_tap_ids,
    ):
        raise ValueError(
            "传感器矩阵tap_id顺序或内容与"
            "all_surface_practical_sensors.csv不一致"
        )

    observed_cp = matrix_table[
        case_tag
    ].to_numpy(dtype=np.float64)

    if len(observed_cp) != len(
        sensor_definition
    ):
        raise ValueError(
            f"{case_tag}传感器数量不一致："
            f"matrix={len(observed_cp)}，"
            f"definition={len(sensor_definition)}"
        )

    if np.any(~np.isfinite(observed_cp)):
        raise ValueError(
            f"{case_tag}的传感器Cp包含NaN或Inf"
        )

    definition_global_ids = (
        sensor_definition[
            "global_face_id"
        ].to_numpy(dtype=np.int64)
    )

    matrix_global_ids = matrix_table[
        "global_face_id"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        matrix_global_ids,
        definition_global_ids,
    ):
        raise ValueError(
            f"{case_tag}的global_face_id与传感器定义不一致"
        )

    topology_global_ids = topology[
        "global_face_ids"
    ]

    global_to_local = {
        int(global_id): local_index
        for local_index, global_id in enumerate(
            topology_global_ids
        )
    }

    missing_ids = [
        int(global_id)
        for global_id in definition_global_ids
        if int(global_id) not in global_to_local
    ]

    if missing_ids:
        raise ValueError(
            f"{case_tag}的传感器global_face_id不属于当前分析部件："
            f"{missing_ids[:10]}"
        )

    topology_indices = np.asarray([
        global_to_local[int(global_id)]
        for global_id in definition_global_ids
    ], dtype=np.int64)

    return {
        "cp": observed_cp,
        "topology_indices": topology_indices,
        "global_face_ids": definition_global_ids,
        "sensor_table": None,
        "sensor_definition": sensor_definition,
        "sensor_path": sensor_matrix["path"],
        "case_tag": case_tag,
        "observation_source": "cgns_extracted_sensor_matrix",
        "max_star_sensor_distance_m": np.nan,
        "max_sensor_topology_distance_m": np.nan,
    }

def load_sensor_definition(method):
    """
    读取传感器定义。

    practical:
        读取经过工程间距和面元面积筛选后的
        all_surface_practical_sensors.csv。

    qr/deim:
        保留旧格式兼容性。
    """
    method = str(method).lower()

    if method == "practical":
        path = Path(
            cfg.PRACTICAL_SENSOR_DEFINITION_PATH
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"工程测点定义不存在：{path}"
            )

        table = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        required = {
            "tap_id",
            "global_face_id",
            "x_m",
            "y_m",
            "z_m",
            "face_area_cm2",
            "component",
        }

        missing = required - set(table.columns)

        if missing:
            raise KeyError(
                f"{path.name}缺少字段：{sorted(missing)}"
            )

        table = table.copy()

        table["tap_id"] = pd.to_numeric(
            table["tap_id"],
            errors="raise",
        ).astype(np.int64)

        table["global_face_id"] = pd.to_numeric(
            table["global_face_id"],
            errors="raise",
        ).astype(np.int64)

        for column in [
            "x_m",
            "y_m",
            "z_m",
            "face_area_cm2",
        ]:
            table[column] = pd.to_numeric(
                table[column],
                errors="raise",
            )

        table = table.sort_values(
            "tap_id"
        ).reset_index(drop=True)

        if table["global_face_id"].duplicated().any():
            duplicated = table.loc[
                table["global_face_id"].duplicated(),
                "global_face_id",
            ].tolist()

            raise ValueError(
                f"工程测点存在重复global_face_id："
                f"{duplicated[:10]}"
            )

        table["sensor_rank"] = np.arange(
            len(table),
            dtype=np.int64,
        ) + 1

        if len(table) != cfg.MAIN_SENSOR_COUNT:
            print(
                f"警告：工程测点定义文件包含"
                f"{len(table)}个点，"
                f"配置要求{cfg.MAIN_SENSOR_COUNT}个点"
            )

        return table

    if method in {"qr", "deim"}:
        if method == "qr":
            path = (
                cfg.DATASET_ROOT
                / "modal_analysis"
                / "cp_static"
                / "all_surface_qr_sensor.csv"
            )
        else:
            path = (
                cfg.DATASET_ROOT
                / "modal_analysis"
                / "cp_static"
                / "all_surface_deim_sensor.csv"
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"旧模态测点定义不存在：{path}"
            )

        table = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        required = {
            "part_name",
            "method",
            "sensor_rank",
            "global_face_id",
            "x_m",
            "y_m",
            "z_m",
            "face_area_m2",
        }

        missing = required - set(table.columns)

        if missing:
            raise KeyError(
                f"{path.name}缺少字段：{sorted(missing)}"
            )

        table = table.loc[
            table["part_name"].astype(str).str.lower()
            == str(cfg.PART_NAME).lower()
        ].copy()

        return table.sort_values(
            "sensor_rank"
        ).drop_duplicates(
            subset=["sensor_rank"],
            keep="first",
        ).reset_index(drop=True)

    raise ValueError(
        f"未知传感器方法：{method}"
    )

def match_sensor_coordinates_to_topology(
    sensor_xyz,
    topology_xyz,
    sensor_name,
):
    """将STAR传感器坐标一一匹配到topology面心。"""
    sensor_xyz = np.asarray(sensor_xyz, dtype=np.float64)
    topology_xyz = np.asarray(topology_xyz, dtype=np.float64)

    matched_indices = []
    distances = []
    used = set()

    for sensor_index, point in enumerate(sensor_xyz):
        delta = topology_xyz - point[None, :]
        distance = np.linalg.norm(delta, axis=1)

        order = np.argsort(distance)
        selected = None

        for candidate in order:
            candidate = int(candidate)
            if candidate in used:
                continue

            tolerance = max(
                float(cfg.SENSOR_COORD_ABS_TOLERANCE_M),
                float(cfg.SENSOR_COORD_REL_TOLERANCE)
                * max(float(np.linalg.norm(point)), 1.0),
            )

            if distance[candidate] <= tolerance:
                selected = candidate
                break

        if selected is None:
            nearest = int(order[0])
            raise ValueError(
                f"{sensor_name}第{sensor_index + 1}个传感器无法匹配topology面心；"
                f"最近距离={distance[nearest]:.6e} m，"
                f"允许容差={tolerance:.6e} m，"
                f"坐标={point}"
            )

        used.add(selected)
        matched_indices.append(selected)
        distances.append(float(distance[selected]))

    matched_indices = np.asarray(matched_indices, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)

    if len(np.unique(matched_indices)) != len(matched_indices):
        raise RuntimeError(
            f"{sensor_name}坐标匹配后出现重复topology面"
        )

    return matched_indices, distances

def convert_star_pressure_to_cp(star_table, velocity):
    """使用STAR绝对静压和三向来流动压计算Cp。"""
    velocity = np.asarray(velocity, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))

    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError(
            f"速度向量无效：{velocity}"
        )

    q_inf = 0.5 * float(cfg.RHO) * speed ** 2

    cp = (
        star_table["Absolute Pressure (Pa)"].to_numpy(dtype=np.float64)
        - float(cfg.P_STATIC_REF_PA)
    ) / q_inf

    if np.any(~np.isfinite(cp)):
        raise ValueError(
            "STAR传感器转换后的Cp包含NaN或Inf"
        )

    return cp

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

def get_velocity_vector(row):
    """读取三向速度向量。"""
    return np.asarray([
        float(row["u_inf_m_s"]),
        float(row["v_inf_m_s"]),
        float(row["w_inf_m_s"]),
    ], dtype=np.float64)

def calculate_speed_from_row(row):
    """计算三向速度模长。"""
    velocity = get_velocity_vector(row)

    speed = float(
        np.linalg.norm(velocity)
    )

    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError(
            f"工况速度无效：{velocity}"
        )

    return speed

# =============================================================================
# STAR-CCM+传感器读取与坐标匹配
# =============================================================================

STAR_SENSOR_COLUMNS = [
    "Absolute Pressure (Pa)",
    "Absolute Total Pressure (Pa)",
    "Velocity: Magnitude (m/s)",
    "Velocity[i] (m/s)",
    "Velocity[j] (m/s)",
    "Velocity[k] (m/s)",
    "Vorticity: Magnitude (/s)",
    "Vorticity[i] (/s)",
    "Vorticity[j] (/s)",
    "Vorticity[k] (/s)",
    "X (m)",
    "Y (m)",
    "Z (m)",
]

def read_star_sensor_csv(path):
    """读取STAR-CCM+导出的传感器CSV。"""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"STAR传感器文件不存在：{path}"
        )

    table = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    table.columns = [
        str(column).strip()
        for column in table.columns
    ]

    missing = (
        set(STAR_SENSOR_COLUMNS)
        - set(table.columns)
    )

    if missing:
        raise KeyError(
            f"{path.name}缺少字段：{sorted(missing)}"
        )

    for column in STAR_SENSOR_COLUMNS:
        table[column] = pd.to_numeric(
            table[column],
            errors="coerce",
        )

    if table[STAR_SENSOR_COLUMNS].isna().any().any():
        raise ValueError(
            f"{path.name}包含NaN或非数值数据"
        )

    return table

def match_points_one_to_one(
    source_xyz,
    target_xyz,
    source_name,
):
    """
    使用带容差的最近邻方法进行一一坐标匹配。

    返回：
        source_xyz中每个点对应的target_xyz行号
        每个匹配点的距离
    """
    source_xyz = np.asarray(
        source_xyz,
        dtype=np.float64,
    )

    target_xyz = np.asarray(
        target_xyz,
        dtype=np.float64,
    )

    selected_indices = []
    selected_distances = []
    used_targets = set()

    for source_index, point in enumerate(
        source_xyz
    ):
        distances = np.linalg.norm(
            target_xyz - point[None, :],
            axis=1,
        )

        tolerance = max(
            float(
                cfg.SENSOR_COORD_ABS_TOLERANCE_M
            ),
            float(
                cfg.SENSOR_COORD_REL_TOLERANCE
            )
            * max(
                float(np.linalg.norm(point)),
                1.0,
            ),
        )

        order = np.argsort(distances)
        selected = None

        for candidate in order:
            candidate = int(candidate)

            if candidate in used_targets:
                continue

            if distances[candidate] <= tolerance:
                selected = candidate
                break

        if selected is None:
            nearest = int(order[0])

            raise ValueError(
                f"{source_name}第{source_index + 1}个点匹配失败；"
                f"最近距离={distances[nearest]:.6e} m，"
                f"容差={tolerance:.6e} m，"
                f"坐标={point}"
            )

        used_targets.add(selected)
        selected_indices.append(selected)
        selected_distances.append(
            float(distances[selected])
        )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    selected_distances = np.asarray(
        selected_distances,
        dtype=np.float64,
    )

    if len(np.unique(selected_indices)) != len(
        selected_indices
    ):
        raise RuntimeError(
            f"{source_name}匹配结果存在重复面"
        )

    return (
        selected_indices,
        selected_distances,
    )

def convert_star_sensor_pressure_to_cp(
    star_table,
    velocity,
):
    """
    使用Absolute Pressure (Pa)计算传感器Cp。

    注意：这里不能使用Absolute Total Pressure (Pa)。
    """
    velocity = np.asarray(
        velocity,
        dtype=np.float64,
    )

    speed = float(
        np.linalg.norm(velocity)
    )

    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError(
            f"速度无效：{velocity}"
        )

    q_inf = (
        0.5
        * float(cfg.RHO)
        * speed ** 2
    )

    cp = (
        star_table[
            "Absolute Pressure (Pa)"
        ].to_numpy(dtype=np.float64)
        - float(cfg.P_STATIC_REF_PA)
    ) / q_inf

    if np.any(~np.isfinite(cp)):
        raise ValueError(
            "STAR传感器Cp包含NaN或Inf"
        )

    return cp

def resolve_runtime_sensor_path(
    unseen_row,
    method,
):
    """
    根据工况清单和工况目录解析STAR测点文件。
    """
    method = str(method).lower()

    if method == "practical":
        path_column = "practical_sensor_path"
        prefix = "PracticalSensorFromStar_"

    elif method == "qr":
        path_column = "qr_sensor_path"
        prefix = "QRSensorFromStar_"

    elif method == "deim":
        path_column = "deim_sensor_path"
        prefix = "DEIMSensorFromStar_"

    else:
        raise ValueError(
            f"未知传感器方法：{method}"
        )

    if path_column in unseen_row.index:
        manifest_value = unseen_row[path_column]

        if pd.notna(manifest_value):
            manifest_path = Path(
                str(manifest_value)
            ).expanduser()

            if manifest_path.is_file():
                return manifest_path.resolve()
        else:
            manifest_path = Path("")

    else:
        manifest_path = Path("")

    if "case_directory" in unseen_row.index:
        case_directory = Path(
            str(unseen_row["case_directory"])
        ).expanduser()
    else:
        case_directory = manifest_path.parent

    if not case_directory.is_dir():
        case_directory = manifest_path.parent

    candidates = sorted(
        case_directory.glob(
            f"{prefix}*.csv"
        )
    )

    if len(candidates) == 1:
        return candidates[0].resolve()

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"没有找到{method.upper()} STAR测点文件；"
            f"工况目录：{case_directory}；"
            f"清单路径：{manifest_path}"
        )

    raise RuntimeError(
        f"工况目录中找到多个{method.upper()} STAR测点文件：\n"
        + "\n".join(
            str(path)
            for path in candidates
        )
    )

def load_case_star_sensor_data(
    unseen_row,
    method,
    topology,
):
    """
    读取STAR-CCM+传感器压力，并按照工程测点定义排序。

    返回的cp顺序与：
        all_surface_practical_sensors.csv
    的tap_id顺序一致。
    """
    method = str(method).lower()

    sensor_path = resolve_runtime_sensor_path(
        unseen_row,
        method,
    )

    star_table = read_star_sensor_csv(
        sensor_path
    )

    velocity = np.asarray([
        float(unseen_row["u_inf_m_s"]),
        float(unseen_row["v_inf_m_s"]),
        float(unseen_row["w_inf_m_s"]),
    ], dtype=np.float64)

    star_xyz = star_table[
        ["X (m)", "Y (m)", "Z (m)"]
    ].to_numpy(dtype=np.float64)

    sensor_definition = load_sensor_definition(
        method
    )

    definition_xyz = sensor_definition[
        ["x_m", "y_m", "z_m"]
    ].to_numpy(dtype=np.float64)

    if len(star_table) != len(
        sensor_definition
    ):
        raise ValueError(
            f"{method.upper()} STAR测点数量不一致："
            f"STAR={len(star_table)}，"
            f"定义文件={len(sensor_definition)}"
        )

    star_to_definition, definition_distances = (
        match_points_one_to_one(
            source_xyz=star_xyz,
            target_xyz=definition_xyz,
            source_name=(
                f"{method.upper()} STAR到测点定义"
            ),
        )
    )

    star_cp = convert_star_sensor_pressure_to_cp(
        star_table,
        velocity,
    )

    ordered_cp = np.full(
        len(sensor_definition),
        np.nan,
        dtype=np.float64,
    )

    ordered_cp[star_to_definition] = star_cp

    if np.any(~np.isfinite(ordered_cp)):
        raise ValueError(
            f"{method.upper()} STAR压力无法完整排序"
        )

    definition_global_ids = (
        sensor_definition[
            "global_face_id"
        ].to_numpy(dtype=np.int64)
    )

    # all_surface下global_face_id就是topology数组下标。
    # 这里仍然通过坐标重新匹配，避免拓扑编号或导出顺序不一致。
    topology_indices, topology_distances = (
        match_points_one_to_one(
            source_xyz=definition_xyz,
            target_xyz=topology["centroid"],
            source_name=(
                f"{method.upper()}测点定义到topology"
            ),
        )
    )

    topology_global_ids = (
        topology["global_face_ids"][
            topology_indices
        ]
    )

    if not np.array_equal(
        definition_global_ids,
        topology_global_ids,
    ):
        mismatched = np.flatnonzero(
            definition_global_ids
            != topology_global_ids
        )

        print(
            f"警告：{method.upper()}定义文件中的"
            f"global_face_id与坐标匹配结果有"
            f"{len(mismatched)}处不同。"
            "后续以坐标匹配结果为准。"
        )

    return {
        "cp": ordered_cp,
        "topology_indices": topology_indices,
        "global_face_ids": topology_global_ids,
        "sensor_table": star_table,
        "sensor_definition": sensor_definition,
        "sensor_path": str(sensor_path),
        "max_star_sensor_distance_m": float(
            np.max(definition_distances)
        ),
        "max_sensor_topology_distance_m": float(
            np.max(topology_distances)
        ),
    }


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

def velocity_metadata(
    velocity,
):
    velocity = np.asarray(
        velocity,
        dtype=np.float64,
    )

    return {
        "u_inf_m_s": float(
            velocity[0]
        ),
        "v_inf_m_s": float(
            velocity[1]
        ),
        "w_inf_m_s": float(
            velocity[2]
        ),
        "speed_m_s": float(
            np.linalg.norm(velocity)
        ),
    }
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

    pod_variances = singular_values ** 2

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
        "pod_variances": singular_values ** 2,
        "cumulative_energy": cumulative_energy,
        "numerical_rank": numerical_rank,
    }


def classify_flow_regime(u, v, w):
    """按主导方向分类流态"""
    speed = (u ** 2 + v ** 2 + w ** 2) ** 0.5
    if speed < 1e-12:
        return "zero"

    abs_u, abs_v, abs_w = abs(u), abs(v), abs(w)

    # 轴向主导：|u| > 1.5 * max(|v|, |w|)
    if abs_u > 1.5 * max(abs_v, abs_w):
        return "axial"

    # 横向主导：|v| > 1.5 * max(|u|, |w|)
    if abs_v > 1.5 * max(abs_u, abs_w):
        return "lateral_y"

    # 垂向主导：|w| > 1.5 * max(|u|, |v|)
    if abs_w > 1.5 * max(abs_u, abs_v):
        return "lateral_z"

    # 三向均衡
    return "oblique"


def build_target_specific_training_set(
        all_train_cases,
        all_train_snapshots,
        target_velocity,
):
    """
    按流态相似性筛选训练集子集，并剔除同速工况。

    参数
    ----
    all_train_cases : pd.DataFrame
        全部训练工况元数据

    all_train_snapshots : np.ndarray
        shape=(n_train, n_faces)，对应all_train_cases的压力快照

    target_velocity : array-like
        目标工况的三向速度 [u, v, w]

    返回
    ----
    filtered_cases : pd.DataFrame
        筛选后的训练工况

    filtered_snapshots : np.ndarray
        对应的压力快照

    excluded_count : int
        被剔除的同速工况数量
    """
    target_velocity = np.asarray(
        target_velocity,
        dtype=np.float64,
    )

    target_speed = float(
        np.linalg.norm(target_velocity)
    )

    # 1. 先按流态分类，筛选同一流态的训练工况
    test_regime = classify_flow_regime(
        target_velocity[0],
        target_velocity[1],
        target_velocity[2],
    )

    regime_mask = all_train_cases.apply(
        lambda row: classify_flow_regime(
            row["u_inf_m_s"],
            row["v_inf_m_s"],
            row["w_inf_m_s"],
        ) == test_regime,
        axis=1,
    )

    selected_cases = all_train_cases[regime_mask].copy()
    selected_snapshots = all_train_snapshots[regime_mask]

    # 如果同流态工况太少（<30个），扩展到oblique作为备用
    if len(selected_cases) < 30 and test_regime != "oblique":
        oblique_mask = all_train_cases.apply(
            lambda row: classify_flow_regime(
                row["u_inf_m_s"],
                row["v_inf_m_s"],
                row["w_inf_m_s"],
            ) == "oblique",
            axis=1,
        )

        additional_cases = all_train_cases[oblique_mask]
        additional_snapshots = all_train_snapshots[oblique_mask]

        selected_cases = pd.concat(
            [selected_cases, additional_cases],
            ignore_index=True,
        )
        selected_snapshots = np.vstack([
            selected_snapshots,
            additional_snapshots,
        ])

    # 2. 剔除与目标速度相同的训练工况（数据泄漏控制）
    if not cfg.EXCLUDE_TEST_SPEED_FROM_TRAINING:
        return (
            selected_cases.reset_index(drop=True),
            selected_snapshots,
            0,
        )


    train_velocity = selected_cases[
        [
            "u_inf_m_s",
            "v_inf_m_s",
            "w_inf_m_s",
        ]
    ].to_numpy(dtype=np.float64)

    target_velocity = np.asarray(
        target_velocity,
        dtype=np.float64,
    )

    same_velocity_mask = np.all(
        np.isclose(
            train_velocity,
            target_velocity[None, :],
            atol=cfg.SPEED_MATCH_TOLERANCE,
            rtol=0.0,
        ),
        axis=1,
    )

    excluded_count = int(
        np.count_nonzero(same_velocity_mask)
    )

    keep_mask = ~same_velocity_mask

    return (
        selected_cases.loc[
            keep_mask
        ].reset_index(drop=True),
        selected_snapshots[keep_mask],
        excluded_count,
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
def build_parameter_features(
    cases,
    feature_set="uvw",
):
    """
    构造参数预测模型输入。

    当前正式模型使用uvw，因为交叉验证结果显示：
        uvw + GP-RBF优于U-alpha-beta + GP-RBF。
    """
    u = cases[
        "u_inf_m_s"
    ].to_numpy(dtype=np.float64)

    v = cases[
        "v_inf_m_s"
    ].to_numpy(dtype=np.float64)

    w = cases[
        "w_inf_m_s"
    ].to_numpy(dtype=np.float64)

    if feature_set == "uvw":
        return np.column_stack([
            u,
            v,
            w,
        ])

    if feature_set in {
        "Ualpha",
        "UalphaBeta",
    }:
        speed = np.sqrt(
            u ** 2 + v ** 2 + w ** 2
        )

        alpha = np.degrees(
            np.arctan2(
                w,
                np.hypot(u, v),
            )
        )

        beta = np.degrees(
            np.arctan2(v, u)
        )

        return np.column_stack([
            speed,
            alpha,
            beta,
        ])

    raise ValueError(
        f"未知参数特征组：{feature_set}"
    )

def standardize_train_target_features(
    x_train,
    x_target,
):
    mean = np.mean(
        x_train,
        axis=0,
    )

    std = np.std(
        x_train,
        axis=0,
        ddof=0,
    )

    std = np.where(
        std > 1.0e-12,
        std,
        1.0,
    )

    return (
        (x_train - mean) / std,
        (x_target - mean) / std,
    )

def rbf_kernel_matrix(
    x_a,
    x_b,
    length_scale,
):
    x_a = np.asarray(
        x_a,
        dtype=np.float64,
    )

    x_b = np.asarray(
        x_b,
        dtype=np.float64,
    )

    norm_a = np.sum(
        x_a ** 2,
        axis=1,
    )[:, None]

    norm_b = np.sum(
        x_b ** 2,
        axis=1,
    )[None, :]

    distance = (
        norm_a
        + norm_b
        - 2.0 * x_a @ x_b.T
    )

    distance = np.maximum(
        distance,
        0.0,
    )

    return np.exp(
        -0.5
        * distance
        / max(
            length_scale ** 2,
            1.0e-12,
        )
    )

def fit_rbf_parameter_model(
    train_cases,
    train_snapshots,
    target_case,
    pod,
    area,
    *,
    feature_set="uvw",
    length_scale=0.5,
    noise=0.05,
    ridge_lambda=1.0e-8,
    mode_count=70,
):
    """
    使用训练工况参数预测目标工况的POD系数。

    输出：
        prediction:
            目标工况完整压力场预测。

        coefficients:
            预测POD系数。

        model_metadata:
            模型诊断信息。
    """
    mode_count = min(
        int(mode_count),
        pod["basis"].shape[1],
    )

    basis = pod[
        "basis"
    ][:, :mode_count]

    mean_field = pod[
        "mean_field"
    ]

    x_train_raw = build_parameter_features(
        train_cases,
        feature_set=feature_set,
    )

    target_df = pd.DataFrame([
        target_case
    ])

    x_target_raw = build_parameter_features(
        target_df,
        feature_set=feature_set,
    )

    x_train, x_target = (
        standardize_train_target_features(
            x_train_raw,
            x_target_raw,
        )
    )

    sqrt_area = np.sqrt(
        np.maximum(
            area,
            EPS,
        )
    )

    fluctuations = (
        train_snapshots
        - mean_field[None, :]
    )

    weighted_fluctuations = (
        fluctuations
        * sqrt_area[None, :]
    )

    weighted_basis = (
        basis
        * sqrt_area[:, None]
    )

    # 面积加权正交投影得到训练POD系数
    coefficients_train = (
        weighted_fluctuations
        @ weighted_basis
    )

    y_mean = np.mean(
        coefficients_train,
        axis=0,
        keepdims=True,
    )

    y_centered = (
        coefficients_train
        - y_mean
    )

    kernel_train = rbf_kernel_matrix(
        x_train,
        x_train,
        length_scale,
    )

    kernel_train = (
        kernel_train
        + (
            noise ** 2
            + ridge_lambda
        )
        * np.eye(
            len(x_train),
            dtype=np.float64,
        )
    )

    kernel_target = rbf_kernel_matrix(
        x_target,
        x_train,
        length_scale,
    )

    try:
        alpha = np.linalg.solve(
            kernel_train,
            y_centered,
        )
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(
            kernel_train,
            y_centered,
            rcond=1.0e-12,
        )[0]

    predicted_coefficients = (
        kernel_target @ alpha
        + y_mean
    ).reshape(-1)

    prediction = (
        mean_field
        + basis
        @ predicted_coefficients
    )

    return {
        "prediction": prediction,
        "coefficients": predicted_coefficients,
        "coefficients_train": coefficients_train,
        "mode_count": mode_count,
        "feature_set": feature_set,
        "length_scale": length_scale,
        "noise": noise,
        "ridge_lambda": ridge_lambda,
    }

def solve_coefficients(
    sampled_basis,
    sampled_fluctuation,
    *,
    solver_name=None,
    ridge_lambda=None,
    pod_variances=None,
    sensor_weights=None,
    coefficient_prior=None,
    prior_lambda=0.0,
):
    A = np.asarray(
        sampled_basis,
        dtype=np.float64,
    )

    y = np.asarray(
        sampled_fluctuation,
        dtype=np.float64,
    ).reshape(-1)

    solver = str(
        solver_name
        if solver_name is not None
        else cfg.COEFFICIENT_SOLVER
    ).lower()

    if sensor_weights is not None:
        weights = np.asarray(
            sensor_weights,
            dtype=np.float64,
        )

        sqrt_weights = np.sqrt(
            np.maximum(weights, 1.0e-12)
        )

        A_work = A * sqrt_weights[:, None]
        y_work = y * sqrt_weights

    else:
        A_work = A
        y_work = y

    if solver == "lstsq":
        return np.linalg.lstsq(
            A_work,
            y_work,
            rcond=cfg.LSTSQ_RCOND,
        )[0]

    gram = A_work.T @ A_work
    rhs = A_work.T @ y_work
    n_modes = A.shape[1]

    lambda_value = float(
        ridge_lambda
        if ridge_lambda is not None
        else cfg.RIDGE_LAMBDA
    )

    if solver in {
        "ridge",
        "scaled_ridge",
        "weighted_ridge",
    }:
        # 【新增】尺度归一化因子
        gram_trace = np.trace(gram)
        normalization_factor = gram_trace / max(n_modes, 1)
        normalization_factor = max(
            normalization_factor,
            1.0e-14,
        )

        lambda_effective = (
                lambda_value * normalization_factor
        )

        if solver == "ridge":
            regularizer = np.eye(
                n_modes,
                dtype=np.float64,
            )

        else:
            regularizer = np.diag(
                np.maximum(
                    np.diag(gram),
                    1.0e-14,
                )
            )

        # lambda=0时退回lstsq
        if lambda_effective <= 1.0e-14:
            return np.linalg.lstsq(
                A_work,
                y_work,
                rcond=cfg.LSTSQ_RCOND,
            )[0]

        return np.linalg.solve(
            gram
            + lambda_effective * regularizer,  # 使用归一化后的λ
            rhs,
        )

    if solver == "energy_ridge":
        if pod_variances is None:
            raise ValueError(
                "energy_ridge需要pod_variances"
            )

        variances = np.asarray(
            pod_variances[:n_modes],
            dtype=np.float64,
        )

        normalized_variances = (
            variances
            / max(
                np.max(variances),
                1.0e-14,
            )
        )

        penalty = 1.0 / np.maximum(
            normalized_variances,
            1.0e-8,
        )

        regularizer = np.diag(penalty)

        if lambda_value <= 0.0:
            return np.linalg.lstsq(
                A_work,
                y_work,
                rcond=cfg.LSTSQ_RCOND,
            )[0]

        return np.linalg.solve(
            gram
            + lambda_value * regularizer,
            rhs,
        )

    if solver == "prior_ridge":
        if coefficient_prior is None:
            raise ValueError(
                "prior_ridge需要coefficient_prior"
            )

        prior = np.asarray(
            coefficient_prior,
            dtype=np.float64,
        ).reshape(-1)

        if len(prior) != n_modes:
            raise ValueError(
                "coefficient_prior长度错误"
            )

        if lambda_value <= 0.0:
            lambda_value = 1.0e-8

        lhs = (
            gram
            + lambda_value
            * np.eye(n_modes)
        )

        rhs_prior = (
            prior_lambda * prior
        )

        return np.linalg.solve(
            lhs,
            rhs + rhs_prior,
        )

    raise ValueError(
        f"未知系数求解器：{solver}"
    )

def _poly_features(X, degree):
    """uvw 的多项式特征（含 1 与交互项）。"""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    cols = [np.ones(n)]
    u, v, w = X[:, 0], X[:, 1], X[:, 2]
    if degree >= 1:
        cols += [u, v, w]
    if degree >= 2:
        cols += [u * u, v * v, w * w, u * v, u * w, v * w]
    if degree >= 3:
        cols += [
            u ** 3, v ** 3, w ** 3,
            u * u * v, u * u * w, v * v * u,
            v * v * w, w * w * u, w * w * v,
            u * v * w,
        ]
    return np.column_stack(cols)

def build_velocity_prior_coefficients(
    *,
    train_cases,
    train_snapshots,
    pod,
    target_velocity,
    area,
    mode_count,
    feature_degree=2,
    ridge=1.0e-3,
):
    """
    【改进】用训练工况在速度参数空间（uvw）对 POD 系数做多项式岭回归，
    预测目标工况的 POD 系数，作为稀疏重构的先验（prior_ridge 用）。

    通过把"参数空间平滑性"作为先验注入，显著缓解三向速度工况，
    尤其是外推 / 纯纵向低速工况（v=0 或 w=0）下的病态重构。
    """
    basis = np.asarray(pod["basis"], dtype=np.float64)[:, :mode_count]
    mean_field = np.asarray(pod["mean_field"], dtype=np.float64)
    snapshots = np.asarray(train_snapshots, dtype=np.float64)
    area_arr = np.asarray(area, dtype=np.float64)

    sqrt_area = np.sqrt(np.maximum(area_arr, 1.0e-14))
    fluctuations = snapshots - mean_field[None, :]

    # 面积加权投影得到训练工况 POD 系数 (n_train, mode_count)
    train_coeff = (
        (fluctuations * sqrt_area[None, :])
        @ (basis * sqrt_area[:, None])
    )

    u = train_cases["u_inf_m_s"].to_numpy(dtype=np.float64)
    v = train_cases["v_inf_m_s"].to_numpy(dtype=np.float64)
    w = train_cases["w_inf_m_s"].to_numpy(dtype=np.float64)
    X = np.column_stack([u, v, w])

    Xp = _poly_features(X, feature_degree)
    mu = Xp[:, 1:].mean(axis=0)
    sd = Xp[:, 1:].std(axis=0) + 1.0e-9
    Xn = np.column_stack([
        np.ones(Xp.shape[0]),
        (Xp[:, 1:] - mu) / sd,
    ])

    gram = Xn.T @ Xn + float(ridge) * np.eye(Xn.shape[1])
    weight_mat = np.linalg.solve(gram, Xn.T @ train_coeff)

    Xt = _poly_features(np.atleast_2d(
        np.asarray(target_velocity, dtype=np.float64)
    ), feature_degree)
    Xtn = np.column_stack([
        np.ones(1),
        (Xt[:, 1:] - mu) / sd,
    ])

    prior_coefficients = (Xtn @ weight_mat).reshape(-1)
    return prior_coefficients

def compute_sensor_residual_weights(
    train_snapshots,
    mean_field,
    basis,
    sensor_indices,
    area,
):
    """
    使用训练工况估计固定测点的残差可靠性。
    """

    snapshots = np.asarray(
        train_snapshots,
        dtype=np.float64,
    )

    sensor_indices = np.asarray(
        sensor_indices,
        dtype=np.int64,
    )

    fluctuations = (
        snapshots
        - mean_field[None, :]
    )

    sqrt_area = np.sqrt(
        np.maximum(area, 1.0e-14)
    )

    weighted_basis = (
        basis * sqrt_area[:, None]
    )

    weighted_fluctuations = (
        fluctuations * sqrt_area[None, :]
    )

    # 面积加权投影得到训练系数
    coefficients = (
        weighted_fluctuations
        @ weighted_basis
    )

    reconstructed = (
        mean_field[None, :]
        + coefficients @ basis.T
    )

    residual = (
        snapshots - reconstructed
    )

    sensor_residual = residual[
        :,
        sensor_indices,
    ]

    variance = np.var(
        sensor_residual,
        axis=0,
        ddof=1,
    )

    weights = 1.0 / np.maximum(
        variance,
        1.0e-10,
    )

    # 归一化，避免整体权重改变目标函数尺度
    weights = weights / np.mean(
        weights
    )

    return weights


def compute_sensor_residual_weights_by_mode(
        train_snapshots,
        mean_field,
        basis,
        sensor_indices,
        area,
):
    """
    使用训练集计算固定传感器的POD残差权重。

    对每个训练工况：
        1. 面积加权投影得到POD系数
        2. 重构该工况压力场
        3. 计算传感器位置的残差
        4. 统计残差方差

    权重定义为残差方差的倒数（归一化后）。
    """
    snapshots = np.asarray(
        train_snapshots,
        dtype=np.float64,
    )

    sensor_indices = np.asarray(
        sensor_indices,
        dtype=np.int64,
    )

    n_train = snapshots.shape[0]
    n_sensors = len(sensor_indices)

    if n_train < 2:
        # 训练样本太少，返回等权重
        return np.ones(n_sensors, dtype=np.float64)

    # 面积权重矩阵
    sqrt_area = np.sqrt(
        np.maximum(area, 1.0e-14)
    )

    weighted_basis = (
            basis * sqrt_area[:, None]
    )

    # 每个工况的波动场
    fluctuations = (
            snapshots - mean_field[None, :]
    )

    weighted_fluctuations = (
            fluctuations * sqrt_area[None, :]
    )

    # 面积加权投影得到系数
    coefficients = (
            weighted_fluctuations @ weighted_basis
    )

    # 重构
    reconstructed = (
            mean_field[None, :]
            + coefficients @ basis.T
    )

    # 残差
    residual = snapshots - reconstructed

    # 传感器位置的残差
    sensor_residual = residual[:, sensor_indices]

    # 计算方差
    variance = np.var(
        sensor_residual,
        axis=0,
        ddof=1,
    )

    # 权重 = 1 / (方差 + 小量)
    weights = 1.0 / np.maximum(
        variance,
        1.0e-10,
    )

    # 归一化，使平均权重为1
    weights = weights / np.mean(weights)

    return weights


def reconstruct_sparse_field(
    truth,
    observed_sensor_cp,
    mean_field,
    basis,
    sensor_indices,
    noise_level,
    rng,
    *,
    solver_name=None,
    ridge_lambda=None,
    pod_variances=None,
    sensor_weights=None,
    coefficient_prior=None,
    prior_lambda=0.0,
):
    """
    使用实际传感器Cp重构完整Cp场。
    """
    sensor_indices = np.asarray(
        sensor_indices,
        dtype=np.int64,
    )

    observed_cp = np.asarray(
        observed_sensor_cp,
        dtype=np.float64,
    ).copy()

    if len(observed_cp) != len(sensor_indices):
        raise ValueError(
            f"传感器Cp数量={len(observed_cp)}，"
            f"传感器索引数量={len(sensor_indices)}"
        )

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
        solver_name=solver_name,
        ridge_lambda=ridge_lambda,
        pod_variances=pod_variances,
        sensor_weights=sensor_weights,
        coefficient_prior=coefficient_prior,
        prior_lambda=prior_lambda,
    )

    raw_condition_number = float(
        np.linalg.cond(sampled_basis)
    )

    column_norms = np.linalg.norm(
        sampled_basis,
        axis=0,
    )

    normalized_basis = (
            sampled_basis
            / np.maximum(
        column_norms[None, :],
        1.0e-14,
    )
    )

    scaled_condition_number = float(
        np.linalg.cond(normalized_basis)
    )

    prediction = (
        mean_field
        + basis @ coefficient
    )

    return {
        "prediction": prediction,
        "coefficient": coefficient,
        "condition_number": raw_condition_number,
        "scaled_condition_number": scaled_condition_number,
        "sampling_rank": int(np.linalg.matrix_rank(sampled_basis)),
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

    # 方差归一化的面积加权相对L2误差
    # 分母用真值的面积加权方差而不是面积加权平方和
    # 等价于 sqrt(1 - R2_area_weighted)，不受Cp均值偏移影响
    truth_mean_area_weighted = (
            np.sum(area * truth)
            / max(np.sum(area), EPS)
    )

    area_weighted_variance = np.sum(
        area
        * (truth - truth_mean_area_weighted) ** 2
    )

    if area_weighted_variance > EPS:
        cp_relative_l2_variance_normalized = float(
            np.sqrt(
                weighted_numerator
                / area_weighted_variance
            )
        )
    else:
        cp_relative_l2_variance_normalized = np.nan

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
        "cp_relative_l2_variance_normalized": (
            cp_relative_l2_variance_normalized
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
    velocity,
    topology,
    normal,
):
    """根据Cp、速度向量和表面法向量计算压力积分。"""
    cp = np.asarray(
        cp,
        dtype=np.float64,
    )

    velocity = np.asarray(
        velocity,
        dtype=np.float64,
    )

    normal = np.asarray(
        normal,
        dtype=np.float64,
    )

    if len(cp) != topology[
        "face_count_part"
    ]:
        raise ValueError(
            "Cp长度与目标部件面数不一致"
        )

    if normal.shape != (
        topology["face_count_part"],
        3,
    ):
        raise ValueError(
            f"法向量形状异常：{normal.shape}"
        )

    speed = float(
        np.linalg.norm(velocity)
    )

    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError(
            f"速度无效：{velocity}"
        )

    q_inf = (
        0.5
        * float(cfg.RHO)
        * speed ** 2
    )

    pressure = cp * q_inf
    area = topology["area"]
    centroid = topology["centroid"]

    force_density = (
        pressure[:, None] * normal
    )

    force = np.sum(
        force_density * area[:, None],
        axis=0,
    )

    reference_point = np.asarray(
        cfg.MOMENT_REFERENCE_POINT_M,
        dtype=np.float64,
    )

    arm = (
        centroid
        - reference_point[None, :]
    )

    moment_density = np.cross(
        arm,
        force_density,
    )

    moment = np.sum(
        moment_density * area[:, None],
        axis=0,
    )

    reference_area = float(
        cfg.REFERENCE_AREA_M2
    )

    reference_length = float(
        cfg.REFERENCE_LENGTH_M
    )

    return {
        "force_x_n": float(force[0]),
        "force_y_n": float(force[1]),
        "force_z_n": float(force[2]),
        "moment_x_nm": float(moment[0]),
        "moment_y_nm": float(moment[1]),
        "moment_z_nm": float(moment[2]),
        "coefficient_force_x": float(
            force[0]
            / (q_inf * reference_area)
        ),
        "coefficient_force_y": float(
            force[1]
            / (q_inf * reference_area)
        ),
        "coefficient_force_z": float(
            force[2]
            / (q_inf * reference_area)
        ),
        "coefficient_moment_x": float(
            moment[0]
            / (
                q_inf
                * reference_area
                * reference_length
            )
        ),
        "coefficient_moment_y": float(
            moment[1]
            / (
                q_inf
                * reference_area
                * reference_length
            )
        ),
        "coefficient_moment_z": float(
            moment[2]
            / (
                q_inf
                * reference_area
                * reference_length
            )
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
    target_velocity,
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
    solver_name=None,
    ridge_lambda=None,
    pod_variances=None,
    sensor_weights=None,
    coefficient_prior=None,
    prior_lambda=0.0,
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

    global CURRENT_STAR_SENSOR_DATA

    sensor_method = str(
        sensor_method
    ).lower()

    if sensor_method in {
        "qr",
        "deim",
        "practical",
    }:
        # qr、deim、practical优先读取STAR测量数据
        if sensor_method in CURRENT_STAR_SENSOR_DATA:
            star_data = CURRENT_STAR_SENSOR_DATA[
                sensor_method
            ]

            if sensor_count > len(
                    star_data["cp"]
            ):
                raise ValueError(
                    f"{sensor_method.upper()}请求"
                    f"{sensor_count}个测点，但STAR文件只有"
                    f"{len(star_data['cp'])}个"
                )

            # topology_indices是局部面索引
            sensor_indices = star_data[
                "topology_indices"
            ][:sensor_count]

            # cp顺序与topology_indices一一对应
            observed_sensor_cp = star_data[
                "cp"
            ][:sensor_count]

        else:
            # 没有STAR数据时，不允许静默使用真值。
            # 只有显式设置为数值基线时才允许。
            raise RuntimeError(
                f"{sensor_method.upper()}没有加载STAR测量数据。"
                f"当前工况不能执行真实测压重构。"
            )

    else:
        # random和uniform仅用于数值模拟基线。
        # 它们直接从CFD真值中抽取测点压力。
        sensor_indices = select_sensors(
            method=sensor_method,
            basis=basis,
            sensor_count=sensor_count,
            xyz=topology["centroid"],
            rng=rng,
        )

        observed_sensor_cp = truth[
            sensor_indices
        ]

    start = time.perf_counter()
    if pod_variances is None:
        if "pod_variances" in pod:
            active_pod_variances = pod[
                "pod_variances"
            ][:actual_mode_count]
        else:
            active_pod_variances = None
    else:
        active_pod_variances = np.asarray(
            pod_variances[:actual_mode_count],
            dtype=np.float64,
        )

    result = reconstruct_sparse_field(
        truth=truth,
        observed_sensor_cp=observed_sensor_cp,
        mean_field=pod["mean_field"],
        basis=basis,
        sensor_indices=sensor_indices,
        noise_level=noise_level,
        rng=rng,
        solver_name=solver_name,
        ridge_lambda=ridge_lambda,
        pod_variances=active_pod_variances,
        sensor_weights=sensor_weights,
        coefficient_prior=coefficient_prior,
        prior_lambda=prior_lambda,
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
        "observation_source": (
            CURRENT_STAR_SENSOR_DATA.get(
                sensor_method,
                {}
            ).get(
                "observation_source",
                "unknown",
            )
        ),
        "sensor_layout": (
            "engineering_practical"
            if sensor_method == "practical"
            else sensor_method
        ),
        "case_id": case_id,
        "u_inf_m_s": float(target_velocity[0]),
        "v_inf_m_s": float(target_velocity[1]),
        "w_inf_m_s": float(target_velocity[2]),
        "speed_m_s": target_speed,
        "split_type": split_type,
        "part_name": cfg.PART_NAME,
        "method": sensor_method,
        "sensor_count": len(sensor_indices),
        "requested_sensor_count": (sensor_count),
        "mode_count": actual_mode_count,
        "requested_mode_count": (mode_count),
        "noise_level": noise_level,
        "noise_percent": (100.0 * noise_level),
        "repeat": repeat,
        "sampling_rank": result["sampling_rank"],
        "condition_number": result["condition_number"],
        "runtime_ms": elapsed_ms,
    }
    # 【修正】明确标记实际求解器和参数
    if solver_name is not None:
        row["solver"] = solver_name
        row["ridge_lambda_used"] = (
            ridge_lambda
            if ridge_lambda is not None
            else 0.0
        )
    else:
        row["solver"] = cfg.COEFFICIENT_SOLVER
        row["ridge_lambda_used"] = (
            cfg.RIDGE_LAMBDA
            if cfg.COEFFICIENT_SOLVER == "ridge"
            else 0.0
        )

    row["prior_lambda"] = prior_lambda

    # 【新增】标记是否使用了传感器权重
    row["sensor_weights_enabled"] = (
            sensor_weights is not None
    )
    row["scaled_condition_number"] = result[
        "scaled_condition_number"
    ]

    row.update(metrics)

    return (
        row,
        result["prediction"],
        sensor_indices,
    )

# =============================================================================
# 10. 主实验流程
# =============================================================================

def ensure_unseen_cases_built():
    """
    【健壮性守卫】未见工况真值(h5)或清单(csv)缺失/不完整时，
    自动调用 01_build_unseen_cases_from_sbd.py 重建。
    仅在缺失时触发，不影响已有完整数据时的正常流程。
    """
    expected = len(cfg.UNSEEN_SBD_CASES)
    out_dir = cfg.UNSEEN_H5_ROOT
    csv_path = cfg.UNSEEN_CASES_CSV_PATH

    built = (
        len(list(out_dir.glob("*.h5")))
        if out_dir.is_dir() else 0
    )
    csv_ok = csv_path.is_file()

    if built >= expected and csv_ok:
        return

    print(
        f"检测到未见工况真值不完整："
        f"h5={built}/{expected}，"
        f"csv={'存在' if csv_ok else '缺失'}"
    )
    print("自动调用 01_build_unseen_cases_from_sbd.py 重建 ...")

    builder = (
        Path(__file__).resolve().parent
        / "01_build_unseen_cases_from_sbd.py"
    )

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(builder)],
        cwd=str(builder.parent),
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "01_build_unseen_cases_from_sbd.py "
            "自动重建失败，请手动运行该脚本排查。"
        )

    rebuilt = (
        len(list(out_dir.glob("*.h5")))
        if out_dir.is_dir() else 0
    )
    if rebuilt < expected:
        raise RuntimeError(
            f"重建后仍不完整：h5={rebuilt}/{expected}"
        )
    print(
        f"重建完成：{rebuilt} 个未见工况真值已就绪。"
    )


def run():
    print("当前运行脚本：", Path(__file__).resolve())
    print("当前配置文件：", Path(cfg.__file__).resolve())
    print("COEFFICIENT_SOLVER：", cfg.COEFFICIENT_SOLVER)
    print("RIDGE_LAMBDA：", cfg.RIDGE_LAMBDA)
    print("=" * 100)
    print("SUBOFF稀疏POD压力重构SCI对照实验")
    print("=" * 100)

    topology = load_topology()

    extracted_sensor_matrix = None

    if USE_EXTRACTED_SENSOR_MATRIX:
        extracted_sensor_matrix = (
            load_extracted_sensor_matrix(
                EXTRACTED_SENSOR_MATRIX_PATH
            )
        )

    train_df, train_snapshots = (
        load_training_data(topology)
    )

    # 【健壮性守卫】真值缺失时自动重建
    ensure_unseen_cases_built()

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

        target_velocity = get_velocity_vector(
            unseen_row
        )
        global CURRENT_STAR_SENSOR_DATA

        CURRENT_STAR_SENSOR_DATA = {}

        for sensor_method in [
            "practical",
        ]:
            if (
                    sensor_method == "practical"
                    and USE_EXTRACTED_SENSOR_MATRIX
            ):
                CURRENT_STAR_SENSOR_DATA[
                    sensor_method
                ] = load_case_sensor_data_from_matrix(
                    unseen_row=unseen_row,
                    method=sensor_method,
                    topology=topology,
                    sensor_matrix=extracted_sensor_matrix,
                )

            else:
                # 兼容旧的逐工况STAR传感器CSV流程。
                CURRENT_STAR_SENSOR_DATA[
                    sensor_method
                ] = load_case_star_sensor_data(
                    unseen_row=unseen_row,
                    method=sensor_method,
                    topology=topology,
                )

            sensor_data = CURRENT_STAR_SENSOR_DATA[
                sensor_method
            ]

            print(
                f"{sensor_method.upper()}工程测点："
                f"{len(sensor_data['cp'])}个，"
                f"数据来源："
                f"{sensor_data.get('observation_source', 'legacy_star_csv')}"
            )

            if np.isfinite(
                    sensor_data.get(
                        "max_star_sensor_distance_m",
                        np.nan,
                    )
            ):
                print(
                    "  STAR到定义最大坐标误差："
                    f"{sensor_data['max_star_sensor_distance_m']:.6e} m"
                )

            if np.isfinite(
                    sensor_data.get(
                        "max_sensor_topology_distance_m",
                        np.nan,
                    )
            ):
                print(
                    "  定义到topology最大坐标误差："
                    f"{sensor_data['max_sensor_topology_distance_m']:.6e} m"
                )


        target_speed = float(
            np.linalg.norm(target_velocity)
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
            target_velocity,
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
        # 10.0 参数到压力场的RBF预测
        # ---------------------------------------------------------------------
        parameter_prediction = None
        parameter_result = None

        if getattr(
                cfg,
                "RUN_PARAMETER_RBF_EXPERIMENT",
                False,
        ):
            parameter_result = (
                fit_rbf_parameter_model(
                    train_cases=local_train_df,
                    train_snapshots=local_train_snapshots,
                    target_case=unseen_row,
                    pod=pod,
                    area=topology["area"],
                    feature_set=(
                        cfg.PARAMETER_FEATURE_SET
                    ),
                    length_scale=(
                        cfg.PARAMETER_RBF_LENGTH_SCALE
                    ),
                    noise=(
                        cfg.PARAMETER_RBF_NOISE
                    ),
                    ridge_lambda=(
                        cfg.PARAMETER_RBF_RIDGE
                    ),
                    mode_count=(
                        cfg.PARAMETER_MODEL_MODE_COUNT
                    ),
                )
            )

            parameter_prediction = (
                parameter_result["prediction"]
            )

            parameter_metrics = calculate_metrics(
                truth,
                parameter_prediction,
                topology["area"],
            )

            parameter_row = {
                "experiment": (
                    "parameter_rbf_prediction"
                ),
                "case_id": case_id,
                "u_inf_m_s": float(
                    target_velocity[0]
                ),
                "v_inf_m_s": float(
                    target_velocity[1]
                ),
                "w_inf_m_s": float(
                    target_velocity[2]
                ),
                "speed_m_s": target_speed,
                "split_type": split_type,
                "part_name": cfg.PART_NAME,
                "method": "rbf_kernel_ridge",
                "feature_set": (
                    cfg.PARAMETER_FEATURE_SET
                ),
                "sensor_count": 0,
                "requested_sensor_count": 0,
                "mode_count": (
                    parameter_result["mode_count"]
                ),
                "requested_mode_count": (
                    cfg.PARAMETER_MODEL_MODE_COUNT
                ),
                "noise_level": 0.0,
                "noise_percent": 0.0,
                "repeat": 0,
                "sampling_rank": 0,
                "condition_number": np.nan,
                "scaled_condition_number": np.nan,
                "runtime_ms": np.nan,
                "parameter_rbf_length_scale": (
                    parameter_result["length_scale"]
                ),
                "parameter_rbf_noise": (
                    parameter_result["noise"]
                ),
                "parameter_rbf_ridge": (
                    parameter_result["ridge_lambda"]
                ),
            }

            parameter_row.update(
                parameter_metrics
            )

            all_rows.append(
                parameter_row
            )

            if getattr(
                    cfg,
                    "SAVE_PARAMETER_PREDICTIONS",
                    False,
            ):
                parameter_table = pd.DataFrame({
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
                    "cp_static_parameter_prediction": (
                        parameter_prediction
                    ),
                    "cp_error": (
                            parameter_prediction
                            - truth
                    ),
                })

                parameter_table.to_csv(
                    cfg.PREDICTION_ROOT
                    / (
                        f"{case_id}_"
                        f"{cfg.PART_NAME}_"
                        "parameter_rbf_prediction.csv"
                    ),
                    index=False,
                    encoding="utf-8-sig",
                    float_format=(
                        cfg.CSV_FLOAT_FORMAT
                    ),
                )

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
                        target_velocity,
                        topology,
                        normal_information[
                            "normal"
                        ],
                    )
                )

                pred_integral = (
                    calculate_pressure_integrals(
                        prediction,
                        target_velocity,
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
                        target_velocity,
                        topology,
                        normal_information[
                            "normal"
                        ],
                    )
                )

                pred_integral = (
                    calculate_pressure_integrals(
                        prediction,
                        target_velocity,
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
                        target_velocity=target_velocity,
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
                        target_velocity=target_velocity,
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
        # 10.4.1 固定80传感器下的系数反演算法比较
        # ---------------------------------------------------------------------

        if cfg.RUN_INVERSION_ALGORITHM_EXPERIMENT:

            inversion_sensor_count = int(
                cfg.INVERSION_SENSOR_COUNT
            )

            # 【新增】获取80个传感器的索引
            sensor_indices_80 = (
                CURRENT_STAR_SENSOR_DATA[
                    cfg.MAIN_SENSOR_METHOD
                ]["topology_indices"][:inversion_sensor_count]
            )

            for mode_count in (
                    cfg.INVERSION_MODE_COUNTS
            ):
                if mode_count > pod[
                    "basis"
                ].shape[1]:
                    continue

                # 【新增】为当前模态数计算传感器残差权重
                if cfg.USE_SENSOR_RESIDUAL_WEIGHTS:
                    current_basis = pod["basis"][
                        :, :mode_count
                    ]

                    sensor_weights = (
                        compute_sensor_residual_weights_by_mode(
                            train_snapshots=local_train_snapshots,
                            mean_field=pod["mean_field"],
                            basis=current_basis,
                            sensor_indices=sensor_indices_80,
                            area=topology["area"],
                        )
                    )
                else:
                    sensor_weights = None

                for solver_name in (
                        cfg.INVERSION_SOLVERS
                ):
                    lambda_values = (
                        cfg.INVERSION_LAMBDAS
                        if solver_name != "lstsq"
                        else [0.0]
                    )

                    for ridge_lambda in (
                            lambda_values
                    ):
                        row, _prediction, _ids = (
                            evaluate_sparse_method(
                                case_id=case_id,
                                target_speed=target_speed,
                                target_velocity=target_velocity,
                                split_type=split_type,
                                truth=truth,
                                pod=pod,
                                topology=topology,
                                sensor_method=(
                                    cfg.MAIN_SENSOR_METHOD
                                ),
                                sensor_count=(
                                    inversion_sensor_count
                                ),
                                mode_count=mode_count,
                                noise_level=0.0,
                                repeat=0,
                                rng=master_rng,
                                experiment_name=(
                                    "inversion_algorithm"
                                ),
                                solver_name=solver_name,
                                ridge_lambda=ridge_lambda,
                                pod_variances=pod.get(
                                    "pod_variances",
                                    None,
                                ),
                                sensor_weights=sensor_weights,  # 【新增】传入权重
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
                            target_velocity=target_velocity,
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
                            target_velocity=target_velocity,
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
        # ---------------------------------------------------------
        # 10.7 【改进】主重构：速度参数先验 + 传感器残差权重 + 最优求解
        # ---------------------------------------------------------
        main_mode_eff = min(
            cfg.MAIN_MODE_COUNT,
            pod["basis"].shape[1],
        )
        main_basis = pod["basis"][:, :main_mode_eff]

        main_sensor_idx = CURRENT_STAR_SENSOR_DATA[
            cfg.MAIN_SENSOR_METHOD
        ]["topology_indices"][:cfg.MAIN_SENSOR_COUNT]

        # 传感器残差权重（提升可靠测点、抑制低质量测点）
        main_sensor_weights = None
        if cfg.MAIN_USE_SENSOR_WEIGHTS:
            main_sensor_weights = (
                compute_sensor_residual_weights_by_mode(
                    train_snapshots=local_train_snapshots,
                    mean_field=pod["mean_field"],
                    basis=main_basis,
                    sensor_indices=main_sensor_idx,
                    area=topology["area"],
                )
            )

        # 速度参数先验（prior_ridge）：参数空间平滑性约束系数
        main_prior = None
        main_prior_lambda = 0.0
        main_solver = cfg.MAIN_SOLVER_FALLBACK
        if cfg.MAIN_USE_VELOCITY_PRIOR:
            main_prior = build_velocity_prior_coefficients(
                train_cases=local_train_df,
                train_snapshots=local_train_snapshots,
                pod=pod,
                target_velocity=target_velocity,
                area=topology["area"],
                mode_count=main_mode_eff,
                feature_degree=cfg.VELOCITY_PRIOR_FEATURE_DEGREE,
                ridge=cfg.VELOCITY_PRIOR_RIDGE,
            )
            main_prior_lambda = cfg.MAIN_PRIOR_LAMBDA
            main_solver = "prior_ridge"

        main_row, main_prediction, (
            main_sensor_indices
        ) = evaluate_sparse_method(
            case_id=case_id,
            target_speed=target_speed,
            target_velocity=target_velocity,
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
            solver_name=main_solver,
            ridge_lambda=cfg.RIDGE_LAMBDA,
            pod_variances=pod.get(
                "pod_variances",
                None,
            ),
            sensor_weights=main_sensor_weights,
            coefficient_prior=main_prior,
            prior_lambda=main_prior_lambda,
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

            main_sensor_method = str(
                cfg.MAIN_SENSOR_METHOD
            ).lower()

            if main_sensor_method in (
                    CURRENT_STAR_SENSOR_DATA
            ):
                main_observed_cp = (
                    CURRENT_STAR_SENSOR_DATA[
                        main_sensor_method
                    ]["cp"][
                        :len(main_sensor_indices)
                    ]
                )
            else:
                main_observed_cp = truth[
                    main_sensor_indices
                ]

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

                "cp_static_observed": (
                    main_observed_cp
                ),

                "sensor_layout": (
                    "practical"
                ),
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
                    target_velocity,
                    topology,
                    normal_information[
                        "normal"
                    ],
                )
            )

            prediction_integral = (
                calculate_pressure_integrals(
                    main_prediction,
                    target_velocity,
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
        if (
            normal_information is not None
            and getattr(
                cfg,
                "RUN_DEIM_EXPERIMENT",
                False,
            )
        ):
            deim_row, deim_prediction, (
                deim_sensor_indices
            ) = evaluate_sparse_method(
                case_id=case_id,
                target_speed=target_speed,
                target_velocity=target_velocity,
                split_type=split_type,
                truth=truth,
                pod=pod,
                topology=topology,
                sensor_method="deim",
                sensor_count=cfg.MAIN_SENSOR_COUNT,
                mode_count=cfg.MAIN_MODE_COUNT,
                noise_level=0.0,
                repeat=0,
                rng=master_rng,
                experiment_name="main_model_deim",
            )

            deim_prediction_integral = (
                calculate_pressure_integrals(
                    deim_prediction,
                    target_velocity,
                    topology,
                    normal_information["normal"],
                )
            )

            deim_integral_row = {
                "case_id": case_id,
                "speed_m_s": target_speed,
                "split_type": split_type,
                "part_name": cfg.PART_NAME,
                "method": "deim",
                "sensor_count": cfg.MAIN_SENSOR_COUNT,
                "mode_count": cfg.MAIN_MODE_COUNT,
                "noise_level": 0.0,
            }

            deim_integral_row.update(
                build_integral_error_row(
                    truth_integral,
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
            "inversion_algorithm",
            "inversion_algorithm_results.csv",
        ),
        (
            "noise_robustness",
            "noise_robustness_raw.csv",
        ),
        (
                "parameter_rbf_prediction",
                "parameter_rbf_prediction.csv",
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