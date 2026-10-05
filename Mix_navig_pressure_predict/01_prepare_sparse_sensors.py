# -*- coding: utf-8 -*-
"""
为三向流场稀疏重构准备传感器位置。

功能：
1. 从POD分析结果中读取DEIM/QR传感器位置
2. 验证传感器位置与拓扑一致性
3. 生成传感器坐标表
"""

from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))

import importlib

cfg = importlib.import_module("00_sparse_config")


def load_topology():
    """加载Master拓扑"""
    with h5py.File(cfg.TOPOLOGY_PATH, mode="r") as file:
        centroid = np.asarray(
            file["geometry/centroid"][()],
            dtype=np.float64,
        )

        face_area = np.asarray(
            file["geometry/face_area"][()],
            dtype=np.float64,
        )

        part_name = file["components/part_name"][()]

        part_name = np.asarray([
            value.decode() if isinstance(value, bytes) else str(value)
            for value in part_name
        ], dtype=str)

        part_name = np.asarray([
            value.strip().lower() for value in part_name
        ], dtype=str)

    return centroid, face_area, part_name


def get_sensor_source_path(sensor_method: str) -> Path:
    """获取指定传感器方法对应的源文件路径。"""
    sensor_method = str(
        sensor_method
    ).strip().lower()

    if sensor_method == "deim":
        return (
            cfg.CP_STATIC_ROOT
            / f"{cfg.PART_NAME}_deim_sensor.csv"
        )

    if sensor_method == "qr":
        return (
            cfg.CP_STATIC_ROOT
            / f"{cfg.PART_NAME}_qr_sensor.csv"
        )

    if sensor_method == "custom":
        return cfg.CUSTOM_SENSOR_XYZ_CSV

    raise ValueError(
        f"未知传感器方法：{sensor_method}"
    )


def prepare_sensors(
    sensor_method: str | None = None,
):
    """准备指定传感器方法的位置表。"""
    if sensor_method is None:
        sensor_method = str(
            cfg.SENSOR_METHOD
        ).strip().lower()
    else:
        sensor_method = str(
            sensor_method
        ).strip().lower()

    if sensor_method not in {
        "qr",
        "deim",
        "custom",
    }:
        raise ValueError(
            f"未知传感器方法：{sensor_method}"
        )

    centroid, face_area, part_name = load_topology()

    source_path = get_sensor_source_path( sensor_method )

    if not source_path.is_file():
        raise FileNotFoundError(f"传感器文件不存在：{source_path}")

    print(f"\n读取传感器位置：{source_path}")

    source = pd.read_csv(source_path, encoding="utf-8-sig")

    # 方式1：直接使用global_face_id
    if "global_face_id" in source.columns and sensor_method != "custom" :
        global_face_id = source["global_face_id"].to_numpy(dtype=np.int64)
        sensor_xyz = centroid[global_face_id]
        matched_distance = np.zeros(len(global_face_id), dtype=np.float64)

    # 方式2：通过坐标匹配
    else:
        required = {"x_m", "y_m", "z_m"}
        missing = required - set(source.columns)

        if missing:
            raise KeyError(f"传感器坐标文件缺少列：{missing}")

        sensor_xyz = source[["x_m", "y_m", "z_m"]].to_numpy(dtype=np.float64)

        tree = cKDTree(centroid)
        matched_distance, global_face_id = tree.query(sensor_xyz, k=1)
        global_face_id = np.asarray(global_face_id, dtype=np.int64)

    matched_part = part_name[global_face_id]

    # 验证部件一致性
    if cfg.PART_NAME != "all_surface":
        invalid = matched_part != cfg.PART_NAME
        if np.any(invalid):
            raise ValueError(
                f"存在测点不属于目标部件：{np.flatnonzero(invalid).tolist()}"
            )

    # 生成传感器表
    result = pd.DataFrame({
        "sensor_id": np.arange(len(global_face_id)) + 1,
        "global_face_id": global_face_id,
        "x_m": centroid[global_face_id, 0],
        "y_m": centroid[global_face_id, 1],
        "z_m": centroid[global_face_id, 2],
        "face_area_m2": face_area[global_face_id],
        "part_name": matched_part,
        "match_distance_m": matched_distance,
    })

    if sensor_method == "qr":
        expected_count = int(
            getattr(
                cfg,
                "QR_SENSOR_COUNT",
                cfg.N_SENSORS,
            )
        )
    else:
        expected_count = int(
            getattr(
                cfg,
                "DEIM_SENSOR_COUNT",
                cfg.EXPECTED_POD_MODES,
            )
        )

    if len(result) != expected_count:
        raise ValueError(
            f"{sensor_method.upper()}传感器数量不一致："
            f"配置={expected_count}，"
            f"实际={len(result)}。"
        )

    if result["global_face_id"].duplicated().any():
        raise ValueError("传感器存在重复global_face_id")

    output_path = (
            cfg.SENSOR_ROOT
            / f"{cfg.PART_NAME}_{sensor_method}_sensors.csv"
    )

    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✓ 已保存传感器表：{output_path}")
    print(f"  传感器数量：{len(result)}")
    print(f"  坐标匹配误差（最大）：{matched_distance.max():.6e} m")
    print("\n传感器位置预览：")
    print(result.head(10))

if __name__ == "__main__":
    # 一次生成QR和DEIM两套传感器文件，
    # 供03_reconstruct_from_measurements.py同时使用。
    for sensor_method in [
        "deim",
        "qr",
    ]:
        print("\n" + "=" * 80)
        print(
            f"准备{sensor_method.upper()}传感器"
        )
        print("=" * 80)

        prepare_sensors(
            sensor_method=sensor_method
        )