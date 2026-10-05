# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(
    0,
    str(Path(__file__).parent),
)

import importlib

cfg = importlib.import_module(
    "00_sparse_config"
)

def load_topology():
    with h5py.File(
        cfg.TOPOLOGY_PATH,
        mode="r",
    ) as file:
        centroid = np.asarray(
            file["geometry/centroid"][()],
            dtype=np.float64,
        )

        face_area = np.asarray(
            file["geometry/face_area"][()],
            dtype=np.float64,
        )

        part_name = file[
            "components/part_name"
        ][()]

        part_name = np.asarray(
            [
                value.decode()
                if isinstance(value, bytes)
                else str(value)
                for value in part_name
            ],
            dtype=str,
        )

        part_name = np.asarray(
            [
                value.strip().lower()
                for value in part_name
            ],
            dtype=str,
        )

    return centroid, face_area, part_name

def get_sensor_source_path():
    if cfg.SENSOR_METHOD == "deim":
        return (
            cfg.CP_STATIC_ROOT
            / f"{cfg.PART_NAME}_deim_sensor.csv"
        )

    if cfg.SENSOR_METHOD == "qr":
        return (
            cfg.CP_STATIC_ROOT
            / f"{cfg.PART_NAME}_qr_sensor.csv"
        )

    if cfg.SENSOR_METHOD == "custom":
        return cfg.CUSTOM_SENSOR_XYZ_CSV

    raise ValueError(
        f"未知传感器方法：{cfg.SENSOR_METHOD}"
    )

def prepare_sensors():
    sensor_method = str(
        cfg.SENSOR_METHOD
    ).strip().lower()

    centroid, face_area, part_name = (
        load_topology()
    )

    source_path = get_sensor_source_path()

    if not source_path.is_file():
        raise FileNotFoundError(
            f"传感器文件不存在：{source_path}"
        )

    source = pd.read_csv(
        source_path,
        encoding="utf-8-sig",
    )

    if (
        "global_face_id" in source.columns
        and cfg.SENSOR_METHOD != "custom"
    ):
        global_face_id = source[
            "global_face_id"
        ].to_numpy(dtype=np.int64)

        sensor_xyz = centroid[
            global_face_id
        ]

        matched_distance = np.zeros(
            len(global_face_id),
            dtype=np.float64,
        )

    else:
        required = {
            "x_m",
            "y_m",
            "z_m",
        }

        missing = required - set(
            source.columns
        )

        if missing:
            raise KeyError(
                f"传感器坐标文件缺少列：{missing}"
            )

        sensor_xyz = source[
            ["x_m", "y_m", "z_m"]
        ].to_numpy(dtype=np.float64)

        tree = cKDTree(centroid)

        matched_distance, global_face_id = (
            tree.query(
                sensor_xyz,
                k=1,
            )
        )

        global_face_id = np.asarray(
            global_face_id,
            dtype=np.int64,
        )

    matched_part = part_name[
        global_face_id
    ]

    if cfg.PART_NAME != "all_surface":
        invalid = (
            matched_part != cfg.PART_NAME
        )

        if np.any(invalid):
            raise ValueError(
                "存在测点不属于目标部件："
                f"{np.flatnonzero(invalid).tolist()}"
            )

    result = pd.DataFrame({
        "sensor_id": np.arange(
            len(global_face_id)
        ) + 1,
        "global_face_id": global_face_id,
        "x_m": centroid[
            global_face_id,
            0,
        ],
        "y_m": centroid[
            global_face_id,
            1,
        ],
        "z_m": centroid[
            global_face_id,
            2,
        ],
        "face_area_m2": face_area[
            global_face_id
        ],
        "part_name": matched_part,
        "match_distance_m": matched_distance,
    })

    if len(result) != cfg.N_SENSORS:
        print(
            f"警告：配置传感器数为"
            f"{cfg.N_SENSORS}，"
            f"实际读取到{len(result)}个"
        )

    if result["global_face_id"].duplicated().any():
        raise ValueError(
            "传感器存在重复global_face_id"
        )


    output_path = (
            cfg.SENSOR_ROOT
            / (
                f"{cfg.PART_NAME}_"
                f"{sensor_method}_sensors.csv"
            )
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"已保存稀疏传感器表：{output_path}"
    )

    print(result)

if __name__ == "__main__":
    prepare_sensors()