# -*- coding: utf-8 -*-

from pathlib import Path
import json
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).parent),
)

import importlib

cfg = importlib.import_module(
    "00_sparse_config"
)

def load_pod_model():
    if not cfg.POD_H5_PATH.is_file():
        raise FileNotFoundError(
            f"POD文件不存在：{cfg.POD_H5_PATH}"
        )

    with h5py.File(
        cfg.POD_H5_PATH,
        mode="r",
    ) as file:
        mean_field = np.asarray(
            file["fields/mean_field"][()],
            dtype=np.float64,
        )

        basis = np.asarray(
            file["fields/basis"][()],
            dtype=np.float64,
        )

        global_face_id = np.asarray(
            file[
                "index/global_face_id"
            ][()],
            dtype=np.int64,
        )

        target_field = file.attrs[
            "target_field"
        ]

        if isinstance(
            target_field,
            bytes,
        ):
            target_field = target_field.decode()

        topology_hash = file.attrs[
            "topology_hash_sha256"
        ]

        if isinstance(
            topology_hash,
            bytes,
        ):
            topology_hash = topology_hash.decode()

    return {
        "mean_field": mean_field,
        "basis": basis,
        "global_face_id": global_face_id,
        "target_field": str(target_field),
        "topology_hash": str(topology_hash),
    }

def build_rom():
    pod = load_pod_model()

    sensor_method = str(
        cfg.SENSOR_METHOD
    ).strip().lower()

    sensor_path = (
            cfg.SENSOR_ROOT
            / (
                f"{cfg.PART_NAME}_"
                f"{sensor_method}_sensors.csv"
            )
    )

    sensors = pd.read_csv(
        sensor_path,
        encoding="utf-8-sig",
    )

    sensor_global_ids = sensors[
        "global_face_id"
    ].to_numpy(dtype=np.int64)

    lookup = {
        int(global_id): local_index
        for local_index, global_id in enumerate(
            pod["global_face_id"]
        )
    }

    missing = [
        int(global_id)
        for global_id in sensor_global_ids
        if int(global_id) not in lookup
    ]

    if missing:
        raise ValueError(
            "传感器global_face_id不属于POD部件："
            f"{missing}"
        )

    sensor_local_indices = np.asarray(
        [
            lookup[int(global_id)]
            for global_id in sensor_global_ids
        ],
        dtype=np.int64,
    )

    basis = pod["basis"]
    mean_field = pod["mean_field"]

    sensor_basis = basis[
        sensor_local_indices,
        :
    ]

    sensor_mean = mean_field[
        sensor_local_indices
    ]

    n_modes = basis.shape[1]

    if len(sensor_local_indices) < n_modes:
        print(
            "警告：传感器数量小于POD模态数，"
            "系数求解为欠定问题。"
        )

    condition_number = np.linalg.cond(
        sensor_basis
    )

    rom = {
        "part_name": cfg.PART_NAME,
        "target_field": pod["target_field"],
        "topology_hash": pod["topology_hash"],
        "global_face_id": pod[
            "global_face_id"
        ],
        "sensor_global_face_id": sensor_global_ids,
        "sensor_local_indices": sensor_local_indices,
        "mean_field": mean_field,
        "basis": basis,
        "sensor_mean": sensor_mean,
        "sensor_basis": sensor_basis,
        "n_faces": int(basis.shape[0]),
        "n_modes": int(basis.shape[1]),
        "n_sensors": int(
            len(sensor_local_indices)
        ),
        "condition_number": float(
            condition_number
        ),
    }

    output_path = (
            cfg.ROM_ROOT
            / (
                f"{cfg.PART_NAME}_"
                f"{sensor_method}_sparse_rom.h5"
            )
    )

    with h5py.File(
        output_path,
        mode="w",
    ) as file:
        file.attrs["part_name"] = (
            rom["part_name"]
        )
        file.attrs["sensor_method"] = (
            sensor_method
        )
        file.attrs["target_field"] = (
            rom["target_field"]
        )
        file.attrs["topology_hash"] = (
            rom["topology_hash"]
        )
        file.attrs["n_faces"] = (
            rom["n_faces"]
        )
        file.attrs["n_modes"] = (
            rom["n_modes"]
        )
        file.attrs["n_sensors"] = (
            rom["n_sensors"]
        )
        file.attrs["condition_number"] = (
            rom["condition_number"]
        )

        file.create_dataset(
            "global_face_id",
            data=rom["global_face_id"],
        )

        file.create_dataset(
            "sensor_global_face_id",
            data=rom["sensor_global_face_id"],
        )

        file.create_dataset(
            "sensor_local_indices",
            data=rom["sensor_local_indices"],
        )

        file.create_dataset(
            "mean_field",
            data=rom["mean_field"],
            compression="gzip",
            compression_opts=4,
        )

        file.create_dataset(
            "basis",
            data=rom["basis"],
            compression="gzip",
            compression_opts=4,
        )

        file.create_dataset(
            "sensor_mean",
            data=rom["sensor_mean"],
        )

        file.create_dataset(
            "sensor_basis",
            data=rom["sensor_basis"],
        )

    print(
        f"已建立稀疏POD模型：{output_path}"
    )

    print(
        f"部件：{cfg.PART_NAME}"
    )
    print(
        f"表面面数：{basis.shape[0]:,}"
    )
    print(
        f"POD模态数：{basis.shape[1]}"
    )
    print(
        f"传感器数：{len(sensor_local_indices)}"
    )
    print(
        f"传感矩阵条件数：{condition_number:.6e}"
    )

if __name__ == "__main__":
    build_rom()