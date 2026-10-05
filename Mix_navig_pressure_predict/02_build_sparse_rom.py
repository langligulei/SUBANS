# -*- coding: utf-8 -*-
"""
为三向流场构建稀疏POD降阶模型（ROM）。

功能：
1. 加载POD模态基
2. 提取传感器位置对应的模态子矩阵
3. 计算传感器矩阵条件数
4. 保存ROM到HDF5
"""

from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import importlib

cfg = importlib.import_module("00_sparse_config")


def load_pod_model():
    """加载POD模型"""
    if not cfg.POD_H5_PATH.is_file():
        raise FileNotFoundError(f"POD文件不存在：{cfg.POD_H5_PATH}")

    print(f"\n加载POD模型：{cfg.POD_H5_PATH}")

    with h5py.File(cfg.POD_H5_PATH, mode="r") as file:
        mean_field = np.asarray(
            file["fields/mean_field"][()],
            dtype=np.float64,
        )

        basis = np.asarray(
            file["fields/basis"][()],
            dtype=np.float64,
        )

        global_face_id = np.asarray(
            file["index/global_face_id"][()],
            dtype=np.int64,
        )

        target_field = file.attrs["target_field"]
        if isinstance(target_field, bytes):
            target_field = target_field.decode()

        topology_hash = file.attrs["topology_hash_sha256"]
        if isinstance(topology_hash, bytes):
            topology_hash = topology_hash.decode()

    print(f"  部件面数：{len(global_face_id):,}")
    print(f"  POD模态数：{basis.shape[1]}")
    print(f"  目标场：{target_field}")

    return {
        "mean_field": mean_field,
        "basis": basis,
        "global_face_id": global_face_id,
        "target_field": str(target_field),
        "topology_hash": str(topology_hash),
    }


def build_rom():
    """构建稀疏ROM"""
    pod = load_pod_model()

    sensor_method = str(cfg.SENSOR_METHOD).strip().lower()

    sensor_path = (
            cfg.SENSOR_ROOT
            / f"{cfg.PART_NAME}_{sensor_method}_sensors.csv"
    )

    if not sensor_path.is_file():
        raise FileNotFoundError(
            f"传感器文件不存在：{sensor_path}\n"
            f"请先运行 01_prepare_sparse_sensors.py"
        )

    sensors = pd.read_csv(sensor_path, encoding="utf-8-sig")
    sensor_global_ids = sensors["global_face_id"].to_numpy(dtype=np.int64)

    print(f"\n构建ROM：")
    print(f"  传感器数量：{len(sensor_global_ids)}")

    # 建立global_face_id → local_index映射
    lookup = {
        int(global_id): local_index
        for local_index, global_id in enumerate(pod["global_face_id"])
    }

    missing = [
        int(global_id)
        for global_id in sensor_global_ids
        if int(global_id) not in lookup
    ]

    if missing:
        raise ValueError(
            f"传感器global_face_id不属于POD部件：{missing}"
        )

    sensor_local_indices = np.asarray([
        lookup[int(global_id)]
        for global_id in sensor_global_ids
    ], dtype=np.int64)

    basis = pod["basis"]
    mean_field = pod["mean_field"]

    sensor_basis = basis[sensor_local_indices, :]
    sensor_mean = mean_field[sensor_local_indices]

    n_modes = basis.shape[1]
    n_sensors = len(sensor_local_indices)

    if sensor_method == "qr":
        expected_sensor_count = int(
            getattr(
                cfg,
                "QR_SENSOR_COUNT",
                cfg.N_SENSORS,
            )
        )
    else:
        expected_sensor_count = int(
            getattr(
                cfg,
                "DEIM_SENSOR_COUNT",
                cfg.EXPECTED_POD_MODES,
            )
        )

    expected_pod_modes = int(
        getattr(
            cfg,
            "EXPECTED_POD_MODES",
            n_modes,
        )
    )

    if n_sensors != expected_sensor_count:
        raise ValueError(
            f"ROM传感器数量不符合配置："
            f"实际={n_sensors}，"
            f"期望={expected_sensor_count}"
        )

    if n_modes != expected_pod_modes:
        raise ValueError(
            f"ROM模态数量不符合配置："
            f"实际={n_modes}，"
            f"期望={expected_pod_modes}"
        )

    if n_sensors < n_modes:
        raise ValueError(
            f"传感器数量({n_sensors})小于POD模态数"
            f"({n_modes})，不能唯一求解POD系数。"
        )

    condition_number = np.linalg.cond(sensor_basis)
    singular_values = np.linalg.svd(
        sensor_basis,
        compute_uv=False,
    )
    print("\n不同POD模态数下的传感矩阵条件数：")

    for test_mode_count in [
        5,
        10,
        15,
        20,
    ]:
        if test_mode_count <= sensor_basis.shape[1]:
            truncated_sensor_basis = (
                sensor_basis[:, :test_mode_count]
            )

            truncated_condition_number = np.linalg.cond(
                truncated_sensor_basis
            )

            print(
                f"  modes={test_mode_count:3d}, "
                f"condition_number="
                f"{truncated_condition_number:.6e}"
            )

    if len(singular_values) == 0:
        raise ValueError(
            "sensor_basis没有奇异值"
        )

    sigma_max = float(
        singular_values[0]
    )
    sigma_min = float(
        singular_values[-1]
    )

    rank_tolerance = (
            np.finfo(np.float64).eps
            * max(sensor_basis.shape)
            * sigma_max
    )

    numerical_rank = int(
        np.count_nonzero(
            singular_values > rank_tolerance
        )
    )

    print(
        f"  传感矩阵形状：{sensor_basis.shape}"
    )
    print(
        f"  传感矩阵数值秩："
        f"{numerical_rank}/{n_modes}"
    )
    print(
        f"  最大奇异值：{sigma_max:.6e}"
    )
    print(
        f"  最小奇异值：{sigma_min:.6e}"
    )

    if numerical_rank < n_modes:
        raise ValueError(
            "传感矩阵不满列秩，不能稳定求解全部POD系数"
        )

    if not np.isfinite(condition_number):
        raise ValueError(
            "传感矩阵条件数不是有限值"
        )

    print(f"  传感器矩阵条件数：{condition_number:.6e}")

    if condition_number > 1e6:
        print(f"  ⚠ 警告：条件数较大，可能导致数值不稳定")

    rom = {
        "part_name": cfg.PART_NAME,
        "target_field": pod["target_field"],
        "topology_hash": pod["topology_hash"],
        "global_face_id": pod["global_face_id"],
        "sensor_global_face_id": sensor_global_ids,
        "sensor_local_indices": sensor_local_indices,
        "mean_field": mean_field,
        "basis": basis,
        "sensor_mean": sensor_mean,
        "sensor_basis": sensor_basis,
        "n_faces": int(basis.shape[0]),
        "n_modes": int(basis.shape[1]),
        "n_sensors": int(len(sensor_local_indices)),
        "condition_number": float(condition_number),
    }

    output_path = (
            cfg.ROM_ROOT
            / f"{cfg.PART_NAME}_{sensor_method}_sparse_rom.h5"
    )

    with h5py.File(output_path, mode="w") as file:
        file.attrs["part_name"] = rom["part_name"]
        file.attrs["sensor_method"] = sensor_method
        file.attrs["target_field"] = rom["target_field"]
        file.attrs["topology_hash"] = rom["topology_hash"]
        file.attrs["n_faces"] = rom["n_faces"]
        file.attrs["n_modes"] = rom["n_modes"]
        file.attrs["n_sensors"] = rom["n_sensors"]
        file.attrs["condition_number"] = rom["condition_number"]

        # 三向速度标识
        file.attrs["velocity_space_dimension"] = 3
        file.attrs["test_case_id"] = cfg.TEST_CASE_ID
        file.attrs["test_u_inf"] = cfg.TEST_U_INF
        file.attrs["test_v_inf"] = cfg.TEST_V_INF
        file.attrs["test_w_inf"] = cfg.TEST_W_INF

        file.create_dataset("global_face_id", data=rom["global_face_id"])
        file.create_dataset("sensor_global_face_id", data=rom["sensor_global_face_id"])
        file.create_dataset("sensor_local_indices", data=rom["sensor_local_indices"])

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

        file.create_dataset("sensor_mean", data=rom["sensor_mean"])
        file.create_dataset("sensor_basis", data=rom["sensor_basis"])

    print(f"\n✓ 已建立稀疏ROM：{output_path}")
    print(f"  表面面数：{basis.shape[0]:,}")
    print(f"  POD模态数：{basis.shape[1]}")
    print(f"  传感器数：{len(sensor_local_indices)}")
    print(f"  条件数：{condition_number:.6e}")


if __name__ == "__main__":
    build_rom()