# -*- coding: utf-8 -*-

from pathlib import Path
import importlib
import importlib.util
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
cfg = importlib.import_module("00_sparse_config")

def load_module(path: Path, name: str):
    path = path.expanduser().resolve()
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"无法加载程序：{path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module

def decode_part_names(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8").strip().lower()
        if isinstance(value, bytes)
        else str(value).strip().lower()
        for value in values
    ])

def prepare_truth():
    builder = load_module(
        cfg.DATASET_BUILDER_PATH,
        "modal_dataset_builder_for_truth",
    )
    legacy = builder.load_legacy_module()

    with h5py.File(cfg.TOPOLOGY_PATH, "r") as file:
        centroid = np.asarray(
            file["geometry/centroid"][()], dtype=np.float64
        )
        area = np.asarray(
            file["geometry/face_area"][()], dtype=np.float64
        )
        part_name = decode_part_names(
            file["components/part_name"][()]
        )
        topology_hash = file.attrs["topology_hash_sha256"]
        if isinstance(topology_hash, bytes):
            topology_hash = topology_hash.decode("utf-8")

    speed = float(np.sqrt(
        cfg.TEST_U_INF ** 2
        + cfg.TEST_V_INF ** 2
        + cfg.TEST_W_INF ** 2
    ))
    if speed <= 0.0:
        raise ValueError("测试工况速度必须大于零")

    sbd_data = builder.read_sbd_case(
        legacy=legacy,
        sbd_path=cfg.TEST_SBD_PATH,
        p_static_ref_pa=float(cfg.P_STATIC_REF_PA),
    )

    mapping, mapping_metadata = builder.build_sbd_to_cgns_mapping(
        sbd_data=sbd_data,
        cgns_centroids=centroid,
        cgns_areas=area,
    )

    p_absolute = builder.reorder_to_cgns(
        sbd_data.p_static_absolute,
        mapping,
    )
    q_inf = 0.5 * float(cfg.RHO) * speed ** 2
    cp_static = (
        p_absolute - float(cfg.P_STATIC_REF_PA)
    ) / q_inf

    if cfg.PART_NAME == "all_surface":
        selected_global_ids = np.arange(len(centroid), dtype=np.int64)
    else:
        selected_global_ids = np.flatnonzero(
            part_name == cfg.PART_NAME
        ).astype(np.int64)

    truth = pd.DataFrame({
        "global_face_id": selected_global_ids,
        "x_m": centroid[selected_global_ids, 0],
        "y_m": centroid[selected_global_ids, 1],
        "z_m": centroid[selected_global_ids, 2],
        "face_area_m2": area[selected_global_ids],
        "cp_static_starccm": cp_static[selected_global_ids],
        "p_static_absolute_starccm_pa": (
            p_absolute[selected_global_ids]
        ),
    })

    truth_path = (
        cfg.TRUTH_ROOT
        / f"{cfg.TEST_CASE_ID}_{cfg.PART_NAME}_truth.csv"
    )
    truth.to_csv(truth_path, index=False, encoding="utf-8-sig")

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

    measurement_path = (
            Path(__file__).parent
            / (
                f"measurements_unseen_speed_"
                f"{sensor_method}.csv"
            )
    )
    sensors = pd.read_csv(sensor_path, encoding="utf-8-sig")
    sensor_ids = sensors["global_face_id"].to_numpy(dtype=np.int64)

    measurements = pd.DataFrame({
        "sensor_id": sensors["sensor_id"].to_numpy(dtype=np.int64),
        "global_face_id": sensor_ids,
        "pressure_value": p_absolute[sensor_ids],
    })

    measurement_path = (
        Path(__file__).parent / "measurements_unseen_speed.csv"
    )
    measurements.to_csv(
        measurement_path, index=False, encoding="utf-8-sig"
    )

    print(f"已保存STAR-CCM+真值：{truth_path}")
    print(f"已提取验证用测点压力：{measurement_path}")
    print(f"测试速度：{speed:.8f} m/s，q_inf={q_inf:.8f} Pa")
    print(f"目标部件面数：{len(selected_global_ids):,}")
    print(f"拓扑哈希：{topology_hash}")
    print(f"映射方法：{mapping_metadata['method']}")

if __name__ == "__main__":
    prepare_truth()