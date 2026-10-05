# -*- coding: utf-8 -*-
"""
将STAR-CCM+导出的六个SBD未见工况转换成统一拓扑顺序的HDF5。

输入目录结构：

SurfaceCheck/
├── v0p75/
│   ├── suboff_0000_v0p75.sbd
│   ├── DEIMSensorFromStar__v0p75.csv
│   └── QRSensorFromStar__v0p75.csv
├── v2p5/
│   └── ...
├── v4p5/
│   └── ...
├── v6p5/
│   └── ...
├── v8p5/
│   └── ...
└── v10p5/
    └── ...

输出：

results/unseen_cases/
├── suboff_0000_v0p75.h5
├── suboff_0000_v2p5.h5
├── suboff_0000_v4p5.h5
├── suboff_0000_v6p5.h5
├── suboff_0000_v8p5.h5
└── suboff_0000_v10p5.h5

以及：

Experiment/unseen_cases.csv
"""

from __future__ import annotations

from pathlib import Path
import csv
import importlib.util
import json
import sys
import traceback

import h5py
import numpy as np
import pandas as pd

# =============================================================================
# 1. 导入配置
# =============================================================================

EXPERIMENT_ROOT = Path(
    __file__
).resolve().parent

sys.path.insert(
    0,
    str(EXPERIMENT_ROOT),
)

import importlib

cfg = importlib.import_module(
    "00_experiment_config"
)

# =============================================================================
# 2. 动态加载现有数据集构建程序
# =============================================================================

def load_module_from_path(
    path: Path,
    module_name: str,
):
    """按照绝对路径加载现有Python模块。"""
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"构建程序不存在：{path}"
        )

    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            f"无法加载程序：{path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    sys.modules[module_name] = module
    specification.loader.exec_module(module)

    return module

# =============================================================================
# 3. 拓扑读取
# =============================================================================

def load_topology():
    """读取统一Master表面拓扑。"""
    with h5py.File(
        cfg.TOPOLOGY_PATH,
        "r",
    ) as file:
        centroid = np.asarray(
            file["geometry/centroid"][()],
            dtype=np.float64,
        )

        area = np.asarray(
            file["geometry/face_area"][()],
            dtype=np.float64,
        )

        normal = None

        if (
            cfg.FACE_NORMAL_DATASET
            in file
        ):
            normal = np.asarray(
                file[
                    cfg.FACE_NORMAL_DATASET
                ][()],
                dtype=np.float64,
            )

        topology_hash = file.attrs.get(
            "topology_hash_sha256",
            "",
        )

        if isinstance(
            topology_hash,
            bytes,
        ):
            topology_hash = topology_hash.decode(
                "utf-8",
                errors="replace",
            )

    if len(centroid) != len(area):
        raise ValueError(
            "topology.h5中centroid与face_area长度不一致"
        )

    if normal is not None:
        if normal.shape != (
            len(centroid),
            3,
        ):
            raise ValueError(
                f"法向量形状异常：{normal.shape}"
            )

        magnitude = np.linalg.norm(
            normal,
            axis=1,
        )

        if np.any(magnitude <= 1.0e-15):
            raise ValueError(
                "法向量中存在零向量"
            )

        normal = (
            normal
            / magnitude[:, None]
        )

        normal *= float(
            cfg.NORMAL_DIRECTION_SIGN
        )

    return {
        "centroid": centroid,
        "area": area,
        "normal": normal,
        "face_count": len(centroid),
        "topology_hash": str(topology_hash),
    }

# =============================================================================
# 4. HDF5写入
# =============================================================================

def write_unseen_hdf5(
    output_path: Path,
    case_id: str,
    speed: float,
    cp_static: np.ndarray,
    p_static_absolute: np.ndarray,
    p_total_absolute: np.ndarray,
    topology,
    sbd_path: Path,
    mapping_metadata: dict,
):
    """保存统一拓扑顺序的未见工况HDF5。"""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    q_inf = (
        0.5
        * float(cfg.RHO)
        * float(speed) ** 2
    )

    with h5py.File(
        output_path,
        "w",
    ) as file:
        file.attrs["format_name"] = (
            "SUBOFF Experiment unseen case"
        )
        file.attrs["case_id"] = case_id
        file.attrs["face_count"] = (
            topology["face_count"]
        )
        file.attrs["topology_hash_sha256"] = (
            topology["topology_hash"]
        )
        file.attrs["u_inf_m_s"] = float(speed)
        file.attrs["v_inf_m_s"] = 0.0
        file.attrs["w_inf_m_s"] = 0.0
        file.attrs["speed_inf_m_s"] = float(speed)
        file.attrs["rho_kg_m3"] = float(cfg.RHO)
        file.attrs["p_static_ref_pa"] = float(
            cfg.P_STATIC_REF_PA
        )
        file.attrs["q_inf_pa"] = q_inf
        file.attrs["sbd_path"] = str(
            sbd_path.resolve()
        )
        file.attrs["mapping_metadata_json"] = (
            json.dumps(
                mapping_metadata,
                ensure_ascii=False,
            )
        )

        geometry = file.create_group(
            "geometry"
        )

        geometry.create_dataset(
            "global_face_id",
            data=np.arange(
                topology["face_count"],
                dtype=np.int64,
            ),
        )

        fields = file.create_group(
            "fields"
        )

        fields.create_dataset(
            "cp_static",
            data=cp_static,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        fields.create_dataset(
            "p_static_absolute",
            data=p_static_absolute,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        # 如果SBD没有总压，该数组通常全部为NaN。
        fields.create_dataset(
            "p_total_absolute",
            data=p_total_absolute,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

    print(
        f"已保存统一顺序HDF5：{output_path}"
    )

# =============================================================================
# 5. 主转换流程
# =============================================================================

def build_all_unseen_cases():
    """将六个SBD逐一转换为统一拓扑顺序的HDF5。"""
    builder = load_module_from_path(
        cfg.BUILDER_SCRIPT_PATH,
        "suboff_builder_for_experiment",
    )

    topology = load_topology()

    if topology["normal"] is None:
        print(
            "警告：没有读取到表面法向量。"
            "压力场实验仍可运行，"
            "但压力积分实验需要人工补充法向量。"
        )
    else:
        print(
            "已读取表面法向量："
            f"{cfg.FACE_NORMAL_DATASET}"
        )

    legacy = builder.load_legacy_module()

    manifest_rows = []
    validation_rows = []

    for case_index, case in enumerate(
        cfg.UNSEEN_SBD_CASES,
        start=1,
    ):
        case_id = str(
            case["case_id"]
        )

        speed = float(
            case["speed_m_s"]
        )

        sbd_path = Path(
            case["sbd_path"]
        ).expanduser().resolve()

        print("\n" + "=" * 100)
        print(
            f"转换未见工况[{case_index}/"
            f"{len(cfg.UNSEEN_SBD_CASES)}]："
            f"{case_id}"
        )
        print(
            f"速度：{speed:.6f} m/s"
        )
        print(
            f"SBD：{sbd_path}"
        )

        if not sbd_path.is_file():
            raise FileNotFoundError(
                f"SBD不存在：{sbd_path}"
            )

        # 直接调用现有builder中的SBD解析函数。
        # 该函数会：
        #   1. 读取SBD场函数；
        #   2. 保留SUBOFF六个部件；
        #   3. 检查面数和字段有效性。
        sbd_data = builder.read_sbd_case(
            legacy=legacy,
            sbd_path=sbd_path,
            p_static_ref_pa=(
                float(cfg.P_STATIC_REF_PA)
            ),
        )

        if sbd_data.face_count != (
            topology["face_count"]
        ):
            raise ValueError(
                f"{case_id}面数不一致："
                f"SBD={sbd_data.face_count:,}，"
                f"topology={topology['face_count']:,}"
            )

        # 建立SBD原始行到统一拓扑global_face_id的映射。
        sbd_to_topology, mapping_metadata = (
            builder.build_sbd_to_cgns_mapping(
                sbd_data=sbd_data,
                cgns_centroids=(
                    topology["centroid"]
                ),
                cgns_areas=topology["area"],
            )
        )

        # SBD静压重排到统一拓扑顺序。
        p_static_absolute = (
            builder.reorder_to_cgns(
                sbd_data.p_static_absolute,
                sbd_to_topology,
            )
        )

        p_total_absolute = (
            builder.reorder_to_cgns(
                sbd_data.p_total_absolute,
                sbd_to_topology,
            )
        )

        q_inf = (
            0.5
            * float(cfg.RHO)
            * speed ** 2
        )

        cp_static = (
            p_static_absolute
            - float(cfg.P_STATIC_REF_PA)
        ) / q_inf

        # 严格检查。
        for field_name, values in [
            (
                "p_static_absolute",
                p_static_absolute,
            ),
            (
                "cp_static",
                cp_static,
            ),
        ]:
            if len(values) != (
                topology["face_count"]
            ):
                raise ValueError(
                    f"{case_id}/{field_name}长度异常"
                )

            if np.any(~np.isfinite(values)):
                raise ValueError(
                    f"{case_id}/{field_name}包含NaN或Inf"
                )

        output_path = (
            cfg.UNSEEN_H5_ROOT
            / f"{case_id}.h5"
        )

        write_unseen_hdf5(
            output_path=output_path,
            case_id=case_id,
            speed=speed,
            cp_static=cp_static,
            p_static_absolute=p_static_absolute,
            p_total_absolute=p_total_absolute,
            topology=topology,
            sbd_path=sbd_path,
            mapping_metadata=mapping_metadata,
        )

        deim_path = Path(
            case["deim_sensor_path"]
        ).expanduser().resolve()

        qr_path = Path(
            case["qr_sensor_path"]
        ).expanduser().resolve()

        validation_rows.append({
            "case_id": case_id,
            "speed_m_s": speed,
            "sbd_path": str(sbd_path),
            "h5_path": str(output_path),
            "sbd_face_count": int(
                sbd_data.face_count
            ),
            "topology_face_count": int(
                topology["face_count"]
            ),
            "mapping_method": mapping_metadata.get(
                "method",
                "unknown",
            ),
            "deim_sensor_file_exists": (
                deim_path.is_file()
            ),
            "qr_sensor_file_exists": (
                qr_path.is_file()
            ),
            "cp_min": float(
                np.min(cp_static)
            ),
            "cp_max": float(
                np.max(cp_static)
            ),
            "cp_mean": float(
                np.mean(cp_static)
            ),
            "cp_std": float(
                np.std(cp_static)
            ),
        })

        manifest_rows.append({
            "case_id": case_id,
            "speed_m_s": speed,
            "u_inf_m_s": speed,
            "v_inf_m_s": 0.0,
            "w_inf_m_s": 0.0,
            "truth_path": str(
                output_path.resolve()
            ),
            "case_h5_path": str(
                output_path.resolve()
            ),
            "split_type": str(
                case["split_type"]
            ),
            "source_sbd_path": str(sbd_path),
            "deim_sensor_path": str(deim_path),
            "qr_sensor_path": str(qr_path),
        })

        del sbd_data

    manifest_path = (
        EXPERIMENT_ROOT
        / "unseen_cases.csv"
    )

    pd.DataFrame(
        manifest_rows
    ).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    validation_path = (
        cfg.LOG_ROOT
        / "unseen_sbd_conversion_validation.csv"
    )

    pd.DataFrame(
        validation_rows
    ).to_csv(
        validation_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "status": "success",
        "case_count": len(
            manifest_rows
        ),
        "topology_path": str(
            cfg.TOPOLOGY_PATH
        ),
        "topology_face_count": int(
            topology["face_count"]
        ),
        "topology_hash": topology[
            "topology_hash"
        ],
        "normal_dataset": (
            cfg.FACE_NORMAL_DATASET
            if topology["normal"] is not None
            else None
        ),
        "manifest_path": str(
            manifest_path
        ),
        "validation_path": str(
            validation_path
        ),
    }

    summary_path = (
        cfg.LOG_ROOT
        / "unseen_sbd_conversion_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("六个SBD未见工况转换完成")
    print(f"HDF5目录：{cfg.UNSEEN_H5_ROOT}")
    print(f"清单文件：{manifest_path}")
    print(f"检查表：{validation_path}")
    print("=" * 100)

if __name__ == "__main__":
    try:
        build_all_unseen_cases()
    except Exception as exc:
        print("\nSBD转换失败：")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        sys.exit(1)