# -*- coding: utf-8 -*-
"""
检查SUBOFF稀疏ROM对照实验所需的数据。

检查内容
--------
1. topology.h5是否存在；
2. cases.csv是否存在；
3. 训练工况HDF5是否完整；
4. 未见工况是否完整；
5. 每个压力场面数是否与Master拓扑一致；
6. 是否存在NaN或Inf；
7. 未见速度是否混入训练集；
8. topology.h5中是否存在表面法向量；
9. 输出HDF5结构清单，便于检查法向量字段。

执行
----
python 02_validate_experiment_data.py
"""

from pathlib import Path
import importlib
import json
import sys

import h5py
import numpy as np
import pandas as pd

# =============================================================================
# 1. 导入配置
# =============================================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

cfg = importlib.import_module(
    "00_experiment_config"
)

# =============================================================================
# 2. 工具函数
# =============================================================================

def decode_string_array(values):
    """将HDF5字符串数组统一转换为Python字符串数组。"""
    return np.asarray([
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
        for value in values
    ], dtype=str)

def resolve_path(path_value, base_directory):
    """
    解析绝对或相对路径。

    相对路径默认相对于CSV文件所在目录。
    """
    path = Path(str(path_value)).expanduser()

    if not path.is_absolute():
        path = base_directory / path

    return path.resolve()

def calculate_speed(table):
    """根据u、v、w计算来流速度模长。"""
    u = pd.to_numeric(
        table["u_inf_m_s"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    v = pd.to_numeric(
        table["v_inf_m_s"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    w = pd.to_numeric(
        table["w_inf_m_s"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    return np.sqrt(
        u ** 2 + v ** 2 + w ** 2
    )

def list_hdf5_datasets(h5_path):
    """
    返回HDF5中全部数据集的名称、形状和类型。
    """
    rows = []

    with h5py.File(h5_path, "r") as file:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                rows.append({
                    "dataset": name,
                    "shape": str(obj.shape),
                    "dtype": str(obj.dtype),
                })

        file.visititems(visitor)

    return rows

def detect_face_normal(topology_path, face_count):
    """
    自动查找表面法向量。

    返回
    ----
    dataset_name:
        找到的数据集路径；未找到则为None。

    normal:
        shape=(n_faces,3)；未找到则为None。
    """
    candidates = []

    if cfg.FACE_NORMAL_DATASET:
        candidates.append(
            cfg.FACE_NORMAL_DATASET
        )

    candidates.extend(
        cfg.FACE_NORMAL_CANDIDATES
    )

    # 去重并保持顺序。
    candidates = list(
        dict.fromkeys(candidates)
    )

    with h5py.File(topology_path, "r") as file:
        for name in candidates:
            if name not in file:
                continue

            values = np.asarray(
                file[name][()],
                dtype=np.float64,
            )

            if values.shape == (face_count, 3):
                magnitude = np.linalg.norm(
                    values,
                    axis=1,
                )

                if np.all(np.isfinite(values)) and np.all(
                    magnitude > np.finfo(float).tiny
                ):
                    normal = (
                        values
                        / magnitude[:, None]
                    )
                    return name, normal

    return None, None

def load_truth_field(path, face_count):
    """
    读取HDF5或CSV形式的完整Cp真值。
    """
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

            values = np.asarray(
                file[field_path][()],
                dtype=np.float64,
            ).ravel()

    elif suffix == ".csv":
        table = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        field_candidates = [
            cfg.TARGET_FIELD,
            "cp_static_starccm",
            "cp_static_truth",
            "cp_static_reconstructed",
        ]

        field_column = next(
            (
                name
                for name in field_candidates
                if name in table.columns
            ),
            None,
        )

        if field_column is None:
            raise KeyError(
                f"{path}没有Cp字段。"
                f"允许字段：{field_candidates}"
            )

        if "global_face_id" in table.columns:
            values = np.full(
                face_count,
                np.nan,
                dtype=np.float64,
            )

            ids = table[
                "global_face_id"
            ].to_numpy(dtype=np.int64)

            values[ids] = table[
                field_column
            ].to_numpy(dtype=np.float64)

        else:
            values = table[
                field_column
            ].to_numpy(dtype=np.float64)

    else:
        raise ValueError(
            f"暂不支持真值文件格式：{path.suffix}"
        )

    return values

# =============================================================================
# 3. 主检查
# =============================================================================

def validate():
    errors = []
    warnings = []

    print("=" * 90)
    print("SUBOFF Experiment数据检查")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # 检查基础文件
    # -------------------------------------------------------------------------
    for path, name in [
        (cfg.TOPOLOGY_PATH, "topology.h5"),
        (cfg.TRAIN_CASES_CSV_PATH, "训练cases.csv"),
        (cfg.UNSEEN_CASES_CSV_PATH, "unseen_cases.csv"),
    ]:
        if not Path(path).is_file():
            errors.append(
                f"{name}不存在：{path}"
            )

    if errors:
        raise FileNotFoundError(
            "\n".join(errors)
        )

    # -------------------------------------------------------------------------
    # 读取Master拓扑
    # -------------------------------------------------------------------------
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

        raw_part_name = file[
            "components/part_name"
        ][()]

        topology_hash = file.attrs.get(
            "topology_hash_sha256",
            "",
        )

        if isinstance(topology_hash, bytes):
            topology_hash = topology_hash.decode(
                "utf-8",
                errors="replace",
            )

    part_name = np.asarray([
        value.strip().lower()
        for value in decode_string_array(
            raw_part_name
        )
    ])

    face_count = len(centroid)

    if centroid.shape != (face_count, 3):
        errors.append(
            f"centroid形状异常：{centroid.shape}"
        )

    if len(area) != face_count:
        errors.append(
            "face_area长度与面数不一致"
        )

    if np.any(~np.isfinite(centroid)):
        errors.append(
            "面心坐标包含NaN或Inf"
        )

    if np.any(~np.isfinite(area)) or np.any(
        area <= 0.0
    ):
        errors.append(
            "面面积包含非有限值或非正值"
        )

    if cfg.PART_NAME != "all_surface":
        if not np.any(
            part_name == cfg.PART_NAME
        ):
            errors.append(
                f"拓扑中不存在部件：{cfg.PART_NAME}"
            )

    print(f"Master拓扑面数：{face_count:,}")
    print(f"拓扑哈希：{topology_hash}")
    print(f"分析部件：{cfg.PART_NAME}")

    # -------------------------------------------------------------------------
    # 输出HDF5结构，用于检查法向量
    # -------------------------------------------------------------------------
    structure_rows = list_hdf5_datasets(
        cfg.TOPOLOGY_PATH
    )

    structure_path = (
        cfg.LOG_ROOT
        / "topology_hdf5_structure.csv"
    )

    pd.DataFrame(
        structure_rows
    ).to_csv(
        structure_path,
        index=False,
        encoding="utf-8-sig",
    )

    normal_dataset, normal = (
        detect_face_normal(
            cfg.TOPOLOGY_PATH,
            face_count,
        )
    )

    if normal is None:
        warnings.append(
            "没有自动找到表面法向量。"
            "压力积分实验将被跳过。"
            f"请查看：{structure_path}"
        )
        print("表面法向量：未找到")
    else:
        print(
            f"表面法向量：{normal_dataset}"
        )

    # -------------------------------------------------------------------------
    # 检查训练工况
    # -------------------------------------------------------------------------
    train_df = pd.read_csv(
        cfg.TRAIN_CASES_CSV_PATH,
        encoding="utf-8-sig",
    )

    train_required = {
        "case_id",
        "case_h5_path",
        *cfg.VELOCITY_COLUMNS,
    }

    missing = train_required - set(
        train_df.columns
    )

    if missing:
        errors.append(
            f"训练cases.csv缺少字段：{sorted(missing)}"
        )

    train_speeds = calculate_speed(
        train_df
    )

    train_df[
        "speed_inf_m_s_calculated"
    ] = train_speeds

    train_check_rows = []

    for row_index, row in train_df.iterrows():
        case_id = str(row["case_id"])

        path = resolve_path(
            row["case_h5_path"],
            cfg.TRAIN_CASES_CSV_PATH.parent,
        )

        status = "success"
        message = ""

        try:
            values = load_truth_field(
                path,
                face_count,
            )

            if len(values) != face_count:
                raise ValueError(
                    f"场长度{len(values):,} != "
                    f"面数{face_count:,}"
                )

            if np.any(~np.isfinite(values)):
                raise ValueError(
                    "压力场包含NaN或Inf"
                )

        except Exception as exc:
            status = "failed"
            message = repr(exc)
            errors.append(
                f"训练工况{case_id}失败：{exc}"
            )

        train_check_rows.append({
            "case_id": case_id,
            "speed_m_s": train_speeds[
                row_index
            ],
            "path": str(path),
            "status": status,
            "message": message,
        })

    pd.DataFrame(
        train_check_rows
    ).to_csv(
        cfg.LOG_ROOT
        / "training_case_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 检查未见工况
    # -------------------------------------------------------------------------
    unseen_df = pd.read_csv(
        cfg.UNSEEN_CASES_CSV_PATH,
        encoding="utf-8-sig",
    )

    unseen_required = {
        "case_id",
        "truth_path",
        "split_type",
        *cfg.VELOCITY_COLUMNS,
    }

    missing = unseen_required - set(
        unseen_df.columns
    )

    if missing:
        errors.append(
            f"unseen_cases.csv缺少字段：{sorted(missing)}"
        )

    unseen_speeds = calculate_speed(
        unseen_df
    )

    unseen_check_rows = []

    for row_index, row in unseen_df.iterrows():
        case_id = str(row["case_id"])

        truth_path = resolve_path(
            row["truth_path"],
            cfg.UNSEEN_CASES_CSV_PATH.parent,
        )

        status = "success"
        message = ""

        try:
            truth = load_truth_field(
                truth_path,
                face_count,
            )

            if len(truth) != face_count:
                raise ValueError(
                    f"真值长度{len(truth):,} != "
                    f"面数{face_count:,}"
                )

            if np.any(~np.isfinite(truth)):
                raise ValueError(
                    "完整真值包含NaN或Inf；"
                    "如果CSV仅包含某个部件，请提供全表面HDF5，"
                    "或者将PART_NAME改成与CSV一致的部件"
                )

        except Exception as exc:
            status = "failed"
            message = repr(exc)
            errors.append(
                f"未见工况{case_id}失败：{exc}"
            )

        same_speed = np.isclose(
            train_speeds,
            unseen_speeds[row_index],
            atol=cfg.SPEED_MATCH_TOLERANCE,
            rtol=0.0,
        )

        if np.any(same_speed):
            warnings.append(
                f"{case_id}的速度"
                f"{unseen_speeds[row_index]:.6f} m/s"
                "在训练cases.csv中出现。"
                "正式实验运行时将自动删除这些训练样本。"
            )

        unseen_check_rows.append({
            "case_id": case_id,
            "speed_m_s": unseen_speeds[
                row_index
            ],
            "split_type": str(
                row["split_type"]
            ),
            "truth_path": str(
                truth_path
            ),
            "same_speed_training_count": int(
                np.count_nonzero(same_speed)
            ),
            "status": status,
            "message": message,
        })

    pd.DataFrame(
        unseen_check_rows
    ).to_csv(
        cfg.LOG_ROOT
        / "unseen_case_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 保存总检查摘要
    # -------------------------------------------------------------------------
    summary = {
        "topology_path": str(
            cfg.TOPOLOGY_PATH
        ),
        "face_count": face_count,
        "part_name": cfg.PART_NAME,
        "training_case_count": len(
            train_df
        ),
        "unseen_case_count": len(
            unseen_df
        ),
        "normal_dataset": normal_dataset,
        "pressure_integral_available": (
            normal is not None
        ),
        "errors": errors,
        "warnings": warnings,
    }

    summary_path = (
        cfg.LOG_ROOT
        / "validation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "-" * 90)

    if warnings:
        print("警告：")
        for item in warnings:
            print(f"  - {item}")

    if errors:
        print("\n检查失败：")
        for item in errors:
            print(f"  - {item}")

        raise RuntimeError(
            "实验数据检查失败，"
            "请根据上述信息修正数据。"
        )

    print("\n数据检查通过。")
    print(f"检查摘要：{summary_path}")
    print("=" * 90)

if __name__ == "__main__":
    validate()