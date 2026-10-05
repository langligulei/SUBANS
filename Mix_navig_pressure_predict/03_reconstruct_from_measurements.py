# -*- coding: utf-8 -*-
"""
从STAR-CCM+导出的稀疏传感器压力重构三向流场表面压力场。

三向流场特性：
1. 速度矢量：(vx, vy, vz)
2. 动压计算：q = 0.5 * rho * (vx² + vy² + vz²)
3. 支持DEIM和QR两种传感器选择方法
"""

from pathlib import Path
import csv
import importlib
import importlib.util
import re
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
cfg = importlib.import_module("00_sparse_config")

# =============================================================================
# 1. STAR-CCM+测量文件配置
# =============================================================================

STAR_MEASUREMENT_ROOT = Path(
    r"E:\suboff（0-200）\suboff_0000\MeshMixVelo\measurements"
)

STAR_MEASUREMENT_FILES = {
    "deim": STAR_MEASUREMENT_ROOT / "DEIMSensorFromStar.csv",
    "qr": STAR_MEASUREMENT_ROOT / "QRSensorFromStar.csv",
}

# 部件名称映射
PART_TO_STAR_NAME = {
    "all_surface": "all",
    "hull": "hull",
    "sail": "sail",
    "fin1": "fin1",
    "fin2": "fin2",
    "fin3": "fin3",
    "fin4": "fin4",
}

STAR_COORDINATE_TOLERANCE_M = 1.0e-4

# STAR字段名称
STAR_PRESSURE_COLUMN = "Absolute Pressure (Pa)"
STAR_TOTAL_PRESSURE_COLUMN = "Absolute Total Pressure (Pa)"
STAR_X_COLUMN = "X (m)"
STAR_Y_COLUMN = "Y (m)"
STAR_Z_COLUMN = "Z (m)"
STAR_ID_COLUMN = "ID"


# =============================================================================
# 2. 基础工具函数
# =============================================================================

def decode_hdf5_attribute(value) -> str:
    """HDF5属性转字符串"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

def get_expected_sensor_count(
    method: str | None = None,
) -> int:
    """
    返回指定传感器方法应使用的传感器数量。
    """
    if method is None:
        method = get_sensor_method()

    method = str(method).strip().lower()

    if method == "qr":
        return int(
            getattr(
                cfg,
                "QR_SENSOR_COUNT",
                cfg.N_SENSORS,
            )
        )

    if method == "deim":
        return int(
            getattr(
                cfg,
                "DEIM_SENSOR_COUNT",
                cfg.EXPECTED_POD_MODES,
            )
        )

    raise ValueError(
        f"未知传感器方法：{method}"
    )

def get_sensor_method() -> str:
    """获取传感器方法"""
    method = str(cfg.SENSOR_METHOD).strip().lower()
    if method not in {"qr", "deim"}:
        raise ValueError(f"当前脚本只支持QR或DEIM，实际：{method}")
    return method


def get_part_name() -> str:
    """获取部件名称"""
    part_name = str(cfg.PART_NAME).strip().lower()
    if part_name not in PART_TO_STAR_NAME:
        raise ValueError(f"未知PART_NAME：{part_name}")
    return part_name


def calculate_freestream_velocity() -> tuple[float, float, float, float]:
    """
    计算三向来流速度。

    返回：
        (vx, vy, vz, |V|)
    """
    vx = float(cfg.TEST_U_INF)
    vy = float(cfg.TEST_V_INF)
    vz = float(cfg.TEST_W_INF)

    speed_magnitude = float(np.sqrt(vx ** 2 + vy ** 2 + vz ** 2))

    if not np.isfinite(speed_magnitude):
        raise ValueError("来流速度包含NaN或Inf")

    if speed_magnitude <= 0.0:
        raise ValueError("来流速度模长必须大于0")

    return vx, vy, vz, speed_magnitude


def calculate_dynamic_pressure(speed_magnitude: float) -> float:
    """计算动压"""
    rho = float(cfg.RHO)

    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError(f"流体密度RHO无效：{rho}")

    return float(0.5 * rho * speed_magnitude ** 2)


# =============================================================================
# 3. 读取稀疏ROM
# =============================================================================

def resolve_rom_path() -> Path:
    """获取ROM路径"""
    method = get_sensor_method()
    part_name = get_part_name()

    rom_path = (
            Path(cfg.ROM_ROOT)
            / f"{part_name}_{method}_sparse_rom.h5"
    )

    if not rom_path.is_file():
        raise FileNotFoundError(
            f"ROM文件不存在：{rom_path}\n"
            f"请先运行 02_build_sparse_rom.py"
        )

    return rom_path


def load_rom() -> dict:
    """加载稀疏ROM"""
    rom_path = resolve_rom_path()

    with h5py.File(rom_path, mode="r") as file:
        required_datasets = [
            "mean_field",
            "basis",
            "sensor_mean",
            "sensor_basis",
            "global_face_id",
            "sensor_global_face_id",
        ]

        missing = [name for name in required_datasets if name not in file]

        if missing:
            raise KeyError(f"ROM文件缺少数据集：{missing}")

        rom = {
            "mean_field": np.asarray(file["mean_field"][()], dtype=np.float64),
            "basis": np.asarray(file["basis"][()], dtype=np.float64),
            "sensor_mean": np.asarray(file["sensor_mean"][()], dtype=np.float64),
            "sensor_basis": np.asarray(file["sensor_basis"][()], dtype=np.float64),
            "global_face_id": np.asarray(file["global_face_id"][()], dtype=np.int64),
            "sensor_global_face_id": np.asarray(
                file["sensor_global_face_id"][()], dtype=np.int64
            ),
            "rom_path": rom_path,
        }

        rom["topology_hash"] = decode_hdf5_attribute(
            file.attrs.get("topology_hash", "")
        )
        rom["part_name"] = decode_hdf5_attribute(
            file.attrs.get("part_name", get_part_name())
        )
        rom["target_field"] = decode_hdf5_attribute(
            file.attrs.get("target_field", "cp_static")
        )

    # ROM形状检查
    if rom["mean_field"].ndim != 1:
        raise ValueError("ROM mean_field必须是一维数组")

    if rom["basis"].ndim != 2:
        raise ValueError("ROM basis必须是二维数组")

    n_faces, n_modes = rom["basis"].shape
    n_sensors = len(
        rom["sensor_global_face_id"]
    )

    if len(rom["mean_field"]) != n_faces:
        raise ValueError(
            "ROM mean_field长度与basis面数不一致"
        )

    expected_sensor_count = (
        get_expected_sensor_count()
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
            f"ROM传感器数量错误："
            f"实际={n_sensors}，"
            f"期望={expected_sensor_count}"
        )

    if n_modes != expected_pod_modes:
        raise ValueError(
            f"ROM模态数量错误："
            f"实际={n_modes}，"
            f"期望={expected_pod_modes}"
        )

    if n_sensors < n_modes:
        raise ValueError(
            f"传感器数量({n_sensors})小于模态数"
            f"({n_modes})"
        )

    if rom["sensor_basis"].shape != (n_sensors, n_modes):
        raise ValueError("ROM sensor_basis形状不一致")

    return rom


# =============================================================================
# 4. 读取STAR-CCM+传感器测量
# =============================================================================

def parse_star_float(
        row: list[str],
        column_index: dict[str, int],
        column_name: str,
        line_number: int,
) -> float:
    """从STAR CSV行读取浮点数"""
    index = column_index[column_name]

    if index >= len(row):
        raise ValueError(f"STAR CSV第{line_number}行缺少字段：{column_name}")

    text = str(row[index]).strip()

    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(
            f"STAR CSV第{line_number}行{column_name}不是有效数字：{text}"
        ) from exc

    if not np.isfinite(value):
        raise ValueError(f"STAR CSV第{line_number}行{column_name}包含NaN或Inf")

    return value


def load_all_star_measurements(path: Path) -> pd.DataFrame:
    """读取并展开STAR传感器文件"""
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"STAR测点文件不存在：{path}")

    parsed_rows = []

    with path.open(mode="r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"STAR测点文件为空：{path}") from exc

        header = [str(value).strip() for value in raw_header]

        required_columns = {
            STAR_PRESSURE_COLUMN,
            STAR_TOTAL_PRESSURE_COLUMN,
            STAR_X_COLUMN,
            STAR_Y_COLUMN,
            STAR_Z_COLUMN,
            STAR_ID_COLUMN,
        }

        missing_columns = required_columns - set(header)

        if missing_columns:
            raise KeyError(f"STAR测点文件缺少字段：{sorted(missing_columns)}")

        column_index = {name: header.index(name) for name in required_columns}
        id_index = column_index[STAR_ID_COLUMN]

        for line_number, row in enumerate(reader, start=2):
            if not row or not any(str(value).strip() for value in row):
                continue

            absolute_pressure = parse_star_float(
                row, column_index, STAR_PRESSURE_COLUMN, line_number
            )

            absolute_total_pressure = parse_star_float(
                row, column_index, STAR_TOTAL_PRESSURE_COLUMN, line_number
            )

            x_m = parse_star_float(row, column_index, STAR_X_COLUMN, line_number)
            y_m = parse_star_float(row, column_index, STAR_Y_COLUMN, line_number)
            z_m = parse_star_float(row, column_index, STAR_Z_COLUMN, line_number)

            # 处理ID字段（可能被CSV拆分）
            id_fragments = [
                str(value).strip() for value in row[id_index:] if str(value).strip()
            ]
            combined_id = ",".join(id_fragments)

            sensor_names = [
                value.strip() for value in combined_id.split(",") if value.strip()
            ]

            if not sensor_names:
                raise ValueError(f"STAR CSV第{line_number}行没有有效传感器ID")

            for sensor_name in sensor_names:
                match = re.fullmatch(
                    r"(DEIM|QR)_(All|Hull|Sail|Fin[1-4])_(\d+)",
                    sensor_name,
                    flags=re.IGNORECASE,
                )

                if match is None:
                    raise ValueError(
                        f"STAR CSV第{line_number}行传感器ID无法识别：{sensor_name}"
                    )

                parsed_rows.append({
                    "sensor_method": match.group(1).lower(),
                    "sensor_part": match.group(2).lower(),
                    "sensor_rank": int(match.group(3)),
                    "sensor_name": sensor_name,
                    "absolute_pressure_pa": absolute_pressure,
                    "absolute_total_pressure_pa": absolute_total_pressure,
                    "x_m": x_m,
                    "y_m": y_m,
                    "z_m": z_m,
                    "star_csv_line": line_number,
                    "source_combined_id": combined_id,
                })

    if not parsed_rows:
        raise ValueError(f"STAR测点文件中没有有效记录：{path}")

    return pd.DataFrame(parsed_rows)


def load_topology_centroid() -> np.ndarray:
    """读取表面拓扑面心坐标"""
    topology_path = Path(cfg.TOPOLOGY_PATH).expanduser().resolve()

    if not topology_path.is_file():
        raise FileNotFoundError(f"表面拓扑不存在：{topology_path}")

    with h5py.File(topology_path, mode="r") as file:
        if "geometry/centroid" not in file:
            raise KeyError("表面拓扑中不存在/geometry/centroid")

        centroid = np.asarray(file["geometry/centroid"][()], dtype=np.float64)

    if centroid.ndim != 2 or centroid.shape[1] != 3:
        raise ValueError("表面拓扑centroid形状必须为(n_faces,3)")

    return centroid


def load_star_measurements(rom: dict) -> pd.DataFrame:
    """提取当前传感器方法和部件对应的STAR测量值"""
    method = get_sensor_method()
    part_name = get_part_name()
    star_part_name = PART_TO_STAR_NAME[part_name]

    measurement_path = STAR_MEASUREMENT_FILES[method]

    table = load_all_star_measurements(measurement_path)

    selected = table.loc[
        (table["sensor_method"] == method) & (table["sensor_part"] == star_part_name)
        ].copy()

    selected = selected.sort_values("sensor_rank", kind="stable").reset_index(drop=True)

    expected_count = len(rom["sensor_global_face_id"])
    if expected_count != get_expected_sensor_count(
            method
    ):
        raise ValueError(
            f"ROM中的传感器数量与配置不一致："
            f"ROM={expected_count}，"
            f"配置={cfg.N_SENSORS}"
        )
    expected_ranks = np.arange(1, expected_count + 1, dtype=np.int64)
    actual_ranks = selected["sensor_rank"].to_numpy(dtype=np.int64)

    if selected["sensor_rank"].duplicated().any():
        duplicate_ranks = selected.loc[
            selected["sensor_rank"].duplicated(keep=False), "sensor_rank"
        ].tolist()
        raise ValueError(f"{method}/{part_name}存在重复传感器排名：{duplicate_ranks}")

    if not np.array_equal(actual_ranks, expected_ranks):
        raise ValueError(
            f"{method}/{part_name}传感器排名不完整\n"
            f"实际：{actual_ranks.tolist()}\n"
            f"期望：{expected_ranks.tolist()}"
        )

    # 验证坐标
    centroid = load_topology_centroid()
    sensor_global_ids = np.asarray(rom["sensor_global_face_id"], dtype=np.int64)

    if np.any(sensor_global_ids < 0) or np.any(sensor_global_ids >= len(centroid)):
        raise ValueError("ROM中的sensor_global_face_id越界")

    expected_xyz = centroid[sensor_global_ids]
    star_xyz = selected[["x_m", "y_m", "z_m"]].to_numpy(dtype=np.float64)

    coordinate_error = np.linalg.norm(star_xyz - expected_xyz, axis=1)

    invalid = coordinate_error > STAR_COORDINATE_TOLERANCE_M

    if np.any(invalid):
        details = [
            f"rank={int(actual_ranks[i])}, "
            f"ID={selected.iloc[i]['sensor_name']}, "
            f"distance={coordinate_error[i]:.6e} m"
            for i in np.flatnonzero(invalid)
        ]
        raise ValueError(
            "STAR测点与ROM传感器坐标不一致\n" + "\n".join(details)
        )

    selected["global_face_id"] = sensor_global_ids
    selected["expected_x_m"] = expected_xyz[:, 0]
    selected["expected_y_m"] = expected_xyz[:, 1]
    selected["expected_z_m"] = expected_xyz[:, 2]
    selected["coordinate_error_m"] = coordinate_error

    return selected

def resolve_test_case_h5_path() -> Path:
    """
    根据TEST_CASE_ID解析测试工况HDF5路径。

    优先使用cfg.TEST_CASE_H5_PATH；
    如果没有设置，则从cases.csv查找case_id。
    """
    configured_path = getattr(
        cfg,
        "TEST_CASE_H5_PATH",
        None,
    )

    if configured_path is not None:
        path = Path(
            configured_path
        ).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"TEST_CASE_H5_PATH不存在：{path}"
            )

        return path

    cases_path = (
        Path(cfg.CASES_CSV_PATH)
        .expanduser()
        .resolve()
    )

    if not cases_path.is_file():
        raise FileNotFoundError(
            f"cases.csv不存在：{cases_path}"
        )

    cases = pd.read_csv(
        cases_path,
        encoding="utf-8-sig",
    )

    required_columns = {
        "case_id",
        "case_h5_path",
    }

    missing = required_columns - set(cases.columns)

    if missing:
        raise KeyError(
            f"cases.csv缺少字段：{sorted(missing)}"
        )

    selected = cases.loc[
        cases["case_id"].astype(str)
        == str(cfg.TEST_CASE_ID)
    ]

    if len(selected) != 1:
        raise ValueError(
            f"cases.csv中无法唯一找到测试工况："
            f"{cfg.TEST_CASE_ID}，"
            f"匹配数量={len(selected)}"
        )

    case_path = Path(
        str(selected.iloc[0]["case_h5_path"])
    )

    if not case_path.is_absolute():
        case_path = (
            cases_path.parent / case_path
        )

    case_path = case_path.expanduser().resolve()

    if not case_path.is_file():
        raise FileNotFoundError(
            f"测试工况HDF5不存在：{case_path}"
        )

    return case_path

def load_case_h5_measurements(
    rom: dict,
) -> pd.DataFrame:
    """
    从测试工况HDF5直接读取完整Cp场，
    再按照ROM中的sensor_global_face_id抽取传感器Cp。

    要求HDF5中存在：
        fields/cp_static
    """
    case_h5_path = (
        resolve_test_case_h5_path()
    )

    print(
        f"\n从工况HDF5读取测试压力场："
        f"{case_h5_path}"
    )

    with h5py.File(
        case_h5_path,
        mode="r",
    ) as file:
        field_path = None
        field_mode = None

        if "fields/cp_static" in file:
            field_path = "fields/cp_static"
            field_mode = "cp_static"

        elif "fields/p_static_absolute" in file:
            field_path = "fields/p_static_absolute"
            field_mode = "absolute_pressure"

        elif "fields/pressure_absolute" in file:
            field_path = "fields/pressure_absolute"
            field_mode = "absolute_pressure"

        else:
            available = []

            if "fields" in file:
                available = list(
                    file["fields"].keys()
                )

            raise KeyError(
                f"{case_h5_path}中没有可用压力场；"
                f"可用字段：{available}"
            )

        values = np.asarray(
            file[field_path][()],
            dtype=np.float64,
        ).ravel()


        cp_full = np.asarray(
            file[field_path][()],
            dtype=np.float64,
        ).ravel()

        file_face_count = int(
            file.attrs.get(
                "face_count",
                len(cp_full),
            )
        )

        file_topology_hash = file.attrs.get(
            "topology_hash_sha256",
            file.attrs.get(
                "topology_hash",
                "",
            ),
        )

        if isinstance(
            file_topology_hash,
            bytes,
        ):
            file_topology_hash = (
                file_topology_hash.decode(
                    "utf-8",
                    errors="replace",
                )
            )

    if file_face_count != len(
        rom["global_face_id"]
    ):
        raise ValueError(
            "测试工况面数与ROM不一致："
            f"case={file_face_count}, "
            f"rom={len(rom['global_face_id'])}"
        )

    if len(cp_full) != file_face_count:
        raise ValueError(
            "测试工况cp_static长度与face_count不一致："
            f"{len(cp_full)} != {file_face_count}"
        )

    if not np.all(np.isfinite(cp_full)):
        raise ValueError(
            "测试工况cp_static包含NaN或Inf"
        )

    rom_topology_hash = str(
        rom.get("topology_hash", "")
    )

    if (
        rom_topology_hash
        and str(file_topology_hash)
        and rom_topology_hash
        != str(file_topology_hash)
    ):
        raise ValueError(
            "测试工况拓扑哈希与ROM不一致："
            f"case={file_topology_hash}, "
            f"rom={rom_topology_hash}"
        )

    sensor_global_ids = np.asarray(
        rom["sensor_global_face_id"],
        dtype=np.int64,
    )

    if np.any(sensor_global_ids < 0):
        raise ValueError(
            "ROM传感器global_face_id包含负数"
        )

    if np.any(
        sensor_global_ids >= len(cp_full)
    ):
        raise ValueError(
            "ROM传感器global_face_id超出测试工况面数"
        )

    sensor_cp = cp_full[sensor_global_ids]

    expected_count = len(
        sensor_global_ids
    )

    if len(sensor_cp) != expected_count:
        raise ValueError(
            "抽取出的传感器数量不正确"
        )

    return pd.DataFrame({
        "sensor_id": np.arange(
            1,
            expected_count + 1,
            dtype=np.int64,
        ),
        "global_face_id": sensor_global_ids,
        "cp_static": sensor_cp,
        "cp_observed": sensor_cp,
        "source": "case_h5",
        "source_path": str(case_h5_path),
        "coordinate_error_m": 0.0,
    })

def load_module_from_path(
    path: Path,
    module_name: str,
):
    """
    从指定Python文件路径加载模块。

    测试SBD读取必须复用数据集构建程序，
    防止在稀疏重构脚本中另写一份SBD解析或面映射逻辑。
    """
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"数据集构建程序不存在：{path}"
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
            f"无法加载数据集构建程序：{path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    sys.modules[module_name] = module

    specification.loader.exec_module(module)

    return module

def load_sbd_measurements(
    rom: dict,
) -> pd.DataFrame:
    """
    从未参与训练的测试SBD读取完整静压场，并按ROM传感器
    global_face_id抽取Cp观测值。

    该函数复用Mix_build_modal_surface_dataset.py中的：
      1. read_sbd_case
      2. build_sbd_to_cgns_mapping
      3. reorder_to_cgns

    因此测试SBD与POD/ROM严格保持相同的global_face_id顺序。
    """
    sbd_path = Path(
        cfg.TEST_SBD_PATH
    ).expanduser().resolve()

    if not sbd_path.is_file():
        raise FileNotFoundError(
            f"测试SBD不存在：{sbd_path}"
        )

    builder_path = Path(
        cfg.DATASET_BUILDER_PATH
    ).expanduser().resolve()

    print(
        "\n从测试SBD读取稀疏传感器压力："
        f"{sbd_path}"
    )

    builder = load_module_from_path(
        path=builder_path,
        module_name=(
            "mix_modal_dataset_builder_"
            "for_sparse_reconstruction"
        ),
    )

    legacy = builder.load_legacy_module()

    with h5py.File(
        Path(cfg.TOPOLOGY_PATH)
        .expanduser()
        .resolve(),
        mode="r",
    ) as topology_file:
        centroid = np.asarray(
            topology_file[
                "geometry/centroid"
            ][()],
            dtype=np.float64,
        )

        face_area = np.asarray(
            topology_file[
                "geometry/face_area"
            ][()],
            dtype=np.float64,
        )

        topology_hash = decode_hdf5_attribute(
            topology_file.attrs.get(
                "topology_hash_sha256",
                "",
            )
        )

    if len(centroid) != len(
        rom["global_face_id"]
    ):
        raise ValueError(
            "拓扑面数与ROM面数不一致："
            f"topology={len(centroid)}，"
            f"rom={len(rom['global_face_id'])}"
        )

    rom_topology_hash = str(
        rom.get("topology_hash", "")
    )

    if (
        rom_topology_hash
        and topology_hash
        and rom_topology_hash != topology_hash
    ):
        raise ValueError(
            "ROM与topology.h5的拓扑哈希不一致："
            f"rom={rom_topology_hash}，"
            f"topology={topology_hash}"
        )

    sbd_data = builder.read_sbd_case(
        legacy=legacy,
        sbd_path=sbd_path,
        p_static_ref_pa=float(
            cfg.P_STATIC_REF_PA
        ),
    )

    sbd_to_cgns, mapping_metadata = (
        builder.build_sbd_to_cgns_mapping(
            sbd_data=sbd_data,
            cgns_centroids=centroid,
            cgns_areas=face_area,
        )
    )

    p_static_absolute_full = (
        builder.reorder_to_cgns(
            sbd_data.p_static_absolute,
            sbd_to_cgns,
        )
    )

    if len(p_static_absolute_full) != len(
        rom["global_face_id"]
    ):
        raise ValueError(
            "SBD重排后的压力场长度与ROM面数不一致："
            f"sbd={len(p_static_absolute_full)}，"
            f"rom={len(rom['global_face_id'])}"
        )

    if not np.all(
        np.isfinite(p_static_absolute_full)
    ):
        raise ValueError(
            "测试SBD重排后的绝对静压包含NaN或Inf"
        )

    sensor_global_ids = np.asarray(
        rom["sensor_global_face_id"],
        dtype=np.int64,
    )

    if np.any(sensor_global_ids < 0) or np.any(
        sensor_global_ids >= len(
            p_static_absolute_full
        )
    ):
        raise ValueError(
            "ROM中的sensor_global_face_id越界"
        )

    sensor_pressure_absolute = (
        p_static_absolute_full[
            sensor_global_ids
        ]
    )

    sensor_cp = convert_pressure_to_cp(
        sensor_pressure_absolute
    )

    expected_count = get_expected_sensor_count(
        get_sensor_method()
    )

    if len(sensor_cp) != expected_count:
        raise ValueError(
            "SBD抽取的传感器数量与配置不一致："
            f"实际={len(sensor_cp)}，"
            f"期望={expected_count}"
        )

    print(
        "  SBD -> CGNS映射方法："
        f"{mapping_metadata['method']}"
    )

    print(
        "  抽取传感器数量："
        f"{len(sensor_cp)}"
    )

    return pd.DataFrame({
        "sensor_id": np.arange(
            1,
            len(sensor_global_ids) + 1,
            dtype=np.int64,
        ),
        "global_face_id": sensor_global_ids,
        "absolute_pressure_pa": (
            sensor_pressure_absolute
        ),
        "cp_static": sensor_cp,
        "source": "test_sbd",
        "source_path": str(sbd_path),
        "mapping_method": (
            mapping_metadata["method"]
        ),
        "coordinate_error_m": 0.0,
    })

# =============================================================================
# 5. 压力转换
# =============================================================================

def convert_pressure_to_cp(pressure_values: np.ndarray) -> np.ndarray:
    """将输入压力转换为静压力系数"""
    pressure_values = np.asarray(pressure_values, dtype=np.float64)

    if np.any(~np.isfinite(pressure_values)):
        raise ValueError("传感器压力包含NaN或Inf")

    _, _, _, speed_magnitude = calculate_freestream_velocity()
    q_inf = calculate_dynamic_pressure(speed_magnitude)

    pressure_input_type = str(cfg.PRESSURE_INPUT_TYPE).strip().lower()

    if pressure_input_type == "cp_static":
        return pressure_values

    if pressure_input_type == "absolute_pressure":
        return (pressure_values - float(cfg.P_STATIC_REF_PA)) / q_inf

    if pressure_input_type == "static_pressure_relative":
        return pressure_values / q_inf

    raise ValueError(f"未知PRESSURE_INPUT_TYPE：{cfg.PRESSURE_INPUT_TYPE}")


# =============================================================================
# 6. POD系数求解
# =============================================================================

def solve_coefficients(
        sensor_cp: np.ndarray,
        sensor_mean: np.ndarray,
        sensor_basis: np.ndarray,
) -> np.ndarray:
    """根据稀疏传感器Cp求解POD系数"""
    sensor_cp = np.asarray(sensor_cp, dtype=np.float64)
    sensor_mean = np.asarray(sensor_mean, dtype=np.float64)
    sensor_basis = np.asarray(sensor_basis, dtype=np.float64)

    if len(sensor_cp) != len(sensor_mean):
        raise ValueError("传感器Cp数量与ROM传感器平均值数量不一致")

    if sensor_basis.shape[0] != len(sensor_cp):
        raise ValueError("sensor_basis行数与传感器数量不一致")

    observed_fluctuation = (
            sensor_cp - sensor_mean
    )

    if not np.all(
            np.isfinite(observed_fluctuation)
    ):
        raise ValueError(
            "传感器压力扰动包含NaN或Inf"
        )

    singular_values = np.linalg.svd(
        sensor_basis,
        compute_uv=False,
    )

    if len(singular_values) == 0:
        raise ValueError(
            "传感矩阵没有奇异值"
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

    condition_number = (
        sigma_max / sigma_min
        if sigma_min > 0.0
        else float("inf")
    )

    print(
        f"  传感矩阵形状：{sensor_basis.shape}"
    )
    print(
        f"  传感矩阵数值秩："
        f"{numerical_rank}/{sensor_basis.shape[1]}"
    )
    print(
        f"  传感矩阵条件数："
        f"{condition_number:.6e}"
    )

    if numerical_rank < sensor_basis.shape[1]:
        raise np.linalg.LinAlgError(
            "传感矩阵不满列秩"
        )

    solver = str( cfg.COEFFICIENT_SOLVER ).strip().lower()

    if solver == "lstsq":
        coefficients, _, rank, singular_values = np.linalg.lstsq(
            sensor_basis, observed_fluctuation, rcond=None
        )

        print(f"  传感矩阵数值秩：{rank}/{sensor_basis.shape[1]}")

        if len(singular_values) > 0:
            print(f"  传感矩阵最小奇异值：{singular_values[-1]:.6e}")

        return coefficients

    if solver == "ridge":
        ridge_lambda = float(cfg.RIDGE_LAMBDA)

        if ridge_lambda < 0.0:
            raise ValueError(
                "RIDGE_LAMBDA不能小于0"
            )

        n_modes = sensor_basis.shape[1]

        if ridge_lambda == 0.0:
            coefficients, _, _, _ = np.linalg.lstsq(
                sensor_basis,
                observed_fluctuation,
                rcond=None,
            )
            return coefficients

        # 使用增广最小二乘求解：
        #
        # min ||B a - y||_2^2 + lambda ||a||_2^2
        #
        # 等价于：
        #
        # min || [B              ] a - [y] ||_2^2
        #     || [sqrt(lambda) I]     [0] ||
        augmented_matrix = np.vstack([
            sensor_basis,
            np.sqrt(ridge_lambda)
            * np.eye(n_modes, dtype=np.float64),
        ])

        augmented_rhs = np.concatenate([
            observed_fluctuation,
            np.zeros(n_modes, dtype=np.float64),
        ])

        coefficients, _, rank, singular_values = np.linalg.lstsq(
            augmented_matrix,
            augmented_rhs,
            rcond=None,
        )

        print(
            f"岭回归有效秩：{rank}/{n_modes}"
        )

        if len(singular_values) > 0:
            print(
                "岭回归增广矩阵最小奇异值："
                f"{singular_values[-1]:.6e}"
            )

        return coefficients

    raise ValueError(f"未知COEFFICIENT_SOLVER：{cfg.COEFFICIENT_SOLVER}")


# =============================================================================
# 7. 表面压力重构主流程
# =============================================================================

def reconstruct() -> None:
    """主重构流程"""
    method = get_sensor_method()
    part_name = get_part_name()

    prediction_root = Path(cfg.PREDICTION_ROOT).expanduser().resolve()
    prediction_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("三向流场STAR-CCM+稀疏压力测量表面重构")
    print("=" * 90)
    print(f"传感器方法：{method.upper()}")
    print(f"目标部件：{part_name}")
    print(f"测试工况：{cfg.TEST_CASE_ID}")

    vx, vy, vz, speed_magnitude = calculate_freestream_velocity()
    print(f"速度分量：vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f} m/s")
    print(f"速度模长：|V|={speed_magnitude:.4f} m/s")

    rom = load_rom()

    print(f"\nROM信息：")
    print(f"  ROM文件：{rom['rom_path'].name}")
    print(f"  目标场：{rom['target_field']}")
    print(f"  表面面数：{len(rom['global_face_id']):,}")
    print(f"  POD模态数：{rom['basis'].shape[1]}")
    print(f"  传感器数：{len(rom['sensor_global_face_id'])}")

    measurement_source = str(
        getattr(
            cfg,
            "MEASUREMENT_SOURCE",
            "star_csv",
        )
    ).strip().lower()

    if measurement_source == "case_h5":
        measurements = (
            load_case_h5_measurements(rom)
        )

        sensor_cp = measurements[
            "cp_static"
        ].to_numpy(
            dtype=np.float64
        )

        sensor_pressure_absolute = None

    elif measurement_source == "sbd":
        measurements = load_sbd_measurements(
            rom
        )

        sensor_cp = measurements[
            "cp_static"
        ].to_numpy(
            dtype=np.float64
        )

        sensor_pressure_absolute = (
            measurements[
                "absolute_pressure_pa"
            ].to_numpy(
                dtype=np.float64
            )
        )

    elif measurement_source == "star_csv":
        measurements = load_star_measurements(rom)

        sensor_pressure_absolute = (
            measurements[
                "absolute_pressure_pa"
            ].to_numpy(dtype=np.float64)
        )

        sensor_cp = convert_pressure_to_cp(
            sensor_pressure_absolute
        )

    else:
        raise ValueError(
            "未知MEASUREMENT_SOURCE："
            f"{measurement_source}；"
            "可选：case_h5、sbd、star_csv"
        )

    coefficients = solve_coefficients(
        sensor_cp=sensor_cp,
        sensor_mean=rom["sensor_mean"],
        sensor_basis=rom["sensor_basis"],
    )
    coefficient_norm = float(
        np.linalg.norm(coefficients)
    )

    if not np.isfinite(coefficient_norm):
        raise ValueError(
            "POD系数范数包含NaN或Inf"
        )

    print(
        f"POD系数L2范数：{coefficient_norm:.6e}"
    )

    reconstructed_cp = rom["mean_field"] + rom["basis"] @ coefficients

    fitted_sensor_cp = rom["sensor_mean"] + rom["sensor_basis"] @ coefficients

    sensor_residual_cp = fitted_sensor_cp - sensor_cp

    q_inf = calculate_dynamic_pressure(speed_magnitude)
    p_static_ref = float(cfg.P_STATIC_REF_PA)

    reconstructed_pressure = p_static_ref + q_inf * reconstructed_cp

    fitted_sensor_pressure = (
            p_static_ref + q_inf * fitted_sensor_cp
    )

    if sensor_pressure_absolute is None:
        observed_sensor_pressure = (
                p_static_ref + q_inf * sensor_cp
        )
    else:
        observed_sensor_pressure = (
            sensor_pressure_absolute
        )

    sensor_residual_pressure = (
            fitted_sensor_pressure
            - observed_sensor_pressure
    )
    result_prefix = f"{cfg.TEST_CASE_ID}_{part_name}_{method}"

    # 保存完整表面重构
    output = pd.DataFrame({
        "global_face_id": rom["global_face_id"],
        "cp_static_reconstructed": reconstructed_cp,
        "p_static_absolute_reconstructed_pa": reconstructed_pressure,
    })

    output_path = prediction_root / f"{result_prefix}_prediction.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 保存POD系数
    coefficient_output = prediction_root / f"{result_prefix}_coefficients.csv"
    pd.DataFrame({
        "mode_index": np.arange(len(coefficients), dtype=np.int64) + 1,
        "coefficient": coefficients,
    }).to_csv(coefficient_output, index=False, encoding="utf-8-sig")

    # 保存传感器匹配和拟合结果
    sensor_output = measurements.copy()
    sensor_output["cp_observed"] = sensor_cp
    sensor_output["cp_fitted"] = fitted_sensor_cp
    sensor_output["cp_residual"] = sensor_residual_cp
    sensor_output["absolute_pressure_observed_pa"] = observed_sensor_pressure
    sensor_output["absolute_pressure_fitted_pa"] = fitted_sensor_pressure
    sensor_output["pressure_residual_pa"] = sensor_residual_pressure

    sensor_output_path = prediction_root / f"{result_prefix}_sensor_fit.csv"
    sensor_output.to_csv(sensor_output_path, index=False, encoding="utf-8-sig")

    # 保存计算摘要
    condition_number = float(np.linalg.cond(rom["sensor_basis"]))
    cp_rmse = float(np.sqrt(np.mean(sensor_residual_cp ** 2)))
    pressure_rmse = float(np.sqrt(np.mean(sensor_residual_pressure ** 2)))

    summary_output = prediction_root / f"{result_prefix}_summary.csv"
    pd.DataFrame([{
        "test_case_id": str(cfg.TEST_CASE_ID),
        "part_name": part_name,
        "sensor_method": method,
        "test_u_inf_m_s": vx,
        "test_v_inf_m_s": vy,
        "test_w_inf_m_s": vz,
        "speed_magnitude_m_s": speed_magnitude,
        "rho_kg_m3": float(cfg.RHO),
        "dynamic_pressure_pa": q_inf,
        "p_static_ref_pa": p_static_ref,
        "sensor_count": len(measurements),
        "pod_mode_count": len(coefficients),
        "sensor_matrix_condition_number": condition_number,
        "sensor_cp_rmse": cp_rmse,
        "pod_coefficient_l2_norm": coefficient_norm,
        "sensor_pressure_rmse_pa": pressure_rmse,
        "maximum_coordinate_error_m": float(measurements["coordinate_error_m"].max()),
        "rom_path": str(rom["rom_path"]),
    }]).to_csv(summary_output, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("重构完成")
    print("=" * 90)
    print(f"速度矢量：({vx:.3f}, {vy:.3f}, {vz:.3f}) m/s")
    print(f"速度模长：{speed_magnitude:.4f} m/s")
    print(f"动压：{q_inf:.2f} Pa")
    print(f"传感矩阵条件数：{condition_number:.6e}")
    print(f"传感器Cp拟合RMSE：{cp_rmse:.6e}")
    print(f"传感器压力拟合RMSE：{pressure_rmse:.2f} Pa")
    print(f"最大坐标匹配误差：{measurements['coordinate_error_m'].max():.6e} m")
    print(f"\n输出文件：")
    print(f"  完整压力场：{output_path.name}")
    print(f"  POD系数：{coefficient_output.name}")
    print(f"  传感器拟合：{sensor_output_path.name}")
    print(f"  计算摘要：{summary_output.name}")
    print("=" * 90)


# =============================================================================
# 8. 程序入口
# =============================================================================

if __name__ == "__main__":
    original_method = str(cfg.SENSOR_METHOD)

    successful_methods = []
    failed_methods = []

    try:
        for sensor_method in ["deim", "qr"]:
            print("\n" + "#" * 90)
            print(f"开始执行{sensor_method.upper()}传感器压力重构")
            print("#" * 90)

            cfg.SENSOR_METHOD = sensor_method

            try:
                reconstruct()
                successful_methods.append(sensor_method)
            except Exception as exc:
                failed_methods.append({
                    "sensor_method": sensor_method,
                    "error": repr(exc),
                })
                print(f"\n{'!' * 90}")
                print(f"{sensor_method.upper()}重构失败")
                print(f"错误：{exc}")
                print("!" * 90)

    finally:
        cfg.SENSOR_METHOD = original_method

    print("\n" + "=" * 90)
    print("QR和DEIM重构执行摘要")
    print("=" * 90)
    print(
        "成功方法："
        + (
            ", ".join(method.upper() for method in successful_methods)
            if successful_methods
            else "无"
        )
    )

    if failed_methods:
        print("失败方法：")
        for item in failed_methods:
            print(f"  - {item['sensor_method'].upper()}: {item['error']}")
        raise RuntimeError("存在传感器方法重构失败")
    else:
        print("失败方法：无")

    print("=" * 90)