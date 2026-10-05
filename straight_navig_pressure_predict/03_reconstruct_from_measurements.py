# -*- coding: utf-8 -*-
"""
根据STAR-CCM+导出的稀疏传感器压力重构SUBOFF表面压力场。

支持的STAR-CCM+测量文件：
    DEIMSensorFromStar.csv
    QRSensorFromStar.csv

支持的传感器ID：
    DEIM_All_1
    DEIM_Hull_1
    DEIM_Sail_1
    DEIM_Fin1_1
    ...
    QR_All_1
    QR_Hull_1
    ...

同一物理位置对应多个传感器时，ID可以写成：
    "DEIM_All_3,DEIM_Sail_3"

也兼容没有正确加双引号、被CSV拆分为额外列的复合ID。
"""

from pathlib import Path
import csv
import importlib
import re
import sys

import h5py
import numpy as np
import pandas as pd

# =============================================================================
# 1. 导入配置
# =============================================================================

sys.path.insert(
    0,
    str(Path(__file__).parent),
)

cfg = importlib.import_module(
    "00_sparse_config"
)

# =============================================================================
# 2. STAR-CCM+测量文件配置
# =============================================================================

STAR_MEASUREMENT_ROOT = Path(
    r"E:\suboff（0-200）\suboff_0000\SurfaceCheck"
)

STAR_MEASUREMENT_FILES = {
    "deim": (
        STAR_MEASUREMENT_ROOT
        / "DEIMSensorFromStar.csv"
    ),
    "qr": (
        STAR_MEASUREMENT_ROOT
        / "QRSensorFromStar.csv"
    ),
}

# 程序内部部件名称和STAR传感器ID中的名称对应关系。
PART_TO_STAR_NAME = {
    "all_surface": "all",
    "hull": "hull",
    "sail": "sail",
    "fin1": "fin1",
    "fin2": "fin2",
    "fin3": "fin3",
    "fin4": "fin4",
}

# STAR导出的坐标只有有限位小数，因此允许一个很小的坐标误差。
STAR_COORDINATE_TOLERANCE_M = 1.0e-4

# STAR文件中的字段名称。
STAR_PRESSURE_COLUMN = (
    "Absolute Pressure (Pa)"
)

STAR_TOTAL_PRESSURE_COLUMN = (
    "Absolute Total Pressure (Pa)"
)

STAR_X_COLUMN = "X (m)"
STAR_Y_COLUMN = "Y (m)"
STAR_Z_COLUMN = "Z (m)"
STAR_ID_COLUMN = "ID"

# =============================================================================
# 3. 基础工具
# =============================================================================

def decode_hdf5_attribute(value) -> str:
    """
    将HDF5字符串属性转换为普通Python字符串。
    """
    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(value)

def get_sensor_method() -> str:
    """
    读取并检查当前传感器选择方法。
    """
    method = str(
        cfg.SENSOR_METHOD
    ).strip().lower()

    if method not in {
        "qr",
        "deim",
    }:
        raise ValueError(
            "当前脚本只支持QR或DEIM传感器，"
            f"实际SENSOR_METHOD={method}"
        )

    return method

def get_part_name() -> str:
    """
    读取并检查当前分析部件。
    """
    part_name = str(
        cfg.PART_NAME
    ).strip().lower()

    if part_name not in PART_TO_STAR_NAME:
        raise ValueError(
            "未知PART_NAME："
            f"{part_name}\n"
            "允许值："
            f"{sorted(PART_TO_STAR_NAME)}"
        )

    return part_name

def calculate_freestream_speed() -> float:
    """
    根据配置文件计算来流速度模长。
    """
    speed_inf = float(
        np.sqrt(
            float(cfg.TEST_U_INF) ** 2
            + float(cfg.TEST_V_INF) ** 2
            + float(cfg.TEST_W_INF) ** 2
        )
    )

    if not np.isfinite(speed_inf):
        raise ValueError(
            "配置中的测试来流速度包含NaN或Inf"
        )

    if speed_inf <= 0.0:
        raise ValueError(
            "测试工况来流速度模长必须大于0"
        )

    return speed_inf

def calculate_dynamic_pressure(
    speed_inf: float,
) -> float:
    """
    计算来流动压。
    """
    rho = float(cfg.RHO)

    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError(
            f"流体密度RHO无效：{rho}"
        )

    return float(
        0.5
        * rho
        * speed_inf ** 2
    )

# =============================================================================
# 4. 读取稀疏ROM
# =============================================================================

def resolve_rom_path() -> Path:
    """
    获取当前部件和传感器方法对应的独立ROM。
    """
    method = get_sensor_method()
    part_name = get_part_name()

    rom_path = (
        Path(cfg.ROM_ROOT)
        / (
            f"{part_name}_"
            f"{method}_sparse_rom.h5"
        )
    )

    if not rom_path.is_file():
        raise FileNotFoundError(
            "找不到当前传感器方法对应的ROM文件：\n"
            f"{rom_path}\n"
            "请先设置对应的PART_NAME和"
            "SENSOR_METHOD运行"
            "02_build_sparse_rom.py。"
        )

    return rom_path

def load_rom() -> dict:
    """
    读取表面稀疏POD模型。
    """
    rom_path = resolve_rom_path()

    with h5py.File(
        rom_path,
        mode="r",
    ) as file:
        required_datasets = [
            "mean_field",
            "basis",
            "sensor_mean",
            "sensor_basis",
            "global_face_id",
            "sensor_global_face_id",
        ]

        missing = [
            name
            for name in required_datasets
            if name not in file
        ]

        if missing:
            raise KeyError(
                f"ROM文件缺少数据集：{missing}\n"
                f"ROM文件：{rom_path}"
            )

        rom = {
            "mean_field": np.asarray(
                file["mean_field"][()],
                dtype=np.float64,
            ),
            "basis": np.asarray(
                file["basis"][()],
                dtype=np.float64,
            ),
            "sensor_mean": np.asarray(
                file["sensor_mean"][()],
                dtype=np.float64,
            ),
            "sensor_basis": np.asarray(
                file["sensor_basis"][()],
                dtype=np.float64,
            ),
            "global_face_id": np.asarray(
                file["global_face_id"][()],
                dtype=np.int64,
            ),
            "sensor_global_face_id": np.asarray(
                file[
                    "sensor_global_face_id"
                ][()],
                dtype=np.int64,
            ),
            "rom_path": rom_path,
        }

        rom["topology_hash"] = (
            decode_hdf5_attribute(
                file.attrs.get(
                    "topology_hash",
                    "",
                )
            )
        )

        rom["part_name"] = (
            decode_hdf5_attribute(
                file.attrs.get(
                    "part_name",
                    get_part_name(),
                )
            )
        )

        rom["target_field"] = (
            decode_hdf5_attribute(
                file.attrs.get(
                    "target_field",
                    "cp_static",
                )
            )
        )

    # ---------------------------------------------------------
    # ROM形状检查
    # ---------------------------------------------------------
    if rom["mean_field"].ndim != 1:
        raise ValueError(
            "ROM mean_field必须是一维数组"
        )

    if rom["basis"].ndim != 2:
        raise ValueError(
            "ROM basis必须是二维数组"
        )

    if rom["sensor_basis"].ndim != 2:
        raise ValueError(
            "ROM sensor_basis必须是二维数组"
        )

    n_faces, n_modes = rom["basis"].shape
    n_sensors = len(
        rom["sensor_global_face_id"]
    )

    if len(rom["mean_field"]) != n_faces:
        raise ValueError(
            "ROM mean_field长度与basis面数不一致"
        )

    if len(rom["global_face_id"]) != n_faces:
        raise ValueError(
            "ROM global_face_id长度与basis面数不一致"
        )

    if len(rom["sensor_mean"]) != n_sensors:
        raise ValueError(
            "ROM sensor_mean长度与传感器数量不一致"
        )

    if rom["sensor_basis"].shape != (
        n_sensors,
        n_modes,
    ):
        raise ValueError(
            "ROM sensor_basis形状不一致："
            f"{rom['sensor_basis'].shape} != "
            f"{(n_sensors, n_modes)}"
        )

    return rom

# =============================================================================
# 5. 读取STAR-CCM+传感器文件
# =============================================================================

def parse_star_float(
    row: list[str],
    column_index: dict[str, int],
    column_name: str,
    line_number: int,
) -> float:
    """
    从STAR CSV行中读取浮点数并检查有效性。
    """
    index = column_index[column_name]

    if index >= len(row):
        raise ValueError(
            f"STAR CSV第{line_number}行"
            f"缺少字段：{column_name}"
        )

    text = str(row[index]).strip()

    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(
            f"STAR CSV第{line_number}行"
            f"{column_name}不是有效数字：{text}"
        ) from exc

    if not np.isfinite(value):
        raise ValueError(
            f"STAR CSV第{line_number}行"
            f"{column_name}包含NaN或Inf"
        )

    return value

def load_all_star_measurements(
    path: Path,
) -> pd.DataFrame:
    """
    读取并展开STAR-CCM+传感器文件。

    对于：
        "QR_All_5,QR_Fin1_2"

    将展开为两个逻辑传感器记录，但共享同一个物理测量值。
    """
    path = (
        path.expanduser().resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"STAR测点文件不存在：{path}"
        )

    parsed_rows = []

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.reader(file)

        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError(
                f"STAR测点文件为空：{path}"
            ) from exc

        header = [
            str(value).strip()
            for value in raw_header
        ]

        required_columns = {
            STAR_PRESSURE_COLUMN,
            STAR_TOTAL_PRESSURE_COLUMN,
            STAR_X_COLUMN,
            STAR_Y_COLUMN,
            STAR_Z_COLUMN,
            STAR_ID_COLUMN,
        }

        missing_columns = (
            required_columns
            - set(header)
        )

        if missing_columns:
            raise KeyError(
                "STAR测点文件缺少字段："
                f"{sorted(missing_columns)}\n"
                f"实际字段：{header}"
            )

        column_index = {
            name: header.index(name)
            for name in required_columns
        }

        id_index = column_index[
            STAR_ID_COLUMN
        ]

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            if not row:
                continue

            if not any(
                str(value).strip()
                for value in row
            ):
                continue

            absolute_pressure = (
                parse_star_float(
                    row=row,
                    column_index=column_index,
                    column_name=(
                        STAR_PRESSURE_COLUMN
                    ),
                    line_number=line_number,
                )
            )

            absolute_total_pressure = (
                parse_star_float(
                    row=row,
                    column_index=column_index,
                    column_name=(
                        STAR_TOTAL_PRESSURE_COLUMN
                    ),
                    line_number=line_number,
                )
            )

            x_m = parse_star_float(
                row,
                column_index,
                STAR_X_COLUMN,
                line_number,
            )

            y_m = parse_star_float(
                row,
                column_index,
                STAR_Y_COLUMN,
                line_number,
            )

            z_m = parse_star_float(
                row,
                column_index,
                STAR_Z_COLUMN,
                line_number,
            )

            # ID可能正常位于一个带引号的字段中，也可能因没有
            # 加引号而被CSV拆成ID之后的多个字段。
            id_fragments = [
                str(value).strip()
                for value in row[id_index:]
                if str(value).strip()
            ]

            combined_id = ",".join(
                id_fragments
            )

            sensor_names = [
                value.strip()
                for value in combined_id.split(",")
                if value.strip()
            ]

            if not sensor_names:
                raise ValueError(
                    f"STAR CSV第{line_number}行"
                    "没有有效传感器ID"
                )

            for sensor_name in sensor_names:
                match = re.fullmatch(
                    (
                        r"(DEIM|QR)_"
                        r"(All|Hull|Sail|Fin[1-4])_"
                        r"(\d+)"
                    ),
                    sensor_name,
                    flags=re.IGNORECASE,
                )

                if match is None:
                    raise ValueError(
                        f"STAR CSV第{line_number}行"
                        "传感器ID无法识别："
                        f"{sensor_name}"
                    )

                parsed_rows.append({
                    "sensor_method": (
                        match.group(1).lower()
                    ),
                    "sensor_part": (
                        match.group(2).lower()
                    ),
                    "sensor_rank": int(
                        match.group(3)
                    ),
                    "sensor_name": sensor_name,
                    "absolute_pressure_pa": (
                        absolute_pressure
                    ),
                    "absolute_total_pressure_pa": (
                        absolute_total_pressure
                    ),
                    "x_m": x_m,
                    "y_m": y_m,
                    "z_m": z_m,
                    "star_csv_line": line_number,
                    "source_combined_id": (
                        combined_id
                    ),
                })

    if not parsed_rows:
        raise ValueError(
            f"STAR测点文件中没有有效记录：{path}"
        )

    return pd.DataFrame(
        parsed_rows
    )

def load_topology_centroid() -> np.ndarray:
    """
    读取Master表面拓扑面心坐标。
    """
    topology_path = (
        Path(cfg.TOPOLOGY_PATH)
        .expanduser()
        .resolve()
    )

    if not topology_path.is_file():
        raise FileNotFoundError(
            f"表面拓扑不存在：{topology_path}"
        )

    with h5py.File(
        topology_path,
        mode="r",
    ) as file:
        if "geometry/centroid" not in file:
            raise KeyError(
                "表面拓扑中不存在"
                "/geometry/centroid"
            )

        centroid = np.asarray(
            file["geometry/centroid"][()],
            dtype=np.float64,
        )

    if (
        centroid.ndim != 2
        or centroid.shape[1] != 3
    ):
        raise ValueError(
            "表面拓扑centroid形状必须为(n_faces,3)"
        )

    return centroid

def load_star_measurements(
    rom: dict,
) -> pd.DataFrame:
    """
    提取当前传感器方法、当前部件对应的STAR测量值，
    并按照传感器排名排序。
    """
    method = get_sensor_method()
    part_name = get_part_name()
    star_part_name = PART_TO_STAR_NAME[
        part_name
    ]

    measurement_path = (
        STAR_MEASUREMENT_FILES[method]
    )

    table = load_all_star_measurements(
        measurement_path
    )

    selected = table.loc[
        (
            table["sensor_method"]
            == method
        )
        & (
            table["sensor_part"]
            == star_part_name
        )
    ].copy()

    selected = selected.sort_values(
        "sensor_rank",
        kind="stable",
    ).reset_index(drop=True)

    expected_count = len(
        rom["sensor_global_face_id"]
    )

    expected_ranks = np.arange(
        1,
        expected_count + 1,
        dtype=np.int64,
    )

    actual_ranks = selected[
        "sensor_rank"
    ].to_numpy(dtype=np.int64)

    if selected[
        "sensor_rank"
    ].duplicated().any():
        duplicate_ranks = selected.loc[
            selected[
                "sensor_rank"
            ].duplicated(
                keep=False
            ),
            "sensor_rank",
        ].tolist()

        raise ValueError(
            f"{method}/{part_name}"
            "存在重复传感器排名："
            f"{duplicate_ranks}"
        )

    if not np.array_equal(
        actual_ranks,
        expected_ranks,
    ):
        raise ValueError(
            f"{method}/{part_name}"
            "传感器排名不完整或数量不正确。\n"
            f"实际排名：{actual_ranks.tolist()}\n"
            f"期望排名：{expected_ranks.tolist()}\n"
            f"STAR文件：{measurement_path}"
        )

    # ---------------------------------------------------------
    # 使用ROM global_face_id核对STAR测点坐标
    # ---------------------------------------------------------
    centroid = load_topology_centroid()

    sensor_global_ids = np.asarray(
        rom["sensor_global_face_id"],
        dtype=np.int64,
    )

    if np.any(
        sensor_global_ids < 0
    ) or np.any(
        sensor_global_ids >= len(centroid)
    ):
        raise ValueError(
            "ROM中的sensor_global_face_id越界"
        )

    expected_xyz = centroid[
        sensor_global_ids
    ]

    star_xyz = selected[
        [
            "x_m",
            "y_m",
            "z_m",
        ]
    ].to_numpy(dtype=np.float64)

    coordinate_error = np.linalg.norm(
        star_xyz - expected_xyz,
        axis=1,
    )

    invalid = (
        coordinate_error
        > STAR_COORDINATE_TOLERANCE_M
    )

    if np.any(invalid):
        details = []

        for index in np.flatnonzero(
            invalid
        ):
            details.append(
                f"rank={int(actual_ranks[index])}, "
                f"ID={selected.iloc[index]['sensor_name']}, "
                f"distance="
                f"{coordinate_error[index]:.6e} m"
            )

        raise ValueError(
            "STAR测点与当前ROM传感器坐标不一致。\n"
            + "\n".join(details)
            + "\n可能原因："
            "QR/DEIM的ROM文件被覆盖，"
            "或当前PART_NAME/SENSOR_METHOD与ROM不匹配。"
        )

    selected[
        "global_face_id"
    ] = sensor_global_ids

    selected[
        "expected_x_m"
    ] = expected_xyz[:, 0]

    selected[
        "expected_y_m"
    ] = expected_xyz[:, 1]

    selected[
        "expected_z_m"
    ] = expected_xyz[:, 2]

    selected[
        "coordinate_error_m"
    ] = coordinate_error

    return selected

# =============================================================================
# 6. 压力转换
# =============================================================================

def convert_pressure_to_cp(
    pressure_values: np.ndarray,
) -> np.ndarray:
    """
    将输入压力转换为静压力系数。

    cp_static =
        (p_static_absolute - p_static_ref) / q_inf
    """
    pressure_values = np.asarray(
        pressure_values,
        dtype=np.float64,
    )

    if np.any(
        ~np.isfinite(pressure_values)
    ):
        raise ValueError(
            "传感器压力包含NaN或Inf"
        )

    speed_inf = (
        calculate_freestream_speed()
    )

    q_inf = calculate_dynamic_pressure(
        speed_inf
    )

    pressure_input_type = str(
        cfg.PRESSURE_INPUT_TYPE
    ).strip().lower()

    if pressure_input_type == "cp_static":
        return pressure_values

    if pressure_input_type == (
        "absolute_pressure"
    ):
        return (
            pressure_values
            - float(cfg.P_STATIC_REF_PA)
        ) / q_inf

    if pressure_input_type == (
        "static_pressure_relative"
    ):
        return (
            pressure_values
            / q_inf
        )

    raise ValueError(
        "未知PRESSURE_INPUT_TYPE："
        f"{cfg.PRESSURE_INPUT_TYPE}\n"
        "允许值：cp_static、absolute_pressure、"
        "static_pressure_relative"
    )

# =============================================================================
# 7. POD系数求解
# =============================================================================

def solve_coefficients(
    sensor_cp: np.ndarray,
    sensor_mean: np.ndarray,
    sensor_basis: np.ndarray,
) -> np.ndarray:
    """
    根据稀疏传感器Cp求解POD系数。
    """
    sensor_cp = np.asarray(
        sensor_cp,
        dtype=np.float64,
    )

    sensor_mean = np.asarray(
        sensor_mean,
        dtype=np.float64,
    )

    sensor_basis = np.asarray(
        sensor_basis,
        dtype=np.float64,
    )

    if len(sensor_cp) != len(
        sensor_mean
    ):
        raise ValueError(
            "传感器Cp数量与ROM传感器平均值数量不一致"
        )

    if sensor_basis.shape[0] != len(
        sensor_cp
    ):
        raise ValueError(
            "sensor_basis行数与传感器数量不一致"
        )

    observed_fluctuation = (
        sensor_cp
        - sensor_mean
    )

    solver = str(
        cfg.COEFFICIENT_SOLVER
    ).strip().lower()

    if solver == "lstsq":
        coefficients, _, rank, singular_values = (
            np.linalg.lstsq(
                sensor_basis,
                observed_fluctuation,
                rcond=None,
            )
        )

        print(
            f"传感矩阵数值秩："
            f"{rank}/{sensor_basis.shape[1]}"
        )

        if len(singular_values) > 0:
            print(
                "传感矩阵最小奇异值："
                f"{singular_values[-1]:.6e}"
            )

        return coefficients

    if solver == "ridge":
        matrix = (
            sensor_basis.T
            @ sensor_basis
        )

        rhs = (
            sensor_basis.T
            @ observed_fluctuation
        )

        ridge_lambda = float(
            cfg.RIDGE_LAMBDA
        )

        if ridge_lambda < 0.0:
            raise ValueError(
                "RIDGE_LAMBDA不能小于0"
            )

        regularization = (
            ridge_lambda
            * np.eye(
                matrix.shape[0],
                dtype=np.float64,
            )
        )

        return np.linalg.solve(
            matrix + regularization,
            rhs,
        )

    raise ValueError(
        "未知COEFFICIENT_SOLVER："
        f"{cfg.COEFFICIENT_SOLVER}"
    )

# =============================================================================
# 8. 表面压力重构
# =============================================================================

def reconstruct() -> None:
    """
    主重构流程。
    """
    method = get_sensor_method()
    part_name = get_part_name()

    prediction_root = Path(
        cfg.PREDICTION_ROOT
    ).expanduser().resolve()

    prediction_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("STAR-CCM+稀疏压力测量表面重构")
    print("=" * 90)
    print(f"传感器方法：{method.upper()}")
    print(f"目标部件：{part_name}")
    print(
        "STAR测量文件："
        f"{STAR_MEASUREMENT_FILES[method]}"
    )

    rom = load_rom()

    print(
        f"ROM文件：{rom['rom_path']}"
    )
    print(
        f"ROM目标场：{rom['target_field']}"
    )
    print(
        f"ROM表面面数："
        f"{len(rom['global_face_id']):,}"
    )
    print(
        f"ROM模态数："
        f"{rom['basis'].shape[1]}"
    )
    print(
        f"ROM传感器数："
        f"{len(rom['sensor_global_face_id'])}"
    )

    measurements = load_star_measurements(
        rom
    )

    sensor_pressure_absolute = (
        measurements[
            "absolute_pressure_pa"
        ].to_numpy(dtype=np.float64)
    )

    sensor_cp = convert_pressure_to_cp(
        sensor_pressure_absolute
    )

    coefficients = solve_coefficients(
        sensor_cp=sensor_cp,
        sensor_mean=rom["sensor_mean"],
        sensor_basis=rom["sensor_basis"],
    )

    reconstructed_cp = (
        rom["mean_field"]
        + rom["basis"]
        @ coefficients
    )

    fitted_sensor_cp = (
        rom["sensor_mean"]
        + rom["sensor_basis"]
        @ coefficients
    )

    sensor_residual_cp = (
        fitted_sensor_cp
        - sensor_cp
    )

    speed_inf = (
        calculate_freestream_speed()
    )

    q_inf = calculate_dynamic_pressure(
        speed_inf
    )

    p_static_ref = float(
        cfg.P_STATIC_REF_PA
    )

    reconstructed_pressure = (
        p_static_ref
        + q_inf
        * reconstructed_cp
    )

    fitted_sensor_pressure = (
        p_static_ref
        + q_inf
        * fitted_sensor_cp
    )

    sensor_residual_pressure = (
        fitted_sensor_pressure
        - sensor_pressure_absolute
    )

    result_prefix = (
        f"{cfg.TEST_CASE_ID}_"
        f"{part_name}_"
        f"{method}"
    )

    # ---------------------------------------------------------
    # 保存完整表面重构
    # ---------------------------------------------------------
    output = pd.DataFrame({
        "global_face_id": rom[
            "global_face_id"
        ],
        "cp_static_reconstructed": (
            reconstructed_cp
        ),
        "p_static_absolute_reconstructed_pa": (
            reconstructed_pressure
        ),
    })

    output_path = (
        prediction_root
        / f"{result_prefix}_prediction.csv"
    )

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------
    # 保存POD系数
    # ---------------------------------------------------------
    coefficient_output = (
        prediction_root
        / f"{result_prefix}_coefficients.csv"
    )

    pd.DataFrame({
        "mode_index": np.arange(
            len(coefficients),
            dtype=np.int64,
        ) + 1,
        "coefficient": coefficients,
    }).to_csv(
        coefficient_output,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------
    # 保存传感器匹配和拟合结果
    # ---------------------------------------------------------
    sensor_output = (
        measurements.copy()
    )

    sensor_output[
        "cp_observed"
    ] = sensor_cp

    sensor_output[
        "cp_fitted"
    ] = fitted_sensor_cp

    sensor_output[
        "cp_residual_fitted_minus_observed"
    ] = sensor_residual_cp

    sensor_output[
        "absolute_pressure_observed_pa"
    ] = sensor_pressure_absolute

    sensor_output[
        "absolute_pressure_fitted_pa"
    ] = fitted_sensor_pressure

    sensor_output[
        "pressure_residual_fitted_minus_observed_pa"
    ] = sensor_residual_pressure

    sensor_output_path = (
        prediction_root
        / f"{result_prefix}_sensor_fit.csv"
    )

    sensor_output.to_csv(
        sensor_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------
    # 保存本次计算摘要
    # ---------------------------------------------------------
    condition_number = float(
        np.linalg.cond(
            rom["sensor_basis"]
        )
    )

    cp_rmse = float(
        np.sqrt(
            np.mean(
                sensor_residual_cp ** 2
            )
        )
    )

    pressure_rmse = float(
        np.sqrt(
            np.mean(
                sensor_residual_pressure ** 2
            )
        )
    )

    summary_output = (
        prediction_root
        / f"{result_prefix}_summary.csv"
    )

    pd.DataFrame([{
        "test_case_id": str(
            cfg.TEST_CASE_ID
        ),
        "part_name": part_name,
        "sensor_method": method,
        "test_u_inf_m_s": float(
            cfg.TEST_U_INF
        ),
        "test_v_inf_m_s": float(
            cfg.TEST_V_INF
        ),
        "test_w_inf_m_s": float(
            cfg.TEST_W_INF
        ),
        "speed_inf_m_s": speed_inf,
        "rho_kg_m3": float(cfg.RHO),
        "dynamic_pressure_pa": q_inf,
        "p_static_ref_pa": p_static_ref,
        "sensor_count": len(
            measurements
        ),
        "pod_mode_count": len(
            coefficients
        ),
        "sensor_matrix_condition_number": (
            condition_number
        ),
        "sensor_cp_rmse": cp_rmse,
        "sensor_pressure_rmse_pa": (
            pressure_rmse
        ),
        "maximum_coordinate_error_m": float(
            measurements[
                "coordinate_error_m"
            ].max()
        ),
        "rom_path": str(
            rom["rom_path"]
        ),
        "star_measurement_path": str(
            STAR_MEASUREMENT_FILES[method]
        ),
    }]).to_csv(
        summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 90)
    print("重构完成")
    print("=" * 90)
    print(
        f"来流速度模长："
        f"{speed_inf:.6f} m/s"
    )
    print(
        f"来流动压："
        f"{q_inf:.6f} Pa"
    )
    print(
        f"传感矩阵条件数："
        f"{condition_number:.6e}"
    )
    print(
        f"传感器Cp拟合RMSE："
        f"{cp_rmse:.6e}"
    )
    print(
        f"传感器压力拟合RMSE："
        f"{pressure_rmse:.6e} Pa"
    )
    print(
        f"最大坐标匹配误差："
        f"{measurements['coordinate_error_m'].max():.6e} m"
    )
    print(
        f"完整压力场：{output_path}"
    )
    print(
        f"POD系数：{coefficient_output}"
    )
    print(
        f"传感器拟合：{sensor_output_path}"
    )
    print(
        f"计算摘要：{summary_output}"
    )
    print("=" * 90)

# =============================================================================
# 9. 程序入口
# =============================================================================

if __name__ == "__main__":
    original_method = str(
        cfg.SENSOR_METHOD
    )

    successful_methods = []
    failed_methods = []

    try:
        for sensor_method in [
            "deim",
            "qr",
        ]:
            print("\n\n")
            print("#" * 90)
            print(
                f"开始执行"
                f"{sensor_method.upper()}"
                "传感器压力重构"
            )
            print("#" * 90)

            cfg.SENSOR_METHOD = sensor_method

            try:
                reconstruct()

                successful_methods.append(
                    sensor_method
                )

            except Exception as exc:
                failed_methods.append({
                    "sensor_method": (
                        sensor_method
                    ),
                    "error": repr(exc),
                })

                print("\n" + "!" * 90)
                print(
                    f"{sensor_method.upper()}"
                    "重构失败"
                )
                print(
                    f"错误信息：{exc}"
                )
                print("!" * 90)

    finally:
        cfg.SENSOR_METHOD = original_method

    print("\n\n")
    print("=" * 90)
    print("QR和DEIM重构执行摘要")
    print("=" * 90)

    print(
        "成功方法："
        + (
            ", ".join(
                method.upper()
                for method in successful_methods
            )
            if successful_methods
            else "无"
        )
    )

    if failed_methods:
        print("失败方法：")

        for item in failed_methods:
            print(
                f"  - "
                f"{item['sensor_method'].upper()}: "
                f"{item['error']}"
            )
    else:
        print("失败方法：无")

    print("=" * 90)

    if failed_methods:
        raise RuntimeError(
            "存在传感器方法重构失败，"
            "请检查上述错误信息。"
        )