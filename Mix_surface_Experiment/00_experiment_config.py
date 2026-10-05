# -*- coding: utf-8 -*-
"""Mix_surface_Experiment三向复杂工况实验配置。"""

from pathlib import Path
import re
import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parent

RESULT_ROOT = EXPERIMENT_ROOT / "results"
UNSEEN_H5_ROOT = RESULT_ROOT / "unseen_cases"
TABLE_ROOT = RESULT_ROOT / "tables"
FIGURE_ROOT = RESULT_ROOT / "figures"
PREDICTION_ROOT = RESULT_ROOT / "predictions"
LOG_ROOT = RESULT_ROOT / "logs"

for directory in [
    RESULT_ROOT,
    UNSEEN_H5_ROOT,
    TABLE_ROOT,
    FIGURE_ROOT,
    PREDICTION_ROOT,
    LOG_ROOT,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 数据路径
# ============================================================================

DATASET_ROOT = Path(
    r"E:\suboff（0-200）\suboff_0000"
    r"\modal_surface_dataset_mix"
)

TOPOLOGY_PATH = DATASET_ROOT / "topology.h5"
TRAIN_CASES_CSV_PATH = DATASET_ROOT / "cases.csv"

UNSEEN_CASES_CSV_PATH = (
    EXPERIMENT_ROOT / "unseen_cases.csv"
)

# =============================================================================
# 工程试验传感器配置
# =============================================================================

# 真实试验主方法
MAIN_SENSOR_METHOD = "practical"
SENSOR_LAYOUT_EXPERIMENTS = [
    "practical",
    "qr",
    "deim",
]

# 当前工程测点总数
MAIN_SENSOR_COUNT = 126

# 测点数量敏感性只测试工程布点的前缀子集
SENSOR_COUNTS = [60, 80, 100, 126]

# 工程测点定义文件
PRACTICAL_SENSOR_DEFINITION_PATH = (
    DATASET_ROOT
    / "modal_analysis"
    / "cp_static"
    / "all_surface_practical_sensors.csv"
)

# =============================================================================
# 链路B：CGNS提取的工程传感器观测数据
# =============================================================================

# True：
#   03_run_sparse_rom_experiments.py使用
#   01_extract_sensor_values_from_cfd.py生成的传感器矩阵。
#
# False：
#   回退到旧的逐工况PracticalSensorFromStar_*.csv流程。
USE_EXTRACTED_SENSOR_MATRIX = True

# 由01_extract_sensor_values_from_cfd.py生成的传感器Cp矩阵
EXTRACTED_SENSOR_MATRIX_PATH = (
    DATASET_ROOT
    / "sensor_check"
    / "sensor_measurements_matrix.csv"
)

# 主模型使用当前40阶POD基
# 【改进】按模态敏感性最优改为 30 阶（10 阶欠拟合、70 阶病态）
MAIN_MODE_COUNT = 30
MAX_POD_MODES = 70

# 模态数量敏感性
POD_MODE_COUNTS = [
    10,
    20,
    30,
    40,
    50,
    60,
    70,
]

# 模态敏感性实验中使用全部工程测点
MODE_SENSITIVITY_SENSOR_COUNT = 126

# 噪声实验
NOISE_SENSOR_COUNT = 126
NOISE_MODE_COUNT = 70
NOISE_LEVELS = [
    0.0,
    0.001,
    0.005,
    0.010,
    0.020,
]
NOISE_REPEATS = 30

# 关闭旧DEIM、QR主流程
RUN_DEIM_EXPERIMENT = False
RUN_QR_EXPERIMENT = False

# =============================================================================
# 参数预测模型配置
# =============================================================================

RUN_PARAMETER_RBF_EXPERIMENT = True

# 你的交叉验证结果表明 uvw 优于 U-alpha-beta
PARAMETER_FEATURE_SET = "uvw"

# RBF核岭模型参数。
# 因为特征会标准化，length scale为无量纲参数。
PARAMETER_RBF_LENGTH_SCALE = 0.5
PARAMETER_RBF_NOISE = 0.05
PARAMETER_RBF_RIDGE = 1.0e-8

# 参数预测使用70阶模态
PARAMETER_MODEL_MODE_COUNT = 70

# 输出参数预测全场
SAVE_PARAMETER_PREDICTIONS = True
# ============================================================================
# 来流速度
# ============================================================================

VELOCITY_COLUMNS = [
    "u_inf_m_s",
    "v_inf_m_s",
    "w_inf_m_s",
]

SPEED_MATCH_TOLERANCE = 1.0e-8

# 坐标匹配容差。
# 建议先使用1e-6 m；如果Star和modal文件坐标差异较大，
# 可改为1e-5 m，但不建议盲目设置过大。
SENSOR_COORD_ABS_TOLERANCE_M = 1.0e-6
SENSOR_COORD_REL_TOLERANCE = 1.0e-8

def speed_magnitude(u_inf_m_s, v_inf_m_s, w_inf_m_s):
    """计算三向来流速度模长。"""
    return float(
        (
            float(u_inf_m_s) ** 2
            + float(v_inf_m_s) ** 2
            + float(w_inf_m_s) ** 2
        ) ** 0.5
    )

# ============================================================================
# SBD解析程序
# ============================================================================

BUILDER_SCRIPT_PATH = Path(
    r"F:\14_Sub\SUBANS\processed"
    r"\build_modal_surface_dataset.py"
)

# =============================================================================
# 4. 自动识别SurfaceCheck中的三向速度工况
# =============================================================================

SURFACE_CHECK_ROOT = Path(
    r"E:\suboff（0-200）\suboff_0000"
    r"\SurfaceCheck"
)

CASE_FOLDER_PATTERN = re.compile(
    r"^u(?P<u>[-+]?\d+(?:p\d+)?)"
    r"v(?P<v>[-+]?\d+(?:p\d+)?)"
    r"w(?P<w>[-+]?\d+(?:p\d+)?)$",
    re.IGNORECASE,
)

def decode_velocity_token(token):
    """
    将文件夹中的速度字符串转换为浮点数。

    例如：
        0p5  ->  0.5
        -0p5 -> -0.5
        2    ->  2.0
    """
    return float(
        str(token).strip().lower().replace("p", ".")
    )

def velocity_token(value):
    """
    将浮点速度转换为文件名中的格式。

    例如：
        0.5  -> 0p5
        -0.5 -> -0p5
        2.0  -> 2
    """
    value = float(value)

    if abs(value - round(value)) < 1.0e-12:
        text = str(int(round(value)))
    else:
        text = (
            f"{value:.12f}"
            .rstrip("0")
            .rstrip(".")
        )

    return text.replace(".", "p")

def speed_magnitude_from_components(
    u_inf_m_s,
    v_inf_m_s,
    w_inf_m_s,
):
    return float(
        (
            float(u_inf_m_s) ** 2
            + float(v_inf_m_s) ** 2
            + float(w_inf_m_s) ** 2
        ) ** 0.5
    )

def make_case_id(u, v, w):
    """
    工况ID格式：

    suboff_0000_u0p5_v0p5_w-0p5
    """
    return (
        "suboff_0000_"
        f"u{velocity_token(u)}_"
        f"v{velocity_token(v)}_"
        f"w{velocity_token(w)}"
    )

def make_file_stem(u, v, w):
    """
    文件名主体格式：

    suboff_0000_u0p5_v0p5_w-0p5
    """
    return make_case_id(u, v, w)

def make_sensor_stem(u, v, w):
    """
    传感器文件名主体格式：

    suboff_0000_u0p5v0p5w-0p5
    """
    return (
        f"u{velocity_token(u)}"
        f"v{velocity_token(v)}"
        f"w{velocity_token(w)}"
    )

def classify_case_group(u, v, w):
    """
    按Group A/B/C/D分类。

    Group A：
        内插工况，但不是固定模长方向变化工况。

    Group B：
        分量绝对值为0.5、1.0、2.0的方向变化工况。

    Group C：
        弱外推，最大分量绝对值为3.5。

    Group D：
        强外推，存在分量绝对值大于3.5。
    """
    values = [
        abs(float(u)),
        abs(float(v)),
        abs(float(w)),
    ]

    if all(value < 3.0 for value in values):
        sorted_values = sorted(values)

        if np.isclose(
            sorted_values,
            [0.5, 1.0, 2.0],
            atol=1.0e-12,
            rtol=0.0,
        ).all():
            return "Group_B"

        return "Group_A"

    if all(value <= 3.5 for value in values):
        return "Group_C"

    return "Group_D"

def discover_unseen_cases():
    """
    自动扫描SurfaceCheck目录。

    文件夹格式：
        u0p5v0p5w-0p5

    文件格式：
        suboff_0000_u0p5_v0p5_w-0p5.sbd
        DEIMSensorFromStar_u0p5v0p5w-0p5.csv
        QRSensorFromStar_u0p5v0p5w-0p5.csv
    """
    if not SURFACE_CHECK_ROOT.is_dir():
        raise FileNotFoundError(
            f"目录不存在：{SURFACE_CHECK_ROOT}"
        )

    cases = []

    for folder in sorted(
        SURFACE_CHECK_ROOT.iterdir()
    ):
        if not folder.is_dir():
            continue

        matched = CASE_FOLDER_PATTERN.fullmatch(
            folder.name
        )

        if matched is None:
            continue

        u = decode_velocity_token(
            matched.group("u")
        )
        v = decode_velocity_token(
            matched.group("v")
        )
        w = decode_velocity_token(
            matched.group("w")
        )

        case_id = make_case_id(u, v, w)
        file_stem = make_file_stem(u, v, w)
        sensor_stem = make_sensor_stem(u, v, w)

        expected_sbd = (
            folder / f"{file_stem}.sbd"
        )

        expected_deim = (
            folder
            / f"DEIMSensorFromStar_{sensor_stem}.csv"
        )

        expected_qr = (
            folder
            / f"QRSensorFromStar_{sensor_stem}.csv"
        )

        sbd_candidates = sorted(
            folder.glob("*.sbd")
        )

        if expected_sbd.is_file():
            sbd_path = expected_sbd
        elif len(sbd_candidates) == 1:
            sbd_path = sbd_candidates[0]
        else:
            sbd_path = expected_sbd

        cgns_candidates = sorted(
            list(folder.glob("*.cgns"))
            + list(folder.glob("*.cgns*"))
        )

        group_name = classify_case_group(
            u,
            v,
            w,
        )

        if group_name in {
            "Group_A",
            "Group_B",
        }:
            split_type = "interpolation"
        else:
            split_type = "extrapolation"

        cases.append({
            "case_id": case_id,
            "u_inf_m_s": u,
            "v_inf_m_s": v,
            "w_inf_m_s": w,
            "speed_inf_m_s": (
                speed_magnitude_from_components(
                    u,
                    v,
                    w,
                )
            ),
            "group_name": group_name,
            "split_type": split_type,
            "case_directory": str(
                folder.resolve()
            ),
            "sbd_path": str(
                sbd_path.resolve()
            ),
            "cgns_path": (
                str(
                    cgns_candidates[0].resolve()
                )
                if cgns_candidates
                else ""
            ),
            "deim_sensor_path": str(
                expected_deim.resolve()
            ),
            "qr_sensor_path": str(
                expected_qr.resolve()
            ),
        })

    if not cases:
        raise RuntimeError(
            "SurfaceCheck目录中没有识别到三向速度工况。"
        )

    return cases

UNSEEN_SBD_CASES = discover_unseen_cases()

# ============================================================================
# 物理参数
# ============================================================================

RHO = 997.561
P_STATIC_REF_PA = 101325.0

REFERENCE_LENGTH_M = 4.356
REFERENCE_AREA_M2 = 0.2027
MOMENT_REFERENCE_POINT_M = (0.0, 0.0, 0.0)

# ============================================================================
# 分析设置
# ============================================================================

PART_NAME = "all_surface"
TARGET_FIELD = "cp_static"


NOISE_LEVELS = [0.001, 0.005, 0.010, 0.020, 0.050]
RANDOM_SENSOR_REPEATS = 200
NOISE_REPEATS = 100
RANDOM_SEED = 20260828

DETERMINISTIC_SENSOR_METHODS = ["practical"]

RUN_RANDOM_SENSOR_BASELINE = True
RUN_MEAN_FIELD_BASELINE = True
RUN_LINEAR_SPEED_BASELINE = False

EXCLUDE_TEST_SPEED_FROM_TRAINING = True

# COEFFICIENT_SOLVER = "lstsq"
RIDGE_LAMBDA = 0.0
COEFFICIENT_SOLVER = "ridge"
RIDGE_LAMBDA = 1.0e-4
LSTSQ_RCOND = None
# =========================================================
# 系数反演算法实验
# =========================================================

RUN_INVERSION_ALGORITHM_EXPERIMENT = True

INVERSION_SOLVERS = [
    "lstsq",
    "ridge",
    "energy_ridge",
]

# 固定80个测点进行算法比较
INVERSION_SENSOR_COUNT = 80

# 推荐测试的POD模态数
INVERSION_MODE_COUNTS = [
    8,
    10,
    12,
    15,
    20,
]

# 相对岭正则化参数
INVERSION_LAMBDAS = [
    0.0,      # 纯lstsq基线
    1.0e-2,   # 弱正则化
    0.1,      # 中等正则化
    1.0,      # 强正则化
    10.0,     # 很强正则化
]

# 是否采用传感器残差权重
USE_SENSOR_RESIDUAL_WEIGHTS = True

# 是否采用速度参数先验（反演算法对比实验用，默认关闭）
USE_VELOCITY_PRIOR = False

# =========================================================
# 【改进】主方法重构配置（用于降低三向速度工况误差）
# =========================================================
# main_model 使用的系数求解器（先验关闭时的回退）
MAIN_SOLVER_FALLBACK = "energy_ridge"
# 是否对 main_model 启用传感器残差权重
MAIN_USE_SENSOR_WEIGHTS = True
# 是否启用速度参数先验（main_model 用 prior_ridge 求解）
MAIN_USE_VELOCITY_PRIOR = True
# 先验强度（离线扫描：0.1 偏大，0.01 在插值44/44、外推7/8 工况更优）
MAIN_PRIOR_LAMBDA = 1.0e-2
# 速度先验多项式阶数与岭正则
VELOCITY_PRIOR_FEATURE_DEGREE = 2
VELOCITY_PRIOR_RIDGE = 1.0e-3

# 速度先验正则化强度
PRIOR_LAMBDAS = [
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
]

# 系数先验模型
VELOCITY_PRIOR_MODEL = "polynomial_ridge"

# 二阶速度特征
VELOCITY_FEATURE_DEGREE = 2
# ============================================================================
# 输出
# ============================================================================

SAVE_MAIN_PREDICTIONS = True
SAVE_RANDOM_RAW_RESULTS = True

FIGURE_DPI = 300
FIGURE_FORMATS = ["png", "pdf"]
PLOT_FONT_FAMILY = "DejaVu Sans"
CSV_FLOAT_FORMAT = "%.12e"

# ============================================================================
# 法向量
# ============================================================================

FACE_NORMAL_DATASET = "geometry/normal"
FACE_NORMAL_CANDIDATES = [
    "geometry/normal",
    "geometry/face_normal",
    "geometry/face_normals",
    "geometry/normals",
    "face_normal",
    "face_normals",
    "normal",
    "normals",
]

NORMAL_DIRECTION_SIGN = 1.0