# -*- coding: utf-8 -*-
"""
SUBOFF Experiment实验配置。

本实验程序的未见工况原始输入为STAR-CCM+导出的SBD文件。
程序会先将SBD转换成与topology.h5完全一致的统一面顺序HDF5，
然后再运行稀疏POD、QR、DEIM、均匀布点和随机布点实验。
"""

from pathlib import Path

# =============================================================================
# 1. Experiment路径
# =============================================================================

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
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

# =============================================================================
# 2. 已有统一表面数据集路径
# =============================================================================

# 该目录中应已经存在：
#   topology.h5
#   cases.csv
#   cases/
DATASET_ROOT = Path(
    r"E:\suboff（0-200）\suboff_0000"
    r"\modal_surface_dataset"
)

TOPOLOGY_PATH = DATASET_ROOT / "topology.h5"
TRAIN_CASES_CSV_PATH = DATASET_ROOT / "cases.csv"
# 六个未见速度工况的自动生成清单。
#
# 该文件由01_build_unseen_cases_from_sbd.py生成。
UNSEEN_CASES_CSV_PATH = (
    EXPERIMENT_ROOT / "unseen_cases.csv"
)
# =============================================================================
# 2.1 来流速度字段
# =============================================================================

VELOCITY_COLUMNS = [
    "u_inf_m_s",
    "v_inf_m_s",
    "w_inf_m_s",
]
# =============================================================================
# 3. 现有SBD解析程序路径
# =============================================================================

# 该文件就是你当前项目中已有的程序。
# 它内部包含：
#   read_sbd_case
#   build_sbd_to_cgns_mapping
#   reorder_to_cgns
#   calculate_case_fields
# 等函数。
#
# 如果实际文件位置不同，只修改这一行。
BUILDER_SCRIPT_PATH = Path(
    r"F:\14_Sub\SUBANS\processed"
    r"\build_modal_surface_dataset.py"
)

# build_modal_surface_dataset.py内部还会加载旧的SBD底层解析程序。
# 通常该路径已经在builder文件中配置好。
# 如果builder内部的LEGACY_SCRIPT_PATH正确，这里不需要改。

# =============================================================================
# 4. 六个未见速度的SBD路径
# =============================================================================

SURFACE_CHECK_ROOT = Path(
    r"E:\suboff（0-200）\suboff_0000"
    r"\SurfaceCheck"
)

UNSEEN_SBD_CASES = [
    {
        "case_id": "suboff_0000_v0p75",
        "speed_m_s": 0.75,
        "sbd_path": (
            SURFACE_CHECK_ROOT
            / "v0p75"
            / "suboff_0000_v0p75.sbd"
        ),
        "deim_sensor_path": (
            SURFACE_CHECK_ROOT
            / "v0p75"
            / "DEIMSensorFromStar__v0p75.csv"
        ),
        "qr_sensor_path": (
            SURFACE_CHECK_ROOT
            / "v0p75"
            / "QRSensorFromStar__v0p75.csv"
        ),
        "split_type": "extrapolation",
    },
    {
        "case_id": "suboff_0000_v2p5",
        "speed_m_s": 2.5,
        "sbd_path": (
            SURFACE_CHECK_ROOT
            / "v2p5"
            / "suboff_0000_v2p5.sbd"
        ),
        "deim_sensor_path": (
            SURFACE_CHECK_ROOT
            / "v2p5"
            / "DEIMSensorFromStar__v2p5.csv"
        ),
        "qr_sensor_path": (
            SURFACE_CHECK_ROOT
            / "v2p5"
            / "QRSensorFromStar__v2p5.csv"
        ),
        "split_type": "interpolation",
    },
    {
        "case_id": "suboff_0000_v4p5",
        "speed_m_s": 4.5,
        "sbd_path": (
            SURFACE_CHECK_ROOT
            / "v4p5"
            / "suboff_0000_v4p5.sbd"
        ),
        "deim_sensor_path": (
            SURFACE_CHECK_ROOT
            / "v4p5"
            / "DEIMSensorFromStar__v4p5.csv"
        ),
        "qr_sensor_path": (
            SURFACE_CHECK_ROOT
            / "v4p5"
            / "QRSensorFromStar__v4p5.csv"
        ),
        "split_type": "interpolation",
    },
    {
        "case_id": "suboff_0000_v6p5",
        "speed_m_s": 6.5,
        "sbd_path": (
            SURFACE_CHECK_ROOT
            / "v6p5"
            / "suboff_0000_v6p5.sbd"
        ),
        "deim_sensor_path": (
            SURFACE_CHECK_ROOT
            / "v6p5"
            / "DEIMSensorFromStar__v6p5.csv"
        ),
        "qr_sensor_path": (
            SURFACE_CHECK_ROOT
            / "v6p5"
            / "QRSensorFromStar__v6p5.csv"
        ),
        "split_type": "interpolation",
    },
    {
        "case_id": "suboff_0000_v8p5",
        "speed_m_s": 8.5,
        "sbd_path": (
            SURFACE_CHECK_ROOT
            / "v8p5"
            / "suboff_0000_v8p5.sbd"
        ),
        "deim_sensor_path": (
            SURFACE_CHECK_ROOT
            / "v8p5"
            / "DEIMSensorFromStar__v8p5.csv"
        ),
        "qr_sensor_path": (
            SURFACE_CHECK_ROOT
            / "v8p5"
            / "QRSensorFromStar__v8p5.csv"
        ),
        "split_type": "interpolation",
    },
    {
        "case_id": "suboff_0000_v10p5",
        "speed_m_s": 10.5,
        "sbd_path": (
            SURFACE_CHECK_ROOT
            / "v10p5"
            / "suboff_0000_v10p5.sbd"
        ),
        "deim_sensor_path": (
            SURFACE_CHECK_ROOT
            / "v10p5"
            / "DEIMSensorFromStar__v10p5.csv"
        ),
        "qr_sensor_path": (
            SURFACE_CHECK_ROOT
            / "v10p5"
            / "QRSensorFromStar__v10p5.csv"
        ),
        "split_type": "extrapolation",
    },
]

# =============================================================================
# 5. 物理参数
# =============================================================================

RHO = 997.561
P_STATIC_REF_PA = 101325.0

# 与现有build_modal_surface_dataset.py保持一致。
REFERENCE_LENGTH_M = 4.356

# 参考面积。请按你的论文定义修改。
REFERENCE_AREA_M2 = 0.2027

MOMENT_REFERENCE_POINT_M = (
    0.0,
    0.0,
    0.0,
)

# =============================================================================
# 6. 分析部件和变量
# =============================================================================

# 推荐主论文先使用all_surface或hull。
PART_NAME = "all_surface"
TARGET_FIELD = "cp_static"

# =============================================================================
# 7. POD和传感器实验参数
# =============================================================================

MAX_POD_MODES = 9

SENSOR_COUNTS = [
    3,
    5,
    7,
    9,
    12,
    15,
    20,
]

POD_MODE_COUNTS = list(range(1, 10))

MODE_SENSITIVITY_SENSOR_COUNT = 12
NOISE_SENSOR_COUNT = 12
NOISE_MODE_COUNT = 9

NOISE_LEVELS = [
    0.001,
    0.005,
    0.010,
    0.020,
    0.050,
]

RANDOM_SENSOR_REPEATS = 200
NOISE_REPEATS = 100
RANDOM_SEED = 20260828

DETERMINISTIC_SENSOR_METHODS = [
    "qr",
    "deim",
    "uniform",
]

RUN_RANDOM_SENSOR_BASELINE = True
RUN_MEAN_FIELD_BASELINE = True
RUN_LINEAR_SPEED_BASELINE = True

# 当测试速度与已有训练速度相同，自动剔除同速训练工况。
EXCLUDE_TEST_SPEED_FROM_TRAINING = True
SPEED_MATCH_TOLERANCE = 1.0e-8

# 采用Ridge以支持m<r的欠定情况。
COEFFICIENT_SOLVER = "ridge"
RIDGE_LAMBDA = 1.0e-8
LSTSQ_RCOND = None

# =============================================================================
# 8. 输出设置
# =============================================================================

SAVE_MAIN_PREDICTIONS = True
# 保存随机传感器所有重复试验的原始结果。
SAVE_RANDOM_RAW_RESULTS = True

MAIN_SENSOR_METHOD = "qr"
MAIN_SENSOR_COUNT = 12
MAIN_MODE_COUNT = 9

FIGURE_DPI = 300
FIGURE_FORMATS = [
    "png",
    "pdf",
]

PLOT_FONT_FAMILY = "DejaVu Sans"
CSV_FLOAT_FORMAT = "%.12e"

# =============================================================================
# 9. 法向量配置
# =============================================================================

# 当前topology.h5中的法向量路径。
FACE_NORMAL_DATASET = (
    "geometry/normal"
)

# 自动搜索候选路径。
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

# 如果法向量方向与论文定义相反，改为-1.0。
NORMAL_DIRECTION_SIGN = 1.0