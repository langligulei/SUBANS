# -*- coding: utf-8 -*-

from pathlib import Path

# =========================================================
# 1. 数据集路径
# =========================================================

DATASET_ROOT = Path(
    r"E:\suboff（0-200）\suboff_0000"
    r"\modal_surface_dataset"
)

MODAL_ANALYSIS_ROOT = (
    DATASET_ROOT / "modal_analysis"
)

CP_STATIC_ROOT = (
    MODAL_ANALYSIS_ROOT / "cp_static"
)

TOPOLOGY_PATH = (
    DATASET_ROOT / "topology.h5"
)

CASES_CSV_PATH = (
    DATASET_ROOT / "cases.csv"
)

# =========================================================
# 2. 研究部件
# =========================================================

# 当前建议先使用艇体。
PART_NAME = "fin4"

# 可选：
# "all_surface"
# "hull"
# "sail"
# "fin1"
# "fin2"
# "fin3"
# "fin4"

POD_H5_PATH = (
    CP_STATIC_ROOT
    / f"{PART_NAME}_pod.h5"
)

# =========================================================
# 3. 压力定义
# =========================================================

# POD文件中已有的目标变量。
TARGET_FIELD = "cp_static"

# 新STAR结果若提供绝对静压，则使用：
PRESSURE_INPUT_TYPE = "absolute_pressure"

# 可选：
# "absolute_pressure"
# "cp_static"
# "static_pressure_relative"

RHO = 997.561

P_STATIC_REF_PA = 101325.0

# =========================================================
# 4. 新工况参数
# =========================================================

# 新速度必须与STAR-CCM+实际计算速度一致。
TEST_CASE_ID = "suboff_0000_v4p5"

TEST_U_INF = 4.5
TEST_V_INF = 0.0
TEST_W_INF = 0.0

# 新速度STAR-CCM+导出的完整SBD，仅用于验证预测结果。
TEST_SBD_PATH = Path(
    r"E:\suboff（0-200）\suboff_0000\SurfaceCheck"
    r"\suboff_0000_v4p5.sbd"
)

# 数据集构建程序，用于复用SBD读取和SBD->CGNS映射。
DATASET_BUILDER_PATH = Path(
    r"F:\14_Sub\SUBANS\processed"
    r"\build_modal_surface_dataset.py"
)

# =========================================================
# 5. 传感器设置
# =========================================================

# 优先使用modal_analysis_pressure.py选出的DEIM点。
SENSOR_METHOD = "qr"

# 可选：
# "deim"
# "qr"
# "custom"

# 几个测点压力。必须不超过POD有效模态数。
N_SENSORS = 9

# 如果是custom，则提供测点坐标文件。
CUSTOM_SENSOR_XYZ_CSV = (
    Path(__file__).parent
    / "custom_sensor_xyz.csv"
)

# =========================================================
# 6. 稀疏系数求解
# =========================================================

# "lstsq"：最小二乘，推荐
# "ridge"：岭回归，测量有噪声时更稳定
COEFFICIENT_SOLVER = "lstsq"

RIDGE_LAMBDA = 1.0e-8

# 是否对传感器压力加入噪声测试
ADD_MEASUREMENT_NOISE = False

MEASUREMENT_NOISE_STD = 0.0

# =========================================================
# 7. 输出路径
# =========================================================

SPARSE_ROOT = (
    MODAL_ANALYSIS_ROOT
    / "sparse_pressure"
)

SENSOR_ROOT = (
    SPARSE_ROOT / "sensors"
)

ROM_ROOT = (
    SPARSE_ROOT / "rom"
)

PREDICTION_ROOT = (
    SPARSE_ROOT / "predictions"
)

TRUTH_ROOT = (
    SPARSE_ROOT / "truth"
)

COMPARISON_ROOT = (
    SPARSE_ROOT / "comparison"
)

for path in [
    SPARSE_ROOT,
    SENSOR_ROOT,
    ROM_ROOT,
    PREDICTION_ROOT,
    TRUTH_ROOT,
    COMPARISON_ROOT,
]:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )