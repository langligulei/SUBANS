# -*- coding: utf-8 -*-
"""
三向流场稀疏压力重构配置文件。

与直航版本的主要区别：
1. 测试工况包含三个速度分量 (vx, vy, vz)
2. 数据集路径指向 modal_surface_dataset_mix
3. POD分析结果来自 Mix_modal_analysis_pressure.py
"""

from pathlib import Path

# =========================================================
# 1. 数据集路径（三向流场版本）
# =========================================================

DATASET_ROOT = Path(
    r"E:\suboff（0-200）\suboff_0000"
    r"\modal_surface_dataset_mix"
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

PART_NAME = "all_surface"

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

TARGET_FIELD = "cp_static"

PRESSURE_INPUT_TYPE = "absolute_pressure"

# 可选：
# "absolute_pressure"
# "cp_static"
# "static_pressure_relative"

RHO = 997.561
P_STATIC_REF_PA = 101325.0

# =========================================================
# 4. 新工况参数（三向速度）
# =========================================================

# 测试工况ID（三向速度命名格式）
TEST_CASE_ID = "suboff_0000_vx2_vy1_vz0p5"

# 三个速度分量（必须与STAR-CCM+实际计算一致）
TEST_U_INF = 2.0   # X方向速度 (m/s)
TEST_V_INF = 1.0   # Y方向速度 (m/s)
TEST_W_INF = 0.5   # Z方向速度 (m/s)

# 速度模长（自动计算）
import numpy as np
TEST_SPEED_MAGNITUDE = float(np.sqrt(
    TEST_U_INF**2 + TEST_V_INF**2 + TEST_W_INF**2
))

# 新速度STAR-CCM+导出的完整SBD（用于验证）
TEST_SBD_PATH = Path(
    r"E:\suboff（0-200）\suboff_0000\SurfaceCheck"
    r"\u2v1w0p5\suboff_0000_u2_v1_w0p5.sbd"
)

# 数据集构建程序（用于复用SBD读取）
DATASET_BUILDER_PATH = Path(
    r"F:\14_Sub\SUBANS\processed"
    r"\Mix_build_modal_surface_dataset.py"
)
# =========================================================
# 4.1 传感器压力来源
# =========================================================

# "case_h5"：从modal_surface_dataset_mix中的工况HDF5读取
# "sbd"：由SBD经过数据集构建程序转换后读取
# "star_csv"：从STAR-CCM+导出的CSV读取
MEASUREMENT_SOURCE = "sbd"

# 测试工况HDF5路径。
# 如果设置为None，则从CASES_CSV_PATH中根据TEST_CASE_ID自动查找。
TEST_CASE_H5_PATH = None

# =========================================================
# 5. 传感器设置
# =========================================================

SENSOR_METHOD = "deim"  # "deim" 或 "qr" 或 "custom"

N_SENSORS = 80  # 传感器数量
EXPECTED_POD_MODES = 20
EXPECTED_SENSOR_COUNT = 80
QR_SENSOR_COUNT = 80
DEIM_SENSOR_COUNT = 20
# 如果是custom，提供传感器坐标文件
CUSTOM_SENSOR_XYZ_CSV = (
    Path(__file__).parent
    / "custom_sensor_xyz.csv"
)

# =========================================================
# 6. 稀疏系数求解
# =========================================================

COEFFICIENT_SOLVER = "ridge"  # "lstsq" 或 "ridge"
RIDGE_LAMBDA = 1.0e-6

# 噪声测试（可选）
ADD_MEASUREMENT_NOISE = False
MEASUREMENT_NOISE_STD = 0.0

# =========================================================
# 7. 输出路径
# =========================================================

SPARSE_ROOT = (
    MODAL_ANALYSIS_ROOT
    / "sparse_pressure"
)

SENSOR_ROOT = SPARSE_ROOT / "sensors"
ROM_ROOT = SPARSE_ROOT / "rom"
PREDICTION_ROOT = SPARSE_ROOT / "predictions"
TRUTH_ROOT = SPARSE_ROOT / "truth"
COMPARISON_ROOT = SPARSE_ROOT / "comparison"

# 自动创建输出目录
for path in [
    SPARSE_ROOT,
    SENSOR_ROOT,
    ROM_ROOT,
    PREDICTION_ROOT,
    TRUTH_ROOT,
    COMPARISON_ROOT,
]:
    path.mkdir(parents=True, exist_ok=True)

# =========================================================
# 8. 三向流场专用配置
# =========================================================

# 速度空间分析
VELOCITY_SPACE_ANALYSIS = True

# 速度归一化（用于显示）
NORMALIZE_VELOCITY_DISPLAY = True

# 打印诊断信息
print(f"\n{'='*80}")
print(f"三向流场稀疏压力重构配置")
print(f"{'='*80}")
print(f"测试工况: {TEST_CASE_ID}")
print(f"速度分量: vx={TEST_U_INF:.2f}, vy={TEST_V_INF:.2f}, vz={TEST_W_INF:.2f} m/s")
print(f"速度模长: |V|={TEST_SPEED_MAGNITUDE:.4f} m/s")
print(f"动压: q_inf={0.5*RHO*TEST_SPEED_MAGNITUDE**2:.2f} Pa")
print(f"目标部件: {PART_NAME}")
print(f"传感器方法: {SENSOR_METHOD.upper()}")
print(f"传感器数量: {N_SENSORS}")
print(f"{'='*80}\n")