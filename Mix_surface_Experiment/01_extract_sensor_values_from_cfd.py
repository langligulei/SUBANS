# -*- coding: utf-8 -*-
r"""
01_extract_sensor_values_from_cfd.py  (v2)

从 STAR-CCM+ 导出的 HDF5-CGNS 中批量提取工程测压孔压力。

v2 相对 v1 的关键修复
--------------------
1) 只扫描HDF5元信息（obj.shape / obj.dtype），字段延迟加载。
   v1 用 visititems + obj[()] 把所有数据集读进内存（含600MB的
   ElementConnectivity 和40+个40MB的Solution字段，峰值4~6GB），
   导致 MemoryError。

2) 直接使用 Solution 组里已有的 CentroidX/Y/Z 作为采样点。
   v1 试图从 ElementConnectivity 重建 4,960,604 个单元几何，
   会生成496万个Python对象（约1.4GB）并耗时数分钟。v2 完全不读
   ElementConnectivity。

3) 用 GridLocation 判断数据在 Vertex 还是 CellCenter，自动切换。

4) 修复 sniff_magic 未定义（NameError）。

5) 自动比对"文件夹名"与"CGNS文件名"解析出的速度，发现不一致即报警。

6) 从远场速度反推真实来流速度，用于交叉验证工况标签。

用法
----
# 全量提取（52个工况）
python 01_extract_sensor_values_from_cfd.py

# 只跑一个工况，看详细诊断
python 01_extract_sensor_values_from_cfd.py --case u-0p5v-0p5w-0p5

# 只检查工况清单，不读CGNS
python 01_extract_sensor_values_from_cfd.py --list-only
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# =============================================================================
# 1. 配置
# =============================================================================

DEFAULT_SURFACE_CHECK_ROOT = Path(
    r"E:\suboff（0-200）\suboff_0000\SurfaceCheck"
)

DEFAULT_SENSOR_CSV = Path(
    r"E:\suboff（0-200）\suboff_0000"
    r"\modal_surface_dataset_mix"
    r"\modal_analysis\cp_static"
    r"\all_surface_practical_sensors.csv"
)

DEFAULT_CASE_LIST_XLSX = Path(
    r"E:\suboff（0-200）\suboff_0000\SurfaceCheck"
    r"\三向速度工况.xlsx"
)

DEFAULT_TRAIN_CASES_CSV = Path(
    r"E:\suboff（0-200）\suboff_0000"
    r"\modal_surface_dataset_mix\cases.csv"
)

DEFAULT_OUTPUT_ROOT = Path(
    r"E:\suboff（0-200）\suboff_0000"
    r"\modal_surface_dataset_mix\sensor_check"
)

# 流体物性，必须与 00_experiment_config.py 完全一致
RHO = 997.561

# CP计算用的参考压力，必须与 00_experiment_config.py 一致
P_STATIC_REFERENCE_PA = 101325.0

# CGNS里 Pressure 字段是绝对压还是表压
#   "auto"     : 自动判断（中位数 < 50000 Pa 视为表压）
#   "absolute" : 直接当绝对压
#   "gauge"    : 当表压，加 GAUGE_TO_ABSOLUTE_OFFSET_PA
PRESSURE_KIND = "auto"
GAUGE_TO_ABSOLUTE_OFFSET_PA = 101325.0

# 传感器坐标全局平移补偿（若坐标系原点不一致时改这里）
SENSOR_COORDINATE_OFFSET_M = (0.0, 0.0, 0.0)

# 采样模式
#   "nearest"  : 最近面心/节点直接取值（推荐，最忠实反映壁面压力）
#   "idw_knn"  : k近邻反距离加权（做轻度平滑）
SAMPLING_MODE = "nearest"
SAMPLING_K = 4
SAMPLING_POWER = 2.0

# 采样距离超过该值报警（m）
SAMPLING_DISTANCE_WARN_M = 5.0e-3

# 工况速度匹配容差
VELOCITY_TOLERANCE = 1.0e-6

# 是否把生成的CSV复制回各工况目录
WRITE_INTO_CASE_DIR = False

# 调试：只处理前N个工况（None 表示全部）
CASE_LIMIT = None

# =============================================================================
# 2. 名称规范化工具
# =============================================================================

def _norm(text: str) -> str:
    """去掉非字母数字并转小写，用于字段名匹配。"""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())

def _norm_part(text: str) -> str:
    """去掉空白并转小写，用于HDF5路径段匹配。"""
    return re.sub(r"\s+", "", str(text)).lower()

def _decode_h5_string(array) -> str:
    """把 CGNS 的 int8/char 数据集解码成字符串。"""
    if array is None:
        return ""

    flat = np.asarray(array).reshape(-1)

    if flat.dtype.kind in "iu":
        raw = bytes(
            np.clip(flat, 0, 255).astype(np.uint8).tolist()
        )
    elif flat.dtype.kind == "S":
        raw = flat.tobytes()
    else:
        return ""

    return (
        raw.split(b"\x00")[0]
        .decode("utf-8", errors="ignore")
        .strip()
    )

# =============================================================================
# 3. 工况文件夹名 / CGNS文件名 解析
# =============================================================================

# 注意：字符类中必须包含 '.' 以支持 u1.5v0w2 这类写法，
# 但要用 [\dpm\.]*\d 强制"末位必须是数字"，
# 否则贪婪匹配会把文件扩展名前面的 '.' 也吞进去
# （suboff_0000_u-0p5_v-1_w0p5.cgns 的 w 组会变成 "0p5."）。

_FOLDER_PATTERN = re.compile(
    r"^u(?P<u>-?[\dpm\.]*\d)"
    r"v(?P<v>-?[\dpm\.]*\d)"
    r"w(?P<w>-?[\dpm\.]*\d)",
    re.IGNORECASE,
)

_FILENAME_PATTERN = re.compile(
    r"u(?P<u>-?[\dpm\.]*\d)"
    r"[_\-\s]*v(?P<v>-?[\dpm\.]*\d)"
    r"[_\-\s]*w(?P<w>-?[\dpm\.]*\d)",
    re.IGNORECASE,
)

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d*)?|\.\d+")

def _parse_component(text: str) -> float:
    """
    '1p5' / '-1p5' / 'm1p5' / '0' / '0p5.' -> float

    最后用正则只提取连续的数字部分，容忍任何尾部残留
    （例如被正则多吞进来的 '.' 或其它扩展名碎片），
    这样即使上游正则再出偏差也不会抛 ValueError。
    """
    text = str(text).strip().lower()

    sign = 1.0

    if text.startswith("-"):
        sign = -1.0
        text = text[1:]
    elif text.startswith(("m", "n")):
        sign = -1.0
        text = text[1:]

    text = text.replace("p", ".").replace("_", ".")

    match = _NUMBER_PATTERN.search(text)

    if match is None:
        return 0.0

    return sign * float(match.group(0))

def parse_case_folder_name(folder_name: str):
    match = _FOLDER_PATTERN.match(str(folder_name).strip())

    if match is None:
        return None

    return (
        _parse_component(match.group("u")),
        _parse_component(match.group("v")),
        _parse_component(match.group("w")),
    )

def parse_velocity_from_filename(file_name: str):
    """从 suboff_0000_u-0p5_v-0p5_w-0p5.cgns 解析出速度。"""
    match = _FILENAME_PATTERN.search(str(file_name))

    if match is None:
        return None

    return (
        _parse_component(match.group("u")),
        _parse_component(match.group("v")),
        _parse_component(match.group("w")),
    )

def format_velocity_tag(u: float, v: float, w: float) -> str:
    def fmt(value: float) -> str:
        text = f"{value:.10g}"
        return text.replace("-", "m").replace(".", "p")

    return f"u{fmt(u)}v{fmt(v)}w{fmt(w)}"

def format_velocity_text(velocity) -> str:
    if velocity is None:
        return "无法解析"
    return (
        f"({velocity[0]:+.2f}, {velocity[1]:+.2f}, {velocity[2]:+.2f})"
    )

# =============================================================================
# 4. 结果文件发现
# =============================================================================

def discover_result_files(case_dir: Path):
    """
    在工况目录里找结果文件，CGNS 优先。

    .sbd 已确认是专有二进制，排在最后，仅作为兜底记录。
    """
    patterns = ["*.cgns", "*.h5", "*.hdf5", "*.sbd", "*.csv"]

    exclude_tokens = ("sensor", "practical", "measurement", "cases")

    unique = {}

    for pattern in patterns:
        for path in case_dir.glob(pattern):
            name_lower = path.name.lower()

            if path.suffix.lower() == ".csv":
                if any(token in name_lower for token in exclude_tokens):
                    continue
                if not any(
                    token in name_lower
                    for token in ("surface", "pressure", "field", "export")
                ):
                    continue

            try:
                key = str(path.resolve())
            except OSError:
                key = str(path)

            unique.setdefault(key, path)

    def rank(path: Path) -> int:
        return {
            ".cgns": 0,
            ".h5": 1,
            ".hdf5": 1,
            ".csv": 2,
            ".sbd": 3,
        }.get(path.suffix.lower(), 9)

    return sorted(unique.values(), key=lambda p: (rank(p), str(p)))

def sniff_magic(head: bytes) -> str:
    """根据文件头字节判断格式（与 00_diagnose 保持一致）。"""
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "HDF5"

    if head.startswith(b"ADF"):
        return "CGNS-ADF"

    if head[:4] == b"CGNS":
        return "CGNS-magic"

    if b"CGNS" in head[:256]:
        return "CGNS-text"

    if b"<?xml" in head[:64]:
        return "XML"

    return "UNKNOWN"

def sniff_file_format(path: Path) -> str:
    """兼容旧接口，返回小写格式名。"""
    path = Path(path)

    if not path.is_file():
        return "unreadable"

    try:
        with open(path, "rb") as handle:
            head = handle.read(256)
    except OSError:
        return "unreadable"

    return sniff_magic(head).lower()

# =============================================================================
# 5. HDF5-CGNS 读取（元信息先行 + 字段延迟加载）
# =============================================================================

def _scan_h5_metadata(handle):
    """
    只扫描元信息，不读取任何数据。

    返回
    ----
    datasets : {路径: (shape, dtype)}
    groups   : {组路径}
    """
    datasets = {}
    groups = set()

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            groups.add(name)
        elif isinstance(obj, h5py.Dataset):
            datasets[name] = (tuple(obj.shape), str(obj.dtype))

    handle.visititems(visitor)

    return datasets, groups

def _find_group_by_basename(groups, basename):
    """按组名（忽略大小写和空格）查找组。"""
    target = _norm_part(basename)

    hits = sorted(
        name
        for name in groups
        if _norm_part(name.rsplit("/", 1)[-1]) == target
    )

    return hits[0] if hits else None

def _child_dataset(datasets, group_path, field_name):
    """
    在 group_path 下查找名为 field_name 的数据集。

    CGNS/HDF5 里数据集的真实名字常带前导空格，例如
        Base/xxx/Solution00001/Pressure/ data
    所以按"去掉空白的路径段"精确匹配。
    """
    prefix = group_path + "/"
    target = _norm_part(field_name)

    for path in datasets:
        if not path.startswith(prefix):
            continue

        rest = path[len(prefix):]
        parts = [_norm_part(p) for p in rest.split("/")]

        if not parts:
            continue

        if parts[0] != target:
            continue

        # 只接受直接子级：形式为 <field>/ data 或 <field>
        if len(parts) <= 2:
            return path

    return None

def _iter_scalar_fields(datasets, group_path, n_sample):
    """
    列出 group_path 下长度恰好等于 n_sample 的数值型直接子级数据集。
    """
    prefix = group_path + "/"
    numeric_dtypes = {
        "float64", "float32",
        "int64", "int32",
        "uint8", "int8",
    }

    result = {}

    for path, (shape, dtype) in datasets.items():
        if not path.startswith(prefix):
            continue

        rest = path[len(prefix):]
        parts = [_norm_part(p) for p in rest.split("/")]

        if len(parts) > 2 or not parts:
            continue

        name = parts[0]

        if not name or name == "data":
            continue

        if dtype not in numeric_dtypes:
            continue

        size = int(np.prod(shape)) if shape else 0

        if size != n_sample:
            continue

        result[name] = path

    return result

class LazyFieldDict(dict):
    """
    键为字段名，取值时才真正从HDF5读取数据。

    这样即使CGNS里有40多个40MB的Solution字段，也只读取
    resolve_pressure_field / resolve_velocity_fields 真正用到的几个。
    """

    def __init__(self, file_path, index):
        super().__init__({name: None for name in index})
        self._file_path = str(file_path)
        self._index = dict(index)

    def __getitem__(self, key):
        value = super().__getitem__(key)

        if value is None:
            with h5py.File(self._file_path, mode="r") as handle:
                value = (
                    np.asarray(handle[self._index[key]][()])
                    .reshape(-1)
                    .astype(np.float64)
                )

            super().__setitem__(key, value)

        return value

    def get(self, key, default=None):
        if key not in self._index:
            return default

        return self[key]

def _read_coordinate_triplet(handle, datasets, group_path, prefix):
    """读取 <prefix>X / <prefix>Y / <prefix>Z 三元组。"""
    paths = {}
    values = {}

    for axis in ("X", "Y", "Z"):
        path = _child_dataset(datasets, group_path, f"{prefix}{axis}")

        if path is None:
            return None, None

        paths[axis] = path
        values[axis] = (
            np.asarray(handle[path][()])
            .reshape(-1)
            .astype(np.float64)
        )

    sizes = {len(values[axis]) for axis in ("X", "Y", "Z")}

    if len(sizes) != 1:
        raise ValueError(
            f"{prefix}X/Y/Z 长度不一致：{sizes}"
        )

    points = np.column_stack([
        values["X"],
        values["Y"],
        values["Z"],
    ])

    return points, paths

def load_cgns_hdf5(path: Path, sensor_xyz: np.ndarray = None):
    """
    读取 STAR-CCM+ 的 HDF5-CGNS。

    只读取采样点坐标和少量标量场，绝不读取 ElementConnectivity。

    返回 dict
    --------
    points          : (n,3) 采样点坐标
    point_location  : "Vertex" / "CellCenter"
    point_source    : "CoordinateX/Y/Z" / "CentroidX/Y/Z" / "PositionX/Y/Z"
    node_fields     : LazyFieldDict（Vertex数据时非空）
    cell_fields     : LazyFieldDict（CellCenter数据时非空）
    meta            : 诊断信息
    """
    with h5py.File(str(path), mode="r") as handle:
        datasets, groups = _scan_h5_metadata(handle)

        if not datasets:
            raise ValueError("HDF5中没有任何数据集")

        # ---- 定位网格坐标组 ----
        grid_group = _find_group_by_basename(groups, "GridCoordinates")

        if grid_group is None:
            raise ValueError(
                "未找到 GridCoordinates 组，"
                f"顶层组：{sorted(groups)[:10]}"
            )

        zone_group = grid_group.rsplit("/", 1)[0]

        node_points, _ = _read_coordinate_triplet(
            handle, datasets, grid_group, "Coordinate"
        )

        if node_points is None:
            raise ValueError("缺少 CoordinateX/Y/Z")

        n_vertex = len(node_points)

        # ---- 定位 Solution 组 ----
        zone_keys = list(handle[zone_group].keys())

        solution_candidates = sorted(
            key
            for key in zone_keys
            if key.lower().startswith("solution")
            and isinstance(handle[zone_group][key], h5py.Group)
        )

        if not solution_candidates:
            raise ValueError(
                f"{zone_group} 下未找到 Solution 组"
            )

        solution_group = f"{zone_group}/{solution_candidates[0]}"
        solution_name = solution_candidates[0]

        # ---- GridLocation ----
        location_path = _child_dataset(
            datasets, solution_group, "GridLocation"
        )

        grid_location = ""

        if location_path is not None:
            grid_location = _decode_h5_string(
                handle[location_path][()]
            )

        # ---- 确定采样点 ----
        meta = {
            "zone_group": zone_group,
            "solution_group": solution_group,
            "solution_name": solution_name,
            "grid_location": grid_location,
            "n_vertex": n_vertex,
        }

        candidate_sets = []

        for prefix in ("Centroid", "Position"):
            points, _ = _read_coordinate_triplet(
                handle, datasets, solution_group, prefix
            )

            if points is not None:
                candidate_sets.append((f"{prefix}X/Y/Z", points))

        is_cell_centered = (
            "cell" in grid_location.lower()
            or not candidate_sets == []
            and n_vertex != len(candidate_sets[0][1])
        )

        # 若Solution字段长度等于节点数，则数据在节点上
        first_field = _child_dataset(
            datasets, solution_group, "Pressure"
        )

        n_field = None

        if first_field is not None:
            shape, _ = datasets[first_field]
            n_field = int(np.prod(shape)) if shape else None

        use_vertex = (
            n_field is not None
            and n_field == n_vertex
        )

        if use_vertex or not candidate_sets:
            points = node_points
            point_source = "CoordinateX/Y/Z (Vertex)"
            point_location = "Vertex"
        else:
            # 在 Centroid 与 Position 之间选与测点更吻合的那个
            if sensor_xyz is not None and len(candidate_sets) > 1:
                best = None

                for label, candidate in candidate_sets:
                    tree = cKDTree(candidate)
                    distance, _ = tree.query(sensor_xyz, k=1)
                    score = float(np.median(distance))

                    if best is None or score < best[1]:
                        best = (label, score, candidate)

                point_source = best[0]
                points = best[2]
                meta["centroid_choice_score_m"] = best[1]
                meta["centroid_candidates"] = [
                    label for label, _ in candidate_sets
                ]
            else:
                point_source = candidate_sets[0][0]
                points = candidate_sets[0][1]

            point_location = "CellCenter"

        n_sample = len(points)

        meta["point_location"] = point_location
        meta["point_source"] = point_source
        meta["n_sample"] = n_sample
        meta["n_field"] = n_field

        # ---- 字段索引（不读数据）----
        field_index = _iter_scalar_fields(
            datasets, solution_group, n_sample
        )

        meta["available_fields"] = sorted(field_index)

        # 节点场：Solution里长度等于节点数的字段（一般没有）
        vertex_index = {}
        cell_index = {}

        if point_location == "Vertex":
            vertex_index = field_index
        else:
            cell_index = field_index

        # ---- 网格包围盒 ----
        meta["node_bbox"] = {
            "x": [float(node_points[:, 0].min()), float(node_points[:, 0].max())],
            "y": [float(node_points[:, 1].min()), float(node_points[:, 1].max())],
            "z": [float(node_points[:, 2].min()), float(node_points[:, 2].max())],
        }

        # ---- 判别表面网格 vs 体积网格 ----
        # 取y、z方向各收缩25%的内核区域：
        # 体积网格在内核里密布采样点，表面网格在内核里几乎为空。
        y_mid = 0.5 * (points[:, 1].min() + points[:, 1].max())
        z_mid = 0.5 * (points[:, 2].min() + points[:, 2].max())
        y_half = 0.25 * (points[:, 1].max() - points[:, 1].min())
        z_half = 0.25 * (points[:, 2].max() - points[:, 2].min())

        interior = (
                (np.abs(points[:, 1] - y_mid) < y_half)
                & (np.abs(points[:, 2] - z_mid) < z_half)
        )

        meta["interior_fraction"] = float(interior.mean())

        meta["sample_bbox"] = {
            "x": [float(points[:, 0].min()), float(points[:, 0].max())],
            "y": [float(points[:, 1].min()), float(points[:, 1].max())],
            "z": [float(points[:, 2].min()), float(points[:, 2].max())],
        }

    return {
        "points": points,
        "point_location": point_location,
        "point_source": point_source,
        "node_fields": LazyFieldDict(path, vertex_index),
        "cell_fields": LazyFieldDict(path, cell_index),
        "meta": meta,
        "reader": "h5py-lazy",
    }

def _load_from_surface_csv(path: Path):
    """读取 STAR-CCM+ 整面导出的 CSV。"""
    table = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    def find_column(candidates):
        normalized = {
            _norm(name): name
            for name in table.columns
        }

        for candidate in candidates:
            hit = normalized.get(_norm(candidate))

            if hit is not None:
                return hit

        return None

    x_col = find_column(["X (m)", "X", "x_m", "CoordinateX"])
    y_col = find_column(["Y (m)", "Y", "y_m", "CoordinateY"])
    z_col = find_column(["Z (m)", "Z", "z_m", "CoordinateZ"])

    if None in (x_col, y_col, z_col):
        raise KeyError(
            f"{path.name} 缺少坐标列。现有列："
            f"{list(table.columns)[:20]}"
        )

    pressure_col = None

    for candidate in PRESSURE_PRIORITY:
        hit = find_column([candidate])

        if hit is not None:
            pressure_col = hit
            break

    if pressure_col is None:
        raise KeyError(
            f"{path.name} 缺少压力列。现有列："
            f"{list(table.columns)[:20]}"
        )

    points = table[[x_col, y_col, z_col]].to_numpy(dtype=np.float64)

    return {
        "points": points,
        "point_location": "Vertex",
        "point_source": "CSV",
        "node_fields": {
            pressure_col: table[pressure_col].to_numpy(dtype=np.float64)
        },
        "cell_fields": {},
        "meta": {
            "grid_location": "CSV",
            "point_location": "Vertex",
            "point_source": "CSV",
            "n_sample": len(points),
        },
        "reader": "surface-csv",
    }

def load_surface_data(path: Path, sensor_xyz=None):
    """多级回退读取。"""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        loaders = [_load_from_surface_csv]
    else:
        loaders = [load_cgns_hdf5]

    errors = []

    for loader in loaders:
        try:
            if loader is load_cgns_hdf5:
                result = loader(path, sensor_xyz=sensor_xyz)
            else:
                result = loader(path)

            if result is None:
                errors.append(f"{loader.__name__}: 未启用")
                continue

            if len(result["points"]) == 0:
                raise ValueError("采样点数为0")

            return result

        except Exception as exc:
            errors.append(
                f"{loader.__name__}: {type(exc).__name__}: {exc}"
            )

    if path.is_file():
        try:
            with open(path, "rb") as handle:
                format_name = sniff_magic(handle.read(256))
        except OSError:
            format_name = "unreadable"
    else:
        format_name = "unreadable"

    hint = ""

    if format_name == "HDF5":
        hint = (
            "文件是HDF5，但结构不符合预期。\n"
            "  请运行 00a_diagnose_cfd_file.py --tree 查看完整数据集树。"
        )
    elif format_name == "CGNS-ADF":
        hint = "ADF版CGNS，需要 cgnslib。建议改导出HDF5版。"
    elif format_name == "UNKNOWN":
        hint = (
            "STAR-CCM+专有二进制，Python无法解析。\n"
            "  请改导出 HDF5-CGNS 或整面CSV。"
        )

    raise RuntimeError(
        f"无法读取结果文件：{path}\n"
        f"嗅探格式：{format_name}\n"
        f"尝试记录：\n  " + "\n  ".join(errors) + "\n"
        f"处理建议：{hint}"
    )

# =============================================================================
# 6. 字段解析
# =============================================================================

PRESSURE_PRIORITY = [
    "Pressure",
    "Absolute Pressure",
    "AbsolutePressure",
    "Static Pressure",
    "StaticPressure",
    "Pressure (Pa)",
]

TOTAL_PRESSURE_PRIORITY = [
    "AbsoluteTotalPressure",
    "Absolute Total Pressure",
    "Total Pressure",
    "TotalPressure",
]

CP_REFERENCE_PRIORITY = [
    "PressureCoefficient",
    "Pressure Coefficient",
    "Cp",
]

VELOCITY_COMPONENT_KEYS = {
    "u": [
        "VelocityX", "VelocityX (m/s)", "Velocity[i] (m/s)", "Velocity[i]",
        "Velocity_0", "RelativeVelocityX", "RelativeVelocity_0",
    ],
    "v": [
        "VelocityY", "VelocityY (m/s)", "Velocity[j] (m/s)", "Velocity[j]",
        "Velocity_1", "RelativeVelocityY", "RelativeVelocity_1",
    ],
    "w": [
        "VelocityZ", "VelocityZ (m/s)", "Velocity[k] (m/s)", "Velocity[k]",
        "Velocity_2", "RelativeVelocityZ", "RelativeVelocity_2",
    ],
}

def pick_field(fields, priority, contains=None):
    """按优先级挑字段，返回 (字段名, 数组) 或 (None, None)。

    注意：fields 可能是 LazyFieldDict，取值时才真正读盘。
    """
    if not fields and not len(fields):
        return None, None

    normalized = {
        _norm(key): key
        for key in fields.keys()
    }

    for candidate in priority:
        hit = normalized.get(_norm(candidate))

        if hit is not None:
            return hit, fields[hit]

    if contains:
        for key in fields.keys():
            key_norm = _norm(key)

            if any(_norm(token) in key_norm for token in contains):
                return key, fields[key]

    return None, None

def resolve_pressure_field(data):
    for location in ("cell_fields", "node_fields"):
        fields = data.get(location, {})

        name, array = pick_field(fields, PRESSURE_PRIORITY)

        if array is not None:
            return location, name, array

    raise KeyError(
        "CGNS中未找到压力场。可用字段："
        f"{data['meta'].get('available_fields', [])[:30]}"
    )

def resolve_total_pressure_field(data):
    for location in ("cell_fields", "node_fields"):
        name, array = pick_field(
            data.get(location, {}), TOTAL_PRESSURE_PRIORITY
        )

        if array is not None:
            return location, name, array

    return None, None, None

def resolve_cp_reference_field(data):
    for location in ("cell_fields", "node_fields"):
        name, array = pick_field(
            data.get(location, {}), CP_REFERENCE_PRIORITY
        )

        if array is not None:
            return location, name, array

    return None, None, None

def resolve_velocity_fields(data):
    for location in ("cell_fields", "node_fields"):
        fields = data.get(location, {})

        resolved = {}

        for axis, candidates in VELOCITY_COMPONENT_KEYS.items():
            name, array = pick_field(fields, candidates)

            if array is None:
                resolved = None
                break

            resolved[axis] = array

        if resolved:
            return location, resolved

    return None, None

# =============================================================================
# 7. 采样
# =============================================================================

def sample_scalar(
    source_points,
    source_values,
    query_points,
    mode: str,
    k: int,
    power: float,
):
    source_points = np.asarray(source_points, dtype=np.float64)
    source_values = np.asarray(source_values, dtype=np.float64).reshape(-1)
    query_points = np.asarray(query_points, dtype=np.float64)

    if len(source_points) != len(source_values):
        raise ValueError(
            "采样点与场长度不一致："
            f"{len(source_points)} vs {len(source_values)}"
        )

    tree = cKDTree(source_points)

    if mode == "nearest" or k <= 1:
        distance, index = tree.query(query_points, k=1)

        return (
            np.asarray(source_values[index], dtype=np.float64),
            np.asarray(distance, dtype=np.float64),
        )

    k = int(min(max(k, 1), len(source_points)))

    distance, index = tree.query(query_points, k=k)

    if k == 1:
        distance = distance[:, None]
        index = index[:, None]

    nearest = distance[:, 0].copy()

    exact = distance[:, 0] <= 1.0e-12

    safe_distance = np.maximum(distance, 1.0e-12)

    weights = 1.0 / safe_distance ** float(power)

    values = np.sum(
        weights * source_values[index], axis=1
    ) / np.sum(weights, axis=1)

    values[exact] = source_values[index[exact, 0]]

    return values, nearest

def estimate_freestream(points, velocity, center=None):
    """
    用离域中心最远的那一小部分点估计来流速度。

    物面附近速度受船体影响，远场速度≈来流，可用于交叉验证工况标签。
    """
    points = np.asarray(points, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)

    if center is None:
        center = points.mean(axis=0)

    distance = np.linalg.norm(points - center[None, :], axis=1)

    k = max(int(0.001 * len(distance)), 32)

    index = np.argpartition(distance, -k)[-k:]

    return (
        np.median(velocity[index], axis=0),
        float(distance[index].min()),
        float(distance[index].max()),
    )

# =============================================================================
# 8. STAR 兼容输出
# =============================================================================

STAR_SENSOR_COLUMNS = [
    "Absolute Pressure (Pa)",
    "Absolute Total Pressure (Pa)",
    "Velocity: Magnitude (m/s)",
    "Velocity[i] (m/s)",
    "Velocity[j] (m/s)",
    "Velocity[k] (m/s)",
    "Vorticity: Magnitude (/s)",
    "Vorticity[i] (/s)",
    "Vorticity[j] (/s)",
    "Vorticity[k] (/s)",
    "X (m)",
    "Y (m)",
    "Z (m)",
]

def build_star_table(
    sensor_xyz,
    static_pressure,
    total_pressure,
    velocity_xyz,
    cp_starccm=None,
):
    velocity_xyz = np.asarray(velocity_xyz, dtype=np.float64)

    speed = np.linalg.norm(velocity_xyz, axis=1)

    table = pd.DataFrame({
        "Absolute Pressure (Pa)": static_pressure,
        "Absolute Total Pressure (Pa)": total_pressure,
        "Velocity: Magnitude (m/s)": speed,
        "Velocity[i] (m/s)": velocity_xyz[:, 0],
        "Velocity[j] (m/s)": velocity_xyz[:, 1],
        "Velocity[k] (m/s)": velocity_xyz[:, 2],
        "Vorticity: Magnitude (/s)": 0.0,
        "Vorticity[i] (/s)": 0.0,
        "Vorticity[j] (/s)": 0.0,
        "Vorticity[k] (/s)": 0.0,
        "X (m)": sensor_xyz[:, 0],
        "Y (m)": sensor_xyz[:, 1],
        "Z (m)": sensor_xyz[:, 2],
    })

    if cp_starccm is not None:
        table["Cp_STARCCM"] = cp_starccm

    return table

# =============================================================================
# 9. 单工况处理
# =============================================================================
def boundary_face_check(points, pressure_field, velocity_xyz, nominal):
    """
    包围盒6个面附近的压力与速度中位数。

    定常外流的正常情形
    ------------------
    * 6 个面的 |V| 都应等于标称值（偏差 < 5%）
    * 6 个面的 p 都应 ≈ 0（极差 < 0.5*q）

    返回
    ----
    dict: {face_name: {"n": , "p_median": , "speed_median": }}
    以及汇总指标
    """
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = high - low

    speed = np.linalg.norm(velocity_xyz, axis=1)
    nominal_speed = float(np.linalg.norm(nominal))

    faces = {}

    for axis_index, axis_name in enumerate("xyz"):
        for side, extent in (("min", low[axis_index]),
                             ("max", high[axis_index])):
            tolerance = 0.02 * span[axis_index]

            if side == "min":
                mask = points[:, axis_index] <= extent + tolerance
            else:
                mask = points[:, axis_index] >= extent - tolerance

            if not mask.any():
                continue

            faces[f"{axis_name}_{side}"] = {
                "n": int(mask.sum()),
                "p_median": float(np.median(pressure_field[mask])),
                "speed_median": float(np.median(speed[mask])),
            }

    if not faces:
        return faces, {"far_field_ok": False, "speed_rel_spread": np.nan,
                       "speed_rel_error": np.nan, "p_spread": np.nan}

    speeds = np.array([f["speed_median"] for f in faces.values()])
    pressures = np.array([f["p_median"] for f in faces.values()])

    speed_rel_spread = float((speeds.max() - speeds.min()) / max(nominal_speed, 1e-9))
    speed_rel_error = float(
        abs(float(np.median(speeds)) - nominal_speed) / max(nominal_speed, 1e-9)
    )
    p_spread = float(pressures.max() - pressures.min())
    p_level = float(np.median(pressures))

    q_inf = 0.5 * RHO * nominal_speed ** 2

    summary = {
        "speed_rel_spread": speed_rel_spread,
        "speed_rel_error": speed_rel_error,
        "p_spread": p_spread,
        "p_level": p_level,
        "p_level_over_q": p_level / max(q_inf, 1e-9),
        "far_field_ok": bool(
            speed_rel_spread < 0.10
            and speed_rel_error < 0.05
            and p_spread < 0.5 * max(q_inf, 1e-9)
        ),
    }

    return faces, summary

def process_single_case(
    case_dir: Path,
    sensor_table: pd.DataFrame,
    velocity,
    output_dir: Path,
    verbose: bool = True,
):
    u, v, w = velocity

    files = discover_result_files(case_dir)

    if not files:
        print(f"  [跳过] 目录内没有结果文件")
        return None, None

    sensor_xyz = sensor_table[
        ["x_m", "y_m", "z_m"]
    ].to_numpy(dtype=np.float64)

    sensor_xyz = sensor_xyz + np.asarray(
        SENSOR_COORDINATE_OFFSET_M, dtype=np.float64
    )[None, :]

    data = None
    used_path = None
    last_error = None

    started = time.time()

    for candidate in files:
        try:
            data = load_surface_data(candidate, sensor_xyz=sensor_xyz)
            used_path = candidate
            break
        except Exception as exc:
            last_error = exc
            print(f"  [尝试失败] {candidate.name}")
            for line in str(exc).splitlines():
                print(f"            {line}")

    if data is None:
        print(f"  [失败] 所有结果文件均无法读取")
        return None, None

    elapsed = time.time() - started
    meta = data["meta"]

    if verbose:
        print(
            f"  读取：{used_path.name} "
            f"[{data['reader']}] {elapsed:.1f}s"
        )
        print(
            f"  GridLocation={meta.get('grid_location')!r}  "
            f"采样点={meta['point_location']} "
            f"({meta.get('point_source')})"
        )
        print(
            f"  节点={meta.get('n_vertex', 0):,}  "
            f"采样点数={meta['n_sample']:,}  "
            f"字段数={len(meta.get('available_fields', []))}"
        )
        print(
            f"  测点包围盒 x={meta['sample_bbox']['x']} "
            f"y={meta['sample_bbox']['y']} "
            f"z={meta['sample_bbox']['z']}"
        )

    # ---- 压力 ----
    (
        pressure_location,
        pressure_name,
        pressure_values,
    ) = resolve_pressure_field(data)

    raw_min = float(np.nanmin(pressure_values))
    raw_max = float(np.nanmax(pressure_values))
    raw_median = float(np.median(pressure_values))

    if PRESSURE_KIND == "absolute":
        pressure_offset = 0.0
        pressure_kind_used = "absolute"
    elif PRESSURE_KIND == "gauge":
        pressure_offset = GAUGE_TO_ABSOLUTE_OFFSET_PA
        pressure_kind_used = "gauge"
    else:
        if raw_median < 0.5 * GAUGE_TO_ABSOLUTE_OFFSET_PA:
            pressure_offset = GAUGE_TO_ABSOLUTE_OFFSET_PA
            pressure_kind_used = "gauge(auto)"
        else:
            pressure_offset = 0.0
            pressure_kind_used = "absolute(auto)"

    pressure_raw = pressure_values
    pressure_values = pressure_values + pressure_offset

    if verbose:
        print(
            f"  压力字段：{pressure_name} ({pressure_location}) "
            f"原始min={raw_min:.2f} max={raw_max:.2f} "
            f"中位={raw_median:.2f} Pa"
        )
        percentiles = np.percentile(
            pressure_raw, [0.1, 1.0, 50.0, 99.0, 99.9]
        )

        print(
            f"  压力分位数 Pa: "
            f"0.1%={percentiles[0]:.1f}  1%={percentiles[1]:.1f}  "
            f"50%={percentiles[2]:.1f}  99%={percentiles[3]:.1f}  "
            f"99.9%={percentiles[4]:.1f}"
        )
        print(f"  压力基准判定：{pressure_kind_used}")

    # ---- 采样点 ----
    sample_points = data["points"]

    # ---- 速度 ----
    velocity_location, velocity_components = resolve_velocity_fields(data)

    if velocity_components is not None:
        velocity_xyz = np.column_stack([
            sample_scalar(
                sample_points,
                velocity_components[axis],
                sensor_xyz,
                SAMPLING_MODE,
                SAMPLING_K,
                SAMPLING_POWER,
            )[0]
            for axis in ("u", "v", "w")
        ])

        detected_freestream, d_near, d_far = estimate_freestream(
            sample_points,
            np.column_stack([
                velocity_components["u"],
                velocity_components["v"],
                velocity_components["w"],
            ]),
        )
    else:
        velocity_xyz = np.tile(
            np.asarray([u, v, w], dtype=np.float64),
            (len(sensor_xyz), 1),
        )
        detected_freestream = np.asarray([u, v, w], dtype=np.float64)
        d_near = d_far = np.nan
        if verbose:
            freestream_ok = bool(
                np.allclose(
                    np.asarray(velocity, dtype=np.float64),
                    detected_freestream,
                    atol=1.0e-3,
                )
            )

            print(
                f"  远场速度(反推)=({detected_freestream[0]:+.4f}, "
                f"{detected_freestream[1]:+.4f}, "
                f"{detected_freestream[2]:+.4f})  "
                f"文件夹名=({u:+.4f}, {v:+.4f}, {w:+.4f})  "
                f"{'一致' if freestream_ok else '★不一致★'}"
            )

    # ---- 压力采样 ----
    static_pressure, nearest_distance = sample_scalar(
        sample_points,
        pressure_values,
        sensor_xyz,
        SAMPLING_MODE,
        SAMPLING_K,
        SAMPLING_POWER,
    )

    # ---- 总压 ----
    _, _, total_values = resolve_total_pressure_field(data)

    if total_values is not None:
        total_pressure = sample_scalar(
            sample_points,
            total_values + pressure_offset,
            sensor_xyz,
            SAMPLING_MODE,
            SAMPLING_K,
            SAMPLING_POWER,
        )[0]
    else:
        total_pressure = static_pressure + 0.5 * RHO * np.sum(
            velocity_xyz ** 2, axis=1
        )

    # ---- Cp ----
    speed_inf = float(np.linalg.norm([u, v, w]))
    q_inf = 0.5 * RHO * speed_inf ** 2

    if q_inf <= 0.0:
        cp = np.full(len(sensor_xyz), np.nan)
    else:
        cp = (static_pressure - P_STATIC_REFERENCE_PA) / q_inf

    # ---- STAR-CCM+ 自带 Cp：反推它的参考动压 q_ref ----
    #
    # STAR-CCM+ 定义 Cp = (p - p_ref) / (0.5 * rho_ref * |V_ref|^2)。
    # 若 ReferenceValues 未设置，默认 rho_ref=1.0, V_ref=1.0, p_ref=0，
    # 此时 Cp = p_gauge / 0.5，与物理Cp差几个数量级。
    # 这里直接从数据反推 q_ref = p_gauge / Cp，用来判断参考值是否设对。
    _, cp_ref_name, cp_ref_values = resolve_cp_reference_field(data)

    cp_starccm = None
    cp_ref_q_inferred = np.nan
    cp_ref_q_spread = np.nan

    if cp_ref_values is not None:
        cp_starccm = sample_scalar(
            sample_points,
            cp_ref_values,
            sensor_xyz,
            SAMPLING_MODE,
            SAMPLING_K,
            SAMPLING_POWER,
        )[0]

        p_gauge_sensor = static_pressure - pressure_offset

        valid = (
                np.isfinite(cp_starccm)
                & (np.abs(cp_starccm) > 1.0e-9)
        )

        if valid.any():
            ratio = p_gauge_sensor[valid] / cp_starccm[valid]
            cp_ref_q_inferred = float(np.median(ratio))
            cp_ref_q_spread = float(
                np.std(ratio) / max(abs(cp_ref_q_inferred), 1.0e-12)
            )

    if verbose and np.isfinite(cp_ref_q_inferred):
        print(
            f"  STAR-CCM+ Cp 反推 q_ref={cp_ref_q_inferred:.6g} Pa  "
            f"(物理 q_inf={q_inf:.6g} Pa, 相对离散={cp_ref_q_spread:.3%})"
        )

    # ---- 输出 ----
    case_tag = format_velocity_tag(u, v, w)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / f"PracticalSensorFromStar_{case_tag}.csv"
    )

    star_table = build_star_table(
        sensor_xyz=sensor_xyz
        - np.asarray(SENSOR_COORDINATE_OFFSET_M, dtype=np.float64)[None, :],
        static_pressure=static_pressure,
        total_pressure=total_pressure,
        velocity_xyz=velocity_xyz,
        cp_starccm=cp_starccm,
    )

    star_table.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    if WRITE_INTO_CASE_DIR:
        star_table.to_csv(
            case_dir / f"PracticalSensorFromStar_{case_tag}.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.10g",
        )

    # ---- 文件夹名 vs 文件名 ----
    filename_velocity = parse_velocity_from_filename(used_path.name)

    folder_filename_match = None

    if filename_velocity is not None:
        folder_filename_match = bool(
            np.allclose(
                np.asarray(velocity),
                np.asarray(filename_velocity),
                atol=VELOCITY_TOLERANCE,
            )
        )

    freestream_match = None

    if np.isfinite(detected_freestream).all():
        freestream_match = bool(
            np.allclose(
                np.asarray(velocity),
                detected_freestream,
                atol=1.0e-3,
            )
        )

    report = {
        "case_folder": case_dir.name,
        "case_tag": case_tag,
        "u_inf_m_s": u,
        "v_inf_m_s": v,
        "w_inf_m_s": w,
        "speed_m_s": speed_inf,
        "result_file": str(used_path),
        "result_format": sniff_file_format(used_path),
        "reader": data["reader"],
        "grid_location": meta.get("grid_location", ""),
        "point_location": meta.get("point_location", ""),
        "point_source": meta.get("point_source", ""),
        "n_vertex": meta.get("n_vertex", 0),
        "n_sample": meta.get("n_sample", 0),
        "n_field": meta.get("n_field", 0),
        "available_fields": ";".join(
            meta.get("available_fields", [])[:40]
        ),
        "pressure_field": pressure_name,
        "pressure_location": pressure_location,
        "pressure_raw_min_pa": raw_min,
        "pressure_raw_max_pa": raw_max,
        "pressure_raw_median_pa": raw_median,
        "pressure_kind_used": pressure_kind_used,
        "velocity_source": (
            velocity_location
            if velocity_components is not None
            else "freestream"
        ),
        "detected_freestream_u": float(detected_freestream[0]),
        "detected_freestream_v": float(detected_freestream[1]),
        "detected_freestream_w": float(detected_freestream[2]),
        "freestream_matches_folder": freestream_match,
        "filename_velocity_u": (
            filename_velocity[0] if filename_velocity else np.nan
        ),
        "filename_velocity_v": (
            filename_velocity[1] if filename_velocity else np.nan
        ),
        "filename_velocity_w": (
            filename_velocity[2] if filename_velocity else np.nan
        ),
        "folder_matches_filename": folder_filename_match,
        "sensor_count": len(sensor_xyz),
        "max_sampling_distance_m": float(nearest_distance.max()),
        "median_sampling_distance_m": float(np.median(nearest_distance)),
        "p95_sampling_distance_m": float(np.percentile(nearest_distance, 95)),
        "cp_min": float(np.nanmin(cp)),
        "cp_max": float(np.nanmax(cp)),
        "cp_mean": float(np.nanmean(cp)),
        "cp_starccm_field": cp_ref_name or "",
        "q_inf_pa": q_inf,
        "starccm_cp_ref_q_inferred_pa": cp_ref_q_inferred,
        "starccm_cp_ref_q_spread": cp_ref_q_spread,
        "sample_bbox_x": (
            f"{meta['sample_bbox']['x'][0]:.4f}"
            f"~{meta['sample_bbox']['x'][1]:.4f}"
        ),
        "sample_bbox_y": (
            f"{meta['sample_bbox']['y'][0]:.4f}"
            f"~{meta['sample_bbox']['y'][1]:.4f}"
        ),
        "sample_bbox_z": (
            f"{meta['sample_bbox']['z'][0]:.4f}"
            f"~{meta['sample_bbox']['z'][1]:.4f}"
        ),
        "runtime_s": elapsed,
        "output_csv": str(output_path),
    }

    warnings = []

    if report["max_sampling_distance_m"] > SAMPLING_DISTANCE_WARN_M:
        warnings.append("采样距离偏大")

    if folder_filename_match is False:
        warnings.append("文件夹名与CGNS文件名速度不一致")

    if freestream_match is False:
        warnings.append("远场速度与文件夹名速度不一致")

    # 只有当 STAR-CCM+ 的参考动压与物理动压一致（说明ReferenceValues设对了），
    # 而Cp仍然对不上时，才说明是我们的问题
    if (
            np.isfinite(cp_ref_q_inferred)
            and q_inf > 0
            and abs(cp_ref_q_inferred - q_inf) / q_inf < 0.1
            and cp_ref_q_spread > 0.05
    ):
        warnings.append("Cp与STAR-CCM+不一致(参考值已设对)")

    if warnings:
        report["warning"] = "；".join(warnings)

    return report, cp

# =============================================================================
# 10. 工况清单与训练集
# =============================================================================

def load_case_list_from_excel(path: Path):
    if path is None or not Path(path).is_file():
        return None

    try:
        raw = pd.read_excel(path, header=None)
    except Exception as exc:
        print(f"[警告] 读取工况Excel失败：{exc}")
        return None

    rows = []
    current_group = ""
    current_category = ""

    for _, line in raw.iterrows():
        values = list(line.values)

        if len(values) < 6:
            continue

        if isinstance(values[0], str) and values[0].strip():
            current_group = values[0].strip()

        if isinstance(values[1], str) and values[1].strip():
            current_category = values[1].strip()

        if not isinstance(values[2], str):
            continue

        code = values[2].strip()

        if not re.fullmatch(r"[A-Z]\d{2}", code):
            continue

        try:
            u = float(values[3])
            v = float(values[4])
            w = float(values[5])
        except (TypeError, ValueError):
            continue

        rows.append({
            "group": current_group,
            "category": current_category,
            "case_code": code,
            "u_inf_m_s": u,
            "v_inf_m_s": v,
            "w_inf_m_s": w,
        })

    if not rows:
        return None

    return pd.DataFrame(rows)

def load_training_cases(path: Path):
    if path is None or not Path(path).is_file():
        print(f"[提示] 未找到训练集清单，跳过查重：{path}")
        return None

    table = pd.read_csv(path, encoding="utf-8-sig")

    required = ["u_inf_m_s", "v_inf_m_s", "w_inf_m_s"]

    if not set(required).issubset(table.columns):
        print("[提示] 训练集清单缺少速度列，跳过查重。")
        return None

    return table[required].to_numpy(dtype=np.float64)

# =============================================================================
# 11. 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从HDF5-CGNS批量提取工程测压孔压力"
    )

    parser.add_argument("--root", type=str,
                        default=str(DEFAULT_SURFACE_CHECK_ROOT))
    parser.add_argument("--sensors", type=str,
                        default=str(DEFAULT_SENSOR_CSV))
    parser.add_argument("--case-list", type=str,
                        default=str(DEFAULT_CASE_LIST_XLSX))
    parser.add_argument("--train-cases", type=str,
                        default=str(DEFAULT_TRAIN_CASES_CSV))
    parser.add_argument("--output", type=str,
                        default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--check-labels", action="store_true",
                        help="只检查文件夹名与CGNS文件名是否一致")

    args = parser.parse_args()

    root = Path(args.root).expanduser()

    if not root.is_dir():
        print(f"[错误] 根目录不存在：{root}")
        sys.exit(1)

    sensor_csv = Path(args.sensors).expanduser()

    if not sensor_csv.is_file():
        print(f"[错误] 传感器文件不存在：{sensor_csv}")
        sys.exit(1)

    sensor_table = pd.read_csv(sensor_csv, encoding="utf-8-sig")

    required = {"tap_id", "global_face_id", "x_m", "y_m", "z_m"}
    missing = required - set(sensor_table.columns)

    if missing:
        print(f"[错误] 传感器文件缺少字段：{sorted(missing)}")
        sys.exit(1)

    sensor_table = sensor_table.sort_values("tap_id").reset_index(drop=True)

    print("=" * 100)
    print("STAR-CCM+ HDF5-CGNS -> 工程测压孔压力提取  (v2)")
    print("=" * 100)
    print(f"结果根目录 ：{root}")
    print(f"测点文件   ：{sensor_csv}")
    print(f"测点数量   ：{len(sensor_table)}")
    print(f"采样模式   ：{SAMPLING_MODE} (k={SAMPLING_K})")
    print(f"压力基准   ：{PRESSURE_KIND}")
    print(f"参考压力   ：{P_STATIC_REFERENCE_PA:.1f} Pa  密度：{RHO}")
    print("=" * 100)

    # ---- 扫描工况目录 ----
    if args.case:
        case_dirs = [root / args.case]
    else:
        case_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    parsed_cases = []

    for case_dir in case_dirs:
        if not case_dir.is_dir():
            continue

        velocity = parse_case_folder_name(case_dir.name)

        if velocity is None:
            continue

        parsed_cases.append((case_dir, velocity))

    print(f"\n识别到 {len(parsed_cases)} 个速度工况文件夹。")

    if not parsed_cases:
        print("[错误] 未识别到任何工况。")
        sys.exit(1)

    # ---- 标签一致性检查 ----
    print("\n" + "-" * 100)
    print("标签一致性检查：文件夹名 vs CGNS文件名")
    print("-" * 100)

    mismatch_rows = []

    for case_dir, folder_velocity in parsed_cases:
        files = [p for p in discover_result_files(case_dir)
                 if p.suffix.lower() == ".cgns"]

        if not files:
            continue

        file_velocity = parse_velocity_from_filename(files[0].name)

        if file_velocity is None:
            continue

        consistent = bool(
            np.allclose(
                np.asarray(folder_velocity),
                np.asarray(file_velocity),
                atol=VELOCITY_TOLERANCE,
            )
        )

        if not consistent:
            mismatch_rows.append({
                "case_folder": case_dir.name,
                "folder_velocity": format_velocity_text(folder_velocity),
                "cgns_file": files[0].name,
                "filename_velocity": format_velocity_text(file_velocity),
            })

    if mismatch_rows:
        print(f"\n[警告] 发现 {len(mismatch_rows)} 处不一致：\n")

        for row in mismatch_rows:
            print(f"  文件夹 {row['case_folder']:<22s} "
                  f"{row['folder_velocity']:<24s}")
            print(f"    文件 {row['cgns_file']:<48s} "
                  f"{row['filename_velocity']}")

        print(
            "\n  脚本按【文件夹名】作为工况速度。"
            "\n  提取完成后请检查报告中的 detected_freestream_* 列："
            "\n  若远场速度与文件夹名不符而与文件名相符，说明需要交换文件夹名。"
        )
    else:
        print("\n[OK] 全部一致。")

    if args.check_labels:
        return

    if args.list_only:
        print()
        for case_dir, velocity in parsed_cases:
            print(f"  {case_dir.name:<28s} "
                  f"u={velocity[0]:+.2f} "
                  f"v={velocity[1]:+.2f} "
                  f"w={velocity[2]:+.2f}")
        return

    output_root = Path(args.output).expanduser()
    star_csv_dir = output_root / "star_sensor_csv"
    output_root.mkdir(parents=True, exist_ok=True)

    # ---- 逐工况处理 ----
    reports = []
    cp_columns = {}
    failures = []
    total_started = time.time()

    for index, (case_dir, velocity) in enumerate(parsed_cases, start=1):
        print(f"\n[{index}/{len(parsed_cases)}] {case_dir.name}  "
              f"{format_velocity_text(velocity)}")

        try:
            report, cp = process_single_case(
                case_dir=case_dir,
                sensor_table=sensor_table,
                velocity=velocity,
                output_dir=star_csv_dir,
            )
        except Exception as exc:
            print(f"  [异常] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures.append({
                "case_folder": case_dir.name,
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue

        if report is None:
            failures.append({
                "case_folder": case_dir.name,
                "reason": "no readable result file",
            })
            continue

        reports.append(report)
        cp_columns[report["case_tag"]] = cp

        if report.get("warning"):
            print(f"  [警告] {report['warning']}")

    if not reports:
        print("\n[错误] 没有任何工况被成功提取。")
        sys.exit(1)

    # ---- 汇总 ----
    report_df = pd.DataFrame(reports)

    case_list = load_case_list_from_excel(
        Path(args.case_list).expanduser()
    )

    if case_list is not None:
        codes = []

        for _, row in report_df.iterrows():
            mask = (
                np.isclose(case_list["u_inf_m_s"], row["u_inf_m_s"],
                           atol=VELOCITY_TOLERANCE)
                & np.isclose(case_list["v_inf_m_s"], row["v_inf_m_s"],
                             atol=VELOCITY_TOLERANCE)
                & np.isclose(case_list["w_inf_m_s"], row["w_inf_m_s"],
                             atol=VELOCITY_TOLERANCE)
            )

            matched = case_list.loc[mask]

            codes.append(
                "/".join(matched["case_code"].tolist()) if len(matched) else ""
            )

        report_df["case_code"] = codes

    training_velocities = load_training_cases(
        Path(args.train_cases).expanduser()
    )

    if training_velocities is not None:
        in_training = []

        for _, row in report_df.iterrows():
            target = np.asarray(
                [row["u_inf_m_s"], row["v_inf_m_s"], row["w_inf_m_s"]],
                dtype=np.float64,
            )

            mask = np.all(
                np.isclose(training_velocities, target[None, :],
                           atol=1.0e-6, rtol=0.0),
                axis=1,
            )

            in_training.append(bool(mask.any()))

        report_df["in_training_set"] = in_training
        report_df["recommended_use"] = np.where(
            report_df["in_training_set"],
            "不可用(训练集已含,数据泄漏)",
            "可用(未见工况)",
        )
    else:
        report_df["in_training_set"] = False
        report_df["recommended_use"] = "未知"

    report_path = output_root / "extraction_report.csv"
    report_df.to_csv(
        report_path, index=False, encoding="utf-8-sig", float_format="%.10g"
    )

    # ---- 长表 ----
    long_rows = []

    for _, row in report_df.iterrows():
        case_tag = row["case_tag"]

        if case_tag not in cp_columns:
            continue

        cp = cp_columns[case_tag]

        for tap_index, tap in sensor_table.iterrows():
            long_rows.append({
                "case_tag": case_tag,
                "case_code": row.get("case_code", ""),
                "u_inf_m_s": row["u_inf_m_s"],
                "v_inf_m_s": row["v_inf_m_s"],
                "w_inf_m_s": row["w_inf_m_s"],
                "speed_m_s": row["speed_m_s"],
                "in_training_set": row["in_training_set"],
                "tap_id": int(tap["tap_id"]),
                "global_face_id": int(tap["global_face_id"]),
                "x_m": float(tap["x_m"]),
                "y_m": float(tap["y_m"]),
                "z_m": float(tap["z_m"]),
                "cp_sensor": float(cp[tap_index]),
            })

    long_df = pd.DataFrame(long_rows)
    long_path = output_root / "sensor_measurements_long.csv"
    long_df.to_csv(
        long_path, index=False, encoding="utf-8-sig", float_format="%.10g"
    )

    # ---- 矩阵 ----
    matrix = pd.DataFrame(cp_columns, index=np.arange(len(sensor_table)))
    matrix.index.name = "tap_index"
    matrix = matrix.reset_index()
    matrix.insert(1, "tap_id", sensor_table["tap_id"].to_numpy())
    matrix.insert(2, "global_face_id",
                  sensor_table["global_face_id"].to_numpy())

    matrix_path = output_root / "sensor_measurements_matrix.csv"
    matrix.to_csv(
        matrix_path, index=False, encoding="utf-8-sig", float_format="%.10g"
    )

    if failures:
        pd.DataFrame(failures).to_csv(
            output_root / "failed_cases.csv",
            index=False, encoding="utf-8-sig",
        )

    if mismatch_rows:
        pd.DataFrame(mismatch_rows).to_csv(
            output_root / "label_mismatch.csv",
            index=False, encoding="utf-8-sig",
        )

    # ---- 总结 ----
    print("\n" + "=" * 100)
    print("提取完成")
    print("=" * 100)
    print(f"成功工况数：{len(report_df)} / {len(parsed_cases)}")
    print(f"总耗时：{(time.time() - total_started) / 60:.1f} 分钟")

    if failures:
        print(f"失败工况数：{len(failures)}")

    print(
        f"采样距离：max={report_df['max_sampling_distance_m'].max():.3e} m "
        f"中位={report_df['median_sampling_distance_m'].median():.3e} m"
    )
    print(
        f"Cp范围：min={report_df['cp_min'].min():.4f} "
        f"max={report_df['cp_max'].max():.4f}"
    )

    if "starccm_cp_ref_q_inferred_pa" in report_df.columns:
        valid = report_df["starccm_cp_ref_q_inferred_pa"].dropna()

        if len(valid):
            print(
                f"STAR-CCM+ Cp 反推参考动压："
                f"中位={valid.median():.4g} Pa  "
                f"（工况 q_inf 中位="
                f"{report_df['q_inf_pa'].median():.4g} Pa）"
            )

    if "folder_matches_filename" in report_df.columns:
        bad = int(
            (report_df["folder_matches_filename"] == False).sum()
        )

        if bad:
            print(f"[警告] 文件夹名与文件名不一致：{bad} 个")

    if "freestream_matches_folder" in report_df.columns:
        bad = int(
            (report_df["freestream_matches_folder"] == False).sum()
        )

        if bad:
            print(
                f"[警告] 远场速度与文件夹名速度不一致：{bad} 个"
                f"（请查看 extraction_report.csv 的 "
                f"detected_freestream_* 列）"
            )

    print("\n输出文件：")
    print(f"  STAR兼容测压CSV目录 ：{star_csv_dir}")
    print(f"  提取报告             ：{report_path}")
    print(f"  长表                 ：{long_path}")
    print(f"  工况×测点矩阵        ：{matrix_path}")

    if failures:
        print(f"  失败清单             ：{output_root / 'failed_cases.csv'}")

    if mismatch_rows:
        print(f"  标签不一致清单       ：{output_root / 'label_mismatch.csv'}")

    print("=" * 100)

if __name__ == "__main__":
    main()