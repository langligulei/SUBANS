# -*- coding: utf-8 -*-
r"""
00b_probe_pressure_level.py

诊断"压力整体偏移"。对若干工况把压力/速度沿 x/y/z 切片，
判断偏移是常数、梯度还是局部异常，并检查 Solution 组数量。

用法
----
python 00b_probe_pressure_level.py
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(r"E:\suboff（0-200）\suboff_0000\SurfaceCheck")

# (标签, 文件夹名, 标称来流)
CASES = [
    ("正常 ", "u0v0w1p5",       (0.0, 0.0, 1.5)),
    ("异常1", "u1p5v0w0",       (1.5, 0.0, 0.0)),
    ("异常2", "u-1p5v0w0",      (-1.5, 0.0, 0.0)),
    ("异常3", "u-1p5v1p5w0",    (-1.5, 1.5, 0.0)),
    ("异常4", "u0p5v0p5w-0p5",  (0.5, 0.5, -0.5)),
]

def _norm(text):
    return re.sub(r"\s+", "", str(text)).lower()

def _scan(handle):
    datasets, groups = {}, set()

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            groups.add(name)
        elif isinstance(obj, h5py.Dataset):
            datasets[name] = (tuple(obj.shape), str(obj.dtype))

    handle.visititems(visitor)
    return datasets, groups

def _child(datasets, group_path, field):
    prefix = group_path + "/"
    target = _norm(field)

    for path in datasets:
        if not path.startswith(prefix):
            continue
        parts = [_norm(p) for p in path[len(prefix):].split("/")]
        if parts and parts[0] == target and len(parts) <= 2:
            return path

    return None

def _groups_by_base(groups, base):
    target = _norm(base)
    return sorted(
        g for g in groups
        if _norm(g.rsplit("/", 1)[-1]) == target
    )

def _decode(array):
    flat = np.asarray(array).reshape(-1)
    raw = bytes(np.clip(flat, 0, 255).astype(np.uint8).tolist())
    return raw.split(b"\x00")[0].decode("utf-8", errors="ignore").strip()

def probe(tag, case_name, nominal, root=ROOT):
    folder = root / case_name
    files = sorted(folder.glob("*.cgns"))

    print("=" * 100)
    print(f"[{tag}] {case_name}   标称来流={nominal}")

    if not files:
        print("    [跳过] 目录内没有 .cgns")
        print()
        return

    path = files[0]
    print(f"    文件 = {path.name}  ({path.stat().st_size/1024/1024:.1f} MB)")

    with h5py.File(str(path), mode="r") as handle:
        datasets, groups = _scan(handle)

        grid_groups = _groups_by_base(groups, "GridCoordinates")

        if not grid_groups:
            print("    [错误] 未找到 GridCoordinates")
            print()
            return

        zone = grid_groups[0].rsplit("/", 1)[0]

        solutions = sorted(
            k for k in handle[zone].keys()
            if k.lower().startswith("solution")
            and isinstance(handle[zone][k], h5py.Group)
        )

        print(f"    zone = {zone}")
        print(f"    Solution 组数量 = {len(solutions)}  -> {solutions}")

        if len(solutions) > 1:
            print("    ★★★ 存在多个 Solution，脚本只用了第一个，"
                  "可能取到未收敛的旧解！")

        if not solutions:
            print("    [错误] 没有 Solution 组")
            print()
            return

        solution = f"{zone}/{solutions[0]}"

        location_path = _child(datasets, solution, "GridLocation")

        if location_path is not None:
            print(f"    GridLocation = {_decode(handle[location_path][()])!r}")

        # 采样点
        axis_arrays = {}

        for axis in "XYZ":
            p = _child(datasets, solution, f"Centroid{axis}")
            if p is not None:
                axis_arrays[axis] = np.asarray(handle[p][()]).reshape(-1).astype(np.float64)

        if len(axis_arrays) < 3:
            print("    [错误] 缺少 CentroidX/Y/Z")
            print()
            return

        X = np.column_stack([axis_arrays["X"], axis_arrays["Y"], axis_arrays["Z"]])

        def read(field, fallback=None):
            path_ = _child(datasets, solution, field)
            if path_ is None and fallback:
                for name in fallback:
                    path_ = _child(datasets, solution, name)
                    if path_ is not None:
                        break
            if path_ is None:
                return None
            return np.asarray(handle[path_][()]).reshape(-1).astype(np.float64)

        pressure = read("Pressure")

        u = read("VelocityX", ["RelativeVelocityX"])
        v = read("VelocityY", ["RelativeVelocityY"])
        w = read("VelocityZ", ["RelativeVelocityZ"])

        print(f"    采样点数 = {len(X):,}")
        print(
            f"    域包围盒 x=[{X[:,0].min():.4f},{X[:,0].max():.4f}] "
            f"y=[{X[:,1].min():.4f},{X[:,1].max():.4f}] "
            f"z=[{X[:,2].min():.4f},{X[:,2].max():.4f}]"
        )

        if pressure is not None:
            q = np.percentile(pressure, [0, 0.1, 1, 50, 99, 99.9, 100])
            print(
                f"\n    压力[Pa]   min={q[0]:.1f}  0.1%={q[1]:.1f}  1%={q[2]:.1f}  "
                f"50%={q[3]:.1f}  99%={q[4]:.1f}  99.9%={q[5]:.1f}  max={q[6]:.1f}"
            )

        speed = None

        if u is not None and v is not None and w is not None:
            speed = np.sqrt(u**2 + v**2 + w**2)

            q = np.percentile(speed, [0, 1, 50, 99, 100])
            print(
                f"    速度[m/s]  min={q[0]:.4f}  1%={q[1]:.4f}  "
                f"50%={q[2]:.4f}  99%={q[3]:.4f}  max={q[4]:.4f}"
            )

            center = X.mean(axis=0)
            distance = np.linalg.norm(X - center, axis=1)
            k = max(int(0.005 * len(distance)), 64)
            far = np.argpartition(distance, -k)[-k:]

            print(
                f"    远场速度(最外层0.5%) = "
                f"({np.median(u[far]):+.4f}, {np.median(v[far]):+.4f}, "
                f"{np.median(w[far]):+.4f})   标称={nominal}"
            )

        # ---- 沿三个轴切片 ----
        for axis_index, axis_name in enumerate("xyz"):
            lo = X[:, axis_index].min()
            hi = X[:, axis_index].max()
            edges = np.linspace(lo, hi, 11)

            print(f"\n    沿 {axis_name} 分 10 片（片内中位数）：")
            print(
                f"      {'区间':>21s}  {'点数':>9s}  "
                f"{'p中位[Pa]':>12s}  {'|V|中位[m/s]':>12s}"
            )

            for i in range(10):
                mask = (
                    (X[:, axis_index] >= edges[i])
                    & (X[:, axis_index] <= edges[i + 1])
                )
                count = int(mask.sum())

                if count == 0:
                    continue

                p_med = (
                    float(np.median(pressure[mask]))
                    if pressure is not None else float("nan")
                )
                v_med = (
                    float(np.median(speed[mask]))
                    if speed is not None else float("nan")
                )

                print(
                    f"      {edges[i]:+8.3f}~{edges[i+1]:+8.3f}  {count:9d}  "
                    f"{p_med:12.2f}  {v_med:12.4f}"
                )

    print()

def main():
    for tag, case_name, nominal in CASES:
        try:
            probe(tag, case_name, nominal)
        except Exception as exc:
            print(f"[{tag}] {case_name} 异常：{type(exc).__name__}: {exc}\n")

if __name__ == "__main__":
    main()