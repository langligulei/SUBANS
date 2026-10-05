# -*- coding: utf-8 -*-
r"""
00c_list_cgns_zones.py

列出 STAR-CCM+ HDF5-CGNS 里的全部 zone，以及每个 zone 的：
  * GridCoordinates 点数
  * Solution 组数量与名字
  * GridLocation
  * 每个字段的长度（按长度分组，并标出与 topology 面数一致的那一组）

目的：确认"域里到底有几个 zone、表面 zone 是哪一个"，
      从而决定 01_extract_sensor_values_from_cfd.py 该怎么选 zone。

用法
----
python 00c_list_cgns_zones.py "E:\suboff（0-200）\suboff_0000\SurfaceCheck\u0v0w1p5"
python 00c_list_cgns_zones.py --root "E:\suboff（0-200）\suboff_0000\SurfaceCheck" --first 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py

# 已知的 topology 表面面数（从 unseen_sbd_conversion_validation.csv 得到）
DEFAULT_FACE_COUNT = 121772

# 关心的字段名
FIELDS_OF_INTEREST = (
    "Pressure",
    "AbsolutePressure",
    "VelocityX",
    "VelocityY",
    "VelocityZ",
    "CentroidX",
    "CentroidY",
    "CentroidZ",
    "PressureCoefficient",
    "AbsoluteTotalPressure",
)

def n_elem(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n

def decode_string(group) -> str:
    for key in (" data", "data"):
        if key in group:
            arr = group[key][()]
            try:
                flat = [int(v) for v in arr.ravel()]
                return bytes(flat).split(b"\x00")[0].decode("ascii", "ignore").strip()
            except Exception:
                return str(arr)
    return ""

def list_zones(path: Path, face_count: int):
    print("=" * 100)
    print(f"文件：{path}")
    print(f"大小：{path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"待匹配的表面面数：{face_count:,}")
    print("=" * 100)

    with h5py.File(str(path), "r") as handle:
        roots = [k for k in handle.keys() if isinstance(handle[k], h5py.Group)]

        print(f"顶层组：{roots}")

        zone_counter = 0

        for base_name in roots:
            base = handle[base_name]

            zones = [k for k in base.keys() if isinstance(base[k], h5py.Group)]

            for zone_name in zones:
                zone = base[zone_name]
                children = list(zone.keys())
                zone_counter += 1

                print()
                print("-" * 100)
                print(f"[zone {zone_counter}] /{base_name}/{zone_name}")
                print(f"  子节点：{children}")

                # ---- GridCoordinates ----
                if "GridCoordinates" in children:
                    gc = zone["GridCoordinates"]
                    n_node = None
                    for k in gc.keys():
                        g = gc[k]
                        if isinstance(g, h5py.Group) and " data" in g:
                            n_node = n_elem(g[" data"].shape)
                    print(f"  GridCoordinates 点数：{n_node:,}" if n_node else "  GridCoordinates 点数为空")
                else:
                    print("  GridCoordinates：无")

                # ---- Solution 组 ----
                sols = sorted(k for k in children if k.lower().startswith("solution"))
                print(f"  Solution 组数：{len(sols)}  -> {sols}")

                for sol_name in sols:
                    sg = zone[sol_name]

                    loc = ""
                    if "GridLocation" in sg:
                        loc = decode_string(sg["GridLocation"])

                    fields = {}
                    for k in sg.keys():
                        g = sg[k]
                        if isinstance(g, h5py.Group) and " data" in g:
                            fields[k] = n_elem(g[" data"].shape)

                    print(f"\n    {sol_name}   GridLocation={loc!r}   字段数={len(fields)}")

                    # 按长度分组
                    by_size = {}
                    for name, size in fields.items():
                        by_size.setdefault(size, []).append(name)

                    for size, names in sorted(
                        by_size.items(), key=lambda kv: -len(kv[1])
                    ):
                        names = sorted(names)
                        head = ", ".join(names[:10])
                        tail = " ..." if len(names) > 10 else ""
                        mark = "   <== 与 topology 面数一致" if size == face_count else ""
                        print(
                            f"      长度={size:>10,d} ({len(names):>2d}个): "
                            f"{head}{tail}{mark}"
                        )

                    for key in FIELDS_OF_INTEREST:
                        if key in fields:
                            print(f"      [关心] {key:<24s} 长度={fields[key]:,}")

    print()
    print("=" * 100)
    print("判读要点")
    print("=" * 100)
    print("  1. 如果出现多个 zone，且只有其中一个 zone 的字段长度 = 面数，")
    print("     那它才是表面 zone；01 脚本必须显式选中它。")
    print("  2. 如果所有 zone 的字段长度都 = 4,960,604（体单元数），")
    print("     说明这批 CGNS 里根本没有表面解，需要重新导出表面 zone。")
    print("  3. 如果一个 zone 下有多个 Solution 组，")
    print("     01 脚本不能取 solutions[0]，要取编号最大的那个。")
    print("=" * 100)

def main():
    parser = argparse.ArgumentParser(description="列出 HDF5-CGNS 的全部 zone")
    parser.add_argument("path", nargs="?", default=None, help="单个 cgns 文件或工况文件夹")
    parser.add_argument("--root", default=None, help="批量扫描根目录")
    parser.add_argument("--first", type=int, default=2, help="--root 模式下只跑前 N 个工况")
    parser.add_argument("--face-count", type=int, default=DEFAULT_FACE_COUNT)
    args = parser.parse_args()

    targets = []

    if args.root:
        root = Path(args.root).expanduser()
        for case_dir in sorted(p for p in root.iterdir() if p.is_dir())[: args.first]:
            targets.extend(sorted(case_dir.glob("*.cgns")))
    elif args.path:
        p = Path(args.path).expanduser()
        if p.is_dir():
            targets.extend(sorted(p.glob("*.cgns")))
        else:
            targets.append(p)

    if not targets:
        print("[错误] 没找到任何 .cgns")
        return

    for path in targets:
        try:
            list_zones(path, args.face_count)
        except Exception as exc:
            print(f"[异常] {path.name}: {type(exc).__name__}: {exc}\n")

if __name__ == "__main__":
    main()