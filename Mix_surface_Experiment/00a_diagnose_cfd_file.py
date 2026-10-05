# -*- coding: utf-8 -*-
r"""
00a_diagnose_cfd_file.py

诊断 STAR-CCM+ 导出的 .cgns / .sbd / .h5 究竟能不能被 Python 读取，
并给出明确修复路线。

用法
----
# 1. 诊断单个文件
python 00a_diagnose_cfd_file.py "E:/suboff/suboff_0000/SurfaceCheck/u-0p5v-0p5w-0p5/xxx.cgns"

# 2. 诊断一个工况文件夹（自动挑第一个结果文件）
python 00a_diagnose_cfd_file.py --case "E:/suboff/suboff_0000/SurfaceCheck/u-0p5v-0p5w-0p5"

# 3. 批量扫描根目录，只输出格式统计（52个工况一次看完）
python 00a_diagnose_cfd_file.py --root "E:/suboff/suboff_0000/SurfaceCheck" --summary

# 4. 如果是 HDF5-CGNS，把完整数据集树打出来
python 00a_diagnose_cfd_file.py "E:/suboff/suboff_0000/SurfaceCheck/xxx.cgns" --tree

注意
----
本文件所有示例路径统一使用正斜杠，避免 Windows 反斜杠被误认为转义符。
实际传入路径时反斜杠、正斜杠都可以，Path 会自行处理。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# =============================================================================
# 工具
# =============================================================================

def hexdump(data: bytes, width: int = 16, limit: int = 256) -> str:
    data = data[:limit]
    lines = []

    for offset in range(0, len(data), width):
        chunk = data[offset:offset + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(
            chr(b) if 32 <= b < 127 else "."
            for b in chunk
        )
        lines.append(
            f"  {offset:08x}  {hex_part:<{width * 3}}  |{ascii_part}|"
        )

    return "\n".join(lines)

def report_library_status():
    print("-" * 100)
    print("依赖库状态")
    print("-" * 100)

    for module_name in ("numpy", "pandas", "h5py", "meshio", "CGNS.MAP"):
        try:
            if module_name == "CGNS.MAP":
                import CGNS.MAP  # noqa: F401
            else:
                __import__(module_name)

            version = ""

            try:
                module = sys.modules[module_name]
                version = getattr(module, "__version__", "")
            except Exception:
                pass

            print(f"  [OK]   {module_name:<12s} {version}")

        except Exception as exc:
            print(f"  [缺失] {module_name:<12s} ({exc})")

    print()

def sniff_magic(head: bytes) -> str:
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "HDF5"

    if head[:4] == b"SBDF":
        return "SBDF"          # STAR-CCM+ 表面结果，legacy mmap 解析器可读

    if head.startswith(b"ADF"):
        return "CGNS-ADF"

    if head[:4] == b"CGNS":
        return "CGNS-magic"

    if b"CGNS" in head[:256]:
        return "CGNS-text"

    if b"<?xml" in head[:64]:
        return "XML"

    return "UNKNOWN"

# =============================================================================
# 各读取器探测
# =============================================================================

def probe_h5py(path: Path, dump_tree: bool = False):
    result = {"ok": False, "error": None, "is_cgns": False, "keys": []}

    try:
        import h5py
    except ImportError:
        result["error"] = "h5py 未安装"
        return result

    try:
        with h5py.File(str(path), mode="r") as handle:
            result["ok"] = True
            result["keys"] = list(handle.keys())
            result["is_cgns"] = (
                "CGNSLibraryVersion" in handle
                or any(key.startswith("Base") for key in handle.keys())
            )

            if dump_tree:
                datasets = []
                groups = []

                def visitor(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets.append(
                            (name, obj.shape, str(obj.dtype))
                        )
                    else:
                        groups.append(name)

                handle.visititems(visitor)

                print("  [HDF5树] 组：")
                for name in groups[:60]:
                    print(f"      {name}")

                print(f"  [HDF5树] 数据集（共 {len(datasets)} 个）：")
                for name, shape, dtype in datasets[:80]:
                    print(f"      {name:<70s} {str(shape):<20s} {dtype}")

                if len(datasets) > 80:
                    print(f"      ... 其余 {len(datasets) - 80} 个省略")

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result

def probe_meshio(path: Path):
    result = {"ok": False, "error": None, "info": ""}

    try:
        import meshio
    except ImportError:
        result["error"] = "meshio 未安装"
        return result

    try:
        mesh = meshio.read(str(path))

        result["ok"] = True
        result["info"] = (
            f"节点={len(mesh.points):,}  "
            f"point_data={list((mesh.point_data or {}).keys())}  "
            f"cell_data={list((mesh.cell_data or {}).keys())}"
        )

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result

def probe_cgnslib(path: Path):
    result = {"ok": False, "error": None, "info": "", "tree": None}

    try:
        import CGNS.MAP as cgns_map
    except ImportError:
        result["error"] = "CGNS (PyCGNS) 未安装"
        return result

    try:
        tree, links, paths = cgns_map.load(str(path))
        result["ok"] = True
        result["info"] = f"根节点={tree[0]}"
        result["tree"] = tree

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result

# =============================================================================
# 单个文件诊断
# =============================================================================

def diagnose_file(path: Path, dump_tree: bool = False) -> dict:
    print("=" * 100)
    print(f"文件：{path}")
    print("=" * 100)

    if not path.is_file():
        print("[错误] 文件不存在")
        return {"verdict": "missing"}

    size = path.stat().st_size
    print(f"大小：{size:,} 字节 ({size / 1024 / 1024:.2f} MB)")

    with open(path, "rb") as handle:
        head = handle.read(256)

    magic = sniff_magic(head)

    print(f"文件头魔数：{magic}")
    print(f"前64字节 hex：")
    print(hexdump(head[:64], limit=64))
    print()

    print("[1/3] 尝试 h5py（读 HDF5-CGNS）")
    h5_result = probe_h5py(path, dump_tree=dump_tree)

    if h5_result["ok"]:
        print(f"      OK，根节点：{h5_result['keys']}")
        print(f"      是CGNS结构：{h5_result['is_cgns']}")
    else:
        print(f"      失败 -> {h5_result['error']}")

    print()
    print("[2/3] 尝试 meshio（读 CGNS）")
    meshio_result = probe_meshio(path)

    if meshio_result["ok"]:
        print(f"      OK，{meshio_result['info']}")
    else:
        print(f"      失败 -> {meshio_result['error']}")

    print()
    print("[3/3] 尝试 PyCGNS（读 ADF + HDF5 两种 CGNS）")
    cgns_result = probe_cgnslib(path)

    if cgns_result["ok"]:
        print(f"      OK，{cgns_result['info']}")

        if dump_tree and cgns_result["tree"] is not None:
            print("      PyCGNS 顶层结构：")
            try:
                for child in cgns_result["tree"][2][:20]:
                    print(f"        {child[0]}  type={child[3]}")
            except Exception:
                pass
    else:
        print(f"      失败 -> {cgns_result['error']}")

    # ---- 判定 ----
    print()
    print("-" * 100)
    print("判定")
    print("-" * 100)

    if h5_result["ok"] and h5_result["is_cgns"]:
        verdict = "hdf5_cgns"
        action = (
            "这是 HDF5 版 CGNS，可以直接被 01 脚本读取。\n"
            "  若 01 脚本仍失败，说明 01 脚本里的 _load_with_h5py 还是旧版，\n"
            "  请换上最新版（旧版对 CGNS 数据集命名过于严格）。"
        )

    elif h5_result["ok"] and not h5_result["is_cgns"]:
        verdict = "hdf5_not_cgns"
        action = (
            "这是 HDF5 但不是 CGNS（可能是 STAR-CCM+ 原生 HDF5）。\n"
            "  需要用 --tree 看结构后定制解析，请把树结构发我。"
        )

    elif meshio_result["ok"]:
        verdict = "meshio_ok"
        action = "可被 meshio 直接读取，01 脚本应能工作。"

    elif cgns_result["ok"]:
        verdict = "adf_cgns"
        action = (
            "这是 ADF 版 CGNS（CGNS 2.5）。h5py 读不了，PyCGNS 可以。\n"
            "  方案1：装 PyCGNS 并走 cgnslib 分支\n"
            "         conda install -c conda-forge cgnslib\n"
            "  方案2：在 STAR-CCM+ 中改导出为 HDF5 版 CGNS（更省事）\n"
            "  方案3：改为导出整面 CSV"
        )

    elif magic == "UNKNOWN":
        verdict = "proprietary"
        action = (
            "这是 STAR-CCM+ 专有二进制格式，Python 无法解析。\n"
            "  必须改用其他导出格式：\n"
            "    (a) CGNS-HDF5\n"
            "    (b) 整面 CSV（推荐，52个工况可用批处理宏一次完成）\n"
            "    (c) EnSight / Tecplot"
        )

    elif magic == "SBDF":
        verdict = "sbd_sbdf"
        action = (
            "这是 STAR-CCM+ 的 SBDF 表面结果文件，h5py/meshio/PyCGNS 都读不了，\n"
            "  但本项目已有 mmap 解析器可读：\n"
            "    F:\\14_Sub\\SUBANS\\unfold\\unfold_star_sbd_surface.py\n"
            "  经 build_modal_surface_dataset.py -> read_sbd_case() 调用。\n"
            "  可提供 121,772 个 SUBOFF 湿表面面的 Centroid / LocalFaceIndex /\n"
            "  AbsolutePressure / AbsoluteTotalPressure / WallY+ / WallShearStress。"
        )

    else:
        verdict = "unknown"
        action = "请把上面完整输出发我。"

    print(f"结论：{verdict}")
    print(f"建议：{action}")

    return {
        "path": str(path),
        "verdict": verdict,
        "size": size,
        "magic": magic,
        "h5py_ok": h5_result["ok"],
        "h5py_is_cgns": h5_result["is_cgns"],
        "meshio_ok": meshio_result["ok"],
        "pycgns_ok": cgns_result["ok"],
    }

# =============================================================================
# 批量扫描
# =============================================================================

# =============================================================================
# 批量扫描
# =============================================================================

# 这两类是"正常"判定，其余一律算异常，需要单独列出来
BENIGN_VERDICTS = {"hdf5_cgns", "sbd_sbdf"}

def scan_root(root: Path, full: bool = False, summary_only: bool = False):
    print("=" * 100)
    print(f"批量扫描：{root}")
    print("=" * 100)

    result_counter = Counter()
    rows = []
    problem_rows = []

    case_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    for case_dir in case_dirs:
        files = []

        for pattern in ("*.cgns", "*.sbd", "*.h5", "*.hdf5"):
            files.extend(case_dir.glob(pattern))

        unique = {}

        for path in files:
            try:
                key = str(path.resolve())
            except OSError:
                key = str(path)

            unique.setdefault(key, path)

        if not unique:
            result_counter["no_file"] += 1
            row = (case_dir.name, "(无结果文件)", "no_file", "", "")
            rows.append(row)
            problem_rows.append(row)
            continue

        for path in unique.values():
            try:
                with open(path, "rb") as handle:
                    head = handle.read(64)
            except OSError as exc:
                result_counter["unreadable"] += 1
                row = (case_dir.name, path.name, "unreadable", "",
                       f"打开失败: {exc}")
                rows.append(row)
                problem_rows.append(row)
                continue

            magic = sniff_magic(head)
            h5 = probe_h5py(path)

            if h5["ok"] and h5["is_cgns"]:
                verdict = "hdf5_cgns"
            elif h5["ok"]:
                verdict = "hdf5_not_cgns"
            elif magic == "HDF5":
                verdict = "hdf5_broken"          # 头是HDF5但h5py打不开
            elif magic == "CGNS-ADF":
                verdict = "adf_cgns"
            elif path.suffix.lower() == ".sbd":
                verdict = "sbd_proprietary"      # .sbd单独归类
            elif magic == "SBDF":
                verdict = "sbd_sbdf"  # 正常工作格式，不是 proprietary
            else:
                suffix = path.suffix.lower().lstrip(".") or "file"
                verdict = f"{suffix}_broken"     # cgns_broken / h5_broken ...

            error = h5["error"] or ""

            try:
                size_text = f"{path.stat().st_size / 1024 / 1024:.1f} MB"
            except OSError:
                size_text = "?"

            result_counter[verdict] += 1
            row = (case_dir.name, path.name, verdict, size_text, error)
            rows.append(row)

            if verdict not in BENIGN_VERDICTS:
                problem_rows.append(row)

    if not summary_only:
        print(f"{'工况文件夹':<28s} {'文件':<50s} {'判定':<18s} {'大小':>10s}  h5py错误")
        print("-" * 140)

        visible = rows if full else rows[:60]

        for row in visible:
            print(
                f"{row[0]:<28s} {row[1]:<50s} {row[2]:<18s} "
                f"{row[3]:>10s}  {row[4][:40]}"
            )

        if not full and len(rows) > 60:
            print(f"... 其余 {len(rows) - 60} 行省略（加 --full 全部打印）")

        print()

    print("=" * 100)
    print("格式统计")
    print("=" * 100)

    for verdict, count in result_counter.most_common():
        print(f"  {verdict:<20s} {count}")

    print()
    print("=" * 100)
    print("异常文件清单（非 hdf5_cgns / 非 sbd_proprietary）")
    print("=" * 100)

    if not problem_rows:
        print("  无。全部文件正常。")
    else:
        for row in problem_rows:
            print(f"  目录：{row[0]}")
            print(f"    文件：{row[1]}")
            print(f"    判定：{row[2]}    大小：{row[3]}")
            if row[4]:
                print(f"    h5py：{row[4]}")
            print()

        print("  处置建议：")
        print("    hdf5_broken / cgns_broken / unreadable")
        print("        -> 该工况 .cgns 损坏或写盘未完成，必须重新导出")
        print("    adf_cgns")
        print("        -> STAR-CCM+ 里改导出 HDF5 版 CGNS")
        print("    hdf5_not_cgns")
        print("        -> 用 --tree 看结构后定制解析")

    print("=" * 100)

    return problem_rows

# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="诊断 STAR-CCM+ 结果文件格式"
    )

    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="要诊断的文件路径",
    )

    parser.add_argument(
        "--case",
        default=None,
        help="要诊断的工况文件夹（自动挑第一个结果文件）",
    )

    parser.add_argument(
        "--root",
        default=None,
        help="批量扫描根目录",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="配合 --root 使用，只输出统计和异常清单（不打明细表）",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="配合 --root 使用，明细表不截断（默认只打印前60行）",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="如果是 HDF5，打印完整数据集树",
    )

    args = parser.parse_args()

    report_library_status()

    if args.root:
        scan_root(
            Path(args.root).expanduser(),
            full=args.full,
            summary_only=args.summary,
        )
        return


    target = None

    if args.path:
        target = Path(args.path).expanduser()

    elif args.case:
        case_dir = Path(args.case).expanduser()

        if not case_dir.is_dir():
            print(f"[错误] 文件夹不存在：{case_dir}")
            sys.exit(1)

        found = []

        for pattern in ("*.cgns", "*.sbd", "*.h5", "*.hdf5"):
            found.extend(sorted(case_dir.glob(pattern)))

        if not found:
            print(f"[错误] {case_dir} 内没有结果文件")
            sys.exit(1)

        target = found[0]

    if target is None:
        parser.print_help()
        sys.exit(0)

    diagnose_file(target, dump_tree=args.tree)

if __name__ == "__main__":
    main()