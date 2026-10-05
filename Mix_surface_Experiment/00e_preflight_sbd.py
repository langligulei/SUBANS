# -*- coding: utf-8 -*-
r"""
00e_preflight_sbd.py  (v2)

对 52 个三向速度 SBD 做四项体检，全部只用 mmap，不加载完整场数据：

  1. 字段 schema   -> 是否与训练集同构（0/N 与 N/N 都算"一致"）
  2. 参考压力      -> median(AbsolutePressure - StaticPressure) 是否 = 101325
  3. 部件ID        -> FacePartSurfaceIndex 是否包含 {48,72,73,74,75,76}
  4. 分组分类      -> Group A/B/C/D 分布

用法
----
python 00e_preflight_sbd.py
python 00e_preflight_sbd.py --reference-sbd "E:\suboff（0-200）\suboff_0000\MeshMedium\suboff_0000_v1.sbd"
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import mmap
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

cfg = importlib.import_module("00_experiment_config")

# 压力体检：只读每个场的前 N 个值
PRESSURE_PROBE_COUNT = 4096

# 参考压力应为 101325 Pa（与 P_STATIC_REF_PA 一致）
EXPECTED_REFERENCE_PRESSURE_PA = 101325.0

# SUBOFF 六部件在本项目中的 FacePartSurfaceIndex
EXPECTED_PART_IDS = {48, 72, 73, 74, 75, 76}

EXPECTED_SUBOFF_FACE_COUNT = 121_772

def load_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def probe_sbd(builder, legacy, label: str, sbd_path: Path) -> dict:
    """对一个 SBD 做全部体检，只使用 mmap 读取。"""
    row = {"label": label, "path": str(sbd_path), "exists": sbd_path.is_file()}

    if not sbd_path.is_file():
        row["status"] = "文件不存在"
        return row

    with sbd_path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mm:

            if mm[:4] != b"SBDF":
                row["status"] = f"不是SBDF({mm[:8].hex(' ')})"
                return row

            # ---- 1. 字段头 ----
            fields = {
                logical: builder.find_optional_header(legacy, mm, candidates)
                for logical, candidates in builder.SBD_FIELD_CANDIDATES.items()
            }

            missing = builder.find_missing_sbd_fields(fields)
            row["status"] = "通过" if not missing else "缺字段"
            row["missing"] = "; ".join(missing)

            for logical, header in fields.items():
                row[logical] = (
                    str(header.internal_name) if header is not None else None
                )

            face_count = int(
                legacy.infer_face_count(
                    fields["centroid_x"],
                    fields["centroid_y"],
                    sbd_path=sbd_path,
                )
            )
            row["sbd_face_count"] = face_count

            # ---- 2. 参考压力体检 ----
            if (
                fields["p_static_absolute"] is not None
                and fields["p_static_relative"] is not None
            ):
                n = int(min(PRESSURE_PROBE_COUNT, face_count))

                p_abs = np.asarray(
                    legacy.read_float64_array(
                        mm, fields["p_static_absolute"], n
                    ),
                    dtype=np.float64,
                )
                p_rel = np.asarray(
                    legacy.read_float64_array(
                        mm, fields["p_static_relative"], n
                    ),
                    dtype=np.float64,
                )

                delta = p_abs - p_rel

                row["p_abs_median_pa"] = float(np.median(p_abs))
                row["p_rel_median_pa"] = float(np.median(p_rel))
                row["reference_pressure_median_pa"] = float(np.median(delta))
                row["reference_pressure_std_pa"] = float(np.std(delta))

            # ---- 3. 部件ID体检（读整个 part_index，约 4.8 MB）----
            if fields["part_index"] is not None:
                part = np.asarray(
                    legacy.read_float64_array(
                        mm, fields["part_index"], face_count
                    ),
                    dtype=np.float64,
                )

                ids, counts = np.unique(
                    np.rint(part).astype(np.int64), return_counts=True
                )

                row["part_ids"] = ",".join(str(int(v)) for v in ids)
                row["part_id_counts"] = "; ".join(
                    f"{int(v)}:{int(c):,}" for v, c in zip(ids, counts)
                )
                row["part_ids_ok"] = EXPECTED_PART_IDS.issubset(
                    set(ids.tolist())
                )
                row["suboff_face_count_est"] = int(
                    counts[np.isin(ids, list(EXPECTED_PART_IDS))].sum()
                )

    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-sbd",
        default=r"E:\suboff（0-200）\suboff_0000\MeshMedium\suboff_0000_v1.sbd",
        help="训练集用的 SBD，作为口径参照",
    )
    args = parser.parse_args()

    builder = load_module_from_path(
        Path(cfg.BUILDER_SCRIPT_PATH), "suboff_builder_for_experiment"
    )
    legacy = builder.load_legacy_module()

    print(f"SBD 工况数   ：{len(cfg.UNSEEN_SBD_CASES)}")
    print(f"legacy 解析器：{builder.LEGACY_SCRIPT_PATH}")
    print(f"参考压力期望 ：{EXPECTED_REFERENCE_PRESSURE_PA:.1f} Pa")
    print("=" * 130)

    rows = []

    for case in cfg.UNSEEN_SBD_CASES:
        case_id = str(case["case_id"])
        row = probe_sbd(builder, legacy, case_id, Path(case["sbd_path"]))

        u = float(case["u_inf_m_s"])
        v = float(case["v_inf_m_s"])
        w = float(case["w_inf_m_s"])

        row["u"] = u
        row["v"] = v
        row["w"] = w
        row["speed"] = float(np.sqrt(u * u + v * v + w * w))
        row["group"] = cfg.classify_case_group(u, v, w)
        row["split_type"] = (
            "interpolation"
            if row["group"] in {"Group_A", "Group_B"}
            else "extrapolation"
        )

        rows.append(row)

        print(
            f"{case_id:<34s} status={row['status']:<6s} "
            f"faces={row.get('sbd_face_count', 0):>9,d} "
            f"refP={row.get('reference_pressure_median_pa', float('nan')):>12.3f}  "
            f"partIDs={'OK' if row.get('part_ids_ok') else '★BAD★'}  "
            f"suboffFaces={row.get('suboff_face_count_est', 0):>9,d}  "
            f"{row['group']}"
        )

    df = pd.DataFrame(rows)

    print()
    print("=" * 130)
    print("【体检 1】字段 schema 一致性（0/N 与 N/N 都算一致）")
    print("=" * 130)

    meta_cols = {"label", "path", "exists", "status", "missing",
                 "u", "v", "w", "speed", "group", "split_type"}
    logical = [c for c in df.columns if c not in meta_cols]

    inconsistent = []

    for name in logical:
        n = int(df[name].notna().sum())
        ok = (n == 0 or n == len(df))
        flag = "" if ok else "   <== 不一致！"
        if not ok:
            inconsistent.append(name)
        print(f"  {name:<24s} {n:>2d}/{len(df)}{flag}")

    print()
    print("=" * 130)
    print("【体检 2】参考压力（median(AbsolutePressure - StaticPressure)）")
    print("=" * 130)

    if "reference_pressure_median_pa" in df.columns:
        valid = df["reference_pressure_median_pa"].dropna()
        print(f"  工况数        ：{len(valid)}/{len(df)}")
        print(f"  中位数        ：{valid.median():.3f} Pa")
        print(f"  最小 / 最大   ：{valid.min():.3f} / {valid.max():.3f} Pa")
        print(f"  组内离散(std) ：{df['reference_pressure_std_pa'].max():.3e} Pa")

        ref_ok = bool(
            np.allclose(valid.to_numpy(), EXPECTED_REFERENCE_PRESSURE_PA, atol=1.0)
        )
        print(
            f"  判定          ：{'✓ 全部等于 101325 Pa，AbsolutePressure 可直接当绝对压用' if ref_ok else '✗ 与 101325 Pa 不符！'}"
        )
    else:
        ref_ok = False
        print("  ✗ 缺少 AbsolutePressure 或 StaticPressure，无法体检")

    print()
    print("=" * 130)
    print("【体检 3】FacePartSurfaceIndex 与 SUBOFF 面数")
    print("=" * 130)

    if "part_ids_ok" in df.columns:
        bad_parts = df[df["part_ids_ok"] != True]
        print(f"  期望包含      ：{sorted(EXPECTED_PART_IDS)}")
        print(f"  通过工况数    ：{len(df) - len(bad_parts)}/{len(df)}")
        if len(bad_parts):
            print("  不通过：")
            print(bad_parts[["label", "part_ids"]].to_string(index=False))

        if "suboff_face_count_est" in df.columns:
            fc = df["suboff_face_count_est"].dropna()
            face_ok = bool(
                np.all(fc.to_numpy() == EXPECTED_SUBOFF_FACE_COUNT)
            )
            print(
                f"  SUBOFF 面数   ：中位={fc.median():,.0f}  "
                f"最小={fc.min():,.0f} 最大={fc.max():,.0f}"
            )
            print(
                f"  期望          ：{EXPECTED_SUBOFF_FACE_COUNT:,}  -> "
                f"{'✓' if face_ok else '✗ 需要调整 SBD_SUBOFF_PART_IDS'}"
            )
            parts_ok = bool(len(bad_parts) == 0 and face_ok)
        else:
            parts_ok = False
    else:
        parts_ok = False
        print("  ✗ 缺少 part_index，无法体检")

    print()
    print("=" * 130)
    print("【体检 4】分组分布")
    print("=" * 130)

    print(df.groupby(["group", "split_type"]).size().to_string())

    print()
    print("=" * 130)
    print("总结论")
    print("=" * 130)

    schema_ok = (len(inconsistent) == 0)

    print(f"  字段 schema     ：{'✓ 一致' if schema_ok else '✗ ' + str(inconsistent)}")
    print(f"  参考压力 101325 ：{'✓' if ref_ok else '✗'}")
    print(f"  部件ID + 面数   ：{'✓' if parts_ok else '✗'}")

    if schema_ok and ref_ok and parts_ok:
        print()
        print("  >>> 全部通过，直接跑：")
        print("      python 01_build_unseen_cases_from_sbd.py --case suboff_0000_u0_v0_w1p5")
    else:
        print()
        print("  >>> 有问题，先不要跑全量 01_build。")

    out = cfg.LOG_ROOT / "unseen_sbd_preflight_v2.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已保存：{out}")

    # ---- 训练集参照 SBD ----
    ref_path = Path(args.reference_sbd)
    if ref_path.is_file():
        print()
        print("=" * 130)
        print(f"训练集参照 SBD：{ref_path}")
        print("=" * 130)

        ref_row = probe_sbd(builder, legacy, "TRAIN_v1", ref_path)

        for key in (
            "status", "sbd_face_count", "reference_pressure_median_pa",
            "reference_pressure_std_pa", "part_ids", "suboff_face_count_est",
            "p_static_absolute", "p_static_relative", "p_total_absolute",
            "wall_yplus", "wall_shear",
        ):
            print(f"  {key:<32s} {ref_row.get(key)}")

if __name__ == "__main__":
    main()