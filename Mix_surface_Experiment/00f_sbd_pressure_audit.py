# -*- coding: utf-8 -*-
r"""
00f_sbd_pressure_audit.py

直接对 SBD 做压力场体检，不依赖任何中间 HDF5。

每个工况输出：
  * cp 统计（用文件夹名速度 + 指定 rho 归一化）
  * cp_min / cp_max 出现的位置与所属部件
  * 分部件（hull/sail/fin1~4）的 cp_min / cp_mean
  * p_gauge 与 z 的相关系数（判别是否开重力/静水压）

用法
----
python 00f_sbd_pressure_audit.py                       # 全部 52 个工况
python 00f_sbd_pressure_audit.py --case suboff_0000_u2p5_v0_w0
python 00f_sbd_pressure_audit.py --sbd <任意.sbd> --speed 6.5 --label TRAIN_v6p5
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

PART_ID_TO_NAME = {
    48: "sail",
    72: "hull",
    73: "fin1",
    74: "fin2",
    75: "fin3",
    76: "fin4",
}

PART_ORDER = ("hull", "sail", "fin1", "fin2", "fin3", "fin4")

# 训练集（纯轴向）物理基准，来自 unseen_sbd_conversion_validation (v8p5)
TRAINING_BASELINE = {
    "cp_min": -1.1703,
    "cp_max": +1.0019,
    "cp_mean": -0.0569,
    "cp_std": 0.1283,
}

def load_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def audit_sbd(builder, legacy, label, sbd_path: Path, speed: float, rho: float):
    sbd_path = Path(sbd_path)
    q_inf = 0.5 * rho * speed * speed

    with sbd_path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mm:

            if mm[:4] != b"SBDF":
                raise ValueError(f"{sbd_path.name} 不是 SBDF")

            fields = {
                key: builder.find_optional_header(legacy, mm, cands)
                for key, cands in builder.SBD_FIELD_CANDIDATES.items()
            }

            face_count = int(
                legacy.infer_face_count(
                    fields["centroid_x"], fields["centroid_y"], sbd_path=sbd_path
                )
            )

            def read(key):
                return np.asarray(
                    legacy.read_float64_array(mm, fields[key], face_count),
                    dtype=np.float64,
                )

            x = read("centroid_x")
            y = read("centroid_y")
            z = read("centroid_z")
            part = np.rint(read("part_index")).astype(np.int64)
            p_abs = read("p_static_absolute")

    keep = np.isin(part, list(PART_ID_TO_NAME))
    x, y, z, part, p_abs = x[keep], y[keep], z[keep], part[keep], p_abs[keep]

    p_gauge = p_abs - float(cfg.P_STATIC_REF_PA)
    cp = p_gauge / q_inf

    i_min = int(np.argmin(cp))
    i_max = int(np.argmax(cp))

    row = {
        "label": label,
        "speed_m_s": speed,
        "q_inf_pa": q_inf,
        "n_faces": int(len(cp)),
        "cp_min": float(cp.min()),
        "cp_max": float(cp.max()),
        "cp_mean": float(cp.mean()),
        "cp_std": float(cp.std()),
        "p_gauge_min_pa": float(p_gauge.min()),
        "p_gauge_max_pa": float(p_gauge.max()),
        "cp_min_xyz": f"({x[i_min]:+.3f},{y[i_min]:+.3f},{z[i_min]:+.3f})",
        "cp_min_part": PART_ID_TO_NAME.get(int(part[i_min]), "?"),
        "cp_max_xyz": f"({x[i_max]:+.3f},{y[i_max]:+.3f},{z[i_max]:+.3f})",
        "cp_max_part": PART_ID_TO_NAME.get(int(part[i_max]), "?"),
    }

    if np.std(z) > 0 and np.std(p_gauge) > 0:
        row["corr_pgauge_z"] = float(np.corrcoef(p_gauge, z)[0, 1])
    else:
        row["corr_pgauge_z"] = float("nan")

    for name in PART_ORDER:
        pid = next(k for k, v in PART_ID_TO_NAME.items() if v == name)
        sub = cp[part == pid]
        if len(sub):
            row[f"{name}_cp_min"] = float(sub.min())
            row[f"{name}_cp_mean"] = float(sub.mean())

    if abs(row["corr_pgauge_z"]) > 0.8:
        row["verdict"] = "GRAVITY?"
    elif row["cp_min"] > -3.0 and row["cp_max"] < 1.5:
        row["verdict"] = "OK"
    elif row["cp_min"] > -6.0 and row["cp_max"] < 2.5:
        row["verdict"] = "WARN"
    else:
        row["verdict"] = "BAD"

    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None, help="只体检指定 case_id")
    parser.add_argument("--sbd", default=None, help="体检任意 SBD 文件")
    parser.add_argument("--speed", type=float, default=None, help="配合 --sbd 指定来流速度")
    parser.add_argument("--label", default=None)
    parser.add_argument("--rho", type=float, default=None)
    args = parser.parse_args()

    rho = float(args.rho) if args.rho else float(cfg.RHO)

    builder = load_module_from_path(
        Path(cfg.BUILDER_SCRIPT_PATH), "suboff_builder_for_experiment"
    )
    legacy = builder.load_legacy_module()

    print("=" * 140)
    print(f"rho = {rho}    P_STATIC_REF_PA = {cfg.P_STATIC_REF_PA}")
    print(
        f"训练集基准(v8p5)：cp_min={TRAINING_BASELINE['cp_min']:+.4f}  "
        f"cp_max={TRAINING_BASELINE['cp_max']:+.4f}  "
        f"cp_mean={TRAINING_BASELINE['cp_mean']:+.4f}  "
        f"cp_std={TRAINING_BASELINE['cp_std']:.4f}"
    )
    print("=" * 140)

    rows = []

    if args.sbd:
        if args.speed is None:
            raise SystemExit("--sbd 必须配合 --speed")
        rows.append(
            audit_sbd(
                builder, legacy,
                args.label or Path(args.sbd).stem,
                Path(args.sbd), args.speed, rho,
            )
        )
    else:
        for case in cfg.UNSEEN_SBD_CASES:
            case_id = str(case["case_id"])

            if args.case and case_id != args.case:
                continue

            u = float(case["u_inf_m_s"])
            v = float(case["v_inf_m_s"])
            w = float(case["w_inf_m_s"])
            speed = float(np.sqrt(u * u + v * v + w * w))

            rows.append(
                audit_sbd(
                    builder, legacy, case_id,
                    Path(case["sbd_path"]), speed, rho,
                )
            )

    df = pd.DataFrame(rows)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 80)

    show = [
        "label", "speed_m_s", "cp_min", "cp_max", "cp_mean", "cp_std",
        "cp_min_part", "cp_max_part", "corr_pgauge_z", "verdict",
    ]
    print(df[show].to_string(index=False))

    print()
    print("=" * 140)
    print("分部件 cp_min / cp_mean")
    print("=" * 140)

    part_cols = [
        c for c in df.columns
        if c.endswith("_cp_min") or c.endswith("_cp_mean")
    ]
    print(df[["label"] + part_cols].to_string(index=False))

    if "verdict" in df.columns:
        print()
        print("=" * 140)
        print("判定汇总")
        print("=" * 140)
        print(df["verdict"].value_counts().to_string())
        bad = df[df["verdict"] != "OK"]
        if len(bad):
            print("\n非 OK 的工况：")
            print(bad[["label", "cp_min", "cp_max", "cp_mean", "verdict"]].to_string(index=False))

    out = cfg.LOG_ROOT / "sbd_pressure_audit.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已保存：{out}")

if __name__ == "__main__":
    main()