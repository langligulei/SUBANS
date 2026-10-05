# -*- coding: utf-8 -*-
r"""
00g_sbd_freestream_probe.py

从 SBD 自身的"外边界部件"面（47/56/57，共 2,424 面）上直接测出真实来流，
用来判定 cp 错误到底出在"速度归一化"还是"压力水平面"。

输出每个工况：
  * V_label（文件夹名）  vs  V_meas（外边界面实测）  -> V_ratio
  * 外边界实测来流方向
  * p_inf,meas = 外边界绝对压中位数，及其相对 101325 Pa 的偏差
  * 三种归一化下的 SUBOFF cp 统计：
        raw    : (p - 101325) / q_label     <- 你现在用的
        pcorr  : (p - p_inf,meas) / q_label
        pVcorr : (p - p_inf,meas) / q_meas

用法
----
python 00g_sbd_freestream_probe.py
python 00g_sbd_freestream_probe.py --sbd "E:\suboff（0-200）\suboff_0000\MeshMedium\suboff_0000_v1.sbd" --speed 1.0 --label TRAIN_v1
python 00g_sbd_freestream_probe.py --sbd "E:\suboff（0-200）\suboff_0000\SurfaceCheck\v6p5\suboff_0000_v6p5.sbd" --speed 6.5 --label TRAIN_v6p5
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

SUBOFF_PART_IDS = (48, 72, 73, 74, 75, 76)
BOUNDARY_PART_IDS = (47, 56, 57)

BOUND_PART_NAME = {47: "bnd47", 56: "bnd56", 57: "bnd57"}

def load_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def probe(builder, legacy, label, sbd_path: Path, rho: float,
          v_label: float | None, p_ref: float):
    sbd_path = Path(sbd_path)

    with sbd_path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mm:

            fields = {
                key: builder.find_optional_header(legacy, mm, cands)
                for key, cands in builder.SBD_FIELD_CANDIDATES.items()
            }

            n = int(legacy.infer_face_count(
                fields["centroid_x"], fields["centroid_y"], sbd_path=sbd_path))

            def read(key):
                return np.asarray(
                    legacy.read_float64_array(mm, fields[key], n), dtype=np.float64)

            x = read("centroid_x")
            y = read("centroid_y")
            z = read("centroid_z")
            part = np.rint(read("part_index")).astype(np.int64)
            p_abs = read("p_static_absolute")
            vx, vy, vz = read("velocity_x"), read("velocity_y"), read("velocity_z")

    sub = np.isin(part, SUBOFF_PART_IDS)
    bnd = np.isin(part, BOUNDARY_PART_IDS)

    row = {"label": label, "n_suboff": int(sub.sum()), "n_boundary": int(bnd.sum())}

    # ---------- 外边界逐部件 ----------
    for pid in BOUNDARY_PART_IDS:
        m = part == pid
        if not m.any():
            continue
        vmag = np.sqrt(vx[m] ** 2 + vy[m] ** 2 + vz[m] ** 2)
        name = BOUND_PART_NAME[pid]
        row[f"{name}_n"] = int(m.sum())
        row[f"{name}_V"] = float(np.median(vmag))
        row[f"{name}_dir"] = (f"({np.median(vx[m]):+.3f},"
                              f"{np.median(vy[m]):+.3f},"
                              f"{np.median(vz[m]):+.3f})")
        row[f"{name}_pg"] = float(np.median(p_abs[m]) - p_ref)

    # ---------- 外边界合并 ----------
    vmag_b = np.sqrt(vx[bnd] ** 2 + vy[bnd] ** 2 + vz[bnd] ** 2)

    V_meas = float(np.median(vmag_b))
    dir_meas = (float(np.median(vx[bnd])),
                float(np.median(vy[bnd])),
                float(np.median(vz[bnd])))
    p_far = float(np.median(p_abs[bnd]))
    p_far_std = float(np.std(p_abs[bnd]))

    row["V_label"] = float("nan") if v_label is None else float(v_label)
    row["V_meas"] = V_meas
    row["V_ratio"] = (V_meas / v_label) if v_label else float("nan")
    row["V_dir_meas"] = f"({dir_meas[0]:+.3f},{dir_meas[1]:+.3f},{dir_meas[2]:+.3f})"
    row["p_far_abs_pa"] = p_far
    row["p_far_gauge_pa"] = p_far - p_ref
    row["p_far_std_pa"] = p_far_std

    # ---------- 三种归一化 ----------
    if v_label:
        V_for_q = float(v_label)
    else:
        V_for_q = V_meas

    q_label = 0.5 * rho * V_for_q ** 2
    q_meas = 0.5 * rho * V_meas ** 2

    def stats(p_gauge, q):
        cp = p_gauge / q
        return (float(cp.min()), float(cp.max()), float(cp.mean()), float(cp.std()))

    raw = stats(p_abs[sub] - p_ref, q_label)
    pcorr = stats(p_abs[sub] - p_far, q_label)
    pVcorr = stats(p_abs[sub] - p_far, q_meas)

    for tag, s in (("raw", raw), ("pcorr", pcorr), ("pVcorr", pVcorr)):
        row[f"cp_{tag}_min"] = s[0]
        row[f"cp_{tag}_max"] = s[1]
        row[f"cp_{tag}_mean"] = s[2]
        row[f"cp_{tag}_std"] = s[3]

    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None)
    parser.add_argument("--sbd", default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--p-ref", type=float, default=None)
    args = parser.parse_args()

    rho = float(args.rho) if args.rho else float(cfg.RHO)
    p_ref = float(args.p_ref) if args.p_ref else float(cfg.P_STATIC_REF_PA)

    builder = load_module_from_path(
        Path(cfg.BUILDER_SCRIPT_PATH), "suboff_builder_for_experiment")
    legacy = builder.load_legacy_module()

    print("=" * 150)
    print(f"rho={rho}   P_REF={p_ref}   "
          f"训练集基准(v8p5) cp: min=-1.1703 max=+1.0019 mean=-0.0569 std=0.1283")
    print("=" * 150)

    rows = []

    if args.sbd:
        if args.speed is None:
            raise SystemExit("--sbd 需要 --speed")
        rows.append(probe(builder, legacy, args.label or Path(args.sbd).stem,
                          Path(args.sbd), rho, args.speed, p_ref))
    else:
        for case in cfg.UNSEEN_SBD_CASES:
            cid = str(case["case_id"])
            if args.case and cid != args.case:
                continue
            u, v, w = (float(case["u_inf_m_s"]), float(case["v_inf_m_s"]),
                       float(case["w_inf_m_s"]))
            rows.append(probe(builder, legacy, cid, Path(case["sbd_path"]),
                              rho, float(np.sqrt(u * u + v * v + w * w)), p_ref))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 100)

    print("\n【A】来流与压力水平面实测")
    print(df[["label", "V_label", "V_meas", "V_ratio", "V_dir_meas",
              "p_far_gauge_pa", "p_far_std_pa"]].to_string(index=False))

    print("\n【B】三种归一化下的 SUBOFF cp")
    print(df[["label",
              "cp_raw_min", "cp_raw_max", "cp_raw_mean",
              "cp_pcorr_mean", "cp_pVcorr_mean",
              "cp_pVcorr_min", "cp_pVcorr_max", "cp_pVcorr_std"]].to_string(index=False))

    print("\n【C】外边界逐部件")
    extra = [c for c in df.columns if c.startswith("bnd")]
    print(df[["label"] + extra].to_string(index=False))

    print("\n" + "=" * 150)
    print("判读矩阵")
    print("=" * 150)
    print("  V_ratio≈1 且 |p_far_gauge|≈0  -> 原始cp就是对的，那些工况的CFD解本身有问题")
    print("  V_ratio≈1 且 |p_far_gauge|>>0  -> 压力水平面错，cp_pcorr 可救回")
    print("  V_ratio≠1 且 |p_far_gauge|≈0   -> 速度标注/设置错，cp_pVcorr 可救回")
    print("  V_ratio≠1 且 |p_far_gauge|>>0   -> 两者都错，必须回 STAR-CCM+ 重修")
    print("=" * 150)

    out = cfg.LOG_ROOT / "sbd_freestream_probe.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已保存：{out}")

if __name__ == "__main__":
    main()