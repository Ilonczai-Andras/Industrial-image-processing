#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Pontfelhő Regisztrációs Pipeline - Stanford Bunny dataset
=============================================================

Lépések:
  1. Adatbevitel & előfeldolgozás  (voxel downsampling + Statistical Outlier Removal)
  2. Jellemzők kinyerése           (normálisok PCA-alapon + FPFH deszkriptorok)
  3. Durva illesztés               (RANSAC globális regisztráció FPFH-val)
  4. Finom illesztés               (ICP Point-to-Point ÉS Point-to-Plane)
  5. Minőségértékelés              (RMSE, fitness/overlap arány)
  6. Eredmény exportálása          (.ply fájl + JSON transzformációk + log)

Vizualizáció: Matplotlib (2D) + PyVista (3D, off-screen)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

# ─────────────────────────────────────────────────────────────────────
# Konfiguráció
# ─────────────────────────────────────────────────────────────────────

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(WORKSPACE_DIR, "input", "bunny", "data")
CONF_FILE     = os.path.join(DATA_DIR, "bun.conf")
OUTPUT_DIR    = os.path.join(WORKSPACE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Voxel méret (méter egységben; a bunny ~0.2 m magas)
VOXEL_SIZE = 0.005           # 5 mm

# Normális becslés
NORMAL_RADIUS = VOXEL_SIZE * 2
NORMAL_MAX_NN = 30

# FPFH
FPFH_RADIUS = VOXEL_SIZE * 5
FPFH_MAX_NN = 100

# RANSAC
RANSAC_DIST_THRESH  = VOXEL_SIZE * 1.5
RANSAC_MAX_ITER     = 100_000
RANSAC_CONFIDENCE   = 0.999

# ICP
ICP_DIST_THRESH = VOXEL_SIZE * 0.4
ICP_MAX_ITER    = 200

# ─────────────────────────────────────────────────────────────────────
# Naplózás – konzolra + fájlba is ír
# ─────────────────────────────────────────────────────────────────────

log_path = os.path.join(OUTPUT_DIR, "pipeline_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# 1a. LÉPÉS: bun.conf beolvasása (ground-truth transzformációk)
# ─────────────────────────────────────────────────────────────────────

def parse_conf(conf_file: str) -> list:
    """
    Beolvassa a bun.conf fájlt.
    Formátum: bmesh <filename> tx ty tz  qx qy qz qw
    Visszatér: [{'file': str, 'gt_transform': np.ndarray (4x4)}, ...]
    """
    scans = []
    with open(conf_file, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts or parts[0] != "bmesh":
                continue
            fname = parts[1]
            tx, ty, tz       = float(parts[2]), float(parts[3]), float(parts[4])
            qx, qy, qz, qw   = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
            rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rot.T
            T[:3, 3]  = [tx, ty, tz]
            scans.append({"file": fname, "gt_transform": T})
    return scans


# ─────────────────────────────────────────────────────────────────────
# 1b. LÉPÉS: Pontfelhő betöltése & előfeldolgozása
# ─────────────────────────────────────────────────────────────────────

def load_and_preprocess(filepath: str, voxel_size: float) -> o3d.geometry.PointCloud:
    """
    1) Betölti a PLY fájlt (o3d.io.read_point_cloud)
    2) Voxel-grid ritkítás (voxel_down_sample)
    3) Statistical Outlier Removal (SOR) zajszűrés
    """
    log.info(f"  Betöltés: {os.path.basename(filepath)}")
    pcd = o3d.io.read_point_cloud(filepath)
    n_raw = len(pcd.points)
    log.info(f"    Eredeti pontszám      : {n_raw}")

    # Voxel grid ritkítás
    pcd_down = pcd.voxel_down_sample(voxel_size)
    log.info(f"    Voxel ritkítás után   : {len(pcd_down.points)}")

    # Statistical Outlier Removal
    pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    log.info(f"    SOR szűrés után       : {len(pcd_clean.points)}")

    return pcd_clean


# ─────────────────────────────────────────────────────────────────────
# 2. LÉPÉS: Jellemzők kinyerése – normálisok + FPFH
# ─────────────────────────────────────────────────────────────────────

def extract_features(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """
    1) Normálisok PCA-alapon (estimate_normals)
    2) Konzisztens orientáció (orient_normals_consistent_tangent_plane)
    3) FPFH 33-dimenziós hisztogram deszkriptorok (compute_fpfh_feature)
    Visszatér: open3d Feature objektum
    """
    # Normálisok becslése
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)

    # FPFH deszkriptorok
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    log.info(f"    FPFH dims: {fpfh.data.shape[0]}, pontok: {fpfh.data.shape[1]}")
    return fpfh


# ─────────────────────────────────────────────────────────────────────
# 3. LÉPÉS: Durva illesztés – RANSAC
# ─────────────────────────────────────────────────────────────────────

def coarse_registration(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    src_fpfh,
    tgt_fpfh,
    voxel_size: float,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    FPFH jellemzők alapján RANSAC globális regisztráció.
    Cél: megközelítőleg helyes kezdeti transzformáció az ICP-hez.
    """
    dist_thresh = voxel_size * 1.5
    log.info(f"    RANSAC futtatása (max_iter={RANSAC_MAX_ITER}, conf={RANSAC_CONFIDENCE})...")
    t0 = time.time()

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt,
        src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            RANSAC_MAX_ITER, RANSAC_CONFIDENCE
        ),
    )

    elapsed = time.time() - t0
    log.info(
        f"    RANSAC kész ({elapsed:.1f}s) | "
        f"fitness={result.fitness:.4f} | "
        f"RMSE={result.inlier_rmse:.6f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────
# 4. LÉPÉS: Finom illesztés – ICP
# ─────────────────────────────────────────────────────────────────────

def fine_registration(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    mode: str = "point_to_plane",
) -> o3d.pipelines.registration.RegistrationResult:
    """
    ICP finom illesztés.
    mode: 'point_to_point'  - forrás- és célpontok távolságát minimalizálja
          'point_to_plane'  - gyorsabban konvergál sík felületek esetén
    Leállási feltétel: RMSE konvergencia VAGY max iteráció elérése.
    """
    if mode == "point_to_point":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=ICP_MAX_ITER,
    )

    t0 = time.time()
    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        ICP_DIST_THRESH,
        init=init_transform,
        estimation_method=estimation,
        criteria=criteria,
    )
    elapsed = time.time() - t0

    log.info(
        f"    ICP [{mode}] kész ({elapsed:.1f}s) | "
        f"fitness={result.fitness:.4f} | "
        f"RMSE={result.inlier_rmse:.6f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────
# 5. LÉPÉS: Minőségértékelés
# ─────────────────────────────────────────────────────────────────────

def evaluate_registration(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    transform: np.ndarray,
    threshold: float,
    label: str = "",
) -> dict:
    """
    Kiszámítja:
    - fitness  : inlier megfeleltetések aránya (overlap)
    - RMSE     : Root Mean Square Error az inlier pontokon
    - #corr    : inlier megfeleltetések száma
    """
    res = o3d.pipelines.registration.evaluate_registration(
        src, tgt, threshold, transform
    )
    log.info(
        f"    [{label:12s}] fitness={res.fitness:.4f} | "
        f"RMSE={res.inlier_rmse:.6f} | "
        f"#corr={len(res.correspondence_set)}"
    )
    return {
        "label":           label,
        "fitness":         res.fitness,
        "inlier_rmse":     res.inlier_rmse,
        "correspondences": len(res.correspondence_set),
    }


# ─────────────────────────────────────────────────────────────────────
# Vizualizáció segédfüggvények
# ─────────────────────────────────────────────────────────────────────

def plot_registration_quality(metrics: list, title: str, save_path: str) -> None:
    """Matplotlib bar-chart a regisztrációs metrikákról."""
    labels    = [m["label"]       for m in metrics]
    fitness   = [m["fitness"]     for m in metrics]
    rmse_vals = [m["inlier_rmse"] for m in metrics]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
              "#00BCD4", "#FF5722", "#607D8B", "#795548", "#CDDC39"]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, fitness, color=colors[:len(labels)])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Fitness (overlap arány)")
    ax1.set_title("Fitness")
    ax1.set_ylim(0, 1.05)
    for xi, v in zip(x, fitness):
        ax1.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    ax2.bar(x, rmse_vals, color=colors[:len(labels)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Inlier RMSE (m)")
    ax2.set_title("RMSE")
    for xi, v in zip(x, rmse_vals):
        ax2.text(xi, v + max(rmse_vals) * 0.01, f"{v:.4f}", ha="center", fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"    Minőségábra mentve: {save_path}")


def pyvista_render(
    pcd_list: list,
    colors: list,
    title: str,
    save_path: str,
) -> None:
    """Off-screen PyVista 3D renderelés, ha a könyvtár elérhető."""
    if not HAS_PYVISTA:
        log.info("    PyVista nem érhető el - 3D render kihagyva.")
        return
    try:
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("white")
        for pcd, col in zip(pcd_list, colors):
            pts = np.asarray(pcd.points)
            if len(pts) == 0:
                continue
            cloud = pv.PolyData(pts)
            plotter.add_points(cloud, color=col, point_size=2)
        plotter.add_title(title, font_size=11)
        plotter.camera_position = "iso"
        plotter.screenshot(save_path)
        log.info(f"    PyVista kép mentve: {save_path}")
    except Exception as exc:
        log.warning(f"    PyVista render sikertelen: {exc}")


def plot_merged_top_view(merged_pcd: o3d.geometry.PointCloud, save_path: str, title: str) -> None:
    """Felülnézeti szóróábra (X-Z sík) a merged pontfelhőről."""
    pts = np.asarray(merged_pcd.points)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pts[:, 0], pts[:, 2], s=0.3, c=pts[:, 1],
               cmap="viridis", alpha=0.6)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"    Felülnézeti ábra mentve: {save_path}")


# ─────────────────────────────────────────────────────────────────────
# Teljes pipeline egyetlen scan-párra
# ─────────────────────────────────────────────────────────────────────

def register_pair(src_file: str, tgt_file: str) -> dict:
    """
    Teljes regisztrációs pipeline két scan között.
    Visszatér szótárral: ransac result, icp_p2p result, icp_p2plane result,
    metrics lista, legjobb transzformáció, pontfelhők.
    """
    pair_name = (
        f"{os.path.splitext(os.path.basename(src_file))[0]}"
        f"_to_"
        f"{os.path.splitext(os.path.basename(tgt_file))[0]}"
    )

    log.info("\n" + "=" * 65)
    log.info(f"  PÁR REGISZTRÁCIÓ: {pair_name}")
    log.info("=" * 65)

    # ── 1. Betöltés & előfeldolgozás ─────────────────────────────────
    log.info("[1] Betöltés & előfeldolgozás")
    src = load_and_preprocess(src_file, VOXEL_SIZE)
    tgt = load_and_preprocess(tgt_file, VOXEL_SIZE)

    # ── 2. Jellemzők kinyerése ────────────────────────────────────────
    log.info("[2] Jellemzők kinyerése (normálisok + FPFH)")
    src_fpfh = extract_features(src, VOXEL_SIZE)
    tgt_fpfh = extract_features(tgt, VOXEL_SIZE)

    # ── 3. RANSAC ────────────────────────────────────────────────────
    log.info("[3] Durva illesztés - RANSAC")
    ransac_res = coarse_registration(src, tgt, src_fpfh, tgt_fpfh, VOXEL_SIZE)

    # ── 4a. ICP Point-to-Point ───────────────────────────────────────
    log.info("[4a] Finom illesztés - ICP Point-to-Point")
    icp_p2p = fine_registration(
        src, tgt, ransac_res.transformation, mode="point_to_point"
    )

    # ── 4b. ICP Point-to-Plane ───────────────────────────────────────
    log.info("[4b] Finom illesztés - ICP Point-to-Plane")
    icp_p2plane = fine_registration(
        src, tgt, ransac_res.transformation, mode="point_to_plane"
    )

    # ── 5. Minőségértékelés ──────────────────────────────────────────
    log.info("[5] Minőségértékelés")
    metrics = [
        evaluate_registration(src, tgt, ransac_res.transformation, VOXEL_SIZE, "RANSAC"),
        evaluate_registration(src, tgt, icp_p2p.transformation,    VOXEL_SIZE, "ICP P2P"),
        evaluate_registration(src, tgt, icp_p2plane.transformation, VOXEL_SIZE, "ICP P2Plane"),
    ]

    # Legjobb transzformáció kiválasztása (legnagyobb fitness)
    candidates = [
        (ransac_res.transformation,  metrics[0]["fitness"]),
        (icp_p2p.transformation,     metrics[1]["fitness"]),
        (icp_p2plane.transformation, metrics[2]["fitness"]),
    ]
    best_transform = max(candidates, key=lambda x: x[1])[0]

    # Matplotlib minőségábrák
    plot_path = os.path.join(OUTPUT_DIR, f"{pair_name}_quality.png")
    plot_registration_quality(
        metrics, f"Regisztrációs minőség – {pair_name}", plot_path
    )

    # Vizualizáció: illesztett eredmény
    src_aligned = copy.deepcopy(src)
    src_aligned.transform(best_transform)
    src_aligned.paint_uniform_color([0.862, 0.129, 0.129])
    tgt_vis = copy.deepcopy(tgt)
    tgt_vis.paint_uniform_color([0.129, 0.510, 0.863])

    pyvista_render(
        [tgt_vis, src_aligned],
        ["#2196F3", "#F44336"],
        f"Illesztett: {pair_name}",
        os.path.join(OUTPUT_DIR, f"{pair_name}_3d.png"),
    )

    return {
        "pair_name":      pair_name,
        "ransac":         ransac_res,
        "icp_p2p":        icp_p2p,
        "icp_p2plane":    icp_p2plane,
        "metrics":        metrics,
        "best_transform": best_transform,
        "src_pcd":        src,
        "tgt_pcd":        tgt,
    }


# ─────────────────────────────────────────────────────────────────────
# 6. LÉPÉS: Eredmény exportálása
# ─────────────────────────────────────────────────────────────────────

def export_result(
    src: o3d.geometry.PointCloud,
    tgt: o3d.geometry.PointCloud,
    transform: np.ndarray,
    pair_name: str,
) -> str:
    """
    - Transzformálja a source pontfelhőt
    - Összefűzi a target-tel
    - Menti .ply formátumban
    - Naplózza a végső transzformációs mátrixot
    """
    src_t = copy.deepcopy(src)
    src_t.transform(transform)
    merged = src_t + tgt

    out_ply = os.path.join(OUTPUT_DIR, f"{pair_name}_merged.ply")
    o3d.io.write_point_cloud(out_ply, merged)
    log.info(f"  Merged PLY mentve: {out_ply}  ({len(merged.points)} pont)")

    T_str = np.array2string(transform, precision=6, suppress_small=True)
    log.info(f"  Végső transzformációs mátrix [{pair_name}]:\n{T_str}")
    return out_ply


# ─────────────────────────────────────────────────────────────────────
# Teljes multi-scan fúzió (összes bun.conf scan)
# ─────────────────────────────────────────────────────────────────────

def full_reconstruction(scans: list) -> tuple:
    """
    Inkrementális regisztráció:
    - bun000.ply lesz a referencia (target)
    - A bun.conf ground-truth transzformációkat használja kezdeti illesztésként
    - Minden scant ICP-vel finomít a már merged felhőhöz képest
    Visszatér: (merged_pcd, {filename: 4x4 transform})
    """
    log.info("\n" + "=" * 65)
    log.info("  TELJES REKONSTRUKCIÓ - multi-scan fúzió (GT init + ICP refine)")
    log.info("=" * 65)

    ref_file = os.path.join(DATA_DIR, scans[0]["file"])
    log.info(f"[REF] Referencia scan: {scans[0]['file']}")
    merged_pcd = load_and_preprocess(ref_file, VOXEL_SIZE)

    # GT transzformáció alkalmazása a referenciára (szkenner→világ)
    ref_gt = scans[0]["gt_transform"]
    merged_pcd.transform(ref_gt)

    # Normálisok a referencián (P2Plane ICP-hez)
    merged_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30)
    )
    merged_pcd.orient_normals_consistent_tangent_plane(k=30)

    cumulative_transforms = {scans[0]["file"]: ref_gt.tolist()}
    all_metrics = []

    for scan in scans[1:]:
        src_file = os.path.join(DATA_DIR, scan["file"])
        log.info(f"\n--- Regisztrálás: {scan['file']} → merged felhő ---")

        src = load_and_preprocess(src_file, VOXEL_SIZE)

        # GT transzformáció (szkenner→világ) mint kezdő becslés
        gt_T = scan["gt_transform"]
        log.info(f"    GT transzformáció alkalmazása kezdő becslésként")

        # Normálisok a source-on
        src.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30)
        )
        src.orient_normals_consistent_tangent_plane(k=30)

        # GT-vel transzformált source értékelése
        m_gt = evaluate_registration(src, merged_pcd, gt_T, VOXEL_SIZE, f"{scan['file']} GT")

        # ICP Point-to-Plane finomítás a GT-ből indulva
        icp_res = fine_registration(
            src, merged_pcd, gt_T, mode="point_to_plane"
        )

        m_icp = evaluate_registration(src, merged_pcd, icp_res.transformation, VOXEL_SIZE, f"{scan['file']} ICP")

        # A jobbik választása (GT vs ICP-refined)
        if m_icp["fitness"] >= m_gt["fitness"]:
            best_T = icp_res.transformation
            log.info(f"    → ICP finomított transzformáció használata (fitness={m_icp['fitness']:.4f})")
        else:
            best_T = gt_T
            log.info(f"    → GT transzformáció használata (fitness={m_gt['fitness']:.4f})")

        cumulative_transforms[scan["file"]] = best_T.tolist()
        all_metrics.append(m_icp if m_icp["fitness"] >= m_gt["fitness"] else m_gt)

        # Transzformált source fúziója a merged felhőbe
        src_t = copy.deepcopy(src)
        src_t.transform(best_T)
        merged_pcd = merged_pcd + src_t

        # Ritkítás, hogy kezelhető maradjon
        merged_pcd = merged_pcd.voxel_down_sample(VOXEL_SIZE)
        # Normálisok frissítése a következő iterációhoz
        merged_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 2, max_nn=30)
        )
        merged_pcd.orient_normals_consistent_tangent_plane(k=30)

        log.info(f"    Merged felhő mérete: {len(merged_pcd.points)} pont")

    # ── Teljes modell mentése ────────────────────────────────────────
    out_ply = os.path.join(OUTPUT_DIR, "bunny_full_reconstruction.ply")
    o3d.io.write_point_cloud(out_ply, merged_pcd)
    log.info(f"\nTeljes rekonstrukció mentve: {out_ply}  ({len(merged_pcd.points)} pont)")


    # ── Transzformációs mátrixok mentése JSON-ba ─────────────────────
    transforms_json = os.path.join(OUTPUT_DIR, "transforms.json")
    with open(transforms_json, "w", encoding="utf-8") as fh:
        json.dump(cumulative_transforms, fh, indent=2)
    log.info(f"Transzformációk JSON: {transforms_json}")

    # ── Mátrixok konzolra/logba ──────────────────────────────────────
    log.info("\n── Végső transzformációs mátrixok ──────────────────────")
    for fname, T_list in cumulative_transforms.items():
        T = np.array(T_list)
        log.info(f"\n  {fname}:\n{np.array2string(T, precision=6, suppress_small=True)}")

    return merged_pcd, cumulative_transforms


# ─────────────────────────────────────────────────────────────────────
# FŐPROGRAM
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Pontfelhő regisztrációs pipeline indul...")
    log.info(f"  Adatkönyvtár   : {DATA_DIR}")
    log.info(f"  Kimenet        : {OUTPUT_DIR}")
    log.info(f"  Voxel méret    : {VOXEL_SIZE} m")
    log.info(f"  PyVista elérh. : {HAS_PYVISTA}")

    # bun.conf beolvasása
    scans = parse_conf(CONF_FILE)
    log.info(f"  Scan-ek száma  : {len(scans)}")
    for s in scans:
        log.info(f"    {s['file']}")

    # ─────────────────────────────────────────────────────────────────
    # Teljes multi-scan rekonstrukció (mind a 10 scan)
    # ─────────────────────────────────────────────────────────────────
    full_reconstruction(scans)

    log.info("\n" + "=" * 65)
    log.info("  PIPELINE KÉSZ")
    log.info(f"  Kimeneti fájlok: {OUTPUT_DIR}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
