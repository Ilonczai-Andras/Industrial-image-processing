"""
Interaktív 3D pontfelhő megjelenítő
- Fájlválasztó a .ply fájlok közül (checkbox)
- Forgatható, zoomolható 3D nézet
- PyVista alapú interaktív ablak
"""

import os
import glob
import numpy as np
import open3d as o3d
import pyvista as pv
from pyvista import themes

# ── Konfiguráció ──────────────────────────────────────────────────────
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(WORKSPACE_DIR, "input", "data")
OUTPUT_DIR    = os.path.join(WORKSPACE_DIR, "output")

COLORS = [
    "#E74C3C",  # piros
    "#3498DB",  # kék
    "#2ECC71",  # zöld
    "#F39C12",  # narancs
    "#9B59B6",  # lila
    "#1ABC9C",  # türkiz
    "#E67E22",  # sötét narancs
    "#34495E",  # szürke-kék
]

# ── PLY fájlok összegyűjtése ──────────────────────────────────────────

def find_ply_files() -> dict:
    """Visszaad egy {név: útvonal} szótárt az összes .ply fájlról."""
    files = {}

    # Input data könyvtár
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.ply"), recursive=True)):
        name = f"[scan] {os.path.basename(p)}"
        files[name] = p

    # Output könyvtár (merged/reconstructed)
    for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.ply"))):
        name = f"[output] {os.path.basename(p)}"
        files[name] = p

    return files


# ── Betöltés ──────────────────────────────────────────────────────────

def load_pcd(filepath: str) -> o3d.geometry.PointCloud | None:
    pcd = o3d.io.read_point_cloud(filepath)
    if len(pcd.points) == 0:
        return None
    pts = np.asarray(pcd.points)
    # Korrupt pontok szűrése
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts) < 1e6).all(axis=1)
    pcd = pcd.select_by_index(np.where(valid)[0])
    return pcd if len(pcd.points) > 0 else None


# ── Interaktív megjelenítő ────────────────────────────────────────────

def launch_viewer(selected_files) -> None:
    """PyVista interaktív ablak – forgatható, zoomolható."""
    pl = pv.Plotter(title="3D Pontfelhő Nézegető")
    pl.set_background(color="#1a1a2e")

    loaded_any = False
    legend_entries = []

    for i, (name, path) in enumerate(selected_files.items()):
        pcd = load_pcd(path)
        if pcd is None:
            print(f"  ⚠ Nem sikerült betölteni: {name}")
            continue

        pts = np.asarray(pcd.points)
        color = COLORS[i % len(COLORS)]

        cloud = pv.PolyData(pts)
        pl.add_points(
            cloud,
            color=color,
            point_size=3,
            render_points_as_spheres=True,
            label=name,
        )

        legend_entries.append([name, color])
        print(f"  ✓ Betöltve: {name}  ({len(pts)} pont)  {color}")
        loaded_any = True

    if not loaded_any:
        print("Nincs betöltött fájl!")
        return

    if legend_entries:
        pl.add_legend(labels=legend_entries, bcolor="#00000088", border=True, size=(0.3, 0.3))

    # Vezérlési útmutató
    pl.add_text(
        "Bal egér: forgatás  |  Jobb egér / scroll: zoom  |  Középső: mozgatás  |  R: reset  |  Q: kilépés",
        position="lower_left",
        font_size=9,
        color="white",
    )

    pl.camera.zoom(0.8)
    pl.reset_camera()
    pl.show()


# ── Fájlválasztó (terminál alapú) ─────────────────────────────────────

def file_selector(files: dict[str, str]) -> dict[str, str]:
    """Terminál-alapú checkbox fájlválasztó."""
    names = list(files.keys())

    print("\n" + "=" * 60)
    print("  3D PONTFELHŐ NÉZEGETŐ – Fájlválasztó")
    print("=" * 60)
    print("  Elérhető .ply fájlok:\n")

    for i, name in enumerate(names):
        print(f"  [{i+1:2d}]  {name}")

    print("\n  Lehetőségek:")
    print("    • Számok vesszővel: pl. 1,3,5")
    print("    • Tartomány: pl. 1-4")
    print("    • Összes: all  (vagy enter)")
    print("    • Output merged: m")
    print()

    raw = input("  Választás: ").strip().lower()

    if raw == "" or raw == "all":
        selected_names = names

    elif raw == "m":
        selected_names = [n for n in names if "[output]" in n]

    else:
        selected_names = []
        for part in raw.replace(" ", "").split(","):
            if "-" in part:
                try:
                    a, b = part.split("-")
                    for idx in range(int(a) - 1, int(b)):
                        if 0 <= idx < len(names):
                            selected_names.append(names[idx])
                except ValueError:
                    pass
            else:
                try:
                    idx = int(part) - 1
                    if 0 <= idx < len(names):
                        selected_names.append(names[idx])
                except ValueError:
                    pass

    if not selected_names:
        print("  Nincs érvényes választás, összes fájl betöltve.")
        selected_names = names

    print(f"\n  Kiválasztva ({len(selected_names)} db):")
    for n in selected_names:
        print(f"    • {n}")
    print()

    return {n: files[n] for n in selected_names}


# ── Főprogram ─────────────────────────────────────────────────────────

def main():
    files = find_ply_files()

    if not files:
        print(f"Nem található .ply fájl sem a '{DATA_DIR}', sem a '{OUTPUT_DIR}' mappában.")
        return

    selected = file_selector(files)
    launch_viewer(selected)


if __name__ == "__main__":
    main()