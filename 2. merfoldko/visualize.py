"""
Interaktív 3D megjelenítő – Open3D visualizer
Irányítás:
  Egér bal gomb  : forgatás
  Egér jobb gomb : közelítés / távolítás
  Egér közép     : eltolás
  Q / Esc        : kilépés
  R              : nézet visszaállítása
  [, ]           : pontméret csökkentése / növelése
"""

import sys
import os
import open3d as o3d

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(WORKSPACE_DIR, "input", "bunny", "data")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output")
OUTPUT_PLY = os.path.join(OUTPUT_DIR, "bunny_full_reconstruction.ply")


def build_file_list():
    """Összeállítja a választható fájlok listáját: input .ply-ok + output rekonstrukció."""
    files = {}
    idx = 1

    # Input .ply fájlok
    if os.path.isdir(INPUT_DIR):
        for fname in sorted(os.listdir(INPUT_DIR)):
            if fname.lower().endswith(".ply"):
                files[str(idx)] = (
                    f"[Input]  {fname}",
                    os.path.join(INPUT_DIR, fname),
                )
                idx += 1

    # Output teljes rekonstrukció
    files[str(idx)] = (
        "[Output] bunny_full_reconstruction.ply",
        OUTPUT_PLY,
    )

    return files


def colorize_by_height(pcd):
    """Magasság szerint színezi a pontfelhőt (kék→zöld→piros)."""
    points = pcd.get_max_bound()
    min_z = pcd.get_min_bound()[1]
    max_z = pcd.get_max_bound()[1]
    import numpy as np
    pts = np.asarray(pcd.points)
    if max_z - min_z < 1e-6:
        pcd.paint_uniform_color([0.3, 0.6, 1.0])
        return pcd
    t = (pts[:, 1] - min_z) / (max_z - min_z) 
    colors = np.zeros((len(pts), 3))

    mask1 = t < 0.5
    colors[mask1, 0] = 0
    colors[mask1, 1] = t[mask1] * 2
    colors[mask1, 2] = 1 - t[mask1] * 2

    mask2 = ~mask1
    colors[mask2, 0] = (t[mask2] - 0.5) * 2
    colors[mask2, 1] = 1 - (t[mask2] - 0.5) * 2
    colors[mask2, 2] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def reconstruct_mesh(pcd):
    """
    Poisson felület-rekonstrukció a pontfelhőből.
    Visszatér: TriangleMesh (magasság szerint színezve).
    """
    import numpy as np

    print("  Normálisok becslése...")
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)

    print("  Poisson felület-rekonstrukció (depth=9)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )

    # Alacsony sűrűségű (zajosabb) háromszögek levágása
    import numpy as np
    densities = np.asarray(densities)
    thresh = np.percentile(densities, 5)
    verts_to_remove = densities < thresh
    mesh.remove_vertices_by_mask(verts_to_remove)

    # Magasság szerint színezés
    pts = np.asarray(mesh.vertices)
    min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
    if max_y - min_y > 1e-6:
        t = (pts[:, 1] - min_y) / (max_y - min_y)
    else:
        t = np.zeros(len(pts))
    colors = np.zeros((len(pts), 3))
    mask = t < 0.5
    colors[mask,  1] = t[mask] * 2
    colors[mask,  2] = 1 - t[mask] * 2
    colors[~mask, 0] = (t[~mask] - 0.5) * 2
    colors[~mask, 1] = 1 - (t[~mask] - 0.5) * 2
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    mesh.compute_vertex_normals()
    print(f"  Mesh: {len(mesh.vertices)} csúcs, {len(mesh.triangles)} háromszög")
    return mesh


def show(filepath, title):
    if not os.path.exists(filepath):
        print(f"[HIBA] Fájl nem található: {filepath}")
        print("Futtasd előbb a registration_pipeline.py-t!")
        return

    print(f"\nBetöltés: {filepath}")
    pcd = o3d.io.read_point_cloud(filepath)
    print(f"Pontok száma: {len(pcd.points)}")

    import numpy as np

    is_output = os.path.abspath(filepath) == os.path.abspath(OUTPUT_PLY)

    bbox = pcd.get_axis_aligned_bounding_box()
    axis_len = max(bbox.get_max_bound() - bbox.get_min_bound()) * 0.2
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len)

    print("\nInteraktív ablak nyílik – irányítás:")
    print("  Bal egér   : forgatás")
    print("  Jobb egér  : zoom")
    print("  Közép/scroll: eltolás / zoom")
    print("  Q / Esc    : bezárás")
    print("  R          : nézet visszaállítása")
    print("  [  /  ]    : pont méret  –  /  +\n")

    if is_output:
        print("Felület-rekonstrukció folyamatban (output fájl)...")
        geom = reconstruct_mesh(pcd)
    else:
        colors = np.asarray(pcd.colors)
        if len(colors) == 0 or colors.max() < 1e-6:
            pcd = colorize_by_height(pcd)
        geom = pcd

    o3d.visualization.draw_geometries(
        [geom, axes],
        window_name=title,
        width=1280,
        height=800,
        mesh_show_back_face=True,
    )


def main():
    print("=" * 50)
    print("  Stanford Bunny - 3D interaktív megjelenítő")
    print("=" * 50)

    FILES = build_file_list()

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\nVálassz fájlt:")
        for k, (name, _) in FILES.items():
            print(f"  {k} – {name}")
        choice = input(f"\nSzám (Enter = {max(FILES, key=int)}): ").strip() or max(FILES, key=int)

    if choice not in FILES:
        print(f"Érvénytelen választás: {choice}")
        sys.exit(1)

    title, filepath = FILES[choice]
    show(filepath, title)


if __name__ == "__main__":
    main()
