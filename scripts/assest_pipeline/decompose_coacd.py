"""
decompose_coacd.py

Standalone convex decomposition utility using the CoACD python library directly.

Use this when you already have a single mesh file (STL or OBJ, visual concerns
aside) and just need MuJoCo-ready convex collision parts out of it -- no
material/texture splitting, no MJCF generation. For a full textured-OBJ ->
MJCF pipeline (mesh + material + texture + XML), use import_obj_mjcf.py instead,
which wraps obj2mjcf (itself built on CoACD) and additionally handles mesh
recentering and mass/inertia computation.

WHAT THIS SCRIPT DOES
  1. Loads the input mesh with trimesh.
  2. If not watertight, repairs it with pymeshfix (this only removes small
     local defects like duplicate/non-manifold faces -- it does NOT fill in
     large legitimate concave features such as a hollow bore; CoACD needs a
     closed manifold to decompose reliably, but a real hole through the part
     stays a real hole).
  3. Runs CoACD to split the (possibly non-convex) mesh into N convex parts.
  4. Takes the exact convex hull of each part (tightens triangle count,
     guarantees exact convexity for MuJoCo's collision geoms).
  5. Exports each part as its own STL file, ready to be referenced as a
     separate <mesh>/<geom> pair in MJCF.

WHY MULTIPLE PARTS: MuJoCo's compiler only auto-generates a SINGLE convex
hull per mesh geom (the `convexhull` compiler attribute) -- it does not do
multi-part decomposition natively. A single hull fills in every concave
feature (lug gaps, bore holes, etc.). This script produces the multi-part
decomposition MuJoCo doesn't do on its own, so those features survive into
the collision geometry.

USAGE
  python decompose_coacd.py --input part.stl --output-dir out/collision_parts \
      --threshold 0.04 --max-hulls 12

  # Skip repair if you already know the mesh is watertight:
  python decompose_coacd.py --input part.stl --output-dir out/collision_parts --no-repair

REQUIRES
  pip install coacd trimesh pymeshfix
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import trimesh


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Repair small non-manifold defects (duplicate/degenerate faces) via
    pymeshfix, without collapsing genuine concave features or holes."""
    import pymeshfix

    mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    mf.repair(remove_smallest_components=False)
    repaired = trimesh.Trimesh(vertices=mf.points, faces=mf.faces)
    return repaired


def decompose(
    input_path: Path,
    output_dir: Path,
    threshold: float = 0.04,
    max_hulls: int = 12,
    do_repair: bool = True,
    seed: int = 0,
) -> list[Path]:
    """Run CoACD decomposition and export each convex part as an STL.

    Returns the list of output file paths, in part order (part00, part01, ...).
    """
    import coacd

    mesh = trimesh.load(input_path, force="mesh")
    print(f"Loaded {input_path.name}: {len(mesh.faces)} faces, "
          f"watertight={mesh.is_watertight}")

    if do_repair and not mesh.is_watertight:
        print("Mesh is not watertight -- repairing with pymeshfix "
              "(fixes local defects only, does not fill real holes)...")
        mesh = repair_mesh(mesh)
        print(f"After repair: {len(mesh.faces)} faces, "
              f"watertight={mesh.is_watertight}")
        if not mesh.is_watertight:
            print("WARNING: mesh still not watertight after repair. "
                  "CoACD may still work but results could be unreliable.",
                  file=sys.stderr)

    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_hulls,
        seed=seed,
    )
    print(f"CoACD produced {len(parts)} convex parts")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for i, (verts, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        hull = part_mesh.convex_hull  # exact convex hull: tightest triangle
                                       # count, guaranteed convex
        out_path = output_dir / f"part{i:02d}.stl"
        hull.export(out_path)
        out_paths.append(out_path)
        print(f"  part{i:02d}.stl: {len(hull.faces)} faces")

    return out_paths


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                    help="Input mesh file (STL or OBJ)")
    p.add_argument("--output-dir", required=True, type=Path,
                    help="Directory to write partNN.stl files into")
    p.add_argument("--threshold", type=float, default=0.04,
                    help="CoACD concavity threshold, 0.01-1 (default 0.04; "
                         "lower = more parts = tighter fit)")
    p.add_argument("--max-hulls", type=int, default=12,
                    help="Max number of convex parts (default 12; -1 for no limit)")
    p.add_argument("--no-repair", action="store_true",
                    help="Skip pymeshfix repair even if mesh is not watertight")
    p.add_argument("--seed", type=int, default=0,
                    help="Random seed for CoACD sampling (default 0)")
    args = p.parse_args()

    decompose(
        input_path=args.input,
        output_dir=args.output_dir,
        threshold=args.threshold,
        max_hulls=args.max_hulls,
        do_repair=not args.no_repair,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
