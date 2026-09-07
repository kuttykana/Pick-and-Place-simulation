"""
import_obj_mjcf.py

Full pipeline: textured OBJ (+ MTL + texture image) -> MuJoCo-ready assets,
dropped directly into your scenes/meshes/<name>/ folder with a ready-to-
<include> MJCF fragment.

Wraps the `obj2mjcf` CLI (which uses CoACD internally for convex
decomposition) and adds several steps obj2mjcf does NOT do on its own,
all learned the hard way while setting up the watch case asset:

  1. RECENTERING. obj2mjcf preserves whatever coordinate origin the source
     OBJ happened to be authored with -- which is very often NOT the
     geometric center. If left alone, this shows up as an unexplained
     offset between where you place the body and where the mesh actually
     renders. This script recenters every output mesh (visual + all collision parts) to
     the true volumetric center of mass, so body pos == mesh center and
     <inertial pos="0 0 0"> is actually correct.

  2. RIGOROUS MASS/INERTIA. Rather than approximate with primitive-shape
     formulas, this script integrates mass and the full inertia tensor
     directly over the real (repaired, watertight) collision-source mesh
     at real-world scale, using a supplied material density. It reports
     off-diagonal inertia terms so you can confirm the principal axes are
     aligned with the body frame (if not, you'll need an inertial quat --
     this script will warn you).

  3. TEXTURE FORMAT. MuJoCo's native texture loader wants PNG, not JPG --
     this script auto-converts if needed.

  4. Generates a <compiler meshdir="..." texturedir="..."/>-compatible
     folder layout and prints the exact <include> line + body scaffold to
     drop into your main scene XML.

USAGE
  python import_obj_mjcf.py \
      --input-obj path/to/part.obj \
      --name my_part \
      --meshes-dir ~/watch-pickplace-sim/scenes/meshes \
      --scale 0.048 \
      --density 7980 \ # stainless steel 316L
      --threshold 0.04 --max-hulls 12

  Then in your scene XML:
    <compiler meshdir="meshes/" texturedir="meshes/"/>
    ...
    <include file="meshes/my_part/my_part.xml"/>
    ...
    <worldbody>
      <body name="my_part" pos="0.25 0 0.02">
        <freejoint/>
        <inertial pos="0 0 0" mass="..." diaginertia="... ... ..."/>
        <!-- geoms come from the included file's body -- see note below -->
      </body>
    </worldbody>

  NOTE ON INCLUDE STRUCTURE: obj2mjcf's generated XML wraps geoms in their
  own <body name="{name}">. MuJoCo does not let you merge two <body> tags,
  so the include's body IS your object body -- this script rewrites that
  body element in place to add pos/freejoint/inertial placeholders (see
  --body-pos, --add-freejoint) so you don't have to hand-edit it.

REQUIRES
  pip install obj2mjcf coacd trimesh pymeshfix pillow
  (obj2mjcf must be runnable as a CLI: `obj2mjcf --help` should work)
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


def convert_texture_to_png(mtl_path: Path, obj_dir: Path) -> None:
    """MuJoCo's native texture loader only accepts PNG. Find any map_Kd
    referencing a non-PNG image, convert it, and rewrite the .mtl."""
    from PIL import Image

    if not mtl_path.exists():
        return

    text = mtl_path.read_text()
    changed = False
    for match in re.finditer(r"^map_Kd\s+(\S+)", text, flags=re.MULTILINE):
        tex_name = match.group(1)
        tex_path = obj_dir / tex_name
        if tex_path.suffix.lower() != ".png" and tex_path.exists():
            png_path = tex_path.with_suffix(".png")
            print(f"Converting texture {tex_path.name} -> {png_path.name}")
            Image.open(tex_path).convert("RGB").save(png_path)
            text = text.replace(tex_name, png_path.name)
            changed = True

    if changed:
        mtl_path.write_text(text)


def find_referenced_file(directory: Path, referenced_name: str,
                          allowed_extensions: set[str] | None = None) -> Path | None:
    """Resolve a filename referenced inside an OBJ/MTL file against what's
    actually on disk. Some upload/transfer paths sanitize filenames (e.g.
    '+' -> '_', spaces -> '_') without updating the referencing file's
    internal text, so an exact match can fail even though the file is
    right there under a slightly different name.

    Tries, in order: exact match, '+'/' '->'_' substitution, a
    case-insensitive scan for a normalized full-name match, then a
    normalized STEM-only match (for files that were converted to a
    different format, e.g. .jpg -> .png, since the referencing file was
    written).

    `allowed_extensions`, if given (e.g. {'.mtl'} or {'.png','.jpg'}),
    restricts every fallback tier to files with one of those suffixes --
    without this, a stem-only match can accidentally hit a completely
    unrelated file that happens to share a normalized name (e.g. matching
    an archive .zip when looking for a .mtl). Always pass this.
    """
    def ext_ok(f: Path) -> bool:
        return allowed_extensions is None or f.suffix.lower() in allowed_extensions

    exact = directory / referenced_name
    if exact.exists() and ext_ok(exact):
        return exact

    sanitized = referenced_name.replace("+", "_").replace(" ", "_")
    candidate = directory / sanitized
    if candidate.exists() and ext_ok(candidate):
        return candidate

    def normalize(s: str) -> str:
        return re.sub(r"[+_ ]", "", s).lower()

    target_norm = normalize(referenced_name)
    for f in directory.iterdir():
        if f.is_file() and ext_ok(f) and normalize(f.name) == target_norm:
            return f

    # Extension-agnostic (but still allowed_extensions-constrained) stem
    # match: the referenced file may have already been converted to a
    # different format since the .mtl was written.
    target_stem_norm = normalize(Path(referenced_name).stem)
    for f in directory.iterdir():
        if f.is_file() and ext_ok(f) and normalize(f.stem) == target_stem_norm:
            return f

    return None


def clean_and_repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Prepare a mesh for reliable volume/mass integration.

    Textured OBJs frequently have many disconnected "components" that are
    not real gaps in the geometry -- they're UV-chart seam boundaries,
    where the same 3D position gets a separate vertex per chart because
    each has different UV coordinates. Naive connectivity checks (and
    pymeshfix's component-based repair) see these as real fragmentation.
    Fix: weld vertices by POSITION only (ignore UV/normal differences)
    before doing anything else. Then isolate the dominant connected
    component (real models sometimes carry stray degenerate sliver
    triangles from export/reconstruction noise -- negligible in size but
    they can confuse repair otherwise), and finally run pymeshfix to close
    any genuinely small local defects (non-manifold edges etc.) without
    filling in real concave/hollow features.
    """
    import pymeshfix

    m = mesh.copy()
    m.merge_vertices(merge_tex=True, merge_norm=True)

    components = m.split(only_watertight=False)
    if len(components) > 1:
        components_sorted = sorted(components, key=lambda c: len(c.faces),
                                    reverse=True)
        main = components_sorted[0]
        frac = len(main.faces) / len(m.faces)
        print(f"Mesh split into {len(components)} components after "
              f"position-welding; using the dominant one "
              f"({frac*100:.2f}% of faces) and dropping the rest as "
              f"export noise.")
        m = main

    if not m.is_watertight:
        mf = pymeshfix.MeshFix(m.vertices, m.faces)
        mf.repair(remove_smallest_components=False)
        m = trimesh.Trimesh(vertices=mf.points, faces=mf.faces)

    return m


def compute_center_and_inertia(collision_source: Path, scale: float, density: float):
    """Repair the mesh if needed, then integrate mass/COM/inertia at real
    scale. Returns (center_raw, mass_kg, diaginertia, off_diag_ratio)."""
    mesh = trimesh.load(collision_source, force="mesh")
    mesh = clean_and_repair_mesh(mesh)
    if not mesh.is_watertight:
        print("WARNING: still not watertight after cleanup; mass/inertia "
              "may be inaccurate.", file=sys.stderr)

    center_raw = mesh.center_mass.copy()  # in raw (unscaled) mesh units

    mesh_real = mesh.copy()
    mesh_real.apply_scale(scale)
    mesh_real.density = density

    mass = mesh_real.mass
    I = mesh_real.moment_inertia
    diag = np.diag(I)
    off_diag = np.array([I[0, 1], I[0, 2], I[1, 2]])
    off_diag_ratio = np.abs(off_diag).max() / np.abs(diag).max()

    return center_raw, mass, diag, off_diag_ratio


def recenter_mesh_file(path: Path, center_raw: np.ndarray) -> None:
    """Translate a mesh file's vertices in place so `center_raw` becomes
    the new origin.

    For .obj files this edits ONLY the "v x y z" vertex lines directly in
    the raw text, leaving mtllib/usemtl/vt/vn/f lines completely
    untouched. This is deliberate: an earlier version of this function
    round-tripped through trimesh's load-shift-export, which sometimes
    silently dropped the real material/texture reference and replaced it
    with a generic "material_0" placeholder (a real bug we hit in
    practice -- the visual mesh came out with the correct texture on some
    runs and a blank placeholder on others, non-deterministically,
    because of how trimesh's OBJ exporter handles material metadata
    during that round trip). Direct text editing has zero risk of losing
    metadata because it never touches any line other than "v ...".

    Non-OBJ files (STL) fall back to the trimesh path since STL carries
    no material data to lose in the first place.
    """
    if path.suffix.lower() == ".obj":
        lines = path.read_text(errors="ignore").splitlines(keepends=True)
        out = []
        for line in lines:
            if line.startswith("v ") or line.startswith("v\t"):
                parts = line.split()
                # "v x y z" (ignore optional w / vertex-color trailing fields
                # by leaving anything past index 3 untouched)
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                x = float(x - center_raw[0])
                y = float(y - center_raw[1])
                z = float(z - center_raw[2])
                rest = parts[4:]
                newline = f"v {x:.9f} {y:.9f} {z:.9f}"
                if rest:
                    newline += " " + " ".join(rest)
                newline += "\n"
                out.append(newline)
            else:
                out.append(line)
        path.write_text("".join(out))
    else:
        mesh = trimesh.load(path, force="mesh", process=False)
        mesh.vertices -= center_raw
        mesh.export(path)


def run_pipeline(
    input_obj: Path,
    name: str,
    meshes_dir: Path,
    scale: float,
    density: float,
    threshold: float,
    max_hulls: int,
    body_pos: str,
    specular: float = 0.9,
    shininess: float = 0.85,
    reflectance: float = 0.4,
):
    work_dir = meshes_dir / f"_staging_{name}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    # Stage input obj + its mtl + textures alongside it (obj2mjcf needs
    # everything in one --obj-dir). IMPORTANT: don't guess the .mtl filename
    # from the .obj's own name -- many exporters (e.g. AI-reconstruction
    # tools) give the .mtl an unrelated filename. Parse the actual
    # `mtllib` directive out of the .obj file instead.
    stage_input = work_dir / "input"
    stage_input.mkdir()

    obj_text = input_obj.read_text(errors="ignore")
    mtllib_match = re.search(r"^mtllib\s+(.+)$", obj_text, flags=re.MULTILINE)

    # We rewrite the staged obj's mtllib line to a clean, predictable name
    # regardless of what the source used, to sidestep filenames obj2mjcf
    # or the filesystem might choke on (e.g. spaces, '+' characters).
    clean_mtl_name = f"{input_obj.stem}.mtl"
    src_mtl_path = None
    if mtllib_match:
        src_mtl_name = mtllib_match.group(1).strip()
        src_mtl_path = find_referenced_file(input_obj.parent, src_mtl_name,
                                             allowed_extensions={".mtl"})
        if src_mtl_path is None:
            # Last-resort fallback: try the obj's own stem as a guess.
            fallback_path = input_obj.with_suffix(".mtl")
            if fallback_path.exists():
                print(f"mtllib reference '{src_mtl_name}' not found on disk; "
                      f"using '{fallback_path.name}' instead (matches the "
                      f".obj's own filename).")
                src_mtl_path = fallback_path
            else:
                print(f"WARNING: mtllib references '{src_mtl_name}' but no "
                      f"matching file was found next to {input_obj.name}. "
                      f"Visual will likely be untextured.", file=sys.stderr)
                shutil.copy(input_obj, stage_input / input_obj.name)
        if src_mtl_path is not None:
            print(f"Found MTL: {src_mtl_path.name}")
            obj_text = obj_text.replace(
                mtllib_match.group(0), f"mtllib {clean_mtl_name}"
            )
            (stage_input / input_obj.name).write_text(obj_text)

            # Resolve + copy every referenced texture, and rewrite the
            # mtl's map_Kd line(s) to match wherever it actually landed --
            # copying the mtl unchanged would leave it pointing at a
            # filename that may not exist on disk under that exact name.
            mtl_text = src_mtl_path.read_text(errors="ignore")
            for match in re.finditer(r"^map_Kd\s+(\S+)", mtl_text,
                                      flags=re.MULTILINE):
                tex_name = match.group(1).strip()
                tex_src = find_referenced_file(
                    input_obj.parent, tex_name,
                    allowed_extensions={".png", ".jpg", ".jpeg", ".bmp",
                                         ".tga", ".tif", ".tiff"},
                )
                if tex_src is not None:
                    shutil.copy(tex_src, stage_input / tex_src.name)
                    if tex_src.name != tex_name:
                        print(f"Texture reference '{tex_name}' resolved to "
                              f"actual file '{tex_src.name}'.")
                        mtl_text = mtl_text.replace(tex_name, tex_src.name)
                else:
                    print(f"WARNING: texture '{tex_name}' referenced "
                          f"but not found next to the source OBJ (tried "
                          f"exact name and '+'/'_' sanitization variants).",
                          file=sys.stderr)
            (stage_input / clean_mtl_name).write_text(mtl_text)
    else:
        print("No mtllib directive found in OBJ -- proceeding untextured.")
        shutil.copy(input_obj, stage_input / input_obj.name)

    # Fix texture format BEFORE running obj2mjcf so it picks up the PNG
    staged_mtl = stage_input / clean_mtl_name
    if staged_mtl.exists():
        convert_texture_to_png(staged_mtl, stage_input)

    # Run obj2mjcf (uses CoACD internally for decompose)
    cmd = [
        "obj2mjcf",
        "--obj-dir", str(stage_input),
        "--save-mjcf",
        "--verbose",
        "--decompose",
        f"--coacd-args.threshold={threshold}",
        f"--coacd-args.max-convex-hull={max_hulls}",
        "--overwrite",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("obj2mjcf failed -- see output above.", file=sys.stderr)
        sys.exit(1)

    gen_dir = stage_input / input_obj.stem
    if not gen_dir.exists():
        print(f"Expected output dir {gen_dir} not found.", file=sys.stderr)
        sys.exit(1)

    # --- Recentering ---
    # Use the FIRST collision part set combined (or just the visual mesh's
    # own watertight repair) to find the true center of mass. Simplest
    # reliable source: reload all collision_N.obj parts, take their
    # combined convex hull as a stand-in solid, OR better -- reuse the
    # original (pre-decomposition) mesh copy trimesh loaded internally.
    # We recompute from the visual mesh directly since it's the full,
    # single-piece geometry.
    visual_obj = gen_dir / f"{input_obj.stem}.obj"
    center_raw, mass, diag, off_ratio = compute_center_and_inertia(
        visual_obj, scale, density
    )

    print(f"\nTrue center of mass (raw mesh units): {center_raw}")
    print(f"Mass: {mass*1000:.3f} g")
    print(f"Diagonal inertia (kg*m^2): {diag}")
    if off_ratio > 0.01:
        print(f"WARNING: off-diagonal inertia terms are {off_ratio*100:.1f}% "
              f"of the diagonal -- principal axes are NOT aligned with the "
              f"body frame. You'll need an <inertial quat=\"...\"/> for "
              f"correctness; this script does not compute that "
              f"automatically.", file=sys.stderr)
    else:
        print("Off-diagonal inertia terms negligible -- axis-aligned "
              "diaginertia is valid, no inertial quat needed.")

    # Recenter every generated mesh file (visual + all collision_N.obj)
    for obj_file in gen_dir.glob("*.obj"):
        recenter_mesh_file(obj_file, center_raw)
    print(f"Recentered {len(list(gen_dir.glob('*.obj')))} mesh files.")

    # --- Move into final location ---
    final_dir = meshes_dir / name
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.move(str(gen_dir), str(final_dir))
    shutil.rmtree(work_dir)

    # --- Rewrite the generated XML: add mesh scale, fix body (pos/freejoint/inertial) ---
    xml_path = final_dir / f"{input_obj.stem}.xml"
    xml_text = xml_path.read_text()

    # add scale="..." to every <mesh file="..."/> that doesn't have one
    xml_text = re.sub(
        r'(<mesh file="[^"]+")\s*/>',
        rf'\1 scale="{scale} {scale} {scale}"/>',
        xml_text,
    )

    diag_str = " ".join(f"{v:.6e}" for v in diag)

    # obj2mjcf always names its default classes "visual"/"collision" -- if
    # any other included file (very commonly a robot model) already
    # defines classes with those exact names, MuJoCo raises a hard
    # "repeated default class name" compile error (classes can't be
    # merged like <asset>/<worldbody> content can). Prefix with the
    # object name so classes stay unique across every included asset.
    xml_text = re.sub(r'class="visual"', f'class="{name}_visual"', xml_text)
    xml_text = re.sub(r'class="collision"', f'class="{name}_collision"', xml_text)

    # Apply material appearance (specular/shininess/reflectance). obj2mjcf's
    # default material values are generic (specular=0.5, shininess=0.5, no
    # reflectance) -- not tuned for any particular surface finish. This
    # rewrites whatever <material .../> obj2mjcf generated to use the
    # requested values instead, so the look survives future re-runs of
    # this script without needing a manual XML edit each time.
    xml_text = re.sub(
        r'(<material name="[^"]+" texture="[^"]+")[^/]*/>',
        rf'\1 specular="{specular}" shininess="{shininess}" '
        rf'reflectance="{reflectance}"/>',
        xml_text,
    )

    body_match = re.search(r'<body name="([^"]+)">', xml_text)
    if body_match:
        old_body_tag = body_match.group(0)
        new_body_tag = (
            f'<body name="{body_match.group(1)}" pos="{body_pos}">\n'
            f'      <freejoint/>\n'
            f'      <inertial pos="0 0 0" mass="{mass:.6f}" '
            f'diaginertia="{diag_str}"/>'
        )
        xml_text = xml_text.replace(old_body_tag, new_body_tag, 1)

    xml_path.write_text(xml_text)

    print(f"\nDone. Assets written to: {final_dir}")
    print(f"Add this to your scene XML:\n")
    print(f'    <include file="meshes/{name}/{input_obj.stem}.xml"/>')
    print(f"\n(Make sure <compiler meshdir=\"meshes/\" texturedir=\"meshes/\"/> "
          f"is set, and that the texture in the generated .mtl/.xml is PNG "
          f"-- already handled above if the source was JPG.)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-obj", required=True, type=Path,
                    help="Path to the source .obj (its .mtl and texture "
                         "should sit alongside it)")
    p.add_argument("--name", required=True,
                    help="Object name -- used for the output folder under "
                         "meshes-dir and the body name")
    p.add_argument("--meshes-dir", required=True, type=Path,
                    help="Your scene's meshes/ directory (matching "
                         "<compiler meshdir=\"...\"/>)")
    p.add_argument("--scale", type=float, required=True,
                    help="Uniform mesh scale factor, meters per source unit "
                         "(e.g. 0.048). Compute this from a known real "
                         "dimension of the part before running.")
    p.add_argument("--density", type=float, required=True,
                    help="Material density in kg/m^3 (e.g. 7980 for 316L "
                         "stainless steel)")
    p.add_argument("--threshold", type=float, default=0.04,
                    help="CoACD concavity threshold (default 0.04)")
    p.add_argument("--max-hulls", type=int, default=12,
                    help="Max convex collision parts (default 12)")
    p.add_argument("--body-pos", default="0.25 0 0.02",
                    help="Initial body pos written into the generated XML "
                         "(default '0.25 0 0.02' -- edit to taste)")
    p.add_argument("--specular", type=float, default=0.9,
                    help="Material specular, 0-1 (default 0.9, tuned for "
                         "polished metal -- lower for matte/plastic parts)")
    p.add_argument("--shininess", type=float, default=0.85,
                    help="Material shininess, 0-1 (default 0.85, sharp "
                         "polished highlight -- lower for a softer sheen)")
    p.add_argument("--reflectance", type=float, default=0.4,
                    help="Material environment reflectance, 0-1 (default "
                         "0.4 -- requires a skybox in the scene to see any "
                         "effect; set to 0 to disable)")
    args = p.parse_args()

    run_pipeline(
        input_obj=args.input_obj,
        name=args.name,
        meshes_dir=args.meshes_dir,
        scale=args.scale,
        density=args.density,
        threshold=args.threshold,
        max_hulls=args.max_hulls,
        body_pos=args.body_pos,
        specular=args.specular,
        shininess=args.shininess,
        reflectance=args.reflectance,
    )


if __name__ == "__main__":
    main()