"""Texture and primitive-distractor pools of the generalization programme (G0 spec D6). Pure: stdlib only, imports no simulator.

Textures are CC0 ambientCG colour maps listed, with their sha256, in `assets_manifest.json` (committed); the files live in
`.cache/gen_textures/` (gitignored). Each surface (`table`, `room` = floor and walls) is split ONCE, by a fixed rule on the file
name, into a train pool (G1, G2) and a test pool (G0 only). The rule is committed before any G1 episode exists and never changes:
a change would move a test texture into training and void every G0 number.

Primitive distractors: pool A (train) and pool B (test) share neither a shape family nor a colour.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("assets_manifest.json")
TEXTURE_DIR = ROOT / ".cache/gen_textures"
SURFACES = ("table", "room")
POOLS = ("train", "test")
TEST_EVERY, TEST_REMAINDER = 5, 4  # after sorting by sha256(file name), index % 5 == 4 goes to the test pool

# name -> (MuJoCo geom type, size). `size` is MuJoCo's own: box half-extents; cylinder (radius, half-height);
# sphere (radius); capsule (radius, half-length of the straight part); ellipsoid half-axes.
PRIMITIVE_SHAPES = {
    "A": {"cube": ("box", (0.025, 0.025, 0.025)), "block": ("box", (0.030, 0.020, 0.040)),
          "slab": ("box", (0.040, 0.030, 0.015)), "drum": ("cylinder", (0.025, 0.040)),
          "tube": ("cylinder", (0.020, 0.050)), "disc": ("cylinder", (0.035, 0.012))},
    "B": {"ball": ("sphere", (0.030,)), "egg": ("ellipsoid", (0.040, 0.025, 0.025)),
          "pill": ("capsule", (0.020, 0.030))},
}
PRIMITIVE_COLOURS = {
    "A": {"red": (0.85, 0.15, 0.15), "green": (0.20, 0.70, 0.25), "yellow": (0.95, 0.85, 0.20),
          "purple": (0.55, 0.25, 0.70), "white": (0.92, 0.92, 0.92), "brown": (0.45, 0.30, 0.15)},
    "B": {"pink": (0.95, 0.55, 0.75), "lime": (0.65, 0.90, 0.15), "black": (0.07, 0.07, 0.07),
          "grey": (0.50, 0.50, 0.55), "olive": (0.55, 0.60, 0.10)},
}
PRIMITIVE_MASS_KG = 0.10
PRIMITIVE_LYING = ("capsule",)  # capsules lie along the body x axis; the other shapes stand
PRIMITIVE_FRICTION = (1.2, 0.005, 0.0001)  # as the props'


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_manifest(path=MANIFEST):
    return json.loads(Path(path).read_text())


def split(names):
    """{"train": [...], "test": [...]} (each sorted by name): sort by sha256 of the name, every 5th (index 4, 9, ...) is test."""
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("duplicate texture names")
    ordered = sorted(names, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    test = {n for i, n in enumerate(ordered) if i % TEST_EVERY == TEST_REMAINDER}
    return {"train": sorted(n for n in names if n not in test), "test": sorted(test)}


def pools(manifest=None):
    """{surface: {"train": [names], "test": [names]}} from the manifest's textures."""
    manifest = manifest if manifest is not None else load_manifest()
    out = {}
    for surface in SURFACES:
        out[surface] = split(t["name"] for t in manifest["textures"] if t["surface"] == surface)
    return out


def texture_path(name, directory=None):
    return Path(directory if directory is not None else TEXTURE_DIR) / name


def verify_files(manifest=None, directory=None):
    """Raises unless every manifest texture exists with its recorded size and sha256 and nothing else is in the directory."""
    manifest = manifest if manifest is not None else load_manifest()
    directory = Path(directory if directory is not None else TEXTURE_DIR)
    wanted = {t["name"]: t for t in manifest["textures"]}
    present = {p.name for p in directory.iterdir() if p.is_file()} if directory.is_dir() else set()
    if present != set(wanted):
        raise ValueError(f"{directory}: missing {sorted(set(wanted) - present)[:3]}, unexpected {sorted(present - set(wanted))[:3]}")
    for name, record in wanted.items():
        path = directory / name
        if path.is_symlink() or path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError(f"{path} does not match the manifest")
    return len(wanted)


def manifest_digest(manifest=None):
    manifest = manifest if manifest is not None else load_manifest()
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def primitive_extent(shape):
    """(footprint radius, height) in metres of a primitive as `compose` places it: boxes, cylinders and spheres stand, an
    ellipsoid's long axis and a lying capsule's axis are horizontal. The radius is the half-diagonal / half-length."""
    for table in PRIMITIVE_SHAPES.values():
        if shape in table:
            kind, size = table[shape]
            break
    else:
        raise ValueError(f"unknown primitive shape {shape!r}")
    if kind == "box":
        return float((size[0] ** 2 + size[1] ** 2) ** 0.5), 2 * size[2]
    if kind == "cylinder":
        return float(size[0]), 2 * size[1]
    if kind == "sphere":
        return float(size[0]), 2 * size[0]
    if kind == "ellipsoid":
        return float(max(size[0], size[1])), 2 * size[2]
    if kind == "capsule":
        return float(size[1] + size[0]), 2 * size[0]
    raise ValueError(kind)
