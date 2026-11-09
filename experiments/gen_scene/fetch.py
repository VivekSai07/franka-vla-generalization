"""Choose, verify and extract the G0/G1 texture set from the pinned Hugging Face mirror of ambientCG. Stdlib only.

The CC0 originals (ambientcg.com, polyhaven.com) were unreachable from this host (proxy 407, 2026-11-09), so the set comes from
`nyuuzyou/ambientcg` (licence cc0-1.0) at a pinned revision. The mirror holds each material's 1K colour map re-compressed as
JPEG q90. Every tar shard and the metadata are verified against the sha256 the Hub published for that revision, then the chosen
colour maps are extracted and recorded, with their own sha256, in the committed manifest. Run it through
`scripts/gen_fetch_textures.sh` on the HOST (containers stay `--network none`).

Selection rule (fixed here, before any policy sees a texture): for each of 12 table-like and 12 floor/wall families, the 5
materials with the smallest sha256(id) among those with a 1K JPG archive and a square 1024 px colour map: 60 + 60 textures.
"""
import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

REPO = "nyuuzyou/ambientcg"
REVISION = "50329ad52d825bfb740080546c30c04014983ec8"
LICENCE = "CC0-1.0"
METADATA = "metadata.jsonl.zst"
METADATA_SHA256 = "649dfa618fc292c5b380c925693d453c45296bbe6dcb800fcacdfd74d5a097b0"
SHARDS = {
    "dataset_00.tar": "0343e9090a4cb119a85b1c218ed99774905ee2abcbddb687a44317705f338252",
    "dataset_01.tar": "e73c9e917a8c76d53f170458d11fa4fbce42893f7921c246210804d54ee5f644",
    "dataset_02.tar": "f035951caa4ed60ff0b805f3ddd6fb6d43157b8dc1c18b56144679a3a6c76a87",
    "dataset_03.tar": "b7bcff8a207d238607adbc43bebda7127f3ea898d0583cd8f30ac35301d90edf",
    "dataset_04.tar": "56a8fddf5ea4be8e3de3547a842e2eb2caac401d79a4df2ae705b6c225521bc4",
    "dataset_05.tar": "d6a9199eff3d006d98cd00d2aeb9dd62808ef6879ac27bc00d7c9603e43e6525",
    "dataset_06.tar": "82104a3894ab2ffa26efdabd58b708ebe558472063d2d5e8e05ffe7bf916e245",
    "dataset_07.tar": "b35decd39f924e2ae1a252469d6710783ea73d2f73eb09e9f252994dd173e3f9",
}
FAMILIES = {
    "table": ("Wood", "Planks", "Marble", "Granite", "Fabric", "Leather", "Plastic", "Metal", "Terrazzo", "Travertine",
              "Chipboard", "PaintedWood"),
    "room": ("Tiles", "Concrete", "Bricks", "PavingStones", "Asphalt", "PaintedPlaster", "Plaster", "Carpet", "WoodFloor",
             "Ground", "Rock", "Wallpaper"),
}
PER_FAMILY = 5
IMAGE_PX = 1024


def member_name(identifier):
    return f"{identifier}_1K-JPG_{identifier}_1K-JPG_Color.jpg"


def family_of(identifier):
    match = re.match(r"^([A-Za-z]+)\d", identifier)
    if not match:
        raise ValueError(f"cannot read the family of {identifier!r}")
    return match.group(1)


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def choose(metadata_rows, member_sizes):
    """{surface: [ids]} (sorted by id within the surface): per family the PER_FAMILY eligible ids with the smallest sha256(id).
    Eligible: a 1K-JPG archive, and a colour map that is IMAGE_PX square (`member_sizes`: {member: (width, height)})."""
    members = {m for m, size in member_sizes.items() if tuple(size) == (IMAGE_PX, IMAGE_PX)}
    eligible = {}
    for row in metadata_rows:
        identifier = row["id"]
        if "HDRI" in identifier or not row["download_url"].endswith("_1K-JPG.zip") or member_name(identifier) not in members:
            continue
        eligible.setdefault(family_of(identifier), []).append(row)
    out = {}
    for surface, families in FAMILIES.items():
        chosen = []
        for family in families:
            rows = sorted(eligible.get(family, []), key=lambda r: sha256_hex(r["id"].encode()))
            if len(rows) < PER_FAMILY:
                raise ValueError(f"family {family}: only {len(rows)} eligible materials, need {PER_FAMILY}")
            chosen += [r["id"] for r in rows[:PER_FAMILY]]
        out[surface] = sorted(chosen)
    if set(out["table"]) & set(out["room"]):
        raise ValueError("a material is in both surfaces")
    return out


def jpeg_size(data):
    """(width, height) of a baseline or progressive JPEG from its first SOF marker."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise ValueError("bad JPEG marker")
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + length
    raise ValueError("no SOF marker")


def verify_stage(stage):
    stage = Path(stage)
    expected = {METADATA: METADATA_SHA256, **SHARDS}
    for name, digest in expected.items():
        path = stage / name
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 22), b""):
                hasher.update(block)
        if hasher.hexdigest() != digest:
            raise ValueError(f"{path} does not match the sha256 the Hub published for revision {REVISION[:10]}")


def read_metadata(stage):
    text = subprocess.run(["zstd", "-dc", str(Path(stage) / METADATA)], check=True, capture_output=True).stdout.decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def shard_members(stage):
    """{colour-map member: (shard, (width, height))} over every shard."""
    out = {}
    for shard in SHARDS:
        with tarfile.open(Path(stage) / shard) as tar:
            for info in tar:
                if info.name.endswith("_Color.jpg"):
                    out[info.name] = (shard, jpeg_size(tar.extractfile(info).read(4096 * 4)))
    return out


def build(stage, out_dir, manifest_path):
    """Verify the stage, choose, extract into `out_dir` (must be empty or absent) and write the manifest. Returns it."""
    stage, out_dir = Path(stage), Path(out_dir)
    verify_stage(stage)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    rows = {r["id"]: r for r in read_metadata(stage)}
    where = shard_members(stage)
    chosen = choose(rows.values(), {m: size for m, (_, size) in where.items()})
    out_dir.mkdir(parents=True, exist_ok=True)
    textures = []
    for surface, ids in chosen.items():
        for identifier in ids:
            member = member_name(identifier)
            shard = where[member][0]
            with tarfile.open(stage / shard) as tar:
                data = tar.extractfile(member).read()
            width, height = jpeg_size(data)
            if (width, height) != (IMAGE_PX, IMAGE_PX):
                raise ValueError(f"{identifier}: {width}x{height}, expected {IMAGE_PX}x{IMAGE_PX}")
            name = f"{identifier}.jpg"
            (out_dir / name).write_bytes(data)
            textures.append(dict(name=name, surface=surface, family=family_of(identifier), upstream_id=identifier,
                                 source_url=rows[identifier]["download_url"], mirror_shard=shard, mirror_member=member,
                                 licence=LICENCE, bytes=len(data), sha256=sha256_hex(data), width=width, height=height))
    manifest = dict(
        source=dict(repo=f"huggingface.co/datasets/{REPO}", revision=REVISION, licence=LICENCE,
                    note="mirror of ambientcg.com 1K-JPG colour maps, re-compressed to JPEG q90; ambientcg.com and polyhaven.com "
                         "returned proxy 407 on 2026-11-09", metadata_sha256=METADATA_SHA256, shards=SHARDS),
        selection=dict(families=FAMILIES, per_family=PER_FAMILY,
                       rule="per family the 5 eligible ids with the smallest sha256(id); eligible = 1K-JPG archive and a 1024x1024 colour map"),
        textures=sorted(textures, key=lambda t: t["name"]))
    Path(manifest_path).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage", required=True, help="folder holding metadata.jsonl.zst and dataset_0N.tar")
    parser.add_argument("--out", required=True, help="texture folder (.cache/gen_textures)")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    manifest = build(args.stage, args.out, args.manifest)
    print("FETCHED", {s: sum(1 for t in manifest["textures"] if t["surface"] == s) for s in FAMILIES}, flush=True)


if __name__ == "__main__":
    main()
