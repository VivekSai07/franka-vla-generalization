"""S0 binding: the sealed P2 root this sidecar is built from. Pure: imports no simulator.

Task 4 extends this with `SOURCE_FILES`, `provenance()` and the sidecar seal record.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P2_ROOT = ROOT / "outputs/props_p2_data/props_p2_20261013"
P2_SEAL = "b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898"
IMAGE = "sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e"
EPISODES = 540
FRAMES = 452
