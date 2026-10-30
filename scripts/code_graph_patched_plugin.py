#!/usr/bin/env python3
"""Build /tmp/ua-patched: symlinks to the real plugin, except prepare-incremental.mjs, which converts
the hardcoded '>30 files => FULL_UPDATE' decision into ARCHITECTURE_UPDATE (same files, architecture and
tour re-run). Every other gate in the plugin is untouched. The real install is never modified."""
import os
import shutil
from pathlib import Path

REAL = Path.home() / ".understand-anything" / "repo" / "understand-anything-plugin"
OUT = Path("/tmp/ua-patched")
SKILL = "skills/understand"

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / SKILL).mkdir(parents=True)

for name in os.listdir(REAL):
    if name != "skills":
        (OUT / name).symlink_to(REAL / name)
for name in os.listdir(REAL / SKILL):
    if name != "prepare-incremental.mjs":
        (OUT / SKILL / name).symlink_to(REAL / SKILL / name)
for name in os.listdir(REAL / "skills"):
    if name != "understand":
        (OUT / "skills" / name).symlink_to(REAL / "skills" / name)

src = (REAL / SKILL / "prepare-incremental.mjs").read_text()
needle = """    oldInventory,
    currentInventory,
  );

  const onlyGeneratedArtifacts ="""
assert src.count(needle) == 1, "patch anchor not found exactly once; plugin changed, re-read it"
patch = """    oldInventory,
    currentInventory,
  );
  if (decision.action === 'FULL_UPDATE') {
    // Local override: the >30-files / >50% heuristic only decides cost; ARCHITECTURE_UPDATE reruns layers and tour.
    decision.reason = `[override: FULL_UPDATE heuristic bypassed] ${decision.reason}`;
    decision.action = 'ARCHITECTURE_UPDATE';
  }

  const onlyGeneratedArtifacts ="""
(OUT / SKILL / "prepare-incremental.mjs").write_text(src.replace(needle, patch))
print("patched copy at", OUT / SKILL / "prepare-incremental.mjs")
