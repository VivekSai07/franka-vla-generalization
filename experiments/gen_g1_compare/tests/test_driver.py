"""The G1 driver `scripts/gen_g1_eval.sh`: its docker commands in dry-run mode and its polling, collecting, skipping and resuming
against a fake `docker`. Only bash runs; robosuite or admission image, CPU."""
import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/gen_g1_eval.sh"
ROB = "sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e"
CANDIDATES = (50760, 76140, 88830, 101520, 114210, 126900)

FAKE_DOCKER = r"""#!/usr/bin/env bash
state="$FAKE_STATE"; mkdir -p "$state/running"
echo "$*" >> "$state/calls.log"
case "$1" in
  run)
    name=""; detach=0; prev=""
    for a in "$@"; do [ "$prev" = "--name" ] && name="$a"; [ "$a" = "-d" ] && detach=1; prev="$a"; done
    if [ "$detach" = 1 ]; then : > "$state/running/$name"; echo "fakeid-$name"; exit 0; fi
    case " $* " in
      *" collect "*)
        n=$(cat "$state/collects" 2>/dev/null || echo 0); echo $((n + 1)) > "$state/collects"
        [ "$n" -lt "${FAKE_COLLECT_FAILS:-0}" ] && exit 3
        exit 0 ;;
    esac
    exit 0 ;;
  ps)
    prefix=""; for a in "$@"; do case "$a" in name=*) prefix="${a#name=}" ;; esac; done
    shown=0
    for f in "$state/running/$prefix"*; do [ -e "$f" ] || continue; basename "$f"; shown=1; done
    if [ -e "$state/seen_$prefix" ]; then rm -f "$state/running/$prefix"*; fi
    [ "$shown" = 1 ] && : > "$state/seen_$prefix"
    exit 0 ;;
  inspect) echo 0; exit 0 ;;
  rm) rm -f "$state/running/${@: -1}"; echo "${@: -1}"; exit 0 ;;
  wait) echo "docker wait must never be used" >&2; exit 99 ;;
esac
exit 0
"""


def commands(text):
    return [line for line in text.splitlines() if line.startswith("docker ")]


class Driver(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.out, self.main, self.wt, self.m0c = base / "campaign", base / "main", base / "wt", base / "m0"
        self.wt.mkdir()
        self.ck = self.main / "outputs/train/xvla_g1_rand_v1/checkpoints"
        for c in CANDIDATES:
            (self.ck / f"step_{c:06d}").mkdir(parents=True)
            (self.ck / f"step_{c:06d}/checkpoint_result.json").write_text("{}")
        (self.out / "arms/A1").mkdir(parents=True)
        self.bin = base / "bin"
        self.bin.mkdir()
        (self.bin / "docker").write_text(FAKE_DOCKER)
        (self.bin / "docker").chmod((self.bin / "docker").stat().st_mode | stat.S_IEXEC)
        self.state = base / "state"

    def run_step(self, *args, check=True, dry=True, fake=False, wt=None, **env):
        e = {**os.environ, "WT": str(wt or self.wt), "OUT": str(self.out), "MAIN": str(self.main), "M0C": str(self.m0c),
             "BUNDLE": str(self.main / "bundle"), "TEXTURES": str(self.main / ".cache/gen_textures"), "POLL_SECONDS": "0", **env}
        if dry:
            e["GEN_NO_RUN"] = "1"
        if fake:
            e.update(DOCKER=f"bash {self.bin / 'docker'}", FAKE_STATE=str(self.state), GEN_NO_RUN="0")
        result = subprocess.run(["bash", str(SCRIPT), *args], env=e, capture_output=True, text=True, timeout=120)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def prepared(self):
        (self.out / "arms/A1/prepared.json").write_text("{}")

    def decided(self, winner=114210):
        (self.out / "arms/A1/selection.json").write_text(json.dumps(dict(winner=winner)))

    def calls(self):
        return (self.state / "calls.log").read_text().splitlines() if (self.state / "calls.log").exists() else []


class Dry(Driver):
    def test_prepare_binds_the_six_candidates_the_adapter_and_g1s_data(self):
        out = self.run_step("prepare").stdout
        self.assertEqual(len(commands(out)), 1)
        self.assertIn("--arm A1 ", out)
        self.assertIn("--adapter experiments.vla_inputs_arm_xvla_g1.adapter:XVLAG1Adapter ", out)
        for c in CANDIDATES:
            self.assertIn(f"{c}=/ckpt/step_{c:06d}", out)
        self.assertIn("--runtime nvidia", out)
        self.assertIn(ROB, out)
        self.assertIn(f"-v {self.wt}:/vla:ro", out)
        self.assertIn(f"-v {self.main}/outputs/gen_g1_data:/vla/outputs/gen_g1_data:ro", out)
        self.assertIn(f"-v {self.ck}:/ckpt:ro", out)
        self.assertNotIn("phase", out)

    def test_prepare_refuses_without_a_candidate(self):
        (self.ck / "step_088830/checkpoint_result.json").unlink()
        result = self.run_step("prepare", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing checkpoint", result.stderr)
        self.assertEqual(commands(result.stdout), [])

    def test_a_stage_is_one_pool_of_six_workers_on_seven_cores_each(self):
        self.prepared()
        cmds = commands(self.run_step("stage", "benchmark", "114210", "6", "7").stdout)
        self.assertEqual(len(cmds), 6)
        for k, cmd in enumerate(cmds):
            lo = k * 7
            self.assertIn(f"--name g1-A1-benchmark-114210-{k} ", cmd)
            self.assertIn(f"--cpuset-cpus {lo}-{lo + 6} ", cmd)
            self.assertIn("-e OMP_NUM_THREADS=7 ", cmd)
            self.assertIn(f"--checkpoint 114210 --shard {k} --shards 6", cmd)
            self.assertIn("--arm A1 --tier benchmark", cmd)
            self.assertIn(" -d ", cmd)

    def test_selection_runs_each_candidate_alone_on_eight_cores_all_at_once(self):
        self.prepared()
        out = self.run_step("selection").stdout
        cmds = commands(out)
        self.assertEqual(len(cmds), 6)
        seen = {}
        for cmd in cmds:
            checkpoint = int(cmd.split("--checkpoint ")[1].split()[0])
            seen[checkpoint] = cmd.split("--cpuset-cpus ")[1].split()[0]
            self.assertIn("--tier selection", cmd)
            self.assertIn("--shard 0 --shards 1", cmd)
            self.assertIn("-e OMP_NUM_THREADS=8 ", cmd)
        self.assertEqual(seen, {c: f"{8 * i}-{8 * i + 7}" for i, c in enumerate(CANDIDATES)})

    def test_steps_refuse_without_prepare(self):
        for step in (("stage", "smoke", "126900", "6", "7"), ("selection",), ("select",), ("compare", "114210")):
            result = self.run_step(*step, check=False)
            self.assertNotEqual(result.returncode, 0, step)
            self.assertIn("run prepare first", result.stderr)

    def test_winner_and_compare(self):
        self.prepared()
        self.decided(114210)
        self.assertEqual(self.run_step("winner").stdout.strip(), "114210")
        cmds = commands(self.run_step("compare", "114210").stdout)
        self.assertEqual(len(cmds), 2)
        self.assertIn("experiments.gen_eval.evaluate --root /out compare --arm A1 --checkpoint 114210", cmds[0])
        self.assertIn("experiments.gen_g1_compare.compare g0 --root /out --checkpoint 114210", cmds[1])
        self.assertNotIn("nvidia", cmds[0] + cmds[1])

    def test_compare_skips_what_exists(self):
        self.prepared()
        (self.out / "arms/A1/compare_step_114210.json").write_text("{}")
        self.assertEqual(len(commands(self.run_step("compare", "114210").stdout)), 1)
        (self.out / "arms/A1/compare_A2_step_114210.json").write_text("{}")
        self.assertEqual(commands(self.run_step("compare", "114210").stdout), [])

    def test_m0_steps_go_through_m0s_driver_at_the_winner(self):
        self.prepared()
        self.decided(114210)
        prep = commands(self.run_step("m0prepare", wt=ROOT).stdout)
        self.assertEqual(len(prep), 1)
        self.assertIn("experiments.vla_inputs_eval.evaluate --root /out prepare --arm A1", prep[0])
        self.assertIn("--checkpoints 114210=/ckpt/step_114210 --phase-checkpoint 114210", prep[0])
        self.assertIn(f"-v {self.m0c}:/out:rw", prep[0])
        self.assertIn(f"-v {self.main}/outputs/gen_g1_data:/vla/outputs/gen_g1_data:ro", prep[0])
        stage = commands(self.run_step("m0comparison", wt=ROOT).stdout)
        self.assertEqual(len(stage), 6)
        self.assertIn("--arm A1 --tier comparison --checkpoint 114210 --shard 0 --shards 6", stage[0])
        compare = commands(self.run_step("m0compare", wt=ROOT).stdout)
        self.assertIn("experiments.gen_g1_compare.compare m0 --root /out --g0-root /g0 --checkpoint 114210", compare[0])
        self.assertIn(f"-v {self.out}:/g0:ro", compare[0])
        (self.m0c / "arms/A1").mkdir(parents=True)
        (self.m0c / "arms/A1/prepared.json").write_text("{}")
        (self.m0c / "arms/A1/compare_A2_step_114210_vs_088830.json").write_text("{}")
        self.assertEqual(commands(self.run_step("m0prepare", wt=ROOT).stdout), [])
        self.assertEqual(commands(self.run_step("m0compare", wt=ROOT).stdout), [])


class Fake(Driver):
    def test_a_stage_polls_with_ps_collects_and_never_waits(self):
        self.prepared()
        r = self.run_step("stage", "benchmark", "114210", "6", "7", dry=False, fake=True)
        self.assertIn("STAGE COMPLETE A1 benchmark 114210", r.stdout)
        calls = self.calls()
        self.assertEqual(sum(1 for c in calls if c.startswith("run ") and " -d " in c), 6)
        self.assertTrue(any(c.startswith("ps ") for c in calls))
        self.assertEqual(sum(1 for c in calls if " collect " in c), 1)
        self.assertFalse(any(c.startswith("wait") for c in calls))

    def test_an_incomplete_stage_is_resumed_once_then_loud(self):
        self.prepared()
        r = self.run_step("stage", "benchmark", "114210", "6", "7", dry=False, fake=True, FAKE_COLLECT_FAILS="1")
        self.assertIn("resuming the missing trials once", r.stdout)
        self.assertIn("STAGE COMPLETE", r.stdout)
        self.assertEqual(sum(1 for c in self.calls() if c.startswith("run ") and " -d " in c), 12)
        self.state = self.state.parent / "state2"
        r = self.run_step("stage", "benchmark", "114210", "6", "7", dry=False, fake=True, FAKE_COLLECT_FAILS="2", check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("STAGE STILL INCOMPLETE", r.stderr)
        self.assertEqual(sum(1 for c in self.calls() if c.startswith("run ") and " -d " in c), 12)

    def test_a_complete_stage_is_skipped(self):
        self.prepared()
        folder = self.out / "arms/A1/benchmark/step_114210"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text('{\n  "status": "complete"\n}\n')
        r = self.run_step("stage", "benchmark", "114210", "6", "7", dry=False, fake=True)
        self.assertIn("STAGE ALREADY COMPLETE", r.stdout)
        self.assertEqual(self.calls(), [])
        (folder / "result.json").write_text('{\n  "status": "incomplete"\n}\n')
        self.run_step("stage", "benchmark", "114210", "6", "7", dry=False, fake=True)
        self.assertNotEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
