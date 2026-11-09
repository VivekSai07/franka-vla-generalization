"""The G0 driver `scripts/gen_eval.sh`: its docker commands in dry-run mode (`GEN_NO_RUN=1` prints them and runs nothing) and its
polling, collecting and resuming against a fake `docker` on the PATH. Only bash runs; admission or robosuite image, CPU."""
import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/gen_eval.sh"
ROB = "sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e"
ADM = "sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc"
ADAPTERS = {"A0": ("experiments.vla_inputs_arm_xvla.adapter:XVLAAdapter", 101520, "/bundle/checkpoints/step_101520"),
            "A2": ("experiments.vla_inputs_arm_xvla_force.adapter:XVLAForceAdapter", 88830, "/ckpt/step_088830"),
            "A3": ("experiments.vla_inputs_arm_xvla_force_box.adapter:XVLAForceBoxAdapter", 101520, "/ckpt/step_101520")}

FAKE_DOCKER = r"""#!/usr/bin/env bash
# a fake docker: logs every call; `run -d --name N` starts N, `ps` shows the running ones once then they have "finished"
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
    # the containers finish after being seen once
    if [ -e "$state/seen_$prefix" ]; then rm -f "$state/running/$prefix"*; fi
    [ "$shown" = 1 ] && : > "$state/seen_$prefix"
    exit 0 ;;
  inspect)
    name="${@: -1}"; var="FAKE_EXIT_${name//-/_}"; echo "${!var:-0}"; exit 0 ;;
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
        self.out, self.main, self.m0, self.wt = base / "campaign", base / "main", base / "m0", base / "wt"
        self.wt.mkdir()
        for arm in ("A0", "A2", "A3"):
            (self.out / "arms" / arm).mkdir(parents=True)
        (self.m0 / "arms/A2").mkdir(parents=True)
        for rel in ("bundle/checkpoints/step_101520", "outputs/train/xvla_p2_force_v1/checkpoints/step_088830",
                    "outputs/train/xvla_p2_force_box_v1/checkpoints/step_101520"):
            (self.main / rel).mkdir(parents=True)
            (self.main / rel / "checkpoint_result.json").write_text("{}")
        self.bin = base / "bin"
        self.bin.mkdir()
        (self.bin / "docker").write_text(FAKE_DOCKER)
        (self.bin / "docker").chmod((self.bin / "docker").stat().st_mode | stat.S_IEXEC)
        self.state = base / "state"

    def env(self, **extra):
        return {**os.environ, "WT": str(self.wt), "OUT": str(self.out), "MAIN": str(self.main), "M0": str(self.m0),
                "BUNDLE": str(self.main / "bundle"), "TEXTURES": str(self.main / ".cache/gen_textures"), "POLL_SECONDS": "0", **extra}

    def run_step(self, *args, check=True, dry=True, fake=False, **env):
        e = self.env(**env)
        if dry:
            e["GEN_NO_RUN"] = "1"
        if fake:
            e.update(DOCKER=f"bash {self.bin / 'docker'}", FAKE_STATE=str(self.state), GEN_NO_RUN="0")
        result = subprocess.run(["bash", str(SCRIPT), *args], env=e, capture_output=True, text=True, timeout=120)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def prepared(self, arm):
        (self.out / "arms" / arm / "prepared.json").write_text("{}")

    def calls(self):
        return (self.state / "calls.log").read_text().splitlines() if (self.state / "calls.log").exists() else []


class Dry(Driver):
    def test_prepare_names_the_arm_its_adapter_the_single_checkpoint_and_the_mounts(self):
        for arm, (adapter, update, container) in ADAPTERS.items():
            with self.subTest(arm=arm):
                out = self.run_step("prepare", arm).stdout
                self.assertEqual(len(commands(out)), 1)
                self.assertIn(f"--arm {arm} ", out)
                self.assertIn(f"--adapter {adapter} ", out)
                self.assertIn(f"--checkpoints {update}={container}", out)
                self.assertIn("--runtime nvidia", out)
                self.assertIn(ROB, out)
                self.assertIn(f"-v {self.wt}:/vla:ro", out)
                self.assertIn(f"-v {self.main}/.cache/gen_textures:/vla/.cache/gen_textures:ro", out)
                self.assertIn(f"-v {self.m0}:/m0:ro", out)
                self.assertNotIn("phase", out)
        self.assertNotIn("/ckpt:ro", self.run_step("prepare", "A0").stdout)
        self.assertIn("outputs/train/xvla_p2_force_v1/checkpoints:/ckpt:ro", self.run_step("prepare", "A2").stdout)
        self.assertIn("outputs/train/xvla_p2_force_box_v1/checkpoints:/ckpt:ro", self.run_step("prepare", "A3").stdout)

    def test_only_a0_a2_and_a3_can_be_prepared(self):
        for arm in ("A1", "A4", "A5", "A10", "a2", ""):
            with self.subTest(arm=arm):
                result = self.run_step("prepare", arm, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(commands(result.stdout), [])

    def test_prepare_refuses_without_the_checkpoint(self):
        (self.main / "outputs/train/xvla_p2_force_v1/checkpoints/step_088830/checkpoint_result.json").unlink()
        result = self.run_step("prepare", "A2", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing checkpoint", result.stderr)

    def test_stages_refuse_an_arm_that_was_not_prepared(self):
        (self.out / "arms/A2").rmdir()
        for step in ("smoke", "selection", "benchmark", "compare", "select"):
            result = self.run_step(step, "A2", check=False)
            self.assertNotEqual(result.returncode, 0, step)
            self.assertIn("run prepare first", result.stderr)

    def test_benchmark_is_one_pool_of_six_workers_on_seven_cores_each(self):
        for arm, (adapter, update, container) in ADAPTERS.items():
            self.prepared(arm)
            cmds = commands(self.run_step("benchmark", arm).stdout)
            self.assertEqual(len(cmds), 6)
            shards = []
            for k, command in enumerate(cmds):
                self.assertIn(f"--arm {arm} --tier benchmark --checkpoint {update} ", command)
                shards.append(re.search(r"--shard (\d) --shards 6", command).group(1))
                self.assertIn(f"--cpuset-cpus {k * 7}-{k * 7 + 6}", command)
                self.assertIn("OMP_NUM_THREADS=7", command)
                self.assertIn(f"--name g0-{arm}-benchmark-{update}-{k} ", command)
                self.assertIn("--runtime nvidia", command)
                self.assertIn(" -d ", command)
                self.assertIn("-m experiments.gen_eval.evaluate --root /out worker", command)
            self.assertEqual(sorted(shards), list("012345"))

    def test_smoke_runs_with_six_workers_on_purpose(self):
        self.prepared("A2")
        cmds = commands(self.run_step("smoke", "A2").stdout)
        self.assertEqual(len(cmds), 6)
        self.assertTrue(all("--tier smoke" in c and "--shards 6" in c for c in cmds))

    def test_selection_runs_the_selection_tier_and_select_is_a_cli_call(self):
        self.prepared("A3")
        self.assertTrue(all("--tier selection" in c for c in commands(self.run_step("selection", "A3").stdout)))
        out = self.run_step("select", "A3").stdout
        self.assertIn("evaluate --root /out select --arm A3", out)
        self.assertIn("--runtime runc", out)

    def test_workers_and_cores_default_to_six_by_seven_and_can_be_overridden(self):
        text = SCRIPT.read_text()
        self.assertIn('WORKERS="${WORKERS:-6}"', text)
        self.assertIn('CORES="${CORES:-7}"', text)
        self.prepared("A2")
        cmds = commands(self.run_step("benchmark", "A2", WORKERS="3", CORES="5").stdout)
        self.assertEqual(len(cmds), 3)
        self.assertIn("--cpuset-cpus 10-14", cmds[2])
        self.assertIn("OMP_NUM_THREADS=5", cmds[2])

    def test_compare_and_final_run_without_a_gpu(self):
        self.prepared("A0")
        out = self.run_step("compare", "A0").stdout
        self.assertIn("compare --arm A0 --checkpoint 101520", out)
        self.assertIn("--runtime runc", out)
        self.assertIn("evaluate --root /out final", self.run_step("final").stdout)

    def test_plan_runs_in_the_admission_image_and_feasibility_in_the_robosuite_image_without_a_gpu(self):
        plan = commands(self.run_step("plan").stdout)
        self.assertEqual(len(plan), 1)
        self.assertIn(ADM, plan[0])
        feas = commands(self.run_step("feasibility", "12", "3").stdout)
        self.assertEqual(len(feas), 12)
        for k, command in enumerate(feas):
            self.assertIn(ROB, command)
            self.assertIn("--runtime runc", command)
            self.assertIn(f"--cpuset-cpus {k * 3}-{k * 3 + 2}", command)
            self.assertIn(f"--shard {k} --shards 12", command)
            self.assertIn("experiments.gen_eval.feasibility --out /out/feasibility --plan /out/plan", command)

    def test_the_identity_gate_is_one_gpu_container_with_the_workers_runtime(self):
        out = self.run_step("identity_gate", "t1").stdout
        cmds = commands(out)
        self.assertEqual(len(cmds), 1)
        c = cmds[0]
        self.assertIn("--runtime nvidia", c)
        self.assertIn("--cpuset-cpus 0-6", c)
        self.assertIn("OMP_NUM_THREADS=7", c)
        self.assertIn("MKL_NUM_THREADS=7", c)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", c)
        self.assertIn("-m experiments.gen_eval.identity_gate --campaign /m0 --out /out/identity_t1", c)
        self.assertIn(f"-v {self.m0}:/m0:ro", c)
        self.assertIn("outputs/train/xvla_p2_force_v1/checkpoints:/ckpt:ro", c)
        self.assertIn("--rm", c)

    def test_the_identity_gate_needs_a_tag_m0s_campaign_and_the_checkpoint(self):
        self.assertNotEqual(self.run_step("identity_gate", check=False).returncode, 0)
        (self.m0 / "arms/A2").rmdir()
        self.assertIn("has no arms/A2", self.run_step("identity_gate", "t", check=False).stderr)

    def test_unknown_commands_and_a_missing_out_fail(self):
        self.assertEqual(self.run_step("train", check=False).returncode, 2)
        env = self.env()
        del env["OUT"]
        self.assertNotEqual(subprocess.run(["bash", str(SCRIPT), "final"], env=env, capture_output=True, text=True).returncode, 0)

    def test_every_container_is_offline_read_only_and_mounts_the_worktree_read_only(self):
        self.prepared("A2")
        for command in commands(self.run_step("benchmark", "A2").stdout) + commands(self.run_step("plan").stdout):
            for needle in ("--network none", "--read-only", "-e HF_HUB_OFFLINE=1", "-e PYTHONPATH=/vla", f"-v {self.wt}:/vla:ro",
                           "--ulimit nofile=65536:65536", "--tmpfs /tmp:rw,size=16g", "-e MUJOCO_GL=osmesa"):
                self.assertIn(needle, command)

    def test_the_script_never_waits_with_docker_wait_and_polls_with_docker_ps(self):
        text = SCRIPT.read_text()
        self.assertNotRegex("\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#")), r"docker\"? wait|\" wait ")
        self.assertIn('"${DOCKER_CMD[@]}" ps -q --filter', text)


class WithFakeDocker(Driver):
    def test_a_stage_launches_polls_inspects_removes_and_collects(self):
        self.prepared("A2")
        result = self.run_step("benchmark", "A2", dry=False, fake=True)
        self.assertIn("STAGE COMPLETE A2 benchmark 88830", result.stdout)
        calls = self.calls()
        runs = [c for c in calls if c.startswith("run ") and " -d " in c]
        self.assertEqual(len(runs), 6)
        self.assertTrue(any(c.startswith("ps -q --filter name=g0-A2-benchmark-88830-") for c in calls))
        self.assertEqual(len([c for c in calls if c.startswith("inspect")]), 6)
        self.assertEqual(len([c for c in calls if c.startswith("rm ")]), 6)
        collects = [c for c in calls if " collect " in c]
        self.assertEqual(len(collects), 1)
        self.assertIn("--tier benchmark --checkpoint 88830", collects[0])
        self.assertFalse([c for c in calls if c.startswith("wait")])

    def test_an_incomplete_stage_is_resumed_once_and_then_completes(self):
        self.prepared("A3")
        result = self.run_step("benchmark", "A3", dry=False, fake=True, FAKE_COLLECT_FAILS="1")
        self.assertIn("resuming the missing trials once", result.stdout)
        self.assertIn("STAGE COMPLETE A3 benchmark 101520", result.stdout)
        calls = self.calls()
        self.assertEqual(len([c for c in calls if c.startswith("run ") and " -d " in c]), 12)
        self.assertEqual(len([c for c in calls if " collect " in c]), 2)

    def test_a_stage_that_stays_incomplete_is_loud_and_fails(self):
        self.prepared("A0")
        result = self.run_step("benchmark", "A0", dry=False, fake=True, check=False, FAKE_COLLECT_FAILS="5")
        self.assertEqual(result.returncode, 1)
        self.assertIn("STAGE STILL INCOMPLETE A0 benchmark 101520", result.stderr)
        self.assertEqual(len([c for c in self.calls() if " collect " in c]), 2)

    def test_a_worker_that_exited_nonzero_is_reported(self):
        self.prepared("A2")
        result = self.run_step("benchmark", "A2", dry=False, fake=True, FAKE_EXIT_g0_A2_benchmark_88830_4="137")
        self.assertIn("worker 4 exited 137", result.stdout)

    def test_feasibility_polls_then_merges_and_a_failed_shard_stops_before_the_merge(self):
        result = self.run_step("feasibility", "4", "3", dry=False, fake=True)
        calls = self.calls()
        self.assertEqual(len([c for c in calls if c.startswith("run ") and " -d " in c]), 4)
        self.assertTrue(calls[-1].endswith("--merge"))
        self.assertIn("--runtime runc", calls[0])
        shutil_state = self.state
        for f in shutil_state.rglob("*"):
            if f.is_file():
                f.unlink()
        bad = self.run_step("feasibility", "4", "3", dry=False, fake=True, check=False, FAKE_EXIT_g0_feas_2="1")
        self.assertEqual(bad.returncode, 1)
        self.assertIn("feasibility shard 2 exited 1", bad.stderr)
        self.assertFalse([c for c in self.calls() if c.endswith("--merge")])


if __name__ == "__main__":
    unittest.main()
