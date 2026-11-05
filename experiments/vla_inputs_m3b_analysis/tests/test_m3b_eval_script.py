"""The M3b evaluation driver in dry-run mode (`M0_NO_RUN=1` prints the docker commands and runs nothing), and `EXTRA_MOUNTS` in M0's driver.

Admission or robosuite image, CPU: only bash is run.
"""
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/vla_inputs_m3b_eval.sh"
DRIVER = ROOT / "scripts/vla_inputs_eval.sh"
PREFIX = "experiments.vla_inputs_arm_xvla_force_box_noisy.adapter:"
ADAPTERS = {"A5": "experiments.vla_inputs_arm_xvla_force_box_pred.adapter:XVLAPredictedBoxAdapter", "A6": PREFIX + "XVLAJitterSmall",
            "A7": PREFIX + "XVLAJitterLarge", "A8": PREFIX + "XVLADropout", "A9": PREFIX + "XVLASwap"}
ROB = "sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e"


class Driver(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "campaign"
        self.ck = Path(self.tmp.name) / "ckpt"
        self.detector = Path(self.tmp.name) / "detector"
        self.out.mkdir()
        self.detector.mkdir()
        (self.ck / "step_101520").mkdir(parents=True)
        (self.ck / "step_101520/checkpoint_result.json").write_text("{}")

    def run_step(self, *args, check=True, **env):
        full = {**os.environ, "WT": str(ROOT), "OUT": str(self.out), "CK": str(self.ck), "MAIN": str(ROOT), "M0_NO_RUN": "1",
                "BUNDLE": "/nonexistent/bundle", "DETECTOR_DIR": str(self.detector), **env}
        result = subprocess.run(["bash", str(SCRIPT), *args], env=full, capture_output=True, text=True, timeout=60)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def commands(self, result):
        return [line for line in result.stdout.splitlines() if line.startswith("docker ")]

    def prepared(self, arm):
        path = self.out / f"arms/{arm}/prepared.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({}))


class PrepareSmokeComparison(Driver):
    def test_prepare_names_the_arm_its_adapter_the_single_checkpoint_and_the_phase_checkpoint(self):
        for arm, adapter in ADAPTERS.items():
            with self.subTest(arm=arm):
                out = self.run_step("prepare", arm).stdout
                self.assertIn(f"--arm {arm} ", out)
                self.assertIn(f"--adapter {adapter} ", out)
                self.assertIn("--checkpoints 101520=/ckpt/step_101520 ", out + " ")
                self.assertIn("--phase-checkpoint 101520", out)
                self.assertIn(f"-v {self.ck}:/ckpt:ro", out)
                self.assertEqual(len(self.commands(self.run_step("prepare", arm))), 1)

    def test_only_the_five_m3b_arms_can_be_prepared(self):
        for arm in ("A0", "A1", "A2", "A3", "A4", "A10", "A5x", "a6", ""):
            with self.subTest(arm=arm):
                result = self.run_step("prepare", arm, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.commands(result), [])

    def test_prepare_refuses_without_the_checkpoint(self):
        (self.ck / "step_101520/checkpoint_result.json").unlink()
        result = self.run_step("prepare", "A6", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing checkpoint step_101520", result.stderr)

    def test_smoke_and_comparison_refuse_an_arm_that_was_not_prepared(self):
        for step in ("smoke", "comparison"):
            result = self.run_step(step, "A7", check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("run prepare first", result.stderr)

    def test_smoke_runs_the_two_smoke_layouts_with_one_worker_at_101520(self):
        self.prepared("A8")
        commands = self.commands(self.run_step("smoke", "A8"))
        self.assertEqual(len(commands), 1)
        self.assertIn("--arm A8 --tier smoke --checkpoint 101520 --shard 0 --shards 1", commands[0])
        self.assertIn("--runtime nvidia", commands[0])

    def test_comparison_is_one_pool_of_six_workers_on_seven_cores_each_and_never_a_selection(self):
        self.prepared("A9")
        commands = self.commands(self.run_step("comparison", "A9"))
        self.assertEqual(len(commands), 6)
        shards = []
        for k, command in enumerate(commands):
            self.assertIn("--arm A9 --tier comparison --checkpoint 101520", command)
            shards.append(re.search(r"--shard (\d) --shards 6", command).group(1))
            self.assertIn(f"--cpuset-cpus {k * 7}-{k * 7 + 6}", command)
            self.assertIn("OMP_NUM_THREADS=7", command)
        self.assertEqual(sorted(shards), list("012345"))

    def test_workers_and_cores_can_be_overridden_but_default_to_six_by_seven(self):
        text = SCRIPT.read_text()
        self.assertIn('WORKERS="${WORKERS:-6}"', text)
        self.assertIn('CORES="${CORES:-7}"', text)

    def test_there_is_no_selection_no_compare_and_no_final_step(self):
        for step in ("selection", "select", "compare", "compare_a2", "final", "train"):
            self.assertEqual(self.run_step(step, "A6", check=False).returncode, 2, step)

    def test_analyse_runs_the_analysis_module_in_the_robosuite_image_without_a_gpu(self):
        commands = self.commands(self.run_step("analyse"))
        self.assertEqual(len(commands), 1)
        self.assertIn(ROB, commands[0])
        self.assertIn("--runtime runc", commands[0])
        self.assertTrue(commands[0].rstrip().endswith("-m experiments.vla_inputs_m3b_analysis.analyse --root /out analyse"))


class DetectorMount(Driver):
    def test_the_predicted_box_arm_mounts_the_detector_directory_read_only_in_every_step(self):
        self.prepared("A5")
        for step in ("prepare", "smoke", "comparison"):
            for command in self.commands(self.run_step(step, "A5")):
                self.assertRegex(command, rf"-v {re.escape(str(self.detector))}:/vla/outputs/vla_inputs_box_detector/[A-Za-z0-9_./-]+:ro")

    def test_the_predicted_box_arm_refuses_without_a_detector_directory(self):
        result = self.run_step("prepare", "A5", check=False, DETECTOR_DIR="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DETECTOR_DIR", result.stderr)
        result = self.run_step("prepare", "A5", check=False, DETECTOR_DIR="/nonexistent/detector")
        self.assertNotEqual(result.returncode, 0)

    def test_the_corruption_arms_mount_no_detector(self):
        self.prepared("A6")
        for step in ("prepare", "smoke", "comparison"):
            for command in self.commands(self.run_step(step, "A6")):
                self.assertNotIn("vla_inputs_box_detector", command)
                self.assertNotIn(str(self.detector), command)


class IdentityGate(Driver):
    def test_the_gate_runs_with_a_gpu_reads_the_real_campaign_read_only_and_writes_to_the_scratch_out(self):
        campaign = Path(self.tmp.name) / "real_campaign"
        campaign.mkdir()
        commands = self.commands(self.run_step("identity_gate", CAMPAIGN=str(campaign)))
        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertIn("--runtime nvidia", command)
        self.assertIn(f"-v {campaign}:/campaign:ro", command)
        self.assertIn(f"-v {self.out}:/out:rw", command)
        self.assertIn(f"-v {self.ck}:/ckpt:ro", command)
        self.assertTrue(command.rstrip().endswith("-m experiments.vla_inputs_m3b_analysis.identity_gate --campaign /campaign --out /out --checkpoint 101520 --ordinals 1 291"))

    def test_the_gate_refuses_a_missing_campaign_or_the_campaign_as_its_own_output(self):
        result = self.run_step("identity_gate", check=False, CAMPAIGN="/nonexistent/campaign")
        self.assertNotEqual(result.returncode, 0)
        result = self.run_step("identity_gate", check=False, CAMPAIGN=str(self.out))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scratch", result.stderr)


class ExtraMountsInM0sDriver(unittest.TestCase):
    def run_driver(self, **env):
        full = {**os.environ, "WT": str(ROOT), "OUT": "/tmp/out", "M0_NO_RUN": "1", "MAIN": str(ROOT), "BUNDLE": "/nonexistent/bundle", **env}
        result = subprocess.run(["bash", str(DRIVER), "module", "experiments.some_module", "--root", "/out"], env=full, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_extra_mounts_are_added_verbatim_when_set(self):
        line = self.run_driver(EXTRA_MOUNTS="-v /host/a:/vla/outputs/a:ro -v /host/b:/b:ro")
        self.assertIn("-v /host/a:/vla/outputs/a:ro -v /host/b:/b:ro", line)
        self.assertTrue(line.endswith("-m experiments.some_module --root /out"))

    def test_nothing_is_added_when_it_is_unset_or_empty(self):
        base = self.run_driver()
        self.assertEqual(self.run_driver(EXTRA_MOUNTS=""), base)
        self.assertNotIn("/host/a", base)

    def test_gpu_module_runs_a_repository_module_with_the_gpu(self):
        full = {**os.environ, "WT": str(ROOT), "OUT": "/tmp/out", "M0_NO_RUN": "1", "MAIN": str(ROOT), "BUNDLE": "/nonexistent/bundle"}
        result = subprocess.run(["bash", str(DRIVER), "gpu_module", "experiments.some_module", "--x"], env=full, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertIn(ROB, line)
        self.assertIn("--runtime nvidia --gpus all", line)
        self.assertTrue(line.endswith("-m experiments.some_module --x"))


class Safety(unittest.TestCase):
    def code(self, path):
        return "\n".join(line for line in path.read_text().splitlines() if not line.lstrip().startswith("#"))

    def test_the_script_never_waits_on_docker_deletes_kills_or_touches_git(self):
        code = self.code(SCRIPT)
        for forbidden in ("docker wait", "docker rm", "docker kill", "docker stop", "prune"):
            self.assertNotIn(forbidden, code)
        for command in ("rm", "kill", "git", "mv", "cp"):
            self.assertIsNone(re.search(rf"(^|[\s;&|(]){command}\s", code), command)

    def test_the_script_is_executable_and_pins_its_constants(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))
        text = SCRIPT.read_text()
        for needle in ("CHECKPOINT=101520", "xvla_p2_force_box_v1/checkpoints", "m0_20261030", 'MAIN="${MAIN:-/home/utxowqr/smolvla_franka}"'):
            self.assertIn(needle, text)
        for adapter in ADAPTERS.values():
            module, name = adapter.split(":")
            self.assertIn(name, text)
            self.assertTrue(module in text or module.rsplit(".", 1)[0] + ".adapter" in text or "NOISY=" + module in text)

    def test_m0s_driver_stays_outside_the_shared_core(self):
        from experiments.vla_inputs_eval import contract
        for rel in contract.core_provenance(ROOT):
            self.assertNotIn("vla_inputs_eval.sh", rel)
            self.assertNotIn("vla_inputs_m3b", rel)


if __name__ == "__main__":
    unittest.main()
