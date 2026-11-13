"""The G1 chain `scripts/gen_g1_chain.sh` against a fake driver, `docker` and `nvidia-smi`: order, skipping, resuming, stopping, the
GPU and disk guards, and the winner deciding the second benchmark. Only bash runs."""
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHAIN = ROOT / "scripts/gen_g1_chain.sh"
CANDIDATES = (50760, 76140, 88830, 101520, 114210, 126900)

FAKE_DRIVER = r"""#!/usr/bin/env bash
echo "$*" >> "$LOGF"
[ "${FAIL_AT:-}" = "$1" ] && exit 5
ck() { printf '%06d' "$1"; }
done_stage() { mkdir -p "$1"; printf '{\n  "status": "complete"\n}\n' > "$1/result.json"; }
case "$1" in
  prepare) mkdir -p "$OUT/arms/A1"; echo '{}' > "$OUT/arms/A1/prepared.json" ;;
  stage) done_stage "$OUT/arms/A1/$2/step_$(ck "$3")" ;;
  selection) for c in 50760 76140 88830 101520 114210 126900; do done_stage "$OUT/arms/A1/selection/step_$(ck $c)"; done ;;
  select) echo "{\"winner\": ${WINNER:-114210}}" > "$OUT/arms/A1/selection.json" ;;
  winner) python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['winner'])" "$OUT/arms/A1/selection.json" ;;
  compare) echo '{}' > "$OUT/arms/A1/compare_A2_step_$(ck "$2").json" ;;
  m0prepare) mkdir -p "$M0C/arms/A1"; echo '{}' > "$M0C/arms/A1/prepared.json" ;;
  m0comparison) done_stage "$M0C/arms/A1/comparison/step_$(ck "${WINNER:-114210}")" ;;
  m0compare) echo '{}' > "$M0C/arms/A1/compare_A2_step_$(ck "${WINNER:-114210}")_vs_088830.json" ;;
esac
exit 0
"""


class Chain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.out, self.m0c, self.wt, self.bin = base / "out", base / "m0", base / "wt", base / "bin"
        (self.out / "arms/A2").mkdir(parents=True)
        (self.out / "arms/A2/prepared.json").write_text("{}")
        (self.wt / "scripts").mkdir(parents=True)
        self.bin.mkdir()
        self.log = base / "driver.log"
        driver = self.wt / "scripts/gen_g1_eval.sh"
        driver.write_text(FAKE_DRIVER)
        driver.chmod(driver.stat().st_mode | stat.S_IEXEC)
        for name, body in (("docker", "#!/usr/bin/env bash\nexit 0\n"),
                           ("nvidia-smi", '#!/usr/bin/env bash\n[ -n "${FAKE_GPU_BUSY:-}" ] && echo 4242\nexit 0\n')):
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)

    def run_chain(self, **env):
        e = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "WT": str(self.wt), "OUT": str(self.out), "M0C": str(self.m0c),
             "MAIN": str(self.out), "LOGF": str(self.log), "CHAIN_POLL_SECONDS": "0", **env}
        return subprocess.run(["bash", str(CHAIN)], env=e, capture_output=True, text=True, timeout=120)

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_the_whole_sequence_in_order_with_the_winner_deciding_the_second_benchmark(self):
        r = self.run_chain()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        order = [c.split()[0] + (" " + c.split()[1] if c.startswith("stage") else "") for c in self.calls() if c != "winner"]
        self.assertEqual(order, ["prepare", "stage smoke", "selection", "select", "stage benchmark", "compare",
                                 "stage benchmark", "compare", "m0prepare", "m0comparison", "m0compare"])
        self.assertIn("stage benchmark 114210 6 7", self.calls())
        self.assertIn("stage benchmark 101520 6 7", self.calls())
        self.assertTrue((self.out / "chain_g1.done").is_file())

    def test_a_winner_of_101520_runs_one_benchmark(self):
        r = self.run_chain(WINNER="101520")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(sum(1 for c in self.calls() if c.startswith("stage benchmark")), 1)
        self.assertEqual(sum(1 for c in self.calls() if c.startswith("compare")), 1)

    def test_rerunning_a_finished_chain_does_nothing(self):
        self.run_chain()
        before = self.calls()
        r = self.run_chain()
        self.assertEqual(r.returncode, 0)
        self.assertEqual([c for c in self.calls()[len(before):] if c != "winner"], [])

    def test_it_stops_at_the_first_failing_step_and_resumes_from_there(self):
        r = self.run_chain(FAIL_AT="selection")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.calls(), ["prepare", "stage smoke 126900 6 7", "selection"])
        self.assertIn("selection tier, six candidates failed", (self.out / "chain_g1.log").read_text())
        self.assertFalse((self.out / "chain_g1.done").exists())
        self.log.unlink()
        r = self.run_chain()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.calls()[0], "selection")                      # prepare and smoke were not repeated

    def test_an_incomplete_stage_result_is_not_skipped(self):
        self.run_chain()
        result = self.out / "arms/A1/benchmark/step_114210/result.json"
        result.write_text('{\n  "status": "incomplete"\n}\n')
        self.log.unlink()
        self.run_chain()
        self.assertIn("stage benchmark 114210 6 7", self.calls())

    def test_a_missing_selection_stage_stops_before_select(self):
        driver = self.wt / "scripts/gen_g1_eval.sh"
        driver.write_text(FAKE_DRIVER.replace('for c in 50760 76140 88830 101520 114210 126900; do', 'for c in 50760 76140 88830 101520 114210; do'))
        r = self.run_chain()
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("select", self.calls())
        self.assertIn("selection stage 126900 is not complete", (self.out / "chain_g1.log").read_text())

    def test_it_waits_for_a_busy_gpu_then_stops_resumably(self):
        r = self.run_chain(FAKE_GPU_BUSY="1", MAX_GPU_WAIT="0")
        self.assertEqual(r.returncode, 3)
        self.assertEqual(self.calls(), [])
        self.assertIn("GPU held by another job", (self.out / "chain_g1.log").read_text())

    def test_it_stops_when_the_disk_is_nearly_full(self):
        r = self.run_chain(MIN_FREE_GB="100000000")
        self.assertEqual(r.returncode, 4)
        self.assertEqual(self.calls(), [])
        self.assertIn("ask the user", (self.out / "chain_g1.log").read_text())

    def test_a_cancel_file_stops_before_the_next_step(self):
        (self.out / "chain_g1.cancel").write_text("")
        r = self.run_chain()
        self.assertEqual(r.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_it_refuses_without_g0s_a2_arm(self):
        (self.out / "arms/A2/prepared.json").unlink()
        r = self.run_chain()
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
