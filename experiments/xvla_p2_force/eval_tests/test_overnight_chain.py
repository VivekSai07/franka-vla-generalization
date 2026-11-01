"""The overnight chain (break, dry night, pilot night 1, M2 selection) against stub drivers, a fake `docker` and a fake `nvidia-smi`.

What is proven is the chain's logic: the order, the break, what is skipped when the dry night fails or time is short, that it
never starts the comparison tier and never deletes or kills anything. Nothing here touches a GPU or starts a container.
"""
import json
import os
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT/'scripts/vla_inputs_overnight_chain.sh'

PILOT_STUB = r'''#!/usr/bin/env bash
echo "pilot_night PILOT_SMOKE_ONLY=${PILOT_SMOKE_ONLY:-} PILOT_SKIP_CONTEXT=${PILOT_SKIP_CONTEXT:-} UPDATES=${UPDATES:-} WT=$WT args=$*" >> "$FAKE_LOG"
night="$1"; mkdir -p "$night/plan" "$night/feasibility" "$night/training/control" "$night/training/force_zeroeq" "$night/training/force"
if [[ "$night" == *dry* ]]; then
  [ "${STUB_DRY_FAIL:-0}" = 1 ] && exit 3
  for d in training/control training/force_zeroeq training/force arms/A0/smoke/step_000040 arms/A1/smoke/step_000040; do
    mkdir -p "$night/$d"; echo '{"status": "complete"}' > "$night/$d/result.json"; done
  [ "${STUB_DRY_EVENT_FAIL:-0}" = 1 ] && echo '{"event": "fail", "phase": "x", "variant": "y"}' > "$night/events.jsonl"
else
  echo "# SUMMARY stub" > "$night/SUMMARY.md"
fi
exit 0
'''
EVAL_STUB = r'''#!/usr/bin/env bash
echo "m2_eval $1 WT=$WT OUT=$OUT" >> "$FAKE_LOG"
case "$1" in
  prepare) [ "${STUB_PREPARE_FAIL:-0}" = 1 ] && exit 1; mkdir -p "$OUT/arms/A2"; echo '{}' > "$OUT/arms/A2/prepared.json"; echo "PREPARED A2" ;;
  select) echo '{"comparison_checkpoints": [101520]}' > "$OUT/arms/A2/selection.json" ;;
esac
exit 0
'''
DOCKER = '#!/usr/bin/env bash\nexit 0\n'
SMI = r'''#!/usr/bin/env bash
case "$*" in
  *compute-apps*) [ "${FAKE_GPU_BUSY:-0}" = 1 ] && echo 4242 ;;
  *) echo "40, 300 MiB" ;;
esac
exit 0
'''


class Chain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.main = Path(self.tmp.name)/'main'
        self.log = Path(self.tmp.name)/'calls.log'
        self.log.write_text('')
        train = self.main/'outputs/train/xvla_p2_force_v1'
        (train/'checkpoints/step_126900').mkdir(parents=True)
        (train/'checkpoints/step_126900/checkpoint_result.json').write_text('{}')
        self.set_status('complete')
        for rel, text in (('.worktrees/pilot-run/scripts/vla_inputs_pilot_night.sh', PILOT_STUB),
                          ('.worktrees/m2-eval/scripts/vla_inputs_m2_eval.sh', EVAL_STUB)):
            path = self.main/rel
            path.parent.mkdir(parents=True)
            path.write_text(text)
            path.chmod(0o755)
        (self.main/'outputs/vla_inputs_eval/m0_20261030').mkdir(parents=True)
        bins = Path(self.tmp.name)/'bin'
        bins.mkdir()
        for name, text in (('docker', DOCKER), ('nvidia-smi', SMI)):
            (bins/name).write_text(text)
            (bins/name).chmod(0o755)
        self.bins = bins

    def set_status(self, status):
        (self.main/'outputs/train/xvla_p2_force_v1/result.json').write_text(json.dumps(dict(status=status, completed_updates=126900, elapsed_seconds=3600)))

    def run_chain(self, **env):
        full = {**os.environ, 'PATH': f'{self.bins}:{os.environ["PATH"]}', 'MAIN': str(self.main), 'FAKE_LOG': str(self.log),
                'CHAIN_POLL_SECONDS': '0.2', 'BREAK_SECONDS': '1', 'MIN_FREE_GB': '0', **env}
        return subprocess.run(['bash', str(SCRIPT)], env=full, capture_output=True, text=True, timeout=60)

    def calls(self):
        return [line for line in self.log.read_text().splitlines() if line.strip()]

    def deadline(self, seconds):
        return str(int(time.time())+seconds)


class OrderTests(Chain):
    def test_full_path_dry_night_then_pilot_night_then_m2_selection_after_the_break(self):
        started = time.time()
        result = self.run_chain(DEADLINE=self.deadline(40*3600))
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        calls = self.calls()
        kinds = [c.split()[0]+' '+(c.split()[1] if c.startswith('m2_eval') else '') for c in calls]
        self.assertEqual([k.strip() for k in kinds], ['pilot_night', 'pilot_night', 'm2_eval prepare', 'm2_eval selection', 'm2_eval select'])
        dry, night = calls[0], calls[1]
        self.assertIn('PILOT_SMOKE_ONLY=1', dry)
        self.assertIn('UPDATES=40', dry)
        self.assertIn('args=', dry)
        self.assertTrue(dry.rstrip().endswith('force'))
        self.assertNotIn('PILOT_SMOKE_ONLY=1', night)
        self.assertIn('PILOT_SKIP_CONTEXT=1', night)
        self.assertNotIn('UPDATES=40', night)
        self.assertTrue(night.rstrip().endswith('force'))
        self.assertIn(f'WT={self.main}/.worktrees/pilot-run', dry+night)
        self.assertIn(f'WT={self.main}/.worktrees/m2-eval', calls[2])
        self.assertIn('OUT='+str(self.main/'outputs/vla_inputs_eval/m0_20261030'), calls[2])
        self.assertGreaterEqual(time.time()-started, 1.0)  # the break
        self.assertIn('chain finished', result.stdout)

    def test_the_pilot_night_reuses_the_dry_nights_plan_and_feasibility(self):
        self.assertEqual(self.run_chain(DEADLINE=self.deadline(40*3600)).returncode, 0)
        nights = sorted((self.main/'outputs/vla_inputs_eval').glob('m0_pilot_night1_*'))
        self.assertEqual(len(nights), 1)
        self.assertTrue((nights[0]/'plan').is_dir() and (nights[0]/'feasibility').is_dir())

    def test_the_comparison_tier_is_never_started(self):
        self.assertEqual(self.run_chain(DEADLINE=self.deadline(40*3600)).returncode, 0)
        self.assertEqual([c for c in self.calls() if 'comparison' in c], [])
        code = '\n'.join(line for line in SCRIPT.read_text().splitlines() if not line.lstrip().startswith('#'))
        self.assertNotRegex(code, r'm2_eval\.sh"? +comparison')
        self.assertNotIn('PILOT_DECISION_RECORDED', code)

    def test_the_script_never_deletes_kills_waits_on_docker_or_switches_branches(self):
        code = '\n'.join(line for line in SCRIPT.read_text().splitlines() if not line.lstrip().startswith('#'))
        for forbidden in ('rm -', 'rm ', 'docker kill', 'docker stop', 'docker wait', 'docker rm', 'git checkout', 'git switch', 'git reset',
                          'prune', 'kill '):
            self.assertNotIn(forbidden, code)


class FallbackTests(Chain):
    def test_a_failed_dry_night_skips_the_pilots_and_runs_the_m2_selection(self):
        result = self.run_chain(DEADLINE=self.deadline(40*3600), STUB_DRY_FAIL='1')
        self.assertEqual(result.returncode, 0, result.stdout)
        kinds = [c.split()[0] for c in self.calls()]
        self.assertEqual(kinds, ['pilot_night', 'm2_eval', 'm2_eval', 'm2_eval'])
        self.assertIn('the pilots are skipped', result.stdout)

    def test_a_failure_event_in_an_otherwise_complete_dry_night_also_skips_the_pilots(self):
        result = self.run_chain(DEADLINE=self.deadline(40*3600), STUB_DRY_EVENT_FAIL='1')
        self.assertEqual(sum(c.startswith('pilot_night') for c in self.calls()), 1)
        self.assertIn('events.jsonl holds a failure', result.stdout)

    def test_too_little_time_for_the_pilots_runs_the_m2_selection_only(self):
        result = self.run_chain(DEADLINE=self.deadline(9*3600+0), EST_NIGHT=str(9*3600))
        self.assertEqual([c.split()[0] for c in self.calls()], ['m2_eval', 'm2_eval', 'm2_eval'])
        self.assertIn('M2 selection only', result.stdout)

    def test_a_failed_m2_prepare_stops_before_the_selection_stages(self):
        result = self.run_chain(DEADLINE=self.deadline(40*3600), STUB_PREPARE_FAIL='1')
        self.assertEqual([c.split()[1] for c in self.calls() if c.startswith('m2_eval')], ['prepare'])
        self.assertIn('M2 prepare failed', result.stdout)

    def test_too_little_time_for_anything_starts_nothing(self):
        result = self.run_chain(DEADLINE=self.deadline(600))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])
        self.assertIn('not started', result.stdout)

    def test_the_selection_is_skipped_when_the_pilot_night_used_up_the_time(self):
        # 3h40 left at the decision point: the dry night and the pilot night "fit" by the estimates, the selection afterwards does not.
        result = self.run_chain(DEADLINE=self.deadline(3*3600+2400), EST_DRY='1', EST_NIGHT='3600', EST_SELECTION='99999')
        self.assertEqual([c.split()[0] for c in self.calls()], ['pilot_night', 'pilot_night'])
        self.assertIn('M2 selection needs', result.stdout)


class SafetyTests(Chain):
    def test_a_training_that_died_and_is_not_being_resumed_aborts_the_chain(self):
        self.set_status('failed')
        result = self.run_chain(DEADLINE=self.deadline(40*3600))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.calls(), [])
        self.assertIn('aborting', result.stdout)

    def test_a_complete_status_without_the_final_checkpoint_aborts(self):
        (self.main/'outputs/train/xvla_p2_force_v1/checkpoints/step_126900/checkpoint_result.json').unlink()
        result = self.run_chain(DEADLINE=self.deadline(40*3600))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.calls(), [])

    def test_a_gpu_that_stays_busy_aborts_before_anything_starts(self):
        result = self.run_chain(DEADLINE=self.deadline(40*3600), FAKE_GPU_BUSY='1')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.calls(), [])

    def test_too_little_disk_aborts(self):
        result = self.run_chain(DEADLINE=self.deadline(40*3600), MIN_FREE_GB='999999')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.calls(), [])

    def test_the_cancel_file_stops_the_chain_before_the_next_phase(self):
        cancel = self.main/'outputs/train/m2_overnight_chain.cancel'
        cancel.write_text('')
        result = self.run_chain(DEADLINE=self.deadline(40*3600), BREAK_SECONDS='3')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])
        self.assertIn('cancelled', result.stdout)

    def test_it_waits_for_a_training_that_is_still_running_then_proceeds(self):
        self.set_status('training')
        (self.bins/'pgrep').write_text('#!/usr/bin/env bash\necho 1234\n')  # the supervisor is alive
        (self.bins/'pgrep').chmod(0o755)
        process = subprocess.Popen(['bash', str(SCRIPT)], env={**os.environ, 'PATH': f'{self.bins}:{os.environ["PATH"]}', 'MAIN': str(self.main),
                                   'FAKE_LOG': str(self.log), 'CHAIN_POLL_SECONDS': '0.2', 'BREAK_SECONDS': '1', 'MIN_FREE_GB': '0',
                                   'DEADLINE': self.deadline(40*3600)}, stdout=subprocess.PIPE, text=True)
        time.sleep(1.5)
        self.assertIsNone(process.poll())
        self.assertEqual(self.calls(), [])
        self.set_status('complete')
        out, _ = process.communicate(timeout=60)
        self.assertEqual(process.returncode, 0, out)
        self.assertEqual(len(self.calls()), 5)


if __name__ == '__main__':
    unittest.main()
