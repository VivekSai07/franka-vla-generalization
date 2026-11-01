"""The pilot night driver, run against a fake `docker` and a fake `nvidia-smi`: order, gating, failure handling, resumption.

Nothing here starts a container or touches a GPU: `docker` is a shell stub on PATH that logs every call and, for the training
runner, writes the result files a real run would. What is proven is the driver's logic (which command runs when, what is skipped,
what is never retried), not that a real pilot trains.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT/'scripts/vla_inputs_pilot_night.sh'
FAKE_DOCKER = r'''#!/usr/bin/env bash
echo "$@" >> "$FAKE_LOG"
case "$1" in
  run)
    args=("$@"); name=""; out_arg=""; runner=0; detached=0; fail=0
    for ((i = 0; i < ${#args[@]}; i++)); do
      case "${args[$i]}" in
        --name) name="${args[$((i + 1))]}" ;;
        --out) out_arg="${args[$((i + 1))]}" ;;
        -d) detached=1 ;;
        experiments.xvla_p2_pilot.runner) runner=1 ;;
      esac
    done
    if [ -n "${FAKE_FAIL_MATCH:-}" ] && [[ "$*" == *"$FAKE_FAIL_MATCH"* ]]; then fail=1; fi
    if [ "$runner" = 1 ]; then
      out_host=""
      for a in "${args[@]}"; do case "$a" in *":$out_arg:rw") out_host="${a%%:*}" ;; esac; done
      mkdir -p "$out_host" "$FAKE_STATE"
      echo '{}' > "$out_host/run_claim.json"
      if [ "$fail" = 1 ]; then
        echo '{"status": "failed", "error_type": "FakeError", "error": "injected"}' > "$out_host/result.json"
        echo 1 > "$FAKE_STATE/exit_$name"
      else
        echo '{"status": "complete", "recipe_version": "pilot_v1", "max_optimizer_updates": 300}' > "$out_host/result.json"
        case "$out_arg" in *_zeroeq) ;; *) mkdir -p "$out_host/checkpoints/step_000300" ;; esac
        echo 0 > "$FAKE_STATE/exit_$name"
      fi
    elif [ -n "$name" ]; then
      mkdir -p "$FAKE_STATE"; if [ "$fail" = 1 ]; then echo 1 > "$FAKE_STATE/exit_$name"; else echo 0 > "$FAKE_STATE/exit_$name"; fi
    fi
    [ "$detached" = 1 ] && echo fakeid
    [ "$detached" = 1 ] && exit 0
    exit "$fail" ;;
  ps) exit 0 ;;
  inspect) cat "$FAKE_STATE/exit_${@: -1}" 2>/dev/null || echo 0 ;;
  logs) echo "fake log" ;;
  rm) exit 0 ;;
  *) exit 0 ;;
esac
'''


class Night(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        (base/'bin').mkdir()
        self.log, self.state, self.night = base/'docker.log', base/'state', base/'night'
        (base/'bin/docker').write_text(FAKE_DOCKER)
        (base/'bin/docker').chmod(0o755)
        self.smi(None)
        self.log.write_text('')

    def smi(self, pids):
        path = Path(self.tmp.name)/'bin/nvidia-smi'
        path.write_text('#!/usr/bin/env bash\n' + (f'echo {pids}\n' if pids else 'exit 0\n'))
        path.chmod(0o755)

    def run_night(self, *variants, night=None, **env):
        full = {**os.environ, 'PATH': f'{self.tmp.name}/bin:{os.environ["PATH"]}', 'WT': str(ROOT), 'MAIN': '/main', 'BUNDLE': '/bundle',
                'FAKE_LOG': str(self.log), 'FAKE_STATE': str(self.state), 'PILOT_POLL_SECONDS': '0', 'WORKERS': '2', 'CORES': '2',
                'FEAS_SHARDS': '2', 'FEAS_CORES': '2', **env}
        return subprocess.run(['bash', str(SCRIPT), str(night or self.night), *variants], env=full, capture_output=True, text=True)

    def calls(self):
        return [line for line in self.log.read_text().splitlines() if line.strip()]

    def index(self, pattern, start=0, calls=None):
        calls = calls if calls is not None else self.calls()
        for i in range(start, len(calls)):
            if re.search(pattern, calls[i]):
                return i
        self.fail(f'no call matching {pattern!r} after {start}')

    def events(self):
        return [json.loads(line) for line in (self.night/'events.jsonl').read_text().splitlines()]


class OrderTests(Night):
    def test_steps_run_in_the_planned_order_for_every_variant(self):
        result = self.run_night('force', 'box')
        self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
        calls = self.calls()
        at = lambda pattern, start=0: self.index(pattern, start, calls)
        plan = at(r'vla_inputs_pilot_plan')
        feasibility = at(r'vla_inputs_eval\.feasibility', plan)
        control = at(r'runner --train --variant control --out /out/control', feasibility)
        prep_a0 = at(r'prepare --arm A0 .*XVLAPilotAdapter', control)
        work_a0 = at(r'worker --arm A0 --tier comparison', prep_a0)
        prep_a9 = at(r'prepare --arm A9 .*XVLAAdapter .*--phase-checkpoint 101520', work_a0)
        work_a9 = at(r'worker --arm A9 --tier comparison --checkpoint 101520', prep_a9)
        for k, name in enumerate(('force', 'box'), start=1):
            gate = at(rf'runner --zero-equivalence --variant {name} --out /out/{name}_zeroeq', work_a9)
            train = at(rf'runner --train --variant {name} --out /out/{name}\b', gate)
            prepare = at(rf'prepare --arm A{k} .*XVLAPilotAdapter .*--checkpoints 300=/out/training/{name}/checkpoints/step_000300 '
                         rf'--phase-checkpoint 300', train)
            selection = at(rf'worker --arm A{k} --tier selection --checkpoint 300', prepare)
            stage = at(rf'worker --arm A{k} --tier comparison --checkpoint 300', selection)
            compare = at(rf'evaluate --root /out compare --arm A{k} --checkpoint 300', stage)
            work_a9 = compare

    def test_the_night_record_names_the_variants_their_arms_and_the_checkpoint(self):
        self.assertEqual(self.run_night('force', 'box').returncode, 0)
        night = json.loads((self.night/'night.json').read_text())
        self.assertEqual((night['recipe_version'], night['updates'], night['checkpoint'], night['control_arm'], night['context_arm']),
                         ('pilot_v1', 300, 300, 'A0', 'A9'))
        self.assertEqual(night['variants'], [['force', 'A1'], ['box', 'A2']])

    def test_every_phase_leaves_start_and_end_events(self):
        self.assertEqual(self.run_night('force').returncode, 0)
        events = [(e['event'], e['phase'], e['variant']) for e in self.events()]
        for phase, variant in (('train', 'control'), ('zero_equivalence', 'force'), ('train', 'force'), ('evaluate', 'force'),
                               ('prepare', 'A0'), ('plan', 'all'), ('feasibility', 'all')):
            self.assertIn(('start', phase, variant), events)
            self.assertIn(('end', phase, variant), events)
        self.assertEqual([e for e in events if e[0] == 'fail'], [])

    def test_a_dry_run_updates_count_goes_to_training_only_and_the_gate_runs_its_own_30(self):
        self.assertEqual(self.run_night('force', UPDATES='300').returncode, 0)
        trains = [c for c in self.calls() if re.search(r'runner --train', c)]
        self.assertTrue(trains and all('--updates 300' in c for c in trains))
        gates = [c for c in self.calls() if 'runner --zero-equivalence' in c]
        self.assertTrue(gates and all('--updates' not in c for c in gates))

    def test_the_smoke_layouts_run_first_for_every_arm_only_when_asked(self):
        self.assertEqual(self.run_night('force').returncode, 0)
        self.assertEqual([c for c in self.calls() if '--tier smoke' in c], [])
        self.log.write_text('')
        shutil.rmtree(self.night)
        self.assertEqual(self.run_night('force', PILOT_SMOKE='1').returncode, 0)
        calls = self.calls()
        for arm, ckpt in (('A0', 300), ('A9', 101520), ('A1', 300)):
            smoke = self.index(rf'worker --arm {arm} --tier smoke --checkpoint {ckpt}', 0, calls)
            self.index(rf'worker --arm {arm} --tier comparison --checkpoint {ckpt}', smoke, calls)

    def test_the_summary_is_written_at_the_end(self):
        self.assertEqual(self.run_night('force').returncode, 0)
        self.assertIn('experiments.xvla_p2_pilot.summary /out', self.calls()[-1])

    def test_a_smoke_only_night_proves_the_gpu_path_and_never_runs_the_pilot_tier_or_the_comparison(self):
        result = self.run_night('force', PILOT_SMOKE_ONLY='1', PILOT_SKIP_CONTEXT='1', UPDATES='40')
        self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
        calls = self.calls()
        for arm in ('A0', 'A1'):
            self.index(rf'worker --arm {arm} --tier smoke --checkpoint 300', 0, calls)
        self.assertEqual([c for c in calls if '--tier comparison' in c], [])
        self.assertEqual([c for c in calls if re.search(r'evaluate --root /out compare ', c)], [])
        self.index(r'runner --zero-equivalence --variant force', 0, calls)  # the GPU-only parts all ran
        self.index(r'runner --train --variant force .*--updates 40', 0, calls)
        self.index(r'worker --arm A1 --tier selection', 0, calls)
        self.assertEqual([e for e in self.events() if e['event'] == 'fail'], [])

    def test_the_pilot_tier_still_runs_without_the_smoke_only_knob(self):
        self.assertEqual(self.run_night('force', PILOT_SKIP_CONTEXT='1').returncode, 0)
        self.index(r'worker --arm A1 --tier comparison', 0)
        self.index(r'evaluate --root /out compare --arm A1', 0)


class SafetyTests(Night):
    def test_every_container_mounts_the_worktree_read_only_and_only_the_night_or_a_run_directory_writable(self):
        self.assertEqual(self.run_night('force').returncode, 0)
        for call in self.calls():
            if not call.startswith('run'):
                continue
            self.assertIn(f'-v {ROOT}:/vla:ro', call)
            writable = re.findall(r'-v (\S+):(\S+):rw', call)
            self.assertLessEqual(len(writable), 2, call)  # /tmp is a tmpfs; /out is the night root or one run directory
            for host, container in writable:
                self.assertTrue(container.startswith('/out'), call)
                self.assertTrue(host.startswith(str(self.night)), call)
            self.assertIn('--network none', call)
            self.assertIn('--read-only', call)

    def test_the_gpu_is_given_to_training_and_workers_only(self):
        self.assertEqual(self.run_night('force').returncode, 0)
        for call in self.calls():
            if not call.startswith('run'):
                continue
            gpu = '--runtime nvidia' in call
            wants = 'runner --' in call or ' worker --arm' in call
            self.assertEqual(gpu, wants, call)

    def test_the_script_never_waits_on_docker_kills_or_switches_branches(self):
        text = SCRIPT.read_text()
        code = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
        for forbidden in ('docker wait', 'docker kill', 'docker stop', 'rm -f', 'git checkout', 'git switch', 'git reset', 'docker rmi', 'prune'):
            self.assertNotIn(forbidden, code)
        self.assertIn('docker ps', code)

    def test_a_busy_gpu_stops_the_night_before_anything_runs(self):
        self.smi('12345')
        result = self.run_night('force')
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertEqual(self.run_night('force', PILOT_ALLOW_BUSY='1').returncode, 0)

    def test_bad_invocations_are_refused_before_anything_runs(self):
        for variants, night in ((('nonesuch',), None), (('control',), None), (('force', 'force'), None), (('Force',), None),
                                (tuple(f'v{i}' for i in range(9)), None), (('force',), 'relative/night')):
            with self.subTest(variants=variants, night=night):
                result = self.run_night(*variants, night=night)
                self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertNotEqual(subprocess.run(['bash', str(SCRIPT)], capture_output=True, env={**os.environ, 'WT': str(ROOT)}).returncode, 0)


class FailureTests(Night):
    def test_a_failed_gate_skips_that_variant_only_is_reported_and_is_not_retried(self):
        result = self.run_night('force', 'box', FAKE_FAIL_MATCH='pilot-zero_equivalence-force')
        self.assertEqual(result.returncode, 3, result.stderr+result.stdout)
        calls = self.calls()
        self.assertEqual(len([c for c in calls if 'runner --zero-equivalence --variant force' in c]), 1)
        self.assertEqual([c for c in calls if 'runner --train --variant force' in c], [])
        self.assertEqual([c for c in calls if re.search(r'--arm A1\b', c)], [])
        self.assertTrue([c for c in calls if 'runner --train --variant box' in c])
        failures = [e for e in self.events() if e['event'] == 'fail']
        self.assertEqual([(e['phase'], e['variant']) for e in failures], [('zero_equivalence', 'force')])

    def test_a_failed_training_is_kept_not_retried_and_reported(self):
        result = self.run_night('force', 'box', FAKE_FAIL_MATCH='pilot-train-force')
        self.assertEqual(result.returncode, 3)
        self.assertEqual([c for c in self.calls() if re.search(r'--arm A1\b', c)], [])
        self.assertEqual(json.loads((self.night/'training/force/result.json').read_text())['status'], 'failed')
        self.assertEqual([(e['phase'], e['variant']) for e in self.events() if e['event'] == 'fail'], [('train', 'force')])
        # A second invocation neither retries the failed training nor repeats the finished work.
        before = len(self.calls())
        again = self.run_night('force', 'box')
        self.assertEqual(again.returncode, 3)
        later = self.calls()[before:]
        self.assertEqual([c for c in later if 'runner --train --variant force' in c or 'runner --zero-equivalence --variant force' in c], [])
        self.assertEqual([c for c in later if 'runner --train --variant control' in c], [])
        self.assertTrue(any(e['event'] == 'fail' and 'not retried' in e['detail'] for e in self.events()))

    def test_a_failed_control_stops_the_night(self):
        result = self.run_night('force', FAKE_FAIL_MATCH='pilot-train-control')
        self.assertEqual(result.returncode, 2)
        self.assertIn('the control did not train', result.stderr)
        self.assertEqual([(e['phase'], e['variant']) for e in self.events() if e['event'] == 'fail'], [('train', 'control')])
        self.assertEqual([c for c in self.calls() if '--variant force' in c], [])

    def test_an_incomplete_evaluation_skips_the_comparison_and_is_reported(self):
        result = self.run_night('force', 'box', FAKE_FAIL_MATCH='collect --arm A1 --tier comparison')
        self.assertEqual(result.returncode, 3, result.stderr+result.stdout)
        calls = self.calls()
        self.assertEqual([c for c in calls if 'compare --arm A1' in c], [])
        self.assertTrue([c for c in calls if 'compare --arm A2' in c])
        self.assertIn(('evaluate', 'force'), [(e['phase'], e['variant']) for e in self.events() if e['event'] == 'fail'])

    def test_a_failure_in_the_context_arm_does_not_stop_the_variants(self):
        result = self.run_night('force', FAKE_FAIL_MATCH='prepare --arm A9')
        self.assertEqual(result.returncode, 0, result.stderr+result.stdout)
        self.assertTrue([c for c in self.calls() if 'runner --train --variant force' in c])
        self.assertIn(('evaluate', 'context'), [(e['phase'], e['variant']) for e in self.events() if e['event'] == 'fail'])


class ResumeTests(Night):
    def test_a_finished_night_repeats_no_training_and_no_gate(self):
        self.assertEqual(self.run_night('force').returncode, 0)
        before = len(self.calls())
        self.assertEqual(self.run_night('force').returncode, 0)
        later = self.calls()[before:]
        self.assertEqual([c for c in later if 'runner --' in c], [])

    def test_a_prepared_arm_and_a_complete_stage_are_skipped(self):
        self.assertEqual(self.run_night('force').returncode, 0)
        (self.night/'arms/A0').mkdir(parents=True, exist_ok=True)
        (self.night/'arms/A0/prepared.json').write_text('{}')
        stage = self.night/'arms/A0/comparison/step_000300'
        stage.mkdir(parents=True)
        (stage/'result.json').write_text('{"status": "complete"}')
        before = len(self.calls())
        self.assertEqual(self.run_night('force').returncode, 0)
        later = self.calls()[before:]
        self.assertEqual([c for c in later if 'prepare --arm A0' in c or 'worker --arm A0 --tier comparison' in c], [])
        self.assertTrue([c for c in later if 'prepare --arm A1' in c])  # not marked prepared: it runs again

    def test_the_plan_and_the_feasibility_are_made_once(self):
        self.assertEqual(self.run_night('force').returncode, 0)
        (self.night/'plan').mkdir(exist_ok=True)
        (self.night/'plan/plan.sha256').write_text('x')
        (self.night/'feasibility').mkdir(exist_ok=True)
        (self.night/'feasibility/feasibility.json').write_text('{}')
        before = len(self.calls())
        self.assertEqual(self.run_night('force').returncode, 0)
        later = self.calls()[before:]
        self.assertEqual([c for c in later if 'vla_inputs_pilot_plan' in c or 'vla_inputs_eval.feasibility' in c], [])


class DryRunTests(unittest.TestCase):
    def test_pilot_no_run_prints_the_commands_and_runs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(['bash', str(SCRIPT), f'{tmp}/night', 'force'], capture_output=True, text=True,
                                    env={**os.environ, 'WT': str(ROOT), 'PILOT_NO_RUN': '1'})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('docker run -d --name pilot-train-control', result.stdout)
            self.assertFalse(Path(tmp, 'night').exists())


if __name__ == '__main__':
    unittest.main()
