"""The M3 evaluation driver in dry-run mode (`M0_NO_RUN=1` prints the docker commands and runs nothing). Robosuite image, CPU.

Which commands each step prints, which pairs `compare_a2` builds, and what the script is not allowed to contain.
"""
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT/'scripts/vla_inputs_m3_eval.sh'
DRIVER = ROOT/'scripts/vla_inputs_eval.sh'
CANDIDATES = (50760, 76140, 88830, 101520, 114210, 126900)
ADAPTER = 'experiments.vla_inputs_arm_xvla_force_box.adapter:XVLAForceBoxAdapter'


class Driver(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)/'campaign'
        self.ck = Path(self.tmp.name)/'ckpt'
        self.out.mkdir()
        for c in CANDIDATES:
            (self.ck/f'step_{c:06d}').mkdir(parents=True)
            (self.ck/f'step_{c:06d}/checkpoint_result.json').write_text('{}')

    def run_step(self, step, check=True, **env):
        full = {**os.environ, 'WT': str(ROOT), 'OUT': str(self.out), 'CK': str(self.ck), 'MAIN': str(ROOT), 'M0_NO_RUN': '1',
                'BUNDLE': '/nonexistent/bundle', **env}
        result = subprocess.run(['bash', str(SCRIPT), step], env=full, capture_output=True, text=True, timeout=60)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return result

    def put(self, relative, value):
        path = self.out/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def selection(self, arm, winner, phase=101520):
        self.put(f'arms/{arm}/selection.json', dict(winner=winner, phase_checkpoint=phase, comparison_checkpoints=sorted({winner, phase})))

    def commands(self, result):
        return [line for line in result.stdout.splitlines() if line.startswith('docker ')]


class StepTests(Driver):
    def test_prepare_names_arm_a3_its_adapter_the_six_candidates_and_the_phase_checkpoint(self):
        out = self.run_step('prepare').stdout
        self.assertIn('--arm A3', out)
        self.assertIn(f'--adapter {ADAPTER}', out)
        listed = ','.join(f'{c}=/ckpt/step_{c:06d}' for c in CANDIDATES)
        self.assertIn(f'--checkpoints {listed}', out)
        self.assertIn('--phase-checkpoint 101520', out)
        self.assertIn(f'-v {self.ck}:/ckpt:ro', out)

    def test_prepare_refuses_when_a_candidate_checkpoint_is_missing(self):
        (self.ck/'step_088830/checkpoint_result.json').unlink()
        result = self.run_step('prepare', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing checkpoint step_088830', result.stderr)

    def test_steps_that_need_a_prepared_arm_or_a_selection_refuse_without_one(self):
        for step in ('smoke', 'selection', 'comparison'):
            with self.subTest(step=step):
                result = self.run_step(step, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(result.stderr, r'run (prepare|select) first')

    def test_smoke_runs_two_layouts_at_the_matched_phase_checkpoint_with_one_worker(self):
        self.put('arms/A3/prepared.json', {})
        commands = self.commands(self.run_step('smoke'))
        self.assertEqual(len(commands), 1)
        self.assertIn('--arm A3 --tier smoke --checkpoint 101520 --shard 0 --shards 1', commands[0])

    def test_selection_starts_one_worker_per_candidate_on_its_own_cores(self):
        self.put('arms/A3/prepared.json', {})
        commands = self.commands(self.run_step('selection'))
        self.assertEqual(len(commands), 6)
        for i, c in enumerate(CANDIDATES):
            mine = [x for x in commands if f'--checkpoint {c} ' in x or x.endswith(f'--checkpoint {c} --shard 0 --shards 1')]
            self.assertEqual(len(mine), 1, c)
            self.assertIn(f'--tier selection', mine[0])
            self.assertIn(f'--cpuset-cpus {i*8}-{i*8+7}', mine[0])
            self.assertIn('--shard 0 --shards 1', mine[0])

    def test_comparison_runs_a_six_worker_pool_for_each_decided_checkpoint_and_needs_no_pilot_flag(self):
        self.put('arms/A3/prepared.json', {})
        self.selection('A3', 88830)
        commands = self.commands(self.run_step('comparison'))
        self.assertEqual(len(commands), 12)
        for c in (88830, 101520):
            pool = [x for x in commands if f'--tier comparison --checkpoint {c} ' in x]
            self.assertEqual(len(pool), 6, c)
            self.assertEqual(sorted(re.findall(r'--shard (\d) --shards 6', ' '.join(pool))), list('012345'))
        self.assertTrue(all('PILOT_DECISION_RECORDED' not in x for x in commands))

    def test_compare_is_m0s_a3_against_a0_at_each_decided_checkpoint(self):
        self.selection('A3', 76140)
        commands = self.commands(self.run_step('compare'))
        self.assertEqual(len(commands), 2)
        self.assertIn('evaluate --root /out compare --arm A3 --checkpoint 76140', commands[0])
        self.assertIn('evaluate --root /out compare --arm A3 --checkpoint 101520', commands[1])

    def test_an_unknown_step_is_refused(self):
        self.assertEqual(self.run_step('train', check=False).returncode, 2)


class PairTests(Driver):
    def pairs(self, a3, a2):
        self.selection('A3', a3)
        self.selection('A2', a2)
        commands = self.commands(self.run_step('compare_a2'))
        found = [tuple(int(x) for x in re.search(r'--checkpoint (\d+) --control-checkpoint (\d+)', c).groups()) for c in commands]
        for command in commands:
            self.assertIn(f'-m experiments.xvla_p2_force_box.eval_compare --root /out compare', command)
        return found

    def test_winners_agree_gives_the_primary_pair_and_the_matched_phase_pair(self):
        self.assertEqual(self.pairs(88830, 88830), [(88830, 88830), (101520, 101520)])

    def test_a3_winner_that_is_the_phase_checkpoint_still_gives_both_pairs(self):
        self.assertEqual(self.pairs(101520, 88830), [(101520, 88830), (101520, 101520)])

    def test_a_different_winner_is_compared_with_a2s_winner_not_with_a_chosen_checkpoint(self):
        self.assertEqual(self.pairs(76140, 88830), [(76140, 88830), (101520, 101520)])

    def test_both_winners_at_the_phase_checkpoint_give_one_pair(self):
        self.assertEqual(self.pairs(101520, 101520), [(101520, 101520)])

    def test_it_refuses_without_a2s_selection(self):
        self.selection('A3', 88830)
        result = self.run_step('compare_a2', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('A2 has no selection.json', result.stderr)

    def test_final_runs_m0s_final_and_the_amended_family(self):
        commands = self.commands(self.run_step('final'))
        self.assertEqual(len(commands), 2)
        self.assertTrue(commands[0].rstrip().endswith('evaluate --root /out final'))
        self.assertTrue(commands[1].rstrip().endswith('-m experiments.xvla_p2_force_box.eval_compare --root /out final'))


class SafetyTests(unittest.TestCase):
    def code(self, path):
        return '\n'.join(line for line in path.read_text().splitlines() if not line.lstrip().startswith('#'))

    def test_the_script_never_waits_on_docker_deletes_kills_or_touches_git(self):
        code = self.code(SCRIPT)
        for forbidden in ('docker wait', 'docker rm', 'docker kill', 'docker stop', 'rm -', 'rm ', 'kill ', 'git ', 'prune'):
            self.assertNotIn(forbidden, code)

    def test_the_driver_runs_any_module_in_the_robosuite_image_without_a_gpu(self):
        result = subprocess.run(['bash', str(DRIVER), 'module', 'experiments.some_module', '--root', '/out', 'final'],
                                env={**os.environ, 'WT': str(ROOT), 'OUT': '/tmp/out', 'M0_NO_RUN': '1', 'MAIN': str(ROOT),
                                     'BUNDLE': '/nonexistent/bundle'}, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        line = result.stdout.strip()
        self.assertIn('sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e', line)
        self.assertIn('--runtime runc', line)
        self.assertNotIn('--gpus', line)
        self.assertTrue(line.endswith('-m experiments.some_module --root /out final'))

    def test_the_script_is_executable_and_names_its_arm_control_and_default_checkpoint_directory(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))
        text = SCRIPT.read_text()
        for needle in ('ARM=A3', 'CONTROL=A2', ADAPTER, 'xvla_p2_force_box_v1/checkpoints', 'PHASE=101520'):
            self.assertIn(needle, text)


if __name__ == '__main__':
    unittest.main()
