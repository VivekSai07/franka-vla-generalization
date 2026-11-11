"""The G1 recipe is M2's (A2's), read from M2's contract and runner and compared here, never retyped.

One variable changes between A2's run and this one: the data. So every constant, schedule, sampler, optimiser setting, config and
update function must equal M2's, apart from what has to differ because the data is another dataset. This file, the equivalence
test and the hygiene test are the only places in the package that import M2.
"""
import ast
import dataclasses
import importlib
import inspect
import itertools
import textwrap
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_contract import FRAMES, TASK_OF_EPISODE

# What must differ: where the data and the outputs live, what the data is (its seal, its records, its split and counts), the
# files each contract binds. Everything else is compared.
DATA_NAMES = {'PARENT', 'DATA', 'SEAL', 'SEAL_RECORD', 'VALIDATION_RECORD', 'TRAIN', 'VAL', 'TRAINING_ROWS', 'COMPLETE_STARTS',
              'VALIDATION_ROWS', 'TOTAL_FRAMES', 'EPISODES'}
OUTPUT_NAMES = {'OUTPUT', 'SCRATCH', 'PROOF', 'EQUIVALENCE', 'ADMISSION', 'CPU_ADMISSION'}
BOUND_FILE_NAMES = {'SOURCE_FILES', 'REFERENCE_FILES'}
ONLY_IN_M2 = {'ZERO_EQUIVALENCE', 'TRAINED_PAIRS'}  # M2's zero-equivalence gate; P2's trained pairs (G1 derives its from the plan)
ONLY_IN_G1 = {'EPISODES', 'TRAIN_EPISODES', 'M2_EQUIVALENCE'}
ALLOWED_TO_DIFFER = DATA_NAMES | OUTPUT_NAMES | BOUND_FILE_NAMES

# Functions that must be the same code, apart from docstrings and comments (those name the data).
SAME_CONTRACT_FUNCTIONS = ('learning_rate', 'set_learning_rate', 'make_config', 'checkpoint_updates', 'checkpoint_directory',
                           'snapshot_updates', 'require_marker', 'verify_checkpoints', 'validate_scratch', 'validate_equivalence',
                           'validate_resume_proof', 'sample_batches', 'collate_rows', 'policy_row', 'observation',
                           'check_model_inputs', 'joined_state', 'selected_stats', 'approved_instruction', 'task_index_map',
                           'require_fields', 'digest_json', 'is_mountpoint', 'source_hashes', 'statistics_json', 'semantic_gate',
                           'admission', 'prepare_data', 'JoinedDataset')
SAME_RUNNER_FUNCTIONS = ('one_update', 'verify_precision_fast', 'serial_fetch', 'LockedDataset', 'Prefetcher', 'processor_check',
                         'load_fresh_policy', 'adam_roundtrip', '_reload_and_verify', 'save_checkpoint', 'save_snapshot',
                         'load_snapshot', 'quarantine', 'compare_with_reference', 'capture_rng', 'restore_rng', 'state_digest',
                         'optimizer_digest', 'require_disk', 'require_resources', 'should_pause', 'cool_down', 'gpu_telemetry',
                         'parse_telemetry', 'too_hot', 'failure', 'equivalence', 'statistics_json')
SAME_RUNNER_CONSTANTS = ('EXPECTED_PARAM_TENSORS', 'EXPECTED_PARAM_NUMEL', 'EQUIVALENCE_UPDATES', 'TELEMETRY_EVERY', 'HOT_C', 'COOL_C',
                         'HOT_LIMIT_S', 'SNAPSHOT_WRITE_S', 'SNAPSHOT_FILES')


def module_constants(module):
    return {name: value for name, value in vars(module).items()
            if name.isupper() and not name.startswith('_') and not inspect.ismodule(value) and not callable(value)}


def code_of(thing):
    """The code of a function or class as an AST dump without docstrings: equal means the same code."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(thing)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


class RecipeIsM2sTests(unittest.TestCase):
    def setUp(self):
        self.m2 = importlib.import_module('experiments.xvla_p2_force.contract')
        self.c = importlib.import_module('experiments.xvla_g1_rand.contract')

    def test_the_recipe_is_m2s_apart_from_the_purpose_and_the_episode_lists(self):
        for preflight in (True, False):
            a, b = self.m2.recipe(preflight), self.c.recipe(preflight)
            self.assertEqual(set(a), set(b))
            self.assertEqual(a.pop('purpose'), 'xvla_p2_force_preflight' if preflight else 'xvla_p2_force')
            self.assertEqual(b.pop('purpose'), 'xvla_g1_rand_preflight' if preflight else 'xvla_g1_rand')
            self.assertEqual((a.pop('train_episodes'), a.pop('validation_episodes')), (list(self.m2.TRAIN), list(self.m2.VAL)))
            self.assertEqual((b.pop('train_episodes'), b.pop('validation_episodes')), (list(self.c.TRAIN), list(self.c.VAL)))
            self.assertEqual(a, b)

    def test_every_constant_is_m2s_apart_from_the_data_and_the_outputs(self):
        a, b = module_constants(self.m2), module_constants(self.c)
        self.assertEqual(set(a)-ONLY_IN_M2, set(b)-ONLY_IN_G1, 'the two contracts must name the same constants')
        self.assertTrue(ONLY_IN_M2 & set(a) and ONLY_IN_G1 <= set(b))
        compared = sorted((set(a) & set(b))-ALLOWED_TO_DIFFER)
        for name in ('INSTRUCTIONS', 'CAMERAS', 'CHECKPOINT_UPDATES', 'TOTAL_UPDATES', 'SNAPSHOT_EVERY', 'KEEP_SNAPSHOTS', 'SEED',
                     'FRAMES_PER_EPISODE', 'CHUNK_SIZE', 'REAL_STATE_DIM', 'JOINT_DIM', 'OBS', 'DATASET_OBS', 'IMAGE', 'CACHE', 'ROOT'):
            self.assertIn(name, compared)
        for name in compared:
            self.assertEqual(b[name], a[name], name)

    def test_what_differs_is_exactly_the_data_the_outputs_and_the_bound_files(self):
        m2, c = self.m2, self.c
        for name in sorted(ALLOWED_TO_DIFFER & set(module_constants(m2)) & set(module_constants(c))):
            self.assertNotEqual(getattr(c, name), getattr(m2, name), name)
        self.assertEqual((m2.TRAINING_ROWS, m2.COMPLETE_STARTS, m2.VALIDATION_ROWS, m2.TOTAL_FRAMES), (216_960, 203_040, 27_120, 244_080))
        self.assertEqual(m2.TRAINING_ROWS, len(m2.TRAIN)*m2.FRAMES_PER_EPISODE)
        self.assertEqual(c.TRAINING_ROWS, len(c.TRAIN)*c.FRAMES_PER_EPISODE)

    def test_the_budget_is_equal_updates_not_equal_exposure(self):
        m2, c = self.m2, self.c
        self.assertEqual(c.TOTAL_UPDATES, m2.TOTAL_UPDATES)
        self.assertEqual(m2.TOTAL_UPDATES*8, m2.COMPLETE_STARTS*5)  # A2: 5.0 epochs of 203,040 starts
        self.assertEqual(round(c.TOTAL_UPDATES*8/c.COMPLETE_STARTS, 2), 0.89)  # here: 0.89 epoch of 1,142,100

    def test_the_output_directories_cannot_collide_with_m2s(self):
        m2, c = self.m2, self.c
        mine = {c.OUTPUT, c.SCRATCH, c.PROOF, c.EQUIVALENCE, c.ADMISSION, c.M2_EQUIVALENCE}
        theirs = {m2.OUTPUT, m2.SCRATCH, m2.PROOF, m2.EQUIVALENCE, m2.ADMISSION, m2.ZERO_EQUIVALENCE}
        self.assertEqual(len(mine), 6)
        self.assertFalse(mine & theirs)
        self.assertEqual(c.OUTPUT.name, 'xvla_g1_rand_v1')
        self.assertNotEqual(c.PARENT, m2.PARENT)

    def test_schedules_and_learning_rates_are_m2s(self):
        m2, c = self.m2, self.c
        for preflight in (True, False):
            self.assertEqual(c.checkpoint_updates(preflight), m2.checkpoint_updates(preflight))
            self.assertEqual(c.snapshot_updates(preflight), m2.snapshot_updates(preflight))
        for update in sorted({1, 250, 499, 500, 501, 997, 50_760, 76_140, 101_520, 126_899, 126_900}|set(range(1, 126_901, 1999))):
            self.assertEqual(c.learning_rate(update), m2.learning_rate(update), update)
        for module in (m2, c):
            with self.assertRaises(ValueError):
                module.learning_rate(126_901)

    def test_the_optimizer_groups_get_m2s_learning_rates_and_checks(self):
        m2, c = self.m2, self.c

        def optimizer(weight_decay_scale=1.):
            parameter = lambda: [torch.nn.Parameter(torch.zeros(1))]
            return torch.optim.AdamW([{'params': parameter(), 'name': name, 'lr': 1., 'betas': (.9, .95), 'eps': 1e-8,
                                       'weight_decay': 1e-4*(.1 if name == 'vlm' else 1.)*weight_decay_scale}
                                      for name in ('vlm', 'soft_prompts', 'other')])
        for update in (1, 250, 500, 60_000, 126_900):
            a, b = optimizer(), optimizer()
            m2.set_learning_rate(a, update)
            c.set_learning_rate(b, update)
            self.assertEqual([g['lr'] for g in a.param_groups], [g['lr'] for g in b.param_groups])
        for module in (m2, c):
            with self.assertRaises(ValueError):
                module.set_learning_rate(optimizer(2.), 1)

    def test_the_sampler_draws_m2s_batches_in_m2s_order(self):
        m2, c = self.m2, self.c
        small = (0, 1, 2, 3, 4)
        starts = [i*FRAMES+j for i in small for j in range(423)]
        with patch.object(m2, 'TRAIN', small), patch.object(c, 'TRAIN', small):
            ours = list(itertools.islice(c.sample_batches(starts), 300))
            theirs = list(itertools.islice(m2.sample_batches(starts), 300))
        self.assertEqual(ours, theirs)
        self.assertEqual(len(set(itertools.chain.from_iterable(ours))), 1200)
        for module in (m2, c):  # unpatched, each names its own full training set
            with self.assertRaises(ValueError):
                next(module.sample_batches(starts))

    def test_the_first_nine_state_statistics_equal_m2s_byte_for_byte(self):
        """Both `selected_stats` over the same 3 episodes of synthetic rows (their extras stubbed: only the first 9 are compared)."""
        m2, c = self.m2, self.c
        small = 3*FRAMES
        generator = np.random.default_rng(5)
        state, action = (generator.normal(size=(small, w)).astype(np.float32) for w in (9, 8))

        class Rows:
            def __getitem__(self, i):
                return dict(episode_index=i//FRAMES, frame_index=i % FRAMES, task_index=TASK_OF_EPISODE[i//FRAMES],
                            action=torch.from_numpy(action[i]), **{'observation.state': torch.from_numpy(state[i])})

        stub = np.zeros(8, np.float32)
        with patch.object(m2, 'TRAINING_ROWS', small), patch.object(c, 'TRAINING_ROWS', small), \
                patch.object(m2.forces, 'extra_state', return_value=stub), patch.object(c.forces, 'extra_state', return_value=stub):
            selected = list(range(small))
            ours, theirs = c.selected_stats(Rows(), selected, TASK_OF_EPISODE), m2.selected_stats(Rows(), selected, TASK_OF_EPISODE)
        for name in ('mean', 'std', 'min', 'max'):
            self.assertEqual(ours['observation.state'][name][:9].numpy().tobytes(), theirs['observation.state'][name][:9].numpy().tobytes(), name)
            self.assertEqual(ours['action'][name].numpy().tobytes(), theirs['action'][name].numpy().tobytes(), name)

    def test_the_contract_functions_are_the_same_code(self):
        for name in SAME_CONTRACT_FUNCTIONS:
            with self.subTest(function=name):
                self.assertEqual(code_of(getattr(self.c, name)), code_of(getattr(self.m2, name)))

    def test_the_comparison_can_tell_code_from_docstrings(self):
        def one():
            def f(x):
                """one"""
                return x+1
            return f

        def two():
            def f(x):
                """two"""
                return x+1
            return f

        def three():
            def f(x):
                """one"""
                return x+2
            return f
        self.assertEqual(code_of(one()), code_of(two()))
        self.assertNotEqual(code_of(one()), code_of(three()))

    def test_the_model_config_is_m2s(self):
        m2, c = self.m2, self.c
        if not (c.CACHE/'model/config.json').exists():
            self.skipTest('base bundle not mounted')
        a, b = m2.make_config('cpu'), c.make_config('cpu')
        self.assertEqual(a.input_features['observation.state'].shape, (17,))
        self.assertEqual(dataclasses.asdict(a), dataclasses.asdict(b))


class RunnerIsM2sTests(unittest.TestCase):
    def setUp(self):
        self.m2 = importlib.import_module('experiments.xvla_p2_force.runner')
        self.r = importlib.import_module('experiments.xvla_g1_rand.runner')

    def test_the_update_math_and_everything_around_it_is_the_same_code(self):
        for name in SAME_RUNNER_FUNCTIONS:
            with self.subTest(function=name):
                self.assertEqual(code_of(getattr(self.r, name)), code_of(getattr(self.m2, name)))

    def test_the_runner_constants_are_m2s(self):
        for name in SAME_RUNNER_CONSTANTS:
            self.assertEqual(getattr(self.r, name), getattr(self.m2, name), name)
        self.assertEqual(self.r.M2_EQUIVALENCE_UPDATES, self.m2.EQUIVALENCE_UPDATES)


if __name__ == '__main__':
    unittest.main()
