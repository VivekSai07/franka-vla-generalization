"""CPU contracts: X-VLA B3 robosuite adapter, real seal admission, provenance enforcement."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from experiments.b3_robosuite_data import plan as b3_plan

# Fixtures are driven by the real plan so they cannot drift from the contract's expectation.
PLAN = b3_plan.make_plan()
PLANNED = tuple(episode['instruction'] for episode in PLAN['episodes'])
EPISODES = len(PLANNED)
TRAINED = tuple(b3_plan.INSTRUCTIONS)
HELD_OUT = 'pick up the yellow cube and place it on top of the red cube'
FIRST_APPEARANCE = tuple(dict.fromkeys(PLANNED))  # how LeRobot numbers tasks
TASK_OF_EPISODE = tuple(FIRST_APPEARANCE.index(x) for x in PLANNED)
SMALL_PARENT_FILES = ('seal.json', 'plan.json', 'substitutions.json', 'provenance.json', 'validation.json')


def metadata():
    return [dict(episode_index=i, length=528, tasks=[PLANNED[i]],
                 dataset_from_index=i*528, dataset_to_index=(i+1)*528) for i in range(EPISODES)]


def frame(episode=0):
    return {'episode_index': torch.tensor(episode), 'task': PLANNED[episode],
            'observation.state': torch.zeros(9),
            'observation.images.image': torch.zeros(3, 512, 512),
            'observation.images.image2': torch.zeros(3, 512, 512),
            'action': torch.zeros(30, 8), 'action_is_pad': torch.zeros(30, dtype=torch.bool)}


def task_table(order):
    return pd.DataFrame({'task_index': list(range(len(order)))}, index=list(order))


class _DS:
    def __init__(self, order):
        self.meta = type('Meta', (), {'tasks': task_table(order)})()


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).parents[1]/'contract.py').is_file(),
                        'X-VLA B3 robosuite adapter missing')
        self.c = importlib.import_module('experiments.xvla_b3_robosuite.contract')

    def _require_parent(self):
        if not (self.c.PARENT/'seal.json').exists():
            self.skipTest('B3 parent not mounted')

    def _copy_parent(self, tmp):
        self._require_parent()
        parent = Path(tmp)/'parent'
        parent.mkdir()
        for name in SMALL_PARENT_FILES:
            shutil.copy(self.c.PARENT/name, parent/name)
        return parent

    def _binding_with(self, parent, **patches):
        stack = [patch.object(self.c, 'PARENT', parent),
                 patch.object(self.c, 'require_assets', return_value={'inventory': 'x'}),
                 patch.object(self.c, 'source_hashes', return_value={})]
        stack += [patch.object(self.c, key, value) for key, value in patches.items()]
        from contextlib import ExitStack
        with ExitStack() as exit_stack:
            for item in stack:
                exit_stack.enter_context(item)
            return self.c.current_binding()

    # --- counts, split, statistics -------------------------------------------------------------

    def test_counts_last_offset_and_no_validation_sampling(self):
        c = self.c
        rows, starts = c.indices(metadata())
        n = len(c.TRAIN)
        self.assertEqual((len(rows), len(starts)), (n*528, n*499))
        self.assertEqual((len(rows), len(starts)), (c.TRAINING_ROWS, c.COMPLETE_STARTS))
        self.assertEqual((c.TRAINING_ROWS, c.COMPLETE_STARTS), (264_000, 249_500))
        self.assertEqual((c.VALIDATION_ROWS, c.TOTAL_FRAMES), (31_680, 295_680))
        self.assertEqual(rows, list(range(c.TRAINING_ROWS)))
        self.assertEqual(starts[-1], (n-1)*528+498)
        self.assertTrue(all(i//528 < n and i % 528 <= 498 for i in starts))
        self.assertEqual(len(set(starts)), len(starts))
        self.assertEqual((c.TRAIN, c.VAL), (tuple(range(500)), tuple(range(500, 560))))

    def test_split_comes_from_the_plan_not_from_lerobot(self):
        c = self.c
        self.assertEqual(c.TRAIN, tuple(PLAN['train_ids']))
        self.assertEqual(c.VAL, tuple(PLAN['validation_ids']))
        self.assertEqual({e['split'] for e in PLAN['episodes'] if e['episode_index'] in c.TRAIN}, {'train'})
        self.assertEqual({e['split'] for e in PLAN['episodes'] if e['episode_index'] in c.VAL}, {'validation'})
        # LeRobot's own metadata calls all 560 episodes "train"; load_data must pin that fact
        # and never read the split from it.
        source = (Path(__file__).parents[1]/'contract.py').read_text()
        self.assertIn("splits={'train': '0:560'}", source)

    def test_reject_malformed_metadata_and_other_packages(self):
        for episodes in (metadata()[:200], metadata()[:24], metadata()+metadata()[:1]):
            with self.assertRaises(ValueError):
                self.c.indices(episodes)
        for change in ({'episode_index': 900}, {'length': 527}, {'tasks': ['wrong']},
                       {'dataset_from_index': 0}, {'dataset_to_index': 10**7}):
            for victim in (1, 499, 500, 559):
                episodes = metadata()
                episodes[victim].update(change)
                with self.subTest(change=change, victim=victim), self.assertRaises(ValueError):
                    self.c.indices(episodes)

    def test_train_statistics_use_exactly_the_training_rows(self):
        c = self.c

        class Rows:
            def __getitem__(self, i):
                if i >= c.TRAINING_ROWS:
                    raise AssertionError('validation accessed for stats')
                return dict(episode_index=i//528, task_index=TASK_OF_EPISODE[i//528],
                            action=torch.full((8,), float(i % 2)),
                            **{'observation.state': torch.full((9,), float(i % 2))})
        stats = c.selected_stats(Rows(), list(range(c.TRAINING_ROWS)), TASK_OF_EPISODE)
        self.assertEqual(set(stats['action']), {'mean', 'std', 'min', 'max'})
        self.assertEqual(stats['action']['mean'].tolist(), [.5]*8)
        self.assertEqual(stats['action']['std'].tolist(), [.5]*8)
        for rows in (list(range(c.TRAINING_ROWS+1)), [0]*c.TRAINING_ROWS, list(range(1, c.TRAINING_ROWS+1))):
            with self.assertRaises(ValueError):
                c.selected_stats(Rows(), rows, TASK_OF_EPISODE)

    def test_sampler_yields_exactly_the_249500_training_starts_and_only_those(self):
        import itertools
        c = self.c
        _, starts = c.indices(metadata())
        self.assertEqual(len(starts), 500*499)
        self.assertEqual(len(starts), 249_500)
        self.assertFalse(any(s//528 >= 500 for s in starts))
        self.assertTrue(all(s % 528 <= 498 for s in starts))
        batches = list(itertools.islice(c.sample_batches(starts), len(starts)//4))
        flat = list(itertools.chain.from_iterable(batches))
        self.assertEqual(sorted(flat), starts)
        self.assertTrue(all(s//528 < 500 for s in flat))
        self.assertEqual(list(itertools.islice(c.sample_batches(starts), len(starts)//4)), batches)
        for bad in (starts[:100], starts[:-1]+[500*528+10], starts+[500*528]):
            with self.assertRaises(ValueError):
                next(c.sample_batches(bad))

    # --- instructions --------------------------------------------------------------------------

    def test_task_index_is_mapped_by_string_not_by_number(self):
        c = self.c
        self.assertNotEqual(FIRST_APPEARANCE, tuple(sorted(FIRST_APPEARANCE)),
                            'premise: LeRobot numbers tasks by first appearance, not in sorted order')
        for order in (FIRST_APPEARANCE, tuple(sorted(TRAINED)), tuple(reversed(TRAINED))):
            table = c.task_index_map(_DS(order))
            inverse = {v: k for k, v in table.items()}
            mapped = tuple(table[x] for x in c.planned_instructions())
            for episode in range(EPISODES):
                self.assertEqual(inverse[mapped[episode]], PLANNED[episode], (order[:1], episode))
        # The plan's sorted numbering is NOT what the first-appearance table gives.
        by_appearance = c.task_index_map(_DS(FIRST_APPEARANCE))
        self.assertNotEqual([by_appearance[x] for x in sorted(TRAINED)], list(range(10)))

    def test_load_data_maps_tasks_through_the_dataset_table_only(self):
        source = (Path(__file__).parents[1]/'contract.py').read_text()
        self.assertIn('task_index_map(ds)[x] for x in planned_instructions()', source)
        self.assertNotIn('INSTRUCTIONS.index', source)

    def test_real_dataset_task_table_maps_every_episode_to_its_planned_string(self):
        c = self.c
        path = c.DATA/'meta/tasks.parquet'
        if not path.exists():
            self.skipTest('B3 parent not mounted')
        tasks = pd.read_parquet(path)
        table = c.task_index_map(type('DS', (), {'meta': type('M', (), {'tasks': tasks})()})())
        self.assertEqual(set(table), set(TRAINED))
        self.assertEqual(len(table), 10)

    def test_approved_set_is_exactly_the_ten_trained_instructions(self):
        c = self.c
        self.assertEqual(tuple(c.INSTRUCTIONS), TRAINED)
        self.assertEqual(len(c.INSTRUCTIONS), 10)
        self.assertEqual(set(PLANNED), set(TRAINED))
        self.assertNotIn(HELD_OUT, c.INSTRUCTIONS)
        self.assertNotIn(HELD_OUT, PLANNED)
        self.assertNotIn('pick up the green cube and place it on top of the blue cube', PLANNED)
        with self.assertRaises(ValueError):
            with c.approved_instruction(HELD_OUT):
                pass
        with self.assertRaises(ValueError):
            c.policy_row({**frame(), 'task': HELD_OUT}, training=True)

    def test_planned_instructions_cover_every_episode_and_split_contains_every_instruction(self):
        self._require_parent()
        c = self.c
        planned = c.planned_instructions()
        self.assertEqual(planned, PLANNED)
        self.assertEqual(len(planned), EPISODES)
        for name, ids in (('train', c.TRAIN), ('validation', c.VAL)):
            self.assertEqual(set(planned[i] for i in ids), set(TRAINED), name)

    def test_whitelist_real_state_action_padding_and_train_membership(self):
        c = self.c
        row = {**frame(), 'success': True, 'object_positions': [1, 2, 3]}
        selected = c.policy_row(row, training=True)
        self.assertEqual(set(selected), set(c.OBS)|{'task', 'action', 'action_is_pad'})
        for change in ({'episode_index': torch.tensor(500)}, {'task': 'wrong'},
                       {'observation.objects': torch.ones(3)}, {'action': torch.zeros(50, 8)},
                       {'action_is_pad': torch.ones(30, dtype=torch.bool)},
                       {'observation.state': torch.zeros(32)},
                       {'action': torch.full((30, 8), float('nan'))}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                c.policy_row({**row, **change}, training=True)
        c.policy_row(frame(559), training=False)  # validation: observation-only inference row
        with self.assertRaises(ValueError):
            c.policy_row(frame(559), training=True)
        with self.assertRaises(ValueError):
            c.collate_rows([frame(500)]*4)
        with self.assertRaises(ValueError):
            c.collate_rows([frame()]*3)

    def test_policy_row_accepts_every_trained_instruction(self):
        """FAILURES #29: fixtures built only from episode 0 pass while other instructions fail."""
        c = self.c
        seen = set()
        for episode in c.TRAIN:
            if PLANNED[episode] in seen:
                continue
            seen.add(PLANNED[episode])
            self.assertEqual(c.policy_row(frame(episode), training=True)['task'], PLANNED[episode])
        self.assertEqual(seen, set(TRAINED))

    def test_all_instructions_survive_a_single_microbatch(self):
        c = self.c
        others = [next(i for i in c.TRAIN if PLANNED[i] == t) for t in TRAINED[:4]]
        batch = c.collate_rows([frame(i) for i in others])
        self.assertEqual(sorted(set(batch['task'])), sorted(set(TRAINED[:4])))

    def test_approved_instruction_restores_the_shared_task(self):
        c = self.c
        before = c.single_task.TASK
        with c.approved_instruction(PLANNED[0]):
            self.assertEqual(c.single_task.TASK, PLANNED[0])
        self.assertEqual(c.single_task.TASK, before)
        with self.assertRaises(RuntimeError):
            with c.approved_instruction(PLANNED[0]):
                raise RuntimeError('boom')
        self.assertEqual(c.single_task.TASK, before)

    def test_task_index_map_rejects_a_single_task_or_incomplete_dataset(self):
        c = self.c
        for order in (TRAINED[:1], TRAINED[:9], TRAINED+(HELD_OUT,)):
            with self.assertRaises(ValueError):
                c.task_index_map(_DS(order))

    # --- parent binding ------------------------------------------------------------------------

    def test_seal_record_matches_the_collection_report(self):
        c = self.c
        self.assertEqual(c.SEAL, 'e14af864596caf7cb515ebc500957e970a3cf12a40a3a65c1692696d92b8e6c0')
        self.assertEqual(set(c.SEAL_RECORD), {'fingerprint', 'provenance_sha256', 'plan_sha256',
                                              'substitutions_sha256', 'image', 'min_clearance_m',
                                              'episodes'})
        self.assertEqual(c.SEAL_RECORD['fingerprint'], c.SEAL)
        self.assertEqual(c.SEAL_RECORD['episodes'], 560)
        self.assertEqual(c.SEAL_RECORD['min_clearance_m'], 0.08)
        self.assertEqual(c.SEAL_RECORD['plan_sha256'], b3_plan.digest(PLAN))

    def test_current_binding_accepts_the_real_parent(self):
        self._require_parent()
        c = self.c
        from experiments.b3_robosuite_data import contract as b3_contract
        binding = c.current_binding()
        self.assertEqual(binding['dataset_fingerprint'], c.SEAL)
        self.assertEqual(binding['b3_provenance'], b3_contract.provenance())
        self.assertEqual(binding['required_image'], c.IMAGE)
        for key in ('seal_sha256', 'plan_sha256', 'provenance_sha256', 'validation_sha256',
                    'sources', 'assets'):
            self.assertIn(key, binding)
        self.assertEqual(binding['plan_sha256'], c.SEAL_RECORD['plan_sha256'])
        self.assertEqual(binding['provenance_sha256'], c.SEAL_RECORD['provenance_sha256'])

    def test_a_copy_of_the_parent_files_binds_and_each_tamper_is_rejected(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            parent = self._copy_parent(tmp)
            self.assertEqual(self._binding_with(parent)['dataset_fingerprint'], c.SEAL)
            seal = json.loads((parent/'seal.json').read_text())
            for key in seal:
                changed = {**seal, key: ('x'*64 if isinstance(seal[key], str) else seal[key]+1)}
                (parent/'seal.json').write_text(json.dumps(changed))
                with self.subTest(seal_field=key), self.assertRaises(ValueError):
                    self._binding_with(parent)
            (parent/'seal.json').write_text(json.dumps(seal))
            self.assertEqual(self._binding_with(parent)['dataset_fingerprint'], c.SEAL)

    def test_changed_plan_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = self._copy_parent(tmp)
            plan = json.loads((parent/'plan.json').read_text())
            plan['episodes'][0]['seed'] += 1
            (parent/'plan.json').write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                self._binding_with(parent)

    def test_non_empty_substitutions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = self._copy_parent(tmp)
            (parent/'substitutions.json').write_text(json.dumps({'12': 1}))
            with self.assertRaises(ValueError):
                self._binding_with(parent)

    def test_drifted_provenance_is_rejected_both_ways(self):
        from experiments.b3_robosuite_data import contract as b3_contract
        with tempfile.TemporaryDirectory() as tmp:
            parent = self._copy_parent(tmp)
            real = b3_contract.provenance()
            with patch.object(b3_contract, 'provenance', return_value={**real, 'experiments/x.py': '0'*64}):
                with self.assertRaises(ValueError):
                    self._binding_with(parent)
            changed = dict(real)
            first = next(iter(changed))
            changed[first] = '0'*64
            with patch.object(b3_contract, 'provenance', return_value=changed):
                with self.assertRaises(ValueError):
                    self._binding_with(parent)
            (parent/'provenance.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):  # recorded copy edited: digest no longer the sealed one
                self._binding_with(parent)

    def test_wrong_validation_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = self._copy_parent(tmp)
            validation = json.loads((parent/'validation.json').read_text())
            changes = {'episodes': 559, 'train': 499, 'validation': 61, 'per_cell_train': 24,
                       'per_cell_validation': 2, 'substituted_slots': 1, 'held_out_pairs_collected': 1,
                       'distractors_train': {'0': 99, '1': 201, '2': 200},
                       'edge_train': {'inner': 51, 'interior': 399, 'outer': 50}}
            for key, value in changes.items():
                (parent/'validation.json').write_text(json.dumps({**validation, key: value}))
                with self.subTest(field=key), self.assertRaises(ValueError):
                    self._binding_with(parent)
            (parent/'validation.json').write_text(json.dumps({k: v for k, v in validation.items()
                                                              if k != 'episodes'}))
            with self.assertRaises(ValueError):
                self._binding_with(parent)

    def test_parent_fingerprint_recomputes_to_the_seal(self):
        self._require_parent()
        self.assertEqual(self.c.parent_fingerprint(), self.c.SEAL)

    def test_cpu_binding_semantic_stats_and_processor_tampering_rejected(self):
        c = self.c
        semantic = {'episodes': 560}
        binding = {'validation_sha256': c.digest_json(semantic)}
        report = c.cpu_proof(binding, semantic, {'action': {'mean': [0.]*8}}, {'batch_size': 4},
                             parent_fingerprint=c.SEAL)
        c.validate_cpu(report, binding)
        self.assertTrue(report['parent_fingerprint_recomputed'])
        for change in ({'binding': {}}, {'training_rows': 4752}, {'complete_training_chunks': 9980},
                       {'train_episodes': list(range(9))}, {'training_stats': {}},
                       {'processor_check': {}}, {'semantic_gate': {}}, {'gpu_executed': True},
                       {'parent_fingerprint_recomputed': False}, {'validation_episodes': []}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.validate_cpu({**report, **change}, binding)
        wrong = c.cpu_proof(binding, semantic, {'action': {'mean': [0.]*8}}, {'batch_size': 4},
                            parent_fingerprint='0'*64)
        self.assertFalse(wrong['parent_fingerprint_recomputed'])
        with self.assertRaises(ValueError):
            c.validate_cpu(wrong, binding)

    def test_later_admission_rehashes_without_full_fingerprint_repeat(self):
        c = self.c
        semantic = {'episodes': 560}
        binding = {'validation_sha256': c.digest_json(semantic)}
        report = c.cpu_proof(binding, semantic, {'action': {'mean': [0.]*8}}, {'batch_size': 4},
                             parent_fingerprint=c.SEAL)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cpu.json'
            path.write_text(json.dumps(report))
            with patch.object(c, 'CPU_ADMISSION', path), \
                    patch.object(c, 'current_binding', return_value=binding), \
                    patch.object(c, 'parent_fingerprint', side_effect=AssertionError('full hash must not repeat')):
                self.assertEqual(c.admission(), binding)
                with patch.object(c, 'current_binding', return_value={**binding, 'assets': 'changed'}):
                    with self.assertRaises(ValueError):
                        c.admission()

    # --- assets, claims, schedule, checkpoints -------------------------------------------------

    def test_output_existing_mount_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileExistsError):
                self.c.claim_output(root)
            with patch.object(self.c, 'is_mountpoint', return_value=True):
                self.c.claim_output(root)
                with self.assertRaises(FileExistsError):
                    self.c.claim_output(root)
            new = root/'new'
            self.c.claim_output(new)
            self.assertEqual(json.loads((new/'run_claim.json').read_text())['purpose'], 'xvla_b3_robosuite')
            link = root/'link'
            link.symlink_to(new, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.c.claim_output(link)

    def test_require_assets_accepts_real_bundle_and_rejects_tamper(self):
        from experiments.xvla_trial.assets import require_assets, ASSETS
        if not (self.c.CACHE/'manifest.json').exists():
            self.skipTest('base bundle not mounted')
        report = require_assets(self.c.CACHE)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['assets'], ASSETS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'manifest.json').write_text(json.dumps({'status': 'incomplete', 'assets': ASSETS, 'files': {}}))
            with self.assertRaises(ValueError):
                require_assets(root)

    def test_checkpoint_naming_is_step_prefixed_and_rejects_legacy(self):
        c = self.c
        self.assertEqual(c.checkpoint_directory(1), 'step_000001')
        self.assertEqual(c.checkpoint_directory(156000), 'step_156000')
        for update in (0, -1, 1.0, '1', True):
            with self.assertRaises(ValueError):
                c.checkpoint_directory(update)

    def test_recipe_and_checkpoint_schedule(self):
        c = self.c
        for scratch in (True, False):
            recipe = c.recipe(scratch)
            self.assertEqual((recipe['microbatch'], recipe['accumulation'], recipe['effective_batch']),
                             (4, 2, 8))
            self.assertEqual((recipe['chunk_size'], recipe['n_action_steps']), (30, 10))
            self.assertEqual(recipe['max_optimizer_updates'], 2 if scratch else c.TOTAL_UPDATES)
            self.assertEqual(recipe['purpose'], 'xvla_b3_robosuite_preflight' if scratch else 'xvla_b3_robosuite')
            self.assertEqual(recipe['seed'], 20260917)
            self.assertEqual(recipe['train_episodes'], list(range(500)))
            self.assertEqual(recipe['validation_episodes'], list(range(500, 560)))
            self.assertFalse(recipe['resume'])
            self.assertEqual(recipe['precision'], 'fp32_parameters_gradients_adam_bf16_autocast')
            self.assertEqual(recipe['optimizer_storage'], 'final_only')
            self.assertEqual(recipe['schedule']['total'], c.TOTAL_UPDATES)
        self.assertEqual(c.TOTAL_UPDATES, 156_000)
        self.assertEqual(c.checkpoint_updates(True), [1, 2])
        updates = c.checkpoint_updates(False)
        self.assertEqual(updates, [1, 31200, 62400, 93600, 124800, 156000])
        self.assertEqual(updates[-1], c.TOTAL_UPDATES)
        self.assertEqual(updates[1:], [c.TOTAL_UPDATES//5*k for k in range(1, 6)])
        # Exposure matched to B2: 5.0 epochs (B2: 50,000 x 8 / 79,840 = 5.01).
        self.assertAlmostEqual(c.TOTAL_UPDATES*8/c.COMPLETE_STARTS, 5.0, delta=0.01)
        self.assertAlmostEqual(c.learning_rate(500), 1e-4)
        self.assertAlmostEqual(c.learning_rate(c.TOTAL_UPDATES), 1e-5, places=9)
        self.assertGreater(c.learning_rate(c.TOTAL_UPDATES//2), 1e-5)
        with self.assertRaises(ValueError):
            c.learning_rate(c.TOTAL_UPDATES+1)

    def test_make_config_stretches_the_decay_horizon(self):
        self.assertEqual(self.c.make_config('cpu').scheduler_decay_steps, self.c.TOTAL_UPDATES)

    def test_verify_checkpoints_requires_step_dirs_and_rejects_legacy(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'checkpoints').mkdir()
            for update in (1, 2):
                (out/f'checkpoints/{update:06d}').mkdir()
            markers = [dict(status='complete', update=u, reload_fp32_weights_exact=True,
                            reload_max_action_error=0., model_sha256='x') for u in (1, 2)]
            report = dict(checkpoints=markers,
                          checkpoint_files={c.checkpoint_directory(u): {} for u in (1, 2)})
            with self.assertRaises(ValueError):
                c.verify_checkpoints(report, out, True)

    def test_forbidden_imports_absent(self):
        for name in ('contract.py', 'runner.py'):
            source = (Path(__file__).parents[1]/name).read_text()
            for needle in ('from experiments.smolvla_local_varied', 'import experiments.smolvla_local_varied',
                           'from experiments.xvla_fixed_layout', 'import experiments.xvla_fixed_layout',
                           'from .evaluate'):
                self.assertNotIn(needle, source, msg=f'Forbidden import in {name}: {needle}')

    def test_explicit_source_files_only(self):
        names = self.c.SOURCE_FILES
        self.assertIn('experiments/xvla_trial/preflight.py', names)
        self.assertIn('experiments/xvla_single_task/contract.py', names)
        for own in ('contract.py', 'runner.py', 'README.md', 'tests/test_contract.py',
                    'tests/test_runner.py', 'tests/test_fork_hygiene.py', '__init__.py'):
            self.assertIn(f'experiments/xvla_b3_robosuite/{own}', names)
        for bound in ('__init__', 'plan', 'contract', 'validate'):
            self.assertIn(f'experiments/b3_robosuite_data/{bound}.py', names)
        previous = ('b2'+'_distractor', 'xvla_b2')  # the fork source's names must not be bound
        self.assertFalse(any(old in n for n in names for old in previous))
        self.assertFalse(any('smolvla_local_varied' in n or 'xvla_fixed_layout' in n for n in names))
        self.assertEqual(len(names), len(set(names)))

    def test_output_paths_are_isolated(self):
        c = self.c
        self.assertEqual(c.OUTPUT.name, 'xvla_franka_stacking_b3_robosuite_v1')
        self.assertEqual(c.SCRATCH.name, 'xvla_franka_stacking_b3_robosuite_v1_preflight')
        self.assertEqual(c.ADMISSION.name, 'xvla_franka_stacking_b3_robosuite_v1_admission')
        self.assertEqual(c.PARENT, c.ROOT/'outputs/b3_robosuite_data/b3_robosuite_20261003')
        self.assertEqual(c.DATA, c.PARENT/'dataset')
        self.assertEqual(c.CACHE, c.ROOT/'.cache/xvla_trial/bundle')

    def test_scratch_proof_recipe_binding_and_checkpoint_bytes_bound(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c.write_json(root/'training_stats.json', {'action': {'mean': [0.]*8}})
            (root/'checkpoints').mkdir()
            markers, files = [], {}
            for update in (1, 2):
                folder = root/'checkpoints'/c.checkpoint_directory(update)
                folder.mkdir()
                for name in ('model.safetensors', 'config.json',
                             'policy_preprocessor.json', 'policy_postprocessor.json'):
                    (folder/name).write_text('bytes')
                if update == 2:
                    (folder/'training_state.pt').write_bytes(b'opt')
                marker = dict(status='complete', update=update, reload_fp32_weights_exact=True,
                              reload_max_action_error=0.,
                              model_sha256=c.sha256(folder/'model.safetensors'))
                c.write_json(folder/'checkpoint_result.json', marker)
                if update == 2:
                    c.write_json(folder/'adam_reload.json',
                                 dict(adam_reload_fp32_exact=True, update=update,
                                      training_state_sha256=c.sha256(folder/'training_state.pt')))
                markers.append(marker)
                files[c.checkpoint_directory(update)] = c.inventory(folder)
            binding = {'fingerprint': 'x'}
            report = {**c.recipe(True), 'status': 'complete', 'completed_updates': 2,
                      'binding': binding, 'training_rows': c.TRAINING_ROWS,
                      'complete_training_chunks': c.COMPLETE_STARTS,
                      'cpu_admission_sha256': 'cpu', 'final_data_binding_passed': True,
                      'adam_reload_fp32_exact': True, 'selected_checkpoint_update': None,
                      'rollouts': 0, 'checkpoints': markers, 'checkpoint_files': files,
                      'training_stats_sha256': c.sha256(root/'training_stats.json')}
            c.validate_scratch(report, binding, root, {'action': {'mean': [0.]*8}}, 'cpu')
            for change in ({'completed_updates': 1}, {'cpu_admission_sha256': 'stale'},
                           {'binding': {}}, {'adam_reload_fp32_exact': False},
                           {'training_rows': 84_480}, {'complete_training_chunks': 79_840},
                           {'rollouts': 1}, {'checkpoints': [markers[0]]}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    c.validate_scratch({**report, **change}, binding, root,
                                       {'action': {'mean': [0.]*8}}, 'cpu')

    def test_intermediate_checkpoint_must_not_carry_optimizer_state(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'checkpoints').mkdir()
            markers, files = [], {}
            for update in (1, 2):
                folder = out/'checkpoints'/c.checkpoint_directory(update)
                folder.mkdir()
                for name in ('model.safetensors', 'config.json',
                             'policy_preprocessor.json', 'policy_postprocessor.json',
                             'training_state.pt'):  # optimizer on BOTH: illegal for update 1
                    (folder/name).write_bytes(b'x')
                marker = dict(status='complete', update=update, reload_fp32_weights_exact=True,
                              reload_max_action_error=0.,
                              model_sha256=c.sha256(folder/'model.safetensors'))
                c.write_json(folder/'checkpoint_result.json', marker)
                markers.append(marker)
                files[c.checkpoint_directory(update)] = c.inventory(folder)
            c.write_json(out/'checkpoints'/c.checkpoint_directory(2)/'adam_reload.json',
                         dict(adam_reload_fp32_exact=True, update=2,
                              training_state_sha256=c.sha256(out/'checkpoints/step_000002/training_state.pt')))
            files[c.checkpoint_directory(2)] = c.inventory(out/'checkpoints/step_000002')
            report = dict(checkpoints=markers, checkpoint_files=files)
            with self.assertRaises(ValueError):
                c.verify_checkpoints(report, out, True)


if __name__ == '__main__':
    unittest.main()
