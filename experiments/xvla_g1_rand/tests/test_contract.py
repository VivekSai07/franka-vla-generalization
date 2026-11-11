"""CPU contracts: the X-VLA G1 package, real seal admission, provenance enforcement.

Fixtures come from the real G1 plan (`gen_g1_data.plan.make_plan()`), so they cannot drift from what the contract expects;
the plan's instructions are patched in for `planned_instructions` (the sealed `plan.json` is checked against the plan by the
tests that need the mounted parent), so the logic tests need no data.
"""
from contextlib import ExitStack
import importlib
import itertools
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.gen_g1_data import plan as g1_plan

PLAN = g1_plan.make_plan()
PLANNED = tuple(episode['instruction'] for episode in PLAN['episodes'])
EPISODES = len(PLANNED)
TRAINED = tuple(sorted({g1_plan.instruction(obj, bin_) for obj, bin_, _ in g1_plan.TRAINED_CELLS}))
HELD_OUT = g1_plan.instruction('milk', 'blue')
HELD_OUT_2 = g1_plan.instruction('bread', 'orange')
FIRST_APPEARANCE = tuple(dict.fromkeys(PLANNED))  # how LeRobot numbers tasks
TASK_OF_EPISODE = tuple(FIRST_APPEARANCE.index(x) for x in PLANNED)
SMALL_PARENT_FILES = ('seal.json', 'plan.json', 'substitutions.json', 'provenance.json', 'validation.json')
FRAMES = 452
LAST_START = 422  # 452 - 30: the last frame of an episode with a complete action chunk


def use_planned(case, contract):
    """`planned_instructions()` returns the real plan's instructions without reading the sealed `plan.json`."""
    patcher = patch.object(contract, 'planned_instructions', return_value=PLANNED)
    patcher.start()
    case.addCleanup(patcher.stop)


def metadata():
    return [dict(episode_index=i, length=FRAMES, tasks=[PLANNED[i]],
                 dataset_from_index=i*FRAMES, dataset_to_index=(i+1)*FRAMES) for i in range(EPISODES)]


def frame(episode=0):
    return {'episode_index': torch.tensor(episode), 'task': PLANNED[episode],
            'observation.state': torch.zeros(17),
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
        self.assertTrue((Path(__file__).parents[1]/'contract.py').is_file(), 'X-VLA G1 contract missing')
        self.c = importlib.import_module('experiments.xvla_g1_rand.contract')
        self.sealed_planned_instructions = self.c.planned_instructions
        use_planned(self, self.c)

    def _require_parent(self):
        if not (self.c.PARENT/'seal.json').exists():
            self.skipTest('sealed G1 dataset not mounted')

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
                 patch.object(self.c, 'source_hashes', return_value={}),
                 patch.object(self.c.forces, 'binding', return_value={'patched': True}),
                 patch.object(g1_plan, 'make_plan', return_value=PLAN)]  # the canonical plan, computed once above
        stack += [patch.object(self.c, key, value) for key, value in patches.items()]
        with ExitStack() as exit_stack:
            for item in stack:
                exit_stack.enter_context(item)
            return self.c.current_binding()

    # --- counts, split, statistics -------------------------------------------------------------

    def test_counts_last_offset_and_no_validation_sampling(self):
        c = self.c
        rows, starts = c.indices(metadata())
        n = len(c.TRAIN)
        self.assertEqual((len(rows), len(starts)), (n*FRAMES, n*(LAST_START+1)))
        self.assertEqual((len(rows), len(starts)), (c.TRAINING_ROWS, c.COMPLETE_STARTS))
        self.assertEqual((c.TRAINING_ROWS, c.COMPLETE_STARTS), (1_220_400, 1_142_100))
        self.assertEqual((c.VALIDATION_ROWS, c.TOTAL_FRAMES), (135_600, 1_356_000))
        self.assertEqual(c.TRAINING_ROWS+c.VALIDATION_ROWS, c.TOTAL_FRAMES)
        self.assertEqual(rows, list(range(c.TRAINING_ROWS)))
        self.assertEqual(starts[-1], (n-1)*FRAMES+LAST_START)
        self.assertTrue(all(i//FRAMES < n and i % FRAMES <= LAST_START for i in starts))
        self.assertEqual(len(set(starts)), len(starts))
        self.assertEqual((c.EPISODES, len(c.TRAIN), len(c.VAL)), (3000, 2700, 300))
        self.assertEqual((c.TRAIN, c.VAL), (tuple(range(2700)), tuple(range(2700, 3000))))

    def test_split_comes_from_the_plan_not_from_lerobot(self):
        c = self.c
        self.assertEqual(c.TRAIN, tuple(PLAN['train_ids']))
        self.assertEqual(c.VAL, tuple(PLAN['validation_ids']))
        self.assertEqual({e['split'] for e in PLAN['episodes'] if e['episode_index'] in c.TRAIN}, {'train'})
        self.assertEqual({e['split'] for e in PLAN['episodes'] if e['episode_index'] in c.VAL}, {'validation'})
        # LeRobot's own metadata calls all 3,000 episodes "train"; load_data must pin that fact
        # and never read the split from it.
        source = (Path(__file__).parents[1]/'contract.py').read_text()
        self.assertIn("splits={'train': f'0:{EPISODES}'}", source)
        self.assertEqual(c.EPISODES, EPISODES)

    def test_the_signals_and_the_contract_agree_on_the_split_and_the_geometry(self):
        c, f = self.c, self.c.forces
        self.assertEqual((f.EPISODES, f.TRAIN_EPISODES, f.FRAMES, f.N_EXTRA), (c.EPISODES, len(c.TRAIN), c.FRAMES_PER_EPISODE, 8))
        self.assertEqual(c.REAL_STATE_DIM-c.JOINT_DIM, f.N_EXTRA)

    def test_reject_malformed_metadata_and_other_packages(self):
        for episodes in (metadata()[:200], metadata()[:540], metadata()[:2999], metadata()+metadata()[:1]):
            with self.assertRaises(ValueError):
                self.c.indices(episodes)
        for change in ({'episode_index': 9000}, {'length': 451}, {'tasks': ['wrong']},
                       {'dataset_from_index': 0}, {'dataset_to_index': 10**7}):
            for victim in (1, 2699, 2700, 2999):
                episodes = metadata()
                episodes[victim].update(change)
                with self.subTest(change=change, victim=victim), self.assertRaises(ValueError):
                    self.c.indices(episodes)

    def test_train_statistics_use_exactly_the_training_rows(self):
        """Run at 3 episodes (TRAINING_ROWS patched): the full-size statistics are computed in test_inputs."""
        c = self.c
        small = 3*FRAMES

        class Rows:
            def __getitem__(self, i):
                if i >= small:
                    raise AssertionError('validation accessed for stats')
                return dict(episode_index=i//FRAMES, frame_index=i % FRAMES, task_index=TASK_OF_EPISODE[i//FRAMES],
                            action=torch.full((8,), float(i % 2)),
                            **{'observation.state': torch.full((9,), float(i % 2))})

        with patch.object(c, 'TRAINING_ROWS', small), patch.object(c.forces, 'extra_state', return_value=np.zeros(8, np.float32)):
            stats = c.selected_stats(Rows(), list(range(small)), TASK_OF_EPISODE)
            self.assertEqual(set(stats['action']), {'mean', 'std', 'min', 'max'})
            self.assertEqual(stats['action']['mean'].tolist(), [.5]*8)
            self.assertEqual(stats['action']['std'].tolist(), [.5]*8)
            self.assertEqual(stats['observation.state']['mean'].tolist(), [.5]*9+[0.]*8)
            for rows in (list(range(small+1)), [0]*small, list(range(1, small+1)), list(range(small-1))):
                with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                    c.selected_stats(Rows(), rows, TASK_OF_EPISODE)

    def test_the_statistics_contract_names_the_full_training_set(self):
        c = self.c
        with self.assertRaises(ValueError):  # a prefix of the training rows is not the training set
            c.selected_stats(object(), list(range(c.TRAINING_ROWS-1)), TASK_OF_EPISODE)
        with self.assertRaises(ValueError):
            c.selected_stats(object(), list(range(c.TRAINING_ROWS+c.VALIDATION_ROWS)), TASK_OF_EPISODE)

    def test_sampler_yields_exactly_the_training_starts_and_only_those(self):
        c = self.c
        n = len(c.TRAIN)
        _, starts = c.indices(metadata())
        self.assertEqual(len(starts), n*(LAST_START+1))
        self.assertFalse(any(s//FRAMES >= n for s in starts))
        self.assertTrue(all(s % FRAMES <= LAST_START for s in starts))
        batches = list(itertools.islice(c.sample_batches(starts), len(starts)//4))
        flat = list(itertools.chain.from_iterable(batches))
        self.assertEqual(sorted(flat), starts)
        self.assertTrue(all(s//FRAMES < n for s in flat))
        self.assertEqual(list(itertools.islice(c.sample_batches(starts), len(starts)//4)), batches)
        for bad in (starts[:100], starts[:-1]+[n*FRAMES+10], starts+[n*FRAMES]):
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

    def test_load_data_maps_tasks_through_the_dataset_table_only(self):
        source = (Path(__file__).parents[1]/'contract.py').read_text()
        self.assertIn('task_index_map(ds)[x] for x in planned_instructions()', source)
        self.assertNotIn('INSTRUCTIONS.index', source)

    def test_real_dataset_task_table_maps_every_episode_to_its_planned_string(self):
        c = self.c
        path = c.DATA/'meta/tasks.parquet'
        if not path.exists():
            self.skipTest('sealed G1 dataset not mounted')
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
        self.assertNotIn(HELD_OUT_2, c.INSTRUCTIONS)
        self.assertNotIn(HELD_OUT, PLANNED)
        self.assertNotIn(HELD_OUT_2, PLANNED)
        self.assertEqual((HELD_OUT, HELD_OUT_2), ('put the milk in the blue bin', 'put the bread in the orange bin'))
        with self.assertRaises(ValueError):
            with c.approved_instruction(HELD_OUT):
                pass
        with self.assertRaises(ValueError):
            c.policy_row({**frame(), 'task': HELD_OUT}, training=True)

    def test_planned_instructions_are_the_real_plans_and_cover_both_splits(self):
        self._require_parent()
        c = self.c
        self.assertEqual(len(PLANNED), c.EPISODES)
        for name, ids in (('train', c.TRAIN), ('validation', c.VAL)):
            self.assertEqual(set(PLANNED[i] for i in ids), set(TRAINED), name)
        sealed = self.sealed_planned_instructions  # the real function reading the sealed plan.json, not the patched one
        sealed.cache_clear()
        self.addCleanup(sealed.cache_clear)
        self.assertEqual(sealed(), PLANNED)

    def test_whitelist_real_state_action_padding_and_train_membership(self):
        c = self.c
        row = {**frame(), 'success': True, 'object_positions': [1, 2, 3]}
        selected = c.policy_row(row, training=True)
        self.assertEqual(set(selected), set(c.OBS)|{'task', 'action', 'action_is_pad'})
        for change in ({'episode_index': torch.tensor(len(c.TRAIN))}, {'episode_index': torch.tensor(EPISODES)}, {'task': 'wrong'},
                       {'observation.objects': torch.ones(3)}, {'action': torch.zeros(50, 8)},
                       {'action_is_pad': torch.ones(30, dtype=torch.bool)},
                       {'observation.state': torch.zeros(32)},
                       {'observation.state': torch.zeros(9)},  # a dataset row's state, not yet joined to the extras
                       {'action': torch.full((30, 8), float('nan'))}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                c.policy_row({**row, **change}, training=True)
        last = EPISODES-1
        c.policy_row(frame(last), training=False)  # validation: observation-only inference row
        with self.assertRaises(ValueError):
            c.policy_row(frame(last), training=True)
        with self.assertRaises(ValueError):
            c.policy_row({**frame(last), 'episode_index': torch.tensor(EPISODES)}, training=False)
        with self.assertRaises(ValueError):
            c.collate_rows([frame(len(c.TRAIN))]*4)
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

    def test_the_parent_is_the_sealed_g1_dataset_and_the_outputs_are_this_packages(self):
        c = self.c
        self.assertEqual(c.PARENT, c.forces.SEALED)
        self.assertEqual(c.PARENT.name, 'g1_20261110')
        self.assertEqual(c.PARENT.parent.name, 'gen_g1_data')
        self.assertEqual(c.DATA, c.PARENT/'dataset')
        names = {c.OUTPUT.name, c.SCRATCH.name, c.PROOF.name, c.EQUIVALENCE.name, c.M2_EQUIVALENCE.name, c.ADMISSION.name}
        self.assertEqual(names, {'xvla_g1_rand_v1', 'xvla_g1_rand_v1_preflight', 'xvla_g1_rand_v1_resumeproof',
                                 'xvla_g1_rand_v1_equivalence', 'xvla_g1_rand_v1_m2equivalence', 'xvla_g1_rand_v1_admission'})
        self.assertEqual(c.CPU_ADMISSION, c.ADMISSION/'result.json')
        self.assertFalse(hasattr(c, 'ZERO_EQUIVALENCE'))
        self.assertFalse(hasattr(c.forces, 'SIDECAR'))

    def test_seal_record_matches_the_collection_report(self):
        c = self.c
        self.assertEqual(c.SEAL, '845ac94063fd367cc315b8d287fc141df76ea171f9227f5794eef0bd8e4e161e')
        self.assertIs(c.SEAL_RECORD, c.forces.SEAL_RECORD)
        self.assertEqual(set(c.SEAL_RECORD), {'fingerprint', 'provenance_sha256', 'plan_sha256',
                                              'substitutions_sha256', 'image', 'episodes'})
        self.assertEqual(c.SEAL_RECORD['fingerprint'], c.SEAL)
        self.assertEqual(c.SEAL_RECORD['episodes'], 3000)
        self.assertEqual(c.SEAL_RECORD['plan_sha256'], g1_plan.digest(PLAN))
        self.assertEqual(c.SEAL_RECORD['plan_sha256'], g1_plan.PLAN_SHA256)
        self.assertEqual(c.SEAL_RECORD['image'],  # the robosuite image the collection ran in
                         'sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e')

    def test_current_binding_accepts_the_real_parent(self):
        self._require_parent()
        if not (self.c.CACHE/'manifest.json').exists():
            self.skipTest('base bundle not mounted')
        c = self.c
        from experiments.gen_g1_data import contract as data_contract
        binding = c.current_binding()
        self.assertEqual(binding['dataset_fingerprint'], c.SEAL)
        self.assertEqual(binding['g1_provenance'], data_contract.provenance())
        self.assertEqual(binding['required_image'], c.IMAGE)
        for key in ('seal_sha256', 'plan_sha256', 'provenance_sha256', 'validation_sha256',
                    'sources', 'assets', 'signals', 'g1_provenance'):
            self.assertIn(key, binding)
        self.assertFalse([key for key in binding if 'sidecar' in key or 'props_p2' in key])
        self.assertEqual(binding['plan_sha256'], c.SEAL_RECORD['plan_sha256'])
        self.assertEqual(binding['provenance_sha256'], c.SEAL_RECORD['provenance_sha256'])
        self.assertEqual(binding['signals']['seal'], c.forces.SEAL_RECORD)
        self.assertEqual(binding['signals']['constants_sha256'], c.forces.CONSTANTS_SHA256)
        self.assertEqual(set(binding['sources']), set(c.SOURCE_FILES))

    def test_a_copy_of_the_parent_files_binds_and_each_tamper_is_rejected(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            parent = self._copy_parent(tmp)
            binding = self._binding_with(parent)
            self.assertEqual(binding['dataset_fingerprint'], c.SEAL)
            self.assertEqual(binding['signals'], {'patched': True})
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
            with self.assertRaisesRegex(ValueError, 'plan|Noncanonical'):
                self._binding_with(parent)

    def test_the_sealed_plan_json_is_the_canonical_g1_plan(self):
        self._require_parent()
        plan = json.loads((self.c.PARENT/'plan.json').read_text())
        self.assertEqual(g1_plan.digest(plan), g1_plan.digest(PLAN))
        self.assertEqual(g1_plan.digest(plan), self.c.SEAL_RECORD['plan_sha256'])

    def test_substitutions_must_match_the_sealed_record(self):
        """G1 sealed with 9 substituted slots; any other record is not the sealed one."""
        with tempfile.TemporaryDirectory() as tmp:
            parent = self._copy_parent(tmp)
            original = json.loads((parent/'substitutions.json').read_text())
            self.assertEqual(len(original), self.c.VALIDATION_RECORD['substituted_slots'])
            self.assertEqual(g1_plan.digest(original), self.c.SEAL_RECORD['substitutions_sha256'])
            first = next(iter(original))
            for name, changed in (('empty', {}), ('extra slot', {**original, '5': 1}),
                                  ('one missing', {k: v for k, v in original.items() if k != first}),
                                  ('count bumped', {**original, first: original[first]+1}),
                                  ('slot moved', {**{k: v for k, v in original.items() if k != first}, '2999': original[first]})):
                (parent/'substitutions.json').write_text(json.dumps(changed))
                with self.subTest(changed=name), self.assertRaisesRegex(ValueError, 'substitutions'):
                    self._binding_with(parent)
            (parent/'substitutions.json').write_text(json.dumps(original))
            self.assertEqual(self._binding_with(parent)['dataset_fingerprint'], self.c.SEAL)

    def test_drifted_provenance_is_rejected_both_ways(self):
        from experiments.gen_g1_data import contract as data_contract
        with tempfile.TemporaryDirectory() as tmp:
            parent = self._copy_parent(tmp)
            real = data_contract.provenance()
            with patch.object(data_contract, 'provenance', return_value={**real, 'experiments/x.py': '0'*64}):
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    self._binding_with(parent)
            changed = dict(real)
            first = next(iter(changed))
            changed[first] = '0'*64
            with patch.object(data_contract, 'provenance', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    self._binding_with(parent)
            (parent/'provenance.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'provenance'):  # recorded copy edited: digest no longer the sealed one
                self._binding_with(parent)

    def test_wrong_validation_values_are_rejected(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            parent = self._copy_parent(tmp)
            validation = json.loads((parent/'validation.json').read_text())
            self.assertEqual({k: validation[k] for k in c.VALIDATION_RECORD}, c.VALIDATION_RECORD)
            changes = {'episodes': 2999, 'train': 2699, 'validation': 301,
                       'distractor_counts': {**validation['distractor_counts'], '1': 449},
                       'height_strata': {**validation['height_strata'], '0': 599},
                       'start_regions': {'outer': 899, 'p2': 2101},
                       'substituted_slots': 8, 'held_out_pairs_collected': 1, 'signals_checked': False}
            self.assertEqual(set(changes), set(c.VALIDATION_RECORD))
            for key, value in changes.items():
                (parent/'validation.json').write_text(json.dumps({**validation, key: value}))
                with self.subTest(field=key), self.assertRaises(ValueError):
                    self._binding_with(parent)
            (parent/'validation.json').write_text(json.dumps({**validation, 'episodes': 3000.0}))
            with self.assertRaises(ValueError):  # the type is part of the record
                self._binding_with(parent)
            (parent/'validation.json').write_text(json.dumps({**validation, 'extra': 1}))
            with self.assertRaises(ValueError):
                self._binding_with(parent)
            (parent/'validation.json').write_text(json.dumps({k: v for k, v in validation.items() if k != 'episodes'}))
            with self.assertRaises(ValueError):
                self._binding_with(parent)

    def test_parent_fingerprint_is_the_collections_own_recomputation_of_the_parent(self):
        """The 33 GB recomputation is `--check`'s job (set G1_FULL_FINGERPRINT=1 to run it here); this pins what it calls."""
        c = self.c
        with patch.object(c.data_validate, 'fingerprint', return_value='abc') as call:
            self.assertEqual(c.parent_fingerprint(), 'abc')
        call.assert_called_once_with(c.PARENT)

    def test_parent_fingerprint_recomputes_to_the_seal_when_asked_to(self):
        import os
        self._require_parent()
        if os.environ.get('G1_FULL_FINGERPRINT') != '1':
            self.skipTest('hashes the 33 GB parent; `--check` does it at admission (G1_FULL_FINGERPRINT=1 to run here)')
        self.assertEqual(self.c.parent_fingerprint(), self.c.SEAL)

    def test_cpu_binding_semantic_stats_and_processor_tampering_rejected(self):
        c = self.c
        constants = {key: np.arange(8, dtype=np.float64)+i for i, key in enumerate(('low', 'high', 'mean', 'std'))}
        rates = {'train': {'rows': 1, 'fraction_clipped': [0.]*8}, 'validation': {'rows': 1, 'fraction_clipped': [0.]*8}}
        with patch.object(c.forces, 'constants', return_value=constants), patch.object(c.forces, 'clipping_rates', return_value=rates):
            semantic = {'episodes': 3000}
            binding = {'validation_sha256': c.digest_json(semantic)}
            report = c.cpu_proof(binding, semantic, {'action': {'mean': [0.]*8}}, {'batch_size': 4}, parent_fingerprint=c.SEAL)
            c.validate_cpu(report, binding)
            self.assertTrue(report['parent_fingerprint_recomputed'])
            self.assertFalse([key for key in report if 'sidecar' in key])
            self.assertEqual(report['extra_constants'], c.forces.constants_json(constants))
            self.assertEqual(set(report['extra_clipping_rates']), {'train', 'validation'})
            self.assertEqual((report['training_rows'], report['complete_training_chunks']), (1_220_400, 1_142_100))
            self.assertEqual((report['train_episodes'], report['validation_episodes']), (list(range(2700)), list(range(2700, 3000))))
            with self.assertRaises(TypeError):  # the sidecar's recomputation flag no longer exists
                c.cpu_proof(binding, semantic, {}, {}, parent_fingerprint=c.SEAL, sidecar_fingerprint='x')
            for change in ({'binding': {}}, {'training_rows': 216_960}, {'complete_training_chunks': 203_040},
                           {'train_episodes': list(range(480))}, {'training_stats': {}},
                           {'processor_check': {}}, {'semantic_gate': {}}, {'gpu_executed': True},
                           {'parent_fingerprint_recomputed': False}, {'validation_episodes': []},
                           {'extra_constants': {}}, {'optimizer_updates': 1}, {'status': 'failed'}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    c.validate_cpu({**report, **change}, binding)
            wrong = c.cpu_proof(binding, semantic, {'action': {'mean': [0.]*8}}, {'batch_size': 4}, parent_fingerprint='0'*64)
            self.assertFalse(wrong['parent_fingerprint_recomputed'])
            with self.assertRaises(ValueError):
                c.validate_cpu(wrong, binding)
            changed = {**constants, 'mean': constants['mean']+1e-9}  # the constants of the proof are the ones now derived
            with patch.object(c.forces, 'constants', return_value=changed), self.assertRaises(ValueError):
                c.validate_cpu(report, binding)

    def test_later_admission_rehashes_without_full_fingerprint_repeat(self):
        c = self.c
        constants = {key: np.arange(8, dtype=np.float64) for key in ('low', 'high', 'mean', 'std')}
        with patch.object(c.forces, 'constants', return_value=constants), patch.object(
                c.forces, 'clipping_rates', return_value={'train': {}, 'validation': {}}):
            semantic = {'episodes': 3000}
            binding = {'validation_sha256': c.digest_json(semantic)}
            report = c.cpu_proof(binding, semantic, {'action': {'mean': [0.]*8}}, {'batch_size': 4}, parent_fingerprint=c.SEAL)
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
            self.assertEqual(json.loads((new/'run_claim.json').read_text())['purpose'], 'xvla_g1_rand')
            link = root/'link'
            link.symlink_to(new, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.c.claim_output(link)

    def test_snapshot_schedule_excludes_the_final_update(self):
        c = self.c
        updates = c.snapshot_updates(False)
        self.assertEqual((updates[0], updates[1]), (2600, 5200))
        self.assertEqual(updates[-1], max(u for u in range(c.SNAPSHOT_EVERY, c.TOTAL_UPDATES, c.SNAPSHOT_EVERY)))
        self.assertNotIn(c.TOTAL_UPDATES, updates)  # the final checkpoint carries the optimizer
        self.assertEqual(c.snapshot_updates(True), [1])
        self.assertEqual(c.recipe(False)['snapshot_updates'], updates)

    def test_resume_claim_needs_the_original_mount_and_numbers_each_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(self.c, 'is_mountpoint', return_value=True):
                with self.assertRaises(FileExistsError):  # nothing was ever claimed here
                    self.c.claim_resume(root)
                self.c.claim_output(root)
                self.assertEqual(self.c.claim_resume(root), 1)
                self.assertEqual(self.c.claim_resume(root), 2)
                self.assertEqual(json.loads((root/'resume_claim_02.json').read_text())['resume'], 2)
                (root/'run_claim.json').write_text(json.dumps({'purpose': 'someone_else', 'resume': False}))
                with self.assertRaises(ValueError):
                    self.c.claim_resume(root)
            with self.assertRaises(FileExistsError):  # not a bind mount
                self.c.claim_resume(root)
            link = root/'link'
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.c.claim_resume(link)

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
        self.assertEqual(c.checkpoint_directory(126900), 'step_126900')
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
            self.assertEqual(recipe['purpose'], 'xvla_g1_rand_preflight' if scratch else 'xvla_g1_rand')
            self.assertEqual(recipe['seed'], 20261007)
            self.assertEqual(recipe['train_episodes'], list(range(2700)))
            self.assertEqual(recipe['validation_episodes'], list(range(2700, 3000)))
            self.assertEqual(recipe['resume'], 'manual_from_verified_snapshot')
            self.assertEqual(recipe['snapshot_updates'], c.snapshot_updates(scratch))
            self.assertEqual(recipe['keep_snapshots'], 2)
            self.assertFalse(recipe['retries'])
            self.assertEqual(recipe['precision'], 'fp32_parameters_gradients_adam_bf16_autocast')
            self.assertEqual(recipe['optimizer_storage'], 'final_only')
            self.assertEqual(recipe['schedule']['total'], c.TOTAL_UPDATES)
        self.assertEqual(c.TOTAL_UPDATES, 126_900)
        self.assertEqual(c.checkpoint_updates(True), [1, 2])
        updates = c.checkpoint_updates(False)
        self.assertEqual(updates, [1, 50760, 76140, 88830, 101520, 114210, 126900])
        self.assertEqual(updates[-1], c.TOTAL_UPDATES)
        self.assertEqual(len(updates), 7)
        # Exposure: M2's 126,900 updates are 0.89 epoch of the 1,142,100 complete starts here (A2's were 5.0 of 203,040).
        self.assertEqual(c.TOTAL_UPDATES*8, 1_015_200)
        self.assertEqual(round(c.TOTAL_UPDATES*8/c.COMPLETE_STARTS, 2), 0.89)
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
        # The fork-source residue check lives in test_fork_hygiene.py; duplicating those exact needles here
        # would self-match this file's own string literals once copied forward to a future fork.
        for name in ('contract.py', 'runner.py'):
            source = (Path(__file__).parents[1]/name).read_text()
            self.assertNotIn('from .evaluate', source, msg=f'Forbidden import in {name}')

    def test_explicit_source_files_only(self):
        names = self.c.SOURCE_FILES
        self.assertEqual(len(names), len(set(names)), 'duplicate source file listed')
        for name in names:
            self.assertTrue(self.c.source_path(name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
