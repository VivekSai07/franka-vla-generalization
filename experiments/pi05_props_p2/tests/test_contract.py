"""CPU contracts: the pi0.5 P2 package's data binding, windows, statistics, recipe and config.

Runs in `admission-pi`. Tests marked as needing the parent skip when the sealed P2 root is not mounted at
outputs/props_p2_data/props_p2_20261013.
"""
import copy
import importlib
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.props_p2_data import plan as p2_plan

PLAN = p2_plan.make_plan()
PLANNED = tuple(episode['instruction'] for episode in PLAN['episodes'])
TRAINED = tuple(sorted(p2_plan.instruction(obj, bin_) for obj, bin_ in p2_plan.TRAINED_PAIRS))
FIRST_APPEARANCE = tuple(dict.fromkeys(PLANNED))
TASK_OF_EPISODE = tuple(FIRST_APPEARANCE.index(x) for x in PLANNED)
SMALL_PARENT_FILES = ('seal.json', 'plan.json', 'substitutions.json', 'provenance.json', 'validation.json')
FRAMES = 452
CHUNK = 50


def metadata():
    return [dict(episode_index=i, length=FRAMES, tasks=[PLANNED[i]],
                 dataset_from_index=i*FRAMES, dataset_to_index=(i+1)*FRAMES) for i in range(540)]


def task_table(order):
    return pd.DataFrame({'task_index': list(range(len(order)))}, index=list(order))


class _DS:
    def __init__(self, order):
        self.meta = type('Meta', (), {'tasks': task_table(order)})()


class ContractTestCase(unittest.TestCase):
    def setUp(self):
        self.c = importlib.import_module('experiments.pi05_props_p2.contract')

    def require_parent(self):
        if not (self.c.PARENT/'seal.json').exists():
            self.skipTest('Props P2 parent not mounted')

    def copy_parent(self, tmp):
        self.require_parent()
        parent = Path(tmp)/'parent'
        parent.mkdir()
        for name in SMALL_PARENT_FILES:
            shutil.copy(self.c.PARENT/name, parent/name)
        return parent

    def binding_with(self, parent, **patches):
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.c, 'PARENT', parent))
            stack.enter_context(patch.object(self.c.assets, 'require_assets', return_value={'inventory': 'x'}))
            stack.enter_context(patch.object(self.c, 'source_hashes', return_value={}))
            for key, value in patches.items():
                stack.enter_context(patch.object(self.c, key, value))
            return self.c.current_binding()


class WindowTests(ContractTestCase):
    def test_complete_fifty_step_windows_only(self):
        c = self.c
        windows = c.windows()
        self.assertEqual(c.CHUNK_SIZE, CHUNK)
        self.assertEqual(len(windows), 193_440)
        self.assertEqual(len(windows), 480*(FRAMES-CHUNK+1))
        self.assertEqual(len(set(windows)), len(windows))
        self.assertTrue(all(w//FRAMES < 480 and w % FRAMES + CHUNK <= FRAMES for w in windows))
        self.assertEqual(windows[:2], [0, 1])
        self.assertEqual(windows[402:404], [402, FRAMES])
        self.assertEqual(windows[-1], 479*FRAMES+402)

    def test_validation_episodes_never_start_a_window(self):
        self.assertFalse(any(w//FRAMES >= 480 for w in self.c.windows()))

    def test_a_chunk_30_window_set_is_not_accepted(self):
        with patch.object(self.c, 'CHUNK_SIZE', 30):
            self.assertEqual(len(self.c.windows()), 203_040)
        self.assertEqual(len(self.c.windows()), 193_440)


class ExposureAndScheduleTests(ContractTestCase):
    def test_horizon_is_derived_from_five_epochs_not_typed(self):
        c = self.c
        derived = round(5.0*len(c.windows())/8)
        self.assertEqual(derived, 120_900)
        self.assertEqual(c.recipe(False)['max_optimizer_updates'], derived)
        self.assertEqual(c.TOTAL_UPDATES, derived)
        self.assertEqual(c.recipe(True)['max_optimizer_updates'], 2)
        source = (Path(c.__file__)).read_text()
        for typed in ('120900', '120_900', '48360', '72540', '84630', '96720', '108810'):
            self.assertNotIn(typed, source)

    def test_five_epochs_is_exact(self):
        self.assertEqual(self.c.TOTAL_UPDATES*8, 5*len(self.c.windows()))

    def test_checkpoint_grid_is_the_x_vla_fractions_plus_update_one(self):
        c = self.c
        grid = c.checkpoint_updates(False)
        self.assertEqual(grid, [1] + [round(f*c.TOTAL_UPDATES) for f in (.4, .6, .7, .8, .9, 1.0)])
        self.assertEqual(grid, [1, 48_360, 72_540, 84_630, 96_720, 108_810, 120_900])
        self.assertEqual(grid[-1], c.recipe(False)['max_optimizer_updates'])
        self.assertEqual(c.checkpoint_updates(True), [1, 2])

    def test_exposure_is_reported_as_three_numbers(self):
        e = self.c.exposure()
        self.assertEqual(e, dict(observation_starts_per_epoch=193_440, samples=967_200,
                                 supervised_action_positions=967_200*50, chunk_size=50,
                                 x_vla_observation_starts_per_epoch=203_040, x_vla_samples=1_015_200,
                                 x_vla_supervised_action_positions=1_015_200*30))
        self.assertEqual(self.c.recipe(False)['exposure'], e)

    def test_recipe_is_the_native_one_with_the_named_deviations(self):
        r = self.c.recipe(False)
        self.assertEqual((r['effective_batch'], r['chunk_size'], r['n_action_steps']), (8, 50, 10))
        self.assertEqual(r['optimizer'], dict(peak_lr=2.5e-5, betas=[.9, .95], eps=1e-8, weight_decay=.01, clip=1.0))
        self.assertEqual(r['schedule'], dict(warmup=1000, total=120_900, decay_lr=2.5e-6))
        self.assertEqual(r['deviations'], ['effective_batch_8', 'decay_spans_run', 'n_action_steps_10', 'vlm_frozen_expert_only'])
        self.assertEqual(r['arm'], 'pi0.5, action expert fine-tuned (VLM frozen)')
        self.assertEqual(r['normalization'], 'quantiles_train_only_q01_q99_no_clipping')
        self.assertEqual(r['train_episodes'], list(range(480)))
        self.assertEqual(r['validation_episodes'], list(range(480, 540)))

    def test_preflight_recipe_differs_only_in_purpose_and_horizon(self):
        full, pre = self.c.recipe(False), self.c.recipe(True)
        self.assertEqual(pre['purpose'], 'pi05_props_p2_preflight')
        self.assertEqual(full['purpose'], 'pi05_props_p2')
        for key in set(full) - {'purpose', 'max_optimizer_updates'}:
            self.assertEqual(full[key], pre[key], key)


class ConfigTests(ContractTestCase):
    def test_config_carries_the_specs_values(self):
        c = self.c.make_config('cpu')
        self.assertEqual((c.chunk_size, c.n_action_steps, c.max_state_dim, c.max_action_dim), (50, 10, 32, 32))
        self.assertEqual({k: tuple(v.shape) for k, v in c.input_features.items()},
                         {'observation.state': (9,), 'observation.images.image': (3, 512, 512),
                          'observation.images.image2': (3, 512, 512)})
        self.assertEqual({k: tuple(v.shape) for k, v in c.output_features.items()}, {'action': (8,)})
        self.assertEqual(c.empty_cameras, 0)
        self.assertEqual((c.image_resolution, c.tokenizer_max_length), ((224, 224), 200))
        self.assertEqual((c.paligemma_variant, c.action_expert_variant), ('gemma_2b', 'gemma_300m'))
        self.assertEqual((c.optimizer_lr, c.optimizer_betas, c.optimizer_eps, c.optimizer_weight_decay,
                          c.optimizer_grad_clip_norm), (2.5e-5, (.9, .95), 1e-8, .01, 1.0))
        self.assertEqual((c.scheduler_warmup_steps, c.scheduler_decay_steps, c.scheduler_decay_lr),
                         (1000, self.c.TOTAL_UPDATES, 2.5e-6))
        self.assertEqual({str(getattr(k, 'name', k)): str(getattr(v, 'name', v))
                          for k, v in c.normalization_mapping.items()},
                         {'VISUAL': 'IDENTITY', 'STATE': 'QUANTILES', 'ACTION': 'QUANTILES'})
        self.assertFalse(c.gradient_checkpointing)
        self.assertEqual(c.device, 'cpu')

    def test_camera_order_is_the_datasets(self):
        self.assertEqual(self.c.CAMERAS, ('observation.images.image', 'observation.images.image2'))
        self.assertEqual(list(self.c.make_config('cpu').image_features), list(self.c.CAMERAS))


class IndexAndStatisticsTests(ContractTestCase):
    def test_indices_accept_the_sealed_layout_and_reject_malformed_metadata(self):
        rows, starts = self.c.indices(metadata())
        self.assertEqual(rows, list(range(216_960)))
        self.assertEqual(starts, self.c.windows())
        for episodes in (metadata()[:200], metadata()[:24], metadata()+metadata()[:1]):
            with self.assertRaises(ValueError):
                self.c.indices(episodes)
        for change in ({'episode_index': 900}, {'length': 451}, {'tasks': ['wrong']},
                       {'dataset_from_index': 0}, {'dataset_to_index': 10**7}):
            for victim in (1, 479, 480, 539):
                episodes = metadata()
                episodes[victim].update(change)
                with self.subTest(change=change, victim=victim), self.assertRaises(ValueError):
                    self.c.indices(episodes)

    def test_one_changed_instruction_is_rejected(self):
        episodes = metadata()
        other = next(t for t in TRAINED if t != PLANNED[7])
        episodes[7]['tasks'] = [other]
        with self.assertRaises(ValueError):
            self.c.indices(episodes)

    def test_statistics_read_exactly_the_training_rows_and_no_validation_row(self):
        c = self.c
        seen = []

        class Rows:
            def __getitem__(self, i):
                if i >= c.TRAINING_ROWS:
                    raise AssertionError('validation row read for statistics')
                seen.append(i)
                value = float(i % 100)
                return dict(episode_index=i//FRAMES, task_index=TASK_OF_EPISODE[i//FRAMES],
                            action=torch.full((8,), value), **{'observation.state': torch.full((9,), value)})
        stats = c.statistics(Rows(), list(range(c.TRAINING_ROWS)), TASK_OF_EPISODE)
        self.assertEqual(len(seen), 216_960)
        self.assertEqual(sorted(seen), list(range(216_960)))
        expected = np.arange(216_960) % 100
        for key, width in (('observation.state', 9), ('action', 8)):
            self.assertEqual(set(stats[key]), {'mean', 'std', 'min', 'max', 'q01', 'q99'})
            for name, reference in dict(mean=expected.mean(), std=expected.std(), min=0., max=99.,
                                        q01=np.quantile(expected, .01), q99=np.quantile(expected, .99)).items():
                self.assertEqual(tuple(stats[key][name].shape), (width,))
                np.testing.assert_allclose(stats[key][name].numpy(), reference, rtol=1e-5, atol=1e-5,
                                           err_msg=f'{key} {name}')
        for rows in (list(range(c.TRAINING_ROWS+1)), [0]*c.TRAINING_ROWS, list(range(1, c.TRAINING_ROWS+1))):
            with self.assertRaises(ValueError):
                c.statistics(Rows(), rows, TASK_OF_EPISODE)

    def test_statistics_reject_a_wrong_dtype_width_or_nonfinite_row(self):
        c = self.c
        good = dict(episode_index=0, task_index=TASK_OF_EPISODE[0], action=torch.zeros(8),
                    **{'observation.state': torch.zeros(9)})
        for change in ({'action': torch.zeros(8, dtype=torch.float64)}, {'action': torch.zeros(7)},
                       {'observation.state': torch.full((9,), float('nan'))},
                       {'observation.state': torch.zeros(32)}, {'episode_index': 1}):
            rows = [{**good, **change}] + [good]*(c.TRAINING_ROWS-1)
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                c.statistics(rows, list(range(c.TRAINING_ROWS)), TASK_OF_EPISODE)


class InstructionTests(ContractTestCase):
    def test_the_approved_set_is_the_ten_trained_instructions(self):
        c = self.c
        self.assertEqual(tuple(c.INSTRUCTIONS), TRAINED)
        self.assertEqual(len(c.INSTRUCTIONS), 10)
        for held_out in (p2_plan.instruction('milk', 'blue'), p2_plan.instruction('bread', 'orange')):
            self.assertNotIn(held_out, c.INSTRUCTIONS)
            with self.assertRaises(ValueError):
                c.require_instruction(held_out)
        for text in TRAINED:
            c.require_instruction(text)

    def test_task_index_is_mapped_by_string_not_by_number(self):
        c = self.c
        self.assertNotEqual(FIRST_APPEARANCE, tuple(sorted(FIRST_APPEARANCE)))
        for order in (FIRST_APPEARANCE, TRAINED, tuple(reversed(TRAINED))):
            table = c.task_index_map(_DS(order))
            inverse = {v: k for k, v in table.items()}
            for episode, text in enumerate(PLANNED):
                self.assertEqual(inverse[table[text]], text)
        for order in (TRAINED[:1], TRAINED[:9], TRAINED+(p2_plan.instruction('milk', 'blue'),)):
            with self.assertRaises(ValueError):
                c.task_index_map(_DS(order))

    def test_load_data_maps_tasks_through_the_dataset_table_only(self):
        source = Path(self.c.__file__).read_text()
        self.assertIn('task_index_map(ds)[x] for x in planned_instructions()', source)
        self.assertNotIn('INSTRUCTIONS.index', source)
        self.assertIn("splits={'train': '0:540'}", source)

    def test_planned_instructions_cover_every_episode_and_both_splits(self):
        self.require_parent()
        c = self.c
        self.assertEqual(c.planned_instructions(), PLANNED)
        for ids in (c.TRAIN, c.VAL):
            self.assertEqual({PLANNED[i] for i in ids}, set(TRAINED))

    def test_real_dataset_metadata_is_accepted_and_one_change_rejected(self):
        self.require_parent()
        c = self.c
        episodes = pd.read_parquet(next((c.DATA/'meta/episodes').rglob('*.parquet')))
        self.assertEqual(len(episodes), 540)
        rows = episodes.to_dict('records')
        rows = [dict(r, tasks=list(r['tasks'])) for r in rows]
        c.indices(rows)
        rows[5]['tasks'] = [next(t for t in TRAINED if t != rows[5]['tasks'][0])]
        with self.assertRaises(ValueError):
            c.indices(rows)


class BindingTests(ContractTestCase):
    def test_seal_record_matches_the_collection_report(self):
        c = self.c
        self.assertEqual(c.SEAL, 'b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898')
        self.assertEqual(set(c.SEAL_RECORD), {'fingerprint', 'provenance_sha256', 'plan_sha256',
                                              'substitutions_sha256', 'image', 'episodes'})
        self.assertEqual(c.SEAL_RECORD['plan_sha256'], p2_plan.digest(PLAN))

    def test_current_binding_accepts_the_real_parent(self):
        self.require_parent()
        c = self.c
        from experiments.props_p2_data import contract as p2_contract
        with patch.object(c.assets, 'require_assets', return_value={'inventory': 'x'}):
            binding = c.current_binding()
        self.assertEqual(binding['dataset_fingerprint'], c.SEAL)
        self.assertEqual(binding['props_p2_provenance'], p2_contract.provenance())
        self.assertEqual(binding['required_image'], c.IMAGE)
        self.assertEqual(binding['plan_sha256'], c.SEAL_RECORD['plan_sha256'])
        for key in ('seal_sha256', 'provenance_sha256', 'validation_sha256', 'sources', 'assets'):
            self.assertIn(key, binding)

    def test_a_copy_of_the_parent_binds_and_each_tamper_is_rejected(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            parent = self.copy_parent(tmp)
            self.assertEqual(self.binding_with(parent)['dataset_fingerprint'], c.SEAL)
            seal = json.loads((parent/'seal.json').read_text())
            for key in seal:
                changed = {**seal, key: ('x'*64 if isinstance(seal[key], str) else seal[key]+1)}
                (parent/'seal.json').write_text(json.dumps(changed))
                with self.subTest(seal_field=key), self.assertRaises(ValueError):
                    self.binding_with(parent)
            (parent/'seal.json').write_text(json.dumps(seal))
            plan = json.loads((parent/'plan.json').read_text())
            plan['episodes'][0]['seed'] += 1
            (parent/'plan.json').write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                self.binding_with(parent)
            shutil.copy(c.PARENT/'plan.json', parent/'plan.json')
            for changed in ({'5': 1}, {'539': 3}):
                (parent/'substitutions.json').write_text(json.dumps(changed))
                with self.subTest(substitutions=changed), self.assertRaises(ValueError):
                    self.binding_with(parent)
            shutil.copy(c.PARENT/'substitutions.json', parent/'substitutions.json')
            validation = json.loads((parent/'validation.json').read_text())
            (parent/'validation.json').write_text(json.dumps({**validation, 'train': 479}))
            with self.assertRaises(ValueError):
                self.binding_with(parent)
            (parent/'validation.json').write_text(json.dumps({**validation, 'extra': 1}))
            with self.assertRaises(ValueError):
                self.binding_with(parent)

    def test_drifted_provenance_is_rejected_both_ways(self):
        from experiments.props_p2_data import contract as p2_contract
        with tempfile.TemporaryDirectory() as tmp:
            parent = self.copy_parent(tmp)
            real = p2_contract.provenance()
            with patch.object(p2_contract, 'provenance', return_value={**real, 'experiments/x.py': '0'*64}):
                with self.assertRaises(ValueError):
                    self.binding_with(parent)
            changed = dict(real)
            changed[next(iter(changed))] = '0'*64
            with patch.object(p2_contract, 'provenance', return_value=changed):
                with self.assertRaises(ValueError):
                    self.binding_with(parent)

    def test_source_files_are_explicit_and_exclude_the_evaluation_adapter(self):
        names = self.c.SOURCE_FILES
        self.assertIsInstance(names, tuple)
        self.assertEqual(len(names), len(set(names)))
        for banned in ('eval_adapter', 'eval_tests'):
            self.assertFalse([n for n in names if banned in n], banned)
        self.assertIn('experiments/pi05_props_p2/README.md', names)
        self.assertGreater((Path(self.c.ROOT)/'experiments/pi05_props_p2/README.md').stat().st_size, 0)

    def test_source_hashes_cover_the_installed_pi05_and_processor_sources(self):
        names = self.c.SOURCE_FILES
        for needed in ('installed/lerobot/policies/pi05/modeling_pi05.py',
                       'installed/lerobot/policies/pi05/processor_pi05.py',
                       'installed/lerobot/policies/pi05/configuration_pi05.py',
                       'installed/lerobot/processor/normalize_processor.py',
                       'installed/lerobot/processor/tokenizer_processor.py'):
            self.assertIn(needed, names)


def c_root(contract):
    return contract.ROOT


class RunStructureTests(ContractTestCase):
    def test_learning_rate_is_the_installed_cosine_with_warmup_scheduler(self):
        c = self.c
        config = c.make_config('cpu')
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=config.optimizer_lr)
        scheduler = config.get_scheduler_preset().build(optimizer, c.TOTAL_UPDATES)
        checked = {1, 2, 500, 999, 1000, 1001, 1002, 5000, 60_000, c.TOTAL_UPDATES-1, c.TOTAL_UPDATES}
        for update in range(1, c.TOTAL_UPDATES+1):
            if update in checked:
                self.assertAlmostEqual(optimizer.param_groups[0]['lr'], c.learning_rate(update), delta=1e-15, msg=update)
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(c.learning_rate(1), 2.5e-5/1001, delta=1e-15)
        self.assertAlmostEqual(c.learning_rate(1001), 2.5e-5, delta=1e-8)
        self.assertAlmostEqual(c.learning_rate(c.TOTAL_UPDATES), 2.5e-6, delta=2e-9)
        for bad in (0, c.TOTAL_UPDATES+1, 1.0, '1'):
            with self.assertRaises(ValueError):
                c.learning_rate(bad)

    def test_the_epoch_sampler_covers_every_window_once_and_is_deterministic(self):
        import itertools
        c = self.c
        starts = c.windows()
        per_epoch = len(starts)//c.EFFECTIVE_BATCH
        self.assertEqual(per_epoch, 24_180)
        groups = list(itertools.islice(c.sample_updates(starts), 2*per_epoch))
        for epoch in (groups[:per_epoch], groups[per_epoch:]):
            flat = list(itertools.chain.from_iterable(epoch))
            self.assertEqual(sorted(flat), starts)
        self.assertNotEqual(groups[:per_epoch], groups[per_epoch:])
        self.assertTrue(all(len(g) == 8 for g in groups))
        self.assertEqual(list(itertools.islice(c.sample_updates(starts), 50)), groups[:50])
        for bad in (starts[:-1], starts[::-1], starts+[0]):
            with self.assertRaises(ValueError):
                next(c.sample_updates(bad))

    def test_policy_row_accepts_every_trained_instruction_and_rejects_everything_else(self):
        c = self.c
        def row(episode, **change):
            base = {'episode_index': torch.tensor(episode), 'task': PLANNED[episode],
                    'observation.state': torch.zeros(9), 'observation.images.image': torch.zeros(3, 512, 512),
                    'observation.images.image2': torch.zeros(3, 512, 512), 'action': torch.zeros(50, 8),
                    'action_is_pad': torch.zeros(50, dtype=torch.bool)}
            return {**base, **change}
        with patch.object(c, 'planned_instructions', return_value=PLANNED):
            seen = set()
            for episode in c.TRAIN:
                if PLANNED[episode] not in seen:
                    seen.add(PLANNED[episode])
                    self.assertEqual(c.policy_row(row(episode))['task'], PLANNED[episode])
            self.assertEqual(seen, set(TRAINED))
            self.assertEqual(set(c.policy_row({**row(0), 'success': True})), set(c.OBS)|{'task', 'action', 'action_is_pad'})
            c.policy_row(row(539), training=False)
            for label, change in {
                    'validation episode': dict(episode_index=torch.tensor(480)),
                    'wrong task': dict(task='wrong'), 'extra observation': {'observation.objects': torch.ones(3)},
                    'chunk 30': dict(action=torch.zeros(30, 8), action_is_pad=torch.zeros(30, dtype=torch.bool)),
                    'padded': dict(action_is_pad=torch.ones(50, dtype=torch.bool)),
                    'state 32': {'observation.state': torch.zeros(32)}, 'nan action': dict(action=torch.full((50, 8), float('nan'))),
                    'image out of range': {'observation.images.image': torch.full((3, 512, 512), 2.0)},
                    'image dtype': {'observation.images.image2': torch.zeros(3, 512, 512, dtype=torch.float64)},
                    'image size': {'observation.images.image': torch.zeros(3, 224, 224)},
                    'float episode': dict(episode_index=torch.tensor(1.0))}.items():
                with self.subTest(label), self.assertRaises(ValueError):
                    c.policy_row(row(0, **change))
            with self.assertRaises(ValueError):
                c.policy_row(row(539), training=True)

    def test_every_strategy_changes_only_what_it_names(self):
        c = self.c
        base = c.make_config('cpu')
        for name, extra in c.STRATEGIES.items():
            config = c.make_config('cpu', name)
            for key, value in extra.items():
                self.assertEqual(getattr(config, key), value, (name, key))
            for key in ('chunk_size', 'n_action_steps', 'max_state_dim', 'optimizer_lr', 'scheduler_decay_steps',
                        'tokenizer_max_length', 'image_resolution', 'empty_cameras'):
                self.assertEqual(getattr(config, key), getattr(base, key), (name, key))
        with self.assertRaises(KeyError):
            c.make_config('cpu', 'fp32_everything')

    def test_the_users_decision_is_recorded_and_the_recipe_carries_it(self):
        c = self.c
        self.assertEqual(c.DECISION, {'strategy': 'expert_only', 'microbatch': 8, 'snapshot_every': 2600,
                                      'checkpoint_bytes': 8_622_597_928, 'snapshot_bytes': 12_063_804_506,
                                      'final_state_bytes': 3_440_963_403, 'gpu_peak_bytes': 32_088_522_752})
        self.assertEqual(c.require_decision(), c.DECISION)
        self.assertEqual(c.recipe(False)['decision'], c.DECISION)
        self.assertEqual(c.recipe(True)['decision'], c.DECISION)
        self.assertEqual(c.accumulation(), 1)
        self.assertEqual(c.snapshot_updates(False)[:3], [2600, 5200, 7800])
        self.assertEqual(c.snapshot_updates(False)[-1], 2600*46)
        self.assertEqual(c.snapshot_updates(True), [1])
        self.assertEqual((c.KEEP_SNAPSHOTS, c.SNAPSHOT_SLOTS), (1, 2))
        self.assertEqual(c.disk_requirement_bytes(False), 7*c.DECISION['checkpoint_bytes'] + c.DECISION['final_state_bytes']
                         + 2*c.DECISION['snapshot_bytes'] + 2*1024**3)
        self.assertEqual(c.gpu_requirement_bytes(), c.DECISION['gpu_peak_bytes'] + 2*1024**3)

    def test_the_decision_matches_the_probes_measurements(self):
        probe = Path(c_root(self.c))/'outputs/train/pi05_props_p2_probe/probe_expert_only_mb8.json'
        if not probe.exists():
            self.skipTest('probe output not mounted')
        d = json.loads(probe.read_text())
        self.assertEqual(self.c.DECISION['checkpoint_bytes'], d['sizes']['checkpoint_bytes'])
        self.assertEqual(self.c.DECISION['snapshot_bytes'], d['sizes']['snapshot_bytes'])
        self.assertEqual(self.c.DECISION['final_state_bytes'], d['sizes']['final_state_bytes'])
        self.assertEqual(self.c.DECISION['gpu_peak_bytes'], d['memory_peak']['peak_reserved'])
        self.assertEqual(d['strategy'], self.c.DECISION['strategy'])

    def test_with_no_decision_everything_that_needs_one_refuses(self):
        c = self.c
        with patch.object(c, 'DECISION', None):
            self.assertIsNone(c.recipe(False)['decision'])
            for call in (c.require_decision, lambda: c.snapshot_updates(False), c.accumulation,
                         lambda: c.disk_requirement_bytes(False), c.gpu_requirement_bytes):
                with self.assertRaisesRegex(RuntimeError, 'not decided'):
                    call()

    def test_claims_are_exclusive_and_resume_needs_the_original_claim(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run'
            c.claim_output(path)
            self.assertEqual(json.loads((path/'run_claim.json').read_text())['purpose'], 'pi05_props_p2')
            with self.assertRaises(FileExistsError):
                c.claim_output(path)
            with self.assertRaises(FileExistsError):
                c.claim_resume(path)  # a plain directory is not the bind-mounted output the claim requires

    def test_gate_validators_reject_a_changed_binding(self):
        c = self.c
        binding = {'validation_sha256': c.digest_json({'a': 1})}
        report = c.cpu_proof(binding, {'a': 1}, {'s': 1}, {'p': 1}, parent_fingerprint=c.SEAL)
        c.validate_cpu(report, binding)
        for key, value in (('status', 'failed'), ('gpu_executed', True), ('binding', {'x': 1}),
                           ('training_stats', {'s': 2}), ('processor_check', {'p': 2}),
                           ('parent_fingerprint_recomputed', False), ('complete_training_windows', 203_040)):
            with self.subTest(key), self.assertRaises(ValueError):
                c.validate_cpu({**report, key: value}, binding)

    def test_marker_requires_an_exact_complete_reload(self):
        c = self.c
        good = dict(status='complete', update=3, reload_weights_exact=True, reload_max_action_error=0.0)
        c.require_marker(good, 3)
        for change in ({'status': 'x'}, {'update': 4}, {'reload_weights_exact': False},
                       {'reload_max_action_error': 1e-3}, {'reload_max_action_error': float('nan')}):
            with self.assertRaises(ValueError):
                c.require_marker({**good, **change}, 3)


if __name__ == '__main__':
    unittest.main()
