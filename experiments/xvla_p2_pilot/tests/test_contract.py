"""CPU contract of the X-VLA P2 continuation pilots: recipe, schedule, variants, state builders, start binding, model inputs."""
import copy
import importlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.props_p2_data import plan as p2_plan

PLAN = p2_plan.make_plan()
PLANNED = tuple(episode['instruction'] for episode in PLAN['episodes'])
FIRST_APPEARANCE = tuple(dict.fromkeys(PLANNED))
TASK_OF_EPISODE = tuple(FIRST_APPEARANCE.index(x) for x in PLANNED)
FRAMES = 452
ROWS = [0, 421, 422, 203_039]  # first and last starts of episodes 0 and 449
VARIANTS = ('control', 'force', 'box')


def contract():
    return importlib.import_module('experiments.xvla_p2_pilot.contract')


def mounted():
    c = contract()
    return (c.forces.SIDECAR/'seal.json').is_file()


def bundle_mounted():
    c = contract()
    return (c.CACHE/'tokenizer').exists()


def start_mounted():
    return (contract().START/'model.safetensors').is_file()


def frame(episode=0, state=None):
    return {'episode_index': torch.tensor(episode), 'task': PLANNED[episode],
            'observation.state': torch.zeros(19) if state is None else state,
            'observation.images.image': torch.zeros(3, 512, 512),
            'observation.images.image2': torch.zeros(3, 512, 512),
            'action': torch.zeros(30, 8), 'action_is_pad': torch.zeros(30, dtype=torch.bool)}


def metadata():
    return [dict(episode_index=i, length=FRAMES, tasks=[PLANNED[i]],
                 dataset_from_index=i*FRAMES, dataset_to_index=(i+1)*FRAMES) for i in range(540)]


def joined_row(c, builder, index, state=None, zero=False, extras=None):
    """A training row at dataset position `index` with its state joined to the builder's extras."""
    episode, position = divmod(index, FRAMES)
    state = torch.linspace(-1, 1, 9)+episode*1e-3 if state is None else state
    joined = c.joined_state(state, episode, position, builder, zero)
    if extras is not None:
        joined[9:] = torch.as_tensor(extras, dtype=torch.float32)
    return {**frame(episode), 'observation.state': joined}


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()
        self.m2 = importlib.import_module('experiments.xvla_p2_force.contract')

    def test_the_pilot_recipe_is_fixed(self):
        r = self.c.recipe()
        self.assertEqual((r['max_optimizer_updates'], r['effective_batch'], r['microbatch'], r['accumulation']),
                         (10_000, 8, 4, 2))
        self.assertEqual(r['schedule'], dict(kind='linear_warmup_then_constant', warmup=200, lr=1e-5, total=10_000,
                                             vlm_scale=.1))
        self.assertEqual((r['start']['update'], r['optimizer_state']), (101_520, 'fresh'))
        self.assertEqual(r['optimizer_storage'], 'none')
        self.assertEqual(r['seed'], self.m2.SEED)
        self.assertEqual((r['autotune'], r['retries']), (False, False))
        self.assertNotIn('snapshot_updates', r)
        self.assertNotIn('checkpoint_updates', r)

    def test_the_optimizer_is_p2s_apart_from_the_peak_learning_rate(self):
        ours, theirs = dict(self.c.recipe()['optimizer']), dict(self.m2.recipe(False)['optimizer'])
        self.assertEqual(ours.pop('peak_lr'), 1e-5)
        self.assertEqual(theirs.pop('peak_lr'), 1e-4)
        self.assertEqual(ours, theirs)

    def test_the_data_split_and_sampler_are_m2s(self):
        c, m = self.c, self.m2
        for name in ('SEED', 'TRAIN', 'VAL', 'FRAMES_PER_EPISODE', 'CHUNK_SIZE', 'TRAINING_ROWS', 'COMPLETE_STARTS',
                     'TOTAL_FRAMES', 'SEAL', 'SEAL_RECORD', 'VALIDATION_RECORD', 'INSTRUCTIONS', 'CAMERAS'):
            self.assertEqual(getattr(c, name), getattr(m, name), name)
        self.assertIs(c.sample_batches, m.sample_batches)
        self.assertEqual(c.recipe()['train_episodes'], list(m.TRAIN))
        self.assertEqual(c.recipe()['validation_episodes'], list(m.VAL))

    def test_exposure_is_a_fraction_of_one_epoch(self):
        c = self.c
        self.assertEqual(c.UPDATES*8, 80_000)
        self.assertAlmostEqual(c.UPDATES*8/c.COMPLETE_STARTS, 0.394, places=3)

    def test_the_sample_order_is_the_same_for_every_variant(self):
        c = self.c
        starts = [i*FRAMES+j for i in c.TRAIN for j in range(FRAMES-c.CHUNK_SIZE+1)]
        reference = self.m2.sample_batches(starts)
        for _ in VARIANTS:  # the sampler takes no variant: one function, one seed
            stream = c.sample_batches(starts)
            again = self.m2.sample_batches(starts)
            for _ in range(25):
                self.assertEqual(next(stream), next(again))
        first = [next(reference) for _ in range(3)]
        self.assertTrue(all(len(batch) == 4 for batch in first))
        with self.assertRaises(ValueError):
            c.sample_batches(starts[:-1]).__next__()


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()

    def test_warm_up_is_linear_from_zero_to_1e_minus_5_over_200_updates_then_constant(self):
        c = self.c
        self.assertAlmostEqual(c.learning_rate(1), 1e-5/200, places=15)
        self.assertAlmostEqual(c.learning_rate(100), 5e-6, places=15)
        self.assertEqual(c.learning_rate(200), 1e-5)
        for update in (201, 1000, 5000, 9999, 10_000):
            self.assertEqual(c.learning_rate(update), 1e-5)
        steps = [c.learning_rate(u) for u in range(1, 201)]
        self.assertTrue(all(b > a for a, b in zip(steps, steps[1:])))
        self.assertAlmostEqual(steps[1]-steps[0], 1e-5/200, places=15)

    def test_the_schedule_refuses_updates_outside_the_horizon(self):
        for bad in (0, -1, 10_001, 1.0, True, '5', None):
            with self.subTest(update=bad), self.assertRaises(ValueError):
                self.c.learning_rate(bad)

    def test_the_horizon_is_one_number_shared_by_recipe_and_schedule(self):
        c = self.c
        r = c.recipe()
        self.assertEqual(r['max_optimizer_updates'], c.UPDATES)
        self.assertEqual(r['schedule']['total'], c.UPDATES)
        self.assertEqual(r['schedule']['warmup'], c.WARMUP)
        self.assertEqual(r['schedule']['lr'], c.LR)

    def test_set_learning_rate_scales_the_vlm_group_by_a_tenth_and_checks_the_recipe(self):
        c = self.c

        def optimizer(**changes):
            groups = []
            for name in ('vlm', 'soft_prompts', 'other'):
                scale = .1 if name == 'vlm' else 1.
                group = dict(params=[torch.nn.Parameter(torch.zeros(1))], name=name, lr=0., betas=(.9, .95), eps=1e-8,
                             weight_decay=1e-4*scale)
                group.update(changes)
                groups.append(group)
            return torch.optim.AdamW(groups)

        opt = optimizer()
        c.set_learning_rate(opt, 400)
        for group, expected in zip(opt.param_groups, (1e-6, 1e-5, 1e-5)):
            self.assertAlmostEqual(group['lr'], expected, places=15)
        c.set_learning_rate(opt, 100)
        self.assertAlmostEqual(opt.param_groups[2]['lr'], 5e-6, places=15)
        with self.assertRaises(ValueError):
            c.set_learning_rate(optimizer(betas=(.9, .99)), 1)
        with self.assertRaises(ValueError):
            c.set_learning_rate(optimizer(eps=1e-6), 1)
        opt = optimizer()
        opt.param_groups.pop()
        with self.assertRaises(ValueError):
            c.set_learning_rate(opt, 1)


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()

    def spec(self, **changes):
        return {**json.loads((self.c.VARIANTS/'force.json').read_text()), **changes}

    def write(self, directory, name, spec):
        (Path(directory)/f'{name}.json').write_text(json.dumps(spec))

    def test_the_shipped_variants_load_with_their_own_hash(self):
        for name in VARIANTS:
            variant = self.c.load_variant(name)
            self.assertEqual(variant['name'], name)
            self.assertEqual(variant['spec']['state_builder'], name)
            self.assertEqual(variant['sha256'], self.c.sha256(self.c.VARIANTS/f'{name}.json'))
            self.assertEqual(variant['path'], f'experiments/xvla_p2_pilot/variants/{name}.json')
            self.assertIsNone(variant['spec']['loss_term'])

    def test_a_file_outside_the_schema_is_rejected(self):
        bad = {
            'extra key': {**self.spec(), 'learning_rate': 1e-3},
            'missing key': {k: v for k, v in self.spec().items() if k != 'note'},
            'wrong schema': self.spec(schema='vla_inputs_pilot_variant_v2'),
            'unknown builder': self.spec(state_builder='drawn_boxes'),
            'loss term': self.spec(loss_term='aux_box_head'),
            'loss weight': self.spec(loss_weight=0.1),
            'loss weight type': self.spec(loss_weight=False),
            'name type': self.spec(name=3),
            'note type': self.spec(note=None),
            'name differs from the file': self.spec(name='control'),
        }
        for label, spec in bad.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                self.write(tmp, 'force', spec)
                with self.assertRaises(ValueError):
                    self.c.load_variant('force', Path(tmp))

    def test_malformed_json_duplicate_keys_and_non_objects_are_rejected(self):
        for label, text in {'list': '[]', 'garbage': '{', 'nan': '{"loss_weight": NaN}',
                            'duplicate': json.dumps(self.spec())[:-1]+', "name": "force"}'}.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp)/'force.json').write_text(text)
                with self.assertRaises(ValueError):
                    self.c.load_variant('force', Path(tmp))

    def test_a_variant_name_is_a_name_never_a_path(self):
        for bad in ('../control', 'a/b', 'Force', '', 'x'*40, 'force.json', '1force', None, 3):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                self.c.load_variant(bad)

    def test_a_missing_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            self.c.load_variant('force', Path(tmp))

    def test_a_changed_variant_changes_its_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp, 'force', self.spec(note='another note'))
            other = self.c.load_variant('force', Path(tmp))
        self.assertNotEqual(other['sha256'], self.c.load_variant('force')['sha256'])

    def test_variants_are_data_not_bound_source(self):
        for name in self.c.SOURCE_FILES:
            self.assertNotIn('variants/', name)
            self.assertNotIn('eval_adapter', name)
            self.assertNotIn('eval_tests', name)
        self.assertIn('variants/', self.c.VARIANTS.as_posix()+'/')


class BuilderTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.c = contract()

    def test_widths_and_tables(self):
        c = self.c
        for name, width in (('control', 0), ('force', 8), ('box', 10)):
            builder = c.BUILDERS[name]
            self.assertEqual(builder.width, width)
            table = builder.table()
            self.assertEqual((table.shape, table.dtype), ((540, FRAMES, width), np.float32))
            self.assertFalse(table.flags.writeable)
        self.assertEqual(c.MAX_EXTRAS, 10)
        self.assertEqual(c.STATE_WIDTH, 19)

    def test_force_is_m2s_table_itself(self):
        c = self.c
        self.assertIs(c.BUILDERS['force'].table(), c.forces.table())
        self.assertEqual(c.forces.constants_digest(c.forces.constants()), c.forces.CONSTANTS_SHA256)

    def test_box_is_the_sidecars_target_box_raw_and_flattened_front_camera_first(self):
        c = self.c
        table = c.BUILDERS['box'].table()
        for episode in (0, 137, 539):
            with np.load(c.forces.SIDECAR/'episodes'/f'episode_{episode:03d}.npz', allow_pickle=False) as stored:
                box = stored['target_box']
            self.assertEqual(box.shape, (FRAMES, 2, 5))
            np.testing.assert_array_equal(table[episode], box.reshape(FRAMES, 10).astype(np.float32))
        self.assertTrue(((table >= 0) & (table <= 1)).all())  # normalised pixel coordinates and the flag
        for camera in (0, 1):  # u_min < u_max and v_min < v_max wherever the target is in frame
            sub = table[:, :, 5*camera:5*camera+5].reshape(-1, 5)
            inside = sub[sub[:, 4] == 1]
            self.assertTrue((inside[:, 0] <= inside[:, 2]).all() and (inside[:, 1] <= inside[:, 3]).all())

    def test_box_table_is_pinned(self):
        c = self.c
        self.assertEqual(c.table_digest(c.BUILDERS['box'].table()), c.BOX_TABLE_SHA256)
        other = c.BUILDERS['box'].table().copy()
        other[3, 4, 0] += np.float32(1e-6)
        self.assertNotEqual(c.table_digest(other), c.BOX_TABLE_SHA256)

    def test_the_box_loader_verifies_the_sidecar_seal_first(self):
        c = self.c
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'sidecar'
            shutil.copytree(c.forces.SIDECAR, root)
            path = root/'episodes'/'episode_005.npz'
            with np.load(path, allow_pickle=False) as stored:
                arrays = {key: stored[key] for key in stored.files}
            arrays['target_box'] = arrays['target_box']+1e-3
            path.unlink()
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                c.load_boxes(root)

    def test_the_records_name_their_sources_and_pins(self):
        c = self.c
        force = c.builder_record(c.BUILDERS['force'])
        self.assertEqual(force['transform_source'], {'experiments/xvla_p2_force/forces.py':
                         c.sha256(c.source_path('experiments/xvla_p2_force/forces.py'))})
        self.assertEqual(force['constants_sha256'], c.forces.CONSTANTS_SHA256)
        box = c.builder_record(c.BUILDERS['box'])
        self.assertEqual(box['table_sha256'], c.BOX_TABLE_SHA256)
        self.assertEqual(c.builder_record(c.BUILDERS['control'])['width'], 0)


class JoinTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.c = contract()

    def test_the_state_is_19_values_joints_then_extras_then_zero_padding(self):
        c = self.c
        state = torch.linspace(-1, 1, 9)
        for name in VARIANTS:
            builder = c.BUILDERS[name]
            for episode, position in ((0, 0), (200, 451), (539, 17)):
                joined = c.joined_state(state, episode, position, builder)
                self.assertEqual((joined.shape, joined.dtype), ((19,), torch.float32))
                self.assertTrue(torch.equal(joined[:9], state))
                self.assertTrue(torch.equal(joined[9:9+builder.width],
                                            torch.from_numpy(np.array(builder.table()[episode, position]))))
                self.assertEqual(torch.count_nonzero(joined[9+builder.width:]).item(), 0)

    def test_control_and_zero_forced_variants_are_byte_equal(self):
        c = self.c
        state = torch.linspace(-1, 1, 9)
        for name in ('force', 'box'):
            for episode, position in ((0, 0), (12, 300), (539, 451)):
                forced = c.joined_state(state, episode, position, c.BUILDERS[name], zero=True)
                control = c.joined_state(state, episode, position, c.BUILDERS['control'])
                self.assertEqual(forced.numpy().tobytes(), control.numpy().tobytes())
                live = c.joined_state(state, episode, position, c.BUILDERS[name])
                self.assertFalse(torch.equal(live, control))

    def test_the_join_refuses_what_is_not_the_nine_dataset_values_or_not_a_frame(self):
        c = self.c
        builder = c.BUILDERS['force']
        for bad in (torch.zeros(17), torch.zeros(9, dtype=torch.float64), torch.full((9,), float('nan')),
                    torch.zeros(1, 9)):
            with self.subTest(shape=tuple(bad.shape), dtype=bad.dtype), self.assertRaises(ValueError):
                c.joined_state(bad, 0, 0, builder)
        for episode, position in ((540, 0), (-1, 0), (0, 452), (True, 0), (0.0, 0)):
            with self.subTest(episode=episode, position=position), self.assertRaises(ValueError):
                c.joined_state(torch.zeros(9), episode, position, builder)

    def test_expected_extras_are_padded_rows_of_the_table(self):
        c = self.c
        for name in VARIANTS:
            builder = c.BUILDERS[name]
            expected = c.expected_extras(builder, ROWS)
            self.assertEqual((expected.shape, expected.dtype), ((4, 10), np.float32))
            for k, i in enumerate(ROWS):
                np.testing.assert_array_equal(expected[k, :builder.width], builder.table()[i//FRAMES, i % FRAMES])
                self.assertEqual(np.count_nonzero(expected[k, builder.width:]), 0)
            self.assertEqual(np.count_nonzero(c.expected_extras(builder, ROWS, zero=True)), 0)

    def test_the_dataset_view_joins_by_sealed_position_and_nothing_else(self):
        c = self.c

        class Raw:
            def __len__(self):
                return 540*FRAMES

            def __getitem__(self, i):
                episode, position = divmod(i, FRAMES)
                return {'episode_index': episode, 'frame_index': position, 'observation.state': torch.zeros(9),
                        'marker': i}

            note = 'passes through'

        view = c.VariantDataset(Raw(), c.BUILDERS['box'])
        row = view[3*FRAMES+7]
        self.assertEqual((row['observation.state'].shape, row['marker']), ((19,), 3*FRAMES+7))
        self.assertTrue(torch.equal(row['observation.state'][9:19],
                                    torch.from_numpy(np.array(c.BUILDERS['box'].table()[3, 7]))))
        self.assertEqual((len(view), view.note), (540*FRAMES, 'passes through'))

        class Shifted(Raw):
            def __getitem__(self, i):
                return {**super().__getitem__(i), 'frame_index': (i+1) % FRAMES}

        with self.assertRaises(ValueError):
            c.VariantDataset(Shifted(), c.BUILDERS['force'])[100]
        zero = c.VariantDataset(Raw(), c.BUILDERS['force'], zero_extras=True)[100]['observation.state']
        self.assertEqual(torch.count_nonzero(zero[9:]).item(), 0)


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.c = contract()
        generator = np.random.default_rng(0)
        self.values = {w: generator.normal(size=(self.c.TRAINING_ROWS, w)).astype(np.float32) for w in (8, 9)}
        values, c = self.values, self.c

        class Rows:
            def __getitem__(self, i):
                if i >= c.TRAINING_ROWS:
                    raise AssertionError('validation accessed for stats')
                return dict(episode_index=i//FRAMES, frame_index=i % FRAMES, task_index=TASK_OF_EPISODE[i//FRAMES],
                            action=torch.from_numpy(values[8][i]), **{'observation.state': torch.from_numpy(values[9][i])})
        self.rows = Rows()

    def stats(self, name, zero=False):
        c = self.c
        return c.selected_stats(self.rows, list(range(c.TRAINING_ROWS)), TASK_OF_EPISODE, c.BUILDERS[name], zero)

    def test_19_state_and_8_action_values_whose_joint_and_action_part_do_not_depend_on_the_variant(self):
        base = self.stats('control')
        for name in VARIANTS:
            stats = self.stats(name)
            self.assertEqual({k: tuple(v.shape) for k, v in stats['observation.state'].items()},
                             {'mean': (19,), 'std': (19,), 'min': (19,), 'max': (19,)})
            self.assertEqual({k: tuple(v.shape) for k, v in stats['action'].items()},
                             {'mean': (8,), 'std': (8,), 'min': (8,), 'max': (8,)})
            for key in ('mean', 'std', 'min', 'max'):
                self.assertTrue(torch.equal(stats['observation.state'][key][:9], base['observation.state'][key][:9]))
                self.assertTrue(torch.equal(stats['action'][key], base['action'][key]))

    def test_the_extras_statistics_are_of_the_training_rows_of_the_variants_table(self):
        c = self.c
        for name in ('force', 'box'):
            builder = c.BUILDERS[name]
            table = builder.table()[:480].reshape(-1, builder.width).astype(np.float64)
            state = self.stats(name)['observation.state']
            np.testing.assert_allclose(state['mean'][9:9+builder.width].numpy(), table.mean(0), atol=2e-5)
            np.testing.assert_allclose(state['std'][9:9+builder.width].numpy(), table.std(0), atol=2e-5)
            self.assertEqual(torch.count_nonzero(state['mean'][9+builder.width:]).item(), 0)
            self.assertEqual(torch.count_nonzero(state['std'][9+builder.width:]).item(), 0)

    def test_control_and_zero_forced_statistics_are_the_same_bytes(self):
        a, b = self.stats('control'), self.stats('force', zero=True)
        for group in a:
            for key in a[group]:
                self.assertEqual(a[group][key].numpy().tobytes(), b[group][key].numpy().tobytes())

    def test_statistics_refuse_a_row_out_of_position_and_a_state_that_is_not_nine_values(self):
        c, rows = self.c, self.rows

        class Shifted:
            def __getitem__(self, i):
                value = rows[i]
                return {**value, 'frame_index': (value['frame_index']+1) % FRAMES} if i == 700 else value

        with self.assertRaises(ValueError):
            c.selected_stats(Shifted(), list(range(c.TRAINING_ROWS)), TASK_OF_EPISODE, c.BUILDERS['force'])

        class Wide:
            def __getitem__(self, i):
                return {**rows[i], 'observation.state': torch.zeros(17)}

        with self.assertRaises(ValueError):
            c.selected_stats(Wide(), list(range(c.TRAINING_ROWS)), TASK_OF_EPISODE, c.BUILDERS['force'])
        with self.assertRaises(ValueError):
            c.selected_stats(rows, list(range(100)), TASK_OF_EPISODE, c.BUILDERS['force'])


class ModelInputTests(unittest.TestCase):
    def setUp(self):
        if not mounted():
            self.skipTest('S0 sidecar not mounted')
        self.c = contract()
        if not bundle_mounted():
            self.skipTest('base bundle not mounted')
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
        self.config = self.c.make_config('cpu')
        self.policy = XVLAPolicy.__new__(XVLAPolicy)
        torch.nn.Module.__init__(self.policy)
        self.policy.config, self.policy.model = self.config, SimpleNamespace(dim_proprio=20, dim_action=20)
        self.pre, _ = make_xvla_pre_post_processors(self.config, {'action': {'mean': torch.arange(8).float(),
                                                                         'std': torch.ones(8)}})

    def batch(self, name, rows=ROWS, **kwargs):
        c = self.c
        raw = c.collate_rows([joined_row(c, c.BUILDERS[name], i, **kwargs) for i in rows])
        return raw, self.pre(raw)

    def check(self, name, batch, raw_state, rows=ROWS, **kwargs):
        return self.c.check_model_inputs(self.policy, batch, raw_state, rows, self.c.BUILDERS[name], **kwargs)

    def test_state_normalisation_stays_identity_and_the_state_feature_has_19_values(self):
        mapping = self.config.normalization_mapping
        self.assertEqual((mapping['STATE'].value, mapping['ACTION'].value), ('IDENTITY', 'MEAN_STD'))
        self.assertEqual(self.config.input_features['observation.state'].shape, (19,))
        self.assertEqual(self.config.max_state_dim, 20)

    def test_the_check_passes_for_every_variant_and_reports_what_the_model_sees(self):
        for name in VARIANTS:
            raw, batch = self.batch(name)
            evidence = self.check(name, batch, raw['observation.state'])
            self.assertEqual(evidence['proprio_shape'], [4, 20])
            self.assertEqual((evidence['state_width'], evidence['extra_width'], evidence['real_action_dim']),
                             (19, self.c.BUILDERS[name].width, 8))
            self.assertTrue(evidence['state_identity_exact'] and evidence['extras_exact'])
            proprio = self.policy._build_model_inputs(batch)['proprio']
            self.assertTrue(torch.equal(proprio[:, :19], raw['observation.state']))
            self.assertEqual(torch.count_nonzero(proprio[:, 19:]).item(), 0)

    def test_control_reaches_the_model_exactly_as_p2s_nine_value_state_does(self):
        raw, batch = self.batch('control')
        proprio = self.policy._build_model_inputs(batch)['proprio']
        self.assertEqual(torch.count_nonzero(proprio[:, 9:]).item(), 0)
        self.assertTrue(torch.equal(proprio[:, :9], raw['observation.state'][:, :9]))

    def test_the_check_fails_if_the_extras_are_missing_raw_or_belong_to_other_frames(self):
        c = self.c
        for name in ('force', 'box'):
            builder = c.BUILDERS[name]
            raw, batch = self.batch(name)
            with self.assertRaises(ValueError):  # state truncated to the joints
                bad = dict(batch)
                bad['observation.state'] = batch['observation.state'][:, :9]
                self.check(name, bad, raw['observation.state'])
            with self.assertRaises(ValueError):
                self.check(name, batch, raw['observation.state'], [r+1 for r in ROWS])
            with self.assertRaises(ValueError):
                self.check(name, batch, raw['observation.state'], ROWS[:3])
            shifted = [c.extras_row(builder, (i+1)//FRAMES, (i+1) % FRAMES) for i in ROWS]
            raw2 = c.collate_rows([joined_row(c, builder, i, extras=shifted[k]) for k, i in enumerate(ROWS)])
            with self.assertRaises(ValueError):
                self.check(name, self.pre(raw2), raw2['observation.state'])

    def test_a_variant_with_forced_zero_extras_is_checked_against_zeros_not_its_table(self):
        raw, batch = self.batch('force', zero=True)
        self.check('force', batch, raw['observation.state'], zero=True)
        with self.assertRaises(ValueError):
            self.check('force', batch, raw['observation.state'])

    def test_the_check_fails_on_a_different_batch_state_nonzero_padding_or_nonfinite_input(self):
        raw, batch = self.batch('force')
        other = dict(batch)
        other['observation.state'] = batch['observation.state'].clone()
        other['observation.state'][0, 3] += 1e-3
        with self.assertRaises(ValueError):
            self.check('force', other, raw['observation.state'])
        wide = dict(batch)
        wide['observation.state'] = torch.cat([batch['observation.state'], torch.ones(4, 1)], dim=1)  # a 20th value
        with self.assertRaises(ValueError):
            self.check('force', wide, raw['observation.state'])
        bad = dict(batch)
        bad['observation.state'] = batch['observation.state'].clone()
        bad['observation.state'][1, 12] = float('nan')
        with self.assertRaises(ValueError):
            self.check('force', bad, raw['observation.state'])
        padded = raw['observation.state'].clone()
        padded[2, 18] = .5  # the box slot force does not use must stay zero in the raw state too
        with self.assertRaises(ValueError):
            self.check('force', batch, padded)


class RowTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()

    def test_rows_are_checked_against_the_19_value_schema_and_the_planned_instruction(self):
        c = self.c
        good = c.policy_row(frame(0), True)
        self.assertEqual(good['observation.state'].shape, (19,))
        for label, row in {
            'state 9': {**frame(0), 'observation.state': torch.zeros(9)},
            'state 17': {**frame(0), 'observation.state': torch.zeros(17)},
            'nan state': {**frame(0), 'observation.state': torch.full((19,), float('nan'))},
            'wrong task': {**frame(0), 'task': 'put the lemon in the blue bin' if PLANNED[0] != 'put the lemon in the blue bin' else 'x'},
            'validation episode': frame(500),
            'padded chunk': {**frame(0), 'action_is_pad': torch.ones(30, dtype=torch.bool)},
            'extra key': {**frame(0), 'observation.environment_state': torch.zeros(3)},
        }.items():
            with self.subTest(label), self.assertRaises(ValueError):
                c.policy_row(row, True)
        c.policy_row(frame(500), False)

    def test_exactly_four_rows_per_microbatch(self):
        c = self.c
        self.assertEqual(c.collate_rows([frame(0)]*4)['observation.state'].shape, (4, 19))
        with self.assertRaises(ValueError):
            c.collate_rows([frame(0)]*3)


class StartTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()

    def test_the_pinned_start_is_the_bundles_checkpoint_101520(self):
        c = self.c
        self.assertEqual(c.START_UPDATE, 101_520)
        self.assertEqual(c.START.as_posix().split('outputs/')[-1], 'downloads_20261017/props_p2/checkpoints/step_101520')
        self.assertEqual(c.START_MODEL_SHA256, '59d02d0a578f3b70590534837f65b802764e68d02e05ae513c334ed99975932e')
        self.assertEqual(set(c.START_FILES), {'checkpoint_result.json', 'config.json', 'policy_postprocessor.json',
                                              'policy_postprocessor_step_0_unnormalizer_processor.safetensors',
                                              'policy_preprocessor.json',
                                              'policy_preprocessor_step_7_normalizer_processor.safetensors'})
        self.assertEqual(c.recipe()['start'], dict(update=101_520, model_sha256=c.START_MODEL_SHA256,
                                                   source='bundle props_p2 checkpoint, weights only'))

    def test_the_start_files_hash_to_their_pins_and_the_weights_to_their_marker(self):
        c = self.c
        if not start_mounted():
            self.skipTest('bundle checkpoint not mounted')
        record = c.start_binding()
        self.assertEqual(record['model_sha256'], c.START_MODEL_SHA256)
        self.assertEqual(record['files'], c.START_FILES)
        marker = json.loads((c.START/'checkpoint_result.json').read_text())
        self.assertEqual((marker['update'], marker['model_sha256']), (101_520, c.START_MODEL_SHA256))
        self.assertEqual(sorted(record), ['files', 'model_sha256', 'path', 'update'])

    def test_a_start_that_differs_from_the_pins_is_refused(self):
        c = self.c
        if not start_mounted():
            self.skipTest('bundle checkpoint not mounted')
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)/'step_101520'
            fake.mkdir()
            for name in c.START_FILES:
                shutil.copy(c.START/name, fake/name)
            (fake/'model.safetensors').write_bytes(b'not the weights')
            with patch.object(c, 'START', fake), self.assertRaises(ValueError):
                c.start_binding()
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)/'step_101520'
            fake.mkdir()
            for name in c.START_FILES:
                shutil.copy(c.START/name, fake/name)
            (fake/'model.safetensors').symlink_to(c.START/'model.safetensors')
            (fake/'config.json').write_text('{}')
            with patch.object(c, 'START', fake), self.assertRaises(ValueError):
                c.start_binding()

    def test_action_statistics_must_equal_the_ones_saved_in_the_start_checkpoint(self):
        c = self.c
        if not start_mounted():
            self.skipTest('bundle checkpoint not mounted')
        from safetensors.torch import load_file
        saved = load_file(str(c.START/'policy_preprocessor_step_7_normalizer_processor.safetensors'))
        stats = {'action': {key: saved[f'action.{key}'].clone() for key in ('mean', 'std', 'min', 'max')}}
        c.require_start_stats(stats)
        for key in ('mean', 'std', 'min', 'max'):
            other = copy.deepcopy(stats)
            other['action'][key][3] = torch.nextafter(other['action'][key][3], torch.tensor(1e9))
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.require_start_stats(other)
        with self.assertRaises(ValueError):
            c.require_start_stats({'action': {'mean': stats['action']['mean']}})


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()

    def test_the_only_checkpoint_is_step_010000(self):
        self.assertEqual(self.c.CHECKPOINT, 'step_010000')

    def test_a_marker_must_be_complete_exact_and_this_runs(self):
        c = self.c
        variant = dict(name='force', sha256='a'*64)
        marker = dict(status='complete', update=10_000, reload_fp32_weights_exact=True, reload_max_action_error=0.,
                      model_sha256='m', optimizer_saved=False, start_update=101_520,
                      start_model_sha256=c.START_MODEL_SHA256, variant='force', variant_sha256='a'*64)
        c.require_marker(marker, variant)
        for change in ({'update': 9_999}, {'status': 'partial'}, {'reload_fp32_weights_exact': False},
                       {'reload_max_action_error': 2e-5}, {'reload_max_action_error': float('nan')},
                       {'optimizer_saved': True}, {'start_update': 1}, {'start_model_sha256': 'x'},
                       {'variant': 'box'}, {'variant_sha256': 'b'*64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                c.require_marker({**marker, **change}, variant)

    def test_checkpoint_verification_wants_every_file_the_evaluator_loads_and_no_optimizer_state(self):
        c = self.c
        variant = dict(name='force', sha256='a'*64)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            folder = out/'checkpoints'/c.CHECKPOINT
            folder.mkdir(parents=True)
            for name in ('model.safetensors', 'config.json', 'policy_preprocessor.json', 'policy_postprocessor.json'):
                (folder/name).write_bytes(b'bytes')
            marker = dict(status='complete', update=10_000, reload_fp32_weights_exact=True, reload_max_action_error=0.,
                          model_sha256=c.sha256(folder/'model.safetensors'), optimizer_saved=False, start_update=101_520,
                          start_model_sha256=c.START_MODEL_SHA256, variant='force', variant_sha256='a'*64)
            c.write_json(folder/'checkpoint_result.json', marker)
            report = dict(checkpoints=[marker], checkpoint_files={c.CHECKPOINT: c.inventory(folder)})
            c.verify_checkpoint(report, out, variant)
            (folder/'training_state.pt').write_bytes(b'opt')
            report['checkpoint_files'][c.CHECKPOINT] = c.inventory(folder)
            with self.assertRaises(ValueError):
                c.verify_checkpoint(report, out, variant)
            (folder/'training_state.pt').unlink()
            (out/'checkpoints'/'step_005000').mkdir()
            with self.assertRaises(ValueError):
                c.verify_checkpoint(report, out, variant)
            (out/'checkpoints'/'step_005000').rmdir()
            (folder/'policy_postprocessor.json').unlink()
            with self.assertRaises(ValueError):
                c.verify_checkpoint(report, out, variant)


class BindingTests(unittest.TestCase):
    def test_the_binding_records_the_sources_the_start_and_the_recipe(self):
        c = contract()
        if not (mounted() and (c.PARENT/'seal.json').exists() and bundle_mounted() and start_mounted()):
            self.skipTest('data, sidecar, bundle or start checkpoint not mounted')
        binding = c.current_binding()
        self.assertEqual(binding['dataset_fingerprint'], c.SEAL)
        self.assertEqual(binding['start']['model_sha256'], c.START_MODEL_SHA256)
        self.assertEqual(binding['recipe_sha256'], c.digest_json(c.recipe()))
        self.assertEqual(binding['sources'], c.source_hashes())
        self.assertEqual(binding['sidecar']['constants_sha256'], c.forces.CONSTANTS_SHA256)
        self.assertEqual(binding['box_table_sha256'], c.BOX_TABLE_SHA256)
        self.assertEqual(c.current_binding(), binding)  # repeatable


if __name__ == '__main__':
    unittest.main()
