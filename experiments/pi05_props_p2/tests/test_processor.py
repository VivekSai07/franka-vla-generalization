"""The locally built pi0.5 pre-processor: prompt, state tokens, tails and truncation.

Needs the tokenizer files at outputs/research/pi05_assets (skipped without them) and the sealed P2 root for the
real-frame test. The reference prompt is computed here from the stored formulas, not by calling the installed step.
"""
import glob
import re
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.pi05_props_p2 import assets, contract as c

HAVE_TOKENIZER = (assets.TOKENIZER_DIR/'tokenizer.json').is_file()
HAVE_PARENT = (c.DATA/'meta/info.json').is_file()
SUFFIX = ';\nAction: '


def stats_with(lo, hi):
    def block(n, lo, hi):
        return {'mean': torch.zeros(n), 'std': torch.ones(n), 'min': torch.full((n,), lo*2),
                'max': torch.full((n,), hi*2), 'q01': torch.full((n,), lo), 'q99': torch.full((n,), hi)}
    return {'observation.state': block(9, lo, hi), 'action': block(8, lo, hi)}


def batch(state, task='put the lemon in the blue bin'):
    return {'observation.state': torch.tensor(state, dtype=torch.float32),
            'observation.images.image': torch.zeros(3, 512, 512),
            'observation.images.image2': torch.zeros(3, 512, 512), 'task': task}


def numbers(prompt):
    match = re.fullmatch(r'Task: (.*), State: ([-0-9 ]*);\nAction: ', prompt, flags=re.S)
    assert match, prompt
    return match.group(1), [int(x) for x in match.group(2).split()]


@unittest.skipUnless(HAVE_TOKENIZER, 'pi05 tokenizer files are not mounted')
class ProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(str(assets.TOKENIZER_DIR))
        cls.config = c.make_config('cpu')
        cls.pre, cls.post = c.make_processors(cls.config, stats_with(-1., 1.))

    def run_pre(self, state, task='put the lemon in the blue bin', pre=None):
        out = (pre or self.pre)(batch(state, task))
        tokens, mask = out['observation.language.tokens'][0], out['observation.language.attention_mask'][0]
        return out, numbers(out['task'][0]), self.tokenizer.decode(tokens[mask], skip_special_tokens=True), int(mask.sum())

    def test_quantile_boundaries_and_tails_use_the_native_tokens_and_do_not_clip(self):
        state = [-1.0, 1.0, -1.01, -3.0, 1.01, 5.0, 0.0, 0.5, -0.5]
        _, (task, tokens), _, _ = self.run_pre(state)
        self.assertEqual(task, 'put the lemon in the blue bin')
        # q01 -> 0, q99 -> 255; below q01 the native discretiser gives -1, above q99 it saturates at 255.
        self.assertEqual(tokens[:9], [0, 255, -1, -1, 255, 255, 128, 192, 64])

    def test_the_normalised_state_itself_is_not_clipped(self):
        out, _, _, _ = self.run_pre([-3.0, 5.0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(out['observation.state'][0, :2].tolist(), [-3.0, 5.0])

    def test_state_has_32_tokens_and_the_padding_is_zero(self):
        _, (_, tokens), _, _ = self.run_pre([0.25]*9)
        self.assertEqual(len(tokens), 32)
        self.assertEqual(tokens[:9], [160]*9)
        self.assertEqual(tokens[9:], [128]*23)

    def test_a_constant_dimension_stays_finite(self):
        stats = stats_with(-1., 1.)
        stats['observation.state']['q99'][3] = stats['observation.state']['q01'][3]
        pre, _ = c.make_processors(self.config, stats)
        _, (_, tokens), _, _ = self.run_pre([0, 0, 0, -1.0, 0, 0, 0, 0, 0], pre=pre)
        self.assertEqual(tokens[3], 0)
        _, (_, tokens), _, _ = self.run_pre([0, 0, 0, 0.0, 0, 0, 0, 0, 0], pre=pre)
        self.assertIn(tokens[3], range(-1, 256))

    def test_every_p2_instruction_survives_untruncated_even_with_the_longest_state(self):
        for text in c.INSTRUCTIONS:
            for state in ([-3.0]*9, [5.0]*9, [0.0]*9):
                with self.subTest(text=text, state=state[0]):
                    out, (task, tokens), decoded, length = self.run_pre(state, text)
                    self.assertEqual(task, text)
                    self.assertEqual(len(tokens), 32)
                    self.assertTrue(decoded.startswith(f'Task: {text}, State: '))
                    self.assertTrue(decoded.endswith(SUFFIX))
                    self.assertLess(length, self.config.tokenizer_max_length)
                    self.assertEqual(out['observation.language.tokens'].shape, (1, 200))

    def test_the_prompt_is_built_locally_and_the_hub_preprocessor_is_never_read(self):
        source = Path(c.__file__).read_text()
        for banned in ('policy_preprocessor', 'policy_postprocessor', 'from_pretrained', 'hf_hub_download'):
            self.assertNotIn(banned, source)
        self.assertEqual(sorted(type(s).__name__ for s in self.pre.steps),
                         sorted(['RenameObservationsProcessorStep', 'AddBatchDimensionProcessorStep',
                                 'NormalizerProcessorStep', 'Pi05PrepareStateTokenizerProcessorStep',
                                 'TokenizerProcessorStep', 'DeviceProcessorStep']))

    def test_the_factory_hook_is_restored_and_refuses_another_tokenizer(self):
        import lerobot.policies.pi05.processor_pi05 as module
        before = module.TokenizerProcessorStep
        with c.local_tokenizer():
            with self.assertRaises(ValueError):
                module.TokenizerProcessorStep(tokenizer_name='google/some-other-tokenizer')
        self.assertIs(module.TokenizerProcessorStep, before)
        with self.assertRaises(RuntimeError):
            with c.local_tokenizer():
                raise RuntimeError('boom')
        self.assertIs(module.TokenizerProcessorStep, before)

    def test_postprocessor_inverts_the_action_normalisation(self):
        action = torch.tensor([[-.5, 0., .25, .5, .75, 1.0, -1.0, 0.3]])
        stats = stats_with(-2., 2.)
        _, post = c.make_processors(self.config, stats)
        normalised = 2.0*(action-stats['action']['q01'])/(stats['action']['q99']-stats['action']['q01'])-1.0
        back = post(normalised)
        torch.testing.assert_close(back, action, atol=1e-6, rtol=0)


@unittest.skipUnless(HAVE_TOKENIZER and HAVE_PARENT, 'tokenizer files or the sealed P2 root are not mounted')
class RealFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        files = sorted(glob.glob(str(c.DATA/'data/**/*.parquet'), recursive=True))
        cls.frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        tasks = pd.read_parquet(c.DATA/'meta/tasks.parquet')
        cls.text_of = {int(v): str(k) for k, v in zip(tasks.index, tasks['task_index'])}
        train = cls.frames.iloc[:c.TRAINING_ROWS]
        state = torch.tensor(np.stack(train['observation.state'].to_numpy()), dtype=torch.float32)
        action = torch.tensor(np.stack(train['action'].to_numpy()), dtype=torch.float32)
        cls.stats = {}
        for key, x in (('observation.state', state), ('action', action)):
            q = torch.quantile(x, torch.tensor([.01, .99]), dim=0)
            cls.stats[key] = {'mean': x.mean(0), 'std': x.std(0, unbiased=False), 'min': x.amin(0),
                              'max': x.amax(0), 'q01': q[0], 'q99': q[1]}
        cls.config = c.make_config('cpu')
        cls.pre, _ = c.make_processors(cls.config, cls.stats)

    def reference_prompt(self, state, text):
        s = self.stats['observation.state']
        norm = 2.0*(torch.tensor(state, dtype=torch.float32)-s['q01'])/(s['q99']-s['q01'])-1.0
        padded = np.concatenate([norm.numpy(), np.zeros(23, dtype=np.float32)])
        tokens = np.digitize(padded, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        return f'Task: {text}, State: {" ".join(str(int(t)) for t in tokens)};\nAction: '

    def test_prompts_of_real_training_and_validation_frames_match_the_reference(self):
        picks = [0, 1, 200, 451, 452*100+17, 452*479+300, 452*500+10]
        for i in picks:
            row = self.frames.iloc[i]
            state = np.asarray(row['observation.state'], dtype=np.float32)
            text = self.text_of[int(row['task_index'])]
            out = self.pre(batch(state.tolist(), text))
            self.assertEqual(out['task'][0], self.reference_prompt(state, text), i)

    def test_out_of_range_state_values_are_rare_in_the_training_rows(self):
        sample = self.frames.iloc[:c.TRAINING_ROWS:97]
        outside = total = 0
        for state in np.stack(sample['observation.state'].to_numpy())[:300]:
            s = self.stats['observation.state']
            norm = 2.0*(torch.tensor(state, dtype=torch.float32)-s['q01'])/(s['q99']-s['q01'])-1.0
            outside += int(((norm < -1) | (norm > 1)).sum())
            total += 9
        self.assertLess(outside/total, 0.05)


if __name__ == '__main__':
    unittest.main()
