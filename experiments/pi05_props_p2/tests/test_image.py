"""The pi0.5 images differ from their parents by transformers and tokenizers and nothing else.

Runs inside `admission-pi` (MuJoCo 3.13.0) or `robosuite-pi` (MuJoCo 3.4.0); either is told apart by the
installed mujoco version, which selects the parent freeze to compare with.
"""
import importlib.metadata as metadata
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FREEZES = ROOT/'docker/pi05'
ADMISSION = '08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc'
ROBOSUITE = '54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e'
COMMIT = 'dcddb970176382c0fcf4521b0c0e6fc15894dfe0'
ZIP_SHA256 = '87a4dc40543efab2d52e5579574169e561ecc29b1fe8d3edadf962df6f062155'
CHANGED = (
    'tokenizers==0.21.4',
    f'transformers @ https://github.com/huggingface/transformers/archive/{COMMIT}.zip#sha256={ZIP_SHA256}',
)
PARENT_BY_MUJOCO = {'3.13.0': ADMISSION, '3.4.0': ROBOSUITE}
# robosuite 1.5.2 is installed --no-deps in the robosuite image; its four unmet requirements are intended.
EXPECTED_PIP_CHECK = {
    ADMISSION: ['No broken requirements found.'],
    ROBOSUITE: [f'robosuite 1.5.2 requires {name}, which is not installed.'
                for name in ('mink', 'opencv-python', 'pytest', 'qpsolvers')],
}
PARAMETERS = 3_616_757_520
NAMES = ('transformers', 'tokenizers')


def parent():
    return PARENT_BY_MUJOCO[metadata.version('mujoco')]


def run(*args):
    return subprocess.run([sys.executable, '-m', *args], capture_output=True, text=True, check=False)


def unchanged(lines):
    return [line for line in lines if line.partition(' ')[0].partition('==')[0].lower() not in NAMES]


class ImageTest(unittest.TestCase):
    def test_siglip_check_is_installed_correctly(self):
        from transformers.models.siglip import check
        self.assertTrue(check.check_whether_transformers_replace_is_installed_correctly())

    def test_pi05_constructs_on_cpu_with_the_p2_features(self):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        config = PI05Config(
            device='cpu', n_action_steps=10,
            input_features={'observation.state': PolicyFeature(FeatureType.STATE, (9,)),
                            'observation.images.image': PolicyFeature(FeatureType.VISUAL, (3, 512, 512)),
                            'observation.images.image2': PolicyFeature(FeatureType.VISUAL, (3, 512, 512))},
            output_features={'action': PolicyFeature(FeatureType.ACTION, (8,))})
        policy = PI05Policy(config)
        self.assertEqual(sum(p.numel() for p in policy.parameters()), PARAMETERS)

    def test_pip_check_is_what_the_parent_gives(self):
        result = run('pip', 'check')
        self.assertEqual(result.stdout.strip().splitlines(), EXPECTED_PIP_CHECK[parent()])

    def test_freeze_differs_from_the_parent_in_the_two_pinned_lines_only(self):
        frozen = sorted(run('pip', 'freeze').stdout.splitlines())
        expected = sorted((FREEZES/f'parent_freeze_{parent()}.txt').read_text().splitlines())
        self.assertEqual(unchanged(frozen), unchanged(expected))
        self.assertEqual([line for line in frozen if line not in unchanged(frozen)], sorted(CHANGED, key=str.lower))
        self.assertEqual(len(expected)-len(unchanged(expected)), 2)

    def test_the_parent_freezes_are_the_parents(self):
        for digest in (ADMISSION, ROBOSUITE):
            text = (FREEZES/f'parent_freeze_{digest}.txt').read_text()
            self.assertIn('transformers==4.57.6', text)
            self.assertIn('tokenizers==0.22.2', text)
            self.assertIn('lerobot==0.4.4', text)


if __name__ == '__main__':
    unittest.main()
