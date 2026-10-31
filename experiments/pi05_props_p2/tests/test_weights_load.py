"""The pinned weights load into `PI05Policy` completely: no missing key, no unexpected key, every tensor equal.

`PI05Policy.from_pretrained` prints missing keys and carries on (and returns a randomly initialised model if anything in
the load raises), so success is asserted on the key sets and the tensors, never on the absence of an exception.
Needs ~30 GB of RAM and the assets mounted at outputs/research/pi05_assets; skipped without them.
"""
import unittest

import torch

from experiments.pi05_props_p2 import assets

REAL = (assets.WEIGHTS_DIR/'model.safetensors').is_file()


def p2_config():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    return PI05Config(
        device='cpu', n_action_steps=10,
        input_features={'observation.state': PolicyFeature(FeatureType.STATE, (9,)),
                        'observation.images.image': PolicyFeature(FeatureType.VISUAL, (3, 512, 512)),
                        'observation.images.image2': PolicyFeature(FeatureType.VISUAL, (3, 512, 512))},
        output_features={'action': PolicyFeature(FeatureType.ACTION, (8,))})


@unittest.skipUnless(REAL, 'pi05 weights are not mounted at outputs/research/pi05_assets')
class WeightsLoadTest(unittest.TestCase):
    def test_every_checkpoint_tensor_lands_in_the_policy_and_nothing_is_left_over(self):
        from safetensors import safe_open
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        policy = PI05Policy.from_pretrained(str(assets.WEIGHTS_DIR), config=p2_config(), local_files_only=True)
        state = policy.state_dict()
        with safe_open(str(assets.WEIGHTS_DIR/'model.safetensors'), framework='pt', device='cpu') as stored:
            keys = list(stored.keys())
            mapped, skipped = {}, []
            for key in keys:
                fixed = policy._fix_pytorch_state_dict_keys({key: None}, policy.config)
                if not fixed:
                    skipped.append(key)
                    continue
                (new,) = fixed
                mapped[key] = new if new.startswith('model.') else f'model.{new}'
            self.assertEqual(skipped, [])
            unexpected = sorted(set(mapped.values()) - set(state))
            loaded = {state[name].data_ptr() for name in mapped.values()}
            # The checkpoint stores the tied embedding once, under lm_head; the model also lists it as embed_tokens.
            # strict=True therefore always reports it missing and LeRobot only prints a warning.
            missing = sorted(name for name in set(state) - set(mapped.values()) if state[name].data_ptr() not in loaded)
            self.assertEqual(unexpected, [])
            self.assertEqual(missing, [])
            for key in keys:
                self.assertTrue(torch.equal(stored.get_tensor(key), state[mapped[key]]), key)


if __name__ == '__main__':
    unittest.main()
