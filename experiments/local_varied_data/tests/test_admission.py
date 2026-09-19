"""Real 77-state envelope: reject coherently rehashed wrong resets, no dynamics."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from experiments.fixed_layout.source import Expert, load_reset
from experiments.fixed_layout.record import write_json, write_npz, file_hash
from experiments.fixed_layout.validate import fingerprint
from experiments.local_varied_data import plan as p, record as r, scenes as s, validate as v


class AdmissionTests(unittest.TestCase):
    def test_real_bank_binding_rejects_wrong_reset_even_with_recomputed_snapshots(self):
        root = Path(tempfile.mkdtemp(prefix='admission-', dir='/vla/outputs/implementation/local_varied_data_20260919'))
        r.claim_output(root)
        with ExitStack() as guards:
            for name in ('mj_step', 'mj_step1', 'mj_step2', 'Renderer'):
                guards.enter_context(patch.object(mujoco, name, side_effect=AssertionError('No native execution')))
            guards.enter_context(patch.object(Expert, 'run', side_effect=AssertionError('No real expert')))
            expert = Expert(); source = load_reset(expert); plan = p.make_plan(); bank = s.make_bank(expert.model, source, plan)
            write_json(root / 'plan.json', plan); write_json(root / 'configuration.json', expert.config)
            write_json(root / 'provenance.json', s.provenance()); write_json(root / 'bank.json', s.bank_manifest(plan, bank))
            (root / 'source.npz').write_bytes((p.ROOT / p.SOURCE).read_bytes()); (root / 'scenes').mkdir()
            for name, arrays in bank.items(): write_npz(root / 'scenes' / f'{name}.npz', arrays)
            hashes = {name: file_hash(root / name) for name in r.frozen_names(plan)}
            write_json(root / 'snapshots.json', hashes)
            self.assertEqual(v.bindings(root, expert, source, bank, lambda: None), plan)
            for layout_id in ('train_000', 'evaluation_000'):
                changed = copy.deepcopy(bank); changed[layout_id] = copy.deepcopy(bank['center'])
                path = root / 'scenes' / f'{layout_id}.npz'
                with path.open('wb') as stream: np.savez_compressed(stream, **changed[layout_id])
                (root / 'bank.json').write_text(json.dumps(s.bank_manifest(plan, changed)))
                (root / 'snapshots.json').write_text(json.dumps({name: file_hash(root / name) for name in r.frozen_names(plan)}))
                with self.subTest(layout=layout_id), self.assertRaisesRegex(ValueError, 'scene manifest|reset'):
                    v.bindings(root, expert, source, changed, lambda: None)
                with path.open('wb') as stream: np.savez_compressed(stream, **bank[layout_id])
                (root / 'bank.json').write_text(json.dumps(s.bank_manifest(plan, bank)))
                (root / 'snapshots.json').write_text(json.dumps({name: file_hash(root / name) for name in r.frozen_names(plan)}))
            self.assertEqual(v.bindings(root, expert, source, bank, lambda: None), plan)
            # Even a self-consistent report+seal cannot admit a 24-attempt root.
            (root / 'attempts').mkdir()
            (root / 'dataset').mkdir()
            for i in range(24): (root / 'attempts' / f'{i:06d}').mkdir()
            events = []
            for slot in plan['episodes']:
                events.extend([dict(event='started', **slot, training_ready=False, elapsed_seconds=0.),
                               dict(event='admitted', episode_index=slot['episode_index'], training_ready=False, elapsed_seconds=0.)])
                if slot['episode_index'] == 11: events.append(dict(event='preparation_pass', episode_ids=list(range(12)), training_ready=False, elapsed_seconds=0.))
            (root / 'journal.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            write_json(root / 'validation.json', dict(status='passed', training_ready=True, duplicates={'exact_duplicates': []}, provenance=s.provenance()))
            write_json(root / 'seal.json', {'fingerprint': fingerprint(root)})
            with self.assertRaisesRegex(ValueError, 'sixty'): v.require_local_varied_pass(root)

    def test_final_inventory_rejects_missing_extra_and_bad_claim(self):
        root = Path(tempfile.mkdtemp())
        expected = {'claim.json', 'plan.json', 'provenance.json', 'configuration.json', 'source.npz',
                    'bank.json', 'snapshots.json', 'journal.jsonl', 'scenes', 'attempts', 'dataset'}
        for name in expected:
            if '.' in name: (root / name).write_text('{}')
            else: (root / name).mkdir()
        claim = dict(version=p.VERSION, output=str(p.OUTPUT.relative_to(p.ROOT)), resume=False)
        (root / 'claim.json').write_text(json.dumps(claim))
        try: v.validate_inventory(root)
        except AttributeError as exc: self.fail(f'Missing exact root inventory gate: {exc}')
        (root / 'unexpected.json').write_text('{}')
        with self.assertRaises(ValueError): v.validate_inventory(root)
        # Separate malformed-claim fixture; no deletion of the previous evidence.
        other = Path(tempfile.mkdtemp())
        for name in expected:
            if '.' in name: (other / name).write_text('{}')
            else: (other / name).mkdir()
        with self.assertRaises(ValueError): v.validate_inventory(other)


if __name__ == '__main__': unittest.main()