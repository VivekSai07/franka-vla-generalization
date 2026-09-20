"""Single-use collector. --preflight never executes an expert or physics."""
import argparse
import copy
import json
import os
from pathlib import Path
import resource
from shutil import disk_usage
import signal
import sys
from time import monotonic, process_time

import numpy as np

from experiments.fixed_layout.record import write_json, write_npz, file_hash, record_attempt
from experiments.fixed_layout.media import BoundPairedDataset
from experiments.fixed_layout.source import Expert, load_reset, storage_path
from experiments.paired_demonstrations.media import FEATURES, ENCODING
from . import plan as p
from .scenes import make_bank, bank_manifest, provenance, preflight, exact

OUTPUT = p.OUTPUT


class Budget:
    def __init__(self, root):
        self.root, self.started = Path(root), monotonic()

    def check(self):
        if monotonic()-self.started >= p.MAX_SECONDS: raise TimeoutError('Local-varied six-hour deadline')
        if disk_usage(storage_path(self.root)).free < p.FLOOR_BYTES: raise ValueError('30 GiB floor required')

    def check_start(self):
        self.check()
        if disk_usage(storage_path(self.root)).free < p.START_BYTES: raise ValueError('40 GiB start reserve required')

    def __enter__(self):
        self.check()
        if signal.getitimer(signal.ITIMER_REAL) != (0., 0.): raise RuntimeError('Existing deadline cannot be replaced')
        self.old_handler = signal.getsignal(signal.SIGALRM)
        def expire(signum, frame): raise TimeoutError('Local-varied six-hour deadline')
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, max(.001, p.MAX_SECONDS-(monotonic()-self.started)))
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old_handler)


def claim_output(root):
    """Exclusive claim of an absent directory OR an empty destination bind mount."""
    root = Path(root)
    if any(x.is_symlink() for x in (root, *root.parents)): raise ValueError('Output symlinks forbidden')
    if root.exists():
        if not root.is_dir() or any(root.iterdir()): raise FileExistsError('No retry/resume/nonempty output')
    else:
        root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'claim.json', dict(version=p.VERSION, output=str(p.OUTPUT.relative_to(p.ROOT)), resume=False))
    if {x.name for x in root.iterdir()} != {'claim.json'}:
        raise FileExistsError('Concurrent writer during exclusive claim')


def frozen_names(plan):
    return {'plan.json', 'provenance.json', 'configuration.json', 'source.npz', 'bank.json'} | {
        f"scenes/{x['layout_id']}.npz" for x in plan['layouts']}


def check_snapshots(root, hashes, check=lambda: None):
    from experiments.fixed_layout.validate import _json
    root = Path(root); plan = _json(root / 'plan.json'); p.validate_plan(plan)
    if set(hashes) != frozen_names(plan): raise ValueError('Frozen snapshot inventory mismatch')
    for name, digest in hashes.items():
        path = root / name
        if any(x.is_symlink() for x in (path, *path.parents)) or file_hash(path, check) != digest:
            raise ValueError('Frozen source/plan/77-state bank changed')
    if _json(root / 'provenance.json') != provenance(check): raise ValueError('Source/runtime changed')


def run_schedule(plan, bank, execute, validate, journal, before=lambda: None):
    p.validate_plan(plan)
    frozen = copy.deepcopy(plan); states = copy.deepcopy(bank)
    if set(bank) != {x['layout_id'] for x in plan['layouts']}: raise ValueError('All 77 layouts required before experts')
    for arrays in bank.values():
        a = arrays['initial_state']
        if a.dtype != np.float64 or a.ndim != 1 or not np.isfinite(a).all(): raise ValueError('Invalid full reset')
    def unchanged():
        p.validate_plan(plan)
        if set(bank) != set(states) or any(set(bank[k]) != set(states[k]) or any(
                not exact(v, states[k][key]) for key, v in bank[k].items()) for k in states):
            raise ValueError('Frozen bank mutated, including reserved evaluation layouts')
    admitted = []
    for slot in frozen['episodes']:
        i = slot['episode_index']
        try:
            unchanged(); before()
            journal(dict(event='started', **slot, training_ready=False))
            outcome = execute(copy.deepcopy(slot), states[slot['layout_id']]['initial_state'].copy())
            if outcome.get('success') is not True or type(outcome.get('frames')) is not int or outcome['frames'] != 528:
                raise RuntimeError('First failed/wrong-duration expert; no retry')
            validate(copy.deepcopy(slot), outcome)
            unchanged(); before()
            journal(dict(event='admitted', episode_index=i, training_ready=False)); admitted.append(i)
            if i == 11: journal(dict(event='preparation_pass', episode_ids=list(range(12)), training_ready=False))
        except BaseException as exc:
            try: journal(dict(event='failed', episode_index=i, training_ready=False, error=repr(exc)))
            except BaseException as secondary: print(f'Journal failure: {secondary!r}', file=sys.stderr)
            raise
    return admitted


def create_dataset(output, check):
    dataset = BoundPairedDataset.create(repo_id='local_varied_data', root=Path(output)/'dataset', fps=30,
        features=copy.deepcopy(FEATURES), vcodec=ENCODING['vcodec'], encoder_threads=ENCODING['encoder_threads'],
        metadata_buffer_size=1)
    dataset.meta.info['data_files_size_in_mb'] = 0
    dataset.meta.info['video_files_size_in_mb'] = 0
    dataset.fixed_layout_check = check  # Existing explicit encoder checkpoint hook, not a module patch.
    return dataset


def collect():
    """Main-only real collection after independent review; no arbitrary overrides."""
    from .validate import validate_episode, validate_dataset
    from experiments.fixed_layout.validate import fingerprint
    output = OUTPUT; budget = Budget(output)
    budget.check_start()
    if any(x.is_symlink() for x in (output, *output.parents)): raise ValueError('Output symlinks forbidden')
    if output.exists() and (not output.is_dir() or any(output.iterdir())): raise FileExistsError('No retry/resume')
    with budget:
        # All scene validity checks precede both the claim and dataset construction.
        plan = p.make_plan(); bound = provenance(budget.check)
        expert = Expert(); source = load_reset(expert)
        bank = make_bank(expert.model, source, plan, budget.check)
        if provenance(budget.check) != bound: raise ValueError('Source changed during scene freeze')
        budget.check_start(); claim_output(output)
        dataset = None; active = None; started = {}; accepted = []
        try:
            budget.check_start()  # Check the claimed filesystem, not repository free space.
            write_json(output / 'plan.json', plan); write_json(output / 'provenance.json', bound)
            write_json(output / 'configuration.json', expert.config)
            write_json(output / 'bank.json', bank_manifest(plan, bank))
            with (output / 'source.npz').open('xb') as stream:
                stream.write((p.ROOT / p.SOURCE).read_bytes()); stream.flush(); os.fsync(stream.fileno())
            (output / 'scenes').mkdir()
            for name, arrays in bank.items(): budget.check(); write_npz(output / 'scenes' / f'{name}.npz', arrays)
            hashes = {name: file_hash(output / name, budget.check) for name in sorted(frozen_names(plan))}
            write_json(output / 'snapshots.json', hashes)
            (output / 'attempts').mkdir()

            def before():
                budget.check(); check_snapshots(output, hashes, budget.check)
                fresh = load_reset(expert)
                if expert.config != json.loads((output / 'configuration.json').read_text()) or any(
                        not exact(source[k], fresh[k]) for k in source): raise ValueError('Live expert/source reset changed')
                budget.check()

            before()
            dataset = create_dataset(output, budget.check)

            def journal(event):
                nonlocal active
                if event['event'] == 'started': active = event['episode_index']
                if event['event'] == 'admitted': accepted.append(event['episode_index'])
                with (output / 'journal.jsonl').open('a') as stream:
                    stream.write(json.dumps({**event, 'elapsed_seconds': monotonic()-budget.started}, allow_nan=False)+'\n')
                    stream.flush(); os.fsync(stream.fileno())

            def execute(slot, initial):
                i = slot['episode_index']; path = output / 'attempts' / f'{i:06d}'
                path.mkdir(exist_ok=False); write_json(path / 'slot.json', slot)
                started[i] = (monotonic(), process_time())
                result = record_attempt(path, expert, dataset, slot, initial, budget.check)
                dataset._writer_closed_for_reading = True
                return result

            def validate(slot, outcome):
                before(); validate_episode(output, slot, expert, bank[slot['layout_id']], budget.check)
                wall, cpu = started[slot['episode_index']]
                usage = resource.getrusage(resource.RUSAGE_SELF)
                write_json(output / 'attempts' / f"{slot['episode_index']:06d}" / 'resources.json', dict(
                    wall_seconds=monotonic()-wall, process_cpu_seconds=process_time()-cpu,
                    process_maxrss_kib=usage.ru_maxrss, process_user_seconds=usage.ru_utime,
                    process_system_seconds=usage.ru_stime, logical_cpus=os.cpu_count(),
                    affinity_cpus=sorted(os.sched_getaffinity(0)), free_bytes=disk_usage(output).free,
                    scope='CPU process cumulative high-water; no GPU attribution'))
                before()

            run_schedule(plan, bank, execute, validate, journal, before)
            dataset.finalize(); dataset = None; before()
            info_path = output / 'dataset/meta/info.json'; info = json.loads(info_path.read_text())
            info['splits'] = {'train': '0:48', 'validation': '48:60'}
            temporary = info_path.with_suffix('.local-varied.json'); write_json(temporary, info); temporary.replace(info_path)
            report = validate_dataset(output, expert, source, bank, budget.check)
            before(); write_json(output / 'validation.json', report)
            write_json(output / 'seal.json', {'fingerprint': fingerprint(output, budget.check)})
            budget.check()
            return report
        except BaseException as exc:
            try:
                attempts = sorted(int(x.name) for x in (output / 'attempts').iterdir()) if (output / 'attempts').exists() else []
                wall, cpu = started.get(active, (monotonic(), process_time()))
                write_json(output / 'failure.json', dict(training_ready=False, error=repr(exc), active_episode=active,
                    admitted_ids=accepted, attempted_ids=attempts, missing_ids=[i for i in range(60) if i not in attempts],
                    elapsed_seconds=monotonic()-budget.started, active_wall_seconds=monotonic()-wall,
                    active_process_cpu_seconds=process_time()-cpu, process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            except BaseException as secondary: print(f'Failure evidence unavailable: {secondary!r}', file=sys.stderr)
            raise
        finally:
            if dataset is not None:
                try: dataset.finalize()
                except BaseException as secondary: print(f'Cleanup failed; primary preserved: {secondary!r}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--preflight', action='store_true'); choice.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    result = preflight() if args.preflight else collect()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())