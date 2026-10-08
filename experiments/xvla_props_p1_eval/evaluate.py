"""Roll a props P1 checkpoint on the reserved layouts: --mode smoke|screen|report|confirm --checkpoint N.

Robosuite image (MuJoCo 3.4.0) with a GPU. Refuses to start unless the expert feasibility pass
exists, the training run's checkpoints verify, and the sealed parent still reproduces its
fingerprint. Writes only under `outputs/xvla_props_p1_eval/<mode>_step_<N>/`; the dataset and the
training output are read-only inputs. After every trial `result.json` is rewritten with the trials
so far and every read-out.

`--resume` continues a `running` or `failed` run: re-checks the checkpoint, parent and trial list
are the ones recorded, keeps every finished trial, moves any half-written trial folder (and
`failure.json`) under `quarantine/`, and runs only the rest. Each trial reseeds itself, so a
resumed trial equals an uninterrupted one. It is never automatic.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from experiments.xvla_props_p1_eval import analysis, protocol

MUJOCO_VERSION = '3.4.0'
FEASIBILITY = protocol.EVAL_ROOT/'feasibility'/'feasibility.json'


def select_trials(mode):
    if mode == 'screen':
        return protocol.screen_trials()
    if mode == 'report':
        return protocol.report_trials()
    if mode == 'confirm':
        return protocol.confirm_trials()
    if mode == 'smoke':
        held_out = next(t for t in protocol.report_trials() if t['kind'] == 'held_out')
        return [protocol.screen_trials()[0], held_out]
    raise ValueError(f'Unapproved evaluation mode; expected one of {protocol.MODES}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=list(protocol.MODES), required=True)
    parser.add_argument('--checkpoint', type=int, choices=protocol.CANDIDATES, required=True)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_feasibility():
    if not FEASIBILITY.is_file():
        raise RuntimeError(f'Run the feasibility pass first: {FEASIBILITY} is missing')
    summary = json.loads(FEASIBILITY.read_text())
    if summary['keys'] != 80:
        raise ValueError('Feasibility pass did not cover all 80 reserved layouts')
    return summary


def verify_checkpoint(candidate):
    """The training run is complete and this checkpoint is one of its verified, unchanged files."""
    from experiments.xvla_props_p1 import contract as training
    report = json.loads((training.OUTPUT/'result.json').read_text())
    if report.get('status') != 'complete':
        raise RuntimeError('Training has not completed')
    training.verify_checkpoints(report, training.OUTPUT, False)
    checkpoint = training.OUTPUT/'checkpoints'/training.checkpoint_directory(candidate)
    config = json.loads((checkpoint/'config.json').read_text())
    if (config['type'], config['chunk_size'], config['n_action_steps'], config['n_obs_steps'],
            config['dtype']) != ('xvla', 30, 10, 1, 'float32'):
        raise ValueError('Checkpoint config/schema changed')
    marker = json.loads((checkpoint/'checkpoint_result.json').read_text())
    return checkpoint, marker['model_sha256'], training.parent_fingerprint(), training.SEAL


def load_policy(checkpoint):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from experiments.xvla_trial import preflight
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = 'cpu'
    policy = preflight.load_weights(XVLAPolicy(config), checkpoint/'model.safetensors').to('cuda')
    policy.config.device = 'cuda'
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    if (policy.config.chunk_size, policy.config.n_action_steps) != (30, 10):
        raise ValueError('Expected chunk30/actionsteps10')
    return policy


TRIAL_FILES = ('cameras.mp4', 'trace.npz', 'episode.json')


def resume_state(directory, expected):
    """Finished rows to keep and folders to quarantine; raises unless this is the run `expected` describes."""
    path = Path(directory)/'result.json'
    if not path.is_file():
        raise FileNotFoundError(f'Nothing to resume: {path} is missing')
    prior = json.loads(path.read_text())
    if prior.get('status') not in ('running', 'failed'):
        raise ValueError(f"Cannot resume a run whose status is {prior.get('status')!r}")
    for key, value in expected.items():
        if prior.get(key) != json.loads(json.dumps(value)):
            raise ValueError(f'Cannot resume: {key} differs from the recorded run')
    done, ordinals = [], set()
    for row in prior.get('episodes', []):
        folder = Path(directory)/f"trial_{row['ordinal']:03d}_{row['layout_id']}"
        if not all((folder/name).is_file() for name in TRIAL_FILES):
            raise ValueError(f'Recorded trial {row["ordinal"]} has no complete folder')
        if json.loads((folder/'episode.json').read_text()) != row:
            raise ValueError(f'Recorded trial {row["ordinal"]} differs from its episode.json')
        done.append(row)
        ordinals.add(row['ordinal'])
    stale = [p for p in sorted(Path(directory).glob('trial_*')) if int(p.name.split('_')[1]) not in ordinals]
    stale += [p for p in (Path(directory)/'failure.json',) if p.exists()]
    return done, stale, prior


def quarantine(directory, paths):
    """Move, never delete."""
    target = Path(directory)/'quarantine'
    target.mkdir(exist_ok=True)
    for path in paths:
        n = 0
        while (target/f'{path.name}.{n}').exists():
            n += 1
        path.rename(target/f'{path.name}.{n}')


def run(mode, candidate, resume=False):
    import mujoco
    from experiments.xvla_props_p1_eval import rollout
    if mujoco.__version__ != MUJOCO_VERSION:
        raise RuntimeError(f'Run in the robosuite image (MuJoCo {MUJOCO_VERSION}), not {mujoco.__version__}')
    directory = protocol.output_dir(mode, candidate)
    feasibility = load_feasibility()
    excluded = set(feasibility['failed_layouts'])
    trials = select_trials(mode)
    checkpoint, model_sha256, fingerprint, seal = verify_checkpoint(candidate)
    if fingerprint != seal:
        raise ValueError('Props P1 parent fingerprint differs from the seal')
    report = dict(status='running', mode=mode, candidate=candidate, model_sha256=model_sha256,
                  parent_fingerprint=fingerprint, max_control_ticks=rollout.MAX_TICKS, strict_score_ticks=30,
                  excluded_layouts=sorted(excluded), planned_trials=len(trials), retries=False,
                  trials=trials, episodes=[])
    done = []
    if resume:
        recorded = {k: report[k] for k in ('mode', 'candidate', 'model_sha256', 'parent_fingerprint',
                                           'max_control_ticks', 'strict_score_ticks', 'excluded_layouts',
                                           'planned_trials', 'trials')}
        done, stale, prior = resume_state(directory, recorded)
        quarantine(directory, stale)
        report.update(resumes=prior.get('resumes', []) + [dict(finished_trials=len(done),
                                                                quarantined=[p.name for p in stale])])
    else:
        directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'result.json', report)
    try:
        _run_trials(directory, report, trials, excluded, checkpoint, done)
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc), retries=False)
        write_json(directory/'failure.json', report)
        write_json(directory/'result.json', report)
        raise
    report.update(status='complete')
    write_json(directory/'result.json', report)


def remaining_by_group(trials, done):
    """Unfinished trials grouped by (object, blue_slot): one renderer per group, no overlapping OSMesa contexts."""
    finished = {row['ordinal'] for row in done}
    groups = defaultdict(list)
    for trial in trials:
        if trial['ordinal'] not in finished:
            groups[(trial['object'], trial['blue_slot'])].append(trial)
    return groups


def _run_trials(directory, report, trials, excluded, checkpoint, done=()):
    import mujoco
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from experiments.props_scene.scene import build_model
    from experiments.xvla_props_p1_eval import rollout, scenes
    torch.use_deterministic_algorithms(True)
    policy = load_policy(checkpoint)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    groups = remaining_by_group(trials, done)
    rows = list(done)
    if rows:
        ordered = sorted(rows, key=lambda r: r['ordinal'])
        report.update(episodes=ordered, readouts=analysis.readouts(ordered, excluded))
    for (obj, blue_slot), members in groups.items():
        _, model = build_model((obj,), blue_slot)
        with mujoco.Renderer(model, height=512, width=512) as renderer:
            for trial in members:
                folder = directory/f"trial_{trial['ordinal']:03d}_{trial['layout_id']}"
                folder.mkdir()
                model, data = scenes.restore(trial['layout_id'])
                row = rollout.rollout(policy, pre, post, model, data, renderer, trial, folder)
                if row['outcome'] == 'nonfinite_action':
                    raise ValueError('Nonfinite-action failure; no retry')
                rows.append(row)
                ordered = sorted(rows, key=lambda r: r['ordinal'])
                report.update(episodes=ordered, readouts=analysis.readouts(ordered, excluded))
                write_json(directory/'result.json', report)
                print('EVAL', directory.name, json.dumps({k: row[k] for k in (
                    'ordinal', 'layout_id', 'kind', 'task_success', 'cause')}), flush=True)
    if len(rows) != len(trials):
        raise ValueError('Not every planned trial ran')


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    run(args.mode, args.checkpoint, args.resume)


if __name__ == '__main__':
    main()
