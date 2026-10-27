"""Roll props P3 checkpoints on the reserved P3 layouts, in parallel workers, then decide and analyse.

Robosuite image (MuJoCo 3.4.0) with a GPU for `worker`; `prepare`, `collect`, `decide` and `final` need no GPU.

    prepare   verify every checkpoint, the seal and the feasibility pass once; writes prepared.json
    worker    run one interleaved share of a stage's (checkpoint, trial) jobs; writes one folder per trial
    collect   merge a stage's trial folders into its result.json (complete only if every planned trial exists)
    decide    apply the pre-registered selection rules (screen survivors; report winner)
    final     the pre-registered bin-choice criteria on the winner's report tier, plus the confirm tier

Workers are independent processes (each ~4 GB GPU, ~5 CPU cores) writing disjoint trial folders under
`outputs/xvla_props_p3_eval/<mode>_step_<N>/`, so a stage can be sharded across processes and resumed: a worker keeps
every complete trial folder, moves a half-written one under `quarantine/`, and runs only the rest. Each trial reseeds
itself, so a resumed trial equals an uninterrupted one. The dataset and training outputs are read-only inputs. A trial
is never retried.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.xvla_props_p3_eval import analysis, protocol

MUJOCO_VERSION = '3.4.0'
FEASIBILITY = protocol.EVAL_ROOT/'feasibility'/'feasibility.json'
PREPARED = protocol.EVAL_ROOT/'prepared.json'
DECISIONS = protocol.EVAL_ROOT/'decisions'
HOT_C, COOL_C = 83, 75
TRIAL_FILES = ('cameras.mp4', 'trace.npz', 'episode.json')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def training_contract():
    from experiments.xvla_props_p3 import contract
    return contract


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def load_feasibility():
    if not FEASIBILITY.is_file():
        raise RuntimeError(f'Run the feasibility pass first: {FEASIBILITY} is missing')
    summary = json.loads(FEASIBILITY.read_text())
    if summary['keys'] != protocol.LAYOUTS:
        raise ValueError(f'Feasibility pass did not cover all {protocol.LAYOUTS} reserved layouts')
    return summary


# --- prepare -----------------------------------------------------------------------------------------------------------

def verify_checkpoints(candidates):
    """Training complete, checkpoint inventory byte-verified, parent reproduces its seal. Returns per-candidate facts."""
    training = training_contract()
    report = json.loads((training.OUTPUT/'result.json').read_text())
    if report.get('status') != 'complete':
        raise RuntimeError('P3 training has not completed')
    training.verify_checkpoints(report, training.OUTPUT, False)
    fingerprint = training.parent_fingerprint()
    if fingerprint != training.SEAL:
        raise ValueError('P3 parent fingerprint differs from its seal')
    facts = {}
    for candidate in candidates:
        checkpoint = training.OUTPUT/'checkpoints'/training.checkpoint_directory(candidate)
        config = json.loads((checkpoint/'config.json').read_text())
        if (config['type'], config['chunk_size'], config['n_action_steps'], config['n_obs_steps'],
                config['dtype']) != ('xvla', 30, 10, 1, 'float32'):
            raise ValueError('Checkpoint config/schema changed')
        marker = json.loads((checkpoint/'checkpoint_result.json').read_text())
        model = checkpoint/'model.safetensors'
        if sha256_file(model) != marker['model_sha256']:
            raise ValueError(f'{checkpoint}: model hash differs from its marker')
        stat = model.stat()
        facts[str(candidate)] = dict(checkpoint=str(checkpoint), model_sha256=marker['model_sha256'],
                                     model_bytes=stat.st_size, model_mtime_ns=stat.st_mtime_ns)
    return dict(seal=training.SEAL, parent_fingerprint=fingerprint, candidates=facts)


def prepare():
    feasibility = load_feasibility()
    record = dict(prepared_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  excluded_layouts=sorted(feasibility['failed_layouts']),
                  feasibility_succeeded=feasibility['succeeded'],
                  failed_by_object=feasibility.get('failed_by_object'),
                  p3=verify_checkpoints(protocol.CANDIDATES))
    write_json(PREPARED, record)
    print('PREPARED', json.dumps({k: record[k] for k in ('excluded_layouts', 'feasibility_succeeded')}), flush=True)


def load_prepared():
    if not PREPARED.is_file():
        raise RuntimeError(f'Run prepare first: {PREPARED} is missing')
    return json.loads(PREPARED.read_text())


def check_prepared(prepared, candidate):
    """Cheap per-worker re-check against the once-verified record: marker hash, size and mtime of the weights."""
    facts = prepared['p3']['candidates'][str(candidate)]
    checkpoint = Path(facts['checkpoint'])
    marker = json.loads((checkpoint/'checkpoint_result.json').read_text())
    stat = (checkpoint/'model.safetensors').stat()
    if (marker['model_sha256'] != facts['model_sha256'] or stat.st_size != facts['model_bytes']
            or stat.st_mtime_ns != facts['model_mtime_ns']):
        raise ValueError(f'{checkpoint} changed since prepare; run prepare again')
    return checkpoint, facts['model_sha256']


# --- worker ------------------------------------------------------------------------------------------------------------

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


def gpu_state():
    """Temperature (C), utilisation (%) and memory (MiB) of GPU 0, or None where nvidia-smi is unavailable."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used',
                              '--format=csv,noheader,nounits', '-i', '0'], capture_output=True, text=True, timeout=20)
        t, u, m = (float(x) for x in out.stdout.strip().split(','))
        return dict(temperature_c=t, utilization_pct=u, memory_mib=m)
    except Exception:  # telemetry is diagnostic; its absence must not fail a trial
        return None


def cool_down(sleep=time.sleep):
    """Pause while the GPU is at or above HOT_C until it falls to COOL_C. Returns seconds paused."""
    paused, state = 0.0, gpu_state()
    if state is None or state['temperature_c'] < HOT_C:
        return paused
    while state is not None and state['temperature_c'] > COOL_C:
        sleep(15)
        paused += 15
        state = gpu_state()
    return paused


def trial_is_complete(folder):
    return all((folder/name).is_file() for name in TRIAL_FILES)


def quarantine(directory, path):
    target = Path(directory)/'quarantine'
    target.mkdir(exist_ok=True)
    n = 0
    while (target/f'{path.name}.{n}').exists():
        n += 1
    path.rename(target/f'{path.name}.{n}')


def run_job(policy, pre, post, job, directory, renderer_size=512):
    import mujoco
    from experiments.xvla_props_p3_eval import rollout, scenes
    trial = job['trial']
    folder = protocol.trial_folder(directory, trial)
    if folder.exists():
        if trial_is_complete(folder):
            return None
        quarantine(directory, folder)
    folder.mkdir()
    paused = cool_down()
    before = gpu_state()
    model, data, names = scenes.restore(trial['layout_id'])
    with mujoco.Renderer(model, height=renderer_size, width=renderer_size) as renderer:
        row = rollout.rollout(policy, pre, post, model, data, names, renderer, trial, folder)
    write_json(folder/'telemetry.json', dict(before=before, after=gpu_state(), cooled_seconds=paused))
    if row['outcome'] == 'nonfinite_action':
        raise ValueError('Nonfinite-action failure; no retry')
    return row


def worker(mode, candidates, shard, shards):
    import mujoco
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    if mujoco.__version__ != MUJOCO_VERSION:
        raise RuntimeError(f'Run in the robosuite image (MuJoCo {MUJOCO_VERSION}), not {mujoco.__version__}')
    torch.use_deterministic_algorithms(True)
    prepared = load_prepared()
    mine = protocol.shard_jobs(protocol.jobs(mode, candidates), shard, shards)
    current, policy, pre, post, directory = None, None, None, None, None
    try:
        for job in mine:
            if job['candidate'] != current:
                policy = pre = post = None
                torch.cuda.empty_cache()
                current = job['candidate']
                checkpoint, _ = check_prepared(prepared, current)
                policy = load_policy(checkpoint)
                pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
                directory = protocol.output_dir(mode, current)
                directory.mkdir(parents=True, exist_ok=True)
            row = run_job(policy, pre, post, job, directory)
            if row is not None:
                print('EVAL', directory.name, json.dumps({k: row[k] for k in (
                    'ordinal', 'layout_id', 'object', 'task_success', 'placed_bin', 'cause')}),
                    f'{row["wall_seconds"]:.0f}s', flush=True)
    except BaseException as exc:
        if directory is not None:
            write_json(directory/f'failure_shard{shard}.json',
                       dict(mode=mode, candidate=current, shard=shard, shards=shards,
                            error_type=type(exc).__name__, error=str(exc)))
        raise
    print('WORKER_DONE', mode, list(candidates), shard, shards, flush=True)


# --- collect -----------------------------------------------------------------------------------------------------------

def collect(mode, candidate):
    prepared = load_prepared()
    directory = protocol.output_dir(mode, candidate)
    trials = protocol.select_trials(mode)
    facts = prepared['p3']['candidates'][str(candidate)]
    excluded = set(prepared['excluded_layouts'])
    rows, missing = [], []
    for trial in trials:
        folder = protocol.trial_folder(directory, trial)
        if not folder.is_dir() or not trial_is_complete(folder):
            missing.append(trial['ordinal'])
            continue
        row = json.loads((folder/'episode.json').read_text())
        planned = json.loads(json.dumps(trial))
        if any(row.get(k) != v for k, v in planned.items()):
            raise ValueError(f'Trial {trial["ordinal"]} does not match the planned trial')
        if row['outcome'] == 'nonfinite_action':
            raise ValueError(f'Trial {trial["ordinal"]} had a nonfinite action')
        rows.append(row)
    failures = sorted(p.name for p in directory.glob('failure_shard*.json'))
    report = dict(status='complete' if not missing else 'incomplete', mode=mode, candidate=candidate,
                  model_sha256=facts['model_sha256'], parent_fingerprint=prepared['p3']['parent_fingerprint'],
                  excluded_layouts=sorted(excluded), planned_trials=len(trials), missing_ordinals=missing,
                  failure_files=failures, retries=False, trials=trials, episodes=rows,
                  readouts=analysis.readouts(rows, excluded) if rows else None)
    write_json(directory/'result.json', report)
    print('COLLECT', directory.name, report['status'], len(rows), 'of', len(trials),
          'missing', missing, 'failures', failures, flush=True)
    return report


def load_stage(mode, candidate):
    path = protocol.output_dir(mode, candidate)/'result.json'
    if not path.is_file():
        raise RuntimeError(f'{path} is missing; run collect')
    report = json.loads(path.read_text())
    if report['status'] != 'complete':
        raise RuntimeError(f'{path.parent.name} is not complete (missing {report["missing_ordinals"]})')
    return report


# --- decide / final ----------------------------------------------------------------------------------------------------

def decide(stage):
    prepared = load_prepared()
    excluded = prepared['excluded_layouts']
    DECISIONS.mkdir(exist_ok=True)
    if stage == 'screen':
        counts = {c: analysis.selection_count(load_stage('screen', c)['episodes'], excluded)
                  for c in protocol.CANDIDATES}
        survivors = analysis.screen_survivors(counts, protocol.SCREEN_TIE_TRIALS)
        decision = dict(stage='screen', counts={str(k): v for k, v in counts.items()}, survivors=survivors,
                        rule='every candidate within 2 successes of the best screen count')
    elif stage == 'report':
        survivors = json.loads((DECISIONS/'screen.json').read_text())['survivors']
        reports = {c: load_stage('report', c)['episodes'] for c in survivors}
        counts = {c: analysis.selection_count(rows, excluded) for c, rows in reports.items()}
        decision = dict(stage='report', counts={str(k): v for k, v in counts.items()},
                        # Not used for selection: held-out layouts never decide which checkpoint is reported.
                        all_layout_successes={str(c): sum(1 for r in rows if r['layout_id'] not in set(excluded)
                                                          and r['task_success']) for c, rows in reports.items()},
                        winner=analysis.report_winner(counts),
                        rule='highest report success count on trained-object layouts; ties go to the later checkpoint')
    else:
        raise ValueError('stage must be screen or report')
    write_json(DECISIONS/f'{stage}.json', decision)
    print('DECISION', json.dumps(decision), flush=True)
    return decision


def final():
    prepared = load_prepared()
    excluded = prepared['excluded_layouts']
    decision = json.loads((DECISIONS/'report.json').read_text())
    winner = decision['winner']
    report, confirm = load_stage('report', winner), load_stage('confirm', winner)
    out = dict(winner=winner, excluded_layouts=excluded,
               criteria=analysis.criteria(report['episodes'], excluded),
               readouts=dict(report=analysis.readouts(report['episodes'], excluded),
                             confirm=analysis.readouts(confirm['episodes'], excluded)),
               # Descriptive: every survivor read the same way. Only the winner's verdict is the pre-registered one.
               survivors={str(c): analysis.criteria(load_stage('report', c)['episodes'], excluded)
                          for c in json.loads((DECISIONS/'screen.json').read_text())['survivors']},
               caveat='The winner was selected on the trained-object report layouts, so the trained-object criterion '
                      'is optimistic; the confirm tier (selected on nothing) is the unbiased estimate. Held-out '
                      'layouts did not enter selection. A held-out result below the floor is not interpretable.')
    write_json(protocol.EVAL_ROOT/'final.json', out)
    print('FINAL', json.dumps(out['criteria'], indent=1), flush=True)
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    w = sub.add_parser('worker')
    c = sub.add_parser('collect')
    for p in (w, c):
        p.add_argument('--mode', choices=list(protocol.MODES), required=True)
        p.add_argument('--candidates', required=True, help='comma-separated checkpoint updates')
    w.add_argument('--shard', type=int, required=True)
    w.add_argument('--shards', type=int, required=True)
    d = sub.add_parser('decide')
    d.add_argument('stage', choices=('screen', 'report'))
    sub.add_parser('final')
    return parser.parse_args(argv)


def _ints(text):
    return tuple(int(x) for x in text.split(','))


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if args.command == 'prepare':
        prepare()
    elif args.command == 'worker':
        worker(args.mode, _ints(args.candidates), args.shard, args.shards)
    elif args.command == 'collect':
        for candidate in _ints(args.candidates):
            collect(args.mode, candidate)
    elif args.command == 'decide':
        decide(args.stage)
    else:
        final()


if __name__ == '__main__':
    sys.exit(main())
