"""Final5000 observation-only probes and twenty seeds on ONE immutable full reset."""
from contextlib import ExitStack, nullcontext
import json
import random
import time

import av
import mujoco
import numpy as np
import torch
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

from data_collection.benchmark import (apply_action, capture_state, restore_timing,
                                      step_control, synchronize_observation, write_json)
from validation.evaluate_pilot import restore_scene, advance_streak
from validation.gates import fresh_stack_metrics
from validation.rollout_diagnostics import bound_action, target_diagnostics
from src.controllers.grasp_controller import GraspController
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF, SCENE_PATH
from experiments.fixed_layout.contract import SOURCE_SCENE_SHA256
from . import contract as c


def evaluation_plan():
    return [dict(ordinal=i+1,seed=20260918+i,reset_file='reset.npz',source_episode_index=7)
            for i in range(20)]


def seed_trial(seed):
    if type(seed) is not int or seed not in range(20260918,20260938):
        raise ValueError('Only the twenty approved independent rollout seeds')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def summarize(rows):
    seeds=[r.get('seed') for r in rows]
    def finite(value):
        if isinstance(value,dict): return all(finite(v) for v in value.values())
        if isinstance(value,(list,tuple)): return all(finite(v) for v in value)
        return bool(np.isfinite(value)) if isinstance(value,(float,np.floating)) else True
    complete=(finite(rows) and seeds == [r['seed'] for r in evaluation_plan()]
              and all(type(r.get('task_success')) is bool
                      and r.get('outcome') == ('success' if r['task_success'] else 'timeout') for r in rows))
    successes=sum(r.get('task_success') is True for r in rows)
    return dict(planned_trials=20,finished_trials=len(rows),successes=successes,
                success_rate=successes/len(rows) if rows else None,
                all_trials_valid=complete,capability_passed=complete and successes>=18,
                required_successes=18,scope='sampling_reliability_one_selected_geometry_not_generalization')


def require_complete_trials(rows):
    if not summarize(rows)['all_trials_valid']:
        raise ValueError('All twenty distinct approved trials must finish without nonfinites')


def reset_entry(binding):
    if c.sha256(c.PARENT/'reset.npz') != binding['reset_sha256']:
        raise ValueError('Fixed full reset changed')
    return dict(episode_index=7,source_episode_index=7,pick='red',place='green',colors=['red','green'],
                file='reset.npz',sha256=binding['reset_sha256'],scene_hash=SOURCE_SCENE_SHA256)


def stage_metrics(trace):
    """Descriptive stages from persisted commands/contact; never feeds the policy."""
    actions=np.asarray(trace['applied_action']).reshape(-1,8)
    contact=np.asarray(trace['bilateral_target_contact'],dtype=bool)
    lift=np.asarray(trace['pick_lift_m'],dtype=float)
    released=np.asarray(trace['released_stack'],dtype=bool)
    if (any(len(x)!=len(actions) for x in (contact,lift,released))
            or not np.isfinite(actions).all() or not np.isfinite(lift).all()):
        raise ValueError('Invalid saved action/contact stages')
    closed=actions[:,7]<=127
    def first(mask):
        ticks=np.flatnonzero(mask)
        return int(ticks[0]) if ticks.size else None
    def sustained(mask,count):
        streak=0
        for i,value in enumerate(mask):
            streak=streak+1 if value else 0
            if streak>=count: return i
        return None
    closure=first(closed)
    after=np.arange(len(actions))>closure if closure is not None else np.zeros(len(actions),dtype=bool)
    return dict(first_closure_tick=closure,sustained_closure_30_tick=sustained(closed,30),
                first_bilateral_contact_tick=first(contact),sustained_bilateral_5_tick=sustained(contact,5),
                first_contact_lift_gt_5cm_tick=first(contact & (lift>.05)),
                first_release_after_closure_tick=first(after & ~closed),
                stable_stack_30_tick=sustained(released,30),
                maximum_lift_m=float(lift.max()) if len(lift) else 0.,
                bilateral_contact_ticks=int(contact.sum()),tick_origin=0)


def offline_probes(policy,pre,post,ds,device='cuda',output=None):
    """All20train/4val x60/107/347, real targets retained outside prediction inputs."""
    python_rng,numpy_rng=random.getstate(),np.random.get_state()
    training=policy.training
    devices=[torch.cuda.current_device()] if device=='cuda' else []
    rows,predictions,targets=[],[],[]
    try:
        policy.eval()
        with torch.random.fork_rng(devices=devices):
            for episode in c.TRAIN+c.VAL:
                for frame in (60,107,347):
                    index=episode*528+frame
                    sample=c.policy_row(ds[index],training=False)
                    obs=c.observation({key:sample[key] for key in (*c.OBS,'task')})
                    target=sample['action'].detach().cpu().clone()
                    batch=c.p.process_row(pre,obs)
                    # Saved LeRobot converters may emit action=None; never pass even that key.
                    if batch.get('action') is not None:
                        raise ValueError('Inference processor introduced a target action')
                    batch.pop('action',None)
                    random.seed(c.SEED+index)
                    np.random.seed(c.SEED+index)
                    torch.manual_seed(c.SEED+index)
                    autocast=torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else nullcontext()
                    with torch.inference_mode(),autocast:
                        prediction=post(policy.predict_action_chunk(batch)).detach().cpu()
                    if prediction.shape != (1,30,8) or not torch.isfinite(prediction).all():
                        raise ValueError('Nonfinite or wrong-shaped physical probe prediction')
                    error=(prediction[0]-target).abs()
                    rows.append(dict(episode=episode,frame=frame,row=index,
                                     split='train' if episode in c.TRAIN else 'validation',
                                     arm_mae_rad=float(error[:,:7].mean()),
                                     first10_arm_mae_rad=float(error[:10,:7].mean()),
                                     gripper_binary_accuracy=float(((prediction[0,:,7]>127)==(target[:,7]>127)).float().mean())))
                    predictions.append(prediction[0].numpy()); targets.append(target.numpy())
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        policy.train(training)
    result=dict(checkpoint_update=5000,observation_only=True,validation_used_for_training=False,
                validation_is_spatial_holdout=False,rows=rows)
    for split in ('train','validation'):
        selected=[r for r in rows if r['split']==split]
        result[split]={'probes':len(selected),**{key:float(np.mean([r[key] for r in selected]))
                       for key in ('arm_mae_rad','first10_arm_mae_rad','gripper_binary_accuracy')}}
    if output is not None:
        with (output/'offline_probe_arrays.npz').open('xb') as stream:
            np.savez_compressed(stream,predictions=np.asarray(predictions),targets=np.asarray(targets),
                                rows=np.asarray([r['row'] for r in rows]))
        write_json(output/'offline_probes.json',result)
    return result


class CountedPolicy(XVLAPolicy):
    """Native select_action queue; count real calls without runtime method replacement."""
    chunk_calls=0

    def _get_action_chunk(self,*args,**kwargs):
        self.chunk_calls+=1
        return super()._get_action_chunk(*args,**kwargs)


def run_trial(policy, pre, post, model, manifest, entry, bank, output, ordinal, seed):
    if (policy.config.chunk_size, policy.config.n_action_steps) != (30, 10):
        raise ValueError("Expected chunk 30/action steps 10")
    data, scratch = restore_scene(model, bank, entry), mujoco.MjData(model)
    policy.reset()
    seed_trial(seed)
    grasp = GraspController(model, data, target=f"target_cube_{entry['pick']}")
    initial_z, initial_time = float(data.xpos[grasp.target_id, 2]), float(data.time)
    initial_state = capture_state(model, data)
    instruction = f"pick up the {entry['pick']} cube and place it on top of the {entry['place']} cube"
    trace = {k: [] for k in ("state", "sim_time", "raw_action", "applied_action", "post_state", "post_sim_time")}
    stable = 0
    before_calls = policy.chunk_calls
    success, outcome = False, "timeout"
    started = time.monotonic()
    with ExitStack() as resources:
        renderer = mujoco.Renderer(model, height=512, width=512)
        resources.callback(renderer.close)
        record = resources.enter_context(av.open(str(output / "cameras.mp4"), "w"))
        stream = record.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 1024, 512, "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        def images_and_video():
            images = []
            for camera in manifest["camera_names"]:
                renderer.update_scene(data, camera=camera)
                images.append(renderer.render().copy())
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.concatenate(images, axis=1), format="rgb24")):
                record.mux(packet)
            return images
        try:
            for _ in range(1200):
                synchronize_observation(model, data)
                state = data.qpos[:9].copy()
                frame = {"observation.state": torch.tensor(state, dtype=torch.float32), "task": instruction}
                for key, image in zip(("observation.images.image", "observation.images.image2"), images_and_video()):
                    frame[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
                processed = pre(c.observation(frame))
                if _ == 0:
                    write_json(output/'input_precheck.json',c.check_model_inputs(policy,processed,frame['observation.state'][None]))
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    action = post(policy.select_action(processed)).squeeze(0).cpu().numpy()
                if action.shape != (8,):
                    raise ValueError("Policy action shape is not eight physical commands")
                for key, value in (("state", state), ("sim_time", float(data.time)), ("raw_action", action.copy())):
                    trace[key].append(value)
                if not np.isfinite(action).all():
                    outcome = "nonfinite_action"
                    break
                bounded, clips = bound_action(action, model.actuator_ctrlrange[list(ACTIVE_JOINTS)])
                apply_action(model, data, bounded, ACTIVE_JOINTS)
                if step_control(model, data, manifest["control_dt_requested"], ARM_DOF) != 20:
                    raise ValueError("Physics substep count changed")
                metrics = fresh_stack_metrics(model, data, scratch, entry["pick"], entry["place"])
                diagnostics = target_diagnostics(model, data, entry["pick"], entry["place"], grasp, initial_z)
                for key, value in {"applied_action": bounded.copy(), "post_state": data.qpos[:9].copy(),
                        "post_sim_time": float(data.time), **clips, **metrics, **diagnostics}.items():
                    trace.setdefault(key, []).append(value)
                if not np.isfinite(capture_state(model, data)).all():
                    raise ValueError("Nonfinite simulator state")
                stable, success = advance_streak(stable, metrics)
                if success:
                    outcome = "success"
                    break
            images_and_video()
        finally:
            for packet in stream.encode():
                record.mux(packet)
    calls = policy.chunk_calls - before_calls
    attempts, ticks = len(trace["raw_action"]), len(trace["applied_action"])
    if calls != (attempts + 9)//10:
        raise ValueError("Actual chunk predictions differ from ceil(action_attempts/10)")
    if not np.isclose(data.time - initial_time, ticks/30, rtol=0, atol=1e-8):
        raise ValueError("Simulator duration differs from completed control ticks/30")
    masks = np.asarray(trace.get("joint_clip_mask", []), dtype=bool).reshape(-1, 7)
    grip = np.asarray(trace.get("gripper_clipped", []), dtype=bool)
    interventions = int(np.sum(np.any(masks, axis=1) | grip))
    terminal = {**fresh_stack_metrics(model, data, scratch, entry["pick"], entry["place"]),
                **target_diagnostics(model, data, entry["pick"], entry["place"], grasp, initial_z)}
    arrays = {k: np.asarray(v) for k, v in trace.items()}
    arrays["applied_action"] = arrays["applied_action"].reshape(-1, 8)
    np.savez_compressed(output / "trace.npz", **arrays, initial_integration_state=initial_state,
                        terminal_integration_state=capture_state(model, data), terminal_state=data.qpos[:9].copy(),
                        terminal_qpos=data.qpos.copy(), terminal_qvel=data.qvel.copy(), terminal_sim_time=float(data.time),
                        **{f"terminal_{k}": np.asarray(v) for k, v in terminal.items()})
    stages = stage_metrics(arrays) if ticks else {}
    result = {**entry, "ordinal": ordinal, "seed": seed, "rollout_seed": seed, "instruction": instruction,
              "outcome": outcome, "ticks": attempts, "action_attempts": attempts,
              "completed_control_ticks": ticks, "sim_seconds": ticks/30,
              "actual_chunk_predictions": calls, "chunk_size": 30, "n_action_steps": 10,
              "task_success": success, "unassisted_success": success and interventions == 0,
              "intervention_steps": interventions, "joint_clipped_ticks": int(np.any(masks, axis=1).sum()),
              "joint_clipped_ticks_per_dimension": masks.sum(axis=0).tolist(), "gripper_clipped_ticks": int(grip.sum()),
              "gripper_thresholded_ticks": int(np.sum(trace.get("gripper_thresholded", []))),
              "max_joint_clip_rad": max(trace.get("max_joint_clip_rad", [0.])),
              "gripper_adapter": "threshold_at_127_not_counted_as_clipping", "success_streak": stable,
              "stages": stages, "reset_sha256": entry['sha256'],
              "final_metrics": {k: np.asarray(v).tolist() for k, v in terminal.items()},
              "wall_seconds": time.monotonic() - started}
    write_json(output / "episode.json", result)
    return result


def verify_final(output,binding):
    report=json.loads((output/'result.json').read_text())
    checkpoint=output/'checkpoints/005000'
    c.require_fields(report,{**c.recipe(False),'status':'complete','completed_updates':5000,
                     'selected_checkpoint_update':5000,'final_data_binding_passed':True,
                     'strict_reload_fp32_exact':True,'reload_max_absolute_error':0.,'binding':binding,
                     'training_rows':10560,'complete_training_chunks':9980,
                     'scratch_used_for_initialization':False},'Complete final-only5000 gate failed')
    if (report['cpu_admission_sha256'] != c.sha256(c.CPU_ADMISSION)
            or report['final_checkpoint_files'] != c.inventory(checkpoint)
            or report['training_stats_sha256'] != c.sha256(output/'training_stats.json')
            or report['scratch_result_sha256'] != c.sha256(c.SCRATCH/'result.json')
            or [m['update'] for m in report['checkpoints']] != c.checkpoint_updates(False)):
        raise ValueError('Final checkpoint/statistics/CPU/scratch binding changed')
    config=json.loads((checkpoint/'config.json').read_text())
    if (config != json.loads((output/'initial_config/config.json').read_text())
            or (config['type'],config['chunk_size'],config['n_action_steps'],config['dtype']) != ('xvla',30,10,'float32')
            or {k:tuple(v['shape']) for k,v in config['input_features'].items()} != c.OBS
            or {k:v['shape'] for k,v in config['output_features'].items()} != {'action':[8]}):
        raise ValueError('Final XVLA real schema mismatch')
    return checkpoint


def training_bytes(output):
    return {str(path.relative_to(output)):c.sha256(path) for path in sorted(output.rglob('*'))
            if path.is_file() and path.relative_to(output).parts[0]!='evaluation'}


def run(output):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from .runner import failure,require_disk
    directory=output/'evaluation'
    c.claim_directory(directory)
    report=dict(status='preparing',stage='evaluation_admission',episodes=[],max_control_ticks=1200,
                max_sim_seconds=40,strict_score_ticks=30,final_only=True,
                retries=False,resume=False,**summarize([]))
    before=training_bytes(output)
    try:
        ds,starts,stats,binding=c.prepare_data()
        checkpoint=verify_final(output,binding)
        manifest=json.loads((c.PARENT/'configuration.json').read_text())
        model=mujoco.MjModel.from_xml_path(SCENE_PATH)
        if restore_timing(model,manifest)['substeps_per_control'] != 20:
            raise ValueError('Expected unchanged physics timing')
        # The actual original episode identity never changes to implement a noise seed.
        entry=reset_entry(binding)
        restore_scene(model,c.PARENT,entry)  # Full integration/qpos/hash; never resettle.
        config=PreTrainedConfig.from_pretrained(checkpoint,local_files_only=True)
        config.device='cpu'
        policy=c.p.load_weights(CountedPolicy(config),checkpoint/'model.safetensors').to('cuda')
        policy.config.device='cuda'
        policy.eval()
        for parameter in policy.parameters(): parameter.requires_grad_(False)
        pre,post=make_pre_post_processors(policy.config,pretrained_path=str(checkpoint))
        report.update(status='running',stage='offline_probes',binding=binding,training_files=before,
                      trials=evaluation_plan(),privileged_policy_inputs=False)
        write_json(directory/'result.json',report)
        report['offline_probes']=offline_probes(policy,pre,post,ds,output=directory)
        # Phase rebinding before actual rollouts, not another full semantic audit.
        if c.admission()!=binding: raise ValueError('Probe phase binding changed')
        verify_final(output,binding)
        report['stage']='rollouts'
        for trial in evaluation_plan():
            require_disk(output,2)
            folder=directory/f"trial_{trial['ordinal']:02d}_seed_{trial['seed']}"
            folder.mkdir(exist_ok=False)
            report['active_seed']=trial['seed']
            write_json(directory/'result.json',report)
            row=run_trial(policy,pre,post,model,manifest,entry,c.PARENT,folder,trial['ordinal'],trial['seed'])
            if row['seed']!=trial['seed'] or row['outcome']=='nonfinite_action':
                raise ValueError('Seed mismatch/nonfinite action; no retries')
            report['episodes'].append(row)
            report.update(summarize(report['episodes']))
            write_json(directory/'result.json',report)
            print('EVAL',json.dumps({key:report[key] for key in ('finished_trials','successes','capability_passed')}),flush=True)
        require_complete_trials(report['episodes'])
        after=c.admission()
        verify_final(output,after)
        if after!=binding: raise ValueError('Final evaluation binding changed')
        report.update(status='complete',stage='complete',active_seed=None,training_bytes_unchanged=True,
                      final_data_binding_passed=True,no_extension=True)
    except BaseException as exc:
        failure(directory,report,exc)
        raise
    finally:
        if training_bytes(output)!=before:
            exc=ValueError('Training bytes changed during evaluation')
            failure(directory,report,exc)
            raise exc
    write_json(directory/'result.json',report)