"""Scripted expert for the props scene: one object, either bin, from a random pose and yaw.

Not a fork of the cube expert, whose phase table assumes a 4 cm cube. It reuses the same parts: pinocchio
IK, bounded joint targets, `GraspController`, `step_control` at 30 Hz with 20 substeps, and the
sink/trace shape (frame and trace delivered after each tick's physics). Runs in the robosuite image.

Per object it plans from the compiled hull: the hand yaw that closes the fingers across the narrowest
horizontal side, a grasp point `min(0.03, height/2)` below the top, and a carry that clears the bin
walls. Targets for the carry and release are measured from the object in the hand, so slip or an
off-centre grasp does not move the drop point.
"""
import mujoco
import numpy as np

from experiments.paired_demonstrations.expert_v1 import (
    ACTIVE_JOINTS, ARM_DOF, CAMERAS, FrankaPanda, GraspController, IKController, Q_HOME,
    bounded_joint_target, capture_state, configuration, pin, pixels_digest, restore_state,
    step_control, synchronize_observation, validate_expert_commands,
)
from experiments.pose_snapshot_v2.planner import SnapshotTaskSpaceTrajectory
from experiments.props_scene import constants as c
from experiments.props_scene import geometry as geo
from experiments.props_scene.metrics import in_bin, placement, success_streak
from experiments.props_scene.scene import build_model, present_objects

EXPERT_VERSION = "props_p0_expert_v1"
ROUND_RATIO = 1.1    # widest over narrowest hull width: below this the footprint is round and any finger yaw does
FACE_TOLERANCE = 1.05  # two faces this close in width are equally good; the nearer one needs less hand rotation
HOLDING = ("close", "lift", "move", "lower")


def task_text(obj, bin_colour):
    return f"put the {obj} in the {bin_colour} bin"


def hand_yaw_for(finger_axis_yaw, narrow_yaw):
    """Rotation about world z, in [-pi/2, pi/2), that turns a finger axis at `finger_axis_yaw` onto `narrow_yaw`."""
    return float((narrow_yaw - finger_axis_yaw + np.pi / 2) % np.pi - np.pi / 2)


def _rotation_z(angle):
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


class PropsExpert:
    def __init__(self, objects, blue_slot):
        _, self.model = build_model(objects, blue_slot)
        self.objects, self.blue_slot = tuple(present_objects(self.model)), blue_slot
        self.robot = FrankaPanda()
        self.config = configuration(self.model, self.robot)
        self.config["expert_version"] = EXPERT_VERSION
        self.bounds = np.asarray(self.config["expert_command_contract"]["joint_bounds"])
        self.ik = IKController(self.robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=10., kp_rot=5.)
        self.finger_axis_yaw = self._home_finger_axis_yaw()

    def _home_finger_axis_yaw(self):
        """World yaw (mod pi) of the line from the left to the right finger with the arm at Q_HOME."""
        data = mujoco.MjData(self.model)
        data.qpos[:9] = Q_HOME
        mujoco.mj_forward(self.model, data)
        gap = data.xpos[self.model.body("right_finger").id] - data.xpos[self.model.body("left_finger").id]
        if abs(float(gap[2])) > 1e-3:
            raise ValueError("fingers are not level at home")
        return float(np.arctan2(gap[1], gap[0]) % np.pi)

    def plan_grasp(self, data, name):
        """Hand yaw and grasp point for the prop as it currently lies; pure geometry, no stepping."""
        model = self.model
        widths = geo.footprint_widths(model, data, name)
        round_ = max(widths) / min(widths) < ROUND_RATIO
        faces = list(zip((widths[0], widths[2]), geo.axis_yaws(model, data, name)))
        narrowest = min(width for width, _ in faces)
        candidates = [yaw for width, yaw in faces if width <= narrowest * FACE_TOLERANCE]
        hand_yaw = 0.0 if round_ else min((hand_yaw_for(self.finger_axis_yaw, yaw) for yaw in candidates), key=abs)
        finger_axis_yaw = (self.finger_axis_yaw + hand_yaw) % np.pi
        top, bottom = geo.top_z(model, data, name), geo.bottom_z(model, data, name)
        below_top = min(c.GRASP_BELOW_TOP_M, (top - bottom) / 2)
        centre = geo.centre_world(model, data, name)
        return {"hand_yaw": hand_yaw, "finger_axis_yaw": finger_axis_yaw,
                "round": bool(round_), "hand_point": np.array([centre[0], centre[1], top - below_top + c.HAND_OFFSET_M])}

    def run(self, initial_state, episode, sink, render=True):
        """Restore once, execute once. Sink receives copied pre-step frame/trace after each tick's physics."""
        model, robot = self.model, self.robot
        obj, bin_colour = episode["object"], episode["bin"]
        if obj not in self.objects:
            raise ValueError(f"{obj} is not in this model ({self.objects})")
        if bin_colour not in c.BINS:
            raise ValueError(f"bin must be one of {c.BINS}")
        if len(episode["home_pose_jitter_deg"]) != 7:
            raise ValueError("home_pose_jitter_deg must have 7 values")
        data = mujoco.MjData(model)
        restore_state(model, data, initial_state)
        rng = np.random.RandomState(episode["seed"])
        q_target = np.zeros(model.nq)
        q_target[:7] = data.qpos[:7]
        q_home_pin = pin.neutral(robot.model)
        q_home_pin[:len(Q_HOME)] = Q_HOME
        home_pose = robot.forward_kinematics(q_home_pin)
        home_rotation, home_pos = home_pose.rotation.copy(), home_pose.translation.copy()
        grasp = GraspController(model, data, target=f"prop_{obj}")
        plan = self.plan_grasp(data, obj)
        grasp_rotation = _rotation_z(plan["hand_yaw"]) @ home_rotation
        bin_xy = data.xpos[model.body(f"bin_{bin_colour}").id][:2].copy()
        task = task_text(obj, bin_colour)
        prop_bodies = [model.body(f"prop_{name}").id for name in self.objects]
        renderer = mujoco.Renderer(model, height=c.CAMERA_SIZE, width=c.CAMERA_SIZE) if render else None
        held = {}
        phase_index, start_time, trajectory = 0, data.time, None
        stable, frames, success, reached = 0, 0, False, False
        reason, metrics, grasp_lost = "not_in_bin", {}, None
        try:
            while phase_index < len(c.PHASE_NAMES):
                if frames >= c.MAX_TICKS:
                    reason = "timeout"
                    break
                name = c.PHASE_NAMES[phase_index]
                synchronize_observation(model, data)
                images, digests = {}, []
                if render:
                    for key, camera in CAMERAS.items():
                        renderer.update_scene(data, camera=camera)
                        images[key] = renderer.render().copy()
                        digests.append(pixels_digest(images[key]))
                q_current = data.qpos[:robot.model.nq].copy()
                hand = robot.forward_kinematics(q_current).translation
                trace = {"qpos": data.qpos.copy(), "sim_time": float(data.time), "phase_index": phase_index,
                         "props_pose": np.array([np.concatenate((data.xpos[b], data.xquat[b])) for b in prop_bodies])}
                if render:
                    trace["image_sha256"] = np.asarray(digests)
                if trajectory is None:
                    target_pos, rotation, jitter = self._target(
                        name, data, hand, obj, plan, grasp_rotation, home_pos, home_rotation, bin_xy, held)
                    if jitter:
                        target_pos = target_pos + rng.normal(0, c.TARGET_JITTER_M, 3)
                    trajectory = SnapshotTaskSpaceTrajectory(
                        start_pose=robot.forward_kinematics(q_current), end_pose=pin.SE3(rotation, target_pos),
                        duration=c.PHASE_DURATION_S[name], method="cubic")
                t_state = data.time - start_time
                dq, err, done = self.ik.compute_velocity(q_current, trajectory.get_pose(t_state), q_posture=q_home_pin)
                target, applied, saturated = bounded_joint_target(
                    q_target[ACTIVE_JOINTS], dq[ACTIVE_JOINTS], 1 / c.FPS, self.bounds)
                q_target[ACTIVE_JOINTS] = target
                gripper = 0 if name in HOLDING else 255
                action = np.zeros(8, dtype=np.float32)
                action[:7], action[7] = q_target[:7], float(gripper)
                data.ctrl[ACTIVE_JOINTS] = action[:7]  # the float32 value that is recorded, so a replay is exact
                grasp.command(gripper)
                validate_expert_commands(action[None], self.bounds)
                frame = {**images, "observation.state": q_current[:9].astype(np.float32), "action": action, "task": task}
                trace.update(ctrl=data.ctrl.copy(), target_pos=trajectory.end_pose.translation.copy(), phase_time=t_state)
                holding = grasp.is_grasped()
                if name in ("move", "lower") and not holding and grasp_lost is None:
                    grasp_lost = phase_index
                if t_state > c.PHASE_DURATION_S[name]:
                    if name == "lift" and not self._lifted(data, obj, holding, held):
                        reason = "grasp"
                        trace.update(substeps=0, success_tick=False)
                        sink(frame, trace)
                        frames += 1
                        break
                    phase_index += 1
                    start_time, trajectory = data.time, None
                substeps = step_control(model, data, 1 / c.FPS, ARM_DOF)
                metrics = in_bin(model, data, obj, bin_colour)
                stable, reached = success_streak(stable, metrics)
                trace.update(substeps=substeps, success_tick=metrics["success_tick"])
                sink(frame, trace)
                frames += 1
                if phase_index >= len(c.PHASE_NAMES):
                    success = reached
            if not success and metrics:
                if grasp_lost is not None:
                    reason = "lost_in_carry"
                elif metrics["inside"]:
                    reason = "not_released"
            return {"success": bool(success), "reason": "success" if success else reason, "frames": frames,
                    "stable_ticks": stable, "final_metrics": metrics,
                    "final_placement": placement(model, data, obj, bin_colour), "final_qpos": data.qpos.copy(),
                    "final_state": capture_state(model, data), "phase_reached": min(phase_index, len(c.PHASE_NAMES) - 1),
                    "grasp_lost_phase": grasp_lost, "model": model, "initial_state": np.asarray(initial_state).copy(),
                    "episode": dict(episode)}
        finally:
            if renderer is not None:
                renderer.close()

    def _measure(self, data, hand, obj):
        """Where the object sits relative to the hand now: height of the hand over its bottom, xy offset."""
        centre = geo.centre_world(self.model, data, obj)
        return {"d": float(hand[2] - geo.bottom_z(self.model, data, obj)), "off": centre[:2] - hand[:2],
                "bottom": geo.bottom_z(self.model, data, obj)}

    def _target(self, name, data, hand, obj, plan, grasp_rotation, home_pos, home_rotation, bin_xy, held):
        """(hand position, hand rotation, jitter) for a phase, from the state at the moment it starts."""
        wall = c.BIN_WALL_HEIGHT_M
        if name in ("approach", "descend", "close"):
            point = plan["hand_point"].copy()
            if name == "close":  # re-derive from the object as it lies, like the cube expert's grasp phase
                point = self.plan_grasp(data, obj)["hand_point"]
            if name == "approach":
                point[2] += c.APPROACH_HEIGHT_M
            return point, grasp_rotation, name != "close"
        if name == "lift":
            held.update(self._measure(data, hand, obj))
            held["bottom_at_lift"] = held["bottom"]
            clear = wall + c.LIFT_CLEARANCE_M
            held["lift_rise"] = clear - held["bottom"]
            return np.array([hand[0], hand[1], clear + held["d"]]), grasp_rotation, False
        if name in ("move", "lower", "open"):
            held.update(self._measure(data, hand, obj))
            level = wall + (c.LIFT_CLEARANCE_M if name == "move" else c.RELEASE_ABOVE_WALL_M)
            xy = bin_xy - held["off"]
            return np.array([xy[0], xy[1], level + held["d"]]), grasp_rotation, name != "open"
        if name == "retract":
            return np.array([hand[0], hand[1], hand[2] + c.RETRACT_HEIGHT_M]), grasp_rotation, False
        return home_pos.copy(), home_rotation, False

    def _lifted(self, data, obj, holding, held):
        rise = geo.bottom_z(self.model, data, obj) - held["bottom_at_lift"]
        return bool(holding and rise >= c.LIFT_VERIFY_FRACTION * held["lift_rise"])
