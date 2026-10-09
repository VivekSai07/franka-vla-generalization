"""Two-bin props scene: one cached model per prop set and bin arrangement, and a deterministic reset.

robosuite composes the scene once; the compiled XML is rewritten by `xml_contract.apply_props_contract`
and stepped with plain MuJoCo. `env.step` is banned. Runs in the robosuite image (MuJoCo 3.4.0).
"""
import functools

import mujoco
import numpy as np
import robosuite
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.objects import BottleObject, BreadObject, CanObject, CerealObject, LemonObject, MilkObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_element

from experiments.props_scene import constants as c
from experiments.props_scene.arena import BinsTableArena
from experiments.props_scene.xml_contract import apply_props_contract
from experiments.robosuite_scene.cameras import GRIPPER_CAM

GRIPPER_BODY_NAME = "robot0_right_hand"
PROP_CLASSES = {"can": CanObject, "bread": BreadObject, "lemon": LemonObject,
                "milk": MilkObject, "bottle": BottleObject, "cereal": CerealObject}
SETTLE_SECONDS = 0.2
DROP_HEIGHT_M = 0.001


def _check(objects, blue_slot):
    objects = tuple(objects)
    if (not objects or len(set(objects)) != len(objects) or not set(objects) <= set(c.OBJECTS)):
        raise ValueError(f"objects must be distinct names from {c.OBJECTS}, got {objects}")
    if blue_slot not in c.SLOT_XY:
        raise ValueError(f"blue_slot must be 'left' or 'right', got {blue_slot!r}")
    return objects


class PropsEnv(ManipulationEnv):
    def __init__(self, objects=("can",), blue_slot="left", **kwargs):
        self.objects = _check(objects, blue_slot)
        self.blue_slot = blue_slot
        super().__init__(
            robots="Panda", base_types="RethinkMinimalMount", controller_configs=None,
            initialization_noise={"magnitude": 0.0, "type": "uniform"},
            use_camera_obs=False, has_renderer=False, has_offscreen_renderer=True,
            control_freq=c.FPS, horizon=10000, ignore_done=True, hard_reset=False, renderer="mujoco",
            camera_names=["front_cam", "gripper_cam"], camera_heights=c.CAMERA_SIZE, camera_widths=c.CAMERA_SIZE,
        )

    def _check_robot_configuration(self, robots):
        return

    def _load_model(self):
        super()._load_model()
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(robot_model.base.bottom_offset - robot_model.base.top_offset)
        arena = BinsTableArena(self.blue_slot)
        arena.set_origin([0, 0, 0])
        props = [PROP_CLASSES[name](name=name) for name in self.objects]
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=props)
        for body in self.model.worldbody.iter("body"):
            if body.get("name") == GRIPPER_BODY_NAME:
                body.append(new_element("camera", **GRIPPER_CAM))
                break


@functools.lru_cache(maxsize=None)
def _built(objects, blue_slot):
    env = robosuite.make(
        "PropsEnv", objects=objects, blue_slot=blue_slot, robots="Panda", base_types="RethinkMinimalMount",
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False, control_freq=c.FPS,
        hard_reset=False, renderer="mujoco", initialization_noise={"magnitude": 0.0, "type": "uniform"})
    env.reset()
    model = mujoco.MjModel.from_xml_string(apply_props_contract(env.sim.model.get_xml()))
    from data_collection.benchmark import configure_timing
    configure_timing(model, c.FPS)

    def _banned(*args, **kwargs):
        raise RuntimeError("env.step is forbidden; drive data.ctrl and mujoco.mj_step")

    env.step = _banned
    return env, model  # the env must stay alive or robosuite frees its sim


def build_model(objects, blue_slot):
    """(env, compiled model) holding exactly these props; cached per sorted objects and arrangement."""
    return _built(tuple(sorted(_check(objects, blue_slot), key=c.OBJECTS.index)), blue_slot)


def present_objects(model):
    return [o for o in c.OBJECTS if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"prop_{o}") >= 0]


def _qpos_address(model, name):
    return int(model.joint(int(model.body(f"prop_{name}").jntadr[0])).qposadr[0])


def reset(model, poses, home_jitter_deg):
    """Arm at Q_HOME plus jitter, props upright at (x, y, yaw_deg) on the table, settled 0.2 s; no RNG."""
    from data_collection.benchmark import capture_state, synchronize_observation
    from experiments.props_scene.geometry import bbox
    from src.robots.franka_panda.config import Q_HOME

    if sorted(poses) != sorted(present_objects(model)):
        raise ValueError(f"poses for {sorted(poses)} do not match the model's props {sorted(present_objects(model))}")
    if len(home_jitter_deg) != 7:
        raise ValueError("home_jitter_deg must have 7 values")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for name, (x, y, yaw_deg) in poses.items():
        centre, half = bbox(model, name)
        yaw = np.deg2rad(yaw_deg)
        adr = _qpos_address(model, name)
        data.qpos[adr:adr + 3] = (x, y, half[2] - centre[2] + DROP_HEIGHT_M)  # body-frame bottom is centre - half
        data.qpos[adr + 3:adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    home = np.asarray(Q_HOME[:7], dtype=np.float64) + np.deg2rad(np.asarray(home_jitter_deg, dtype=np.float64))
    data.qpos[:9] = Q_HOME
    data.qpos[:7] = home
    data.ctrl[:] = 0.0
    data.ctrl[:7] = home
    data.ctrl[-1] = 255
    mujoco.mj_forward(model, data)
    for _ in range(round(SETTLE_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)
    synchronize_observation(model, data)
    return {"qpos": data.qpos.copy(), "initial_state": capture_state(model, data)}


def object_pose(model, data, name):
    body = model.body(f"prop_{name}").id
    rotation = data.xmat[body].reshape(3, 3)
    return {"xy": data.xpos[body][:2].copy(), "z": float(data.xpos[body][2]),
            "yaw_deg": float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
            "quat": data.xquat[body].copy()}
