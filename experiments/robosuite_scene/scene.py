import mujoco
import robosuite
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_element

from experiments.robosuite_scene.arena import CopiedCameraTableArena
from experiments.robosuite_scene.cameras import FRONT_CAM, GRIPPER_CAM
from experiments.robosuite_scene.xml_contract import CUBE_RGBA, apply_scripted_contract

GRIPPER_BODY_NAME = "robot0_right_hand"


def _check_colours(colours):
    colours = tuple(colours)
    if len(colours) < 2 or len(set(colours)) != len(colours) or not set(colours) <= set(CUBE_RGBA):
        raise ValueError(f"colours must be 2-4 distinct names from {sorted(CUBE_RGBA)}, got {colours}")
    return colours


class BringupEnv(ManipulationEnv):
    def __init__(self, colours=("red", "green"), **kwargs):
        self.colours = _check_colours(colours)
        super().__init__(
            robots="Panda",
            base_types="RethinkMinimalMount",
            controller_configs=None,
            initialization_noise={"magnitude": 0.0, "type": "uniform"},
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            control_freq=30,
            horizon=10000,
            ignore_done=True,
            hard_reset=False,
            renderer="mujoco",
            camera_names=["front_cam", "gripper_cam"],
            camera_heights=512,
            camera_widths=512,
        )

    def _check_robot_configuration(self, robots):
        return

    def _load_model(self):
        super()._load_model()
        # set_base_xpos places the mount's bottom; lower it so link0 itself is at the world origin.
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(robot_model.base.bottom_offset - robot_model.base.top_offset)

        mujoco_arena = CopiedCameraTableArena()
        mujoco_arena.set_origin([0, 0, 0])

        cubes = [BoxObject(name=f"target_cube_{colour}", size=[0.02, 0.02, 0.02],
                           rgba=[float(v) for v in CUBE_RGBA[colour].split()] + [1])
                 for colour in self.colours]

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

        _attach_gripper_cam(self.model.worldbody)


def _attach_gripper_cam(worldbody):
    for body in worldbody.iter("body"):
        if body.get("name") == GRIPPER_BODY_NAME:
            body.append(new_element("camera", **GRIPPER_CAM))
            return


def build_sim(colours=("red", "green")):
    colours = _check_colours(colours)
    env = robosuite.make(
        "BringupEnv",
        colours=colours,
        robots="Panda",
        base_types="RethinkMinimalMount",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=30,
        hard_reset=False,
        renderer="mujoco",
        initialization_noise={"magnitude": 0.0, "type": "uniform"},
    )
    env.reset()
    xml = apply_scripted_contract(env.sim.model.get_xml())
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # env.sim.model.get_xml() bakes joint reference frames, not the post-reset
    # qpos/qvel; copy those over so our standalone data matches env.sim's state.
    data.qpos[:] = env.sim.data.qpos
    data.qvel[:] = env.sim.data.qvel
    mujoco.mj_forward(model, data)
    from data_collection.benchmark import configure_timing
    configure_timing(model, 30)
    real_step = env.step

    def _banned(*args, **kwargs):
        raise RuntimeError("env.step is forbidden; drive data.ctrl and mujoco.mj_step")

    env.step = _banned
    env._robosuite_step = real_step
    return env, model, data


def overwrite_reset(model, data):
    """Deterministic reset: arm at Q_HOME, cubes at FIXED_CUBES, settled for 0.2 s. No RNG."""
    from data_collection.benchmark import capture_state, synchronize_observation
    from experiments.robosuite_scene.contract import FIXED_CUBES
    from src.robots.franka_panda.config import Q_HOME

    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for name, position in FIXED_CUBES.items():
        body = model.body(name)
        joint = model.joint(int(body.jntadr[0]))
        adr = int(joint.qposadr[0])
        data.qpos[adr:adr + 3] = position
        data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[:9] = Q_HOME
    data.ctrl[:] = 0.0
    data.ctrl[:7] = Q_HOME[:7]
    data.ctrl[-1] = 255
    mujoco.mj_forward(model, data)
    for _ in range(round(0.2 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    synchronize_observation(model, data)
    return {"qpos": data.qpos.copy(), "initial_state": capture_state(model, data)}
