import copy
import unittest

import mujoco
import numpy as np

from data_collection.benchmark import STATE_SPEC, capture_state, restore_state, synchronize_observation
from experiments.props_p2_data import episode as p2_episode
from experiments.props_scene import scene
from experiments.vla_inputs_signals import replay, signals

HAND = "robot0_right_hand"
EPISODES = (0, 271, 539)


def reference_run(index, on_frame):
    """Replay `index`; `on_frame(tick, model, data, reference_model, reference_data)` with a native-sensor twin.

    The twin is the same compiled scene plus a site at the hand origin carrying MuJoCo's own force and torque
    sensors. A model rebuilt from XML text differs from the live one in the last digits (actuator gains), so every
    numeric array both models share is copied from the live model before the twin is used.
    """
    sealed = replay.load_episode(index)
    objects, slot = p2_episode.names_of(sealed["episode"]), sealed["episode"]["blue_slot"]
    scene._built.cache_clear()
    p2_episode._EXPERTS.clear()
    expert = p2_episode.expert_for(objects, slot)
    env, model = scene.build_model(expert.objects, slot)
    spec = mujoco.MjSpec.from_string(env.sim.model.get_xml())
    spec.body(HAND).add_site(name="s0_hand", pos=[0, 0, 0])
    for name, kind in (("s0_force", mujoco.mjtSensor.mjSENS_FORCE), ("s0_torque", mujoco.mjtSensor.mjSENS_TORQUE)):
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="s0_hand")
    twin = spec.compile()
    for name in dir(model):
        value = getattr(model, name)
        if name.startswith(("_", "site_", "sensor_")) or not isinstance(value, np.ndarray) or value.dtype.kind != "f":
            continue
        shared = getattr(twin, name)
        if shared.shape == value.shape:
            shared[...] = value
    twin.opt.timestep = model.opt.timestep
    twin_data = mujoco.MjData(twin)
    return replay.frames(index, lambda tick, m, d: on_frame(tick, m, d, twin, twin_data))


def native_wrench(twin, twin_data, state):
    mujoco.mj_setState(twin, twin_data, state, STATE_SPEC)
    mujoco.mj_forward(twin, twin_data)
    force = twin_data.sensordata[twin.sensor_adr[twin.sensor("s0_force").id]:][:3]
    torque = twin_data.sensordata[twin.sensor_adr[twin.sensor("s0_torque").id]:][:3]
    return np.concatenate([force, torque])


def independent_finger_force(model, data):
    """Same quantity through a different route: geom sets and a plain loop over `mj_contactForce`."""
    robot_root = model.body_rootid[model.body(HAND).id]
    robot_geoms = {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == robot_root}
    finger_geoms = []
    for name in ("left_finger", "right_finger"):
        body = model.body(name).id
        finger_geoms.append({g for g in robot_geoms if _is_below(model, int(model.geom_bodyid[g]), body)})
    out, force = [0.0, 0.0], np.zeros(6)
    for i in range(data.ncon):
        g1, g2 = int(data.contact[i].geom1), int(data.contact[i].geom2)
        mujoco.mj_contactForce(model, data, i, force)
        for k in range(2):
            for mine, other in ((g1, g2), (g2, g1)):
                if mine in finger_geoms[k] and other not in robot_geoms:
                    out[k] += float(force[0])
    return np.array(out)


def _is_below(model, body, ancestor):
    while body != 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


class WristWrenchAgainstMuJoCo(unittest.TestCase):
    def test_wrench_matches_mujoco_sensor(self):
        """G3: on the sealed trajectories, from the same integration state the live forward started from."""
        compared = 0
        for index in EPISODES:
            worst = []

            def on_frame(tick, model, data, twin, twin_data):
                if tick % 10:
                    return
                reference = native_wrench(twin, twin_data, capture_state(model, data))
                worst.append(float(np.abs(signals.wrist_wrench(model, data) - reference).max()))

            reference_run(index, on_frame)
            compared += len(worst)
            self.assertLess(max(worst), 1e-9, f"episode {index}")
        self.assertEqual(compared, 3 * 46)


ROBOT_DOF = 9  # seven arm joints and two fingers


def hold_still(model, data):
    """Joint forces that cancel everything except gravity on the hand subtree: no robot dof accelerates."""
    data.qvel[:] = 0.0
    for _ in range(3):
        mujoco.mj_forward(model, data)
        data.qfrc_applied[:ROBOT_DOF] = (data.qfrc_bias - data.qfrc_actuator - data.qfrc_passive
                                         - data.qfrc_constraint)[:ROBOT_DOF]
    mujoco.mj_forward(model, data)


class StaticFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.model = scene.build_model(("can",), "left")
        initial = scene.reset(cls.model, {"can": (0.33, 0.0, 0.0)}, [0.0] * 7)["initial_state"]
        cls.data = mujoco.MjData(cls.model)
        restore_state(cls.model, cls.data, initial)
        hold_still(cls.model, cls.data)

    def test_the_fixture_is_static(self):
        self.assertLess(float(np.abs(self.data.qacc[:ROBOT_DOF]).max()), 1e-4)
        self.assertLess(float(np.abs(self.data.qvel).max()), 1e-6)

    def test_static_known_load_sign_frame_and_reference_point(self):
        model, data = self.model, self.data
        hand = model.body(HAND).id
        mass, weight = float(model.body_subtreemass[hand]), float(model.body_subtreemass[hand]) * 9.81
        up = np.array([0.0, 0.0, weight])  # the parent holds the hand and everything below it up
        rotation = data.xmat[hand].reshape(3, 3)
        lever = data.subtree_com[hand] - data.xpos[hand]
        expected = np.concatenate([rotation.T @ up, rotation.T @ np.cross(lever, up)])
        wrench = signals.wrist_wrench(model, data)
        self.assertGreater(mass, 0.5)
        self.assertLess(float(np.abs(wrench - expected).max()), 1e-6)
        self.assertGreater(float(np.linalg.norm(wrench[3:])), 1e-3)  # a nonzero torque, so the lever arm is tested

    def test_no_gravity_means_no_load(self):
        model = self.model
        data = copy.copy(self.data)
        gravity = model.opt.gravity.copy()
        try:
            model.opt.gravity[:] = 0.0
            data.qfrc_applied[:] = 0.0
            hold_still(model, data)
            self.assertLess(float(np.abs(data.qacc[:ROBOT_DOF]).max()), 1e-4)
            self.assertLess(float(np.abs(signals.wrist_wrench(model, data)).max()), 1e-4)  # weight is ~10 N
        finally:
            model.opt.gravity[:] = gravity

    def test_no_contact_means_zero_finger_force(self):
        self.assertEqual(signals.finger_force(self.model, self.data).tolist(), [0.0, 0.0])

    def test_signals_do_not_write_data(self):
        model, data = self.model, copy.copy(self.data)
        data.cacc[:] = 7.0  # poison what `mj_rnePostConstraint` writes, so a write on `data` cannot go unseen
        data.cfrc_int[:] = 7.0
        names = ("qpos", "qvel", "qacc", "qacc_warmstart", "efc_force", "cacc", "cfrc_int", "cfrc_ext", "qfrc_constraint")
        before = {name: getattr(data, name).copy() for name in names}
        state = capture_state(model, data)
        signals.wrist_wrench(model, data)
        signals.finger_force(model, data)
        for name in names:
            self.assertTrue(np.array_equal(before[name], getattr(data, name)), name)
        self.assertTrue(np.array_equal(state, capture_state(model, data)))


TINY = """
<mujoco>
  <option timestep="0.002" cone="elliptic"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="block" pos="0 0 {bz}"><freejoint/><geom name="block" type="box" size=".01 .01 .01" mass=".1"/></body>
    <body name="base" pos="0 0 0.3"><freejoint/>
      <body name="robot0_right_hand"><geom name="hand" type="box" size=".02 .02 .005" pos="0 0 .02" contype="0" conaffinity="0"/>
        <body name="left_finger" pos="0 {y} 0"><joint type="slide" axis="0 1 0"/><geom name="left_pad" type="box" size=".01 .01 .01"/></body>
        <body name="right_finger" pos="0 -{y} 0"><joint type="slide" axis="0 1 0"/><geom name="right_pad" type="box" size=".01 .01 .01"/></body>
      </body>
      <body name="link" pos="0 .3 -.205"><geom name="link" type="box" size=".05 .05 .1"/></body>
    </body>
  </worldbody>
</mujoco>
"""


def tiny(y, block_z=0.3):
    model = mujoco.MjModel.from_xml_string(TINY.replace("{y}", str(y)).replace("{bz}", str(block_z)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class FingerForceFilters(unittest.TestCase):
    def test_both_fingers_squeeze_an_external_block(self):
        model, data = tiny(0.0195)  # pad inner faces 0.5 mm inside the block
        out = signals.finger_force(model, data)
        self.assertTrue((out > 0.1).all(), out)
        self.assertLess(abs(out[0] - out[1]) / out.max(), 0.05)
        self.assertTrue(np.allclose(out, independent_finger_force(model, data), rtol=0, atol=1e-12))

    def test_normal_component_only(self):
        model, data = tiny(0.0195)
        force, normals, totals = np.zeros(6), [], []
        for i in range(data.ncon):
            names = {model.geom(data.contact[i].geom1).name, model.geom(data.contact[i].geom2).name}
            if not names & {"left_pad", "right_pad"}:
                continue
            mujoco.mj_contactForce(model, data, i, force)
            normals.append(float(force[0]))
            totals.append(float(np.linalg.norm(force[:3])))
        self.assertGreater(max(totals), max(normals) + 1e-6)  # the block's weight is carried by friction
        self.assertAlmostEqual(float(signals.finger_force(model, data).sum()), sum(normals), places=9)

    def test_finger_on_finger_is_not_external(self):
        model, data = tiny(0.005, block_z=0.8)  # the two pads overlap each other and nothing else
        force = np.zeros(6)
        pads = [i for i in range(data.ncon)
                if {model.geom(data.contact[i].geom1).name, model.geom(data.contact[i].geom2).name} == {"left_pad", "right_pad"}]
        self.assertTrue(pads)
        mujoco.mj_contactForce(model, data, pads[0], force)
        self.assertGreater(abs(float(force[0])), 0.01)  # a real pad-on-pad force exists and must not be counted
        self.assertEqual(signals.finger_force(model, data).tolist(), [0.0, 0.0])

    def test_other_robot_links_are_not_fingers(self):
        model, data = tiny(0.1, block_z=0.8)  # fingers far from everything; the link rests in the floor
        link_floor = [i for i in range(data.ncon)
                      if {model.geom(data.contact[i].geom1).name, model.geom(data.contact[i].geom2).name} == {"floor", "link"}]
        self.assertTrue(link_floor)
        self.assertEqual(signals.finger_force(model, data).tolist(), [0.0, 0.0])


class FingerForceOnTheTrajectories(unittest.TestCase):
    def test_matches_an_independent_implementation_and_is_positive_in_the_hold(self):
        seen = {}

        def on_frame(tick, model, data, twin, twin_data):
            if tick % 10 == 0:
                mine = signals.finger_force(model, data)
                assert np.allclose(mine, independent_finger_force(model, data), rtol=0, atol=1e-12), tick
                assert np.isfinite(mine).all() and (mine >= 0).all()
            if tick == 133:  # end of `close`: the fingers are on the can, nothing has lifted yet
                seen["hold"] = signals.finger_force(model, data)

        reference_run(271, on_frame)
        self.assertTrue((seen["hold"] > 0.05).all(), seen["hold"])


if __name__ == "__main__":
    unittest.main()
