import itertools
import unittest

import mujoco
import numpy as np

from experiments.vla_inputs_signals import replay, signals

SIZE, F = 512, 256.0  # a camera at (0, 0, 1) looking straight down, fovy 90: f = 256 / tan(45 deg)
SCENE = """
<mujoco>
  <statistic extent="1"/>
  <visual><map znear="0.01"/></visual>
  <asset>
    <mesh name="wedge" vertex="0 0 0  .2 0 0  0 .1 0  .2 .1 0  0 0 .05  .2 0 .03  0 .1 .05  .2 .1 .03"/>
  </asset>
  <worldbody>
    <camera name="cam" pos="0 0 1" xyaxes="1 0 0 0 1 0" fovy="90"/>
    {body}
  </worldbody>
</mujoco>
"""
ZNEAR = 0.01


def make(body):
    model = mujoco.MjModel.from_xml_string(SCENE.replace("{body}", body))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def corners(centre, half):
    return np.array([np.add(centre, np.multiply(half, s)) for s in itertools.product((-1, 1), repeat=3)])


def expected(points):
    """The box of fully-in-front points, computed from first principles; zeros if it does not touch the image."""
    points = np.asarray(points, float)
    d = 1.0 - points[:, 2]
    assert (d > ZNEAR).all()
    u, v = SIZE / 2 + F * points[:, 0] / d, SIZE / 2 - F * points[:, 1] / d
    lo, hi = np.maximum([u.min(), v.min()], 0), np.minimum([u.max(), v.max()], SIZE)
    if (hi <= lo).any():
        return np.zeros(5)
    return np.array([lo[0], lo[1], hi[0], hi[1], 1.0]) / np.array([SIZE, SIZE, SIZE, SIZE, 1.0])


def box(body):
    model, data = make(body)
    return signals.box(model, data, "obj", "cam")


class AnalyticFixtures(unittest.TestCase):
    def test_axis_point_projects_to_centre(self):
        got = box('<body name="obj"><geom type="box" size="1e-6 1e-6 1e-6" group="1"/></body>')
        self.assertTrue(np.allclose(got, [0.5, 0.5, 0.5, 0.5, 1.0], atol=1e-5), got)

    def test_tight_box_is_the_projection_of_the_corners(self):
        got = box('<body name="obj" pos="0.2 -0.1 0.05"><geom type="box" size=".1 .05 .08" group="1"/></body>')
        want = expected(corners((0.2, -0.1, 0.05), (0.1, 0.05, 0.08)))
        self.assertTrue(np.allclose(got, want, rtol=0, atol=1e-12), (got, want))
        self.assertGreater(got[0], 0.5)  # +x is right
        self.assertGreater(got[1], 0.5)  # +y is up, so v grows downwards: y = -0.1 sits below the centre

    def test_box_is_not_square_or_symmetric(self):
        got = box('<body name="obj" pos="0.2 -0.1 0.05"><geom type="box" size=".1 .05 .08" group="1"/></body>')
        self.assertNotAlmostEqual(got[2] - got[0], got[3] - got[1], places=3)

    def test_partly_out_of_frame_is_clipped_not_dropped(self):
        got = box('<body name="obj" pos="1.0 0 0"><geom type="box" size=".1 .1 .1" group="1"/></body>')
        want = expected(corners((1.0, 0, 0), (0.1, 0.1, 0.1)))
        self.assertEqual(got[2], 1.0)
        self.assertEqual(got[4], 1.0)
        self.assertTrue(np.allclose(got, want, rtol=0, atol=1e-12), (got, want))

    def test_out_of_frame_is_all_zero(self):
        self.assertEqual(box('<body name="obj" pos="5 0 0"><geom type="box" size=".1 .1 .1" group="1"/></body>').tolist(), [0] * 5)

    def test_behind_the_camera_is_all_zero(self):
        self.assertEqual(box('<body name="obj" pos="0 0 2"><geom type="box" size=".1 .1 .1" group="1"/></body>').tolist(), [0] * 5)

    def test_geometry_straddling_the_near_plane_is_clipped_before_the_division(self):
        # camera plane z = 1 cuts the box: only the z = 0.9 face is in front, but the clipped slice at the near
        # plane makes the box cover the whole image
        got = box('<body name="obj" pos="0.05 0 1"><geom type="box" size=".1 .1 .1" group="1"/></body>')
        self.assertEqual(got.tolist(), [0.0, 0.0, 1.0, 1.0, 1.0])

    def test_vertices_behind_the_camera_do_not_mirror_into_the_image(self):
        # dividing by a negative depth would send the z > 1 vertices to the opposite side of the image
        got = box('<body name="obj" pos="-0.4 0 1"><geom type="box" size=".1 .05 .1" group="1"/></body>')
        self.assertEqual(got.tolist(), [0] * 5)

    def test_only_rendered_geom_groups_count(self):
        body = ('<body name="obj" pos="0.1 0.1 0"><geom type="box" size=".05 .05 .05" group="1"/>'
                '<geom type="box" size=".2 .2 .2" group="3"/></body>')
        got = box(body)
        want = expected(corners((0.1, 0.1, 0), (0.05, 0.05, 0.05)))
        self.assertTrue(np.allclose(got, want, rtol=0, atol=1e-12), (got, want))

    def test_compiled_mesh_transform(self):
        body = '<body name="obj" pos="0.1 0.05 0.2"><geom type="mesh" mesh="wedge" pos="0.02 -0.01 0.03" euler="0 0 30" group="1"/></body>'
        vertices = np.array([[0, 0, 0], [.2, 0, 0], [0, .1, 0], [.2, .1, 0], [0, 0, .05], [.2, 0, .03], [0, .1, .05], [.2, .1, .03]])
        angle = np.radians(30)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        world = vertices @ rotation.T + np.array([0.02, -0.01, 0.03]) + np.array([0.1, 0.05, 0.2])
        got, want = box(body), expected(world)
        self.assertTrue(np.allclose(got, want, rtol=0, atol=1e-6), (got, want))  # mesh vertices are float32

    def test_primitive_bound_is_conservative_and_documented(self):
        body = '<body name="obj" pos="0.3 0 0"><geom type="sphere" size=".1" group="1"/></body>'
        got = box(body)
        rng = np.random.default_rng(0)
        directions = rng.normal(size=(4000, 3))
        surface = np.array([0.3, 0, 0]) + 0.1 * directions / np.linalg.norm(directions, axis=1, keepdims=True)
        true = expected(surface)
        looser = expected(corners((0.3, 0, 0), (0.1, 0.1, 0.1)))
        self.assertTrue((got[:2] <= true[:2] + 1e-12).all() and (got[2:4] >= true[2:4] - 1e-12).all())
        self.assertTrue(np.allclose(got, looser, rtol=0, atol=1e-12))  # the bound is the half-extent cube

    def test_reading_a_box_writes_nothing(self):
        model, data = make('<body name="obj" pos="0.2 0 0"><geom type="box" size=".1 .1 .1" group="1"/></body>')
        before = {n: getattr(data, n).copy() for n in ("qpos", "xpos", "xmat", "cam_xpos", "cam_xmat", "geom_xpos")}
        signals.box(model, data, "obj", "cam")
        for name, value in before.items():
            self.assertTrue(np.array_equal(value, getattr(data, name)), name)


TICKS = tuple(int(t) for t in np.linspace(0, 451, 6))


class SegmentationContainment(unittest.TestCase):
    def test_every_target_pixel_is_inside_the_box(self):
        """G4: 10 episodes x 6 frames x 2 cameras against a segmentation render of the live scene."""
        frames, seen, ious, shapes_checked = 0, {"front_cam": 0, "gripper_cam": 0}, [], 0
        for index in range(0, 540, 54):
            target = replay.load_episode(index)["episode"]["object"]
            holder = {}

            def on_frame(tick, model, data):
                nonlocal frames, shapes_checked
                if tick not in TICKS:
                    return
                if "renderer" not in holder:
                    holder["renderer"] = mujoco.Renderer(model, SIZE, SIZE)
                    holder["renderer"].enable_segmentation_rendering()
                renderer = holder["renderer"]
                body = model.body(f"prop_{target}").id
                geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
                for camera in signals.CAMERAS:
                    renderer.update_scene(data, camera=camera)
                    seg = renderer.render()
                    mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], geoms)
                    frames += 1
                    got = signals.box(model, data, f"prop_{target}", camera)
                    if not mask.any():
                        continue
                    ys, xs = np.nonzero(mask)
                    px = got[:4] * SIZE
                    self.assertEqual(got[4], 1.0, (index, tick, camera))
                    self.assertGreaterEqual(xs.min(), px[0] - 1, (index, tick, camera, "u_min"))
                    self.assertLessEqual(xs.max() + 1, px[2] + 1, (index, tick, camera, "u_max"))
                    self.assertGreaterEqual(ys.min(), px[1] - 1, (index, tick, camera, "v_min"))
                    self.assertLessEqual(ys.max() + 1, px[3] + 1, (index, tick, camera, "v_max"))
                    seen[camera] += 1
                    inter = max(0, min(xs.max() + 1, px[2]) - max(xs.min(), px[0])) * max(0, min(ys.max() + 1, px[3]) - max(ys.min(), px[1]))
                    union = (xs.max() + 1 - xs.min()) * (ys.max() + 1 - ys.min()) + (px[2] - px[0]) * (px[3] - px[1]) - inter
                    ious.append(inter / union)
                stacked = signals.target_box(model, data, target)
                self.assertEqual(stacked.shape, (2, 5))
                self.assertTrue(np.array_equal(stacked[0], signals.box(model, data, f"prop_{target}", "front_cam")))
                self.assertTrue(np.array_equal(stacked[1], signals.box(model, data, f"prop_{target}", "gripper_cam")))
                every = signals.props_box(model, data, [target, target])
                self.assertEqual(every.shape, (2, 2, 5))
                shapes_checked += 1

            try:
                replay.frames(index, on_frame)
            finally:
                if "renderer" in holder:
                    holder["renderer"].close()
        print(f"\nG4: {frames} frame-camera renders, {sum(seen.values())} with target pixels {seen}, "
              f"IoU mean {np.mean(ious):.3f} min {np.min(ious):.3f}")
        self.assertEqual(frames, 120)
        self.assertEqual(shapes_checked, 60)
        self.assertGreaterEqual(seen["front_cam"], 30)
        self.assertGreaterEqual(seen["gripper_cam"], 10)


if __name__ == "__main__":
    unittest.main()
