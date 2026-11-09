"""Engine tests; robosuite image (MuJoCo 3.4.0). The texture folder and the sealed P2 data must be mounted."""
import math
import unittest

import mujoco
import numpy as np

from experiments.gen_scene import compose, gates, levels, pools
from experiments.gen_scene.conditions import NOMINAL, Condition
from helpers import layouts, make_layout

PROPS, SLOT = ("can", "bread"), "left"


def build(condition, primitives=(), props=PROPS, slot=SLOT):
    return compose.build_model(props, primitives, slot, condition)


def lights(model):
    h = model.vis.headlight
    return dict(diffuse=model.light_diffuse[0].copy(), specular=model.light_specular[0].copy(), ambient=model.light_ambient[0].copy(),
                dir=model.light_dir[0].copy(), shadow=int(model.light_castshadow[0]),
                hl_diffuse=np.array(h.diffuse, dtype=np.float64), hl_specular=np.array(h.specular, dtype=np.float64),
                hl_ambient=np.array(h.ambient, dtype=np.float64))


def settled(condition, layout=None):
    layout = layout or make_layout(0)
    model, resolved = compose.model_for(layout, condition)
    scene = compose.reset(model, resolved["poses"], layout["home_pose_jitter_deg"])
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    from data_collection.benchmark import restore_state
    restore_state(model, data, scene["initial_state"])
    mujoco.mj_forward(model, data)
    return model, data, resolved


class NominalIsP2s(unittest.TestCase):
    """G0 spec D9 (i): the engine's nominal scene is props_scene's."""

    def test_three_sealed_episodes_compose_and_render_byte_equal(self):
        for index in gates.GATE_EPISODES:
            result = gates.equality(index)
            self.assertEqual(result["arrays_compared"] > 400, True)
            self.assertEqual(result["differing_arrays"], [], index)
            self.assertTrue(result["initial_state_equal_reference"] and result["initial_state_equal_sealed"], index)
            self.assertEqual(result["renders_byte_equal"], [True, True], index)
            self.assertTrue(all(s > 1.0 for s in result["render_std"]))

    def test_the_comparison_sees_a_change(self):
        a, b = build(NOMINAL), build(levels.lighting("dark", 0))
        count, bad = gates.differing_arrays(a, b)
        self.assertIn("light_diffuse", bad)
        self.assertGreater(count, 400)


class Lights(unittest.TestCase):
    def setUp(self):
        self.base = lights(build(NOMINAL))

    def test_nominal_values(self):
        np.testing.assert_allclose(self.base["diffuse"], 0.7, atol=1e-6)
        np.testing.assert_allclose(self.base["hl_diffuse"], 0.4, atol=1e-6)
        self.assertEqual(self.base["shadow"], 0)

    def test_dark(self):
        got = lights(build(levels.lighting("dark", 0)))
        np.testing.assert_allclose(got["diffuse"], self.base["diffuse"] * 0.3, rtol=1e-6)
        np.testing.assert_allclose(got["ambient"], self.base["ambient"] * 0.3, rtol=1e-6)
        np.testing.assert_allclose(got["hl_diffuse"], self.base["hl_diffuse"] * 0.3, rtol=1e-6)
        np.testing.assert_allclose(got["hl_ambient"], self.base["hl_ambient"] * 0.3, rtol=1e-6)
        np.testing.assert_allclose(got["specular"], self.base["specular"], rtol=1e-6)
        np.testing.assert_allclose(got["hl_specular"], self.base["hl_specular"], rtol=1e-6)

    def test_bright(self):
        got = lights(build(levels.lighting("bright", 0)))
        np.testing.assert_allclose(got["diffuse"], self.base["diffuse"] * 2.5, rtol=1e-6)
        np.testing.assert_allclose(got["specular"], self.base["specular"] * 2.5, rtol=1e-6)
        np.testing.assert_allclose(got["hl_diffuse"], self.base["hl_diffuse"] * 2.5, rtol=1e-6)
        np.testing.assert_allclose(got["hl_specular"], self.base["hl_specular"] * 2.5, rtol=1e-6)
        np.testing.assert_allclose(got["hl_ambient"], self.base["hl_ambient"], rtol=1e-6)

    def test_tinted_warm_and_cool(self):
        for j, tint in ((0, (1.0, 0.75, 0.5)), (1, (0.6, 0.75, 1.0))):
            got = lights(build(levels.lighting("tinted", j)))
            for key in ("diffuse", "specular", "hl_diffuse", "hl_specular", "hl_ambient"):
                np.testing.assert_allclose(got[key], self.base[key] * np.asarray(tint), rtol=1e-6, err_msg=f"{key} {tint}")

    def test_low_angle(self):
        got = lights(build(levels.lighting("low_angle", 0)))
        np.testing.assert_allclose(got["dir"], levels.LOW_ANGLE_DIR, atol=1e-6)
        self.assertEqual(got["shadow"], 1)
        np.testing.assert_allclose(got["diffuse"], self.base["diffuse"], rtol=1e-6)
        for key in ("hl_diffuse", "hl_specular", "hl_ambient"):
            np.testing.assert_allclose(got[key], self.base[key] * 0.3, rtol=1e-6, err_msg=key)
        elevation = math.degrees(math.asin(-got["dir"][2]))
        self.assertAlmostEqual(elevation, 20.0, places=4)

    def test_lights_do_not_touch_the_scene(self):
        a, b = build(NOMINAL), build(levels.lighting("dark", 0))
        _, bad = gates.differing_arrays(a, b)
        self.assertEqual(sorted(bad), ["light_diffuse", "vis.headlight.ambient", "vis.headlight.diffuse"])
        self.assertNotIn("geom_pos", bad)


class Table(unittest.TestCase):
    def check(self, offset):
        condition = levels.spatial({0.05: "table_up5", -0.05: "table_down5", 0.10: "table_up10"}[offset], make_layout(0), "x", 0)
        base, _, base_resolved = settled(NOMINAL)
        model, data, resolved = settled(condition)
        self.assertAlmostEqual(compose.table_top_z(model), offset, places=12)
        for colour in ("blue", "orange"):
            nominal_pos = base.body(f"bin_{colour}").pos
            self.assertAlmostEqual(model.body(f"bin_{colour}").pos[2], offset, places=12)
            self.assertEqual(tuple(model.body(f"bin_{colour}").pos[:2]), tuple(nominal_pos[:2]))
        floor = model.geom("floor").pos[2]
        slab = model.geom("table_collision")
        bottom = model.body("table").pos[2] - slab.size[2]
        self.assertAlmostEqual(bottom, floor, places=12)              # the slab reaches the floor: no gap, no coplanar faces
        self.assertAlmostEqual(floor, min(-0.05, offset - 0.05), places=12)
        np.testing.assert_array_equal(model.geom("table_visual").size, slab.size)
        return base, model, data

    def test_every_on_table_body_moves_by_exactly_the_offset(self):
        for offset in (0.05, -0.05, 0.10):
            base_model, model, data = self.check(offset)
            _, base_data, _ = settled(NOMINAL)
            for name in ("can", "bread"):
                dz = data.xpos[model.body(f"prop_{name}").id][2] - base_data.xpos[base_model.body(f"prop_{name}").id][2]
                self.assertAlmostEqual(dz, offset, delta=1e-4, msg=f"{name} at {offset}")   # settling differs by tens of micrometres
                dxy = data.xpos[model.body(f"prop_{name}").id][:2] - base_data.xpos[base_model.body(f"prop_{name}").id][:2]
                self.assertLess(float(np.abs(dxy).max()), 1e-4)
            np.testing.assert_array_equal(data.qpos[:9], base_data.qpos[:9])   # the arm does not move with the table

    def test_the_arm_stays_where_it_was(self):
        a = settled(NOMINAL)[1]
        b = settled(levels.spatial("table_up10", make_layout(0), "x", 0))[1]
        np.testing.assert_array_equal(a.qpos[:9], b.qpos[:9])


class Bins(unittest.TestCase):
    def test_shift_moves_each_bin_away_or_together_along_y(self):
        for slot in ("left", "right"):
            for shift in (0.04, -0.04):
                model = build(Condition(bin_shift_m=shift), slot=slot)
                blue_y = model.body("bin_blue").pos[1]
                orange_y = model.body("bin_orange").pos[1]
                sign = 1.0 if slot == "left" else -1.0
                self.assertAlmostEqual(blue_y, sign * (0.33 + shift), places=12)
                self.assertAlmostEqual(orange_y, -sign * (0.33 + shift), places=12)
                self.assertAlmostEqual(model.body("bin_blue").pos[0], 0.32, places=12)
                self.assertEqual(model.body("bin_blue").pos[2], 0.0)

    def test_zero_shift_is_the_nominal_scene(self):
        _, bad = gates.differing_arrays(build(NOMINAL), build(Condition(bin_shift_m=0.0)))
        self.assertEqual(bad, [])


class Textures(unittest.TestCase):
    def setUp(self):
        self.base = build(NOMINAL)

    def changed_geoms(self, model):
        a, b = self.base, model
        return sorted(a.geom(i).name for i in range(a.ngeom) if a.geom_matid[i] != b.geom_matid[i])

    def test_table_only(self):
        c = levels.background("table", 0)
        model = build(c)
        self.assertEqual(self.changed_geoms(model), ["table_visual"])
        material = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "gen_table_mat")
        self.assertEqual(model.geom("table_visual").matid[0], material)
        texture = int(model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB])
        self.assertEqual((model.tex_width[texture], model.tex_height[texture]), (1024, 1024))
        self.assertEqual(model.ntex, self.base.ntex + 1)

    def test_room_only(self):
        model = build(levels.background("room", 0))
        walls = sorted(model.geom(i).name for i in range(model.ngeom) if model.geom(i).name.startswith("wall_"))
        self.assertEqual(len(walls), 6)                    # the arena's six walls, read from the model, not from compose
        self.assertEqual(sorted(compose.WALLS), walls)
        self.assertEqual(self.changed_geoms(model), sorted(["floor", *walls]))
        floor = model.geom("floor").matid[0]
        wall = model.geom("wall_rear_visual").matid[0]
        self.assertNotEqual(floor, wall)
        self.assertEqual({int(model.geom(n).matid[0]) for n in walls}, {int(wall)})

    def test_both(self):
        model = build(levels.background("both", 0))
        walls = [model.geom(i).name for i in range(model.ngeom) if model.geom(i).name.startswith("wall_")]
        self.assertEqual(self.changed_geoms(model), sorted(["table_visual", "floor", *walls]))

    def test_the_texture_is_the_manifest_file_decoded(self):
        name = pools.pools()["table"]["test"][0]
        png = compose.texture_png(name)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        from PIL import Image
        import io
        decoded = np.asarray(Image.open(io.BytesIO(png)))
        original = np.asarray(Image.open(pools.texture_path(name)).convert("RGB"))
        np.testing.assert_array_equal(decoded, original)

    def test_a_texture_of_the_wrong_pool_never_reaches_a_model(self):
        both = pools.pools()
        with self.assertRaises(ValueError):
            build(Condition(pool="train", table_texture=both["table"]["test"][0]))
        with self.assertRaises(ValueError):
            build(Condition(pool="test", table_texture=both["table"]["train"][0]))

    def test_the_render_changes(self):
        _, data, _ = settled(NOMINAL)
        model, data2, _ = settled(levels.background("both", 0))
        a = gates.render(*(settled(NOMINAL)[:2]))
        b = gates.render(model, data2)
        self.assertGreater(float(np.abs(a[0].astype(int) - b[0].astype(int)).mean()), 5.0)


class Primitives(unittest.TestCase):
    def test_bodies_shapes_and_placement(self):
        layout = make_layout(0)
        condition = levels.distractors("five", layout, layout["identity"])
        model, data, resolved = settled(condition, layout)
        self.assertEqual(len(condition.primitives), 2)
        for shape, colour, x, y, yaw in condition.primitives:
            name = f"prop_{shape}_{colour}"
            body = model.body(name)
            kind, size = next(t[shape] for t in pools.PRIMITIVE_SHAPES.values() if shape in t)
            geom = model.geom(f"{name}_geom")
            self.assertEqual(mujoco.mjtGeom(geom.type).name.replace("mjGEOM_", "").lower(), kind)
            np.testing.assert_allclose(geom.size[:len(size)], size, rtol=1e-6)
            np.testing.assert_allclose(geom.rgba[:3], pools.PRIMITIVE_COLOURS["B"][colour], rtol=1e-6)
            self.assertAlmostEqual(float(model.body_mass[body.id]), pools.PRIMITIVE_MASS_KG, places=6)
            self.assertLess(abs(data.xpos[body.id][0] - x), 2e-3)
            self.assertLess(abs(data.xpos[body.id][1] - y), 2e-3)
            radius, height = pools.primitive_extent(shape)
            self.assertLess(abs(data.xpos[body.id][2] - height / 2), 2e-3, shape)   # resting on the table, not sunk or floating
        self.assertEqual(resolved["names"][:1 + len(layout["distractors"])], [layout["object"], *(d["object"] for d in layout["distractors"])])
        self.assertEqual(resolved["names"][-2:], [f"{s}_{c}" for s, c, *_ in condition.primitives])

    def test_a_capsule_lies_along_x(self):
        c = Condition(pool="test", primitives=(("pill", "pink", 0.45, 0.0, 0.0),))
        model = build(c, primitives=[("pill", "pink")])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        geom = model.geom("prop_pill_pink_geom")
        axis = data.geom_xmat[geom.id].reshape(3, 3)[:, 2]
        self.assertGreater(abs(float(axis[0])), 0.99)

    def test_a_pool_a_primitive_never_reaches_a_test_scene(self):
        with self.assertRaises(ValueError):
            build(Condition(pool="test", primitives=(("cube", "red", 0.4, 0.0, 0.0),)), primitives=[("cube", "red")])


class EveryLevelRenders(unittest.TestCase):
    def test_both_cameras_of_every_level(self):
        ls = layouts(24)
        conditions = levels.assign(ls)
        seen = set()
        for layout in ls[:12]:
            for axis, c in conditions[layout["layout_id"]].items():
                if (axis, c.level) in seen:
                    continue
                seen.add((axis, c.level))
                model, data, _ = settled(c, layout)
                for image in gates.render(model, data):
                    self.assertEqual(image.shape, (512, 512, 3))
                    self.assertGreater(float(image.std()), 5.0, (axis, c.level))
        self.assertEqual(len(seen), sum(len(v) for v in levels.LEVELS.values()))


if __name__ == "__main__":
    unittest.main()
