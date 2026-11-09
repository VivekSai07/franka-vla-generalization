import json
import unittest

from experiments.gen_scene import pools
from experiments.gen_scene.conditions import NOMINAL, Condition, primitive_name, resolve

NOMINAL_SHA256 = "e4028bdbe137b39766537b3afc28d3ee4cf51d85a06e1477f6836f4197106524"


def layout():
    return dict(object="can", bin="blue", blue_slot="left", start=[0.3, 0.0, 10.0],
                distractors=[dict(object="bread", pose=[0.35, 0.1, -5.0])])


class Nominal(unittest.TestCase):
    def test_nominal_changes_nothing(self):
        self.assertTrue(NOMINAL.is_nominal())
        self.assertEqual((NOMINAL.light_diffuse_scale, NOMINAL.light_specular_scale, NOMINAL.ambient_scale, NOMINAL.headlight_scale),
                         (1.0, 1.0, 1.0, 1.0))
        self.assertEqual((NOMINAL.tint, NOMINAL.light_dir, NOMINAL.light_shadows), ((1.0, 1.0, 1.0), None, False))
        self.assertEqual((NOMINAL.table_texture, NOMINAL.floor_texture, NOMINAL.wall_texture), (None, None, None))
        self.assertEqual((NOMINAL.table_offset_m, NOMINAL.bin_shift_m, NOMINAL.extra_props, NOMINAL.primitives, NOMINAL.start),
                         (0.0, 0.0, (), (), None))
        self.assertEqual((NOMINAL.disturbance, NOMINAL.move_to, NOMINAL.pool, NOMINAL.axis, NOMINAL.level),
                         ("none", None, "none", "nominal", "nominal"))
        NOMINAL.check()

    def test_digest_is_pinned(self):
        self.assertEqual(NOMINAL.digest(), NOMINAL_SHA256)


class Serialisation(unittest.TestCase):
    def sample(self):
        table = pools.pools()["table"]["test"][0]
        return Condition(axis="distractors", level="five", pool="test", table_texture=table, light_dir=(0.0, -0.6, -0.8),
                         extra_props=(("milk", 0.3, 0.1, 5.0),), primitives=(("ball", "pink", 0.4, -0.1, 3.0),),
                         start=(0.45, 0.0, 1.0), disturbance="move", move_to=(0.3, 0.1))

    def test_json_round_trip_is_exact(self):
        c = self.sample()
        again = Condition.from_dict(json.loads(json.dumps(c.to_dict())))
        self.assertEqual(again, c)
        self.assertEqual(again.digest(), c.digest())

    def test_digest_changes_with_any_field(self):
        c = self.sample()
        self.assertNotEqual(c.digest(), Condition.from_dict({**c.to_dict(), "table_offset_m": 0.05}).digest())

    def test_unknown_field_refused(self):
        with self.assertRaises(ValueError):
            Condition.from_dict({"nonsense": 1})


class Pools(unittest.TestCase):
    def setUp(self):
        both = pools.pools()
        self.table_train, self.table_test = both["table"]["train"][0], both["table"]["test"][0]
        self.room_train, self.room_test = both["room"]["train"][0], both["room"]["test"][0]

    def test_a_test_texture_is_refused_in_a_train_condition_and_the_reverse(self):
        Condition(pool="test", table_texture=self.table_test).check()
        Condition(pool="train", table_texture=self.table_train).check()
        with self.assertRaises(ValueError):
            Condition(pool="train", table_texture=self.table_test).check()
        with self.assertRaises(ValueError):
            Condition(pool="test", table_texture=self.table_train).check()
        with self.assertRaises(ValueError):
            Condition(pool="train", floor_texture=self.room_test, wall_texture=self.room_train).check()

    def test_a_room_texture_is_not_a_table_texture(self):
        with self.assertRaises(ValueError):
            Condition(pool="test", table_texture=self.room_test).check()

    def test_textures_need_a_declared_pool(self):
        with self.assertRaises(ValueError):
            Condition(pool="none", table_texture=self.table_test).check()

    def test_primitives_follow_the_pool(self):
        Condition(pool="test", primitives=(("ball", "pink", 0.4, 0.0, 0.0),)).check()
        Condition(pool="train", primitives=(("cube", "red", 0.4, 0.0, 0.0),)).check()
        with self.assertRaises(ValueError):
            Condition(pool="train", primitives=(("ball", "pink", 0.4, 0.0, 0.0),)).check()
        with self.assertRaises(ValueError):
            Condition(pool="test", primitives=(("cube", "red", 0.4, 0.0, 0.0),)).check()
        with self.assertRaises(ValueError):
            Condition(pool="test", primitives=(("ball", "red", 0.4, 0.0, 0.0),)).check()  # an A colour on a B shape
        with self.assertRaises(ValueError):
            Condition(pool="none", primitives=(("ball", "pink", 0.4, 0.0, 0.0),)).check()


class Ranges(unittest.TestCase):
    def test_refusals(self):
        for bad in (dict(table_offset_m=0.25), dict(table_offset_m=-0.11), dict(bin_shift_m=0.2), dict(light_diffuse_scale=0.0),
                    dict(tint=(1.2, 1.0, 1.0)), dict(light_dir=(0.0, 0.0, -2.0)), dict(light_dir=(0.0, 1.0, 0.0)),
                    dict(disturbance="move"), dict(move_to=(0.3, 0.0)), dict(disturbance="fall"), dict(pool="both"),
                    dict(extra_props=(("milk", 0, 0, 0), ("milk", 1, 1, 1))),
                    dict(pool="test", primitives=(("ball", "pink", 0, 0, 0), ("ball", "pink", 1, 1, 1)))):
            with self.assertRaises(ValueError, msg=str(bad)):
                Condition(**bad).check()

    def test_range_ends_are_allowed(self):
        Condition(table_offset_m=0.20).check()
        Condition(table_offset_m=-0.10).check()


class Resolve(unittest.TestCase):
    def test_target_first_then_layout_then_extras_then_primitives(self):
        condition = Condition(pool="test", extra_props=(("milk", 0.4, -0.1, 7.0),), primitives=(("ball", "pink", 0.45, 0.05, 3.0),),
                              start=(0.44, 0.02, -9.0)).check()
        out = resolve(layout(), condition)
        self.assertEqual(out["names"], ["can", "bread", "milk", "ball_pink"])
        self.assertEqual(out["props"], ["can", "bread", "milk"])
        self.assertEqual(out["primitives"], [("ball", "pink")])
        self.assertEqual(out["primitive_names"], ["ball_pink"])
        self.assertEqual(out["poses"], {"can": (0.44, 0.02, -9.0), "bread": (0.35, 0.1, -5.0), "milk": (0.4, -0.1, 7.0),
                                        "ball_pink": (0.45, 0.05, 3.0)})
        self.assertEqual(out["start"], (0.44, 0.02, -9.0))
        self.assertEqual(primitive_name("ball", "pink"), "ball_pink")

    def test_nominal_keeps_the_layouts_own_start(self):
        out = resolve(layout(), NOMINAL)
        self.assertEqual((out["names"], out["start"], out["poses"]["can"]), (["can", "bread"], (0.3, 0.0, 10.0), (0.3, 0.0, 10.0)))


if __name__ == "__main__":
    unittest.main()
