"""Compose the props scene under a `Condition`; robosuite image (MuJoCo 3.4.0).

The scene is P2's: `props_scene.scene.PropsEnv` composes it once, `props_scene.xml_contract.apply_props_contract` rewrites its
XML, and plain MuJoCo steps it (`env.step` stays banned). Everything a condition changes is applied to that XML before compiling
(textures, table height, bin positions, primitive distractors) or to the compiled model (light values), so under the nominal
condition the XML is the very string `props_scene` compiles and the model, its lights and its renders are P2's (gate D9 i).

The robosuite XML must be read right after `env.reset()`: a later `get_xml()` returns the last model MuJoCo loaded, which is no
longer robosuite's. It is cached per (props, slot); compiled models are cached per scene and the cache is bounded (a model is
large, and a rebuild is bit-identical).

Table height: the table's top, both bins and the props' drop height move together by `table_offset_m`; the slab always reaches
down to the floor, and for a table below nominal the floor drops with it, so no gap and no coplanar faces appear.
"""
import functools
import gc
import io
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import robosuite
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"

from experiments.gen_scene import pools
from experiments.gen_scene.conditions import primitive_name, resolve
from experiments.props_scene import constants as c
from experiments.props_scene import scene as props_scene
from experiments.props_scene.xml_contract import apply_props_contract

MAX_CACHED_MODELS = 2
NOMINAL_FLOOR_Z = -0.05
WALLS = ("wall_leftcorner_visual", "wall_rightcorner_visual", "wall_left_visual", "wall_right_visual", "wall_rear_visual",
         "wall_front_visual")
TEXTURE_REPEAT = {"table": ("false", "1 1"), "floor": ("true", "2 2"), "wall": ("true", "2 2")}
LAY_ALONG_X = "0.7071067811865476 0 0.7071067811865476 0"   # a z-axis capsule turned onto the x axis (the XML is in radians)
_MODELS = {}


def _num(value):
    return repr(float(value))


@functools.lru_cache(maxsize=64)
def contract_xml(props, blue_slot):
    """The compiled-scene XML of `props_scene` for these props, exactly as `props_scene._built` compiles it."""
    env = robosuite.make(
        "PropsEnv", objects=props, blue_slot=blue_slot, robots="Panda", base_types="RethinkMinimalMount",
        has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False, control_freq=c.FPS,
        hard_reset=False, renderer="mujoco", initialization_noise={"magnitude": 0.0, "type": "uniform"})
    env.reset()
    xml = apply_props_contract(env.sim.model.get_xml())
    env.close()
    return xml


@functools.lru_cache(maxsize=256)
def texture_png(name):
    """PNG bytes of a manifest texture (MuJoCo reads PNG only; the maps are JPEG)."""
    from PIL import Image
    with Image.open(pools.texture_path(name)) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _by_name(root, tag, name):
    found = [e for e in root.iter(tag) if e.get("name") == name]
    if len(found) != 1:
        raise ValueError(f"expected one <{tag}> named {name}, found {len(found)}")
    return found[0]


def _surface(root, assets, key, texture, geoms, material_defaults):
    """Give `geoms` a material of this texture."""
    uniform, repeat = TEXTURE_REPEAT[key]
    file_name = f"gen_{key}.png"
    assets[file_name] = texture_png(texture)
    asset = root.find("asset")
    ET.SubElement(asset, "texture", type="2d", name=f"gen_{key}_tex", file=file_name, colorspace="sRGB")
    ET.SubElement(asset, "material", name=f"gen_{key}_mat", texture=f"gen_{key}_tex", texuniform=uniform, texrepeat=repeat,
                  **material_defaults)
    for geom in geoms:
        geom.set("material", f"gen_{key}_mat")


def _table(root, offset):
    floor_z = offset - 0.05 if offset < 0 else NOMINAL_FLOOR_Z
    half = (offset - floor_z) / 2
    body = _by_name(root, "body", "table")
    x, y, _ = (float(v) for v in body.get("pos").split())
    body.set("pos", f"{_num(x)} {_num(y)} {_num((offset + floor_z) / 2)}")
    for name in ("table_collision", "table_visual"):
        geom = _by_name(root, "geom", name)
        sx, sy, _ = (float(v) for v in geom.get("size").split())
        geom.set("size", f"{_num(sx)} {_num(sy)} {_num(half)}")
    floor = _by_name(root, "geom", "floor")
    fx, fy, _ = (float(v) for v in floor.get("pos").split())
    floor.set("pos", f"{_num(fx)} {_num(fy)} {_num(floor_z)}")


def _bins(root, offset, shift, blue_slot):
    for colour in c.BINS:
        slot = blue_slot if colour == "blue" else ("right" if blue_slot == "left" else "left")
        body = _by_name(root, "body", f"bin_{colour}")
        x, y = c.SLOT_XY[slot]
        sign = 1.0 if y > 0 else -1.0
        body.set("pos", f"{_num(x)} {_num(y + sign * shift)} {_num(offset)}")


def _primitive_bodies(root, primitives):
    world = root.find("worldbody")
    for shape, colour in primitives:
        kind, size = next(t[shape] for t in pools.PRIMITIVE_SHAPES.values() if shape in t)
        name = f"prop_{primitive_name(shape, colour)}"
        body = ET.SubElement(world, "body", name=name, pos="0 0 0")
        ET.SubElement(body, "joint", name=f"{name}_joint", type="free")
        attrs = dict(name=f"{name}_geom", type=kind, size=" ".join(_num(v) for v in size),
                     rgba=" ".join(_num(v) for v in (*pools.PRIMITIVE_COLOURS["A" if colour in pools.PRIMITIVE_COLOURS["A"] else "B"][colour], 1.0)),
                     mass=_num(pools.PRIMITIVE_MASS_KG), friction=" ".join(_num(v) for v in pools.PRIMITIVE_FRICTION),
                     solref=c.PROP_CONTACT["solref"], solimp=c.PROP_CONTACT["solimp"], margin=c.PROP_CONTACT["margin"],
                     group="1", contype="1", conaffinity="1")
        if kind in pools.PRIMITIVE_LYING:
            attrs["quat"] = LAY_ALONG_X
        ET.SubElement(body, "geom", **attrs)


def edit_xml(xml, primitives, blue_slot, condition):
    """(xml, assets) for the condition; the input string itself, with no assets, when nothing in the scene changes."""
    scene_changes = (condition.table_texture or condition.floor_texture or condition.wall_texture or condition.table_offset_m
                     or condition.bin_shift_m or primitives)
    if not scene_changes:
        return xml, {}
    root, assets = ET.fromstring(xml), {}
    if condition.table_offset_m:
        _table(root, condition.table_offset_m)
    if condition.table_offset_m or condition.bin_shift_m:
        _bins(root, condition.table_offset_m, condition.bin_shift_m, blue_slot)
    if condition.table_texture:
        _surface(root, assets, "table", condition.table_texture, [_by_name(root, "geom", "table_visual")],
                 dict(specular="0.2", shininess="0"))
    if condition.floor_texture:
        _surface(root, assets, "floor", condition.floor_texture, [_by_name(root, "geom", "floor")], dict(specular="0", shininess="0"))
    if condition.wall_texture:
        _surface(root, assets, "wall", condition.wall_texture, [_by_name(root, "geom", n) for n in WALLS],
                 dict(specular="0.1", shininess="0.1"))
    _primitive_bodies(root, primitives)
    return ET.tostring(root, encoding="unicode"), assets


def apply_lights(model, condition):
    """Scale and tint the compiled model's lights and headlight; set the scene light's direction and shadows."""
    if (condition.light_diffuse_scale, condition.light_specular_scale, condition.ambient_scale, condition.headlight_scale,
            condition.tint, condition.light_dir, condition.light_shadows) == (1.0, 1.0, 1.0, 1.0, (1.0, 1.0, 1.0), None, False):
        return model
    if model.nlight != 1:
        raise ValueError(f"expected the arena's one light, found {model.nlight}")
    tint = np.asarray(condition.tint, dtype=np.float64)
    for i in range(model.nlight):
        model.light_diffuse[i] = model.light_diffuse[i] * condition.light_diffuse_scale * tint
        model.light_specular[i] = model.light_specular[i] * condition.light_specular_scale * tint
        model.light_ambient[i] = model.light_ambient[i] * condition.ambient_scale * tint
    headlight = model.vis.headlight
    h = condition.headlight_scale
    headlight.diffuse[:] = np.asarray(headlight.diffuse, dtype=np.float64) * condition.light_diffuse_scale * h * tint
    headlight.specular[:] = np.asarray(headlight.specular, dtype=np.float64) * condition.light_specular_scale * h * tint
    headlight.ambient[:] = np.asarray(headlight.ambient, dtype=np.float64) * condition.ambient_scale * h * tint
    if condition.light_dir is not None:
        model.light_dir[0] = np.asarray(condition.light_dir, dtype=np.float64)
    if condition.light_shadows:
        model.light_castshadow[0] = 1
    return model


def scene_key(props, primitives, blue_slot, condition):
    """What the compiled model depends on (not the start, move or disturbance, which only the episode uses)."""
    fields = (condition.light_diffuse_scale, condition.light_specular_scale, condition.ambient_scale, condition.headlight_scale,
              tuple(condition.tint), condition.light_dir, condition.light_shadows, condition.table_texture,
              condition.floor_texture, condition.wall_texture, condition.table_offset_m, condition.bin_shift_m)
    return (tuple(sorted(props, key=c.OBJECTS.index)), tuple(primitives), blue_slot, fields)


def clear_cache():
    _MODELS.clear()
    gc.collect()


def build_model(props, primitives, blue_slot, condition):
    """The compiled, timed, lit model of the scene; cached (at most MAX_CACHED_MODELS)."""
    from data_collection.benchmark import configure_timing
    condition.check()
    props = tuple(sorted(props, key=c.OBJECTS.index))
    props_scene._check(props, blue_slot)
    key = scene_key(props, primitives, blue_slot, condition)
    if key not in _MODELS:
        if len(_MODELS) >= MAX_CACHED_MODELS:
            clear_cache()
        xml, assets = edit_xml(contract_xml(props, blue_slot), tuple(primitives), blue_slot, condition)
        model = mujoco.MjModel.from_xml_string(xml, assets) if assets else mujoco.MjModel.from_xml_string(xml)
        configure_timing(model, c.FPS)
        _MODELS[key] = apply_lights(model, condition)
    return _MODELS[key]


def table_top_z(model):
    """World height of the table's top: its body height plus the collision slab's half thickness."""
    return float(model.body("table").pos[2] + model.geom("table_collision").size[2])


def model_for(layout, condition):
    """(model, resolved scene) of a plan layout under a condition."""
    resolved = resolve(layout, condition)
    return build_model(resolved["props"], resolved["primitives"], resolved["blue_slot"], condition), resolved


def _qpos_address(model, name):
    return int(model.joint(int(model.body(f"prop_{name}").jntadr[0])).qposadr[0])


def reset(model, poses, home_jitter_deg):
    """Arm at Q_HOME plus jitter, every object upright at (x, y, yaw_deg) on the table, settled 0.2 s; no RNG.
    `props_scene.reset` with the table's height added, and primitives (bodies `prop_<shape>_<colour>`) placed by their height."""
    from data_collection.benchmark import capture_state, synchronize_observation
    from experiments.props_scene.geometry import bbox
    from src.robots.franka_panda.config import Q_HOME

    present = props_scene.present_objects(model)
    primitives = [n for n in poses if n not in c.OBJECTS]
    if sorted(poses) != sorted([*present, *primitives]) or any(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"prop_{n}") < 0 for n in primitives):
        raise ValueError(f"poses for {sorted(poses)} do not match the model's objects {sorted(present)} + {sorted(primitives)}")
    if len(home_jitter_deg) != 7:
        raise ValueError("home_jitter_deg must have 7 values")
    table_z = table_top_z(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    for name, (x, y, yaw_deg) in poses.items():
        if name in c.OBJECTS:
            centre, half = bbox(model, name)
            rest = half[2] - centre[2]
        else:
            rest = pools.primitive_extent(name.split("_")[0])[1] / 2
        yaw = np.deg2rad(yaw_deg)
        adr = _qpos_address(model, name)
        data.qpos[adr:adr + 3] = (x, y, table_z + (rest + props_scene.DROP_HEIGHT_M))
        data.qpos[adr + 3:adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    home = np.asarray(Q_HOME[:7], dtype=np.float64) + np.deg2rad(np.asarray(home_jitter_deg, dtype=np.float64))
    data.qpos[:9] = Q_HOME
    data.qpos[:7] = home
    data.ctrl[:] = 0.0
    data.ctrl[:7] = home
    data.ctrl[-1] = 255
    mujoco.mj_forward(model, data)
    for _ in range(round(props_scene.SETTLE_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)
    synchronize_observation(model, data)
    return {"qpos": data.qpos.copy(), "initial_state": capture_state(model, data)}
