"""Rewrite robosuite's compiled XML to the repo's robot contract and set the props' physics.

The robot half reproduces `experiments/robosuite_scene/xml_contract._rewrite_model` (sealed by B3, so
not edited and not callable: it requires cube bodies). tests/test_scene.py checks the result against
the current MuJoCo scene.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robosuite.utils.mjcf_utils import new_element

from experiments.props_scene import constants as c
from experiments.robosuite_scene.cameras import FRONT_CAM, GRIPPER_CAM
from experiments.robosuite_scene.xml_contract import (
    ARM_GAINS, ARM_JOINTS, FINGER_JOINTS, FINGER_TIPS, GRIPPER_BODY_NAME, RENAMES, _by_name, _need,
)


def _robot(root):
    for old, new in RENAMES.items():
        _by_name(root, "body", old).set("name", new)

    finger2 = _by_name(root, "joint", FINGER_JOINTS[1])
    finger2.set("axis", "0 -1 0")
    finger2.set("range", "0 0.04")
    for name in ARM_JOINTS + FINGER_JOINTS:
        joint = _by_name(root, "joint", name)
        joint.set("armature", "0.1")
        joint.set("damping", "1")
        joint.set("frictionloss", "0")

    _need(root.find("option"), "<option>").set("integrator", "implicitfast")

    actuator = _need(root.find("actuator"), "<actuator>")
    arm_ranges = [_by_name(root, "joint", name).get("range") for name in ARM_JOINTS]
    for child in list(actuator):
        actuator.remove(child)
    common = dict(biastype="affine", ctrllimited="true", forcelimited="true")
    for i, (name, rng, (force, gain)) in enumerate(zip(ARM_JOINTS, arm_ranges, ARM_GAINS), start=1):
        ET.SubElement(actuator, "general", name=f"actuator{i}", joint=name, ctrlrange=rng,
                      forcerange=f"-{force} {force}", gainprm=str(gain),
                      biasprm=f"0 -{gain} -{gain // 10}", **common)
    ET.SubElement(actuator, "general", name="actuator8", tendon="split", ctrlrange="0 255",
                  forcerange="-100 100", gainprm="0.07843137255 0 0", biasprm="0 -500 -50", **common)

    tendon = ET.Element("tendon")
    fixed = ET.SubElement(tendon, "fixed", name="split")
    for name in FINGER_JOINTS:
        ET.SubElement(fixed, "joint", joint=name, coef="0.5")
    equality = ET.Element("equality")
    ET.SubElement(equality, "joint", joint1=FINGER_JOINTS[0], joint2=FINGER_JOINTS[1],
                  solimp="0.95 0.99 0.001", solref="0.005 1")
    index = list(root).index(actuator)
    root.insert(index, equality)
    root.insert(index, tendon)

    # Shared code detects grasp and finger contact by the body ids left_finger/right_finger, but
    # robosuite keeps each pad in a jointless child body. Re-parent the pad geoms, world pose kept.
    for finger, tip in FINGER_TIPS.items():
        finger_body = _by_name(root, "body", finger)
        tip_body = _by_name(finger_body, "body", tip)
        pads = [g for g in tip_body.findall("geom") if g.get("contype") != "0" or g.get("conaffinity") != "0"]
        if not pads:
            raise ValueError(f"{tip} has no collision geom")
        tip_pos = np.array(tip_body.get("pos", "0 0 0").split(), dtype=float)
        tip_quat = np.array(tip_body.get("quat", "1 0 0 0").split(), dtype=float)
        for geom in pads:
            pos, quat = np.zeros(3), np.zeros(4)
            mujoco.mju_rotVecQuat(pos, np.array(geom.get("pos", "0 0 0").split(), dtype=float), tip_quat)
            mujoco.mju_mulQuat(quat, tip_quat, np.array(geom.get("quat", "1 0 0 0").split(), dtype=float))
            geom.set("pos", " ".join(repr(float(v)) for v in pos + tip_pos))
            geom.set("quat", " ".join(repr(float(v)) for v in quat))
            tip_body.remove(geom)
            finger_body.append(geom)


def _props(root):
    found = []
    for body in root.iter("body"):
        name = body.get("name") or ""
        if name.endswith("_main") and name[:-len("_main")] in c.OBJECTS:
            found.append((name[:-len("_main")], body))
    if not found:
        raise ValueError("robosuite XML has no <object>_main prop body")
    for obj, body in found:
        body.set("name", f"prop_{obj}")
        colliding = [g for g in body.findall("geom") if g.get("contype") != "0"]
        if len(colliding) != 1:
            raise ValueError(f"{obj} must have exactly one collision geom, found {len(colliding)}")
        geom = colliding[0]
        geom.attrib.pop("density", None)
        geom.attrib.update(c.PROP_CONTACT)
        geom.set("mass", repr(c.MASS_KG[obj]))
        geom.set("friction", " ".join(repr(v) for v in c.PROP_FRICTION[obj]))
    return [obj for obj, _ in found]


def apply_props_contract(xml):
    root = ET.fromstring(xml)
    _robot(root)
    _props(root)

    # The renderer draws groups 0-2; robosuite keeps coloured collision geoms in group 0.
    for geom in root.iter("geom"):
        if geom.get("group") in (None, "0"):
            geom.set("group", "3")
    for site in root.iter("site"):
        site.set("group", "4")

    worldbody = root.find("worldbody")
    existing = {camera.get("name") for camera in root.iter("camera")}

    visual = root.find("visual")
    global_el = visual.find("global")
    if global_el is None:
        global_el = new_element("global", None, offheight=str(c.CAMERA_SIZE), offwidth=str(c.CAMERA_SIZE))
        visual.insert(0, global_el)
    else:
        global_el.set("offheight", str(c.CAMERA_SIZE))
        global_el.set("offwidth", str(c.CAMERA_SIZE))

    if FRONT_CAM["name"] not in existing:
        worldbody.append(new_element("camera", **FRONT_CAM))
    if GRIPPER_CAM["name"] not in existing:
        target = worldbody
        for body in root.iter("body"):
            if body.get("name") == GRIPPER_BODY_NAME:
                target = body
                break
        target.append(new_element("camera", **GRIPPER_CAM))
    return ET.tostring(root, encoding="unicode")
