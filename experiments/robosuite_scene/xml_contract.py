import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robosuite.utils.mjcf_utils import new_element

from experiments.robosuite_scene.cameras import FRONT_CAM, GRIPPER_CAM

GRIPPER_BODY_NAME = "robot0_right_hand"

RENAMES = {
    "gripper0_right_leftfinger": "left_finger",
    "gripper0_right_rightfinger": "right_finger",
    "target_cube_red_main": "target_cube_red",
    "target_cube_green_main": "target_cube_green",
}
FINGER_TIPS = {
    "left_finger": "gripper0_right_finger_joint1_tip",
    "right_finger": "gripper0_right_finger_joint2_tip",
}
ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]
# (forcerange, gain) per arm joint, from src/robots/franka_panda/model/panda.xml actuators 1-7.
ARM_GAINS = [(87, 4500), (87, 4500), (87, 3500), (87, 3500), (12, 2000), (12, 2000), (12, 2000)]
CUBE_CONTACT = dict(
    mass="0.05", size="0.02 0.02 0.02", friction="1.2 0.005 0.0001",
    solref="0.002 1", solimp="0.9 0.95 0.001", margin="0.001",
)


def _need(element, what):
    if element is None:
        raise ValueError(f"robosuite XML is missing {what}")
    return element


def _by_name(root, tag, name):
    for element in root.iter(tag):
        if element.get("name") == name:
            return element
    raise ValueError(f"robosuite XML is missing {tag} {name!r}")


def _rewrite_model(root):
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

    for cube in ("target_cube_red", "target_cube_green"):
        body = _by_name(root, "body", cube)
        colliding = [g for g in body.findall("geom") if g.get("contype") != "0"]
        if len(colliding) != 1:
            raise ValueError(f"{cube} must have exactly one collision geom, found {len(colliding)}")
        geom = colliding[0]
        geom.attrib.pop("density", None)
        geom.attrib.update(CUBE_CONTACT)

    # Shared code detects grasp and finger contact by the body ids left_finger/right_finger, but
    # robosuite keeps each pad in a jointless child body. Re-parent the pad geoms, world pose kept;
    # the tip bodies and their inertials stay, so masses do not change.
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

    # The expert renders with MuJoCo default options (geom groups 0-2 drawn); robosuite keeps its
    # coloured collision geoms in group 0. Push them (and all sites) out so only visual geoms show.
    for geom in root.iter("geom"):
        if geom.get("group") in (None, "0"):
            geom.set("group", "3")
    for site in root.iter("site"):
        site.set("group", "4")


def apply_scripted_contract(xml):
    root = ET.fromstring(xml)
    _rewrite_model(root)
    worldbody = root.find("worldbody")
    existing = {camera.get("name") for camera in root.iter("camera")}

    # mujoco.Renderer needs the offscreen framebuffer sized for 512x512 up front.
    visual = root.find("visual")
    global_el = visual.find("global")
    if global_el is None:
        global_el = new_element("global", None, offheight="512", offwidth="512")
        visual.insert(0, global_el)
    else:
        global_el.set("offheight", "512")
        global_el.set("offwidth", "512")

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
