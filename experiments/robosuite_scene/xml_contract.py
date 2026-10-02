import xml.etree.ElementTree as ET

from robosuite.utils.mjcf_utils import new_element

from experiments.robosuite_scene.cameras import FRONT_CAM, GRIPPER_CAM

GRIPPER_BODY_NAME = "robot0_right_hand"


def apply_scripted_contract(xml):
    root = ET.fromstring(xml)
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
