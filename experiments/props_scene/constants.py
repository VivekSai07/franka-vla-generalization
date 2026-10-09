"""Every number the props P0 spec fixes. Pure: imports no simulator."""
import hashlib

VERSION = "props_p0_v1"
MASTER = 20261005

OBJECTS = ("can", "bread", "lemon", "milk", "bottle", "cereal")
MASS_KG = {"can": 0.15, "bread": 0.05, "lemon": 0.08, "milk": 0.20, "bottle": 0.20, "cereal": 0.15}

# Left and right are from the robot's point of view (it faces +x): left is y > 0. In front_cam +y is image-right.
BIN_INNER_M = 0.22
BIN_WALL_HEIGHT_M = 0.07
BIN_WALL_THICKNESS_M = 0.01
BIN_FLOOR_THICKNESS_M = 0.01
SLOT_XY = {"left": (0.32, 0.33), "right": (0.32, -0.33)}
BIN_RGBA = {"blue": (0.15, 0.35, 0.85, 1.0), "orange": (0.95, 0.55, 0.10, 1.0)}
BINS = tuple(BIN_RGBA)

START_X_M = (0.27, 0.40)    # amended 2026-10-05; was 0.22, inside the arm's working envelope (spec amendment)
START_Y_M = (-0.12, 0.12)
START_YAW_DEG = (-90.0, 90.0)
# amended 2026-10-06: the x >= 0.27 rectangle still admits a near-corner band the arm cannot reach along
# the straight-line approach; drawn starts are also rejected below this radius.
START_RADIUS_MIN_M = 0.285
# amended 2026-10-06: near +/-90 deg the lemon's grasp axis nearly aligns with its long (7 cm) side, leaving
# the fingers almost no margin (a measured, narrow failure mode, not a reach problem).
YAW_MAX_DEG_BY_OBJECT = {name: 90.0 for name in OBJECTS}
YAW_MAX_DEG_BY_OBJECT["lemon"] = 75.0

HAND_OFFSET_M = 0.105       # EE frame above the pad centre; the sealed cube expert's value
GRASP_BELOW_TOP_M = 0.03    # grasp point is min(this, height / 2) below the object's top
LIFT_CLEARANCE_M = 0.05     # object bottom over the wall top while carrying
RELEASE_ABOVE_WALL_M = 0.02 # object bottom over the wall top at release
APPROACH_HEIGHT_M = 0.15    # hover above the grasp point
RETRACT_HEIGHT_M = 0.15     # hand rise after release
HOME_JITTER_DEG_MAX = 1.0
TARGET_JITTER_M = 0.005
LIFT_VERIFY_FRACTION = 0.5  # the object must have risen this fraction of the planned lift

FPS = 30
SUBSTEPS = 20
SUCCESS_TICKS = 30
MAX_TICKS = 900
GRIPPER_OPEN_CTRL = 127
CAMERA_SIZE = 512

PHASE_NAMES = ("approach", "descend", "close", "lift", "move", "lower", "open", "retract", "home")
PHASE_DURATION_S = {"approach": 2.0, "descend": 1.5, "close": 1.0, "lift": 1.5, "move": 2.5,
                    "lower": 1.5, "open": 1.0, "retract": 1.5, "home": 2.5}

# Contact parameters copied from the cube (robosuite_scene.xml_contract.CUBE_CONTACT) except mass.
PROP_CONTACT = dict(solref="0.002 1", solimp="0.9 0.95 0.001", margin="0.001")
PROP_FRICTION = {name: (1.2, 0.005, 0.0001) for name in OBJECTS}

GATE_ATTEMPTS = 288
GATE_SEEDS_PER_CELL = 12
GATE_OVERALL = 0.95
GATE_PER_OBJECT = 0.90
GATE_PER_SLOT = 0.93


def seed(identity):
    return int.from_bytes(hashlib.sha256(f"{VERSION}:{MASTER}:{identity}".encode()).digest()[:4], "big")
