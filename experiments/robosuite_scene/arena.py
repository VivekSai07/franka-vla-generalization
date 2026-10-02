import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"

from robosuite.models.arenas.table_arena import TableArena
from robosuite.utils.mjcf_utils import new_element

from experiments.robosuite_scene.cameras import FRONT_CAM


class CopiedCameraTableArena(TableArena):
    def __init__(self):
        super().__init__(
            table_full_size=(1.2, 1.2, 0.05),
            table_friction=(1.0, 0.005, 0.0001),
            table_offset=(0.40, 0.0, 0.0),
            has_legs=False,
        )
        self.floor.set("pos", "0 0 -0.05")
        self.worldbody.append(new_element("camera", **FRONT_CAM))
