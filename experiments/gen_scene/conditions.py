"""The condition of a scene: every departure from P2's nominal scene, as one JSON-serializable record. Pure; imports no simulator.

`Condition` is what G0 applies at test time (`pool="test"`) and what G1/G2 sample at training time (`pool="train"`). The
nominal condition changes nothing: `compose.build` must then produce P2's scene byte for byte (G0 spec D9 gate i).
A condition names the pool its textures and primitive distractors come from, and `check` refuses any texture or primitive that
is not in that pool, so a test texture can never reach a training scene or the reverse.

Units: metres and degrees; positions are table-frame (x forward from the robot, y left), the plan's own.
"""
import dataclasses
import hashlib
import json
import math

from experiments.gen_scene import pools

VERSION = "gen_scene_v1"
POOL_NAMES = ("none", "train", "test")
PRIMITIVE_POOL = {"train": "A", "test": "B"}
DISTURBANCES = ("none", "move", "drop")
TABLE_OFFSET_RANGE_M = (-0.10, 0.20)   # GraspVLA's range; G0 tests +-5 cm and +10 cm
BIN_SHIFT_MAX_M = 0.10


@dataclasses.dataclass(frozen=True)
class Condition:
    axis: str = "nominal"
    level: str = "nominal"
    pool: str = "none"
    # Lights. Scales multiply the compiled model's values; `tint` multiplies diffuse, specular and ambient of every light.
    light_diffuse_scale: float = 1.0     # every light's diffuse, the headlight's included
    light_specular_scale: float = 1.0    # every light's specular
    ambient_scale: float = 1.0           # every light's ambient
    headlight_scale: float = 1.0         # extra factor on the headlight's diffuse, specular and ambient only
    tint: tuple = (1.0, 1.0, 1.0)
    light_dir: tuple = None              # replaces the scene light's direction (unit vector, travel direction)
    light_shadows: bool = False
    # Surfaces (names of manifest textures of the pool).
    table_texture: str = None
    floor_texture: str = None
    wall_texture: str = None
    # Geometry.
    table_offset_m: float = 0.0          # table top, bins and the props' drop height move together
    bin_shift_m: float = 0.0             # both bins, along y
    extra_props: tuple = ()              # ((name, x, y, yaw_deg), ...) props of the six, added to the layout's distractors
    primitives: tuple = ()               # ((shape, colour, x, y, yaw_deg), ...) from the pool's primitive set
    start: tuple = None                  # (x, y, yaw_deg) replaces the layout's target start
    # Disturbance during the episode.
    disturbance: str = "none"
    move_to: tuple = None                # (x, y): where `move` puts the target

    def to_dict(self):
        def plain(value):
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            return value
        return {f.name: plain(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown condition fields {sorted(unknown)}")

        def frozen(value):
            return tuple(frozen(v) for v in value) if isinstance(value, list) else value
        return cls(**{k: frozen(v) for k, v in data.items()})

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def is_nominal(self):
        return self == NOMINAL

    def check(self, manifest=None):
        """Raises ValueError unless this condition is well formed and every texture and primitive is in its pool."""
        if self.pool not in POOL_NAMES:
            raise ValueError(f"pool must be one of {POOL_NAMES}")
        if self.disturbance not in DISTURBANCES:
            raise ValueError(f"disturbance must be one of {DISTURBANCES}")
        for name in ("light_diffuse_scale", "light_specular_scale", "ambient_scale", "headlight_scale"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if len(self.tint) != 3 or not all(0 < v <= 1.0 for v in self.tint):
            raise ValueError("tint must be three values in (0, 1]")
        if self.light_dir is not None:
            if len(self.light_dir) != 3 or abs(math.sqrt(sum(v * v for v in self.light_dir)) - 1.0) > 1e-6:
                raise ValueError("light_dir must be a unit vector")
            if self.light_dir[2] >= 0:
                raise ValueError("light_dir must point downward")
        low, high = TABLE_OFFSET_RANGE_M
        if not low <= self.table_offset_m <= high:
            raise ValueError(f"table_offset_m must be within {TABLE_OFFSET_RANGE_M}")
        if abs(self.bin_shift_m) > BIN_SHIFT_MAX_M:
            raise ValueError(f"|bin_shift_m| must be at most {BIN_SHIFT_MAX_M}")
        if (self.disturbance == "move") != (self.move_to is not None):
            raise ValueError("move_to is set exactly when the disturbance is `move`")
        used = [t for t in (self.table_texture, self.floor_texture, self.wall_texture) if t is not None]
        if used or self.primitives:
            if self.pool == "none":
                raise ValueError("a condition with textures or primitives must declare pool `train` or `test`")
        if used:
            both = pools.pools(manifest)
            allowed = {"table": set(both["table"][self.pool]), "room": set(both["room"][self.pool])}
            for surface, names in (("table", [self.table_texture]), ("room", [self.floor_texture, self.wall_texture])):
                for name in names:
                    if name is not None and name not in allowed[surface]:
                        raise ValueError(f"{name} is not in the {self.pool} {surface} pool")
        for shape, colour, *_ in self.primitives:
            pool = PRIMITIVE_POOL[self.pool]
            if shape not in pools.PRIMITIVE_SHAPES[pool] or colour not in pools.PRIMITIVE_COLOURS[pool]:
                raise ValueError(f"primitive {shape}/{colour} is not in primitive pool {pool} (condition pool {self.pool})")
        names = [n for n, *_ in self.extra_props]
        if len(set(names)) != len(names):
            raise ValueError("an extra prop appears twice")
        if len({(s, c) for s, c, *_ in self.primitives}) != len(self.primitives):
            raise ValueError("a primitive (shape, colour) appears twice")
        return self


NOMINAL = Condition()


def primitive_name(shape, colour):
    return f"{shape}_{colour}"


def resolve(layout, condition):
    """What a trial puts in the scene: the target first, then the layout's distractors, the condition's extra props and its
    primitives, with every pose. `layout` is a plan layout (`object`, `bin`, `blue_slot`, `start`, `distractors`)."""
    props = [layout["object"], *(d["object"] for d in layout["distractors"]), *(p[0] for p in condition.extra_props)]
    primitives = [primitive_name(s, c) for s, c, *_ in condition.primitives]
    start = tuple(condition.start) if condition.start is not None else tuple(layout["start"])
    poses = {layout["object"]: start}
    poses.update({d["object"]: tuple(d["pose"]) for d in layout["distractors"]})
    poses.update({n: (x, y, yaw) for n, x, y, yaw in condition.extra_props})
    poses.update({primitive_name(s, c): (x, y, yaw) for s, c, x, y, yaw in condition.primitives})
    return dict(names=props + primitives, props=props, primitives=[(s, c) for s, c, *_ in condition.primitives],
                primitive_names=primitives, poses=poses, start=start, blue_slot=layout["blue_slot"])
