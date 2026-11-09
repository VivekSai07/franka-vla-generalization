"""Layouts for tests: P2-style plan layouts with a fixed home jitter, made from P2's own sampler (read-only)."""
from experiments.props_p2_data import plan as p2


def make_layout(index, target="can", base=("bread",), tier="t"):
    identity = f"gen:test:{tier}:{index}"
    draw = p2._draw_layout(target, list(base), identity, None)
    return dict(layout_id=identity.replace(":", "_"), identity=identity, object=target, bin="blue", blue_slot="left",
                home_pose_jitter_deg=p2._jitter(identity), seed=p2.seed(f"{identity}:expert"), **draw)


def layouts(n=24):
    targets = list(p2.OBJECTS)
    out = []
    for i in range(n):
        target = targets[i % 6]
        others = [o for o in targets if o != target]
        base = others[(i // 6) % 4: (i // 6) % 4 + 1 + (i % 2)]
        out.append(make_layout(i, target, base))
    return out
