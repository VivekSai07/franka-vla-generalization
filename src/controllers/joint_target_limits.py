"""Position/velocity-aware command integration before actuator send and recording."""
import numpy as np


def bounded_joint_target(previous, requested_velocity, dt, position_bounds, max_velocity=2.):
    previous = np.asarray(previous, dtype=np.float64)
    requested = np.asarray(requested_velocity, dtype=np.float64)
    bounds = np.asarray(position_bounds, dtype=np.float64)
    if (previous.ndim != 1 or requested.shape != previous.shape or bounds.shape != (len(previous), 2)
            or not all(np.isfinite(x).all() for x in (previous, requested, bounds))
            or not np.isfinite(dt) or dt <= 0 or not np.isfinite(max_velocity) or max_velocity <= 0
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("Invalid bounded target integration inputs")
    if np.any(previous < bounds[:, 0]) or np.any(previous > bounds[:, 1]):
        raise ValueError("Previous commanded target is already outside limits")
    # Maintain a 1e-6-radian interior margin to keep float32 recorded commands
    # within actuator bounds. It is part of command generation, not label repair.
    margin = np.minimum(1e-6, (bounds[:, 1] - bounds[:, 0]) / 4)
    lower = np.maximum(-max_velocity, (bounds[:, 0] + margin - previous) / dt)
    upper = np.minimum(max_velocity, (bounds[:, 1] - margin - previous) / dt)
    if np.any(lower > upper):
        raise ValueError("No feasible command increment within the velocity bound")
    applied = np.clip(requested, lower, upper)
    target = previous + applied * dt
    if np.any(target < bounds[:, 0]) or np.any(target > bounds[:, 1]):
        raise ValueError("Numerical target integration escaped bounds")
    return target, applied, np.abs(applied - requested) > 1e-12