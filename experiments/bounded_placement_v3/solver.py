"""Finite box-constrained DLS; no unconstrained solve followed by clipping.

The inner convex quadratic uses cyclic exact coordinate minimization with box
projection at every coordinate and a projected KKT stopping test. Fixed damping
1e-3, 2048 sweeps and 12 backtracking trials are numerical implementation caps,
not automatic tuning/restarts. Pose tolerances remain independently enforced.
"""
import numpy as np
import pinocchio as pin


class PlanningFailure(ValueError):
    def __init__(self, gate, **details):
        super().__init__(gate)
        self.gate, self.details = gate, details


def finite(*values):
    if not all(np.isfinite(v).all() for v in values):
        raise PlanningFailure('nonfinite')


def box_quadratic(H, g, lower, upper):
    """Minimize .5*x.T*H*x+g.T*x subject to lower<=x<=upper."""
    H, g, lower, upper = [np.asarray(v, dtype=np.float64) for v in (H, g, lower, upper)]
    finite(H, g, lower, upper)
    n = len(g)
    if (g.shape != (n,) or H.shape != (n, n) or lower.shape != (n,) or upper.shape != (n,)
            or np.any(lower > upper) or not np.allclose(H, H.T, rtol=0, atol=1e-14)):
        raise PlanningFailure('invalid_quadratic')
    try:
        np.linalg.cholesky(H)
    except np.linalg.LinAlgError as exc:
        raise PlanningFailure('nonpositive_quadratic') from exc
    x = np.clip(np.zeros(n), lower, upper)
    for _ in range(2048):
        for j in range(n):
            gradient_j = H[j] @ x + g[j]
            x[j] = np.clip(x[j] - gradient_j/H[j, j], lower[j], upper[j])
        gradient = H @ x + g
        kkt = x - np.clip(x-gradient, lower, upper)
        if np.max(np.abs(kkt)) <= 1e-10:
            return x
    raise PlanningFailure('quadratic_iteration_limit', candidate=x.copy(), kkt=float(np.max(np.abs(kkt))))


def pose_errors(robot, q, target):
    pose = robot.forward_kinematics(q).copy()
    delta = pose.inverse()*target
    residual = pin.log6(delta).vector.copy()
    position = float(np.linalg.norm(pose.translation-target.translation))
    orientation = float(np.linalg.norm(pin.log3(pose.rotation.T@target.rotation)))
    finite(residual, position, orientation)
    return residual, delta, position, orientation


def solve_waypoint(robot, seed, target, bounds, journal):
    q = np.asarray(seed, dtype=np.float64).copy()
    bounds = np.asarray(bounds, dtype=np.float64)
    finite(q, bounds, target.homogeneous)
    if bounds.shape != (7, 2) or q.shape != (robot.model.nq,):
        raise PlanningFailure('invalid_ik_shape')
    lo, hi = bounds[:, 0]+1e-6, bounds[:, 1]-1e-6
    if np.any(lo >= hi) or np.any(q[:7] < lo) or np.any(q[:7] > hi):
        raise PlanningFailure('ik_seed_bounds', q=q.copy())
    scale = np.array([1., 1., 1., .2, .2, .2])
    for iteration in range(101):
        residual, delta, position, orientation = pose_errors(robot, q, target)
        r = scale*residual
        score = float(r@r/2)
        row = dict(iteration=iteration, q=q.copy(), position_error_m=position,
                   orientation_error_rad=orientation, objective=score, backtracking=[])
        journal.append(row)
        if position <= .001 and orientation <= .01:
            return q.copy()
        if iteration == 100:
            raise PlanningFailure('ik_iteration_limit', position_error_m=position, orientation_error_rad=orientation)
        J = -pin.Jlog6(delta.inverse()) @ robot.jacobian(q)[:, :7]
        A = scale[:, None]*J
        gradient = A.T@r
        lower, upper = np.maximum(lo-q[:7], -.05), np.minimum(hi-q[:7], .05)
        step = box_quadratic(A.T@A+1e-3*np.eye(7), gradient, lower, upper)
        accepted = False
        for k in range(12):
            trial = q.copy()
            trial[:7] += (0.5**k)*step
            # No clipping: the optimization and convex backtrack enforce bounds.
            finite(trial)
            if np.any(trial[:7] < lo) or np.any(trial[:7] > hi):
                raise PlanningFailure('ik_numerical_bounds')
            error = scale*pose_errors(robot, trial, target)[0]
            trial_score = float(error@error/2)
            row['backtracking'].append(dict(q=trial.copy(), objective=trial_score))
            if trial_score < score and trial_score <= score+1e-4*float(gradient@(trial[:7]-q[:7])):
                q, accepted = trial, True
                break
        if not accepted:
            raise PlanningFailure('ik_backtracking_failure', position_error_m=position, orientation_error_rad=orientation)