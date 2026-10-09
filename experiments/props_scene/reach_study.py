"""Nearest radius a vertical hand reaches at several heights, by numerical IK with the real Panda joint limits.

    python -m experiments.props_scene.reach_study       (admission or robosuite image; no simulator)
"""
import numpy as np
import pinocchio as pin
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import Q_HOME

robot = FrankaPanda()
m = robot.model
# the real Panda limits that robosuite's model (and so the actuator ctrlrange the expert clips to) uses
lo = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
hi = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
print("pinocchio URDF limits:", m.lowerPositionLimit[:7].round(3), m.upperPositionLimit[:7].round(3))
q_home = pin.neutral(m)
q_home[:9] = Q_HOME
R_home = robot.forward_kinematics(q_home).rotation.copy()


def solve(target, q0):
    q = pin.neutral(m)
    q[:9] = Q_HOME
    q[:7] = q0
    for _ in range(400):
        T = robot.forward_kinematics(q)
        err = pin.log6(T.inverse() * target).vector
        if np.linalg.norm(err[:3]) < 5e-4 and np.linalg.norm(err[3:]) < 0.01:
            return True
        J = robot.jacobian(q)[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        q[:7] = np.clip(q[:7] + 0.5 * dq, lo, hi)
    return False


def reachable(pos, R_t, tries=60, seed=0):
    rng = np.random.RandomState(seed)
    target = pin.SE3(R_t, pos)
    for t in range(tries):
        q0 = np.asarray(Q_HOME[:7]) if t == 0 else rng.uniform(lo, hi)
        if solve(target, q0):
            return True
    return False


for z in (0.129, 0.155, 0.185, 0.225, 0.3):
    ok_radius = None
    for r in np.arange(0.36, 0.15, -0.005):
        if reachable(np.array([r, 0.0, z]), R_home):
            ok_radius = r
        else:
            break
    print(f"hand z {z:.3f}: nearest radius a vertical hand reaches (y=0): {ok_radius:.3f}")
