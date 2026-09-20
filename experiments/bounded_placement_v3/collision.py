"""Enabled MuJoCo geometry on scratch state, with a measured rigid payload.

This uses MuJoCo contact filters, exclusions and convex mesh representation.
Sampling is discrete, not a continuous collision or force/grasp-stability proof.
"""
import math
import mujoco
import numpy as np
import pinocchio as pin

from data_collection.benchmark import capture_state, restore_state
from .solver import PlanningFailure, finite


def body_pose(model, data, name):
    body = model.body(name).id
    return pin.SE3(data.xmat[body].reshape(3, 3).copy(), data.xpos[body].copy())


def unexpected_contact(bodies, distance, robot, gripper, payload, base, final_lowering):
    if not np.isfinite(distance):
        return True
    if payload in bodies and bodies.intersection(gripper):
        return False
    if bodies == {payload, base}:
        return distance < -1e-5 if final_lowering else distance <= 0
    return bool((bodies.intersection(robot) or payload in bodies) and distance < -1e-5)


class CollisionChecker:
    def __init__(self, model, data, hand, payload, base, robot_bodies, gripper_bodies, q_indices):
        self.model, self.hand, self.payload, self.base = model, hand, payload, base
        self.robot, self.gripper = set(robot_bodies), set(gripper_bodies)
        self.q_indices = list(q_indices)
        self.data = mujoco.MjData(model)
        restore_state(model, self.data, capture_state(model, data))
        self.relative = body_pose(model, data, hand).inverse()*body_pose(model, data, payload)
        body = model.body(payload).id
        joint = int(model.body_jntadr[body])
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise PlanningFailure('payload_not_free_body')
        self.payload_address = int(model.jnt_qposadr[joint])
        self.checked, self.journal = 0, []
        self.filters = dict(contype=model.geom_contype.copy(), conaffinity=model.geom_conaffinity.copy(),
                            exclude_signature=model.exclude_signature.copy(),
                            limitation='Enabled MuJoCo contacts; discrete <=0.005 rad/joint, not continuous proof')

    def contacts(self, data, final_lowering):
        rows, unexpected = [], []
        for contact in data.contact:
            geoms = [int(contact.geom1), int(contact.geom2)]
            bodies = {self.model.body(int(self.model.geom_bodyid[g])).name for g in geoms}
            row = dict(geoms=geoms, bodies=sorted(bodies), distance_m=float(contact.dist))
            row['unexpected'] = unexpected_contact(bodies, contact.dist, self.robot, self.gripper,
                                                    self.payload, self.base, final_lowering)
            rows.append(row)
            if row['unexpected']:
                unexpected.append(row)
        return rows, unexpected

    def sample(self, q, final_lowering, carrying=True):
        finite(q)
        if np.asarray(q).shape != (len(self.q_indices),):
            raise PlanningFailure('collision_joint_shape')
        if self.checked >= 10000:
            raise PlanningFailure('collision_sample_limit', checked=self.checked)
        self.data.qpos[self.q_indices] = q
        mujoco.mj_forward(self.model, self.data)
        if carrying:
            pose = body_pose(self.model, self.data, self.hand)*self.relative
            address = self.payload_address
            self.data.qpos[address:address+3] = pose.translation
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, pose.rotation.reshape(-1))
            self.data.qpos[address+3:address+7] = quat
            mujoco.mj_forward(self.model, self.data)
        contacts, unexpected = self.contacts(self.data, final_lowering)
        self.checked += 1
        self.journal.append(dict(q=np.asarray(q).copy(), carrying=carrying,
                                 final_lowering=final_lowering, contacts=contacts, unexpected=unexpected))
        return unexpected

    def segment(self, start, end, final_lowering, carrying=True):
        finite(start, end)
        n = max(1, math.ceil(float(np.max(np.abs(end-start)))/.005))
        for i in range(n+1):
            sample = end.copy() if i == n else start+(end-start)*(i/n)
            if self.sample(sample, final_lowering, carrying):
                raise PlanningFailure('path_collision', sample=self.checked-1,
                                      contacts=self.journal[-1]['unexpected'])