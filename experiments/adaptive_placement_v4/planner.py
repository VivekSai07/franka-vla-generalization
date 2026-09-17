"""Bounded desired-segment bisection; no physics or alternative route search."""
from dataclasses import dataclass
import numpy as np
import pinocchio as pin

from experiments.bounded_placement_v3 import path as v3_path
from experiments.bounded_placement_v3.solver import PlanningFailure, finite, pose_errors, solve_waypoint


@dataclass(frozen=True)
class Limits:
    # Overrides may only tighten limits (for focused finite-budget tests).
    attempts: int = 400
    iterations: int = 40000
    depth: int = 8

    def __post_init__(self):
        for value, maximum in ((self.attempts, 400), (self.iterations, 40000), (self.depth, 8)):
            if type(value) is not int or not 0 <= value <= maximum:
                raise PlanningFailure('invalid_adaptive_limit')


@dataclass(frozen=True)
class Dependencies:
    solve: object = solve_waypoint
    residual: object = pose_errors
    waypoints: object = v3_path.pose_waypoints
    branch: object = v3_path.check_branch
    schedule: object = v3_path.time_parameterize


class FiniteRobot:
    """Validate full joint q before native FK/Jacobian; own mutable FK outputs."""
    def __init__(self, robot):
        self.robot, self.model = robot, robot.model

    def forward_kinematics(self, q):
        finite(q)
        pose = self.robot.forward_kinematics(q).copy()
        finite(pose.homogeneous)
        return pose

    def jacobian(self, q):
        finite(q)
        J = self.robot.jacobian(q).copy()
        finite(J)
        return J


class CandidateJournal(list):
    """Explicit v3 journal adapter: count updates before QP/backtracking work.

    v3 appends the residual observation BEFORE testing convergence/iteration 100.
    Consequently 101 rows can describe 100 updates. A failed QP or backtracking
    consumes an update; a converged residual row does not. No solver rewriting.
    """
    def __init__(self, budget, limits):
        super().__init__()
        self.budget, self.limits, self.updates = budget, limits, 0

    def append(self, row):
        expected = len(self)
        super().append(row)  # Even rejected/nonfinite evidence stays visible.
        row['update_counted'] = False
        if row['iteration'] != expected or expected > 100:
            raise PlanningFailure('solver_iteration_protocol')
        self.budget['residual_evaluations'] += 1
        finite(row['q'], row['position_error_m'], row['orientation_error_rad'], row['objective'])
        converged = row['position_error_m'] <= .001 and row['orientation_error_rad'] <= .01
        if not converged and expected < 100:
            if self.budget['iterations'] >= self.limits.iterations:
                raise PlanningFailure('total_ik_iteration_limit', used=self.budget['iterations'])
            self.updates += 1
            self.budget['iterations'] += 1
            row['update_counted'] = True


def plan_route(robot, measured_q, previous_target, endpoints, bounds, checker,
               minimum_ticks, available_ticks, journal, carrying=True, *,
               dependencies=Dependencies(), limits=Limits()):
    """Return a complete checked schedule or retain first failure; never execute.

    Only residual-valid branch-step rejections subdivide. Both children restart
    from their immediately preceding accepted posture. Rejected parent candidates
    are evidence, never seeds. Original desired endpoint poses are retained exactly.
    """
    robot = FiniteRobot(robot)
    q = np.asarray(measured_q, dtype=np.float64).copy()
    previous_target = np.asarray(previous_target, dtype=np.float64).copy()
    bounds = np.asarray(bounds, dtype=np.float64).copy()
    endpoints = [p.copy() for p in endpoints]
    budget = dict(attempts=0, iterations=0, residual_evaluations=0, collision_samples=checker.checked)
    vertices = [(0, q[:7].copy())]
    journal.update(start_q=q.copy(), previous_target=previous_target.copy(),
                   endpoints=[p.homogeneous.copy() for p in endpoints], attempts=[],
                   vertices=[dict(stage=0, q=q[:7].copy())], budgets=budget,
                   limits=dict(attempts=limits.attempts, iterations=limits.iterations,
                               iterations_per_attempt=100, depth=limits.depth, collision_samples=10000))

    def visit(start, target, seed, stage, depth, parent, original_index):
        if budget['attempts'] >= limits.attempts:
            raise PlanningFailure('waypoint_attempt_limit', used=budget['attempts'])
        candidates = CandidateJournal(budget, limits)
        row = dict(id=budget['attempts'], parent=parent, depth=depth, stage=stage,
                   original_index=original_index, start_pose=start.homogeneous.copy(),
                   pose=target.homogeneous.copy(), seed=seed.copy(), candidates=candidates,
                   status='started')
        journal['attempts'].append(row)
        budget['attempts'] += 1
        try:
            solved = np.asarray(dependencies.solve(robot, seed.copy(), target.copy(), bounds, candidates)).copy()
            finite(solved)
            if solved.shape != q.shape:
                raise PlanningFailure('invalid_ik_shape')
            v3_path.validate_target(solved[:7], bounds)
            _, _, position, rotation = dependencies.residual(robot, solved, target)
            finite(position, rotation)
            row.update(q=solved.copy(), position_error_m=position, orientation_error_rad=rotation,
                       joint_step_rad=float(np.max(np.abs(solved[:7]-seed[:7]))))
            if position > .001 or rotation > .01:
                raise PlanningFailure('independent_fk_residual')
            try:
                dependencies.branch(seed, solved)
            except PlanningFailure as exc:
                if exc.gate != 'waypoint_branch_jump':
                    raise
                rejection = dict(gate=exc.gate, details=exc.details, attempt=row['id'])
                row['rejection'] = rejection
                journal.setdefault('first_branch_rejection', rejection)
                if depth >= limits.depth:
                    raise PlanningFailure('subdivision_depth_limit', depth=depth, rejection=rejection)
                translation = float(np.linalg.norm(target.translation-start.translation))
                rotation_log = pin.log3(start.rotation.T@target.rotation)
                angle = float(np.linalg.norm(rotation_log))
                if translation/2 < 1e-5 and angle/2 < 1e-5:
                    raise PlanningFailure('subdivision_minimum_motion', translation_m=translation,
                                          rotation_rad=angle, rejection=rejection)
                middle = pin.SE3(start.rotation@pin.exp3(.5*rotation_log),
                                 .5*start.translation+.5*target.translation)
                row['status'] = 'subdivided'
                left_q = visit(start, middle, seed, stage, depth+1, row['id'], original_index)
                return visit(middle, target, left_q, stage, depth+1, row['id'], original_index)
            checker.segment(seed[:7], solved[:7], stage >= 1 or not carrying, carrying)
            row['status'] = 'accepted'
            vertices.append((stage, solved[:7].copy()))
            journal['vertices'].append(dict(stage=stage, q=solved[:7].copy(), attempt=row['id']))
            return solved
        except Exception as exc:
            # Do not relabel a parent as the failing attempt when its child fails.
            if row['status'] != 'subdivided':
                row.update(status='failed', failure=dict(gate=getattr(exc, 'gate', type(exc).__name__),
                                                        details=getattr(exc, 'details', {})))
            raise
        finally:
            row['iterations'] = candidates.updates
            budget['collision_samples'] = checker.checked

    try:
        finite(q, previous_target, bounds)
        if q.shape != (robot.model.nq,):
            raise PlanningFailure('invalid_ik_shape')
        v3_path.validate_target(q[:7], bounds)
        v3_path.validate_target(previous_target, bounds)
        start = robot.forward_kinematics(q).copy()
        journal['start_pose'] = start.homogeneous.copy()
        desired = dependencies.waypoints(start, endpoints)
        journal['original_waypoints'] = [dict(stage=s, pose=p.homogeneous.copy()) for s, p in desired]
        checker.segment(previous_target, q[:7], not carrying, carrying)
        for index, (stage, target) in enumerate(desired):
            q = visit(start, target, q, stage, 0, None, index)
            start = target.copy()
        schedule = dependencies.schedule(previous_target, vertices, minimum_ticks, bounds, available_ticks)
        journal.update(commands=[dict(stage=s, q=v.copy()) for s, v in schedule], success=True)
        return schedule
    except Exception as exc:
        journal.update(success=False, failure=dict(first_failed_gate=getattr(exc, 'gate', type(exc).__name__),
                                                   details=getattr(exc, 'details', {}), message=str(exc)))
        raise
    finally:
        budget['collision_samples'] = checker.checked