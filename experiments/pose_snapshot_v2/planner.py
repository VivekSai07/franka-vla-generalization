"""Own SE3 endpoint snapshots; preserve every legacy interpolation operation."""
import pinocchio as pin

from src.planners.trajectory_planner import TaskSpaceTrajectory


class SnapshotTaskSpaceTrajectory(TaskSpaceTrajectory):
    """Later FK or caller input mutation cannot move either stored endpoint."""

    def __init__(self, start_pose, end_pose, duration, method="cubic"):
        super().__init__(pin.SE3(start_pose), pin.SE3(end_pose), duration, method)