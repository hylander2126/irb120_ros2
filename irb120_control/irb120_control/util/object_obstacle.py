"""The detected object as a MoveIt collision box, so approach moves plan around it.

Only for free-space moves (approaches, and the return home). It must be gone
before any contact phase: the Cartesian push and MoveIt Servo both check
collisions and would stop at the box. Use it as

    with object_obstacle(node, xyz):
        ok = plan_and_execute_pose_goal(...)

The box is the cloud's axis-aligned bounds in base_link, grown by PADDING and
extended down to the table. PUSH_STANDOFF / SQUASH_STANDOFF keep the finger
ball ~14 mm outside it at the approach poses (checked on the flashlight).

After a push the object is somewhere between where it was and PUSH_DISTANCE
further in +X, so the move home uses the box stretched by that much in +X
(extend_x). That stays clear of the arm, which is back on the -X side.
"""

from contextlib import contextmanager

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive

OBJECT_ID = "detected_object"
FRAME = "base_link"
TABLE_Z = -0.021   # table top in base_link
PADDING = 0.010    # m, on every side except the bottom


def _apply(node, obj: CollisionObject) -> bool:
    client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    if not client.wait_for_service(timeout_sec=3.0):
        node.get_logger().error("/apply_planning_scene not available")
        return False
    req = ApplyPlanningScene.Request()
    req.scene.is_diff = True
    req.scene.world.collision_objects = [obj]
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
    return future.result() is not None and future.result().success


@contextmanager
def object_obstacle(node, xyz, extend_x: float = 0.0):
    """Add the object's padded bounding box (optionally stretched by extend_x in +X)
    to the planning scene for the duration of the block."""
    lo, hi = xyz.min(axis=0) - PADDING, xyz.max(axis=0) + PADDING
    lo[2] = TABLE_Z
    hi[0] += extend_x
    obj = CollisionObject()
    obj.header.frame_id = FRAME
    obj.id = OBJECT_ID
    obj.operation = CollisionObject.ADD
    obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=(hi - lo).tolist())]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = ((lo + hi) / 2).tolist()
    pose.orientation.w = 1.0
    obj.primitive_poses = [pose]
    if not _apply(node, obj):
        raise RuntimeError("could not add the object to the planning scene -- not moving")
    node.get_logger().info(f"Object collision box {np.round(lo, 3)} .. {np.round(hi, 3)}")
    try:
        yield
    finally:
        remove = CollisionObject()
        remove.header.frame_id = FRAME
        remove.id = OBJECT_ID
        remove.operation = CollisionObject.REMOVE
        if not _apply(node, remove):
            node.get_logger().error("could not remove the object box from the planning scene")
