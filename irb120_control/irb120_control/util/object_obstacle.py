"""The detected object as a MoveIt collision box, so approach moves plan around it.

Only for free-space moves (approaches, and the return home). It must be gone
before any contact phase: the Cartesian push and MoveIt Servo both check
collisions and would stop at the box. Use it as

    with object_obstacle(node, xyz):
        ok = plan_and_execute_pose_goal(...)

The box is the cloud's axis-aligned bounds in base_link, grown by PADDING and
extended down to the table. slabs=True instead adds one padded box per 20 mm slice
(forward_tip_motion.slab_boxes), so the forward-tip pre-pose can sit beside a narrow
part above a wide one while MoveIt still checks the whole arm against the object. The push approach clears the full box by 10 mm beyond the ball radius.
The press approach uses SQUASH_STANDOFF above its selected contact.

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
from irb120_control.util.push_geometry import OBJECT_PADDING

PADDING = OBJECT_PADDING  # m, on every side except the bottom


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


def box_bounds(xyz, extend_x: float = 0.0):
    """The padded box's (lo, hi) corners in base_link, as object_obstacle() adds it."""
    lo, hi = xyz.min(axis=0) - PADDING, xyz.max(axis=0) + PADDING
    lo[2] = TABLE_Z
    hi[0] += extend_x
    return lo, hi


@contextmanager
def object_obstacle(node, xyz, extend_x: float = 0.0, slabs: bool = False):
    """Add the object's padded bounding box (optionally stretched by extend_x in +X), or
    with slabs=True its stack of per-slice boxes, to the planning scene for the block."""
    if slabs:
        from irb120_control.util.forward_tip_motion import slab_boxes
        boxes = slab_boxes(np.asarray(xyz, dtype=float))
        if extend_x:
            for _, hi in boxes:
                hi[0] += extend_x
    else:
        boxes = [box_bounds(xyz, extend_x)]
    obj = CollisionObject()
    obj.header.frame_id = FRAME
    obj.id = OBJECT_ID
    obj.operation = CollisionObject.ADD
    for lo, hi in boxes:
        obj.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=(hi - lo).tolist()))
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = ((lo + hi) / 2).tolist()
        pose.orientation.w = 1.0
        obj.primitive_poses.append(pose)
    if not _apply(node, obj):
        raise RuntimeError("could not add the object to the planning scene -- not moving")
    lo, hi = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
    node.get_logger().info(f"Object collision {'slabs' if slabs else 'box'} ({len(boxes)}) "
                           f"{np.round(lo, 3)} .. {np.round(hi, 3)}")
    try:
        yield
    finally:
        remove = CollisionObject()
        remove.header.frame_id = FRAME
        remove.id = OBJECT_ID
        remove.operation = CollisionObject.REMOVE
        if not _apply(node, remove):
            node.get_logger().error("could not remove the object box from the planning scene")
