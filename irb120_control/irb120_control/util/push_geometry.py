"""Free-space push approach geometry, in base_link (metres)."""
import numpy as np

OBJECT_PADDING = 0.010
BALL_RADIUS = 0.01325
APPROACH_CLEARANCE = 0.010

# Tool geometry behind the ball, along the finger/tool axis (irb120_with_finger.xacro and the
# link_5 collision mesh). The tool stack collision cylinder covers adapters + F/T sensor.
BALL_TO_STACK = 0.0866512   # finger_ball_center_xyz: ball centre -> F/T output face
STACK_LENGTH = 0.0858       # tool_stack_length: F/T output face -> tool0 flange
STACK_RADIUS = 0.045        # tool_stack_collision radius
FLANGE_TO_WRIST = 0.072     # joint_6 origin: wrist centre behind the flange
WRIST_REACH = 0.0796        # link_5 collision mesh: furthest point from the wrist centre


def horizontal_push_direction(normal):
    """Horizontal inward normal, limited to the existing robot-facing 20 deg cone."""
    normal = np.asarray(normal, dtype=float)
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or np.linalg.norm(normal) < 1e-9
            or -normal[0] / np.linalg.norm(normal) < np.cos(np.radians(20.0))):
        raise ValueError('Push normal must face the robot within 20 deg of -X')
    direction = -normal.copy()
    direction[2] = 0.
    return direction / np.linalg.norm(direction)


def push_orientation(direction, pitch_deg):
    """Yaw the pitched tool so its shaft stays behind the push direction."""
    from scipy.spatial.transform import Rotation
    yaw = np.arctan2(direction[1], direction[0])
    return (Rotation.from_rotvec([0., 0., yaw]) *
            Rotation.from_rotvec([0., np.radians(pitch_deg), 0.])).as_quat()


def force_push_check(force, direction, min_force=.3, max_angle_deg=45., hard_limit=20.):
    """Check the reaction in base axes; return waiting/validated/rejected, never steer."""
    force = np.asarray(force, dtype=float)
    if force.shape != (3,) or not np.isfinite(force).all():
        return 'rejected', 'invalid force'
    if np.linalg.norm(force) >= hard_limit:
        return 'rejected', 'force limit exceeded'
    horizontal = np.linalg.norm(force[:2])
    if horizontal < min_force:
        return 'waiting', 'no sustained horizontal contact'
    angle = float(np.degrees(np.arccos(np.clip(
        -force[:2] @ np.asarray(direction)[:2] / horizontal, -1., 1.))))
    return ('validated' if angle <= max_angle_deg else 'rejected',
            f'reaction angle {angle:.1f} deg (limit {max_angle_deg:g})')


def push_approach(ball_center, xyz, standoff, travel, direction=(1., 0., 0.)):
    """Clear the entire padded box while preserving the nominal push endpoint."""
    ball = np.asarray(ball_center, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if (ball.shape != (3,) or xyz.ndim != 2 or xyz.shape[1] != 3
            or not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(ball).all()):
        raise ValueError('Push approach requires finite ball center and nonempty Nx3 cloud')
    if (direction.shape != (3,) or not np.isfinite(direction).all()
            or direction[0] <= 0 or abs(direction[2]) > 1e-9
            or not np.isclose(np.linalg.norm(direction), 1.)):
        raise ValueError('Push direction must be a horizontal unit vector toward +X')
    clear_x = xyz[:, 0].min() - OBJECT_PADDING - BALL_RADIUS - APPROACH_CLEARANCE
    backoff = max(standoff, (ball[0] - clear_x) / direction[0])
    pre = ball - backoff * direction
    return tuple(pre), float(travel + backoff - standoff)


def min_push_pitch(ball_height, clearance):
    """Smallest tip-down pitch (rad) keeping the tool stack and wrist `clearance` above the table.

    `ball_height` is the ball centre above the table. The shaft leans back-and-up from the ball
    at the pitch; the stack's lowest point is the near rim of the cylinder (BALL_TO_STACK back,
    STACK_RADIUS down), and link_5 is bounded by a WRIST_REACH sphere about the wrist centre,
    whatever joint 5 does. Ahead of the ball the tool never reaches past the contact face for
    any pitch >= 0, so the table is the only bound.
    """
    drop = clearance - ball_height   # lowest allowed point, relative to the ball centre
    stack = np.arctan2(STACK_RADIUS, BALL_TO_STACK) + np.arcsin(
        np.clip(drop / np.hypot(BALL_TO_STACK, STACK_RADIUS), -1, 1))
    wrist = np.arcsin(np.clip(
        (drop + WRIST_REACH) / (BALL_TO_STACK + STACK_LENGTH + FLANGE_TO_WRIST), -1, 1))
    return float(max(stack, wrist, 0.0))
