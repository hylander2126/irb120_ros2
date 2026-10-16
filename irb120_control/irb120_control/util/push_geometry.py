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


def push_approach(ball_center, xyz, standoff, travel):
    """Clear the entire padded box while preserving the nominal push endpoint."""
    ball = np.asarray(ball_center, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    if (ball.shape != (3,) or xyz.ndim != 2 or xyz.shape[1] != 3
            or not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(ball).all()):
        raise ValueError('Push approach requires finite ball center and nonempty Nx3 cloud')
    nominal_x = ball[0] - standoff
    clear_x = xyz[:, 0].min() - OBJECT_PADDING - BALL_RADIUS - APPROACH_CLEARANCE
    pre = ball.copy()
    pre[0] = min(nominal_x, clear_x)
    return tuple(pre), float(travel + nominal_x - pre[0])


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
