"""Free-space push approach geometry, in base_link (metres)."""
import numpy as np

OBJECT_PADDING = 0.010
BALL_RADIUS = 0.01325
APPROACH_CLEARANCE = 0.010


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
