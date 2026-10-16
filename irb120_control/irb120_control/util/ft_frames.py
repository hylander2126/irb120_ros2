"""NetFT frame convention retained from the original tool-transform setup.

The old tool0_to_ft joint was identity (commit 9296f57). NetFT's internal
axis transform was configured for those axes. root_finger is a CAD frame
with different axes. Published ft_link labels are retained for consumers.
The log stores the tool0 reference pose. Estimation expresses moments about
the finger mounting plane, recovered from the ball frame and rigid shaft.
Do not treat the historical tool0-to-mount 0.08225 m as the current CAD chain.
"""

WRENCH_AXES_FRAME = 'tool0'
ESTIMATOR_POSE_FRAME = 'tool0'
WRENCH_HEADER_FRAME = 'ft_link'

# finger_ball_center_xyz in irb120_with_finger.xacro, expressed along tool0 +X.
# This is the rigid mount-to-ball distance; it is independent of the adapters
# between tool0 and the mount, which changed when the CAD chain was imported.
FINGER_MOUNT_TO_BALL_X = 0.0866512


def wrench_origin_from_ball(p_ball, q_tool):
    """World position of the existing distal-face wrench reference.

    The current tool0-to-mount chain is 85.8244 mm plus a transverse offset,
    while the historical one was 82.25 mm. Both have the same rigid finger.
    Using the logged ball pose handles either chain without an adapter-length
    guess or any object mass/CoM input. Wrench axes remain tool0 axes.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    p_ball = np.asarray(p_ball, float)
    return p_ball - Rotation.from_quat(q_tool).apply(
        np.broadcast_to([FINGER_MOUNT_TO_BALL_X, 0., 0.], p_ball.shape))


def forces_in_base(d):
    """(fx, fy, fz) of an F/T+pose log rotated from WRENCH_AXES_FRAME into base_link axes.

    Uses the per-sample tool0 orientation logged with each wrench (ft_q*). Needed once the
    finger is pitched (push.PUSH_PITCH_DEG, ~28.5 deg); at the old horizontal pose tool0 is aligned with
    base_link, so this is the identity there. Logs without ft_q*, or rows whose TF lookup
    failed (NaN), are taken as already aligned.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    f = np.c_[d["fx"], d["fy"], d["fz"]]
    if "ft_qw" not in getattr(d, "files", d):
        return f[:, 0], f[:, 1], f[:, 2]
    q = np.c_[d["ft_qx"], d["ft_qy"], d["ft_qz"], d["ft_qw"]]
    q[~np.isfinite(q).all(axis=1)] = (0.0, 0.0, 0.0, 1.0)
    f = Rotation.from_quat(q).apply(f)
    return f[:, 0], f[:, 1], f[:, 2]
