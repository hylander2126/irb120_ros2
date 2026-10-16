"""NetFT frame convention retained from the original tool-transform setup.

The old tool0_to_ft joint was identity (commit 9296f57). NetFT's internal
axis transform was configured for those axes. root_finger is a CAD frame
with different axes. Published ft_link labels are retained for consumers.
The estimator expects the tool0 reference pose and adds its own 0.08225 m
wrench-origin offset; do not log the distal origin in its place.
"""

WRENCH_AXES_FRAME = 'tool0'
ESTIMATOR_POSE_FRAME = 'tool0'
WRENCH_HEADER_FRAME = 'ft_link'


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
