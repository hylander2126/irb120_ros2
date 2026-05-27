#!/usr/bin/env python3
"""
Cross-check the logged EE pose (finger_ball_center) against a forward-kinematic
prediction built from the *confident* FT-sensor pose plus the fixed finger/sensor
geometry taken straight from the xacro.

Logic
-----
The xacro chain is:

    tool0 --(ft_mount rpy)--> ft_link --(z=+ft_length, finger_unwind rpy)--> finger_link
          --(x=+finger_length)--> finger_ball_center

You log `ft_*` (the FT sensor pose in {B}) and `x/y/z` (the EE = finger_ball_center
pose in {B}). The transform that takes you from the FT *link* frame to the tip is
fixed and known. So:

    T_B_tip_pred = T_B_ft @ T_ft_tip

where T_ft_tip = (ft_to_finger joint) @ (finger_to_tip joint).

IMPORTANT ASSUMPTION TO VERIFY: that the pose you log under `ft_*` is the pose of
`ft_link`'s frame (the joint origin shared with tool0_to_ft's child), NOT some other
point along the sensor. If your TF logger reports a different frame (e.g. tool0, or a
sensor measurement frame offset by ft_cog_z), the constant offset below is wrong and
that itself is the bug. This script will surface that as a constant position residual.

We compare position only first (orientation diff printed too). A constant residual ->
frame/offset mislabel. A residual that grows with tip angle -> rotation-in-the-chain
error. A residual that scales with lever length -> length error.
"""
import numpy as np

# ----------------------------------------------------------------------------
# Geometry from the xacro (edit path / keys to match your logs)
# ----------------------------------------------------------------------------
FT_LENGTH     = 0.08225   # ft_to_finger translation along ft_link z
FINGER_LENGTH = 0.100     # finger_to_tip translation along finger_link x

# rpy for ft_to_finger joint (finger_unwind_*)
FINGER_UNWIND_RPY = np.array([0.0, -1.5708, 1.5708])  # roll, pitch, yaw


def rpy_to_R(rpy):
    """URDF rpy = intrinsic X(roll) then Y(pitch) then Z(yaw), i.e. R = Rz @ Ry @ Rx."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T_from(rpy, xyz):
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(rpy)
    T[:3, 3] = xyz
    return T


def quat_to_R(q):
    """q = [x,y,z,w] (matches your ft_qx..ft_qw ordering). Batched (N,4) or (4,)."""
    q = np.atleast_2d(q).astype(float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y*y + z*z); R[:, 0, 1] = 2 * (x*y - z*w); R[:, 0, 2] = 2 * (x*z + y*w)
    R[:, 1, 0] = 2 * (x*y + z*w);     R[:, 1, 1] = 1 - 2 * (x*x + z*z); R[:, 1, 2] = 2 * (y*z - x*w)
    R[:, 2, 0] = 2 * (x*z - y*w);     R[:, 2, 1] = 2 * (y*z + x*w);     R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_T_ft_tip():
    """Fixed transform from the FT *link* frame to finger_ball_center."""
    # ft_to_finger joint: translate +z by ft_length, rotate by finger_unwind rpy
    T_ft_finger = T_from(FINGER_UNWIND_RPY, np.array([0.0, 0.0, FT_LENGTH]))
    # finger_to_tip joint: translate +x by finger_length, no rotation
    T_finger_tip = T_from(np.zeros(3), np.array([FINGER_LENGTH, 0.0, 0.0]))
    return T_ft_finger @ T_finger_tip


def run_tst(npz_path):
    data = np.load(npz_path)

    # logged FT pose in {B}
    p_ft = np.column_stack([data["ft_px"], data["ft_py"], data["ft_pz"]])      # (N,3)
    q_ft = np.column_stack([data["ft_qx"], data["ft_qy"], data["ft_qz"], data["ft_qw"]])
    R_ft = quat_to_R(q_ft)                                                      # (N,3,3)

    # logged EE pose in {B}  (these are on a different time grid in your pipeline!)
    p_ee = np.column_stack([data["x"], data["y"], data["z"]])                   # (M,3)

    T_ft_tip = build_T_ft_tip()
    print("T_ft_tip (FT link -> finger_ball_center):")
    print(np.round(T_ft_tip, 4))
    print(f"  => tip offset in FT-link frame: {np.round(T_ft_tip[:3,3],4)}  "
          f"(|.|={np.linalg.norm(T_ft_tip[:3,3]):.4f} m)\n")

    # FK prediction of tip position in {B}
    p_tip_pred = p_ft + np.einsum('nij,j->ni', R_ft, T_ft_tip[:3, 3])           # (N,3)

    # NOTE: p_ft and p_ee may be on different time grids/lengths. If your npz stores
    # ft on the raw 500Hz grid and EE at ~100Hz, align by nearest time before diffing.
    N, M = len(p_tip_pred), len(p_ee)
    print(f"samples: ft={N}  ee={M}")
    if "ft_time_s" in data and "pose_time_s" in data:
        t_ft = data["ft_time_s"] if len(data["ft_time_s"]) == N else data.get("ft_raw_time_s")
        t_ee = data["pose_time_s"]
        # nearest-neighbour align EE onto ft grid
        idx = np.searchsorted(t_ee, t_ft)
        idx = np.clip(idx, 0, M - 1)
        p_ee_al = p_ee[idx]
        print("aligned EE onto FT time grid by nearest time.\n")
    else:
        n = min(N, M)
        p_tip_pred, p_ee_al = p_tip_pred[:n], p_ee[:n]
        print("WARNING: no time vectors found; truncated to common length (may be misaligned).\n")

    resid = p_ee_al - p_tip_pred                                               # (.,3)
    rnorm = np.linalg.norm(resid, axis=1)

    print("position residual  EE_logged - EE_FKpredicted  (meters, {B} frame)")
    print(f"  mean   : {np.round(resid.mean(0), 5)}")
    print(f"  std    : {np.round(resid.std(0), 5)}")
    print(f"  |resid|: mean={rnorm.mean():.5f}  median={np.median(rnorm):.5f}  max={rnorm.max():.5f}")
    print()
    print("Interpretation:")
    print("  - near-zero mean & std  -> EE pose is consistent with FT+geometry. Trust it.")
    print("  - large CONSTANT mean   -> frame/offset mislabel (logging wrong point, or")
    print("                             ft pose is not the ft_link frame). Fixes lever arm.")
    print("  - std grows over time   -> a rotation in the chain (ft_mount / finger_unwind)")
    print("                             is wrong; angle scaling -> mass inflation.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz-path",
        default="/home/hylander2126/Documents/github/ros2_irb120/runtime_logs/box/arc_squash/most_recent.npz",
        help="Path to the squash log npz file.",
    )
    args = parser.parse_args()
    run_tst(args.npz_path)


if __name__ == "__main__":
    main()
    