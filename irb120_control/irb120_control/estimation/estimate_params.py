#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from irb120_control.estimation.com_estimation import model_fwd_wrench, model_bkwd_wrench, compute_applied_wrench, W_app_arc, construct_T
from irb120_control.estimation.helper_fns import rotvec_to_rot, quat_to_rotvec
from irb120_control.estimation.plotting_helper import plot_wrench_and_tipping, plot_torque_fit_result

ALL_OBJECTS = ["box"]#, "heart", "flashlight", "monitor", "soda"]

_LPF_B,      _LPF_A      = butter(4, 6,   fs=500, btype='low')  # 4, 6 Hz — removes high-freq sensor noise
_LPF_SLOW_B, _LPF_SLOW_A = butter(2, 0.5, fs=500, btype='low')  # 2, 0.5 Hz — removes force-controller hunting


def _lpf(x, axis=0):
    return filtfilt(_LPF_B, _LPF_A, x, axis=axis) if x.shape[0] > 20 else x


def _lpf_slow(x, axis=0):
    return filtfilt(_LPF_SLOW_B, _LPF_SLOW_A, x, axis=axis) if x.shape[0] > 20 else x


def load_and_preprocess(filepath):
    data = np.load(filepath)

    for keys in [
        ("ft_px", "ft_py", "ft_pz", "ft_qx", "ft_qy", "ft_qz", "ft_qw"),
        ("pose_time_s", "x", "y", "z"),
        ("obj_time_s", "obj_x", "obj_y", "obj_z", "obj_qx", "obj_qy", "obj_qz", "obj_qw"),
        ("ft_raw_time_s", "fx_raw", "fy_raw", "fz_raw", "tx_raw", "ty_raw", "tz_raw")
    ]:
        if not all(k in data for k in keys) or len(data[keys[0]]) == 0:
            raise KeyError(f"Missing keys {keys} in {filepath}.")

    # Three independently-sampled time axes: F/T @ 500 Hz, EE pose @ ~100 Hz, object detector @ ~2 Hz (unused)
    time_ft  = data["ft_time_s"]
    time_ft_raw = data["ft_raw_time_s"]
    time_ee  = data["pose_time_s"]
    time_obj = data["obj_time_s"] # Unused object detector time

    f_meas_S = _lpf(np.column_stack([data["fx"], data["fy"], data["fz"]])) # F/T in world frame
    t_meas_S = _lpf(np.column_stack([data["tx"], data["ty"], data["tz"]]))

    # BIASING HAPPENS ONCE ORIENTATION IS CALCULATED - The gravity changes wrt {S}
    f_meas_raw = np.column_stack([data["fx_raw"], data["fy_raw"], data["fz_raw"]]) # F/T in {S} frame (unbiased)
    t_meas_raw = np.column_stack([data["tx_raw"], data["ty_raw"], data["tz_raw"]])

    p_ft_B   = np.column_stack([data["ft_px"], data["ft_py"], data["ft_pz"]])
    Q_ft     = np.column_stack([data["ft_qx"], data["ft_qy"], data["ft_qz"], data["ft_qw"]])
    p_ee_B   = np.column_stack([data["x"], data["y"], data["z"]])
    p_obj_B  = np.column_stack([data["obj_x"], data["obj_y"], data["obj_z"]])

    # Now we can bias the raw F/T using Q_ft
    bias_f_S = np.mean(f_meas_raw[50:250], axis=0) # average of first 50 samples as bias
    bias_t_S = np.mean(t_meas_raw[50:250], axis=0)
    # per-sample rotations: R[n] maps {S} -> {B}
    R_ft_B = rotvec_to_rot(quat_to_rotvec(Q_ft)) # (N,3,3) rotation - {S} in {B}
    # base-frame bias (use rotation at bias samples)
    R0 = R_ft_B[100] # rotation at sample 100 (middle of bias window)
    bias_f_B = R0 @ bias_f_S # (N,3) bias in {B}
    bias_t_B = R0 @ bias_t_S
    # Now rotate base bias into each sample's sensor frame
    bias_f_S_all = np.einsum('nij,j->ni', R_ft_B.transpose(0,2,1), bias_f_B) # (N,3) bias in {S}
    bias_t_S_all = np.einsum('nij,j->ni', R_ft_B.transpose(0,2,1), bias_t_B) # (N,3) bias in {S}
    f_meas_raw -= bias_f_S_all
    t_meas_raw -= bias_t_S_all

    # And finally apply filtering after biasing...
    f_meas_raw = _lpf(f_meas_raw) 
    t_meas_raw = _lpf(t_meas_raw)

    # # If Q_ft idx don't align with time, append zeros
    # if len(Q_ft) < len(time_ft):
    #     n_missing = len(time_ft) - len(Q_ft)
    #     Q_ft = np.vstack((Q_ft, np.zeros((n_missing, 4))))
    #     p_ft_B = np.vstack((p_ft_B, np.zeros((n_missing, 3))))
    # elif len(Q_ft) > len(time_ft):
    #     Q_ft = Q_ft[:len(time_ft)]
    #     p_ft_B = p_ft_B[:len(time_ft)]

    return time_ft, f_meas_S, t_meas_S, time_ft_raw, f_meas_raw, t_meas_raw, p_ft_B, Q_ft, time_ee, p_ee_B, time_obj, p_obj_B


def _run_estimation(obj: str, base_dir: str, squash_file: str) -> None:
    time_ft, f_meas_S, t_meas_S, time_ft_raw, f_meas_raw, t_meas_raw, p_ft_B, Q_ft, time_ee, p_ee_B, time_obj, p_obj_B = load_and_preprocess(squash_file)

    # Temp change of variable name: Set raw data to "meas"...
    time_ft = time_ft_raw
    f_meas_S = f_meas_raw
    t_meas_S = t_meas_raw

    # Loop forward to find contact and find release by doing the opposite loop backwards.
    contact_ft_idx = np.argmax(np.linalg.norm(f_meas_S, axis=1) > 0.5) # argmax on bool array: first idx of norm > 0.5 N
    release_ft_idx = len(time_ft) - np.argmax(np.linalg.norm(f_meas_S[::-1], axis=1) > 0.5) # last idx of norm > 0.5 N
    print(f"[{obj}] contact start: idx={contact_ft_idx} contact end: idx={release_ft_idx}")

    # Pivot: near edge of object = pre-contact centroid x_min (closest to robot is approximate object frame)
    # p_pivot_B = np.array([p_obj_B[time_obj <= time_contact_start, 0].min(), 0.0, 0.0])
    p_pivot_B = np.array([0.6, 0, 0]) # Near-exact pivot from pre-defined object frame (I reset obj to known pose)

    # Proprioceptive tipping angle from EE triangle in X-Z plane
    contact_ee_idx = np.argmin(np.abs(time_ee - time_ft[contact_ft_idx])) # index of first EE (ball center) position after contact
    r_t = p_ee_B                 - p_pivot_B
    r0  = p_ee_B[contact_ee_idx] - p_pivot_B
    print(f"[{obj}] p_pivot_B: {p_pivot_B}, p_ee_B[contact]: {np.round(p_ee_B[contact_ee_idx], 3)}, r0: {np.round(r0, 3)}")
    # Angle of moving r_t wrt stationary r0 in the X-Z plane (positive CCW, +Y into the screen)
    prop_angle = -(np.arctan2(r0[0]*r_t[:, 2] - r0[2]*r_t[:, 0], r0[0]*r_t[:, 0] + r0[2]*r_t[:, 2]))
    pitch_B = np.interp(time_ft, time_ee, prop_angle) # Match the time of F/T data

    # Build batched (N,4,4) homogeneous transforms for sensor and object frames
    T_B_sensor = construct_T(p_ft_B, quat=Q_ft) # (N,4,4) sensor pose in {B}

    rv_obj  = np.zeros((len(time_ft), 3))
    rv_obj[:, 1] = pitch_B 
    p_obj_B = np.tile(p_pivot_B, (len(time_ft), 1)) # Set object position to pivot for all samples
    T_B_obj = construct_T(p_obj_B, rv=rv_obj) # (N,4,4) object pose in {B}

    # ====================================================
    w_meas_S = np.hstack((t_meas_S, f_meas_S))  # (N,6) IMPORTANT: [t, f] CONVENTION
    # ====================================================

    # And mask to contact samples only for estimation and plotting
    contact_mask  = pitch_B < 0.0

    T_B_sensor = T_B_sensor[contact_mask]
    T_B_obj = T_B_obj[contact_mask]
    w_meas_S = w_meas_S[contact_mask]

    ## HERE WE CAN REDEFINE THE r0 lever arm
    if obj == "box":
        # r0 = np.array([0.01, 0.0, 0.3]) # 0.026 old value, new is ~1.4 cm
        COM_GT = np.array([0.0, 0.0, 0.15])
        MASS_GT = 0.635
    elif obj == "heart":
        # r0 = np.array([0.01, 0.0, 0.2]) # 0.026
        COM_GT = np.array([0.0458, 0.0, 0.10])
        MASS_GT = 0.295
    elif obj == "flashlight":
        # r0 = np.array([0.028, 0.0, 0.2]) # 0.028
        COM_GT = np.array([0.028, 0.0, 0.0938])
        MASS_GT = 0.387
    elif obj == "soda":
        # r0 = np.array([0.055, 0.0, 0.3]) # 0.055
        COM_GT = np.array([0.055, 0.0, 0.15])
        MASS_GT = 2.05

    pivot_axis_est = np.array([0.0, 1.0, 0.0])          # HARDCODE: force pure +Y for testing

    w_app_O = model_bkwd_wrench(w_meas_S, T_B_sensor, T_B_obj) #, r0)

    print(f"[{obj}] Percent of applied torque below zero (expect 100%): {(w_app_O[:, 4] < 0).mean() * 100:.1f}%")

    # 0.5 Hz LPF on applied wrench to suppress force-controller hunting for fitting
    pitch_contact  = pitch_B[contact_mask]
    N_c            = len(pitch_contact)
    w_app_O_smooth = _lpf_slow(w_app_O)

    # Project torque onto pivot axis to get scalar torque driving tipping.
    tau_axis        = w_app_O[:, 3:6] @ pivot_axis_est           # (N,) raw projected torque
    tau_axis_smooth = _lpf_slow(tau_axis)                         # (N,) smoothed

    tau_residual_rms = np.sqrt(np.mean(np.linalg.norm(w_app_O[:, 3:6] - tau_axis[:, None] * pivot_axis_est, axis=1)**2))
    print(f"[{obj}] τ_axis RMS={np.sqrt(np.mean(tau_axis**2)):.4f} N·m  "
          f"off-axis residual RMS={tau_residual_rms:.4f} N·m  "
          f"({100*tau_residual_rms/max(np.sqrt(np.mean(tau_axis**2)), 1e-9):.1f}% of signal)")

    # One figure per object: 3 subplots side-by-side
    fig_obj, axes_obj = plt.subplots(1, 3, figsize=(24, 6))
    fig_obj.suptitle(f"[{obj}]", fontsize=14, fontweight="bold")

    # Raw F/T overview plot (post-contact only) with smoothed overlay
    time_ft_rel     = time_ft[contact_ft_idx:] - time_ft[contact_ft_idx]
    R_ft_all        = rotvec_to_rot(quat_to_rotvec(Q_ft[contact_ft_idx:]))
    # f_meas_W        = np.einsum('nij,nj->ni', R_ft_all, f_meas_S[contact_ft_idx:])
    f_meas_W = f_meas_S[contact_ft_idx:]
    f_meas_W_smooth = _lpf_slow(f_meas_W)
    time_contact_xp  = time_ft[contact_mask] - time_ft[contact_ft_idx]
    tau_axis_on_ft   = np.interp(time_ft_rel, time_contact_xp, tau_axis,        left=0.0, right=0.0)
    tau_axis_smooth_ft = np.interp(time_ft_rel, time_contact_xp, tau_axis_smooth, left=0.0, right=0.0)
    plot_wrench_and_tipping(time_ft_rel, f_meas_W, tau_axis_on_ft,
                            ax=axes_obj[0],
                            pitch_rad=pitch_B[contact_ft_idx:], torque_label="τ_axis",
                            force_xyz_smooth=f_meas_W_smooth, torque_primary_smooth=tau_axis_smooth_ft,
                            contact_time=0.0, title=f"Raw F/T + tipping angle", show=False)

    # Calculated Applied Wrench overview plot
    f_app_O_on_ft = np.zeros((len(time_ft_rel), 3))
    f_app_O_smooth_ft = np.zeros((len(time_ft_rel), 3))
    for i in range(3):
        f_app_O_on_ft[:, i] = np.interp(time_ft_rel, time_contact_xp, w_app_O[:, i], left=0.0, right=0.0)
        f_app_O_smooth_ft[:, i] = np.interp(time_ft_rel, time_contact_xp, w_app_O_smooth[:, i], left=0.0, right=0.0)

    plot_wrench_and_tipping(time_ft_rel, f_app_O_on_ft, tau_axis_on_ft,
                            ax=axes_obj[1],
                            pitch_rad=pitch_B[contact_ft_idx:], torque_label="τ_axis",
                            force_xyz_smooth=f_app_O_smooth_ft, torque_primary_smooth=tau_axis_smooth_ft,
                            contact_time=0.0, title=f"Applied Wrench (Object Frame) + tipping angle", show=False)

    # Tipping phase selection: exclude 1.5° from start and peak (angles are negative)
    tip_sel = (pitch_contact < -np.deg2rad(1.6)) & (pitch_contact > pitch_contact.min() + np.deg2rad(1.6))
    print(f"[{obj}] pitch_contact: N={N_c}  min={np.rad2deg(pitch_contact.min()):.2f}°  max={np.rad2deg(pitch_contact.max()):.2f}°")
    print(f"[{obj}] peak={np.rad2deg(pitch_contact.min()):.2f}°  lower_bound=-1.5°  upper_bound={np.rad2deg(pitch_contact.min() + np.deg2rad(1.5)):.2f}°  tip_sel={tip_sel.sum()}")
    if abs(tip_sel.sum()) < 10:
        print(f"[{obj}] Too few tipping samples — skipping fit.")
        return

    # Split tip_sel into push / retract phases
    peak_idx_in_contact = np.argmin(pitch_contact)           # index of most-negative pitch in contact array
    push_phase    = np.arange(N_c) <= peak_idx_in_contact    # (N_c,) bool — up to and including the peak
    push_tip_sel    = tip_sel & push_phase
    retract_tip_sel = tip_sel & ~push_phase

    def _fit_phase(phase_sel, label, COM_GT, tau_y_override=None):
        """Fit COM_z and mass from applied torque.

        phase_sel: (N_c,) bool array selecting samples in the tipping phase to fit
        label: string label for printing
        COM_GT: ground truth center of mass
        tau_y_override: if provided, use this (N,) array as the measured tau_y
        instead of w_app_O_smooth[:, 4]. Useful for fitting against a single
        force component's torque contribution (e.g. F_x only).
        """
        if phase_sel.sum() < 10:
            print(f"[{obj}] Too few {label} samples ({phase_sel.sum()}) — skipping.")
            return None, None
        p_ph    = pitch_contact[phase_sel]
        rv_ph   = np.column_stack([np.zeros(phase_sel.sum()), p_ph, np.zeros(phase_sel.sum())]) # rotation vec for y
        tau_y_meas = tau_y_override[phase_sel] if tau_y_override is not None else tau_axis_smooth[phase_sel]
        com_z0 = mass0 = 0.1
        def _residual(params):
            w_grav, _ = model_fwd_wrench(rv_ph, np.array([COM_GT[0], 0.0, params[0]]), params[1], 0.0)
            # project gravity torque onto estimated pivot axis for consistency
            return (w_grav[:, 3:6] @ pivot_axis_est) + tau_y_meas
        res = least_squares(_residual, x0=[com_z0, mass0],
                            bounds=([1e-6, 1e-6], [np.inf, np.inf]), method='trf')
        com_z, mass = res.x
        print(f"  [{obj}] {label:>7s} fit — COM_z={com_z:.4f} m  Mass={mass:.4f} kg  θ*={np.degrees(np.arctan2(COM_GT[0], com_z)):.1f}°")
        return com_z, mass


    print(f"\n--- [{obj}] PHASE ESTIMATES (full torque) ---")
    com_z_push,    mass_push    = _fit_phase(push_tip_sel,    "push", COM_GT)
    com_z_retract, mass_retract = _fit_phase(retract_tip_sel, "retract", COM_GT)
    print(f"  [{obj}] Ground truth — COM_z={COM_GT[2]:.4f} m  Mass={MASS_GT:.4f} kg  θ*={np.degrees(np.arctan2(COM_GT[0], COM_GT[2])):.1f}°")



    # Use push estimate for friction (more reliable — no friction hysteresis on approach)
    # mass_est = mass_push if mass_push is not None else mass_retract
    # push_log = os.path.join(base_dir, "push", "most_recent.npz")
    # if os.path.exists(push_log) and mass_est is not None:
    #     mu_est, mu_std = estimate_friction(push_log, mass_est)
    #     print(f"  μ_table={mu_est:.4f} ± {mu_std:.4f}")
    # else:
    #     print(f"  No push log — skipping friction estimate.")

    # # Build per-sample predictions over the full tip_sel window for plotting
    pitch_fit = pitch_contact[tip_sel]
    rv_fit    = np.column_stack([np.zeros(tip_sel.sum()), pitch_fit, np.zeros(tip_sel.sum())])
    push_sel_plot = push_phase[tip_sel]   # push mask re-indexed to tip_sel

    tau_pred_push = np.zeros(tip_sel.sum())
    if com_z_push is not None:
        w_grav_push, _ = model_fwd_wrench(rv_fit, np.array([COM_GT[0], 0.0, com_z_push]), mass_push, 0.0)
        tau_pred_push = -(w_grav_push[:, 3:6] @ pivot_axis_est)

    tau_pred_retract = np.zeros(tip_sel.sum())
    if com_z_retract is not None:
        w_grav_ret, _ = model_fwd_wrench(rv_fit, np.array([COM_GT[0], 0.0, com_z_retract]), mass_retract, 0.0)
        tau_pred_retract = -(w_grav_ret[:, 3:6] @ pivot_axis_est)

    theta_push    = np.arctan2(COM_GT[0], com_z_push)    if com_z_push    is not None else None
    theta_retract = np.arctan2(COM_GT[0], com_z_retract) if com_z_retract is not None else None

    plot_torque_fit_result(
        pitch_rad=pitch_fit,
        tau_meas=tau_axis_smooth[tip_sel],
        tau_pred_push=tau_pred_push,
        theta_star_push_rad=theta_push if theta_push is not None else 0.0,
        ax=axes_obj[2],
        tau_pred_retract=tau_pred_retract if com_z_retract is not None else None,
        theta_star_retract_rad=theta_retract,
        theta_star_gt_rad=np.arctan2(COM_GT[0], COM_GT[2]),
        push_sel=push_sel_plot,
        title=f"Torque fit result (full torque)",
        show=False,
    )

    fig_obj.tight_layout()
    fig_obj.savefig(os.path.join(base_dir, "estimation_summary.png"), dpi=150, bbox_inches="tight")


# def estimate_friction(push_log_path: str, mass_est: float) -> tuple:
#     data = np.load(push_log_path)
#     f_planar = np.sqrt(_lpf(data["fx"])**2 + _lpf(data["fy"])**2)

#     active  = f_planar > 0.3
#     first_a = np.argmax(active)
#     last_a  = len(f_planar) - np.argmax(active[::-1])
#     span    = last_a - first_a
#     f_planar = f_planar[first_a + int(0.15 * span) : first_a + int(0.85 * span)]

#     mu_t   = f_planar / (mass_est * 9.81)
#     med    = np.median(f_planar)
#     steady = np.abs(f_planar - med) <= np.median(np.abs(f_planar - med))
#     return float(np.median(mu_t[steady])), float(np.std(mu_t[steady]))


def run_object(obj: str, workspace_root: str) -> None:
    squash_file = os.path.join(workspace_root, "runtime_logs", obj, "arc_squash", "most_recent.npz")
    if not os.path.exists(squash_file):
        print(f"\n[{obj}] No squash log — skipping.")
        return
    print(f"\n{'='*60}\n  OBJECT: {obj}\n{'='*60}")
    _run_estimation(obj, os.path.join(workspace_root, "runtime_logs", obj), squash_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, default=None, choices=ALL_OBJECTS)
    parser.add_argument("--workspace", type=str, default=None)
    args = parser.parse_args()

    workspace_root = args.workspace or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    for obj in ([args.object] if args.object else ALL_OBJECTS):
        run_object(obj, workspace_root)
    plt.show()


if __name__ == "__main__":
    main()
