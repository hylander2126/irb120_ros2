#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from irb120_control.estimation.com_estimation import model_fwd_wrench, model_bkwd_wrench, compute_applied_wrench, compute_applied_wrench_surface
from irb120_control.estimation.helper_fns import rotvec_to_rot, quat_to_rotvec
from irb120_control.estimation.plotting_helper import plot_wrench_and_tipping, plot_torque_fit_result

ALL_OBJECTS = ["box", "heart", "flashlight", "monitor", "soda"]

_LPF_B,      _LPF_A      = butter(4, 6,   fs=500, btype='low')  # 6 Hz — removes high-freq sensor noise
_LPF_SLOW_B, _LPF_SLOW_A = butter(2, 0.5, fs=500, btype='low')  # 0.5 Hz — removes force-controller hunting


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
    ]:
        if not all(k in data for k in keys) or len(data[keys[0]]) == 0:
            raise KeyError(f"Missing keys {keys} in {filepath}.")

    # Three independently-sampled time axes: F/T @ 500 Hz, EE pose @ ~100 Hz, object detector @ ~2 Hz (unused)
    time_ft  = data["ft_time_s"]
    time_ee  = data["pose_time_s"]
    time_obj = data["obj_time_s"]

    f_meas_S = _lpf(np.column_stack([data["fx"], data["fy"], data["fz"]]))
    t_meas_S = _lpf(np.column_stack([data["tx"], data["ty"], data["tz"]]))

    p_ft_B   = np.column_stack([data["ft_px"], data["ft_py"], data["ft_pz"]])
    Q_ft     = np.column_stack([data["ft_qx"], data["ft_qy"], data["ft_qz"], data["ft_qw"]])
    p_ee_B   = np.column_stack([data["x"], data["y"], data["z"]])
    p_obj_B  = np.column_stack([data["obj_x"], data["obj_y"], data["obj_z"]])

    return time_ft, f_meas_S, t_meas_S, p_ft_B, Q_ft, time_ee, p_ee_B, time_obj, p_obj_B


def _run_estimation(obj: str, base_dir: str, squash_file: str) -> None:
    time_ft, f_meas_S, t_meas_S, p_ft_B, Q_ft, time_ee, p_ee_B, time_obj, p_obj_B = load_and_preprocess(squash_file)

    contact_ft_idx = np.argmax(np.linalg.norm(f_meas_S, axis=1) > 0.5) # argmax on bool array: first idx of norm > 0.5 N
    time_contact_start = time_ft[contact_ft_idx] # and the corresponding time

    # Pivot: near edge of object = pre-contact centroid x_min (closest to robot is approximate object frame)
    p_pivot_B = np.array([p_obj_B[time_obj <= time_contact_start, 0].min(), 0.0, 0.0])
    print(f"[{obj}] pivot x={p_pivot_B[0]:.4f} m")

    # p_pivot_B = np.array([0.5, 0, 0])

    # Proprioceptive tipping angle from EE triangle in X-Z plane
    contact_ee_idx = np.argmin(np.abs(time_ee - time_contact_start)) # index of first EE (ball center) position after contact
    p_contact = p_ee_B[contact_ee_idx]
    r0 = p_contact - p_pivot_B
    r_t = p_ee_B - p_pivot_B

    print(f"r0 from proprioception = {r0}")
    
    # Angle of moving r_t wrt stationary r0 in the X-Z plane (positive CCW, +Y into the screen)
    prop_angle_deg = -np.degrees(np.arctan2(r0[0] * r_t[:, 2] - r0[2] * r_t[:, 0],
                                           r0[0] * r_t[:, 0] + r0[2] * r_t[:, 2]))
    
    prop_angle_deg[:contact_ee_idx] = 0.0 # Set the angle to 0 before the contact
    print(f"[{obj}] min angle={prop_angle_deg.min():.2f}°  ee_samples={len(time_ee)}")

    pitch_B = np.deg2rad(np.interp(time_ft, time_ee, prop_angle_deg)) # Match the time of F/T data

    # Applied wrench in object frame — angles are Negative (object tips CCW when looking from -Y)
    contact_mask  = pitch_B < 0.0
    N_c_all = contact_mask.sum()

    # Build batched (N,4,4) homogeneous transforms for sensor and object frames
    R_ft_c  = rotvec_to_rot(quat_to_rotvec(Q_ft[contact_mask]))       # (N,3,3) sensor rotation in {B}
    T_B_sensor = np.zeros((N_c_all, 4, 4))
    T_B_sensor[:, :3, :3] = R_ft_c
    T_B_sensor[:, :3,  3] = p_ft_B[contact_mask]
    T_B_sensor[:,  3,  3] = 1.0

    pitch_c = pitch_B[contact_mask]
    rv_obj  = np.zeros((N_c_all, 3)); rv_obj[:, 1] = pitch_c
    R_obj_c = rotvec_to_rot(rv_obj)                                    # (N,3,3) object rotation in {B}
    T_B_obj = np.zeros((N_c_all, 4, 4))
    T_B_obj[:, :3, :3] = R_obj_c
    T_B_obj[:, :3,  3] = p_pivot_B                                     # pivot = object frame origin in {B}
    T_B_obj[:,  3,  3] = 1.0

    w_meas_S = np.hstack((f_meas_S[contact_mask], t_meas_S[contact_mask]))  # (N,6)

    ## HERE WE CAN REDEFINE THE r0 lever arm
    if obj == "box":
        r0 = np.array([0.023, 0.0, 0.3]) # 0.026
        COM_GT = np.array([0.05, 0.0, 0.15])
        MASS_GT = 0.635
    elif obj == "heart":
        r0 = np.array([0.023, 0.0, 0.2]) # 0.026
        COM_GT = np.array([0.0458, 0.0, 0.10])
        MASS_GT = 0.295
    elif obj == "soda":
        r0 = np.array([0.0, 0.0, 0.3]) # 0.055
        COM_GT = np.array([0.055, 0.0, 0.15])
        MASS_GT = 2.05


    w_app_O = model_bkwd_wrench(w_meas_S, T_B_sensor, T_B_obj, r0)

    print(f"\nr0 harcoded = {r0}")

    ## ================================================================
    # Force-component contribution to applied torque (diagnostic)
    # Decompose measured force into normal (object-surface Z) and tangential (object-surface X)
    # components in world frame, then compute each component's contribution to tau_y.
    # Normal = along object-frame Z axis; Tangent = along object-frame X axis.
    f_meas_W_c = np.einsum('nij,nj->ni', R_ft_c, f_meas_S[contact_mask])  # (N,3) in world frame

    Q_identity = np.zeros((N_c_all, 4)); Q_identity[:, 3] = 1.0           # R_B_S = I

    # Object-frame axes in world frame: normal = R_obj @ z_hat, tangent = R_obj @ x_hat
    pitch_c = pitch_B[contact_mask]
    rv_c = np.zeros((N_c_all, 3)); rv_c[:, 1] = pitch_c
    R_obj_c = rotvec_to_rot(rv_c)                                          # (N,3,3)
    n_hat = R_obj_c[:, :, 2]                                               # (N,3) object normal in world
    t_hat = R_obj_c[:, :, 0]                                               # (N,3) object tangent in world

    # Project world force onto normal and tangential directions
    f_n_scalar = np.einsum('ni,ni->n', f_meas_W_c, n_hat)                 # (N,) normal magnitude
    f_t_scalar = np.einsum('ni,ni->n', f_meas_W_c, t_hat)                 # (N,) tangential magnitude
    f_W_normal  = f_n_scalar[:, None] * n_hat                             # (N,3) normal component vector
    f_W_tangent = f_t_scalar[:, None] * t_hat                             # (N,3) tangential component vector

    w_app_normal  = compute_applied_wrench(f_W_normal,  Q_identity, pitch_c, r0)
    w_app_tangent = compute_applied_wrench(f_W_tangent, Q_identity, pitch_c, r0)

    tau_y_fn   = w_app_normal[:, 4]
    tau_y_ft   = w_app_tangent[:, 4]
    tau_y_full = tau_y_fn + tau_y_ft

    pitch_diag_deg = np.rad2deg(pitch_B[contact_mask])
    fig_diag, ax_diag = plt.subplots(figsize=(10, 5))
    ax_diag.plot(pitch_diag_deg, _lpf_slow(tau_y_full),    color='black',   linewidth=2, label='τ_y full (sum)')
    ax_diag.plot(pitch_diag_deg, _lpf_slow(w_app_O[:, 4]), color='gray',    linewidth=1, linestyle='--', label='τ_y (model_bkwd, verify)')
    ax_diag.plot(pitch_diag_deg, _lpf_slow(tau_y_fn),      color='tab:red', linewidth=2, label='τ_y from normal-F only')
    ax_diag.plot(pitch_diag_deg, _lpf_slow(tau_y_ft),      color='tab:blue',linewidth=2, label='τ_y from tangent-F only')
    ax_diag.plot(pitch_diag_deg, _lpf_slow(f_n_scalar),    color='tab:red', linewidth=1, linestyle='dashed', label='F_normal magnitude (N)')
    ax_diag.plot(pitch_diag_deg, _lpf_slow(f_t_scalar),    color='tab:blue',linewidth=1, linestyle='dashed', label='F_tangent magnitude (N)')
    ax_diag.axhline(0, color='gray', linewidth=1)
    ax_diag.set_xlabel('Object angle (deg)')
    ax_diag.set_ylabel('τ_y contribution (N·m) / Force magnitude (N)')
    ax_diag.set_title(f'[{obj}] Applied torque: normal-F vs tangent-F contribution')
    ax_diag.legend()
    ax_diag.grid(True, alpha=0.4)
    plt.tight_layout()
    fig_diag.savefig(os.path.join(base_dir, "torque_force_decomp.png"), dpi=150, bbox_inches="tight")
    ## ================================================================

    # 0.5 Hz LPF on applied wrench to suppress force-controller hunting for fitting
    pitch_contact  = pitch_B[contact_mask]
    N_c            = len(pitch_contact)
    w_app_O_smooth = _lpf_slow(w_app_O)

    # Raw F/T overview plot (post-contact only) with smoothed overlay
    time_ft_rel     = time_ft[contact_ft_idx:] - time_ft[contact_ft_idx]
    R_ft_all        = rotvec_to_rot(quat_to_rotvec(Q_ft[contact_ft_idx:]))
    f_meas_W        = np.einsum('nij,nj->ni', R_ft_all, f_meas_S[contact_ft_idx:])
    f_meas_W_smooth = _lpf_slow(f_meas_W)
    time_contact_xp  = time_ft[contact_mask] - time_ft[contact_ft_idx]
    tau_y_on_ft      = np.interp(time_ft_rel, time_contact_xp, w_app_O[:, 4],        left=0.0, right=0.0)
    tau_y_smooth_ft  = np.interp(time_ft_rel, time_contact_xp, w_app_O_smooth[:, 4], left=0.0, right=0.0)
    plot_wrench_and_tipping(time_ft_rel, f_meas_W, 2*tau_y_on_ft,
                            pitch_rad=pitch_B[contact_ft_idx:], torque_label="(2x) tau_y",
                            force_xyz_smooth=f_meas_W_smooth, torque_primary_smooth=tau_y_smooth_ft,
                            contact_time=0.0, title=f"[{obj}] Raw F/T + tipping angle", show=False,
                            save_path=os.path.join(base_dir, "squash_forces.png"))

    # Calculated Applied Wrench overview plot
    # f_app_O_on_ft = np.zeros((len(time_ft_rel), 3))
    # f_app_O_smooth_ft = np.zeros((len(time_ft_rel), 3))
    # for i in range(3):
    #     f_app_O_on_ft[:, i] = np.interp(time_ft_rel, time_contact_xp, w_app_O[:, i], left=0.0, right=0.0)
    #     f_app_O_smooth_ft[:, i] = np.interp(time_ft_rel, time_contact_xp, w_app_O_smooth[:, i], left=0.0, right=0.0)

    # plot_wrench_and_tipping(time_ft_rel, f_app_O_on_ft, tau_y_on_ft,
    #                         pitch_rad=pitch_B[contact_ft_idx:], torque_label="tau_y",
    #                         force_xyz_smooth=f_app_O_smooth_ft, torque_primary_smooth=tau_y_smooth_ft,
    #                         contact_time=0.0, title=f"[{obj}] Applied Wrench (Object Frame) + tipping angle", show=False,
    #                         save_path=os.path.join(base_dir, "applied_wrench.png"))

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

    def _fit_phase(phase_sel, label, tau_y_override=None):
        """Fit COM_z and mass from applied torque.

        phase_sel: (N_c,) bool array selecting samples in the tipping phase to fit
        label: string label for printing
        tau_y_override: if provided, use this (N,) array as the measured tau_y
        instead of w_app_O_smooth[:, 4]. Useful for fitting against a single
        force component's torque contribution (e.g. F_x only).
        """
        if phase_sel.sum() < 10:
            print(f"[{obj}] Too few {label} samples ({phase_sel.sum()}) — skipping.")
            return None, None
        p_ph    = pitch_contact[phase_sel]
        rv_ph   = np.column_stack([np.zeros(phase_sel.sum()), p_ph, np.zeros(phase_sel.sum())]) # rotation vec for y
        tau_y_meas = tau_y_override[phase_sel] if tau_y_override is not None else w_app_O_smooth[phase_sel, 4]
        com_z0 = mass0 = 0.1
        def _residual(params):
            w_grav, _ = model_fwd_wrench(rv_ph, np.array([COM_GT[0], 0.0, params[0]]), params[1], 0.0)
            return w_grav[:, 4] + tau_y_meas
        res = least_squares(_residual, x0=[com_z0, mass0],
                            bounds=([1e-6, 1e-6], [np.inf, np.inf]), method='trf')
        com_z, mass = res.x
        print(f"  [{obj}] {label:>7s} fit — COM_z={com_z:.4f} m  Mass={mass:.4f} kg  θ*={np.degrees(np.arctan2(COM_GT[0], com_z)):.1f}°")
        return com_z, mass


    print(f"\n--- [{obj}] PHASE ESTIMATES (full torque) ---")
    com_z_push,    mass_push    = _fit_phase(push_tip_sel,    "push")
    com_z_retract, mass_retract = _fit_phase(retract_tip_sel, "retract")
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
        tau_pred_push = -w_grav_push[:, 4]

    tau_pred_retract = np.zeros(tip_sel.sum())
    if com_z_retract is not None:
        w_grav_ret, _ = model_fwd_wrench(rv_fit, np.array([COM_GT[0], 0.0, com_z_retract]), mass_retract, 0.0)
        tau_pred_retract = -w_grav_ret[:, 4]

    theta_push    = np.arctan2(COM_GT[0], com_z_push)    if com_z_push    is not None else None
    theta_retract = np.arctan2(COM_GT[0], com_z_retract) if com_z_retract is not None else None

    plot_torque_fit_result(
        pitch_rad=pitch_fit,
        tau_meas=w_app_O_smooth[tip_sel, 4],
        tau_pred_push=tau_pred_push,
        theta_star_push_rad=theta_push if theta_push is not None else 0.0,
        tau_pred_retract=tau_pred_retract if com_z_retract is not None else None,
        theta_star_retract_rad=theta_retract,
        push_sel=push_sel_plot,
        title=f"[{obj}] Torque fit result (full torque)",
        show=False,
        save_path=os.path.join(base_dir, "torque_fit.png"),
    )


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
