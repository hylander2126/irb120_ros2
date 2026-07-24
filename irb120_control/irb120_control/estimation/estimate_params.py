#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from irb120_control.estimation.com_estimation import model_fwd_wrench, model_bkwd_wrench, construct_T, rotvec_between
from irb120_control.estimation.helper_fns import rotvec_to_rot, quat_to_rotvec
from irb120_control.estimation.plotting_helper import plot_wrench_and_tipping, plot_torque_fit_result, plot_raw_forces

ALL_OBJECTS = ["box", "heart", "flashlight", "soda", "monitor"]
PLOT_PER_OBJECT = False  # set True to show individual estimation figures per object
# ALL_OBJECTS = ["flashlight"]
# ALL_OBJECTS = ["soda"]

STATE_SQUASH = 1
STATE_LULL = 2
STATE_ARC = 3
STATE_UNARC = 4
STATE_RETRACT = 5

FT_WRENCH_ORIGIN_OFFSET_X = 0.08225

_LPF_B, _LPF_A = butter(4, 6, fs=500, btype='low')

def _lpf(x, axis=0):
    return filtfilt(_LPF_B, _LPF_A, x, axis=axis) if x.shape[0] > 20 else x


def load_and_preprocess(filepath):
    data = np.load(filepath)

    def _require(keys):
        if not all(k in data for k in keys) or any(len(data[k]) == 0 for k in keys):
            raise KeyError(f"Missing keys {keys} in {filepath}.")

    required_keys = (
        "pose_time_s", "x", "y", "z", "qx", "qy", "qz", "qw",
        "controller_state_id",
        "obj_time_s", "obj_x", "obj_y", "obj_z", "obj_qx", "obj_qy", "obj_qz", "obj_qw",
        "ft_time_s", "fx", "fy", "fz", "tx", "ty", "tz",
        "ft_px", "ft_py", "ft_pz", "ft_qx", "ft_qy", "ft_qz", "ft_qw",
    )
    _require(required_keys)

    # EE pose is the sparsest stream (~100 Hz); use it as the common time grid.
    # F/T is subsampled onto this grid via interpolation here so _run_estimation never has to manage multiple time axes.
    time  = data["pose_time_s"]
    time_obj = data["obj_time_s"]
    state_id = data["controller_state_id"].astype(int)
    if len(state_id) != len(time):
        raise ValueError(
            f"controller_state_id length ({len(state_id)}) does not match pose_time_s length ({len(time)}) in {filepath}."
        )

    p_ee_B = np.column_stack([data["x"], data["y"], data["z"]])
    Q_obj    = np.column_stack([data["obj_qx"], data["obj_qy"], data["obj_qz"], data["obj_qw"]])

    # Subsample everything onto the EE time grid
    def _interp_cols(t_src, arr, t_dst):
        return np.column_stack([np.interp(t_dst, t_src, arr[:, i]) for i in range(arr.shape[1])])

    time_ft = data["ft_time_s"]
    ft_aligned_keys = ("fx", "fy", "fz", "tx", "ty", "tz", "ft_px", "ft_py", "ft_pz", "ft_qx", "ft_qy", "ft_qz", "ft_qw")
    bad_lengths = {k: len(data[k]) for k in ft_aligned_keys if len(data[k]) != len(time_ft)}
    if bad_lengths:
        raise ValueError(f"F/T-aligned key lengths do not match ft_time_s ({len(time_ft)}) in {filepath}: {bad_lengths}")

    f_ft = np.column_stack([data["fx"], data["fy"], data["fz"]])
    t_ft = np.column_stack([data["tx"], data["ty"], data["tz"]])
    p_ft_ft = np.column_stack([data["ft_px"], data["ft_py"], data["ft_pz"]])
    Q_ft_ft = np.column_stack([data["ft_qx"], data["ft_qy"], data["ft_qz"], data["ft_qw"]])

    f_meas_S = _interp_cols(time_ft, f_ft, time)
    t_meas_S = _interp_cols(time_ft, t_ft, time)
    p_ft_B = _interp_cols(time_ft, p_ft_ft, time)
    Q_ft = _interp_cols(time_ft, Q_ft_ft, time)
    Q_obj = _interp_cols(time_obj, Q_obj, time)

    # The logged TF pose is ft_link at base of sensor body, but NetFT torque channels behave as moments about the distal face/finger base.
    R_ft_B = rotvec_to_rot(quat_to_rotvec(Q_ft))
    p_ft_B = p_ft_B + np.einsum(
        "nij,j->ni",
        R_ft_B,
        np.array([FT_WRENCH_ORIGIN_OFFSET_X, 0.0, 0.0]), # Shift the wrench origin before applying adjoint transforms.
    )

    return time, f_meas_S, t_meas_S, p_ft_B, Q_ft, p_ee_B, Q_obj, state_id


def estimate_friction(push_log_path: str, mass_est: float, mass_gt: float) -> tuple:
    """
    Estimate mu_t from slip-onset tangential force in the push log.
    Returns (mu_est, mu_gt) where mu_gt uses the known ground-truth mass.
    The PRE-contact window (first 10% of the log) is used to estimate the
    F/T zero bias in f_tan and subtracted before computing mu.
    """
    data = np.load(push_log_path)
    fx = _lpf(data["fx"])
    fy = _lpf(data["fy"])
    f_tan = np.sqrt(fx**2 + fy**2)

    n = len(f_tan)
    f_bias = float(np.mean(f_tan[:int(0.10 * n)]))  # PRE-contact zero offset
    f_slip = float(np.median(f_tan[int(0.50 * n):int(0.85 * n)])) - f_bias

    mu_est = f_slip / (mass_est * 9.81)
    mu_gt  = f_slip / (mass_gt  * 9.81)
    return mu_est, mu_gt


def _run_estimation(obj: str, base_dir: str, squash_file: str, push_file: str | None) -> dict | None:

    ## HERE WE CAN REDEFINE THE r0 lever arm
    if obj == "box":
        # r0 = np.array([0.01, 0.0, 0.3]) # 0.026 old value, new is ~1.4 cm
        MASS_GT = 0.676
        COM_GT = np.array([0.05, 0.0, 0.15])
        THETA_GT_DEG = 17.532
    elif obj == "heart":
        # r0 = np.array([0.01, 0.0, 0.2]) # 0.026
        MASS_GT = 0.239
        COM_GT = np.array([0.0458, 0.0, 0.10])
        THETA_GT_DEG = 23.984
    elif obj == "flashlight":
        # r0 = np.array([0.028, 0.0, 0.2]) # 0.028
        MASS_GT = 0.387
        COM_GT = np.array([0.028, 0.0, 0.0938])
        THETA_GT_DEG = 15.126
    elif obj == "soda":
        # r0 = np.array([0.055, 0.0, 0.3]) # 0.055
        MASS_GT = 2.054
        COM_GT = np.array([0.0525, 0.0, 0.15])
        THETA_GT_DEG = 21.801 #20.126
    elif obj == "monitor":
        # r0 = np.array([0.055, 0.0, 0.3]) # 0.055
        MASS_GT = 5.04
        COM_GT = np.array([0.06, 0.0, 0.232]) # CoM are approximate. +- 1cm
        THETA_GT_DEG = 14.5 # Theta star is approximate. +- 0.2 deg

    time, f_meas_S, t_meas_S, p_ft_B, Q_ft, p_ee_B, Q_obj, state_id = load_and_preprocess(squash_file)

    p_pivot_B = np.array([0.61, 0, 0])# -0.021]) # 0.6 Near-exact pivot from pre-defined object frame (I reset obj to known pose)s

    # Bootstrap from controller state timing rather than inferring contact/release.
    in_contact = np.isin(state_id, [STATE_LULL, STATE_ARC, STATE_UNARC, STATE_RETRACT])
    print(f"[{obj}] Using controller_state_id for contact/phase segmentation.")
    r_t = p_ee_B - p_pivot_B
    r0  = r_t[np.argmax(in_contact)]
    print(f"[{obj}] p_pivot_B: {p_pivot_B}, p_ee_B[contact]: {np.round(p_ee_B[np.argmax(in_contact)], 3)}, r0: {np.round(r0, 3)}")
    rot_vec_obj = rotvec_between(r0, r_t)  # (N, 3) object rotation vector in {B} on unified time grid
    rot_vec_obj[~in_contact] = 0.0         # Keep only the contact window; zero outside

    # Contact mask: within the force window and past the small-angle deadband. (Y neg as obj tips)
    contact_mask = np.isin(state_id, [STATE_ARC, STATE_UNARC]) & (rot_vec_obj[:, 1] < -np.deg2rad(1.0))

    # Build batched (N,4,4) homogeneous transforms for sensor and object frames
    T_B_sensor = construct_T(p_ft_B, quat=Q_ft)
    T_B_obj    = construct_T(np.tile(p_pivot_B, (len(time), 1)), rv=rot_vec_obj) # const pos
    w_meas_S = np.hstack((t_meas_S, f_meas_S))  # (N,6) [tau, f] convention (Modern Robotics)
    w_app_O = model_bkwd_wrench(w_meas_S[contact_mask],
                                T_B_sensor[contact_mask],
                                T_B_obj[contact_mask])

    ## ======== One figure per object: 3 subplots side-by-side =========
    if PLOT_PER_OBJECT:
        fig_obj, axes_obj = plt.subplots(1, 3, figsize=(24, 6))
        fig_obj.suptitle(f"[{obj}]", fontsize=14, fontweight="bold")
        time_plot = time[contact_mask] - time[contact_mask][0]
        # plot_raw_forces(time_plot, f_meas_S[contact_mask], title="Measured Force (Sensor Frame)", show=False) # On it's own
        plot_wrench_and_tipping(time_plot, w_app_O[:, 3:], w_app_O[:, :3],
                                ax=axes_obj[0],
                                pitch_rad=rot_vec_obj[contact_mask, 1], torque_label="τ",
                                contact_time=0.0, title=f"Applied Wrench (Object Frame)", show=False)

    ## Trim to contact window and then separate tipping from retract phase
    rot_vec_during_contact      = rot_vec_obj[contact_mask]    # (N_c, 3) full rotation vectors during contact
    y_pitch_during_contact = rot_vec_during_contact[:, 1] # (N_c,)   y-axis pitch for phase/threshold logic (plot only)
    state_contact               = state_id[contact_mask]

    # Trim ~1.6° from each phase boundary using controller state labels directly.
    trim = 0.0 #np.deg2rad(1.6)
    arc_phase = state_contact == STATE_ARC
    unarc_phase = state_contact == STATE_UNARC

    def _trim_phase(phase_mask, pitch_signal):
        if not np.any(phase_mask):
            return np.zeros_like(phase_mask, dtype=bool)
        phase_min = pitch_signal[phase_mask].min()
        return phase_mask & (pitch_signal < -trim) & (pitch_signal > phase_min + trim)

    arc_phase_trimmed    = _trim_phase(arc_phase, y_pitch_during_contact)
    unarc_phase_trimmed  = _trim_phase(unarc_phase, y_pitch_during_contact)
    tip_sel = arc_phase_trimmed | unarc_phase_trimmed

    TIP_AXIS = rot_vec_during_contact.mean(0) / np.linalg.norm(rot_vec_during_contact.mean(0))
    print(f"\n[{obj}] FORCING TIP AXIS TO: {np.round(TIP_AXIS, 2)}")
    print(f"[{obj}] And testing p_pivot_B at: {p_pivot_B}\n")

    theta_gt_deg = -THETA_GT_DEG #np.degrees(np.arctan2(COM_GT[0], COM_GT[2]))

    # --- Two estimation methods per phase ---
    # A: f_x zero-crossing → z_c fixed, mass-only torque fit
    # B: joint (mass, z_c) fit from torque balance directly
    def _fit_phase(phase_sel, label, COM_GT):
        y_pitch_deg = np.rad2deg(y_pitch_during_contact[phase_sel])
        rv_ph = rot_vec_during_contact[phase_sel]
        tau_meas = w_app_O[phase_sel, :3] @ TIP_AXIS

        # Method A — use torque-corrected signal: g_x = f_x - (r0_x/r0_z)*f_z
        # This zeros at theta* for any contact geometry, not just r0_x=0
        gx = w_app_O[phase_sel, 3] - (r0[0] / r0[2]) * w_app_O[phase_sel, 5]
        fx_coeffs = np.polyfit(y_pitch_deg, gx, 1)
        theta_fx_deg = -fx_coeffs[1] / fx_coeffs[0]
        com_z_fx = COM_GT[0] / np.tan(np.deg2rad(abs(theta_fx_deg)))
        mass_fx = least_squares(
            lambda p: ((model_fwd_wrench(rv_ph, np.array([COM_GT[0], 0.0, com_z_fx]), p[0])[:, :3] @ TIP_AXIS) - tau_meas).ravel(),
            x0=[MASS_GT], bounds=([1e-6], [np.inf]), method='trf').x[0]
        print(f"  [{obj}] {label:>7s} [A: f_x→θ*] theta*={theta_fx_deg:.2f}°  z_c={com_z_fx:.4f}m  m={mass_fx:.4f}kg")

        # Method B
        res_tau = least_squares(
            lambda p: ((model_fwd_wrench(rv_ph, np.array([COM_GT[0], 0.0, p[1]]), p[0])[:, :3] @ TIP_AXIS) - tau_meas).ravel(),
            x0=[MASS_GT, COM_GT[2]], bounds=([1e-6, 1e-3], [np.inf, np.inf]), method='trf')
        mass_tau, com_z_tau = res_tau.x
        theta_tau_deg = -np.degrees(np.arctan2(COM_GT[0], com_z_tau))
        print(f"  [{obj}] {label:>7s} [B: τ jnt ] theta*={theta_tau_deg:.2f}°  z_c={com_z_tau:.4f}m  m={mass_tau:.4f}kg")

        estimates = {
            "A": {"m": mass_fx,  "zc": com_z_fx,  "theta": theta_fx_deg},
            "B": {"m": mass_tau, "zc": com_z_tau, "theta": theta_tau_deg},
        }
        return com_z_fx, mass_fx, theta_fx_deg, fx_coeffs, com_z_tau, mass_tau, theta_tau_deg, estimates


    print(f"\n--- [{obj}] PHASE ESTIMATES ---")
    com_z_push,    mass_push,    theta_fx_push_deg,    fx_coeffs_push,    com_z_tau_push,    mass_tau_push,    theta_tau_push_deg,    est_arc   = _fit_phase(arc_phase_trimmed,   "ARC",   COM_GT)
    com_z_retract, mass_retract, theta_fx_retract_deg, fx_coeffs_retract, com_z_tau_retract, mass_tau_retract, theta_tau_retract_deg, est_unarc = _fit_phase(unarc_phase_trimmed, "UNARC", COM_GT)
    print(f"  [{obj}] Ground truth — COM_z={COM_GT[2]:.4f} m  Mass={MASS_GT:.4f} kg  theta*={theta_gt_deg:.1f}deg")

    # Average push/retract Method B estimates (hysteresis cancellation)
    mass_est      = 0.5 * (mass_tau_push      + mass_tau_retract)
    com_z_est     = 0.5 * (com_z_tau_push     + com_z_tau_retract)
    theta_est_deg = 0.5 * (theta_tau_push_deg + theta_tau_retract_deg)
    tau_pred_arc   = model_fwd_wrench(rot_vec_during_contact[tip_sel], np.array([COM_GT[0], 0.0, com_z_tau_push]),    mass_tau_push)[:, 1]
    tau_pred_unarc = model_fwd_wrench(rot_vec_during_contact[tip_sel], np.array([COM_GT[0], 0.0, com_z_tau_retract]), mass_tau_retract)[:, 1]

    if PLOT_PER_OBJECT:
        plot_torque_fit_result(
            pitch_rad=-y_pitch_during_contact[tip_sel],  # sign-flip for plotting: positive = larger tip
            tau_meas=w_app_O[tip_sel, 1],
            tau_pred_push=tau_pred_arc,
            theta_star_push_rad=np.deg2rad(theta_tau_push_deg),
            ax=axes_obj[2],
            tau_pred_retract=tau_pred_unarc,
            theta_star_retract_rad=np.deg2rad(theta_tau_retract_deg),
            theta_star_gt_rad=np.deg2rad(theta_gt_deg),
            push_sel=arc_phase[tip_sel],
            title=f"Torque fit (B: joint τ)",
            show=False,
        )

        # Extrapolation range: span observed data plus padding toward phase zero crossings.
        extrap_bounds = [np.rad2deg(y_pitch_during_contact[tip_sel]).min(), np.rad2deg(y_pitch_during_contact[tip_sel]).max()]
        extrap_bounds.append(theta_fx_push_deg)
        extrap_bounds.append(theta_fx_retract_deg)
        theta_extrap = np.linspace(min(extrap_bounds) - 1.0, max(extrap_bounds) + 1.0, 200)

        # Plot g_x = f_x - (r0_x/r0_z)*f_z data and extrapolated line (Method A)
        gx_plot = w_app_O[tip_sel, 3] - (r0[0] / r0[2]) * w_app_O[tip_sel, 5]
        axes_obj[1].plot(np.rad2deg(y_pitch_during_contact[tip_sel]), gx_plot, 'o', markersize=3, label="g_x (corrected)")
        axes_obj[1].plot(theta_extrap, np.polyval(fx_coeffs_push, theta_extrap), color='tab:blue', linestyle='--', label="linear fit (ARC)")
        axes_obj[1].axvline(theta_fx_push_deg, color='tab:blue', linestyle=':', label=f"ARC A θ*={theta_fx_push_deg:.2f}°")
        axes_obj[1].axvline(theta_tau_push_deg, color='tab:blue', linestyle='-', linewidth=1.5, label=f"ARC B θ*={theta_tau_push_deg:.2f}°")
        axes_obj[1].plot(theta_extrap, np.polyval(fx_coeffs_retract, theta_extrap), color='tab:orange', linestyle='-.', label="linear fit (UNARC)")
        axes_obj[1].axvline(theta_fx_retract_deg, color='tab:orange', linestyle=':', label=f"UNARC A θ*={theta_fx_retract_deg:.2f}°")
        axes_obj[1].axvline(theta_tau_retract_deg, color='tab:orange', linestyle='-', linewidth=1.5, label=f"UNARC B θ*={theta_tau_retract_deg:.2f}°")
        axes_obj[1].axvline(theta_gt_deg, color='green', linestyle=':', label=f"theta* GT = {theta_gt_deg:.1f}deg")
        axes_obj[1].axhline(0, color='k', linewidth=0.8)
        axes_obj[1].set_xlabel("Pitch angle (degrees)")
        axes_obj[1].set_ylabel("g_x = f_x − (r0_x/r0_z)·f_z  (N)")
        axes_obj[1].set_title("g_x zero-crossing (A: dashed) vs joint τ (B: solid)")
        axes_obj[1].legend()
        axes_obj[1].grid(True)

        # fig_obj.tight_layout()
        fig_obj.savefig(os.path.join(base_dir, "estimation_summary.png"), dpi=150, bbox_inches="tight")

    # Friction estimation from push log
    mu_est = mu_gt = None
    if push_file and os.path.exists(push_file):
        mu_est, mu_gt = estimate_friction(push_file, mass_est, MASS_GT)
        print(f"  [{obj}] Friction: mu_est={mu_est:.3f}  mu_gt={mu_gt:.3f}")
    else:
        print(f"  [{obj}] No push log — friction skipped.")

    return {
        "obj":          obj,
        "mass_est":     mass_est,
        "mass_gt":      MASS_GT,
        "com_z_est":    com_z_est,
        "com_z_gt":     COM_GT[2],
        "theta_est_deg": abs(theta_est_deg),   # store as positive tipping angle
        "theta_gt_deg":  THETA_GT_DEG,
        "mu_est":       mu_est,
        "mu_gt":        mu_gt,
        "phase_estimates": {
            "arc_A":   {"m": est_arc["A"]["m"],   "zc": est_arc["A"]["zc"],   "theta": abs(est_arc["A"]["theta"])},
            "arc_B":   {"m": est_arc["B"]["m"],   "zc": est_arc["B"]["zc"],   "theta": abs(est_arc["B"]["theta"])},
            "unarc_A": {"m": est_unarc["A"]["m"], "zc": est_unarc["A"]["zc"], "theta": abs(est_unarc["A"]["theta"])},
            "unarc_B": {"m": est_unarc["B"]["m"], "zc": est_unarc["B"]["zc"], "theta": abs(est_unarc["B"]["theta"])},
        },
    }


def plot_results_summary(results: list, save_dir: str) -> None:
    """
    Two-figure results summary (2×2 layout for single-column papers):
      Fig 1 — grouped bar chart: estimated vs GT for m, z_c, θ*, µ_t
      Fig 2 — absolute error bar chart for the same four parameters
    """
    results = [r for r in results if r["obj"] != "soda"]
    objs   = [r["obj"] for r in results]
    n      = len(objs)
    x      = np.arange(n)
    labels = ["Flash." if o == "flashlight" else o.capitalize() for o in objs]

    mass_est    = np.array([r["mass_est"]      for r in results])
    mass_gt     = np.array([r["mass_gt"]       for r in results])
    com_z_est   = np.array([r["com_z_est"]     for r in results]) * 100   # m → cm
    com_z_gt    = np.array([r["com_z_gt"]      for r in results]) * 100
    theta_est   = np.array([r["theta_est_deg"] for r in results])
    theta_gt    = np.array([r["theta_gt_deg"]  for r in results])
    mu_mask     = np.array([r["mu_est"] is not None for r in results])
    mu_est      = np.array([r["mu_est"] if r["mu_est"] is not None else 0.0 for r in results])
    mu_gt       = np.array([r["mu_gt"]  if r["mu_gt"]  is not None else 0.0 for r in results])

    bar_w   = 0.75
    col_est = "#2196F3"
    col_gt  = "#4CAF50"
    err_col = "#E53935"
    fs_tick  = 28
    fs_label = 30
    fs_title = 30
    fs_leg   = 24

    mass_err  = np.abs(mass_est  - mass_gt)
    com_z_err = np.abs(com_z_est - com_z_gt)
    theta_err = np.abs(theta_est - theta_gt)
    mu_err    = np.where(mu_mask, np.abs(mu_est - mu_gt), np.nan)

    mass_rel  = mass_err  / mass_gt  * 100
    com_z_rel = com_z_err / com_z_gt * 100
    theta_rel = theta_err / theta_gt * 100
    mu_rel    = np.where(mu_mask, mu_err / mu_gt * 100, np.nan)

    # ── Figure 1: pseudo-whisker bars — one bar per object per param ─────────
    # Bar rises to max(est, gt). Base (0 → lower value) takes the color of
    # whichever quantity is smaller; the overshoot (lower → higher) is red.
    from matplotlib.patches import Patch
    abs_params = [
        (mass_est,  mass_gt,  "kg",  r"$m$",        None),
        (com_z_est, com_z_gt, "cm",  r"$z_c$",      None),
        (theta_est, theta_gt, "deg", r"$\theta^*$", None),
        (mu_est,    mu_gt,    "",    r"$\mu_t$",    mu_mask),
    ]
    fig1, axes = plt.subplots(1, 4, figsize=(28, 8))
    fig1.subplots_adjust(wspace=0.25)

    for idx, (ax, (est_vals, gt_vals, unit, param_label, mask)) in enumerate(zip(axes, abs_params)):
        for i, (ev, gv) in enumerate(zip(est_vals, gt_vals)):
            if mask is not None and not mask[i]:
                ax.text(i, 0, "N/A", ha='center', va='bottom',
                        fontsize=fs_tick, color='gray')
                continue
            lo, hi = min(ev, gv), max(ev, gv)
            base_col = col_est if ev <= gv else col_gt
            ax.bar(i, lo,      width=bar_w, color=base_col, alpha=0.85)
            ax.bar(i, hi - lo, width=bar_w, bottom=lo, color=err_col, alpha=0.90)

        ax.set_xticks(x)
        # rotation_mode='anchor' with ha='right' pivots the label on its right edge,
        # which visually centers each label under its bar group rather than hanging left
        ax.set_xticklabels(labels, fontsize=fs_tick + 2, rotation=35,
                           ha='right', rotation_mode='anchor')
        ax.set_ylabel(unit, fontsize=fs_label + 3)
        ax.tick_params(axis='y', labelsize=fs_tick + 2)
        ax.grid(axis='y', alpha=0.4)
        ax.set_ylim(bottom=0)
        ax.text(0.5, 0.93, param_label, ha='center', va='top',
                fontsize=44, fontweight='bold', transform=ax.transAxes)
        if idx == 0:
            ax.legend(handles=[
                Patch(color=col_est, alpha=0.85, label="Est ≤ GT"),
                Patch(color=col_gt,  alpha=0.85, label="GT < Est"),
                Patch(color=err_col, alpha=0.90, label="Error"),
            ], fontsize=fs_leg, loc='center left')

    fig1.tight_layout(rect=[0, 0, 1, 1])
    fig1.savefig(os.path.join(save_dir, "results_bar.png"), dpi=150, bbox_inches="tight")
    print("Saved results_bar.png")

    # ── Figure 2: relative errors % (single shared axis, 4 groups separated by vertical lines) ──
    params = [
        (mass_rel,  r"$|\Delta m|/m$",               None),
        (com_z_rel, r"$|\Delta z_c|/z_c$",           None),
        (theta_rel, r"$|\Delta \theta^*|/\theta^*$", None),
        (mu_rel,    r"$|\Delta \mu_t|/\mu_t$",       mu_mask),
    ]
    num_groups = len(params)
    bar_width = 0.7
    total_bars = num_groups * n
    all_xs = np.arange(total_bars, dtype=float)  # evenly spaced, no gap

    fig2, ax2 = plt.subplots(figsize=(4 * n + 4, 8))

    for g, (errs, _, mask) in enumerate(params):
        xs = all_xs[g * n:(g + 1) * n]
        colors = [err_col if (mask is None or mask[i]) else "#BDBDBD" for i in range(n)]
        ax2.bar(xs, np.where(np.isnan(errs), 0, errs), width=bar_width, color=colors, alpha=0.85)
        if mask is not None:
            for i, m in enumerate(mask):
                if not m:
                    ax2.text(xs[i], 0.5, "N/A", ha='center', va='bottom', fontsize=fs_tick, color='gray')

    ax2.set_xticks(all_xs)
    ax2.set_xticklabels(labels * num_groups, fontsize=fs_tick + 2, rotation=35, ha='right')
    ax2.set_ylabel("Relative error (%)", fontsize=fs_label + 3)
    ax2.tick_params(axis='y', labelsize=fs_tick + 2)
    ax2.grid(axis='y', alpha=0.4)
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=-0.5, right=total_bars - 0.5)

    # Vertical separators between groups and bold parameter labels inside each region
    ymax = ax2.get_ylim()[1]
    for g, (_, param_label, _) in enumerate(params):
        # Label centered in the group, placed at 88% of ymax so it sits clearly inside
        group_center_x = g * n + (n - 1) / 2.0
        ax2.text(group_center_x, ymax * 0.88, param_label,
                 ha='center', va='top', fontsize=36, fontweight='bold',
                 transform=ax2.transData)
        if g > 0:
            ax2.axvline(g * n - 0.5, color='#333333', linewidth=1.2, linestyle='-')

    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "results_error.png"), dpi=150, bbox_inches="tight")
    print("Saved results_error.png")

    # ── Print summary table ─────────────────────────────────────────────────────
    print(f"\n{'Object':<12} {'|Δm|%':>8} {'|Δzc|%':>8} {'|Δθ*|%':>8} {'|Δμt|%':>8}")
    print("-" * 48)
    for r, mr, cr, tr, mur in zip(results, mass_rel, com_z_rel, theta_rel, mu_rel):
        mu_str = f"{mur:>7.1f}" if not np.isnan(mur) else "    N/A"
        print(f"  {r['obj']:<10} {mr:>7.1f}% {cr:>7.1f}% {tr:>7.1f}% {mu_str}%")


def plot_soda_summary(results: list, save_dir: str) -> None:
    """
    Two separate figures for soda, saved individually for manual composition:
      soda_bar.png   — pseudo-whisker absolute values, 4 subplots (one per param)
      soda_error.png — relative error % bars, one bar per param
    No title. Parameter labels match the bold in-axes style of the summary plots.
    """
    from matplotlib.patches import Patch

    soda = next((r for r in results if r["obj"] == "soda"), None)
    if soda is None:
        return

    bar_w   = 0.75
    col_est = "#2196F3"
    col_gt  = "#4CAF50"
    err_col = "#E53935"
    fs_tick  = 28
    fs_label = 30
    fs_leg   = 24
    fs_param = 44

    has_mu     = soda["mu_est"] is not None
    mu_est_val = soda["mu_est"] if has_mu else 0.0
    mu_gt_val  = soda["mu_gt"]  if has_mu else 0.0

    abs_params = [
        (soda["mass_est"],         soda["mass_gt"],         "kg",  r"$m$",        True),
        (soda["com_z_est"] * 100,  soda["com_z_gt"]  * 100, "cm",  r"$z_c$",      True),
        (soda["theta_est_deg"],    soda["theta_gt_deg"],    "deg", r"$\theta^*$", True),
        (mu_est_val,               mu_gt_val,               "",    r"$\mu_t$",    has_mu),
    ]

    def _rel(est, gt, valid):
        if not valid or gt == 0:
            return float("nan")
        return abs(est - gt) / abs(gt) * 100

    rel_errors = [_rel(ev, gv, valid) for ev, gv, _, _, valid in abs_params]
    rel_param_labels = [r"$m$", r"$z_c$", r"$\theta^*$", r"$\mu_t$"]

    # ── Fig A: pseudo-whisker absolute, 4 independent subplots ───────────────
    figA, axesA = plt.subplots(1, 4, figsize=(18, 7))
    figA.subplots_adjust(wspace=0.65, bottom=0.18)

    for idx, (ax, (ev, gv, unit, param_label, valid)) in enumerate(zip(axesA, abs_params)):
        if not valid:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                    fontsize=fs_tick, color='gray', transform=ax.transAxes)
        else:
            lo, hi = min(ev, gv), max(ev, gv)
            base_col = col_est if ev <= gv else col_gt
            ax.bar(0, lo,      width=bar_w, color=base_col, alpha=0.85)
            ax.bar(0, hi - lo, width=bar_w, bottom=lo, color=err_col, alpha=0.90)

        ax.set_xticks([])   # no per-subplot x-tick
        ax.set_ylabel(unit, fontsize=fs_label + 3)
        ax.tick_params(axis='y', labelsize=fs_tick + 2)
        ax.grid(axis='y', alpha=0.4)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.75, 0.75)
        ax.text(0.5, 0.93, param_label, ha='center', va='top',
                fontsize=fs_param, fontweight='bold', transform=ax.transAxes)
    # Legend placed outside all axes, to the right of the last subplot
    axesA[-1].legend(handles=[
        Patch(color=col_est, alpha=0.85, label="Est ≤ GT"),
        Patch(color=col_gt,  alpha=0.85, label="GT < Est"),
        Patch(color=err_col, alpha=0.90, label="Error"),
    ], fontsize=fs_leg, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # Single "Soda" label centered under all four subplots
    figA.text(0.5, 0.02, "Soda", ha='center', va='bottom', fontsize=fs_label + 3)
    figA.savefig(os.path.join(save_dir, "soda_bar.png"), dpi=150, bbox_inches="tight")
    print("Saved soda_bar.png")

    # ── Fig B: relative error %, one bar per param ────────────────────────────
    figB, ax2 = plt.subplots(figsize=(12, 7))
    figB.subplots_adjust(bottom=0.18)

    all_xs = np.arange(len(rel_errors), dtype=float)
    colors = [err_col if not np.isnan(v) else "#BDBDBD" for v in rel_errors]
    ax2.bar(all_xs, [0 if np.isnan(v) else v for v in rel_errors],
            width=0.7, color=colors, alpha=0.85)
    for i, v in enumerate(rel_errors):
        if np.isnan(v):
            ax2.text(all_xs[i], 0.5, "N/A", ha='center', va='bottom',
                     fontsize=fs_tick, color='gray')

    # Bold in-axes parameter labels, same style as Fig A
    ax2.set_ylim(bottom=0)
    ymax_r = ax2.get_ylim()[1]
    for i, lbl in enumerate(rel_param_labels):
        ax2.text(all_xs[i], ymax_r * 0.93, lbl, ha='center', va='top',
                 fontsize=fs_param, fontweight='bold', transform=ax2.transData)

    ax2.set_xticks([])   # no per-bar x-tick
    ax2.set_xlabel("Soda", fontsize=fs_label + 3, labelpad=12)
    ax2.set_ylabel("Relative error (%)", fontsize=fs_label + 3)
    ax2.tick_params(axis='y', labelsize=fs_tick + 2)
    ax2.grid(axis='y', alpha=0.4)
    ax2.set_xlim(left=-0.5, right=len(rel_errors) - 0.5)

    figB.savefig(os.path.join(save_dir, "soda_error.png"), dpi=150, bbox_inches="tight")
    print("Saved soda_error.png")


def print_discussion_stats(results: list) -> None:
    """
    Print four discussion-section analyses to stdout:
      1. Per-phase consistency (std dev across 4 estimates, normalized spread, best estimator)
      2. Hysteresis cancellation benefit (arc_B vs unarc_B vs average)
      3. Soda bottle diagnosis (arc/unarc divergence vs rigid objects)
      4. Friction identifiability (mu_est vs mu_gt_from_gt_mass, separated error sources)
    """
    RIGID = {"box", "heart", "flashlight", "monitor"}

    # ── helpers ────────────────────────────────────────────────────────────────
    def _pct(err, gt):
        return abs(err) / abs(gt) * 100 if gt != 0 else float("nan")

    # ══════════════════════════════════════════════════════════════════════════
    # 1. PER-PHASE CONSISTENCY TABLE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  1. PER-PHASE CONSISTENCY  (std dev across 4 phase estimates)")
    print("=" * 80)

    hdr = (f"  {'Object':<12} "
           f"{'σ_m (kg)':>10} {'σ_m %GT':>9} {'best_m':>10}  "
           f"{'σ_zc (m)':>10} {'σ_zc %GT':>9} {'best_zc':>10}  "
           f"{'σ_θ (°)':>9} {'σ_θ %GT':>8} {'best_θ':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in results:
        pe  = r["phase_estimates"]
        obj = r["obj"]
        keys = ["arc_A", "arc_B", "unarc_A", "unarc_B"]

        ms  = np.array([pe[k]["m"]     for k in keys])
        zcs = np.array([pe[k]["zc"]    for k in keys])
        ths = np.array([pe[k]["theta"] for k in keys])

        sm  = float(np.std(ms,  ddof=1))
        szc = float(np.std(zcs, ddof=1))
        sth = float(np.std(ths, ddof=1))

        sm_pct  = _pct(sm,  r["mass_gt"])
        szc_pct = _pct(szc, r["com_z_gt"])
        sth_pct = _pct(sth, r["theta_gt_deg"])

        best_m   = keys[int(np.argmin(np.abs(ms  - r["mass_gt"])))]
        best_zc  = keys[int(np.argmin(np.abs(zcs - r["com_z_gt"])))]
        best_th  = keys[int(np.argmin(np.abs(ths - r["theta_gt_deg"])))]

        print(f"  {obj:<12} "
              f"{sm:>10.4f} {sm_pct:>8.1f}% {best_m:>10}  "
              f"{szc:>10.4f} {szc_pct:>8.1f}% {best_zc:>10}  "
              f"{sth:>9.3f} {sth_pct:>7.1f}% {best_th:>10}")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. HYSTERESIS CANCELLATION BENEFIT
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  2. HYSTERESIS CANCELLATION BENEFIT")
    print("     arc_B error  |  unarc_B error  |  average(arc_B, unarc_B) error")
    print("=" * 80)

    for param in ("m", "zc", "theta"):
        if param == "m":
            label, gt_key = "mass (kg)", "mass_gt"
        elif param == "zc":
            label, gt_key = "z_c  (m) ", "com_z_gt"
        else:
            label, gt_key = "theta*(°)", "theta_gt_deg"

        print(f"\n  --- {label} ---")
        print(f"  {'Object':<12} {'arc_B err%':>11} {'unarc_B err%':>13} {'avg err%':>10}  {'Δ(avg-best)%':>14}")
        print("  " + "-" * 64)

        for r in results:
            pe  = r["phase_estimates"]
            gt  = r[gt_key]
            ab  = pe["arc_B"][param]
            ub  = pe["unarc_B"][param]
            avg = 0.5 * (ab + ub)

            e_ab  = _pct(ab  - gt, gt)
            e_ub  = _pct(ub  - gt, gt)
            e_avg = _pct(avg - gt, gt)
            best_single = min(e_ab, e_ub)
            delta = e_avg - best_single  # positive = averaging hurt

            print(f"  {r['obj']:<12} {e_ab:>10.1f}% {e_ub:>12.1f}% {e_avg:>9.1f}%  {delta:>+13.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. SODA BOTTLE DIAGNOSIS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  3. SODA BOTTLE DIAGNOSIS  (arc vs unarc divergence, fluid slosh signature)")
    print("=" * 80)

    soda_results = [r for r in results if r["obj"] == "soda"]
    rigid_results = [r for r in results if r["obj"] in RIGID]

    def _arc_unarc_spread(r, param):
        pe = r["phase_estimates"]
        return abs(pe["arc_B"][param] - pe["unarc_B"][param])

    print(f"\n  {'Object':<12} {'arc_B_m':>9} {'unarc_B_m':>11} {'|Δm|':>8}  "
          f"{'arc_B_zc':>10} {'unarc_B_zc':>11} {'|Δzc|':>8}")
    print("  " + "-" * 74)

    spread_m_vals  = []
    spread_zc_vals = []

    for r in results:
        pe = r["phase_estimates"]
        ab_m  = pe["arc_B"]["m"]
        ub_m  = pe["unarc_B"]["m"]
        ab_zc = pe["arc_B"]["zc"]
        ub_zc = pe["unarc_B"]["zc"]
        dm  = abs(ab_m  - ub_m)
        dzc = abs(ab_zc - ub_zc)
        spread_m_vals.append((r["obj"], dm))
        spread_zc_vals.append((r["obj"], dzc))
        marker = " ← soda" if r["obj"] == "soda" else ""
        print(f"  {r['obj']:<12} {ab_m:>9.4f} {ub_m:>11.4f} {dm:>8.4f}  "
              f"{ab_zc:>10.4f} {ub_zc:>11.4f} {dzc:>8.4f}{marker}")

    if soda_results and rigid_results:
        soda_r = soda_results[0]
        soda_dm  = _arc_unarc_spread(soda_r, "m")
        soda_dzc = _arc_unarc_spread(soda_r, "zc")
        rigid_dm_mean  = float(np.mean([_arc_unarc_spread(r, "m")  for r in rigid_results]))
        rigid_dzc_mean = float(np.mean([_arc_unarc_spread(r, "zc") for r in rigid_results]))
        ratio_m  = soda_dm  / rigid_dm_mean  if rigid_dm_mean  > 0 else float("nan")
        ratio_zc = soda_dzc / rigid_dzc_mean if rigid_dzc_mean > 0 else float("nan")
        print(f"\n  Arc/Unarc spread ratios  (soda ÷ mean-rigid):")
        print(f"    mass  — soda |Δm|={soda_dm:.4f} kg   rigid mean={rigid_dm_mean:.4f} kg   ratio={ratio_m:.2f}×")
        print(f"    z_c   — soda |Δzc|={soda_dzc:.4f} m   rigid mean={rigid_dzc_mean:.4f} m   ratio={ratio_zc:.2f}×")
    else:
        print("\n  (Need both soda and at least one rigid object to compute ratios.)")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. FRICTION IDENTIFIABILITY CHECK
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  4. FRICTION IDENTIFIABILITY CHECK")
    print("     Separates mass-propagation error from sliding-measurement error")
    print("=" * 80)

    print(f"\n  {'Object':<12} {'mu_est':>8} {'mu_gt(GT_m)':>12} {'Δmu':>8}  "
          f"{'mu_est err%':>12} {'mu_gt err%':>12} {'Δ(err)%':>9}")
    print("  " + "-" * 78)

    for r in results:
        if r["mu_est"] is None:
            print(f"  {r['obj']:<12}  (no friction data)")
            continue
        mu_e  = r["mu_est"]
        mu_g  = r["mu_gt"]
        delta = mu_e - mu_g
        # mu_est uses estimated mass as denominator; mu_gt uses GT mass.
        # Both share the same f_slip numerator, so Δmu directly reflects mass error.
        err_est = _pct(mu_e - mu_g, mu_g)  # total friction error (from estimated mass)
        err_gt  = 0.0                        # mu_gt is the best-case friction (GT mass denom)
        # How much of the friction error is attributable purely to mass error:
        # delta_mu = f_slip*(1/m_est - 1/m_gt) ≈ f_slip * Δm / m_gt²
        mass_induced_err = _pct(delta, mu_g)

        print(f"  {r['obj']:<12} {mu_e:>8.4f} {mu_g:>12.4f} {delta:>+8.4f}  "
              f"{err_est:>11.1f}% {err_gt:>11.1f}%  {mass_induced_err:>+8.1f}%")

    print(f"\n  Note: mu_gt(GT_m) uses ground-truth mass as the normal-force denominator.")
    print(f"        Δmu = mu_est - mu_gt isolates the mass-propagation contribution to")
    print(f"        friction error; any residual in mu_gt reflects sliding-measurement noise.")
    print("=" * 80 + "\n")


def run_object(obj: str, workspace_root: str) -> dict | None:
    squash_file = os.path.join(workspace_root, "runtime_logs", obj, "arc_squash", "most_recent.npz")
    push_file   = os.path.join(workspace_root, "runtime_logs", obj, "push",       "most_recent.npz")
    if not os.path.exists(squash_file):
        print(f"\n[{obj}] No squash log — skipping.")
        return None
    print(f"\n{'='*60}\n  OBJECT: {obj}\n{'='*60}")
    try:
        return _run_estimation(obj, os.path.join(workspace_root, "runtime_logs", obj), squash_file, push_file)
    except (KeyError, ValueError) as exc:
        print(f"[{obj}] Incompatible or incomplete arc_squash log — skipping.")
        print(f"[{obj}] {exc}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, default=None, choices=ALL_OBJECTS)
    parser.add_argument("--workspace", type=str, default=None)
    args = parser.parse_args()

    workspace_root = args.workspace or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    results = []
    for obj in ([args.object] if args.object else ALL_OBJECTS):
        r = run_object(obj, workspace_root)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        plot_results_summary(results, workspace_root)
        plot_soda_summary(results, workspace_root)

    if results:
        print_discussion_stats(results)

    plt.show()


if __name__ == "__main__":
    main()
