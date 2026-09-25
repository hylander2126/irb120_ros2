#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt

# Run directly (`python3 .../estimation/estimate_params.py`) and only this file's own
# directory lands on sys.path, so the absolute irb120_control.* imports below fail. Put the
# package root (estimation/ -> irb120_control/ -> irb120_control/) on the path so the script
# works either way. Under `python3 -m` or the ros2 entry point __package__ is set, the root
# is already importable, and this is skipped.
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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

# World-frame pivot edge. Must match ARC_CENTER in arc_static.py — the controller arcs
# about this point, and the estimator measures the object's tilt about it.
# NOTE: this single value is used for every object. The tilt reference it defines is the
# most sensitive input in the whole pipeline (a 1 deg error moves m and z_c by ~11% and is
# INVISIBLE in the fit residual), so prefer pivot_mode="trajectory" to recover it per trial.
PIVOT_DEFAULT = np.array([0.61, 0.0, 0.0])

# Max peak-to-peak variation of |p_ball - p_pivot| across the arc for the rigid no-slip
# contact model (and hence trajectory-based pivot recovery) to be considered valid.
# Measured: box 1.8, heart 1.4, flashlight 2.1, monitor 8.4 mm — the monitor slips.
NOSLIP_TOL_MM = 4.0

# Distance tool0 -> finger_ball_center along tool0's local +X, from the URDF static chain
# (irb120_with_finger.xacro): tool0 -> root_finger = 0.08225 (== FT_WRENCH_ORIGIN_OFFSET_X)
# and root_finger -> finger_ball_center = 0.0866512. The chain also puts finger_ball_center
# at the SAME orientation as tool0, so reconstructing the sensor pose from the logged EE pose
# is a pure translation along the tool axis. See _reconstruct_sensor_pose.
EE_TO_SENSOR_OFFSET_X = 0.08225 + 0.0866512

# Ground-truth object properties. CoM/theta* for box/heart/flashlight are from CAD +
# scale; monitor's are approximate (CoM +-1 cm, theta* +-0.2 deg).
GROUND_TRUTH = {
    "box":        {"mass": 0.676, "com": np.array([0.05,   0.0, 0.15]),   "theta_deg": 17.532},
    "heart":      {"mass": 0.239, "com": np.array([0.0458, 0.0, 0.10]),   "theta_deg": 23.984},
    "flashlight": {"mass": 0.387, "com": np.array([0.028,  0.0, 0.0938]), "theta_deg": 15.126},
    "soda":       {"mass": 2.054, "com": np.array([0.0525, 0.0, 0.15]),   "theta_deg": 21.801},
    "monitor":    {"mass": 5.04,  "com": np.array([0.06,   0.0, 0.232]),  "theta_deg": 14.5},
}

_LPF_B, _LPF_A = butter(4, 6, fs=500, btype='low')

def _lpf(x, axis=0):
    return filtfilt(_LPF_B, _LPF_A, x, axis=axis) if x.shape[0] > 20 else x


def _reconstruct_sensor_pose(p_ee_B, Q_ee):
    """
    Rebuild the F/T sensor pose in {B} from the logged EE (finger_ball_center) pose.

    The loggers (arc_static.py, push.py) look up the frame "ft_link", which a URDF
    refactor removed — the chain is now tool0 -> root_sensor -> ... -> sensor_body ->
    root_finger -> finger_ball_center. Every sample of ft_p*/ft_q* in those logs is
    therefore NaN. Reconstruction is exact here rather than approximate because the
    URDF static chain puts finger_ball_center at the SAME orientation as tool0 (the
    frame "ft_link" named: tool0 -> root_finger is 0.08225 m along tool0's local +X,
    which is exactly the FT_WRENCH_ORIGIN_OFFSET_X this module already applies), so:

        R_ft_B = R_ee_B
        p_ft_B = p_ee_B - R_ee_B @ [EE_TO_SENSOR_OFFSET_X, 0, 0]

    Returns (p_ft_B, Q_ft) on the EE time grid.
    """
    R_ee_B = rotvec_to_rot(quat_to_rotvec(Q_ee))
    p_ft_B = p_ee_B - np.einsum("nij,j->ni", R_ee_B, np.array([EE_TO_SENSOR_OFFSET_X, 0.0, 0.0]))
    return p_ft_B, Q_ee.copy()


def load_and_preprocess(filepath):
    data = np.load(filepath)

    def _require(keys):
        missing = [k for k in keys if k not in data]
        if missing:
            raise KeyError(f"Missing keys {missing} in {os.path.basename(filepath)}.")
        empty = [k for k in keys if len(data[k]) == 0]
        if empty:
            # An aborted run writes the .npz with every stream at length 0.
            raise ValueError(
                f"Empty log (aborted run): {len(empty)}/{len(keys)} required streams "
                f"have no samples, incl. {empty[0]}."
            )

    required_keys = (
        "pose_time_s", "x", "y", "z", "qx", "qy", "qz", "qw",
        "controller_state_id",
        "ft_time_s", "fx", "fy", "fz", "tx", "ty", "tz",
    )
    _require(required_keys)

    # obj_* is the vision-tracked object pose. It is logged but UNUSED by the estimator
    # (object rotation comes from the EE lever arm via rotvec_between), and the
    # arc_static_batch runs recorded no detections at all — so it is optional.
    has_obj = all(
        k in data and len(data[k]) > 0
        for k in ("obj_time_s", "obj_qx", "obj_qy", "obj_qz", "obj_qw")
    )

    # EE pose is the sparsest stream (~100 Hz); use it as the common time grid.
    # F/T is subsampled onto this grid via interpolation here so _run_estimation never has to manage multiple time axes.
    time  = data["pose_time_s"]
    state_id = data["controller_state_id"].astype(int)
    if len(state_id) != len(time):
        raise ValueError(
            f"controller_state_id length ({len(state_id)}) does not match pose_time_s length ({len(time)}) in {filepath}."
        )

    p_ee_B = np.column_stack([data["x"], data["y"], data["z"]])
    Q_ee   = np.column_stack([data["qx"], data["qy"], data["qz"], data["qw"]])

    # Subsample everything onto the EE time grid
    def _interp_cols(t_src, arr, t_dst):
        return np.column_stack([np.interp(t_dst, t_src, arr[:, i]) for i in range(arr.shape[1])])

    time_ft = data["ft_time_s"]
    ft_aligned_keys = ("fx", "fy", "fz", "tx", "ty", "tz")
    bad_lengths = {k: len(data[k]) for k in ft_aligned_keys if len(data[k]) != len(time_ft)}
    if bad_lengths:
        raise ValueError(f"F/T-aligned key lengths do not match ft_time_s ({len(time_ft)}) in {filepath}: {bad_lengths}")

    f_ft = np.column_stack([data["fx"], data["fy"], data["fz"]])
    t_ft = np.column_stack([data["tx"], data["ty"], data["tz"]])

    f_meas_S = _interp_cols(time_ft, f_ft, time)
    t_meas_S = _interp_cols(time_ft, t_ft, time)

    # Sensor pose: prefer the logged ft_link TF, fall back to reconstructing it from the
    # EE pose when the TF lookup failed at record time (all-NaN — see _reconstruct_sensor_pose).
    sensor_pose_keys = ("ft_px", "ft_py", "ft_pz", "ft_qx", "ft_qy", "ft_qz", "ft_qw")
    logged_sensor_pose = (
        all(k in data and len(data[k]) == len(time_ft) for k in sensor_pose_keys)
        and np.isfinite(np.column_stack([data[k] for k in sensor_pose_keys])).all(axis=1).any()
    )
    if logged_sensor_pose:
        p_ft_ft = np.column_stack([data["ft_px"], data["ft_py"], data["ft_pz"]])
        Q_ft_ft = np.column_stack([data["ft_qx"], data["ft_qy"], data["ft_qz"], data["ft_qw"]])
        valid = np.isfinite(np.hstack([p_ft_ft, Q_ft_ft])).all(axis=1)
        p_ft_B = _interp_cols(time_ft[valid], p_ft_ft[valid], time)
        Q_ft = _interp_cols(time_ft[valid], Q_ft_ft[valid], time)
    else:
        p_ft_B, Q_ft = _reconstruct_sensor_pose(p_ee_B, Q_ee)

    if not has_obj:
        Q_obj = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(time), 1))
    else:
        Q_obj = _interp_cols(
            data["obj_time_s"],
            np.column_stack([data["obj_qx"], data["obj_qy"], data["obj_qz"], data["obj_qw"]]),
            time,
        )

    # The logged TF pose is ft_link at base of sensor body, but NetFT torque channels behave as moments about the distal face/finger base.
    R_ft_B = rotvec_to_rot(quat_to_rotvec(Q_ft))
    p_ft_B = p_ft_B + np.einsum(
        "nij,j->ni",
        R_ft_B,
        np.array([FT_WRENCH_ORIGIN_OFFSET_X, 0.0, 0.0]), # Shift the wrench origin before applying adjoint transforms.
    )

    return time, f_meas_S, t_meas_S, p_ft_B, Q_ft, p_ee_B, Q_obj, state_id, logged_sensor_pose


def estimate_friction(push_log_path: str, mass_est: float, mass_gt: float) -> tuple:
    """
    Estimate mu_t from slip-onset tangential force in the push log.
    Returns (mu_est, mu_gt) where mu_gt uses the known ground-truth mass.
    The PRE-contact window (first 10% of the log) is used to estimate the
    F/T zero bias, which is subtracted PER AXIS before taking the tangential
    magnitude. The push logs are not tared (bias ~[6.6, 1.8] N) and the push reads
    as -fx, so subtracting the bias from |f| instead gives a negative slip force.
    """
    data = np.load(push_log_path)
    fx = _lpf(data["fx"])
    fy = _lpf(data["fy"])

    n = len(fx)
    pre = slice(0, int(0.10 * n))                    # PRE-contact zero offset
    f_tan = np.hypot(fx - fx[pre].mean(), fy - fy[pre].mean())
    f_slip = float(np.median(f_tan[int(0.50 * n):int(0.85 * n)]))

    mu_est = f_slip / (mass_est * 9.81)
    mu_gt  = f_slip / (mass_gt  * 9.81)
    return mu_est, mu_gt


def pivot_from_trajectory(p_ee_B, state_id, z_fixed: float = 0.0) -> tuple:
    """
    Recover the pivot from the EE trajectory alone (proprioception only).

    Under no-slip with a non-rotating finger ball, the contact point c is a fixed
    material point of the object, so

        p_ball = c + R_ball * n_hat(theta)   and   p_ball - p_pivot = R(theta) * u0,

    i.e. the BALL CENTRE is rigidly carried by the object — the ball radius and the
    migration of the contact patch cancel exactly. The ball centre therefore sweeps a
    circular arc centred on the pivot, and fitting that circle recovers the pivot with
    no external measurement, no vision, and nothing to occlude.

    The swept arc is only ~13-20°, which leaves the centre well conditioned along x but
    poorly along the radial (z) direction, so z is pinned to the table plane by default
    rather than fitted. Returns (pivot, radius, residual_m) where residual_m is the RMS
    circle-fit error — a direct no-slip quality metric (sub-mm means the model holds).
    """
    arc = np.isin(state_id, [STATE_ARC])
    if arc.sum() < 50:
        return None, np.nan, np.nan
    x, z = p_ee_B[arc, 0], p_ee_B[arc, 2]

    def _resid(p):
        d = np.hypot(x - p[0], z - z_fixed)
        return d - d.mean()

    sol = least_squares(_resid, x0=[float(np.median(x))])
    d = np.hypot(x - sol.x[0], z - z_fixed)
    return np.array([sol.x[0], 0.0, z_fixed]), float(d.mean()), float(np.std(d))


def _run_estimation(obj: str, base_dir: str, squash_file: str, push_file: str | None,
                    verbose: bool = True, trial_label: str | None = None,
                    free_com_x: bool = False, pivot_mode: str = "fixed",
                    tilt_offset_rad: float = 0.0, ft_bias_n: float = 0.0,
                    pivot_override: np.ndarray | None = None) -> dict | None:
    """Estimate (m, z_c, theta*, mu_t) from ONE arc_squash trial log.

    `verbose=False` silences the per-trial diagnostic prints, which is what the
    multi-trial driver wants — 10 trials x 5 objects of this is unreadable.

    free_com_x  — fit com_x instead of taking it from GROUND_TRUTH. The paper's
                  primary method assumes com_x known; this is the ablation that
                  shows the result does not depend on that assumption.
    pivot_mode  — "fixed" uses PIVOT_DEFAULT; "trajectory" recovers the pivot per
                  trial from the EE arc (see pivot_from_trajectory).
    pivot_override — explicit pivot for this trial, used by pivot_mode="relative"
                  (the caller needs all trials to compute the per-object mean first).
    """
    def _say(msg):
        if verbose:
            print(msg)

    gt = GROUND_TRUTH[obj]
    MASS_GT, COM_GT, THETA_GT_DEG = gt["mass"], gt["com"], gt["theta_deg"]

    time, f_meas_S, t_meas_S, p_ft_B, Q_ft, p_ee_B, Q_obj, state_id, logged_sensor_pose = load_and_preprocess(squash_file)

    # Deliberate F/T zero perturbation, used only by uncertainty_budget(). A stale tare is
    # a constant FORCE offset in the sensor frame; letting it propagate through the wrench
    # transform reproduces the resulting pivot-torque bias correctly, which simply adding a
    # torque offset would not (the lever arm changes as the object tips).
    if ft_bias_n:
        f_meas_S = f_meas_S + np.array([ft_bias_n, 0.0, 0.0])

    p_pivot_B = PIVOT_DEFAULT.copy() if pivot_override is None else np.asarray(pivot_override, float).copy()
    pivot_fit_resid = np.nan

    # No-slip check against the DEFAULT pivot, computed before any pivot substitution so
    # it is an independent gate on whether trajectory recovery is even applicable.
    _arc_only = np.isin(state_id, [STATE_ARC, STATE_UNARC])
    _r_def = np.linalg.norm((p_ee_B - PIVOT_DEFAULT)[_arc_only], axis=1)
    noslip_dev_mm = float(np.ptp(_r_def) * 1000) if _arc_only.any() else np.nan

    if pivot_mode == "trajectory" and pivot_override is None:
        # Recovering the pivot from the arc is only valid where the ball centre really
        # does sweep a circle. The formal standard error of the fit is misleadingly small
        # (~0.05-0.2 mm) even when it does not: a non-circular path still fits SOME circle
        # tightly. Gate on the physical assumption instead — if |p_ball - p_pivot| is not
        # near-constant, no-slip is violated and the recovered centre is meaningless.
        if np.isfinite(noslip_dev_mm) and noslip_dev_mm > NOSLIP_TOL_MM:
            _say(f"[{obj}] no-slip deviation {noslip_dev_mm:.2f} mm > {NOSLIP_TOL_MM:.1f} mm "
                 f"— trajectory pivot NOT trustworthy, falling back to PIVOT_DEFAULT.")
        else:
            piv, _rad, pivot_fit_resid = pivot_from_trajectory(p_ee_B, state_id)
            if piv is not None:
                p_pivot_B = piv
                _say(f"[{obj}] pivot from EE arc: x={piv[0]:.4f} m "
                     f"(circle residual {pivot_fit_resid*1000:.3f} mm, no-slip dev {noslip_dev_mm:.2f} mm)")

    # Bootstrap from controller state timing rather than inferring contact/release.
    in_contact = np.isin(state_id, [STATE_LULL, STATE_ARC, STATE_UNARC, STATE_RETRACT])
    _say(f"[{obj}] Using controller_state_id for contact/phase segmentation.")
    if not logged_sensor_pose:
        _say(f"[{obj}] ft_link TF absent from log — sensor pose reconstructed from EE pose.")
    r_t = p_ee_B - p_pivot_B
    r0  = r_t[np.argmax(in_contact)]

    _say(f"[{obj}] p_pivot_B: {p_pivot_B}, p_ee_B[contact]: {np.round(p_ee_B[np.argmax(in_contact)], 3)}, r0: {np.round(r0, 3)}")
    rot_vec_obj = rotvec_between(r0, r_t)  # (N, 3) object rotation vector in {B} on unified time grid
    rot_vec_obj[~in_contact] = 0.0         # Keep only the contact window; zero outside

    # Deliberate perturbation of the tilt reference, used only by uncertainty_budget().
    # Applied about the tipping (y) axis, which is where the theta*/z_c degeneracy lives.
    if tilt_offset_rad:
        rot_vec_obj[in_contact, 1] -= tilt_offset_rad

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
    _say(f"\n[{obj}] FORCING TIP AXIS TO: {np.round(TIP_AXIS, 2)}")
    _say(f"[{obj}] And testing p_pivot_B at: {p_pivot_B}\n")

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
        _say(f"  [{obj}] {label:>7s} [A: f_x→θ*] theta*={theta_fx_deg:.2f}°  z_c={com_z_fx:.4f}m  m={mass_fx:.4f}kg")

        # Method B — joint (mass, z_c) [, com_x] fit from torque balance.
        # With free_com_x the third parameter is solved for instead of being taken from
        # ground truth. Note only the AMPLITUDE m*|r_com| and the PHASE theta* are truly
        # identifiable from tau(theta); fixing com_x is what breaks that tie. Freeing it
        # trades a GT input for a weaker separation of m from z_c.
        if free_com_x:
            res_tau = least_squares(
                lambda p: ((model_fwd_wrench(rv_ph, np.array([p[2], 0.0, p[1]]), p[0])[:, :3] @ TIP_AXIS) - tau_meas).ravel(),
                x0=[MASS_GT, COM_GT[2], COM_GT[0]],
                bounds=([1e-6, 1e-3, -0.2], [np.inf, np.inf, 0.2]), method='trf')
            mass_tau, com_z_tau, com_x_tau = res_tau.x
        else:
            res_tau = least_squares(
                lambda p: ((model_fwd_wrench(rv_ph, np.array([COM_GT[0], 0.0, p[1]]), p[0])[:, :3] @ TIP_AXIS) - tau_meas).ravel(),
                x0=[MASS_GT, COM_GT[2]], bounds=([1e-6, 1e-3], [np.inf, np.inf]), method='trf')
            mass_tau, com_z_tau = res_tau.x
            com_x_tau = COM_GT[0]
        theta_tau_deg = -np.degrees(np.arctan2(com_x_tau, com_z_tau))
        _say(f"  [{obj}] {label:>7s} [B: τ jnt ] theta*={theta_tau_deg:.2f}°  z_c={com_z_tau:.4f}m  "
             f"m={mass_tau:.4f}kg  com_x={com_x_tau:.4f}m{' (fit)' if free_com_x else ' (GT)'}")

        estimates = {
            "A": {"m": mass_fx,  "zc": com_z_fx,  "theta": theta_fx_deg, "cx": COM_GT[0]},
            "B": {"m": mass_tau, "zc": com_z_tau, "theta": theta_tau_deg, "cx": com_x_tau},
        }
        return com_z_fx, mass_fx, theta_fx_deg, fx_coeffs, com_z_tau, mass_tau, theta_tau_deg, estimates


    _say(f"\n--- [{obj}] PHASE ESTIMATES ---")
    com_z_push,    mass_push,    theta_fx_push_deg,    fx_coeffs_push,    com_z_tau_push,    mass_tau_push,    theta_tau_push_deg,    est_arc   = _fit_phase(arc_phase_trimmed,   "ARC",   COM_GT)
    com_z_retract, mass_retract, theta_fx_retract_deg, fx_coeffs_retract, com_z_tau_retract, mass_tau_retract, theta_tau_retract_deg, est_unarc = _fit_phase(unarc_phase_trimmed, "UNARC", COM_GT)
    _say(f"  [{obj}] Ground truth — COM_z={COM_GT[2]:.4f} m  Mass={MASS_GT:.4f} kg  theta*={theta_gt_deg:.1f}deg")

    # Average push/retract Method B estimates (hysteresis cancellation)
    mass_est      = 0.5 * (mass_tau_push      + mass_tau_retract)
    com_z_est     = 0.5 * (com_z_tau_push     + com_z_tau_retract)
    theta_est_deg = 0.5 * (theta_tau_push_deg + theta_tau_retract_deg)
    com_x_est     = 0.5 * (est_arc["B"]["cx"] + est_unarc["B"]["cx"])

    # Tipping moment m*|r_com| (kg·m). tau(theta) = m*g*|r_com|*sin(theta* - theta), so the
    # curve's AMPLITUDE fixes m*|r_com| and its PHASE fixes theta*. Those two are what the
    # data actually identifies; splitting m from z_c needs the extra com_x assumption and
    # is the ill-conditioned step. m*|r_com| is therefore the robust reported quantity —
    # it is invariant to the tilt-reference error that dominates everything else.
    mgr_est = mass_est * np.hypot(com_x_est, com_z_est)
    mgr_gt  = MASS_GT * np.hypot(COM_GT[0], COM_GT[2])
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
        _say(f"  [{obj}] Friction: mu_est={mu_est:.3f}  mu_gt={mu_gt:.3f}")
    else:
        _say(f"  [{obj}] No push log — friction skipped.")

    return {
        "obj":          obj,
        "trial":        trial_label or os.path.basename(squash_file),
        "mass_est":     mass_est,
        "mass_gt":      MASS_GT,
        "com_z_est":    com_z_est,
        "com_z_gt":     COM_GT[2],
        "theta_est_deg": abs(theta_est_deg),   # store as positive tipping angle
        "theta_gt_deg":  THETA_GT_DEG,
        "mgr_est":      mgr_est,               # tipping moment m*|r_com| (kg·m)
        "mgr_gt":       mgr_gt,
        "com_x_est":    com_x_est,
        "com_x_gt":     COM_GT[0],
        "mu_est":       mu_est,
        "mu_gt":        mu_gt,
        "pivot_x":      float(p_pivot_B[0]),
        "pivot_fit_resid_mm": pivot_fit_resid * 1000 if np.isfinite(pivot_fit_resid) else np.nan,
        "noslip_dev_mm": noslip_dev_mm,
        "phase_estimates": {
            "arc_A":   {"m": est_arc["A"]["m"],   "zc": est_arc["A"]["zc"],   "theta": abs(est_arc["A"]["theta"])},
            "arc_B":   {"m": est_arc["B"]["m"],   "zc": est_arc["B"]["zc"],   "theta": abs(est_arc["B"]["theta"])},
            "unarc_A": {"m": est_unarc["A"]["m"], "zc": est_unarc["A"]["zc"], "theta": abs(est_unarc["A"]["theta"])},
            "unarc_B": {"m": est_unarc["B"]["m"], "zc": est_unarc["B"]["zc"], "theta": abs(est_unarc["B"]["theta"])},
        },
    }


# ==========================================================================================
#  FIGURE STYLE (single-column Springer)
# ==========================================================================================
# The figures are drawn large in inches and then scaled down to a ~3.3 in column, so the
# size text ACTUALLY prints at is (fontsize) x (column width / figure width). That is why
# the knobs live here rather than being sprinkled through the plot functions: raising
# FONT_SCALE (or narrowing a figure) is what makes a label readable on the page.
FONT_SCALE       = 1.45   # multiplies every fontsize in the results figures
FIG_HEIGHT_SCALE = 1.45   # taller bar/box panels; widths are left alone
WHISKER_LW_SCALE = 2.2    # box, whisker, cap and +-1 SD error-bar line weight


def _fs(pt: float) -> float:
    """Font size in points after the global scale."""
    return pt * FONT_SCALE


def plot_trial_spread(results: list, save_dir: str) -> None:
    """
    Repeatability figure — results_spread.png.

    One panel per parameter; per object, the N trial estimates as a box (IQR +
    median, whiskers to min/max) with the individual trials overlaid.

    The y axis is DEVIATION FROM THAT OBJECT'S OWN TRIAL MEAN, in percent. That
    normalisation is the point of the figure: the absolute spreads differ by two
    orders of magnitude across objects (monitor SD 0.008 kg vs heart 0.021 kg on a
    0.24 kg object), so on a shared absolute axis every object but the heaviest
    collapses onto the zero line. In percent they are comparable and a 0.2% spread
    is still legible. Each object is annotated with its CV and absolute SD, so the
    physical magnitude is never lost.
    """
    objs = [r for r in results if r["n_trials"] > 0]
    if not objs:
        return
    labels = ["Flash." if r["obj"] == "flashlight" else r["obj"].capitalize() for r in objs]

    # Only plot parameters that actually have data (mu_t is absent without push logs).
    # m|r_com| is deliberately excluded from the FIGURE: it is a useful diagnostic (see the
    # repeatability table and the CSV) but less intuitive for readers than raw m / z_c.
    panels = [(sp_key, lbl, unit) for _, sp_key, _, lbl, unit in PARAM_KEYS
              if sp_key not in FIGURE_EXCLUDED_PARAMS
              and any(r["spread"][sp_key]["n"] > 0 for r in objs)]
    if not panels:
        return

    col_pt   = "#2196F3"   # same "estimate" blue as the other figures
    col_box  = "#BBDEFB"
    ink      = "#212121"
    fs_tick  = _fs(26)
    fs_label = _fs(28)
    fs_param = _fs(40)
    # Bumped less than the rest: this note has to fit in one object's slot, so at the
    # full FONT_SCALE adjacent objects' notes overlap.
    fs_note  = _fs(15) * 0.72

    tex = {"m": r"$m$", "z_c": r"$z_c$", "theta*": r"$\theta^*$",
           "m|r_com|": r"$m\,|r_c|$", "mu_t": r"$\mu_t$"}

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels) + 2, 8 * FIG_HEIGHT_SCALE),
                             squeeze=False)
    axes = axes[0]
    rng = np.random.default_rng(0)   # fixed seed: jitter must not move between runs
    all_ns = set()

    for ax, (sp_key, lbl, unit) in zip(axes, panels):
        dev_sets, notes = [], []
        for r in objs:
            st = r["spread"][sp_key]
            if st["n"] == 0 or st["mean"] == 0:
                dev_sets.append(np.array([]))
                notes.append(None)
                continue
            dev_sets.append((st["values"] - st["mean"]) / abs(st["mean"]) * 100.0)
            notes.append((st["cv_pct"], st["std"], st["n"]))

        bp = ax.boxplot(
            [d if len(d) else [0.0] for d in dev_sets],
            positions=np.arange(len(objs)), widths=0.55, whis=(0, 100),
            patch_artist=True, showfliers=False,
            medianprops=dict(color=ink, linewidth=2.5 * WHISKER_LW_SCALE),
            boxprops=dict(facecolor=col_box, edgecolor=ink, linewidth=1.5 * WHISKER_LW_SCALE),
            whiskerprops=dict(color=ink, linewidth=1.5 * WHISKER_LW_SCALE),
            capprops=dict(color=ink, linewidth=1.5 * WHISKER_LW_SCALE),
        )
        for i, d in enumerate(dev_sets):
            if not len(d):
                continue
            ax.scatter(i + rng.uniform(-0.13, 0.13, size=len(d)), d,
                       s=55 * FONT_SCALE, color=col_pt, edgecolor="white",
                       linewidth=0.8 * WHISKER_LW_SCALE, zorder=3)

        ax.axhline(0, color=ink, linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xticks(np.arange(len(objs)))
        ax.set_xticklabels(labels, fontsize=fs_tick, rotation=35, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel("Deviation from trial mean (%)", fontsize=fs_label)
        ax.tick_params(axis="y", labelsize=fs_tick)
        ax.grid(axis="y", alpha=0.4)
        # Headroom above for the parameter label, below for the per-object notes.
        lo, hi = ax.get_ylim()
        span = hi - lo
        ax.set_ylim(lo - 0.20 * span, hi + 0.16 * span)
        ax.text(0.5, 0.98, tex.get(lbl, lbl), ha="center", va="top",
                fontsize=fs_param, fontweight="bold", transform=ax.transAxes)
        if unit:
            ax.text(0.5, 0.86, f"({unit})", ha="center", va="top",
                    fontsize=fs_note + 1, color=ink, transform=ax.transAxes)

        all_ns.update(nt[2] for nt in notes if nt is not None)

        # One direct label per object (CV + absolute SD) — not a number per point.
        y0, y1 = ax.get_ylim()
        for i, note in enumerate(notes):
            if note is None:
                continue
            cv, sd, _ = note
            # Unit lives in the panel header — repeating it per object overflows the slot.
            sd_txt = f"{sd:.1e}" if 0 < abs(sd) < 0.01 else f"{sd:.3g}"
            ax.text(i, y0 + 0.02 * (y1 - y0), f"CV {cv:.2f}%\nSD {sd_txt}",
                    ha="center", va="bottom", fontsize=fs_note, color=ink)

    # N is identical across every panel and object, so it is stated once for the whole
    # figure — inside each panel it lands on top of the parameter symbol.
    if all_ns:
        ns = sorted(all_ns)
        n_lbl = f"N = {ns[0]}" if len(ns) == 1 else f"N = {ns[0]}–{ns[-1]}"
        fig.text(0.002, 0.998, n_lbl + " trials", ha="left", va="top",
                 fontsize=fs_note + 2, color=ink)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "results_spread.png"), dpi=150, bbox_inches="tight")
    print("Saved results_spread.png")


def plot_results_summary(results: list, save_dir: str) -> None:
    """
    Two-figure results summary (2×2 layout for single-column papers):
      Fig 1 — grouped bar chart: estimated vs GT for m, z_c, θ*, µ_t
      Fig 2 — absolute error bar chart for the same four parameters

    Each estimate is the mean over that object's N trials, and both figures carry
    ±1 SD whiskers computed across those trials. With a single trial the SD is
    zero and no whisker is drawn, so the figures degrade cleanly.
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

    # Across-trial SD of each estimate, in the same units as the plotted value.
    # These become the error bars; zero-length when only one trial exists.
    def _sd(sp_key, scale=1.0):
        return np.array([
            (r["spread"][sp_key]["std"] * scale)
            if r["spread"][sp_key]["n"] > 1 and np.isfinite(r["spread"][sp_key]["std"]) else 0.0
            for r in results
        ])

    mass_sd  = _sd("mass")
    com_z_sd = _sd("com_z", 100.0)   # m → cm, matching com_z_est
    theta_sd = _sd("theta")
    mu_sd    = _sd("mu")
    n_trials = np.array([r["n_trials"] for r in results])
    n_txt    = f"N={n_trials[0]}" if len(set(n_trials.tolist())) == 1 else f"N={n_trials.min()}–{n_trials.max()}"

    bar_w   = 0.75
    col_est = "#2196F3"
    col_gt  = "#4CAF50"
    err_col = "#E53935"
    ink     = "#212121"   # error bars ride in ink, not a series color
    fs_tick  = _fs(28)
    fs_label = _fs(30)
    fs_title = _fs(30)
    fs_leg   = _fs(24)

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
    from matplotlib.lines import Line2D
    abs_params = [
        (mass_est,  mass_gt,  mass_sd,  "kg",  r"$m$",        None),
        (com_z_est, com_z_gt, com_z_sd, "cm",  r"$z_c$",      None),
        (theta_est, theta_gt, theta_sd, "deg", r"$\theta^*$", None),
        (mu_est,    mu_gt,    mu_sd,    "",    r"$\mu_t$",    mu_mask),
    ]
    fig1, axes = plt.subplots(1, 4, figsize=(28, 8 * FIG_HEIGHT_SCALE))
    fig1.subplots_adjust(wspace=0.25)

    for idx, (ax, (est_vals, gt_vals, sd_vals, unit, param_label, mask)) in enumerate(zip(axes, abs_params)):
        for i, (ev, gv) in enumerate(zip(est_vals, gt_vals)):
            if mask is not None and not mask[i]:
                ax.text(i, 0, "N/A", ha='center', va='bottom',
                        fontsize=fs_tick, color='gray')
                continue
            lo, hi = min(ev, gv), max(ev, gv)
            base_col = col_est if ev <= gv else col_gt
            ax.bar(i, lo,      width=bar_w, color=base_col, alpha=0.85)
            ax.bar(i, hi - lo, width=bar_w, bottom=lo, color=err_col, alpha=0.90)
            # ±1 SD across trials, anchored at the mean estimate (the bar edge is
            # max(est, gt), so the whisker is deliberately drawn at `ev`, not the top).
            if sd_vals[i] > 0:
                ax.errorbar(i, ev, yerr=sd_vals[i], fmt='none', ecolor=ink,
                            elinewidth=2.2 * WHISKER_LW_SCALE, capsize=9 * WHISKER_LW_SCALE,
                            capthick=2.2 * WHISKER_LW_SCALE, zorder=5)

        ax.set_xticks(x)
        # rotation_mode='anchor' with ha='right' pivots the label on its right edge,
        # which visually centers each label under its bar group rather than hanging left
        ax.set_xticklabels(labels, fontsize=fs_tick + 2, rotation=35,
                           ha='right', rotation_mode='anchor')
        ax.set_ylabel(unit, fontsize=fs_label + 3)
        ax.tick_params(axis='y', labelsize=fs_tick + 2)
        ax.grid(axis='y', alpha=0.4)
        # Headroom must clear the tallest bar AND its whisker, or the bold parameter
        # label collides with them (heart's theta* whisker did exactly that).
        valid = np.ones(len(est_vals), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if valid.any():
            tops = np.maximum(est_vals, gt_vals)[valid] + sd_vals[valid]
            ax.set_ylim(0, float(np.max(tops)) * 1.22)
        else:
            ax.set_ylim(bottom=0)
        ax.text(0.5, 0.99, param_label, ha='center', va='top',
                fontsize=_fs(44), fontweight='bold', transform=ax.transAxes)
        if idx == 0:
            ax.legend(handles=[
                Patch(color=col_est, alpha=0.85, label="Est ≤ GT"),
                Patch(color=col_gt,  alpha=0.85, label="GT < Est"),
                Patch(color=err_col, alpha=0.90, label="Error"),
                Line2D([0], [0], color=ink, linewidth=2.2 * WHISKER_LW_SCALE, marker='_',
                       markersize=12 * WHISKER_LW_SCALE, label=f"±1 SD ({n_txt})"),
            ], fontsize=fs_leg, loc='center left')

    fig1.tight_layout(rect=[0, 0, 1, 1])
    fig1.savefig(os.path.join(save_dir, "results_bar.png"), dpi=150, bbox_inches="tight")
    print("Saved results_bar.png")

    # ── Figure 2: relative errors % (single shared axis, 4 groups separated by vertical lines) ──
    # Error bars are the across-trial SD expressed in the same normalised units as the
    # bar (SD/|GT|·100), so bar and whisker are the same quantity as in Figure 1 —
    # NOT the SD of the per-trial error, which would not be comparable between figures.
    params = [
        (mass_rel,  mass_sd  / np.abs(mass_gt)  * 100, r"$|\Delta m|/m$",               None),
        (com_z_rel, com_z_sd / np.abs(com_z_gt) * 100, r"$|\Delta z_c|/z_c$",           None),
        (theta_rel, theta_sd / np.abs(theta_gt) * 100, r"$|\Delta \theta^*|/\theta^*$", None),
        (mu_rel,    np.where(mu_mask, mu_sd / np.where(mu_gt == 0, np.nan, np.abs(mu_gt)) * 100, np.nan),
         r"$|\Delta \mu_t|/\mu_t$", mu_mask),
    ]
    num_groups = len(params)
    bar_width = 0.7
    total_bars = num_groups * n
    all_xs = np.arange(total_bars, dtype=float)  # evenly spaced, no gap

    fig2, ax2 = plt.subplots(figsize=(4 * n + 4, 8 * FIG_HEIGHT_SCALE))

    for g, (errs, sds, _, mask) in enumerate(params):
        xs = all_xs[g * n:(g + 1) * n]
        colors = [err_col if (mask is None or mask[i]) else "#BDBDBD" for i in range(n)]
        heights = np.where(np.isnan(errs), 0, errs)
        ax2.bar(xs, heights, width=bar_width, color=colors, alpha=0.85)
        sd_plot = np.where(np.isfinite(sds), sds, 0.0)
        if np.any(sd_plot > 0):
            ax2.errorbar(xs, heights, yerr=sd_plot, fmt='none', ecolor=ink,
                         elinewidth=2.0 * WHISKER_LW_SCALE, capsize=7 * WHISKER_LW_SCALE,
                         capthick=2.0 * WHISKER_LW_SCALE, zorder=5)
        if mask is not None:
            for i, m in enumerate(mask):
                if not m:
                    ax2.text(xs[i], 0.5, "N/A", ha='center', va='bottom', fontsize=fs_tick, color='gray')

    ax2.set_xticks(all_xs)
    ax2.set_xticklabels(labels * num_groups, fontsize=fs_tick + 2, rotation=35, ha='right')
    ax2.set_ylabel("Relative error (%)", fontsize=fs_label + 3)
    ax2.tick_params(axis='y', labelsize=fs_tick + 2)
    ax2.grid(axis='y', alpha=0.4)
    # Headroom set from bar+whisker rather than left to autoscale: autoscale clips the
    # tallest whisker, and the parameter labels below need clear space above the data.
    _tops = [h + s for errs, sds, _, mask in params
             for h, s in zip(np.where(np.isnan(errs), 0, errs),
                             np.where(np.isfinite(sds), sds, 0.0))]
    ax2.set_ylim(0, (max(_tops) * 1.32) if _tops and max(_tops) > 0 else 1.0)
    ax2.set_xlim(left=-0.5, right=total_bars - 0.5)
    # The axis is an absolute error, so it starts at 0 and the lower whisker is clipped
    # there. A whisker reaching 0 means the across-trial SD covers the offset from GT —
    # i.e. that estimate is consistent with ground truth within one SD.
    # Above the axes, not inside them: in-axes it overlapped the first group's label.
    ax2.legend(handles=[
        Line2D([0], [0], color=ink, linewidth=2.0 * WHISKER_LW_SCALE, marker='_',
               markersize=12 * WHISKER_LW_SCALE, label=f"±1 SD across trials ({n_txt})"),
    ], fontsize=fs_leg, loc='lower left', bbox_to_anchor=(0.0, 1.005),
       borderaxespad=0, frameon=False)

    # Vertical separators between groups and bold parameter labels inside each region
    ymax = ax2.get_ylim()[1]
    for g, (_, _, param_label, _) in enumerate(params):
        # Label centered in the group, placed at 88% of ymax so it sits clearly inside
        group_center_x = g * n + (n - 1) / 2.0
        ax2.text(group_center_x, ymax * 0.97, param_label,
                 ha='center', va='top', fontsize=_fs(36), fontweight='bold',
                 transform=ax2.transData)
        if g > 0:
            ax2.axvline(g * n - 0.5, color='#333333', linewidth=1.2, linestyle='-')

    fig2.tight_layout()
    # bbox_inches='tight' also keeps the out-of-axes legend in frame.
    fig2.savefig(os.path.join(save_dir, "results_error.png"), dpi=150, bbox_inches="tight")
    print("Saved results_error.png")

    # ── Print summary table ─────────────────────────────────────────────────────
    print(f"\n  Relative error of the across-trial MEAN estimate (± = across-trial SD, normalised)")
    print(f"{'Object':<12} {'N':>3} {'|Δm|%':>16} {'|Δzc|%':>16} {'|Δθ*|%':>16} {'|Δμt|%':>16}")
    print("-" * 84)
    for i, r in enumerate(results):
        mu_str = (f"{mu_rel[i]:>9.1f}±{params[3][1][i]:<5.1f}"
                  if not np.isnan(mu_rel[i]) else f"{'N/A':>16}")
        print(f"  {r['obj']:<10} {r['n_trials']:>3} "
              f"{mass_rel[i]:>9.1f}±{params[0][1][i]:<5.1f} "
              f"{com_z_rel[i]:>9.1f}±{params[1][1][i]:<5.1f} "
              f"{theta_rel[i]:>9.1f}±{params[2][1][i]:<5.1f} {mu_str}")


def plot_soda_summary(results: list, save_dir: str) -> None:
    """
    Two separate figures for soda, saved individually for manual composition:
      soda_bar.png   — pseudo-whisker absolute values, 4 subplots (one per param)
      soda_error.png — relative error % bars, one bar per param
    No title. Parameter labels match the bold in-axes style of the summary plots.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    soda = next((r for r in results if r["obj"] == "soda"), None)
    if soda is None:
        return

    bar_w   = 0.75
    col_est = "#2196F3"
    col_gt  = "#4CAF50"
    err_col = "#E53935"
    ink     = "#212121"
    fs_tick  = _fs(28)
    fs_label = _fs(30)
    fs_leg   = _fs(24)
    fs_param = _fs(44)

    has_mu     = soda["mu_est"] is not None
    mu_est_val = soda["mu_est"] if has_mu else 0.0
    mu_gt_val  = soda["mu_gt"]  if has_mu else 0.0

    n_soda = soda["n_trials"]

    def _soda_sd(sp_key, scale=1.0):
        st = soda["spread"][sp_key]
        return st["std"] * scale if st["n"] > 1 and np.isfinite(st["std"]) else 0.0

    abs_params = [
        (soda["mass_est"],         soda["mass_gt"],         _soda_sd("mass"),         "kg",  r"$m$",        True),
        (soda["com_z_est"] * 100,  soda["com_z_gt"]  * 100, _soda_sd("com_z", 100.0), "cm",  r"$z_c$",      True),
        (soda["theta_est_deg"],    soda["theta_gt_deg"],    _soda_sd("theta"),        "deg", r"$\theta^*$", True),
        (mu_est_val,               mu_gt_val,               _soda_sd("mu"),           "",    r"$\mu_t$",    has_mu),
    ]

    def _rel(est, gt, valid):
        if not valid or gt == 0:
            return float("nan")
        return abs(est - gt) / abs(gt) * 100

    rel_errors = [_rel(ev, gv, valid) for ev, gv, _, _, _, valid in abs_params]
    rel_sds = [(sd / abs(gv) * 100) if (valid and gv != 0) else float("nan")
               for _, gv, sd, _, _, valid in abs_params]
    rel_param_labels = [r"$m$", r"$z_c$", r"$\theta^*$", r"$\mu_t$"]

    # ── Fig A: pseudo-whisker absolute, 4 independent subplots ───────────────
    figA, axesA = plt.subplots(1, 4, figsize=(18, 7 * FIG_HEIGHT_SCALE))
    figA.subplots_adjust(wspace=0.65, bottom=0.18)

    for idx, (ax, (ev, gv, sd, unit, param_label, valid)) in enumerate(zip(axesA, abs_params)):
        if not valid:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                    fontsize=fs_tick, color='gray', transform=ax.transAxes)
        else:
            lo, hi = min(ev, gv), max(ev, gv)
            base_col = col_est if ev <= gv else col_gt
            ax.bar(0, lo,      width=bar_w, color=base_col, alpha=0.85)
            ax.bar(0, hi - lo, width=bar_w, bottom=lo, color=err_col, alpha=0.90)
            if sd > 0:
                ax.errorbar(0, ev, yerr=sd, fmt='none', ecolor=ink,
                            elinewidth=2.2 * WHISKER_LW_SCALE, capsize=9 * WHISKER_LW_SCALE,
                            capthick=2.2 * WHISKER_LW_SCALE, zorder=5)

        ax.set_xticks([])   # no per-subplot x-tick
        ax.set_ylabel(unit, fontsize=fs_label + 3)
        ax.tick_params(axis='y', labelsize=fs_tick + 2)
        ax.grid(axis='y', alpha=0.4)
        ax.set_ylim(0, (max(ev, gv) + sd) * 1.22 if valid else None)
        ax.set_xlim(-0.75, 0.75)
        ax.text(0.5, 0.93, param_label, ha='center', va='top',
                fontsize=fs_param, fontweight='bold', transform=ax.transAxes)
    # Legend placed outside all axes, to the right of the last subplot
    axesA[-1].legend(handles=[
        Patch(color=col_est, alpha=0.85, label="Est ≤ GT"),
        Patch(color=col_gt,  alpha=0.85, label="GT < Est"),
        Patch(color=err_col, alpha=0.90, label="Error"),
        Line2D([0], [0], color=ink, linewidth=2.2 * WHISKER_LW_SCALE, marker='_',
               markersize=12 * WHISKER_LW_SCALE, label=f"±1 SD (N={n_soda})"),
    ], fontsize=fs_leg, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # Single "Soda" label centered under all four subplots
    figA.text(0.5, 0.02, "Soda", ha='center', va='bottom', fontsize=fs_label + 3)
    figA.savefig(os.path.join(save_dir, "soda_bar.png"), dpi=150, bbox_inches="tight")
    print("Saved soda_bar.png")

    # ── Fig B: relative error %, one bar per param ────────────────────────────
    figB, ax2 = plt.subplots(figsize=(12, 7 * FIG_HEIGHT_SCALE))
    figB.subplots_adjust(bottom=0.18)

    all_xs = np.arange(len(rel_errors), dtype=float)
    colors = [err_col if not np.isnan(v) else "#BDBDBD" for v in rel_errors]
    ax2.bar(all_xs, [0 if np.isnan(v) else v for v in rel_errors],
            width=0.7, color=colors, alpha=0.85)
    _sd_plot = [0.0 if not np.isfinite(v) else v for v in rel_sds]
    if any(v > 0 for v in _sd_plot):
        ax2.errorbar(all_xs, [0 if np.isnan(v) else v for v in rel_errors],
                     yerr=_sd_plot, fmt='none', ecolor=ink,
                     elinewidth=2.0 * WHISKER_LW_SCALE, capsize=7 * WHISKER_LW_SCALE,
                     capthick=2.0 * WHISKER_LW_SCALE, zorder=5)
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


# ==========================================================================================
#  PAPER RESULTS TABLE
# ==========================================================================================
# Columns of the results table, in order. Mirrors Table 2 of the paper: ground truth,
# estimate (across-trial mean +- 1 SD), relative error of that mean.
#   (spread key, estimate key, GT key, GT symbol, estimate symbol, rel-error symbol,
#    unit, scale applied to the stored value, decimals)
# theta* is deliberately absent — it is carried by the figures, and a fourth parameter
# pushes the table past a single column. Add the commented row to report it as well.
TABLE_COLUMNS = (
    ("mass",  "mass_est",  "mass_gt",  r"$m$",     r"$\hat{m}$",      r"$\frac{|\Delta m|}{m}$",           "kg", 1.0,   3),
    ("com_z", "com_z_est", "com_z_gt", r"$z_c$",   r"$\hat{z}_c$",    r"$\frac{|\Delta z_c|}{z_c}$",       "cm", 100.0, 2),
    # ("theta", "theta_est_deg", "theta_gt_deg", r"$\theta^*$", r"$\hat{\theta}^*$",
    #  r"$\frac{|\Delta \theta^*|}{\theta^*}$", "deg", 1.0, 2),
    ("mu",    "mu_est",    "mu_gt",    r"$\mu_t$", r"$\hat{\mu}_t$",  r"$\frac{|\Delta \mu_t|}{\mu_t}$",   "",   1.0,   3),
)

_TABLE_NA = "--"

# Columns estimated from one push log rather than repeated per trial (see _emit_results_table).
SINGLE_PUSH_COLUMNS = frozenset({"mu"})


def _obj_label(obj: str) -> str:
    return "Flashlight" if obj == "flashlight" else obj.capitalize()


def _table_cells(r: dict, col: tuple) -> tuple:
    """(gt, est, sd, rel_pct) for one object/column, already scaled. None = not available."""
    sp_key, est_key, gt_key, _, _, _, _, scale, _ = col
    gt  = r.get(gt_key)
    est = r.get(est_key)
    st  = r.get("spread", {}).get(sp_key, {})
    sd  = st.get("std") if st.get("n", 0) > 1 else None

    gt  = float(gt) * scale  if gt  is not None and np.isfinite(gt)  else None
    est = float(est) * scale if est is not None and np.isfinite(est) else None
    sd  = float(sd) * scale  if sd  is not None and np.isfinite(sd) and sd > 0 else None

    rel = None
    if gt not in (None, 0) and est is not None:
        rel = abs(est - gt) / abs(gt) * 100.0
    return gt, est, sd, rel


def print_results_table(results: list, save_dir: str) -> None:
    """
    Paper results table — ground truth | estimate (mean +- 1 SD) | relative error.

    Printed to stdout and written next to the figures as a ready-to-\\input LaTeX
    table. Soda is emitted as its own table for the same reason it gets its own
    figures: it is the deliberate non-rigid case and does not belong on a shared
    axis with the rigid objects.
    """
    rigid = [r for r in results if r["obj"] != "soda" and r.get("n_trials", 0) > 0]
    soda  = [r for r in results if r["obj"] == "soda" and r.get("n_trials", 0) > 0]
    if rigid:
        _emit_results_table(rigid, save_dir, "results_table.tex", "tab:results",
                            "Estimation results on physical objects")
    if soda:
        _emit_results_table(soda, save_dir, "results_table_soda.tex", "tab:results_soda",
                            "Estimation results on the soda bottle (non-rigid contents)")


def _emit_results_table(rows: list, save_dir: str, filename: str,
                        label: str, caption_lead: str) -> None:
    ns = sorted({r["n_trials"] for r in rows})
    n_txt = str(ns[0]) if len(ns) == 1 else f"{ns[0]}--{ns[-1]}"

    # Which symbols actually carry a +- over repeated trials. mu_t comes from a single
    # push divided by each trial's recovered mass, so its SD is only the propagated mass
    # spread — the caption must not present it as trial-to-trial repeatability.
    with_sd, without_sd, single_push_sd = [], [], False
    for col in TABLE_COLUMNS:
        cells = [_table_cells(r, col) for r in rows]
        if not any(c[1] is not None for c in cells):
            continue   # column is empty for these objects — nothing to say about it
        has_sd = any(c[2] is not None for c in cells)
        if col[0] in SINGLE_PUSH_COLUMNS:
            without_sd.append(col[4])
            single_push_sd |= has_sd
        else:
            (with_sd if has_sd else without_sd).append(col[4])

    # ── stdout ──────────────────────────────────────────────────────────────
    print(f"\n{caption_lead} — mean ± 1 SD across N={n_txt} trials")
    head_gt  = "".join(f"{c[3].strip('$').replace(chr(92), '') + (f' ({c[6]})' if c[6] else ''):>12}"
                       for c in TABLE_COLUMNS)
    head_est = "".join(f"{'est ' + c[3].strip('$').replace(chr(92), ''):>20}" for c in TABLE_COLUMNS)
    head_rel = "".join(f"{'|d' + c[3].strip('$').replace(chr(92), '') + '|%':>12}" for c in TABLE_COLUMNS)
    print(f"  {'Object':<12}{'N':>4}{head_gt}{head_est}{head_rel}")
    print("  " + "-" * (16 + 44 * len(TABLE_COLUMNS)))
    for r in rows:
        line = f"  {_obj_label(r['obj']):<12}{r['n_trials']:>4}"
        for col in TABLE_COLUMNS:
            dec = col[8]
            gt, est, sd, rel = _table_cells(r, col)
            line += f"{(f'{gt:.{dec}f}' if gt is not None else _TABLE_NA):>12}"
        for col in TABLE_COLUMNS:
            dec = col[8]
            gt, est, sd, rel = _table_cells(r, col)
            if est is None:
                cell = _TABLE_NA
            elif sd is None:
                cell = f"{est:.{dec}f}"
            else:
                cell = f"{est:.{dec}f} ± {sd:.{dec}f}"
            line += f"{cell:>20}"
        for col in TABLE_COLUMNS:
            gt, est, sd, rel = _table_cells(r, col)
            line += f"{(f'{rel:.1f}' if rel is not None else _TABLE_NA):>12}"
        print(line)

    # ── LaTeX ───────────────────────────────────────────────────────────────
    k = len(TABLE_COLUMNS)
    def _and_join(syms):
        return syms[0] if len(syms) == 1 else ", ".join(syms[:-1]) + " and " + syms[-1]

    caption = f"{caption_lead}."
    if with_sd:
        caption += (" " + _and_join(with_sd)
                    + (" is" if len(with_sd) == 1 else " are")
                    + f" the mean $\\pm$ one SD over $N={n_txt}$ press-and-pull trials")
        caption += ";" if without_sd else "."
    if without_sd:
        caption += (" " if with_sd else " ") + _and_join(without_sd)
        caption += (" follows" if len(without_sd) == 1 else " follow")
        caption += " from a single Mode~1 push and the recovered mass"
        caption += (", so its $\\pm$ reflects only the mass spread." if single_push_sd else ".")

    gt_hdr  = " & ".join(f"{c[3]}" + (f" ({c[6]})" if c[6] else "") for c in TABLE_COLUMNS)
    est_hdr = " & ".join(f"{c[4]}" + (f" ({c[6]})" if c[6] else "") for c in TABLE_COLUMNS)
    rel_hdr = " & ".join(c[5] for c in TABLE_COLUMNS)

    lines = [
        "% Generated by estimate_params.py — do not edit by hand.",
        "% Requires \\usepackage{booktabs} and \\usepackage{amsmath}.",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\small",
        "\\begin{tabular}{@{}l" + ("|" + "c" * k) * 3 + "@{}}",
        "\\toprule",
        (" & " + f"\\multicolumn{{{k}}}{{c|}}{{Ground Truth}}"
               + f" & \\multicolumn{{{k}}}{{c|}}{{Estimated ($N={n_txt}$)}}"
               + f" & \\multicolumn{{{k}}}{{c}}{{Relative Error (\\%)}} \\\\"),
        "".join(f"\\cmidrule(lr){{{2 + g * k}-{1 + (g + 1) * k}}}" for g in range(3)),
        f"Object & {gt_hdr} & {est_hdr} & {rel_hdr} \\\\",
        "\\midrule",
    ]
    for r in rows:
        cells = [_obj_label(r["obj"])]
        for col in TABLE_COLUMNS:
            dec = col[8]
            gt, est, sd, rel = _table_cells(r, col)
            cells.append(f"{gt:.{dec}f}" if gt is not None else _TABLE_NA)
        for col in TABLE_COLUMNS:
            dec = col[8]
            gt, est, sd, rel = _table_cells(r, col)
            if est is None:
                cells.append(_TABLE_NA)
            elif sd is None:
                cells.append(f"${est:.{dec}f}$")
            else:
                cells.append(f"${est:.{dec}f} \\pm {sd:.{dec}f}$")
        for col in TABLE_COLUMNS:
            gt, est, sd, rel = _table_cells(r, col)
            cells.append(f"{rel:.1f}" if rel is not None else _TABLE_NA)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    path = os.path.join(save_dir, filename)
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved {filename}")


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
    print("     NOTE: each phase estimate is itself a mean over that object's trials,")
    print("     so this measures ARC-vs-UNARC / method-A-vs-B disagreement, NOT")
    print("     trial-to-trial repeatability — see the ACROSS-TRIAL table above.")
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


def find_trial_logs(obj: str, workspace_root: str) -> list:
    """
    All arc_squash trial logs for `obj`, oldest first (filenames are timestamped).

    Excludes most_recent.npz, which is a byte-identical COPY of the newest trial —
    including it would double-weight that trial in every mean and shrink every std.
    Falls back to most_recent.npz only when no timestamped log exists, so
    single-trial datasets recorded before the batch runs still work.

    Trials recorded as episode stages (util/episode.py) follow the legacy
    folder's, in episode then stage order -- i.e. chronologically.
    """
    import glob
    squash_dir = os.path.join(workspace_root, "runtime_logs", obj, "arc_squash")
    trials = sorted(
        f for f in glob.glob(os.path.join(squash_dir, "arc_static*.npz"))
        if os.path.basename(f) != "most_recent.npz"
    )
    trials += sorted(
        f for f in glob.glob(os.path.join(workspace_root, "runtime_logs", obj, "episodes", "*",
                                          "*_press_pull_tip", "arc_static*.npz"))
        if os.path.basename(f) != "most_recent.npz"
    )
    if trials:
        return trials
    fallback = os.path.join(squash_dir, "most_recent.npz")
    return [fallback] if os.path.exists(fallback) else []


def find_push_log(obj: str, workspace_root: str) -> str:
    """The newest push log for `obj`: the legacy push/most_recent.npz or a push
    recorded as an episode stage, whichever was written last. Returns the legacy
    path (possibly nonexistent) when there is neither, as before."""
    import glob
    legacy = os.path.join(workspace_root, "runtime_logs", obj, "push", "most_recent.npz")
    candidates = [f for f in glob.glob(os.path.join(workspace_root, "runtime_logs", obj, "episodes", "*",
                                                    "*_push", "push_ft_pose_*.npz"))]
    if os.path.exists(legacy):
        candidates.append(legacy)
    return max(candidates, key=os.path.getmtime) if candidates else legacy


def _spread(values: list) -> dict:
    """Central tendency + spread for one parameter over N trials.

    std/sem use ddof=1 (sample std) — these are N=10 samples of a repeated
    experiment, not a full population. Both mean and median are reported because
    a single bad trial moves the mean but not the median; if they disagree, say so.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "sem": np.nan,
                "ci95": np.nan, "min": np.nan, "max": np.nan, "iqr": np.nan,
                "cv_pct": np.nan, "values": v}
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(np.median(v)),
        "sem": sem,
        # Student-t 95% half-width. t(0.975, n-1) for the small n we actually use;
        # 1.96 (normal) would understate the interval at n=10 by ~15%.
        "ci95": float(sem * _T95.get(n - 1, 1.96)) if n > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
        "cv_pct": float(std / abs(mean) * 100) if mean != 0 else np.nan,
        "values": v,
    }


# Two-sided t critical values at 95%, indexed by degrees of freedom (n-1).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}

PARAM_KEYS = (
    # (result key, spread key, ground-truth key, label, unit)
    ("mass_est",      "mass",   "mass_gt",      "m",        "kg"),
    ("com_z_est",     "com_z",  "com_z_gt",     "z_c",      "m"),
    ("theta_est_deg", "theta",  "theta_gt_deg", "theta*",   "deg"),
    # m*|r_com| is the combination tau(theta) actually identifies — reported alongside
    # m and z_c because it is immune to the tilt-reference degeneracy that dominates them.
    ("mgr_est",       "mgr",    "mgr_gt",       "m|r_com|", "kg m"),
    ("mu_est",        "mu",     "mu_gt",        "mu_t",     ""),
)

PHASE_KEYS = ("arc_A", "arc_B", "unarc_A", "unarc_B")

# Reported in the text tables and the per-trial CSV, but kept out of the figures.
FIGURE_EXCLUDED_PARAMS = frozenset({"mgr"})


def aggregate_trials(obj: str, trial_results: list) -> dict:
    """
    Collapse N per-trial estimates into one object-level record.

    The scalar keys the plotting/discussion functions already consume
    (mass_est, com_z_est, theta_est_deg, mu_est, *_gt, phase_estimates) are kept
    with the SAME names and now hold the across-trial MEAN, so every existing
    consumer keeps working. Spread lives alongside under "spread" / "n_trials".
    """
    agg = {
        "obj": obj,
        "n_trials": len(trial_results),
        "trials": [r["trial"] for r in trial_results],
        "mass_gt":      trial_results[0]["mass_gt"],
        "com_z_gt":     trial_results[0]["com_z_gt"],
        "theta_gt_deg": trial_results[0]["theta_gt_deg"],
        "mgr_gt":       trial_results[0]["mgr_gt"],
        "mu_gt":        trial_results[0]["mu_gt"],
        "diag": {
            "pivot_x":            _spread([r["pivot_x"] for r in trial_results]),
            "pivot_fit_resid_mm": _spread([r["pivot_fit_resid_mm"] for r in trial_results]),
            "noslip_dev_mm":      _spread([r["noslip_dev_mm"] for r in trial_results]),
            "com_x_est":          _spread([r["com_x_est"] for r in trial_results]),
        },
        "com_x_gt":     trial_results[0]["com_x_gt"],
        "spread": {},
    }

    for est_key, sp_key, gt_key, _, _ in PARAM_KEYS:
        st = _spread([r[est_key] for r in trial_results])
        agg["spread"][sp_key] = st
        # Point estimate reported for the object = mean over trials.
        agg[est_key] = st["mean"] if st["n"] else None

        # Per-trial relative error, so the error bar on the error plot is the
        # spread of the errors themselves — not the error of the mean.
        gt = agg[gt_key]
        if st["n"] and gt not in (None, 0) and np.isfinite(gt):
            rel = np.abs(st["values"] - gt) / abs(gt) * 100
            agg["spread"][sp_key + "_relerr"] = _spread(list(rel))
        else:
            agg["spread"][sp_key + "_relerr"] = _spread([])

    # Phase estimates: mean across trials, spread retained per phase/param.
    agg["phase_estimates"] = {}
    agg["phase_spread"] = {}
    for pk in PHASE_KEYS:
        agg["phase_estimates"][pk] = {}
        agg["phase_spread"][pk] = {}
        for param in ("m", "zc", "theta"):
            st = _spread([r["phase_estimates"][pk][param] for r in trial_results])
            agg["phase_estimates"][pk][param] = st["mean"]
            agg["phase_spread"][pk][param] = st
    return agg


def _relative_pivots(obj: str, trial_files: list, verbose: bool = True) -> dict:
    """
    Per-trial pivots that remove measured creep WITHOUT re-siting the object.

    The object physically creeps between back-to-back trials (the heart moves ~+0.9 mm
    per trial, away from the robot, confirmed both by circle-fitting the EE arc and by
    direct observation). That creep moves the tilt reference, which is the input the fit
    is most sensitive to and is invisible in its residual — so it inflates the spread.

    Only the DEVIATION from this object's own mean recovered pivot is applied; the mean
    stays at PIVOT_DEFAULT. So the absolute pivot claim is unchanged and the reported
    accuracy barely moves (<=0.5 pp on every object measured), while trial-to-trial
    scatter from creep is removed. This deliberately does NOT fit the pivot to make
    estimates approach ground truth — it is a relative correction only.

    Skipped entirely for objects that fail the no-slip check, where the EE arc is not
    circular and the recovered pivot is noise (the monitor: 8.4 mm deviation, and
    applying the correction there made its spread worse, not better).
    """
    cx, dev = [], []
    for f in trial_files:
        try:
            _t, _f, _tq, _p, _q, p_ee_B, _qo, state_id, _ls = load_and_preprocess(f)
        except (KeyError, ValueError, IndexError):
            cx.append(np.nan); dev.append(np.nan); continue
        arc = np.isin(state_id, [STATE_ARC, STATE_UNARC])
        r = np.linalg.norm((p_ee_B - PIVOT_DEFAULT)[arc], axis=1)
        dev.append(float(np.ptp(r) * 1000) if arc.any() else np.nan)
        piv, _rad, _res = pivot_from_trajectory(p_ee_B, state_id)
        cx.append(np.nan if piv is None else float(piv[0]))
    cx, dev = np.array(cx), np.array(dev)
    ok = np.isfinite(cx)
    mean_dev = float(np.nanmean(dev)) if np.isfinite(dev).any() else np.inf
    if not ok.any() or mean_dev > NOSLIP_TOL_MM:
        if verbose:
            print(f"  [{obj}] no-slip deviation {mean_dev:.2f} mm > {NOSLIP_TOL_MM:.1f} mm — "
                  f"relative pivot correction NOT applied (EE arc is not circular).")
        return {}
    mu = float(np.nanmean(cx[ok]))
    out = {i: PIVOT_DEFAULT + np.array([cx[i] - mu, 0.0, 0.0]) for i in np.where(ok)[0]}
    if verbose:
        drift = np.polyfit(np.arange(ok.sum()), cx[ok], 1)[0] * 1000
        print(f"  [{obj}] relative pivot correction: mean x={mu:.4f} m, per-trial spread "
              f"±{np.nanstd(cx[ok], ddof=1)*1000:.2f} mm, trend {drift:+.2f} mm/trial "
              f"(no-slip dev {mean_dev:.2f} mm)")
    return out


def run_object(obj: str, workspace_root: str, verbose_trials: bool = False,
               free_com_x: bool = False, pivot_mode: str = "relative") -> dict | None:
    """Run every recorded trial for `obj` and return the aggregated record."""
    trial_files = find_trial_logs(obj, workspace_root)
    push_file   = find_push_log(obj, workspace_root)
    if not trial_files:
        print(f"\n[{obj}] No squash log — skipping.")
        return None

    print(f"\n{'='*60}\n  OBJECT: {obj}   ({len(trial_files)} trial log(s) found)\n{'='*60}")
    base_dir = os.path.join(workspace_root, "runtime_logs", obj)
    rel_pivots = _relative_pivots(obj, trial_files) if pivot_mode == "relative" else {}
    trial_results, skipped = [], []
    for idx, path in enumerate(trial_files):
        name = os.path.basename(path)
        try:
            r = _run_estimation(obj, base_dir, path, push_file,
                                verbose=verbose_trials, trial_label=name,
                                free_com_x=free_com_x, pivot_mode=pivot_mode,
                                pivot_override=rel_pivots.get(idx))
        except (KeyError, ValueError, IndexError) as exc:
            skipped.append((name, str(exc).split("\n")[0][:90]))
            continue
        trial_results.append(r)
        print(f"  trial {len(trial_results):2d}/{len(trial_files)}  {name:44s} "
              f"m={r['mass_est']:.4f} kg  z_c={r['com_z_est']:.4f} m  θ*={r['theta_est_deg']:.2f}°")

    for name, why in skipped:
        print(f"  SKIPPED  {name:44s} {why}")
    if not trial_results:
        print(f"[{obj}] No usable trials — skipping.")
        return None

    print(f"  → {len(trial_results)} usable trial(s), {len(skipped)} skipped.")
    return aggregate_trials(obj, trial_results)


def write_per_trial_csv(results: list, save_dir: str) -> None:
    """Dump every individual trial estimate, so the stats are reproducible/re-plottable."""
    import csv
    path = os.path.join(save_dir, "results_per_trial.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["object", "trial_index", "trial_log",
                    "mass_est_kg", "mass_gt_kg",
                    "com_z_est_m", "com_z_gt_m",
                    "theta_est_deg", "theta_gt_deg",
                    "tipping_moment_est_kgm", "tipping_moment_gt_kgm",
                    "mu_est", "mu_gt",
                    "pivot_x_m", "pivot_fit_resid_mm", "noslip_dev_mm", "com_x_est_m"])
        for r in results:
            sp = r["spread"]
            for i in range(r["n_trials"]):
                def _v(key):
                    vals = sp[key]["values"]
                    return f"{vals[i]:.6f}" if i < len(vals) else ""
                def _d(key):
                    vals = r["diag"][key]["values"]
                    return f"{vals[i]:.6f}" if i < len(vals) else ""
                w.writerow([r["obj"], i + 1, r["trials"][i],
                            _v("mass"),  f"{r['mass_gt']:.6f}",
                            _v("com_z"), f"{r['com_z_gt']:.6f}",
                            _v("theta"), f"{r['theta_gt_deg']:.6f}",
                            _v("mgr"),   f"{r['mgr_gt']:.6f}",
                            _v("mu"),    "" if r["mu_gt"] is None else f"{r['mu_gt']:.6f}",
                            _d("pivot_x"), _d("pivot_fit_resid_mm"),
                            _d("noslip_dev_mm"), _d("com_x_est")])
    print(f"Saved results_per_trial.csv ({sum(r['n_trials'] for r in results)} trial rows)")


def print_repeatability_stats(results: list) -> None:
    """Across-trial repeatability: mean ± SD, median, range, CV, and 95% CI."""
    print("\n" + "=" * 96)
    print("  ACROSS-TRIAL REPEATABILITY  (N trials per object)")
    print("  mean ± SD (sample, ddof=1) | median | [min, max] | CV% = SD/|mean| | 95% CI = t·SD/√N")
    print("=" * 96)

    for est_key, sp_key, gt_key, label, unit in PARAM_KEYS:
        print(f"\n  --- {label} {('(' + unit + ')') if unit else ''} ---")
        print(f"  {'Object':<12} {'N':>3} {'mean':>11} {'SD':>10} {'median':>11} "
              f"{'min':>10} {'max':>10} {'CV%':>7} {'95% CI half':>12} {'GT':>10} {'err%':>8}")
        print("  " + "-" * 112)
        for r in results:
            st = r["spread"][sp_key]
            gt = r[gt_key]
            if st["n"] == 0:
                print(f"  {r['obj']:<12} {'—':>3}  (no data)")
                continue
            err = (abs(st["mean"] - gt) / abs(gt) * 100) if gt not in (None, 0) else np.nan
            gt_s = f"{gt:10.4f}" if gt is not None else f"{'—':>10}"
            err_s = f"{err:7.2f}%" if np.isfinite(err) else f"{'—':>8}"
            print(f"  {r['obj']:<12} {st['n']:>3} {st['mean']:>11.4f} {st['std']:>10.4f} "
                  f"{st['median']:>11.4f} {st['min']:>10.4f} {st['max']:>10.4f} "
                  f"{st['cv_pct']:>6.2f}% {st['ci95']:>12.4f} {gt_s} {err_s}")

    # Mean-vs-median divergence flags a trial that is pulling the mean around.
    print("\n" + "-" * 96)
    print("  Mean vs median check (|mean − median| as % of SD; > ~50% suggests a skewing trial)")
    print("-" * 96)
    print(f"  {'Object':<12} " + " ".join(f"{lbl:>14}" for _, _, _, lbl, _ in PARAM_KEYS))
    for r in results:
        cells = []
        for _, sp_key, _, _, _ in PARAM_KEYS:
            st = r["spread"][sp_key]
            if st["n"] < 2 or st["std"] == 0 or not np.isfinite(st["std"]):
                cells.append(f"{'—':>14}")
            else:
                cells.append(f"{abs(st['mean'] - st['median']) / st['std'] * 100:>13.0f}%")
        print(f"  {r['obj']:<12} " + " ".join(cells))

    # ── Drift check ───────────────────────────────────────────────────────────
    # Trials are recorded back to back without re-seating the object, so a slow
    # pose creep shows up as trial-order correlation rather than random scatter.
    # Where R^2 is high the SD is NOT measurement repeatability — it is dominated
    # by drift, and quoting it as a +- noise figure would overstate the noise.
    print("\n" + "-" * 96)
    print("  Trial-order dependence check")
    print("    R²  = share of across-trial variance explained by a linear trend in trial order.")
    print("    ρ₁  = lag-1 autocorrelation. For independent repeats both sit near 0.")
    print("  SD and the 95% CI above both assume INDEPENDENT trials. Where R² or ρ₁ is")
    print("  large the trials are serially correlated (object creeping between back-to-back")
    print("  runs), the SD reflects that drift rather than measurement noise, and the CI is")
    print("  optimistic — quote the trend alongside it.")
    print("-" * 96)
    print(f"  {'Object':<12} " + " ".join(f"{lbl + ': R²/ρ₁/slope':>29}" for _, _, _, lbl, _ in PARAM_KEYS))
    for r in results:
        cells = []
        for _, sp_key, _, _, unit in PARAM_KEYS:
            st = r["spread"][sp_key]
            v = st["values"]
            if st["n"] < 3 or np.ptp(v) == 0:
                cells.append(f"{'—':>29}")
                continue
            idx = np.arange(len(v), dtype=float)
            slope, intercept = np.polyfit(idx, v, 1)
            resid = v - (slope * idx + intercept)
            ss_tot = float(np.sum((v - v.mean()) ** 2))
            r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
            d = v - v.mean()
            rho1 = float(np.sum(d[:-1] * d[1:]) / np.sum(d ** 2)) if np.sum(d ** 2) > 0 else np.nan
            flag = "*" if (np.isfinite(r2) and r2 > 0.5) or (np.isfinite(rho1) and rho1 > 0.5) else " "
            cells.append(f"{r2:>6.2f}/{rho1:>+5.2f}/{slope:>+10.2e}{flag}")
        print(f"  {r['obj']:<12} " + " ".join(cells))
    print("  (* = trials are NOT independent for this parameter)")

    # ── Contact / geometry diagnostics ────────────────────────────────────────
    print("\n" + "-" * 96)
    print("  CONTACT & PIVOT DIAGNOSTICS")
    print("    no-slip dev = peak-to-peak variation of |p_ball - p_pivot| during the arc.")
    print("      Under no-slip the ball centre is rigidly carried by the object, so this")
    print("      should be ~0. Large values mean slip, compliance, or a wrong pivot.")
    print("    circle resid = RMS error of the circle fitted to the EE arc (pivot recovery).")
    print("    com_x       = fitted CoM x offset, vs ground truth (only differs with --free-com-x).")
    print("-" * 96)
    print(f"  {'Object':<12} {'pivot_x (m)':>13} {'no-slip dev':>13} {'circle resid':>14} "
          f"{'com_x est':>11} {'com_x GT':>10}")
    for r in results:
        d = r["diag"]
        pr = d["pivot_fit_resid_mm"]["mean"]
        pr_s = f"{pr:>12.3f}mm" if np.isfinite(pr) else f"{'—':>14}"
        print(f"  {r['obj']:<12} {d['pivot_x']['mean']:>13.4f} "
              f"{d['noslip_dev_mm']['mean']:>11.2f}mm {pr_s} "
              f"{d['com_x_est']['mean']:>11.4f} {r['com_x_gt']:>10.4f}")
    print("=" * 96)


OUTPUT_SUBDIR = "figs"   # figures + CSV land here, not in the workspace root


def _output_dir(workspace_root: str) -> str:
    d = os.path.join(workspace_root, OUTPUT_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, default=None, choices=ALL_OBJECTS)
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--verbose-trials", action="store_true",
                        help="Print the full per-trial fit diagnostics (very noisy with 10 trials/object).")
    parser.add_argument("--free-com-x", action="store_true",
                        help="Fit com_x instead of taking it from GROUND_TRUTH (ablation: "
                             "shows the result does not depend on knowing the CoM x offset).")
    parser.add_argument("--uncertainty-budget", action="store_true",
                        help="Attribute the residual error to individually measured input "
                             "uncertainties (pivot creep, pivot height, F/T zero drift).")
    parser.add_argument("--pivot", choices=["fixed", "relative", "trajectory"], default="relative",
                        help="DEFAULT 'relative' keeps the mean pivot at PIVOT_DEFAULT but removes "
                             "each trial's measured deviation from it, cancelling object creep — "
                             "accuracy is essentially unchanged, spread drops (the heart creeps "
                             "~0.9 mm/trial and is the reason this is the default; objects that fail "
                             "the no-slip check are left uncorrected). 'fixed' uses PIVOT_DEFAULT for "
                             "every object and is the uncorrected baseline. 'trajectory' uses the "
                             "per-trial recovered pivot outright, which also shifts the mean. Both "
                             "recover the pivot from the EE arc alone.")
    args = parser.parse_args()

    workspace_root = args.workspace or _find_workspace_root()
    out_dir = _output_dir(workspace_root)
    results = []
    for obj in ([args.object] if args.object else ALL_OBJECTS):
        r = run_object(obj, workspace_root, verbose_trials=args.verbose_trials,
                       free_com_x=args.free_com_x, pivot_mode=args.pivot)
        if r is not None:
            results.append(r)

    if results:
        write_per_trial_csv(results, out_dir)
        print_results_table(results, out_dir)
        print_repeatability_stats(results)
        plot_trial_spread(results, out_dir)

    if len(results) > 1:
        plot_results_summary(results, out_dir)
        plot_soda_summary(results, out_dir)

    if results:
        print_discussion_stats(results)

    if args.uncertainty_budget and results:
        print_uncertainty_budget(results, workspace_root)

    plt.show()


def _find_workspace_root() -> str:
    """Walk up from this file until a directory containing runtime_logs/ is found."""
    d = os.path.abspath(os.path.dirname(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "runtime_logs")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))



# ==========================================================================================
#  UNCERTAINTY BUDGET
# ==========================================================================================
# Why this exists: for the heart, z_c is estimated ~15% high against a ground truth that is
# a DESIGNED value (3D-printed, CoM placed in CAD) and therefore not in doubt. So the gap is
# measurement uncertainty, and the paper should account for it rather than report it as
# method error. This function attributes it to specific, individually measured inputs.
#
# The mechanism is that tau(theta) = m*g*|r_com|*sin(theta* - theta) identifies only an
# AMPLITUDE and a PHASE. Anything that shifts the phase — i.e. the tilt reference, which is
# defined by the assumed pivot — trades directly against z_c, and does so INVISIBLY: the fit
# residual is flat to 5 significant figures across +-1.5 deg of tilt offset. So these terms
# cannot be detected by goodness-of-fit and must be propagated explicitly.

UNCERTAINTY_SOURCES = (
    # (name, param, nominal 1-sigma, unit, how it was obtained)
    ("pivot x (object creep)", "pivot_x", 0.0018, "m",
     "half-range of the per-trial circle-fit pivot drift; heart creeps ~+0.9 mm/trial"),
    ("pivot z (edge radius / base compliance)", "pivot_z", 0.005, "m",
     "unmeasured: a designed-sharp edge still rolls on a fillet; invisible in the residual"),
    ("F/T zero drift between tares", "ft_bias", 0.010, "N",
     "SD of pre-contact |f| across a batch (0.005-0.009 N measured); one tare per batch"),
)


def uncertainty_budget(obj: str, workspace_root: str, trial_index: int = 0) -> dict | None:
    """
    Per-source contribution to the z_c and m uncertainty, by direct perturbation.

    Each input is perturbed by its 1-sigma and the estimator re-run, so the reported
    sensitivity is the true nonlinear response of the full pipeline rather than an
    analytic linearisation. Terms are combined in quadrature (independent sources).
    """
    trials = find_trial_logs(obj, workspace_root)
    if not trials:
        return None
    path = trials[min(trial_index, len(trials) - 1)]
    gt = GROUND_TRUTH[obj]
    base_dir = os.path.join(workspace_root, "runtime_logs", obj)

    def _est(pivot_dx=0.0, pivot_dz=0.0, tilt=0.0, ft_bias=0.0):
        global PIVOT_DEFAULT
        saved = PIVOT_DEFAULT.copy()
        PIVOT_DEFAULT = saved + np.array([pivot_dx, 0.0, pivot_dz])
        try:
            r = _run_estimation(obj, base_dir, path, None, verbose=False,
                                tilt_offset_rad=tilt, ft_bias_n=ft_bias)
        finally:
            PIVOT_DEFAULT = saved
        return (r["mass_est"], r["com_z_est"]) if r else (np.nan, np.nan)

    m0, z0 = _est()
    rows = []
    for name, param, sigma, unit, provenance in UNCERTAINTY_SOURCES:
        kw = {"pivot_x": "pivot_dx", "pivot_z": "pivot_dz",
              "tilt": "tilt", "ft_bias": "ft_bias"}[param]
        mp, zp = _est(**{kw: +sigma})
        mm, zm = _est(**{kw: -sigma})
        rows.append({
            "source": name, "sigma": sigma, "unit": unit, "provenance": provenance,
            # central difference — the response is not symmetric, so use the half-span
            "d_m":   0.5 * abs(mp - mm),
            "d_z":   0.5 * abs(zp - zm),
        })

    tot_m = float(np.sqrt(sum(r["d_m"] ** 2 for r in rows)))
    tot_z = float(np.sqrt(sum(r["d_z"] ** 2 for r in rows)))
    return {"obj": obj, "trial": os.path.basename(path), "m0": m0, "z0": z0,
            "rows": rows, "total_m": tot_m, "total_z": tot_z,
            "gt_m": gt["mass"], "gt_z": gt["com"][2]}


def print_uncertainty_budget(results: list, workspace_root: str) -> None:
    print("\n" + "=" * 100)
    print("  MEASUREMENT UNCERTAINTY BUDGET")
    print("  Each input is perturbed by its 1-sigma and the full estimator re-run.")
    print("  These terms are INVISIBLE to goodness-of-fit (the residual is flat in them),")
    print("  so they must be propagated explicitly rather than inferred from the fit.")
    print("=" * 100)
    for res in results:
        obj = res["obj"] if isinstance(res, dict) else res
        b = uncertainty_budget(obj, workspace_root)
        if b is None:
            continue
        # The paper reports the across-trial mean, so that is what the budget must explain.
        z_report = res["com_z_est"] if isinstance(res, dict) else b["z0"]
        n_rep = res["n_trials"] if isinstance(res, dict) else 1
        print(f"\n  --- {obj}  (trial {b['trial']}) ---")
        print(f"  {'source':<42} {'1σ':>10} {'→ σ(m) kg':>11} {'→ σ(z_c) mm':>13} {'% of z_c':>9}")
        print("  " + "-" * 92)
        for r in b["rows"]:
            print(f"  {r['source']:<42} {r['sigma']:>10.4f} {r['d_m']:>11.4f} "
                  f"{r['d_z']*1000:>13.2f} {r['d_z']/b['gt_z']*100:>8.2f}%")
        print("  " + "-" * 92)
        print(f"  {'combined (quadrature)':<42} {'':>10} {b['total_m']:>11.4f} "
              f"{b['total_z']*1000:>13.2f} {b['total_z']/b['gt_z']*100:>8.2f}%")
        obs = abs(z_report - b["gt_z"])
        print(f"  observed |z_c − GT| = {obs*1000:.2f} mm ({obs/b['gt_z']*100:.2f}%) "
              f"[mean of {n_rep} trial(s)]  →  "
              f"{'WITHIN' if obs <= b['total_z'] else 'EXCEEDS'} the combined budget")
        for r in b["rows"]:
            print(f"    · {r['source']}: {r['provenance']}")
    print("=" * 100)

if __name__ == "__main__":
    main()
