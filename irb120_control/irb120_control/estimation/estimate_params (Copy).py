#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from irb120_control.estimation.com_estimation import model_fwd_wrench, model_bkwd_wrench, construct_T
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

    # EE pose is the sparsest stream (~100 Hz); use it as the common time grid.
    # F/T raw (~500 Hz) is subsampled onto this grid via interpolation here so
    # _run_estimation never has to manage multiple time axes.
    time  = data["pose_time_s"]
    time_ft_raw = data["ft_raw_time_s"]

    p_ee_B = np.column_stack([data["x"], data["y"], data["z"]])

    f_raw_ft = np.column_stack([data["fx_raw"], data["fy_raw"], data["fz_raw"]])
    t_raw_ft = np.column_stack([data["tx_raw"], data["ty_raw"], data["tz_raw"]])
    p_ft_ft  = np.column_stack([data["ft_px"], data["ft_py"], data["ft_pz"]])
    Q_ft_ft  = np.column_stack([data["ft_qx"], data["ft_qy"], data["ft_qz"], data["ft_qw"]])

    # Bias removal on the native F/T grid (needs Q_ft at full rate)
    R_ft_B = rotvec_to_rot(quat_to_rotvec(Q_ft_ft))          # (N_ft, 3,3)
    bias_f_B = R_ft_B[100] @ np.mean(f_raw_ft[50:250], axis=0)
    bias_t_B = R_ft_B[100] @ np.mean(t_raw_ft[50:250], axis=0)
    bias_f_S = np.einsum('nij,j->ni', R_ft_B.transpose(0, 2, 1), bias_f_B)
    bias_t_S = np.einsum('nij,j->ni', R_ft_B.transpose(0, 2, 1), bias_t_B)
    f_raw_ft -= bias_f_S
    t_raw_ft -= bias_t_S

    f_raw_ft = _lpf(f_raw_ft)
    t_raw_ft = _lpf(t_raw_ft)

    # Subsample everything onto the EE time grid
    def _interp_cols(t_src, arr, t_dst):
        return np.column_stack([np.interp(t_dst, t_src, arr[:, i]) for i in range(arr.shape[1])])

    f_meas_S = _interp_cols(time_ft_raw, f_raw_ft,  time)
    t_meas_S = _interp_cols(time_ft_raw, t_raw_ft,  time)
    p_ft_B   = _interp_cols(time_ft_raw, p_ft_ft,   time)
    Q_ft     = _interp_cols(time_ft_raw, Q_ft_ft,   time)

    return time, f_meas_S, t_meas_S, p_ft_B, Q_ft, p_ee_B


def _run_estimation(obj: str, base_dir: str, squash_file: str) -> None:
    time_ft, f_meas_S, t_meas_S, time_ft_raw, f_meas_raw, t_meas_raw, p_ft_B, Q_ft, time_ee, p_ee_B, time_obj, p_obj_B = load_and_preprocess(squash_file)

    # Temp change of variable name: Set raw data to "meas"...
    time_ft = time_ft_raw
    f_meas_S = f_meas_raw
    t_meas_S = t_meas_raw

    p_pivot_B = np.array([0.6, 0, 0]) # Near-exact pivot from pre-defined object frame (I reset obj to known pose)

    # Bootstrap: find EE reference pose using force spike (first physical touch)
    force_contact_mask = np.linalg.norm(f_meas_S, axis=1) > 0.5
    if not np.any(force_contact_mask):
        print(f"[{obj}] No force contact detected — skipping.")
        return

    contact_ft_idx = np.argmax(force_contact_mask)
    release_ft_idx = len(force_contact_mask) - np.argmax(force_contact_mask[::-1])

    contact_ee_idx = np.argmin(np.abs(time_ee - time_ft[contact_ft_idx]))
    r_t = p_ee_B - p_pivot_B
    r0  = p_ee_B[contact_ee_idx] - p_pivot_B
    print(f"[{obj}] p_pivot_B: {p_pivot_B}, p_ee_B[contact]: {np.round(p_ee_B[contact_ee_idx], 3)}, r0: {np.round(r0, 3)}")

    def rotvec_between(a, b):
        # rotation taking unit(a) -> unit(b), as a rotation vector (axis*angle)
        # a: (3,) reference vector; b: (N,3) query vectors
        a = np.atleast_2d(a) / np.linalg.norm(a)
        b = b / np.linalg.norm(b, axis=-1, keepdims=True)
        axis = np.cross(a, b)
        s = np.linalg.norm(axis, axis=-1)
        c = np.einsum('ni,ni->n', np.broadcast_to(a, b.shape), b)
        ang = np.arctan2(s, c)
        axis = axis / np.where(s[:, None] > 1e-9, s[:, None], 1.0)
        return axis * ang[:, None]                   # (N,3) rotation vector

    prop_angle = rotvec_between(r0, r_t)   # (N_ee, 3) full rotation vector from proprioception
    # prop_angle[:contact_ee_idx] = 0.0
    # Interpolate each component of the rotation vector onto the F/T time grid
    rv_obj = np.column_stack([
        np.interp(time_ft, time_ee, prop_angle[:, i]) for i in range(3)
    ])                                     # (N_ft, 3) object rotation vector in {B}

    # Force-trim window: keep only the interval between first contact and final release.
    # Outside this interval, and at the boundaries, zero the calculated object rotation vector.
    force_trim_mask = np.zeros(len(time_ft), dtype=bool)
    force_trim_mask[contact_ft_idx:release_ft_idx] = True
    rv_obj[~force_trim_mask] = 0.0
    rv_obj[contact_ft_idx] = 0.0
    rv_obj[release_ft_idx - 1] = 0.0

    # Signed tipping angle: Y component is negative when object tips toward robot.
    # Keep only the negative-angle region beyond the deadband.
    angle_B = rv_obj[:, 1]   # (N_ft,) negative during tipping
    contact_mask = force_trim_mask & (angle_B < -np.deg2rad(0.5))  # remove small-angle noise around zero

    # Build batched (N,4,4) homogeneous transforms for sensor and object frames
    T_B_sensor = construct_T(p_ft_B, quat=Q_ft) # (N,4,4) sensor pose in {B}
    p_obj_B = np.tile(p_pivot_B, (len(time_ft), 1))
    T_B_obj = construct_T(p_obj_B, rv=rv_obj)    # (N,4,4) object pose in {B}

    # ====================================================
    w_meas_S = np.hstack((t_meas_S, f_meas_S))  # (N,6) [tau, f] convention (Modern Robotics) maintained throughout
    # ====================================================

    T_B_sensor = T_B_sensor[contact_mask]
    T_B_obj    = T_B_obj[contact_mask]
    w_meas_S   = w_meas_S[contact_mask]

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

    w_app_O = model_bkwd_wrench(w_meas_S, T_B_sensor, T_B_obj)

    print(f"[{obj}] Percent of applied torque_y below zero (expect 100%): {(w_app_O[:, 1] < 0).mean() * 100:.1f}%")  # tau_y is index 1

    ## ======== One figure per object: 3 subplots side-by-side =========
    fig_obj, axes_obj = plt.subplots(1, 3, figsize=(24, 6))
    fig_obj.suptitle(f"[{obj}]", fontsize=14, fontweight="bold")

    # Raw F/T overview plot — full wrench: f=(N,3) from sensor, tau=(N,3) interpolated onto post-contact grid
    time_ft_rel     = time_ft[contact_ft_idx:] - time_ft[contact_ft_idx]
    time_contact_xp = time_ft[contact_mask] - time_ft[contact_ft_idx]
    # t_meas_S is on the full ft grid; slice directly. tau_full is contact-only so interpolate.
    tau_full_on_ft = np.zeros((len(time_ft_rel), 3))
    for i in range(3):
        tau_full_on_ft[:, i] = np.interp(time_ft_rel, time_contact_xp, w_app_O[:, i], left=0.0, right=0.0)
    plot_wrench_and_tipping(time_ft_rel, f_meas_S[contact_ft_idx:], tau_full_on_ft[:, 1],
                            ax=axes_obj[0],
                            pitch_rad=angle_B[contact_ft_idx:], torque_label="τ",
                            contact_time=0.0, title=f"Raw F/T (Sensor Frame)", show=False)

    # Applied wrench in object frame — full wrench: f=(N,3), tau=(N,3), both interpolated
    f_app_O_on_ft  = np.zeros((len(time_ft_rel), 3))
    for i in range(3):
        f_app_O_on_ft[:, i]  = np.interp(time_ft_rel, time_contact_xp, w_app_O[:, 3 + i], left=0.0, right=0.0)
    plot_wrench_and_tipping(time_ft_rel, f_app_O_on_ft, tau_full_on_ft[:, 1],
                            ax=axes_obj[1],
                            pitch_rad=angle_B[contact_ft_idx:], torque_label="τ",
                            contact_time=0.0, title=f"Applied Wrench (Object Frame)", show=False)

    rv_contact     = rv_obj[contact_mask]          # (N_c, 3) full rotation vectors during contact
    angle_contact  = angle_B[contact_mask]          # (N_c,)   angle magnitudes during contact (positive)

    # Tipping phase selection: exclude 1.5° from start and peak (use angle magnitude)
    angle_max = angle_contact.max()
    tip_sel = (angle_contact > np.deg2rad(1.6)) & (angle_contact < angle_max - np.deg2rad(1.6))
    print(f"[{obj}] angle_contact: N={len(angle_contact)}  min={np.rad2deg(angle_contact.min()):.2f}°  max={np.rad2deg(angle_max):.2f}°")
    print(f"[{obj}] tip_sel={tip_sel.sum()}")
    if abs(tip_sel.sum()) < 10:
        print(f"[{obj}] Too few tipping samples — skipping fit.")
        return
    
    # Split tip_sel into push / retract phases based on peak angle
    peak_idx_in_contact = np.argmax(angle_contact)
    push_phase      = np.arange(len(angle_contact)) <= peak_idx_in_contact
    push_tip_sel    = tip_sel & push_phase
    retract_tip_sel = tip_sel & ~push_phase

    def _fit_phase(phase_sel, label, COM_GT):
        if phase_sel.sum() < 10:
            print(f"[{obj}] Too few {label} samples ({phase_sel.sum()}) — skipping.")
            return None, None
        rv_ph  = rv_contact[phase_sel]   # (N,3) full rotation vectors for this phase
        com_z0 = mass0 = 0.1
        def _residual(params):
            w_grav, _ = model_fwd_wrench(rv_ph, np.array([COM_GT[0], 0.0, params[0]]), params[1], 0.0)
            # static equilibrium: gravity torque + applied torque = 0
            return (w_grav[:, :3] - w_app_O[phase_sel, :3]).ravel()#tau_full_smooth[phase_sel]).ravel()
        def _residual_tau_y(params):
            w_grav, _ = model_fwd_wrench(rv_ph, np.array([COM_GT[0], 0.0, params[0]]), params[1], 0.0)
            return w_grav[:, 1] - w_app_O[phase_sel, :3]
        
        
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

    # Build per-sample predictions over the full tip_sel window for plotting
    rv_fit        = rv_contact[tip_sel]           # (N,3) full rotation vectors
    angle_fit     = angle_contact[tip_sel]        # (N,) angle magnitudes for x-axis
    push_sel_plot = push_phase[tip_sel]

    tau_pred_push = np.zeros(tip_sel.sum())
    if com_z_push is not None:
        w_grav_push, _ = model_fwd_wrench(rv_fit, np.array([COM_GT[0], 0.0, com_z_push]), mass_push, 0.0)
        tau_pred_push = -w_grav_push[:, 1]

    tau_pred_retract = np.zeros(tip_sel.sum())
    if com_z_retract is not None:
        w_grav_ret, _ = model_fwd_wrench(rv_fit, np.array([COM_GT[0], 0.0, com_z_retract]), mass_retract, 0.0)
        tau_pred_retract = -w_grav_ret[:, 1]

    theta_push    = np.arctan2(COM_GT[0], com_z_push)    if com_z_push    is not None else None
    theta_retract = np.arctan2(COM_GT[0], com_z_retract) if com_z_retract is not None else None

    plot_torque_fit_result(
        pitch_rad=angle_fit,
        tau_meas=w_app_O[tip_sel, 1],  # tau_y for fit result plot
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
