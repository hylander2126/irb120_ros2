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

ALL_OBJECTS = ["box", "heart", "flashlight", "soda"]#, "monitor"]
# ALL_OBJECTS = ["flashlight"]
# ALL_OBJECTS = ["soda"]

STATE_SQUASH = 1
STATE_LULL = 2
STATE_ARC = 3
STATE_UNARC = 4
STATE_RETRACT = 5

FT_WRENCH_ORIGIN_OFFSET_X = 0.08225

# _LPF_B,      _LPF_A      = butter(4, 6,   fs=500, btype='low')  # 4, 6 Hz — removes high-freq sensor noise
# _LPF_SLOW_B, _LPF_SLOW_A = butter(2, 0.5, fs=500, btype='low')  # 2, 0.5 Hz — removes force-controller hunting
# def _lpf(x, axis=0):
#     return filtfilt(_LPF_B, _LPF_A, x, axis=axis) if x.shape[0] > 20 else x
# def _lpf_slow(x, axis=0):
#     return filtfilt(_LPF_SLOW_B, _LPF_SLOW_A, x, axis=axis) if x.shape[0] > 20 else x


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


def _run_estimation(obj: str, base_dir: str, squash_file: str) -> None:
    
    ## HERE WE CAN REDEFINE THE r0 lever arm
    if obj == "box":
        # r0 = np.array([0.01, 0.0, 0.3]) # 0.026 old value, new is ~1.4 cm
        COM_GT = np.array([0.05, 0.0, 0.15])
        MASS_GT = 0.676
        THETA_GT_DEG = 17.532
    elif obj == "heart":
        # r0 = np.array([0.01, 0.0, 0.2]) # 0.026
        COM_GT = np.array([0.0458, 0.0, 0.10])
        MASS_GT = 0.295
        THETA_GT_DEG = 23.984
    elif obj == "flashlight":
        # r0 = np.array([0.028, 0.0, 0.2]) # 0.028
        COM_GT = np.array([0.028, 0.0, 0.0938])
        MASS_GT = 0.387
        THETA_GT_DEG = 15.126
    elif obj == "soda":
        # r0 = np.array([0.055, 0.0, 0.3]) # 0.055
        COM_GT = np.array([0.055, 0.0, 0.15])
        MASS_GT = 2.057
        THETA_GT_DEG = 20.126

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

    ## ======== One figure per object: 2 subplots side-by-side =========
    fig_obj, axes_obj = plt.subplots(1, 3, figsize=(24, 6))
    fig_obj.suptitle(f"[{obj}]", fontsize=14, fontweight="bold")
    time_plot = time[contact_mask] - time[contact_mask][0]
    plot_raw_forces(time_plot, f_meas_S[contact_mask], title="Measured Force (Sensor Frame)", show=False) # On it's own
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

        # Method A
        fx_coeffs = np.polyfit(y_pitch_deg, w_app_O[phase_sel, 3], 1)
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

        return com_z_fx, mass_fx, theta_fx_deg, fx_coeffs, com_z_tau, mass_tau


    print(f"\n--- [{obj}] PHASE ESTIMATES ---")
    com_z_push,    mass_push,    theta_fx_push_deg,    fx_coeffs_push,    com_z_tau_push,    mass_tau_push    = _fit_phase(arc_phase_trimmed,   "ARC",   COM_GT)
    com_z_retract, mass_retract, theta_fx_retract_deg, fx_coeffs_retract, com_z_tau_retract, mass_tau_retract = _fit_phase(unarc_phase_trimmed, "UNARC", COM_GT)
    print(f"  [{obj}] Ground truth — COM_z={COM_GT[2]:.4f} m  Mass={MASS_GT:.4f} kg  theta*={theta_gt_deg:.1f}deg")

    # Torque-plot predictions from Method B (joint τ fit)
    theta_tau_push_deg    = -np.degrees(np.arctan2(COM_GT[0], com_z_tau_push))
    theta_tau_retract_deg = -np.degrees(np.arctan2(COM_GT[0], com_z_tau_retract))
    tau_pred_arc   = model_fwd_wrench(rot_vec_during_contact[tip_sel], np.array([COM_GT[0], 0.0, com_z_tau_push]),    mass_tau_push)[:, 1]
    tau_pred_unarc = model_fwd_wrench(rot_vec_during_contact[tip_sel], np.array([COM_GT[0], 0.0, com_z_tau_retract]), mass_tau_retract)[:, 1]

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

    # Plot f_x data and extrapolated line (Method A)
    axes_obj[1].plot(np.rad2deg(y_pitch_during_contact[tip_sel]), w_app_O[tip_sel, 3], 'o', markersize=3, label="f_x (object frame)")
    axes_obj[1].plot(theta_extrap, np.polyval(fx_coeffs_push, theta_extrap), color='tab:blue', linestyle='--', label="linear fit (ARC)")
    axes_obj[1].axvline(theta_fx_push_deg, color='tab:blue', linestyle=':', label=f"ARC A θ*={theta_fx_push_deg:.2f}°")
    axes_obj[1].axvline(theta_tau_push_deg, color='tab:blue', linestyle='-', linewidth=1.5, label=f"ARC B θ*={theta_tau_push_deg:.2f}°")
    axes_obj[1].plot(theta_extrap, np.polyval(fx_coeffs_retract, theta_extrap), color='tab:orange', linestyle='-.', label="linear fit (UNARC)")
    axes_obj[1].axvline(theta_fx_retract_deg, color='tab:orange', linestyle=':', label=f"UNARC A θ*={theta_fx_retract_deg:.2f}°")
    axes_obj[1].axvline(theta_tau_retract_deg, color='tab:orange', linestyle='-', linewidth=1.5, label=f"UNARC B θ*={theta_tau_retract_deg:.2f}°")
    axes_obj[1].axvline(theta_gt_deg, color='green', linestyle=':', label=f"theta* GT = {theta_gt_deg:.1f}deg")
    axes_obj[1].axhline(0, color='k', linewidth=0.8)
    axes_obj[1].set_xlabel("Pitch angle (degrees)")
    axes_obj[1].set_ylabel("f_x in object frame (N)")
    axes_obj[1].set_title("f_x zero-crossing (A: dashed) vs joint τ (B: solid)")
    axes_obj[1].legend()
    axes_obj[1].grid(True)
    
    # fig_obj.tight_layout()
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
    try:
        _run_estimation(obj, os.path.join(workspace_root, "runtime_logs", obj), squash_file)
    except (KeyError, ValueError) as exc:
        print(f"[{obj}] Incompatible or incomplete arc_squash log — skipping.")
        print(f"[{obj}] {exc}")


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
