#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from irb120_control.estimation.com_estimation import model_fwd_wrench, model_bkwd_wrench, construct_T
from irb120_control.estimation.helper_fns import rotvec_to_rot, quat_to_rotvec
from irb120_control.estimation.plotting_helper import plot_wrench_and_tipping, plot_torque_fit_result, plot_raw_forces

ALL_OBJECTS = ["box"]#, "heart", "flashlight", "soda"] #, "monitor"]

STATE_SQUASH = 1
STATE_LULL = 2
STATE_ARC = 3
STATE_UNARC = 4
STATE_RETRACT = 5

_LPF_B,      _LPF_A      = butter(4, 6,   fs=500, btype='low')  # 4, 6 Hz — removes high-freq sensor noise
_LPF_SLOW_B, _LPF_SLOW_A = butter(2, 0.5, fs=500, btype='low')  # 2, 0.5 Hz — removes force-controller hunting


def _lpf(x, axis=0):
    return filtfilt(_LPF_B, _LPF_A, x, axis=axis) if x.shape[0] > 20 else x


def _lpf_slow(x, axis=0):
    return filtfilt(_LPF_SLOW_B, _LPF_SLOW_A, x, axis=axis) if x.shape[0] > 20 else x


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
    # F/T is subsampled onto this grid via interpolation here so
    # _run_estimation never has to manage multiple time axes.
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

    return time, f_meas_S, t_meas_S, p_ft_B, Q_ft, p_ee_B, Q_obj, state_id


def _run_estimation(obj: str, base_dir: str, squash_file: str) -> None:
    
    ## HERE WE CAN REDEFINE THE r0 lever arm
    if obj == "box":
        # r0 = np.array([0.01, 0.0, 0.3]) # 0.026 old value, new is ~1.4 cm
        COM_GT = np.array([0.05, 0.0, 0.15])
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

    time, f_meas_S, t_meas_S, p_ft_B, Q_ft, p_ee_B, Q_obj, state_id = load_and_preprocess(squash_file)

    # f_meas_S[:, 1] = 0 # zero the z-force to see if results are better
 
    p_pivot_B = np.array([0.605, 0, 0])# -0.021]) # 0.6 Near-exact pivot from pre-defined object frame (I reset obj to known pose)

    # Bootstrap from controller state timing rather than inferring contact/release.
    in_contact = np.isin(state_id, [STATE_LULL, STATE_ARC, STATE_UNARC, STATE_RETRACT])
    print(f"[{obj}] Using controller_state_id for contact/phase segmentation.")
    if not np.any(in_contact):
        print(f"[{obj}] No force contact detected — skipping.")
        return

    contact_idx = np.argmax(in_contact)
    r_t = p_ee_B - p_pivot_B
    r0  = r_t[contact_idx]
    print(f"[{obj}] p_pivot_B: {p_pivot_B}, p_ee_B[contact]: {np.round(p_ee_B[contact_idx], 3)}, r0: {np.round(r0, 3)}")

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
        return axis * ang[:, None]  # (N,3) rotation vector

    rv_obj = rotvec_between(r0, r_t)  # (N, 3) object rotation vector in {B} on unified time grid
    rv_obj[~in_contact] = 0.0         # Keep only the contact window; zero outside
    
    # Let's compare proprioceptive tipping angle vs the raw Q_obj on a plot:
    R_obj_B = rotvec_to_rot(rv_obj).transpose(0, 2, 1)  # (N,3,3) object rotation in {B}
    Q_obj_from_rv = np.empty_like(Q_obj)
    for i in range(len(rv_obj)):
        R_i = R_obj_B[i]
        qw = 0.5 * np.sqrt(1 + R_i[0, 0] + R_i[1, 1] + R_i[2, 2])
        qx = 0.25 * (R_i[2, 1] - R_i[1, 2]) / qw
        qy = 0.25 * (R_i[0, 2] - R_i[2, 0]) / qw
        qz = 0.25 * (R_i[1, 0] - R_i[0, 1]) / qw
        Q_obj_from_rv[i] = [qx, qy, qz, qw]

    # Contact mask: within the force window and past the small-angle deadband.
    # Y-component is negative when the object tips toward the robot.
    contact_mask = np.isin(state_id, [STATE_ARC, STATE_UNARC]) & (rv_obj[:, 1] < -np.deg2rad(1.0))

    # Build batched (N,4,4) homogeneous transforms for sensor and object frames
    T_B_sensor = construct_T(p_ft_B, quat=Q_ft)
    T_B_obj    = construct_T(np.tile(p_pivot_B, (len(time), 1)), rv=rv_obj)

    print(f"[{obj}] \n\n T_sensor pos: {T_B_sensor[50, :3, 3]} \n T_obj pos: {T_B_obj[50, :3, 3]} \n\n")

    w_meas_S = np.hstack((t_meas_S, f_meas_S))  # (N,6) [tau, f] convention (Modern Robotics)

    w_app_O = model_bkwd_wrench(w_meas_S[contact_mask],
                                T_B_sensor[contact_mask],
                                T_B_obj[contact_mask])

    print(f"[{obj}] Percent of applied torque_y below zero (expect 100%): {(w_app_O[:, 1] < 0).mean() * 100:.1f}%")  # tau_y is index 1

    ## ======== One figure per object: 3 subplots side-by-side =========
    fig_obj, axes_obj = plt.subplots(1, 2, figsize=(24, 6))
    fig_obj.suptitle(f"[{obj}]", fontsize=14, fontweight="bold")

    time_plot = time[contact_mask] - time[contact_mask][0]

    plot_raw_forces(time_plot, f_meas_S[contact_mask], title="Measured Force (Sensor Frame)", show=False)
    plot_wrench_and_tipping(time_plot, w_app_O[:, 3:], w_app_O[:, :3],
                            ax=axes_obj[0],
                            pitch_rad=rv_obj[contact_mask, 1], torque_label="τ",
                            contact_time=0.0, title=f"Applied Wrench (Object Frame)", show=False)

    ## Trim to contact window and then separate tipping from retract phase
    rv_contact    = rv_obj[contact_mask]           # (N_c, 3) full rotation vectors during contact
    pitch_contact = rv_contact[:, 1]               # (N_c,)   y-axis pitch for phase/threshold logic (plot only)
    state_contact = state_id[contact_mask]

    tau = w_app_O[:, :3]        # (N,3) applied moment in {O}
    u_tau = tau.mean(0) + 1e-6  # add small bias to avoid zero vector
    u_tau /= np.linalg.norm(u_tau)
    u_kin = rv_contact.mean(0)
    u_kin /= np.linalg.norm(u_kin)
    print(f"[{obj}] Average applied moment direction in object frame: {np.round(u_tau, 3)}")
    print(f"[{obj}] Average kinematic rotation direction: {np.round(u_kin, 3)}")

    # Tipping phase selection: exclude 1.6° from start and peak
    pitch_max = pitch_contact.min()                # most-negative = largest tip
    tip_sel = (pitch_contact < -np.deg2rad(1.6)) & (pitch_contact > pitch_max + np.deg2rad(1.6))
    print(f"[{obj}] pitch_contact: N={len(pitch_contact)}  min={np.rad2deg(pitch_contact.min()):.2f}°  max={np.rad2deg(pitch_contact.max()):.2f}°")
    peak_idx = np.argmin(pitch_contact)
    rv_peak_avg = rv_contact[peak_idx-3:peak_idx+4].mean(axis=0)
    print(f"[{obj}] peak tip rv 6idx avg @ idx={peak_idx} = {np.round(np.rad2deg(rv_peak_avg), 2)}")
    
    if abs(tip_sel.sum()) < 10:
        print(f"[{obj}] Too few tipping samples — skipping fit.")
        return

    # Split tip_sel into push / retract phases using controller state labels.
    push_phase = state_contact == STATE_ARC
    retract_phase = state_contact == STATE_UNARC
    push_tip_sel    = tip_sel & push_phase
    retract_tip_sel = tip_sel & retract_phase


    T_B_sensor_contact = T_B_sensor[contact_mask]
    p_ee_contact = p_ee_B[contact_mask]
    dump = np.column_stack([
        np.rad2deg(rv_contact[tip_sel,1]),
        w_app_O[tip_sel,3], w_app_O[tip_sel,4], w_app_O[tip_sel,5],   # f_O
        w_app_O[tip_sel,0], w_app_O[tip_sel,1], w_app_O[tip_sel,2],   # tau_O
        T_B_sensor_contact[tip_sel,0,3], T_B_sensor_contact[tip_sel,1,3], T_B_sensor_contact[tip_sel,2,3],   # ft pos in B
        p_ee_contact[tip_sel,0], p_ee_contact[tip_sel,1], p_ee_contact[tip_sel,2],  # ee in B
    ])
    np.savetxt("lever_dump.csv", dump[::max(1,len(dump)//150)],
            header="pitch_deg,fxO,fyO,fzO,txO,tyO,tzO,ftx,fty,ftz,eex,eey,eez",
            delimiter=",", comments="")
    

    TIP_AXIS = u_kin #np.array([-0.75, 0.75, 0.0]) # u_kin
    print(f"\n[{obj}] FORCING TIP AXIS TO: {TIP_AXIS}")
    print(f"[{obj}] And testing p_pivot_B at: {p_pivot_B}\n")

    def _fit_phase(phase_sel, label, COM_GT):
        if phase_sel.sum() < 10:
            print(f"[{obj}] Too few {label} samples ({phase_sel.sum()}) — skipping.")
            return None, None
        rv_ph = rv_contact[phase_sel]  # (N,3) full rotation vectors for this phase

        def _residual(params):
            w_grav = model_fwd_wrench(rv_ph, np.array([COM_GT[0], 0.0, params[0]]), params[1])
                
            tau_grav_axis = w_grav[:, :3] @ TIP_AXIS
            tau_meas_axis = w_app_O[phase_sel, :3] @ TIP_AXIS

            # return (w_grav[:, :3] - w_app_O[phase_sel, :3]).ravel()
            return (tau_grav_axis - tau_meas_axis).ravel()
        
        res = least_squares(_residual, x0=[0.1, 0.1],
                            bounds=([1e-6, 1e-6], [np.inf, np.inf]), method='trf')
        com_z, mass = res.x
        print(f"  [{obj}] {label:>7s} fit — COM_z={com_z:.4f} m  Mass={mass:.4f} kg  θ*={np.degrees(np.arctan2(COM_GT[0], com_z)):.1f}°")
        return com_z, mass

    print(f"\n--- [{obj}] PHASE ESTIMATES (full torque) ---")
    com_z_push,    mass_push    = _fit_phase(push_tip_sel,    "push", COM_GT)
    com_z_retract, mass_retract = _fit_phase(retract_tip_sel, "retract", COM_GT)
    print(f"  [{obj}] Ground truth — COM_z={COM_GT[2]:.4f} m  Mass={MASS_GT:.4f} kg  θ*={np.degrees(np.arctan2(COM_GT[0], COM_GT[2])):.1f}°")

    rv_fit        = rv_contact[tip_sel]        # (N,3) full rotation vectors for fit window
    push_sel_plot = push_phase[tip_sel]

    tau_pred_push = np.zeros(tip_sel.sum())
    if com_z_push is not None:
        w_grav_push = model_fwd_wrench(rv_fit, np.array([COM_GT[0], 0.0, com_z_push]), mass_push)
        tau_pred_push = w_grav_push[:, 1]

    tau_pred_retract = np.zeros(tip_sel.sum())
    if com_z_retract is not None:
        w_grav_ret = model_fwd_wrench(rv_fit, np.array([COM_GT[0], 0.0, com_z_retract]), mass_retract)
        tau_pred_retract = w_grav_ret[:, 1]

    theta_push    = np.arctan2(COM_GT[0], com_z_push)    if com_z_push    is not None else None
    theta_retract = np.arctan2(COM_GT[0], com_z_retract) if com_z_retract is not None else None

    # plot_torque_fit_result(
    #     pitch_rad=-pitch_contact[tip_sel],  # sign-flip for plotting: positive = larger tip
    #     tau_meas=w_app_O[tip_sel, 1],
    #     tau_pred_push=tau_pred_push,
    #     theta_star_push_rad=theta_push if theta_push is not None else 0.0,
    #     ax=axes_obj[2],
    #     tau_pred_retract=tau_pred_retract if com_z_retract is not None else None,
    #     theta_star_retract_rad=theta_retract,
    #     theta_star_gt_rad=np.arctan2(COM_GT[0], COM_GT[2]),
    #     push_sel=push_sel_plot,
    #     title=f"Torque fit result (full torque)",
    #     show=False,
    # )

    # --- f_x zero-crossing extrapolation ---
    # Fit a line to f_x vs pitch over the push phase, then find the angle where f_x = 0.
    # That angle is an independent estimate of θ* (tipping point) without relying on the torque model.
    pitch_push_deg = np.rad2deg(pitch_contact[push_tip_sel])
    fx_push        = w_app_O[push_tip_sel, 3]  # f_x in object frame

    fx_coeffs = np.polyfit(pitch_push_deg, fx_push, 1)  # linear fit: f_x = a*θ + b
    theta_fx_zero_deg = -fx_coeffs[1] / fx_coeffs[0]   # zero crossing: θ = -b/a
    print(f"  [{obj}] f_x zero-crossing — θ*={theta_fx_zero_deg:.2f}°  (GT={np.degrees(np.arctan2(COM_GT[0], COM_GT[2])):.1f}°)")

    # Extrapolation range: span observed data plus padding toward zero crossing
    theta_extrap = np.linspace(
        min(pitch_push_deg.min(), theta_fx_zero_deg) - 1.0,
        max(pitch_push_deg.max(), theta_fx_zero_deg) + 1.0,
        200
    )
    fx_extrap = np.polyval(fx_coeffs, theta_extrap)

    # Plot f_x data and extrapolated line
    axes_obj[1].plot(np.rad2deg(pitch_contact[tip_sel]), w_app_O[tip_sel, 3], 'o', markersize=3, label="f_x (object frame)")
    axes_obj[1].plot(theta_extrap, fx_extrap, '--', label=f"linear fit (push)")
    axes_obj[1].axvline(theta_fx_zero_deg, color='red', linestyle=':', label=f"θ* = {theta_fx_zero_deg:.2f}°")
    axes_obj[1].axvline(np.degrees(np.arctan2(COM_GT[0], COM_GT[2])), color='green', linestyle=':', label=f"θ* GT = {np.degrees(np.arctan2(COM_GT[0], COM_GT[2])):.1f}°")
    axes_obj[1].axhline(0, color='k', linewidth=0.8)
    axes_obj[1].set_xlabel("Pitch angle (degrees)")
    axes_obj[1].set_ylabel("f_x in object frame (N)")
    axes_obj[1].set_title("f_x zero-crossing → θ*")
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
