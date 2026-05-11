#!/usr/bin/env python3
import argparse
import os
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
# Custom imports
from irb120_control.estimation.com_estimation import model_fwd_wrench, model_bkwd_wrench
from irb120_control.estimation.helper_fns import rotvec_to_rot, quat_conj, quat_to_rotvec, enforce_quat_continuity
from irb120_control.estimation.plotting_helper import plot_wrench_and_tipping, plot_4vec_vs_angle, plot_torque_fit_result

def load_and_preprocess(filepath):
    print(f"Loading {filepath}")
    data = np.load(filepath)

    t_hist = data["ft_time_s"]
    # Raw sensor-frame forces and torques
    f_hist = np.column_stack([data["fx"], data["fy"], data["fz"]])
    t_hist_torque = np.column_stack([data["tx"], data["ty"], data["tz"]])

    # ft_link pose in base frame, sampled at F/T rate — used to build T_B_sensor
    ft_pose_keys = ("ft_px", "ft_py", "ft_pz", "ft_qx", "ft_qy", "ft_qz", "ft_qw")
    if all(k in data for k in ft_pose_keys) and len(data["ft_px"]) > 0:
        p_ft_B = np.column_stack([data["ft_px"], data["ft_py"], data["ft_pz"]])   # (N, 3)
        Q_ft   = np.column_stack([data["ft_qx"], data["ft_qy"], data["ft_qz"], data["ft_qw"]])  # (N, 4)
    else:
        raise KeyError(f"ft_link pose keys {ft_pose_keys} not found in {filepath}.")

    # EE pose: position in base frame, sampled at control rate (kept for contact-point interpolation)
    ee_keys = ("pose_time_s", "x", "y", "z")
    if all(k in data for k in ee_keys) and len(data["pose_time_s"]) > 0:
        t_ee   = data["pose_time_s"]
        p_ee_B = np.column_stack([data["x"], data["y"], data["z"]])  # (N, 3) in base frame
    else:
        raise KeyError(f"EE pose keys {ee_keys} not found in {filepath}.")

    # Object pose from detector: time + position + quaternion
    obj_keys = ("obj_time_s", "obj_x", "obj_y", "obj_z", "obj_qx", "obj_qy", "obj_qz", "obj_qw")
    if all(k in data for k in obj_keys) and len(data["obj_time_s"]) > 0:
        t_obj  = data["obj_time_s"]
        p_obj_B = np.column_stack([data["obj_x"], data["obj_y"], data["obj_z"]])  # (N, 3)
        Q_obj  = np.column_stack([data["obj_qx"], data["obj_qy"], data["obj_qz"], data["obj_qw"]])
    else:
        raise KeyError(f"Object pose keys {obj_keys} not found in {filepath}.")

    b, a = butter(4, 6, fs=500, btype='low')
    if len(t_hist) > 20:
        f_meas_S = filtfilt(b, a, f_hist, axis=0)
        t_meas_S = filtfilt(b, a, t_hist_torque, axis=0)
    else:
        f_meas_S = f_hist
        t_meas_S = t_hist_torque

    return t_hist, f_meas_S, t_meas_S, p_ft_B, Q_ft, t_ee, p_ee_B, t_obj, p_obj_B, Q_obj

def main():
    parser = argparse.ArgumentParser(description="Estimate mass and COM for a given log.")
    parser.add_argument("--workspace", type=str, default=None, help="Force workspace root path")
    args = parser.parse_args()

    # Try to find workspace root intelligently.
    # From build dir it's 3 up, from install it's 6 up. Let's just use an absolute path for safety, or walk up until we see runtime_logs
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    # if not os.path.exists(os.path.join(workspace_root, "runtime_logs")):
    #     workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        
    base_dir = os.path.join(workspace_root, "runtime_logs", "heart")
    squash_file = os.path.join(base_dir, "squash", "most_recent.npz")

    if not os.path.exists(squash_file):
        print(f"Missing squash log in {base_dir}")
        return

    # Object geometry — distance from tipping pivot edge to COM in object-frame x (= half object depth)
    COM_X = 0.05  # metres — tune to match actual object

    # 1. Load data
    t_hist, f_meas_S, t_meas_S, p_ft_B, Q_ft, t_ee, p_ee_B, t_obj, p_obj_B, Q_obj = load_and_preprocess(squash_file)

    # 2. Compute signed tipping angle from detector quaternions.
    #    Reference: mean quaternion over first second of data (stable rest pose).
    #    Then pin zero to the last obj sample before F/T contact, and enforce positive tipping.
    Q_obj = enforce_quat_continuity(Q_obj)

    rest_mask = t_obj - t_obj[0] < 1.0
    q_ref = Q_obj[rest_mask].mean(axis=0)
    q_ref /= np.linalg.norm(q_ref)

    q_ref_inv = quat_conj(q_ref)
    x1, y1, z1, w1 = q_ref_inv
    x2, y2, z2, w2 = Q_obj[:, 0], Q_obj[:, 1], Q_obj[:, 2], Q_obj[:, 3]
    q_rel = np.column_stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])
    rotvecs = quat_to_rotvec(q_rel)
    angle_y = np.rad2deg(rotvecs[:, 1])

    contact_ft_idx = np.argmax(np.linalg.norm(f_meas_S, axis=1) > 0.5)
    t_contact_start = t_hist[contact_ft_idx] if contact_ft_idx > 0 else t_hist[0]
    pre_contact_obj = np.where(t_obj <= t_contact_start)[0]
    contact_ref_idx = pre_contact_obj[-1] if len(pre_contact_obj) else 0
    angle_y -= angle_y[contact_ref_idx]

    post_contact = angle_y[contact_ref_idx:]
    if len(post_contact) > 10 and post_contact[post_contact.size // 2:].mean() < 0:
        angle_y = -angle_y

    obj_angle_deg = angle_y

    # 3. Interpolate object angle onto F/T time axis for the fitting pipeline.
    pitch_B = np.deg2rad(np.interp(t_hist, t_obj, obj_angle_deg))

    # 4. Compute contact point in object frame at first contact (no-slip → constant thereafter).
    #    Pivot is fixed in world frame — hardcoded. At contact pitch ≈ 0 (pinned above), so R_O_B ≈ I.
    p_pivot_B = np.array([0.54, 0.0, 0.0])  # tipping edge in base frame — tune to object
    t_c = t_hist[contact_ft_idx] if contact_ft_idx > 0 else t_hist[0]
    p_finger_at_contact = np.array([np.interp(t_c, t_ee, p_ee_B[:, i]) for i in range(3)])
    pitch_at_contact = float(np.interp(t_c, t_hist, pitch_B))
    R_O_B_at_contact = rotvec_to_rot(np.array([[0.0, pitch_at_contact, 0.0]]))[0].T
    r_contact_O = R_O_B_at_contact @ (p_finger_at_contact - p_pivot_B)
    print(f"p_finger_at_contact = {p_finger_at_contact}")
    print(f"r_contact_O         = {r_contact_O}")

    # 5. Compute applied wrench in object frame via model_bkwd_wrench.
    #    Sensor frame = EE frame (finger tip); object frame rotates about pivot.
    #    T_B_sensor: EE at constant finger position, rotated with the object.
    #    T_B_obj: object frame origin at pivot p_pivot_B, oriented from detector quaternions.
    contact_mask = pitch_B > 0.0
    w_meas_S_contact = np.column_stack([f_meas_S[contact_mask, :],
                                        t_meas_S[contact_mask, :]])  # (N,6) raw sensor frame
    pitch_contact = pitch_B[contact_mask]
    N_c = contact_mask.sum()

    rv_contact = np.zeros((N_c, 3))
    rv_contact[:, 1] = -pitch_contact  # object tips in -Y direction; pitch_contact is the signed magnitude

    # T_B_sensor: ft_link pose logged directly at F/T rate
    R_ft = rotvec_to_rot(quat_to_rotvec(Q_ft[contact_mask]))  # (N,3,3)
    T_B_sensor = np.zeros((N_c, 4, 4))
    T_B_sensor[:, :3, :3] = R_ft
    T_B_sensor[:, :3, 3]  = p_ft_B[contact_mask]
    T_B_sensor[:, 3, 3]   = 1.0

    # T_B_obj: object frame origin at pivot, oriented with the tipping angle from detector
    R_obj = rotvec_to_rot(rv_contact)  # (N,3,3) R_B_O from tipping angle
    T_B_obj = np.zeros((N_c, 4, 4))
    T_B_obj[:, :3, :3] = R_obj
    T_B_obj[:, :3, 3]  = p_pivot_B
    T_B_obj[:, 3, 3]   = 1.0

    w_app_O = model_bkwd_wrench(w_meas_S_contact, T_B_sensor, T_B_obj, r_contact_O)

    max_idx_contact = np.argmax(pitch_contact)
    tip_trim_frac = 0.85  # use this fraction of the tipping phase (trim noisy peak)
    tip_end_idx = int(max_idx_contact * tip_trim_frac)
    above_1deg = pitch_contact > np.deg2rad(1.5)
    tip_sel     = (np.arange(len(pitch_contact)) <= tip_end_idx) & above_1deg
    retract_sel = (np.arange(len(pitch_contact)) >  max_idx_contact) & above_1deg

    # 6. Raw data overview: rotate sensor-frame forces to world frame for plotting readability
    R_ft_all = rotvec_to_rot(quat_to_rotvec(Q_ft))  # (N,3,3)
    f_meas_W = np.einsum('nij,nj->ni', R_ft_all, f_meas_S)  # (N,3) world-frame forces

    t_ft_rel = t_hist - t_hist[0]
    tau_y_on_ft = np.interp(t_ft_rel, t_ft_rel[contact_mask], w_app_O[:, 4], left=0.0, right=0.0)
    plot_wrench_and_tipping(
        t_ft_rel,
        f_meas_W,
        tau_y_on_ft,
        pitch_rad=pitch_B,
        torque_label="tau_y",
        contact_time=t_ft_rel[contact_ft_idx],
        title="Raw F/T + object tipping angle over time",
        show=True,
    )

    # 8. Pre-fit diagnostic: w_app (fx, fy, fz, tau_y) vs tipping angle — tipping phase only
    # w_app_tip = np.column_stack([w_app_O[tip_sel, :3], w_app_O[tip_sel, 4]])
    # plot_4vec_vs_angle(
    #     w_app_tip,
    #     pitch_rad=pitch_contact[tip_sel],
    #     vec_labels=("f_x", "f_y", "f_z", "tau_y"),
    #     x_label="Tipping angle ||°||",
    #     y_label="Force (N)",
    #     torque_y_label="Torque (Nm)",
    #     title=r"$w_{app}$ vs Tipping Angle in {O} (object)",
    #     show=True,
    # )

    # 9. Fit on tipping phase only
    w_O_app_masked = w_app_O[tip_sel]
    pitch_B_masked = pitch_contact[tip_sel]
    rv_B_masked = rv_contact[tip_sel]

    def model_wrapper(params):
        com_z, mass = params
        com = np.array([COM_X, 0.0, com_z])
        w_grav_pred, w_ground_pred = model_fwd_wrench(rv_B_masked, com, mass, 0.0)#, w_O_app=w_O_app_masked)
        # Only fit tau_y — the torque balance is what constrains com_z and mass.
        # Force residuals are O(4N) vs torque O(0.1Nm) and would drown out the signal.
        tau_y_pred = w_grav_pred[:, 4] #+ w_ground_pred[:, 4]
        tau_y_meas = w_O_app_masked[:, 4]
        return tau_y_pred + tau_y_meas  # grav + applied = 0 at static equilibrium

    theta_star_rad = np.deg2rad(18.5)
    com_z0 = COM_X / np.tan(theta_star_rad)
    mid = len(pitch_B_masked) // 4
    th = pitch_B_masked[:mid]
    tau_mid = w_O_app_masked[:mid, 4]
    lever = COM_X * np.cos(th) - com_z0 * np.sin(th)
    mass0 = np.median(tau_mid / (9.81 * lever + 1e-12))
    mass0 = max(mass0, 0.01)
    print(f"Initial guess — com_z: {com_z0:.4f} m  mass: {mass0:.4f} kg")

    result_torque = least_squares(
        model_wrapper,
        x0=[com_z0, mass0],
        bounds=([1e-6, 1e-6], [np.inf, np.inf]),
        method='trf'
    )

    print("\n--- SQUASH PULL LOG MASS / COM ESTIMATE ---")
    print(f"  COM_z: {result_torque.x[0]:.4f} m")
    print(f"  Mass:  {result_torque.x[1]:.4f} kg")

    # 10. Fit result plot
    com_est_final = np.array([COM_X, 0.0, result_torque.x[0]])
    mass_est_final = result_torque.x[1]
    w_grav_pred, _ = model_fwd_wrench(rv_B_masked, com_est_final, mass_est_final, 0.0)#, w_O_app=w_O_app_masked)

    # 11. Friction estimate from push log
    push_log = os.path.join(base_dir, "push", "most_recent.npz")
    if os.path.exists(push_log):
        mu_est, mu_std = estimate_friction(push_log, mass_est_final)
        print(f"\n--- FRICTION ESTIMATE (push log) ---")
        print(f"  μ_table: {mu_est:.4f}  ±  {mu_std:.4f}")
        print(f"  (from mass estimate: {mass_est_final:.4f} kg)")
    else:
        print(f"\nNo push log found at {push_log}, skipping friction estimate.")

    push_sel = np.arange(len(pitch_B_masked)) <= np.argmax(pitch_B_masked)
    plot_torque_fit_result(
        pitch_rad=pitch_B_masked,
        tau_meas=w_O_app_masked[:, 4],
        tau_pred=(w_grav_pred[:, 4]),
        theta_star_rad=np.arctan2(COM_X, result_torque.x[0]),
        push_sel=push_sel,
        title="Torque fit result",
        show=True,
    )

def estimate_friction(push_log_path: str, mass_est: float) -> tuple:
    """
    Estimate table-object friction coefficient from a pure planar push log.

    During steady sliding with no squash: f_push_planar = μ * m * g
    => μ(t) = norm([fx, fy])(t) / (mass * g)

    Steady sliding is detected automatically as samples where f_planar is
    within one MAD of the global median — this excludes stick-slip transients
    and start/stop ramps without relying on arbitrary window fractions.
    """
    import matplotlib.pyplot as plt

    data = np.load(push_log_path)
    t = data["ft_time_s"]
    t = t - t[0]

    b, a = butter(4, 6, fs=500, btype='low')
    fx = filtfilt(b, a, data["fx"])
    fy = filtfilt(b, a, data["fy"])
    f_planar = np.sqrt(fx**2 + fy**2)

    # Trim: drop pre-contact zeros and end dropoff
    contact_threshold = 0.3  # N — below this is pre-contact or post-retract
    active = f_planar > contact_threshold
    first_active = np.argmax(active)
    last_active = len(f_planar) - np.argmax(active[::-1])  # last sample above threshold
    active_len = last_active - first_active
    trim_start = first_active + int(0.15 * active_len)  # drop first 15% (force ramp)
    trim_end   = first_active + int(0.85 * active_len)  # drop last 15% (retract dropoff)
    f_planar = f_planar[trim_start:trim_end]
    fx = fx[trim_start:trim_end]
    fy = fy[trim_start:trim_end]
    t = t[trim_start:trim_end]

    mu_t = f_planar / (mass_est * 9.81)

    # Steady sliding: samples within 1 MAD of the median (robust to transients)
    med = np.median(f_planar)
    mad = np.median(np.abs(f_planar - med))
    steady_mask = np.abs(f_planar - med) <= mad

    mu_est = float(np.median(mu_t[steady_mask]))
    mu_std = float(np.std(mu_t[steady_mask]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, fx, label="f_x", color="tab:red")
    axes[0].plot(t, fy, label="f_y", color="tab:green")
    axes[0].plot(t, f_planar, label="|f_planar|", color="tab:blue", linewidth=2)
    axes[0].axhline(med, color="k", linestyle="--", linewidth=1, label=f"median={med:.3f} N")
    axes[0].fill_between(t, med - mad, med + mad, alpha=0.15, color="tab:blue", label="±1 MAD (steady window)")
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("Push phase: planar forces")
    axes[0].legend(fontsize=10)
    axes[0].grid(True)

    axes[1].plot(t, mu_t, color="tab:orange", linewidth=1.5, label="μ(t) = |f_planar| / mg")
    axes[1].plot(t[steady_mask], mu_t[steady_mask], color="tab:green", linewidth=2,
                 linestyle="none", marker=".", markersize=2, label="steady samples")
    axes[1].axhline(mu_est, color="k", linestyle="--", linewidth=1.5, label=f"μ estimate = {mu_est:.4f}")
    axes[1].set_ylabel("μ estimate")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title(f"Friction coefficient over time  [μ = {mu_est:.4f} ± {mu_std:.4f}]")
    axes[1].legend(fontsize=10)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    return mu_est, mu_std


if __name__ == "__main__":
    main()
