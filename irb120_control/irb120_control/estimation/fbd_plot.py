#!/usr/bin/env python3
"""
Free-body diagram animation: 2D x-z plane view of the tipping object
at several time steps, showing all applied forces and torques.

Usage:
    python3 fbd_plot.py [--object soda] [--workspace /path/to/ws] [--steps 6]
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt

# Allow running directly without install
_this_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root  = os.path.abspath(os.path.join(_this_dir, "../../.."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from irb120_control.estimation.helper_fns   import rotvec_to_rot, quat_to_rotvec
from irb120_control.estimation.com_estimation import model_bkwd_wrench

# ── tunables ────────────────────────────────────────────────────────────────
ALL_OBJECTS = ["box", "heart", "soda", "flashlight", "monitor"]
COM_X       = 0.05          # metres — half object depth / pivot-to-COM x
OBJ_W       = 0.10          # object half-width  (x, metres) for the box glyph
OBJ_H       = 0.30          # object half-height (z, metres) for the box glyph

PIVOT_X = {
    "soda":       0.4839,
    "box":        0.5100,
    "heart":      0.5100,
    "flashlight": 0.4839,
    "monitor":    0.4839,
}

# LPFs (same as estimate_params)
_B6, _A6   = butter(4, 6,   fs=500, btype='low')
_B05, _A05 = butter(2, 0.5, fs=500, btype='low')

def _lpf(x, ax=0):
    return filtfilt(_B6,  _A6,  x, axis=ax) if x.shape[0] > 20 else x

def _lpf_slow(x, ax=0):
    return filtfilt(_B05, _A05, x, axis=ax) if x.shape[0] > 20 else x


# ── geometry helpers ─────────────────────────────────────────────────────────
def _rot2d(theta):
    """Standard 2-D CCW rotation in x-z plane. Positive theta = CCW."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _obj_corners(pitch, pivot, w=OBJ_W, h=OBJ_H):
    """Return (4,2) corners of the object box in world x-z, rotated by pitch."""
    # object frame corners (x,z) relative to pivot (= object frame origin)
    # pivot at bottom-left corner → corners span [0, 2w] in x, [0, h] in z
    cx = np.array([0, 2*w, 2*w,   0])
    cz = np.array([0,   0,   h,   h])
    R  = _rot2d(pitch)          # pitch is rotation about y → CCW in x-z plane
    pts = R @ np.vstack([cx, cz])   # (2, 4)
    return (pts + np.array([[pivot[0]], [pivot[2]]])).T  # (4, 2) world x-z


def _draw_object(ax, pitch, pivot, alpha=0.35, color='steelblue'):
    corners = _obj_corners(pitch, pivot)
    poly = plt.Polygon(corners, closed=True, facecolor=color, edgecolor='navy',
                       linewidth=1.5, alpha=alpha, zorder=2)
    ax.add_patch(poly)


def _arrow(ax, origin, vec, scale, color, label=None, lw=2, head_w=None, zorder=5):
    """Draw a scaled vector arrow originating at origin (x,z)."""
    if np.linalg.norm(vec) < 1e-9:
        return
    dx, dz = vec[0] * scale, vec[1] * scale
    length = np.hypot(dx, dz)
    hw = head_w if head_w else length * 0.18
    hl = hw * 1.5
    ax.arrow(origin[0], origin[1], dx, dz,
             head_width=hw, head_length=hl,
             fc=color, ec=color, lw=lw,
             length_includes_head=True, zorder=zorder)
    if label:
        ax.text(origin[0] + dx * 1.15, origin[1] + dz * 1.15, label,
                color=color, fontsize=7, ha='center', va='center', zorder=zorder + 1)


def _arc_torque(ax, origin, tau_y, radius=0.025, color='darkorange', label=None, zorder=5):
    """Draw a curved arc arrow to indicate a torque about y (in x-z plane)."""
    if abs(tau_y) < 1e-9:
        return
    theta0 = np.deg2rad(-60) if tau_y > 0 else np.deg2rad(60)
    dtheta  = np.deg2rad(120) * np.sign(tau_y)
    thetas  = np.linspace(theta0, theta0 + dtheta, 60)
    xs = origin[0] + radius * np.cos(thetas)
    zs = origin[1] + radius * np.sin(thetas)
    ax.plot(xs, zs, color=color, lw=2, zorder=zorder)
    # arrowhead at the end using ax.arrow for reliable rendering
    tx = xs[-1]; tz = zs[-1]
    dtx = xs[-1] - xs[-3]; dtz = zs[-1] - zs[-3]
    ax.arrow(tx - dtx, tz - dtz, dtx, dtz,
             head_width=0.012, head_length=0.012,
             fc=color, ec=color, lw=0, length_includes_head=True, zorder=zorder)
    if label:
        lx = origin[0] + radius * 1.55 * np.cos(theta0 + dtheta * 0.5)
        lz = origin[1] + radius * 1.55 * np.sin(theta0 + dtheta * 0.5)
        ax.text(lx, lz, label, color=color, fontsize=7, ha='center', va='center', zorder=zorder + 1)


# ── FBD for a single time-step ───────────────────────────────────────────────
def _draw_fbd(ax, pitch, pivot, p_finger_B, f_app_O, tau_app_O_y,
              f_scale=0.015, tau_scale=None, title=""):
    """
    Draw the 2-D FBD on `ax` in world x-z.

    pitch       : scalar tipping angle (rad, about y)
    pivot       : (3,) world position of pivot (x,y,z)
    p_finger_B  : (3,) finger contact position in world
    f_app_O     : (3,) applied force in object frame
    tau_app_O_y : scalar applied torque about object y-axis
    """
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x  (m)", fontsize=8)
    ax.set_ylabel("z  (m)", fontsize=8)
    ax.set_title(title, fontsize=8)

    # table surface
    ax.axhline(pivot[2], color='saddlebrown', lw=2, zorder=1)

    # pivot marker
    ax.plot(pivot[0], pivot[2], 'k^', ms=6, zorder=6)

    # object box (rotated) — pitch is positive (CCW viewed from +Y)
    _draw_object(ax, pitch, pivot)

    # rotate applied force from object frame to world x-z using R_y(pitch):
    # x_W = x_O*cos + z_O*sin,  z_W = -x_O*sin + z_O*cos
    c, s = np.cos(pitch), np.sin(pitch)
    R_y_xz = np.array([[c, s], [-s, c]])  # R_y in x-z subspace (NOT standard 2D CCW)
    R = _rot2d(pitch)  # kept for COM arrow (box geometry uses CCW convention)
    f_xz_O = np.array([f_app_O[0], f_app_O[2]])
    f_xz_W = R_y_xz @ f_xz_O

    pivot_xz = np.array([pivot[0], pivot[2]])

    # contact point rotates with the box (CCW = _rot2d convention)
    r_xz_O = np.array([p_finger_B[0], p_finger_B[2]])
    contact_xz = pivot_xz + R @ r_xz_O

    # lever arm: pivot → contact point in world (dashed purple)
    r_vec = contact_xz - pivot_xz
    ax.annotate("", xy=contact_xz, xytext=pivot_xz,
                arrowprops=dict(arrowstyle="-|>", color='purple', lw=1.5, linestyle='dashed'),
                zorder=4)
    mid = pivot_xz + r_vec * 0.5
    ax.text(mid[0] - 0.015, mid[1], f"r=({r_vec[0]:.3f}, {r_vec[1]:.3f})",
            color='purple', fontsize=6, ha='right', va='center', zorder=6)

    # applied force arrow at contact point (world frame)
    _arrow(ax, contact_xz, f_xz_W, scale=f_scale,
           color='crimson', label=f"$F_{{app}}$\n({np.linalg.norm(f_app_O):.1f} N)")

    # tau_y > 0 → CCW arc viewed from +Y → consistent with positive tipping rotation
    _arc_torque(ax, pivot_xz, tau_app_O_y, radius=0.05,
                color='darkorange', label=f"τ={tau_app_O_y:.3f} Nm")

    # gravity arrow at object COM — center of box in object frame is (OBJ_W, OBJ_H/2)
    com_O_xz = np.array([OBJ_W, OBJ_H / 2])
    com_W_xz = R @ com_O_xz + np.array([pivot[0], pivot[2]])
    _arrow(ax, com_W_xz, np.array([0.0, -1.0]), scale=f_scale * 3,
           color='forestgreen', label="$W$", lw=1.5)

    # contact point marker
    ax.plot(contact_xz[0], contact_xz[1], 'ro', ms=5, zorder=7)

    # auto-set limits with padding — wide enough for arrows
    pad = 0.08
    ax.set_xlim(pivot[0] - pad, pivot[0] + 2*OBJ_W + pad)
    ax.set_ylim(pivot[2] - 0.03, pivot[2] + OBJ_H + pad)


# ── main routine ─────────────────────────────────────────────────────────────
def run(obj: str, workspace_root: str, n_steps: int):
    squash_file = os.path.join(workspace_root, "runtime_logs", obj, "squash", "most_recent.npz")
    if not os.path.exists(squash_file):
        print(f"[{obj}] No squash file at {squash_file}")
        return

    data = np.load(squash_file)

    time_ft  = data["ft_time_s"]
    time_ee  = data["pose_time_s"]

    f_meas_S = _lpf(np.column_stack([data["fx"], data["fy"], data["fz"]]))
    t_meas_S = _lpf(np.column_stack([data["tx"], data["ty"], data["tz"]]))
    p_ft_B   = np.column_stack([data["ft_px"], data["ft_py"], data["ft_pz"]])
    Q_ft     = np.column_stack([data["ft_qx"], data["ft_qy"], data["ft_qz"], data["ft_qw"]])
    p_ee_B   = np.column_stack([data["x"], data["y"], data["z"]])

    # contact detection
    contact_ft_idx     = np.argmax(np.linalg.norm(f_meas_S, axis=1) > 0.5)
    time_contact_start = time_ft[contact_ft_idx]

    p_pivot_B = np.array([PIVOT_X[obj], 0.0, 0.0])

    # proprioceptive tipping angle
    contact_ee_idx = np.argmin(np.abs(time_ee - time_contact_start))
    p_contact = p_ee_B[contact_ee_idx]
    r0  = np.array(p_contact - p_pivot_B)
    r_t = np.column_stack([p_ee_B[:, 0] - p_pivot_B[0], p_ee_B[:, 2] - p_pivot_B[2]])
    prop_angle_deg = -np.degrees(np.arctan2(r_t[:, 0] * r0[2] - r_t[:, 1] * r0[0],
                                            r_t[:, 0] * r0[0] + r_t[:, 1] * r0[2]))
    prop_angle_deg[:contact_ee_idx] = 0.0

    pitch_B      = np.deg2rad(np.interp(time_ft, time_ee, prop_angle_deg))
    contact_mask = pitch_B > 0.0
    N_c          = contact_mask.sum()

    r_contact_O = r0   # no-slip → constant in object frame

    # applied wrench (object frame) — full contact window
    R_ft_c   = rotvec_to_rot(quat_to_rotvec(Q_ft[contact_mask]))
    T_B_S    = np.zeros((N_c, 4, 4))
    T_B_S[:, :3, :3] = R_ft_c
    T_B_S[:, :3,  3] = p_ft_B[contact_mask]
    T_B_S[:,  3,  3] = 1.0

    pitch_c  = pitch_B[contact_mask]
    rv_obj   = np.zeros((N_c, 3)); rv_obj[:, 1] = -pitch_c  # model_bkwd_wrench expects original negative-convention pitch
    R_obj_c  = rotvec_to_rot(rv_obj)
    T_B_O    = np.zeros((N_c, 4, 4))
    T_B_O[:, :3, :3] = R_obj_c
    T_B_O[:, :3,  3] = p_pivot_B
    T_B_O[:,  3,  3] = 1.0

    w_meas_S = np.hstack([f_meas_S[contact_mask], t_meas_S[contact_mask]])
    w_app_O  = _lpf_slow(model_bkwd_wrench(w_meas_S, T_B_S, T_B_O, r_contact_O))

    # pick n_steps evenly-spaced contact indices
    indices = np.linspace(0, N_c - 1, n_steps, dtype=int)

    # ── figure layout ─────────────────────────────────────────────────────────
    cols = min(n_steps, 3)
    rows = (n_steps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows))
    axes = np.array(axes).ravel()

    # adaptive force scale: normalise so 1 N → ~3 cm on plot
    f_max  = max(np.linalg.norm(w_app_O[:, :3], axis=1).max(), 1.0)
    f_scale = 0.12 / f_max   # metres per Newton

    for k, idx in enumerate(indices):
        ax = axes[k]

        pitch_k   = pitch_c[idx]
        f_app_O_k = w_app_O[idx, :3]
        tau_y_k   = w_app_O[idx, 4]

        _draw_fbd(ax, pitch_k, p_pivot_B, r_contact_O,
                  f_app_O_k, tau_y_k, f_scale=f_scale,
                  title=f"step {k+1}/{n_steps}  θ={np.rad2deg(pitch_k):.1f}°  τ_y={tau_y_k:.3f} Nm")

    # hide unused axes
    for ax in axes[n_steps:]:
        ax.set_visible(False)

    # shared legend
    legend_handles = [
        mpatches.Patch(facecolor='steelblue', alpha=0.4, edgecolor='navy', label='Object'),
        mpatches.Patch(color='crimson',     label='Applied force $F_{app}$'),
        mpatches.Patch(color='darkorange',  label='Applied torque $τ_y$'),
        mpatches.Patch(color='forestgreen', label='Weight $W$'),
        mpatches.Patch(color='black',       label='Pivot'),
        mpatches.Patch(color='red',         label='Contact point'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=6,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f"Free-Body Diagram — {obj}  (x-z plane, {n_steps} time steps)",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_dir  = os.path.join(workspace_root, "runtime_logs", obj)
    out_path = os.path.join(out_dir, "fbd_timesteps.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[{obj}] FBD saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--steps",     type=int, default=6)
    args = parser.parse_args()

    ws = args.workspace or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    for obj in ALL_OBJECTS:
        run(obj, ws, args.steps)

    plt.show()


if __name__ == "__main__":
    main()
