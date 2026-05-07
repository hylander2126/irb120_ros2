"""Reusable plotting helpers for wrench and tipping-angle time histories."""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter


def _title_to_filename(title: str, suffix: str = ".png") -> str:
    """Convert a figure title into a filesystem-safe filename."""
    prefix = "../figures/squash_pull/"
    safe = re.sub(r"[{}$]", "", title.strip())
    safe = re.sub(r'[\\/:*?"<>|]+', "_", safe)
    safe = re.sub(r"\s+", "_", safe)
    safe = safe.strip("._")
    return f"{prefix}{safe or 'figure'}{suffix}"


def _save_figure_if_requested(fig, title: Optional[str], save_to_file: bool) -> None:
    """Save figure using title as filename when requested."""
    if not save_to_file:
        return
    fig.savefig(_title_to_filename(title or "figure"), dpi=300, bbox_inches="tight")


def _maybe_use_scientific_ticks(ax, values: np.ndarray) -> None:
    """Use scientific tick labels when magnitudes would produce long tick strings."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    vmax = float(np.max(np.abs(vals)))
    if vmax <= 0.0:
        return

    exponent = int(np.floor(np.log10(vmax)))
    if exponent >= 3 or exponent <= -3:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def _densify_ticks(ax) -> None:
    """Add minor ticks so visible tickmark density increases."""
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", length=3)


def plot_2d_wrench_geometry(
    pitch_rad: np.ndarray,
    p_finger_O: np.ndarray,
    f_O_app: np.ndarray,
    p_c_O: Optional[np.ndarray] = None,
    p_tipping_edge_O: Optional[np.ndarray] = None,
    *,
    num_snapshots: int = 8,
    figsize: Tuple[float, float] = (16, 10),
    title: Optional[str] = None,
    save_to_file: bool = False,
    show: bool = True,
):
    """Visualize 2D object geometry, COM, contact point, force, and moment arm at selected angles.
    
    Origin is placed at the tipping edge (pivot point during rotation).
    
    Args:
        pitch_rad: Tipping angle in radians, shape (N,).
        p_finger_O: Contact point position in object frame, shape (N,3).
        f_O_app: Applied force in object frame, shape (N,3).
        p_c_O: Center of mass in object frame, shape (3,). If None, assumed at origin.
        p_tipping_edge_O: Tipping edge (pivot) position in object frame, shape (3,). 
                          If None, assumed at (0, 0, -0.05) (bottom edge of object).
        num_snapshots: Number of time points to visualize.
        figsize: Figure size.
        title: Figure title.
        save_to_file: If True, save to disk using title as filename.
        show: If True, call plt.show().
    """
    pitch_rad = np.asarray(pitch_rad)
    p_finger_O = np.asarray(p_finger_O)
    f_O_app = np.asarray(f_O_app)
    
    if p_c_O is None:
        p_c_O = np.array([0.0, 0.0, 0.15])
    else:
        p_c_O = np.asarray(p_c_O)
    
    if p_tipping_edge_O is None:
        # Assume tipping edge is at the bottom-front corner of the object
        p_tipping_edge_O = np.array([0.0, 0.0, 0.0])
    else:
        p_tipping_edge_O = np.asarray(p_tipping_edge_O)
    
    # Select evenly-spaced snapshots
    n_samples = len(pitch_rad)
    indices = np.linspace(0, n_samples - 1, num_snapshots, dtype=int)
    
    # Create subplots
    ncols = min(4, num_snapshots)
    nrows = (num_snapshots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    
    for idx, (ax, sample_idx) in enumerate(zip(axes, indices)):
        if idx >= num_snapshots:
            ax.axis("off")
            continue
        
        pitch = pitch_rad[sample_idx]
        p_contact = p_finger_O[sample_idx, :]
        f_app = f_O_app[sample_idx, :]
        
        # Rotation matrix for the object tilt (rotation about Y-axis)
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)
        R_tilt = np.array([
            [cos_p, 0, sin_p],
            [0, 1, 0],
            [-sin_p, 0, cos_p]
        ])
        
        # Transform positions relative to tipping edge (pivot point)
        # This makes the tipping edge the origin (0, 0) in the 2D view
        p_c_rel_O = p_c_O - p_tipping_edge_O
        p_contact_rel_O = p_contact - p_tipping_edge_O
        
        # Apply rotation to get world/display frame coordinates
        p_c_world = R_tilt @ p_c_rel_O
        p_contact_world = R_tilt @ p_contact_rel_O
        
        # Force in world/display frame for consistent 2D arrow + torque computation
        f_app_world = R_tilt @ f_app

        # Moment arm from tipping edge (origin) to contact point
        r_moment = p_contact_world

        # Torque about tipping edge: tau = r x f (report y-component)
        torque_vec = np.cross(r_moment, f_app_world)
        tau_y = torque_vec[1]
        
        # Plot background grid
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linewidth=1.5, label="table surface")
        ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.5)
        
        # Draw a simple box representation of the object
        # Box anchored at tipping edge, extending upward and back
        box_size_x = 0.10
        box_size_z = 0.30
        # Corners relative to tipping edge (at origin in this coordinate system)
        # Tipping edge is at bottom-front of object
        box_corners = np.array([
            [0, 0, 0],              # bottom-front-left
            [box_size_x, 0, 0],               # bottom-front-right
            [box_size_x, 0, box_size_z],      # top-front-right
            [0, 0, box_size_z],     # top-front-left
        ])
        # Transform to account for tipping edge position in object frame
        box_corners_rel = box_corners - p_tipping_edge_O
        box_rotated = (R_tilt @ box_corners_rel.T).T
        box_2d = np.vstack([box_rotated[:, [0, 2]], box_rotated[0, [0, 2]]])
        ax.plot(box_2d[:, 0], box_2d[:, 1], "k-", linewidth=2, alpha=0.6, label="object")
        ax.fill(box_2d[:, 0], box_2d[:, 1], alpha=0.1, color="gray")
        
        # Plot tipping edge (origin) as a thick dot
        ax.plot(0, 0, "k*", markersize=20, label="tipping edge", zorder=6)
        
        # Plot COM (star)
        ax.plot(p_c_world[0], p_c_world[2], "g*", markersize=15, label="COM", zorder=5)
        
        # Plot contact point (circle)
        ax.plot(p_contact_world[0], p_contact_world[2], "ro", markersize=10, label="contact", zorder=5)
        
        # Plot moment arm (dashed line from tipping edge to contact)
        ax.plot(
            [0.0, p_contact_world[0]],
            [0.0, p_contact_world[2]],
            "b--",
            linewidth=2,
            alpha=0.7,
            label="moment arm r_edge",
        )
        
        # Plot applied force vector (scale for visibility)
        force_scale = 0.025
        ax.arrow(
            p_contact_world[0],
            p_contact_world[2],
            force_scale * f_app_world[0],
            force_scale * f_app_world[2],
            head_width=0.015,
            head_length=0.015,
            fc="red",
            ec="red",
            linewidth=2,
            label="applied force",
            zorder=4,
        )
        
        # Title with angle and torque info
        angle_deg = np.rad2deg(pitch)
        ax.set_title(
            f"Pitch: {angle_deg:.1f}°\n$\\tau_y$ = {tau_y:.4f} Nm",
            fontsize=11,
            fontweight="bold",
        )
        
        # Set axis limits to show full geometry with padding
        ax.set_xlim(-0.20, 0.25)
        ax.set_ylim(-0.08, 0.40)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)
    
    # Hide unused subplots
    for idx in range(num_snapshots, len(axes)):
        axes[idx].axis("off")
    
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    
    plt.tight_layout()
    _save_figure_if_requested(fig, title, save_to_file)
    
    if show:
        plt.show()
    
    return fig, axes


def plot_4vec_vs_angle(
    vec_xyzw: np.ndarray,
    pitch_rad: Optional[np.ndarray] = None,
    *,
    vec_labels: Sequence[str] = ("f_x", "f_y", "f_z", "tau_y"),
    x_label: str = "Tipping angle ||deg||",
    y_label: str = "Force (N)",
    torque_y_label: str = "Torque (Nm)",
    figsize: Tuple[float, float] = (10, 6),
    legend_fontsize: int = 13,
    line_width: float = 3.0,
    title: Optional[str] = None,
    save_to_file: bool = False,
    show: bool = True,
):
    """Plot wrench channels with optional pitch-angle overlay.

    Args:
        vec_xyzw: Vector channels of shape (N,4).
        pitch_rad: Optional tipping angle history in radians, shape (N,).
        vec_labels: Legend labels for vector x/y/z curves.
        y_label: Left-axis y-label text.
        figsize: Matplotlib figure size.
        legend_fontsize: Combined legend font size.
        line_width: Shared line width for plotted curves.
        save_to_file: If True, save figure to disk using the title as filename.
        show: If True, call plt.show().
        title: Optional title for the plot.

    Returns:
        fig, ax1
    """
    vec_xyzw = np.asarray(vec_xyzw)
    if vec_xyzw.ndim != 2 or vec_xyzw.shape[1] != 4:
        raise ValueError(
            f"Expected vec_xyz with shape (N, 4), got {vec_xyzw.shape}"
        )
    if len(vec_labels) != 4:
        raise ValueError("vec_labels must contain exactly 4 entries")
    
    pitch_deg = abs(np.rad2deg(pitch_rad) if pitch_rad is not None else None)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(pitch_deg, vec_xyzw[:, 0], color="tab:red", linewidth=line_width, label=vec_labels[0])
    ax1.plot(pitch_deg, vec_xyzw[:, 1], color="tab:green", linewidth=line_width, label=vec_labels[1])
    ax1.plot(pitch_deg, vec_xyzw[:, 2], color="tab:blue", linewidth=line_width, label=vec_labels[2])
    ax1.axhline(y=0, color="k", linewidth=1.5, label="_", alpha=0.7)

    ax_torque = ax1.twinx()
    ax_torque.plot(pitch_deg, vec_xyzw[:, 3], color="tab:orange", linewidth=line_width, label=vec_labels[3])

    # ax1.axvline(contact_time, color="k", linestyle="-", linewidth=2, label="first contact")
    ax1.set_xlabel(x_label, color="k")
    ax1.set_ylabel(y_label, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax_torque.set_ylabel(torque_y_label, color="tab:orange")
    ax_torque.tick_params(axis="y", labelcolor="tab:orange")
    _maybe_use_scientific_ticks(ax_torque, vec_xyzw[:, 3])
    _densify_ticks(ax1)
    _densify_ticks(ax_torque)
    ax1.grid(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines_t, labels_t = ax_torque.get_legend_handles_labels()
    ax1.legend(lines1 + lines_t, labels1 + labels_t, loc="best", fontsize=legend_fontsize)

    if title is not None:
        plt.title(title)

    align_zeros([ax1, ax_torque])
    plt.tight_layout()
    _save_figure_if_requested(fig, title, save_to_file)
    if show:
        plt.show()

    return fig, ax1

def plot_3vec_vs_angle(
    vec_xyz: np.ndarray,
    pitch_rad: Optional[np.ndarray] = None,
    *,
    vec_labels: Sequence[str] = ("f_x", "f_y", "f_z"),
    x_label: str = "Tipping angle ||deg||",
    y_label: str = "Force (N)",
    contact_time: float = 0.0,
    figsize: Tuple[float, float] = (10, 6),
    legend_fontsize: int = 13,
    line_width: float = 3.0,
    title: Optional[str] = None,
    save_to_file: bool = False,
    show: bool = True,
):
    """Plot wrench channels with optional pitch-angle overlay.

    Args:
        vec_xyz: Vector channels of shape (N,3).
        pitch_rad: Optional tipping angle history in radians, shape (N,).
        vec_labels: Legend labels for vector x/y/z curves.
        y_label: Left-axis y-label text.
        contact_time: X-location (s) for the vertical contact marker.
        figsize: Matplotlib figure size.
        legend_fontsize: Combined legend font size.
        line_width: Shared line width for plotted curves.
        save_to_file: If True, save figure to disk using the title as filename.
        show: If True, call plt.show().
        title: Optional title for the plot.

    Returns:
        fig, ax1
    """
    vec_xyz = np.asarray(vec_xyz)
    if vec_xyz.ndim != 2 or vec_xyz.shape[1] != 3:
        raise ValueError(
            f"Expected vec_xyz with shape (N, 3), got {vec_xyz.shape}"
        )
    if len(vec_labels) != 3:
        raise ValueError("vec_labels must contain exactly 3 entries")
    
    pitch_deg = abs(np.rad2deg(pitch_rad) if pitch_rad is not None else None)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(pitch_deg, vec_xyz[:, 0], color="tab:pink", linewidth=line_width, label=vec_labels[0])
    ax1.plot(pitch_deg, vec_xyz[:, 1], color="tab:olive", linewidth=line_width, label=vec_labels[1])
    ax1.plot(pitch_deg, vec_xyz[:, 2], color="tab:cyan", linewidth=line_width, label=vec_labels[2])

    # ax1.axvline(contact_time, color="k", linestyle="-", linewidth=2, label="first contact")
    ax1.set_xlabel(x_label, color="k")
    ax1.set_ylabel(y_label, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    _densify_ticks(ax1)
    ax1.grid(True)

    ax1.legend(loc="best", fontsize=legend_fontsize)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    _save_figure_if_requested(fig, title, save_to_file)
    if show:
        plt.show()

    return fig, ax1


def plot_wrench_and_tipping(
    t: np.ndarray,
    force_xyz: np.ndarray,
    torque_primary: np.ndarray,
    pitch_rad: Optional[np.ndarray] = None,
    *,
    torque_label: str = "tau_y",
    force_labels: Sequence[str] = ("f_x", "f_y", "f_z"),
    y_label: str = "Force (N)",
    contact_time: float = 0.0,
    figsize: Tuple[float, float] = (10, 6),
    legend_fontsize: int = 13,
    line_width: float = 3.0,
    title: Optional[str] = None,
    save_to_file: bool = False,
    show: bool = True,
):
    """Plot wrench channels with optional pitch-angle overlay.

    Args:
        t: Time axis of shape (N,).
        force_xyz: Force channels of shape (N,3).
        torque_primary: Primary torque channel of shape (N,).
        pitch_rad: Optional tipping angle history in radians, shape (N,).
        torque_label: Legend label for the torque curve.
        force_labels: Legend labels for force x/y/z curves.
        y_label: Left-axis y-label text.
        contact_time: X-location (s) for the vertical contact marker.
        figsize: Matplotlib figure size.
        legend_fontsize: Combined legend font size.
        line_width: Shared line width for plotted curves.
        save_to_file: If True, save figure to disk using the title as filename.
        show: If True, call plt.show().
        title: Optional title for the plot.

    Returns:
        fig, ax1, ax2 where ax2 is None if pitch_rad is not provided.
    """
    t = np.asarray(t)
    force_xyz = np.asarray(force_xyz)
    torque_primary = np.asarray(torque_primary)

    if t.ndim != 1:
        raise ValueError(f"Expected t with shape (N,), got {t.shape}")
    if force_xyz.shape != (t.size, 3):
        raise ValueError(f"Expected force_xyz with shape ({t.size}, 3), got {force_xyz.shape}")
    if torque_primary.shape != (t.size,):
        raise ValueError(
            f"Expected torque_primary with shape ({t.size},), got {torque_primary.shape}"
        )
    if len(force_labels) != 3:
        raise ValueError("force_labels must contain exactly 3 entries")

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(t, force_xyz[:, 0], color="tab:red", linewidth=line_width, label=force_labels[0])
    ax1.plot(t, force_xyz[:, 1], color="tab:green", linewidth=line_width, label=force_labels[1])
    ax1.plot(t, force_xyz[:, 2], color="tab:blue", linewidth=line_width, label=force_labels[2])
    ax1.plot(t, torque_primary, color="tab:orange", linewidth=line_width, label=torque_label)

    ax1.axvline(contact_time, color="k", linestyle="-", linewidth=2, label="first contact")
    ax1.set_xlabel("Time from first contact (s)")
    ax1.set_ylabel(f"{y_label}")
    ax1.tick_params(axis="y")
    # _maybe_use_scientific_ticks(ax1, np.concatenate([force_xyz.ravel(), torque_primary]))
    _densify_ticks(ax1)
    ax1.grid(True)

    ax2 = None
    if pitch_rad is not None:
        pitch_rad = np.asarray(pitch_rad)
        if pitch_rad.shape != (t.size,):
            raise ValueError(f"Expected pitch_rad with shape ({t.size},), got {pitch_rad.shape}")

        ax2 = ax1.twinx()
        ax2.plot(
            t,
            np.rad2deg(pitch_rad),
            color="black",
            linewidth=line_width,
            linestyle="-.",
            label="pitch angle",
        )
        ax2.set_ylabel("Tipping angle (deg)", color="black")
        ax2.tick_params(axis="y", labelcolor="black")
        _densify_ticks(ax2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="lower left",
            fontsize=legend_fontsize,
        )
        align_zeros([ax1, ax2])
    else:
        ax1.legend(loc="best", fontsize=legend_fontsize)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    _save_figure_if_requested(fig, title, save_to_file)
    if show:
        plt.show()

    return fig, ax1, ax2


def plot_torque_fit_result(
    pitch_rad: np.ndarray,
    tau_meas: np.ndarray,
    tau_pred: np.ndarray,
    theta_star_rad: float,
    *,
    push_sel: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    save_to_file: bool = False,
    show: bool = True,
):
    """Scatter measured tau_y and overlay model-predicted tau_y vs tipping angle.

    Args:
        pitch_rad: Tipping angle for each sample, shape (N,).
        tau_meas: Measured torque (tau_y) per sample, shape (N,).
        tau_pred: Model-predicted torque per sample, shape (N,).
        theta_star_rad: Estimated tipping angle (rad) — drawn as a vertical line.
        push_sel: Boolean mask (N,) selecting the push phase; retract = ~push_sel.
                  If None, all samples are treated as push phase.
        figsize: Matplotlib figure size.
        title: Optional figure title.
        save_to_file: If True, save figure to disk using the title as filename.
        show: If True, call plt.show().

    Returns:
        fig, ax
    """
    pitch_rad = np.asarray(pitch_rad)
    tau_meas  = np.asarray(tau_meas)
    tau_pred  = np.asarray(tau_pred)
    pitch_deg = np.abs(np.rad2deg(pitch_rad))

    if push_sel is None:
        push_sel = np.ones(len(pitch_deg), dtype=bool)
    retract_sel = ~push_sel

    sort_push    = np.argsort(pitch_deg[push_sel])
    sort_retract = np.argsort(pitch_deg[retract_sel])

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pitch_deg[push_sel],    tau_meas[push_sel],    alpha=0.5, color="tab:blue",   label="Push phase data")
    ax.scatter(pitch_deg[retract_sel], tau_meas[retract_sel], alpha=0.5, color="tab:orange", label="Retract phase data")
    ax.plot(pitch_deg[push_sel][sort_push],       tau_pred[push_sel][sort_push],       color="black", linewidth=2, label="Model (push)")
    ax.plot(pitch_deg[retract_sel][sort_retract], tau_pred[retract_sel][sort_retract], color="gray",  linewidth=2, label="Model (retract)")

    theta_star_deg = np.rad2deg(theta_star_rad)
    ax.axvline(x=theta_star_deg, color="green", linestyle="--", linewidth=2,
               label=f"Est. $\\theta^*$ ({theta_star_deg:.1f}°)")
    ax.axhline(y=0, color="cyan", linestyle="-", alpha=0.8)

    ax.set_xlabel("Object angle (deg)", fontsize=14)
    ax.set_ylabel("Torque (N·m)", fontsize=14)
    ax.grid(True, alpha=0.4)
    _densify_ticks(ax)
    ax.legend(fontsize=12)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    _save_figure_if_requested(fig, title, save_to_file)
    if show:
        plt.show()

    return fig, ax


## Helper function to align y-axis limits of multiple axes to zero
def align_zeros(axes):
    ylims_current = {}   #  Current ylims
    ylims_mod     = {}   #  Modified ylims
    deltas        = {}   #  ymax - ymin for ylims_current
    ratios        = {}   #  ratio of the zero point within deltas

    for ax in axes:
        ylims_current[ax] = list(ax.get_ylim())
                        # Need to convert a tuple to a list to manipulate elements.
        deltas[ax]        = ylims_current[ax][1] - ylims_current[ax][0]
        ratios[ax]        = -ylims_current[ax][0]/deltas[ax]
    
    for ax in axes:      # Loop through all axes to ensure each ax fits in others.
        ylims_mod[ax]     = [np.nan,np.nan]   # Construct a blank list
        ylims_mod[ax][1]  = max(deltas[ax] * (1-np.array(list(ratios.values()))))
                        # Choose the max value among (delta for ax)*(1-ratios),
                        # and apply it to ymax for ax
        ylims_mod[ax][0]  = min(-deltas[ax] * np.array(list(ratios.values())))
                        # Do the same for ymin
        ax.set_ylim(tuple(ylims_mod[ax]))
