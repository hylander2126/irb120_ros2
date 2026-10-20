"""Shape-free instantaneous-support estimate ("orange" baseline), press-pull stages.

From the cylinder CAD-free replay: the support path comes from ball kinematics alone
(motion_contact), the horizontal CoM is supplied by the caller (cloud axis or hull
centroid, an assumption), and m, z come from a bounded gravity fit per branch.
"""
import numpy as np

import _paths  # noqa: F401
from irb120_control.estimation.physical_bounds import cloud_height_bound
from motion_contact import gravity_fit, origin_x, reconstruct, regularize_support
from rolling import tau_at

BRANCHES = ('tip', 'return')


def estimate(log, com_x_world, degree=3, window_s=.6, height_cap=None):
    """height_cap: only for logs without a pre-action cloud (legacy trials)."""
    cap = height_cap if height_cap is not None else cloud_height_bound(log)[0]
    start = np.where(log.branch == 'tip')[0][0]
    motion = regularize_support(reconstruct(log.t_epoch, log.p_ball, start, window_s=window_s),
                                log.p_ball, log.branch, degree)
    phi = motion['theta']; mask = motion['valid'] & log.contact & (phi > np.deg2rad(1))
    ox = origin_x(motion['smooth'], phi, start, com_x_world)
    tau = tau_at(log, motion['pivot'])
    fits = {b: gravity_fit(phi, tau, ox - motion['pivot'][:, 0], cap, mask & (log.branch == b)) for b in BRANCHES}
    masses = [fits[b]['m'] for b in BRANCHES]
    return dict(branches=fits, mass_kg=float(np.mean(masses)), height_m=float(np.mean([fits[b]['z'] for b in BRANCHES])),
                branch_mass_split=float(np.ptp(masses) / np.mean(masses)), height_cap_m=cap)
