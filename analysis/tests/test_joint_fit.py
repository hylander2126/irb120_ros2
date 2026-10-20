"""Synthetic checks for scene_profile / joint_fit. No recorded data, no GT."""
import numpy as np
from scene_profile import fillet_rho, chi2_profile, interval, R_GRID
from joint_fit import joint_geometry, initial_support, BRANCHES


def bins(R, a=.038, lean=0., flare_below=0., noise=0., seed=0):
    rng = np.random.default_rng(seed)
    h = np.arange(.0025, .045, .001)
    rho = fillet_rho(h, a, R) + lean * (h - .035) + rng.normal(0, noise, len(h))
    rho[h < flare_below] += .004 * (1 - h[h < flare_below] / flare_below)   # wedge-fill flare
    return h, rho


def test_profile_recovers_radius_with_lean():
    for R in (0., .008, .016, .024):
        h, rho = bins(R, lean=.03)
        c, _ = chi2_profile(h, rho, np.full(len(h), .0003))
        assert abs(R_GRID[np.argmin(c)] - R) < .0006


def test_flare_above_cutoff_excluded_no_false_fillet():
    h, rho = bins(0., flare_below=.006, noise=.0002)
    keep = h >= .007
    c, _ = chi2_profile(h[keep], rho[keep], np.full(keep.sum(), .0003))
    lo, hi = interval(c, 3.84)
    assert R_GRID[np.argmin(c)] == 0. and lo == 0.   # flare does not become a fillet
    assert hi <= .007 + .005                          # only radii hidden near the cutoff remain


def test_unobserved_band_widens_interval():
    h, rho = bins(.005)
    keep = h >= .009                          # all fillet below the cutoff
    c, _ = chi2_profile(h[keep], rho[keep], np.full(keep.sum(), .0003))
    lo, hi = interval(c, 3.84)
    assert lo == 0. and hi >= .008            # cannot distinguish 0..cutoff


def test_joint_motion_level_pins_radius_and_registration_cancels_in_arm():
    R, x_face, reg = .02, .60, .0015
    motion = {b: dict(level=x_face + R * (1 - t), mean_theta=t, sigma_level=.0003) for b, t in zip(BRANCHES, (.08, .1))}
    prior = np.zeros(len(R_GRID))             # uninformative camera prior
    g = joint_geometry(prior, x_face, motion, reg)
    assert abs(g['posterior']['best_R_m'] - R) < .0002
    # A registration shift of the camera face is absorbed by the GLS support estimate.
    p0 = initial_support(R, x_face - .001, motion, reg)
    assert abs(p0 - (x_face + R)) < .001


def test_cloud_motion_conflict_detected():
    R_true, x_face = .02, .60
    motion = {b: dict(level=x_face + R_true * (1 - t), mean_theta=t, sigma_level=.0003) for b, t in zip(BRANCHES, (.08, .1))}
    prior = ((R_GRID - .005) / .001) ** 2     # camera confidently says 5 mm
    g = joint_geometry(prior, x_face, motion, .001)
    assert g['cloud_motion_conflict_chi2'] > 6.63
