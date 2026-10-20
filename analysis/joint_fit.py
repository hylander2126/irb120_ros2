"""Prior (pre-contact perception) -> posterior (after interaction) support geometry, then mass.

No CAD/GT in inference. Scoring references are loaded only after all estimates finish.

Geometry is ONE fit over the lower-transition radius R (circular family) using:
  prior   : per-camera scene profile of the support-side view (scene_profile), chi2(R)
            minimized over the flare cutoff (cutoff is a nuisance parameter).
  motion  : shape-free instantaneous support (motion_contact) path LEVEL per branch: mean
            support x and mean tilt over the branch. Under the family the support starts R
            inside the support-side wall face and rolls back R per radian, so
            mean_P_b = x_face + R (1 - mean_theta_b). A free-slope intercept is NOT used: it
            would import the unreliable slope through extrapolation. x_face carries a
            camera-registration error shared by both branches (2x2 covariance), sized from
            cross-camera table disagreement.
The support path SLOPE (-R under the family) is a second derivative of position and is
used only as a diagnostic, not as evidence.

Mass/CoM then reuse the existing rolling-support torque model at the posterior geometry,
with the posterior propagated by refitting at its 2.5/16/50/84/97.5 percentiles.
Gates and thresholds below were fixed before scoring (2026-10-20); see JOINT_FIT.md.
"""
from pathlib import Path
import copy
import numpy as np

import _paths
from irb120_control.estimation.estimate_params import tiplog_from_press_pull
from irb120_control.estimation.physical_bounds import cloud_height_bound, bounded_gravity_fit
from motion_contact import reconstruct, regularize_support
from rolling import rolling_angle, tau_at
from scene_profile import R_GRID, envelope_profile, interval

MAX_ELEVATION_DEG = 45.     # steeper views cannot see a downward-facing lower transition
REG_FLOOR_M = .001
WINDOWS = (.3, .6, 1.2)
DEGREES = (2, 3, 4, 5)
BOOT = 200
P_GATE = 6.63               # chi2(1) at p = .01
SLOPE_GATE = 3.0            # sigma
MASS_SPLIT_BORDERLINE = .10
MASS_SPLIT_REJECT = .20
TORQUE_RMS_GATE = 5.0       # mN m, as quality.py
VISION_K_GATE = .05         # |k_tip - k_return|, vision/motion angle ratio
BRANCHES = ('tip', 'return')


def motion_intercepts(log, start, rng):
    """Branch path level (mean support x, mean tilt) and the diagnostic slope;
    sigma = smoother-setting spread (+) block-bootstrap SE."""
    rows = {b: [] for b in BRANCHES}; boot = {b: [] for b in BRANCHES}
    for w in WINDOWS:
        for deg in DEGREES:
            m = regularize_support(reconstruct(log.t_epoch, log.p_ball, start, window_s=w), log.p_ball, log.branch, deg)
            th = m['theta']
            for b in BRANCHES:
                u = np.where(m['valid'] & log.contact & (th > np.deg2rad(1)) & (log.branch == b))[0]
                beta = np.polyfit(th[u], m['pivot'][u, 0], 1)[0]
                rows[b].append((m['pivot'][u, 0].mean(), beta, th[u].mean()))
                if w == .6 and deg == 3:
                    block = max(m['window_samples'], 5); nb = max(len(u) // block, 2)
                    for _ in range(BOOT):
                        pick = np.concatenate([u[k * block:(k + 1) * block] for k in rng.integers(0, nb, nb)])
                        boot[b].append((m['pivot'][pick, 0].mean(), np.polyfit(th[pick], m['pivot'][pick, 0], 1)[0]))
    out = {}
    for b in BRANCHES:
        r = np.array(rows[b]); bt = np.array(boot[b])
        central = r[WINDOWS.index(.6) * len(DEGREES) + DEGREES.index(3)]
        sd = np.sqrt(r[:, :2].std(0) ** 2 + bt.std(0) ** 2)
        out[b] = dict(level=float(central[0]), mean_theta=float(central[2]), slope=float(central[1]),
                      sigma_level=float(sd[0]), sigma_slope=float(sd[1]),
                      level_settings=r[:, 0].tolist(), slope_settings=r[:, 1].tolist())
    return out


def registration_sigma(views):
    z = [v['table']['plane'][2] for v in views]
    return max(REG_FLOOR_M, float(np.ptp(z) / np.sqrt(2))) if len(z) >= 2 else None


def joint_geometry(prior_chi2, x_face, motion, sig_reg):
    level = np.array([motion[b]['level'] for b in BRANCHES])
    lever = 1 - np.array([motion[b]['mean_theta'] for b in BRANCHES])
    C = np.diag([motion[b]['sigma_level'] ** 2 for b in BRANCHES]) + sig_reg ** 2
    Ci = np.linalg.inv(C)
    mchi = np.array([(level - x_face - R * lever) @ Ci @ (level - x_face - R * lever) for R in R_GRID])
    joint = prior_chi2 + mchi
    def summary(c):
        j = int(np.argmin(c)); p = np.exp(-.5 * (c - c.min())); cdf = np.cumsum(p) / p.sum()
        pct = {q: float(np.interp(q / 100, cdf, R_GRID)) for q in (2.5, 16, 50, 84, 97.5)}
        return dict(best_R_m=float(R_GRID[j]), R_68_m=interval(c, 1.), R_95_m=interval(c, 3.84), percentiles_m=pct, chi2_min=float(c.min()))
    conflict = float(joint.min() - prior_chi2.min() - mchi.min())
    return dict(prior=summary(prior_chi2), motion_only=summary(mchi), posterior=summary(joint),
                cloud_motion_conflict_chi2=conflict, registration_sigma_m=sig_reg,
                chi2=dict(prior=prior_chi2.tolist(), motion=mchi.tolist(), joint=joint.tolist()))


def initial_support(R, x_face, motion, sig_reg):
    """GLS estimate of the zero-tilt support x given R: the registration offset of x_face
    is estimated from the branch path levels (prior mean 0, sigma sig_reg)."""
    w = np.array([1 / motion[b]['sigma_level'] ** 2 for b in BRANCHES])
    r = np.array([motion[b]['level'] - R * (1 - motion[b]['mean_theta']) for b in BRANCHES]) - x_face
    delta = float(np.sum(w * r) / (w.sum() + 1 / sig_reg ** 2))
    return x_face + delta + R


def vision_check(log, start, track):
    """Per-branch vision/motion angle ratio from a tilt_vision TiltTrack (None: unavailable)."""
    if track is None:
        return dict(available=False)
    phi = track.rotvec @ log.tilt_dir
    m = regularize_support(reconstruct(log.t_epoch, log.p_ball, start), log.p_ball, log.branch, 3)
    th = m['theta']; out = dict(available=True, branches={})
    for b in BRANCHES:
        u = m['valid'] & log.contact & (th > np.deg2rad(1)) & (log.branch == b) & track.valid
        if u.sum() < 20:
            return dict(available=False, reason=f'{b}: too few valid vision samples')
        k = float(np.sum(phi[u] * th[u]) / np.sum(th[u] ** 2))
        out['branches'][b] = dict(n=int(u.sum()), vision_over_motion_angle=k,
            residual_rms_deg=float(np.degrees(np.sqrt(np.mean((phi[u] - k * th[u]) ** 2)))),
            vision_max_deg=float(np.degrees(phi[u].max())), motion_max_deg=float(np.degrees(th[u].max())))
    ks = [out['branches'][b]['vision_over_motion_angle'] for b in BRANCHES]
    out['branch_k_difference'] = float(abs(ks[0] - ks[1]))
    out['branches_visually_symmetric'] = out['branch_k_difference'] <= VISION_K_GATE
    return out


def mass_fits(log, R, p0x, arm, cap, mask):
    settled = copy.copy(log); settled.contact = log.contact.copy()
    settled.contact[:np.where(log.branch == 'tip')[0][0]] = False
    p = np.array([p0x, log.pivot[1], 0.])
    angle, _ = rolling_angle(settled, p, R, .01325)
    moving = np.broadcast_to(p, log.p_ball.shape).copy(); moving[:, 0] -= R * angle
    tau = tau_at(log, moving); fits = {}
    for b in BRANCHES:
        use = mask & (log.branch == b)
        m, h, active = bounded_gravity_fit(angle[use], tau[use], arm, R, cap)
        rms = np.sqrt(np.mean((9.81 * m * (arm * np.cos(angle[use]) - (h - R) * np.sin(angle[use])) - tau[use]) ** 2)) * 1000
        fits[b] = dict(m=m, z=h, active_bounds=active, rms_mNm=float(rms), n=int(use.sum()))
    return fits


def support_view(scene_folder, pivot, push_dir):
    """Shape-agnostic support-side envelope profile and the low-elevation camera facing it.
    The scene cloud is read from the snapshot or its derived mirror (_paths.scene_dir)."""
    selector = np.load(Path(scene_folder) / 'object_cloud.npz')['xyz']
    scene = envelope_profile(scene_folder, selector, pivot, push_dir, cloud_dir=_paths.scene_dir(scene_folder))
    center, support_dir = np.array(scene['center_xy']), np.array(scene['support_dir'])
    low = [v for v in scene['views'] if v['status'] == 'ok' and v['elevation_deg'] <= MAX_ELEVATION_DEG]
    if not low:
        raise ValueError('no low-elevation camera sees the support-side face')
    view = max(low, key=lambda v: v['facing'])
    return scene, center, support_dir, low, view


def outer_face_x(scene_folder, pivot, push_dir):
    """World x of the support-side outer wall (35 mm wall line of the support camera), and
    the unit direction from the axis to the support."""
    _, center, support_dir, _, view = support_view(scene_folder, pivot, push_dir)
    return float(center[0] + support_dir[0] * view['fits']['5mm']['outer_radius_m']), support_dir


def run(name, log_path, scene_folder, sig_reg_fallback=None, seed=0, vision_track=None, envelope_d=None,
        com_x_world=None):
    """CoM source, one of: envelope_d = supplied distance from the support-side OUTER base
    edge (table-edge measurement); com_x_world = a world x (e.g. the hull centroid)."""
    rng = np.random.default_rng(seed)
    log = tiplog_from_press_pull(str(log_path)); start = np.where(log.branch == 'tip')[0][0]
    cap, cloud_path = cloud_height_bound(log)
    scene, center, support_dir, low, view = support_view(scene_folder, log.pivot, log.push_dir)
    # Union of per-cutoff intervals: each cutoff's delta-chi2, minimized over cutoffs.
    # (Raw chi2 is not comparable across cutoffs: fewer bins give smaller chi2.)
    prior_chi2 = np.min([np.array(f['chi2_profile']) - min(f['chi2_profile']) for f in view['fits'].values()], axis=0)
    main = view['fits']['5mm']
    # Wall position at the table (lean extrapolated from the wall line at 35 mm).
    a_bottom = main['outer_radius_m'] - .035 * main['wall_lean_mm_per_10mm'] / 10
    x_face = float(center[0] + support_dir[0] * a_bottom)
    # Outer envelope on the support side (wall at 35 mm, no lean extrapolation): the face a
    # table-edge measurement is referenced to.
    x_face_outer = float(center[0] + support_dir[0] * main['outer_radius_m'])
    sig_reg = registration_sigma(low)
    reg_source = 'this scene, low-elevation views'
    if sig_reg is None:
        if sig_reg_fallback is None:
            raise ValueError('single usable low-elevation view and no pooled registration sigma')
        sig_reg, reg_source = sig_reg_fallback, 'pooled median of other scenes (single usable view)'
    motion = motion_intercepts(log, start, rng)
    geom = joint_geometry(prior_chi2, x_face, motion, sig_reg)
    post = geom['posterior']
    m3 = regularize_support(reconstruct(log.t_epoch, log.p_ball, start), log.p_ball, log.branch, 3)
    mask = m3['valid'] & log.contact & (m3['theta'] > np.deg2rad(1))
    propagated = {}
    for q, R in post['percentiles_m'].items():
        R = max(R, 0.)
        p0x = initial_support(R, x_face, motion, sig_reg)
        # Arm = cloud axis (all cameras) - support (robot frame), as replay.py. Not a_bottom - R:
        # that would drop the registration offset the GLS support estimate absorbs.
        if envelope_d is not None:
            com_x = x_face_outer - support_dir[0] * envelope_d
        elif com_x_world is not None:
            com_x = com_x_world
        else:
            raise ValueError('joint_fit.run needs envelope_d or com_x_world')
        arm = float(com_x - p0x) * float(-np.sign(support_dir[0]))
        propagated[str(q)] = dict(R_m=R, p0_x=p0x, arm_m=arm,
                                  branches=mass_fits(log, R, p0x, arm, cap, mask))
    # Registration error of the cloud axis moves the arm directly (mass ~ 1/arm).
    med = propagated['50']
    for tag, sign in (('reg_minus', -1), ('reg_plus', 1)):
        arm = med['arm_m'] + sign * 1.96 * sig_reg
        propagated[tag] = dict(R_m=med['R_m'], p0_x=med['p0_x'], arm_m=arm,
                               branches=mass_fits(log, med['R_m'], med['p0_x'], arm, cap, mask))
    vision = vision_check(log, start, vision_track)
    result = dict(object=name, log=str(log_path), scene=str(scene_folder),
        support_view=view['camera'], support_view_elevation_deg=view['elevation_deg'], x_face_m=x_face,
        outer_radius_bottom_m=a_bottom, axis_xy_m=scene['center_xy'], height_cap_m=cap,
        registration_sigma_source=reg_source, envelope_d=envelope_d, motion=motion, geometry=geom, propagated=propagated, vision=vision,
        views=[dict(camera=v['camera'], status=v['status'], elevation_deg=v.get('elevation_deg'),
                    R_95_envelope_m=v.get('R_95_envelope_m'), fits={k: {kk: vv for kk, vv in f.items() if kk != 'chi2_profile'}
                    for k, f in v.get('fits', {}).items()}) for v in scene['views']])
    result['gates'] = gates(result)
    return result, scene, view


def gates(r):
    flags, branch_flags = [], {b: [] for b in BRANCHES}
    g = r['geometry']; post = g['posterior']; R = post['best_R_m']
    if g['cloud_motion_conflict_chi2'] > P_GATE:
        flags.append('cloud_motion_conflict')
    mo = r['motion']
    da = (mo['tip']['level'] - mo['return']['level']) - R * (mo['return']['mean_theta'] - mo['tip']['mean_theta'])
    if da ** 2 / (mo['tip']['sigma_level'] ** 2 + mo['return']['sigma_level'] ** 2) > P_GATE:
        flags.append('branch_support_levels_disagree')
    sigR = (post['percentiles_m'][84] - post['percentiles_m'][16]) / 2
    for b in BRANCHES:
        z = abs(mo[b]['slope'] + R) / np.hypot(mo[b]['sigma_slope'], sigR)
        if z > SLOPE_GATE:
            branch_flags[b].append('support_path_not_circular_rolling')
    fits = r['propagated']['50']['branches']
    for b in BRANCHES:
        if fits[b]['active_bounds']:
            branch_flags[b].append('parameter_bound_active')
        if fits[b]['rms_mNm'] > TORQUE_RMS_GATE:
            branch_flags[b].append('torque_rms_above_5mNm')
    masses = [fits[b]['m'] for b in BRANCHES]
    split = float(np.ptp(masses) / np.mean(masses))
    v = r['vision']
    if v.get('available') and not v['branches_visually_symmetric']:
        flags.append('vision_branch_asymmetry')
    hard = ('cloud_motion_conflict' in flags or split > MASS_SPLIT_REJECT
            or any('parameter_bound_active' in branch_flags[b] for b in BRANCHES)
            or all('torque_rms_above_5mNm' in branch_flags[b] for b in BRANCHES))
    soft = flags or split > MASS_SPLIT_BORDERLINE or any(branch_flags.values())
    status = 'rejected' if hard else 'borderline' if soft else 'accepted'
    return dict(status=status, flags=flags, branch_flags=branch_flags, branch_mass_split=split)


def summarize(r):
    """Median-R point; 95% band combines posterior-R spread and registration (arm) error
    in quadrature, side by side. Excludes F/T, angle-source and model-form error."""
    P = r['propagated']
    mean = lambda q, k: float(np.mean([P[q]['branches'][b][k] for b in BRANCHES]))
    out = {}
    for k, unit, name in (('m', 1000, 'mass_g'), ('z', 1000, 'height_mm')):
        mid = mean('50', k)
        ends = [mean(q, k) - mid for q in ('2.5', '97.5', 'reg_minus', 'reg_plus')]
        lo = np.hypot(min(0, *ends[:2]), min(0, *ends[2:])); hi = np.hypot(max(0, *ends[:2]), max(0, *ends[2:]))
        out[name] = unit * mid; out[name + '_95'] = [unit * (mid - lo), unit * (mid + hi)]
    return out
