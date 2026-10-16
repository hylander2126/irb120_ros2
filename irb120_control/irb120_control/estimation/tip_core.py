#!/usr/bin/env python3
"""Primitive-agnostic tipping estimator shared by press-pull and forward tip.

Both primitives rotate a rigid object about a known pivot edge while the F/T sensor reads
the finger wrench, so the physics is identical:

    tau_pivot(phi) = m g |r_com| sin(theta* - phi)

What differs between them is only HOW THE OBJECT'S TILT phi(t) IS OBTAINED. This module
takes the tilt as an input (a TiltTrack) and does everything else once:

    TipLog      one log on the EE pose grid, from a per-primitive loader
                (estimate_params.tiplog_from_press_pull, fwd_tip_estimation.load_fwd_tip)
    TiltTrack   the object's rotation vector per sample, plus where it is trustworthy
                and (when the source can see it) how far the object has slid
    detect_onset          first contact, rocking onset, proprioceptive slide
    pivot_wrench          applied wrench about the (possibly moving) pivot
    fit_track             methods A, B, D on the tip and return branches

Two families of tilt source exist, for each primitive:

    proprioceptive  accurate and at the control rate, but blind to tip vs slide: it
                    assumes a contact model (no-slip finger for press-pull, a ball
                    sliding on a planar face for the forward tip) and attributes all
                    finger motion to rotation about the pivot.
                        tilt_noslip            press-pull
                        tilt_sliding_face      forward tip
    vision          ~30 Hz, noisier and lower confidence, but it measures the object
                    itself, so it separates tipping from sliding and gives the slide.
                        tilt_vision.track_tilt_vision (both primitives)

Methods. All of them take com_x (horizontal CoM distance from the pivot) as known, fit
the measured pivot torque tau(phi) on the tip and return branches separately, and
average the two branches.
    A  Zero crossing. Fit a straight line to a signal that is zero at the balance angle
       (press-pull: g_x; forward tip: tau itself) and extrapolate it to zero to get theta*.
       Then z_c = com_x / tan(theta*), and m is the least-squares amplitude.
    B  Free fit, absolute tilt. Fit m and z_c together so that
           tau(phi) = m g (com_x cos phi - z_c sin phi)
       matches the data, with phi taken literally from the tilt source. It is only right
       if the tilt is right in absolute terms: a 1 deg error in the tilt zero moves z_c
       by ~10 % and does not show up in the residual.
    D  Onset-anchored, offset-tolerant. At rocking onset the object is still upright and
       just leaving the far support, so tau_on = m g com_x, which gives m directly
       (no tilt needed). With m fixed, fit z_c AND a constant tilt offset phi0:
           tau(phi) = m g (com_x cos(phi - phi0) - z_c sin(phi - phi0))
       So the tilt only has to be right in increments; its zero is fitted, not assumed.
       Robust to a slide before rocking or a biased tilt zero. Relies on detecting the
       onset torque correctly, and the fitted phi0 should come out near 0.
Press-pull reports B as primary (paper numbers); forward tip reports D as primary.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares

from irb120_control.estimation.com_estimation import construct_T, model_bkwd_wrench, model_fwd_wrench

G = 9.81
TILT_DEADBAND = np.deg2rad(1.0)   # samples tilted less than this carry no theta* information
F_CONTACT_N = 0.12                # applied force along the push marking first touch
ONSET_SMOOTH = 25                 # pose samples (0.25 s at 100 Hz) smoothing the onset search
ONSET_WINDOW = 150                # samples after onset searched for the onset torque
ROCK_DROP = 0.5                   # rocking: torque must fall below this fraction of its onset value
SLIDE_MIN_M = 0.003               # finger travel before onset below this is compliance, not slide

BRANCHES = ("tip", "return")


@dataclass
class TipLog:
    """One tipping log on the EE pose grid. Wrench as MEASURED in the sensor frame.

    branch   per-sample 'tip' (object tilting further), 'return' (coming back) or ''.
    contact  per-sample finger-on-object window.
    tilt_dir unit axis about which a positive tip is a positive rotation
             (forward tip pushes the top along +x: +y; press-pull pulls it back: -y).
    push_dir horizontal direction the finger moves while tipping (proprioceptive slide).
    """
    primitive: str
    path: str
    t_epoch: np.ndarray
    f_S: np.ndarray
    tau_S: np.ndarray
    p_S: np.ndarray
    q_S: np.ndarray
    p_ball: np.ndarray
    branch: np.ndarray
    contact: np.ndarray
    pivot: np.ndarray
    tilt_dir: np.ndarray
    push_dir: np.ndarray
    a_signal: str = "tau"             # method A zero-crossing signal: 'tau' or 'gx'
    episode_dir: object = None        # pathlib.Path of the episode, when there is one
    tip_stage: str = ""               # stage folder name ('01_forward_tip', ...)
    extra: dict = field(default_factory=dict)

    @property
    def time(self):
        return self.t_epoch - self.t_epoch[0]

    @property
    def f_push(self):
        """Applied force along the push (the sensor reads the reaction)."""
        R = _rotations(self.q_S)
        f_w = np.einsum("nij,nj->ni", R, self.f_S)
        return -(f_w @ self.push_dir)

    def sensor_T(self):
        return construct_T(self.p_S, quat=self.q_S)


@dataclass
class TiltTrack:
    """Object rotation vector per log sample and where it can be used.

    rotvec  (N,3) object rotation in the base frame relative to its pose before contact
    valid   (N,)  samples this source vouches for
    slide   (N,)  object translation along the push [m], NaN where unknown
    sigma   (N,)  1-sigma tilt uncertainty [rad], NaN where not modelled
    pivot_t (N,3) pivot position per sample (moves with a slide); None = log.pivot
    """
    rotvec: np.ndarray
    valid: np.ndarray
    source: str
    slide: np.ndarray = None
    sigma: np.ndarray = None
    pivot_t: np.ndarray = None
    i_ref: int = 0
    info: dict = field(default_factory=dict)

    def tilt(self, tilt_dir):
        return self.rotvec @ tilt_dir


def _rotations(q):
    from irb120_control.estimation.helper_fns import quat_to_rotvec, rotvec_to_rot
    return rotvec_to_rot(quat_to_rotvec(q))


# ==========================================================================================
#  PROPRIOCEPTIVE TILT
# ==========================================================================================
def tilt_noslip(log: TipLog, pivot) -> TiltTrack:
    """Press-pull: a no-slip finger is a material point of the object, so the ball centre
    rotates rigidly about the pivot and the object rotation is the rotation of the
    pivot->ball vector from its value at first contact."""
    from irb120_control.estimation.com_estimation import rotvec_between
    r_t = log.p_ball - pivot
    i0 = int(np.argmax(log.contact))
    rv = rotvec_between(r_t[i0], r_t)
    rv[~log.contact] = 0.0
    return TiltTrack(rotvec=rv, valid=log.contact.copy(), source="proprio:noslip", i_ref=i0,
                     info={"r0": r_t[i0]})


def tilt_rolling_sphere(log: TipLog, pivot, radius=0.01325) -> TiltTrack:
    """Optional planar press model for a world-fixed spherical finger rolling without slip.

    A ball centre is not a material point of the rotating object. For backward
    tilt phi, sticking contact velocities give b=Ry(-phi)*(r0+radius*phi*ex).
    Thus the observed ball polar-angle increment is
    phi+atan2(z0,x0+radius*phi)-atan2(z0,x0), rather than phi itself.
    This assumes an initially horizontal top and a nonrotating finger. It is
    an explicit alternative to the point approximation, not an angle calibration.
    """
    if log.primitive != 'press_pull' or log.tilt_dir[1] > -.999:
        raise ValueError('rolling-sphere tilt requires backward planar press-pull')
    track = tilt_noslip(log, pivot)
    r0 = track.info['r0']
    if r0[2] <= radius:
        raise ValueError('contact height must exceed the ball radius')
    observed = track.rotvec @ log.tilt_dir
    phi = observed.copy()
    for _ in range(8):
        x = r0[0] + radius * phi
        error = phi + np.arctan2(r0[2], x) - np.arctan2(r0[2], r0[0]) - observed
        derivative = 1. - radius*r0[2]/(x*x+r0[2]*r0[2])
        phi -= error / derivative
    track.rotvec = phi[:, None]*log.tilt_dir
    track.rotvec[~log.contact] = 0.
    track.source = 'proprio:rolling_sphere'
    predicted_radius = np.sqrt((r0[0]+radius*phi)**2+r0[1]**2+r0[2]**2)
    actual_radius = np.linalg.norm(log.p_ball-pivot, axis=1)
    rocking = log.contact & np.isin(log.branch, BRANCHES) & (phi > TILT_DEADBAND)
    track.info['rolling_radial_rms_mm'] = float(np.sqrt(np.mean(
        (actual_radius[rocking]-predicted_radius[rocking])**2)))*1000 if rocking.any() else None
    return track


def tilt_sliding_face(log: TipLog, pivot, i_ref: int, upright_distance=None) -> TiltTrack:
    """Forward tip: a fixed-height ball sliding on a planar face at horizontal distance d0
    (ball centre to pivot) when upright touches the face tilted by phi where

        z_b sin(phi) + d cos(phi) = d0,       phi(i_ref) = 0.

    Samples where no rigid tilt reaches the finger (ratio > 1: the object must have slid)
    are invalid, as are samples before i_ref and, on the way back, samples where the finger
    is behind its onset position (the object is back on the table there)."""
    d = (pivot - log.p_ball) @ log.push_dir
    zb = log.p_ball[:, 2] - pivot[2]
    # When a slide moved the support, the upright face distance comes from
    # the original support/contact geometry, not the moved support at first touch.
    d0 = d[i_ref] if upright_distance is None else float(upright_distance)
    ratio = d0 / np.hypot(d, zb)
    phi = np.arctan2(zb, d) - np.arccos(np.clip(ratio, -1.0, 1.0))
    bad = ratio > 1.0
    phi = np.maximum(np.where(bad, 0.0, phi), 0.0)
    idx = np.arange(len(phi))
    x = log.p_ball @ log.push_dir
    valid = log.contact & (idx >= i_ref) & ~bad & (x >= x[i_ref] - 1e-3)
    rv = phi[:, None] * log.tilt_dir[None, :]
    return TiltTrack(rotvec=rv, valid=valid, source="proprio:sliding_face", i_ref=i_ref,
                     info={"ratio": ratio})


# ==========================================================================================
#  WRENCH / ONSET
# ==========================================================================================
def pivot_wrench(log: TipLog, rotvec, pivot, mask=None, T_B_S=None):
    """Applied wrench [tau, f] on the object in its own frame at the pivot, for `mask`.

    pivot is (3,) or (N,3) (a pivot that moves with a slide)."""
    n = len(log.t_epoch)
    mask = np.ones(n, bool) if mask is None else mask
    piv = np.broadcast_to(np.asarray(pivot, float), (n, 3))
    T_B_S = log.sensor_T() if T_B_S is None else T_B_S
    T_B_O = construct_T(np.ascontiguousarray(piv[mask]), rv=rotvec[mask])
    w_S = np.hstack((log.tau_S, log.f_S))
    return model_bkwd_wrench(w_S[mask], T_B_S[mask], T_B_O)


def detect_onset(log: TipLog, pivot) -> dict:
    """First contact, rocking onset and proprioceptive slide.

    The torque about the pivot is flat while the object slides and falls monotonically once
    it rocks (tau = m g |r| sin(theta* - phi)), so the onset is its (smoothed) maximum on
    the tip branch. A log where it never falls below ROCK_DROP of that value never rocked.
    The slide is finger travel along the push between first contact and onset -- only
    meaningful for a fixed-height push; it cannot tell a slide from finger compliance or
    from the finger sliding on the object, which is what vision is for.
    """
    active = log.branch != ""
    if log.primitive == "press_pull":
        # The normal press maintains contact even when its horizontal reaction
        # opposes the arc direction (notably a CoM close to the support).
        # Controller LULL..RETRACT is the loader's contact window.
        on = np.where(log.contact)[0]
    else:
        on = np.where(active & (log.f_push > F_CONTACT_N))[0]
    out = dict(contact=len(on) > 0, rocking=False,
               contact_source="controller" if log.primitive == "press_pull" else "push_force")
    if not len(on):
        return out
    i0, i_last = int(on[0]), int(on[-1])
    out["i0"], out["i_last"] = i0, i_last
    tip = np.where(log.contact & (log.branch == "tip") & (np.arange(len(active)) >= i0))[0]
    if len(tip) < 2 * ONSET_SMOOTH:
        return out
    tau0 = pivot_wrench(log, np.zeros((len(active), 3)), pivot)[:, :3] @ log.tilt_dir
    sm = uniform_filter1d(tau0, ONSET_SMOOTH)
    i_sm = int(tip[np.argmax(sm[tip])])
    # The moving average lags a step-then-decline peak by ~half its window; refine on the
    # (already low-passed) torque itself near the smoothed peak.
    near = tip[np.abs(tip - i_sm) <= ONSET_SMOOTH]
    i_on = int(near[np.argmax(tau0[near])])
    tail = sm[tip[-ONSET_SMOOTH:]].mean()
    slide = float((log.p_ball[i_on] - log.p_ball[i0]) @ log.push_dir)
    out.update(i_on=i_on, slide_m=slide if log.primitive != "press_pull" and slide >= SLIDE_MIN_M else 0.0,
               rocking=bool(tail < ROCK_DROP * sm[i_on]), tau_end_frac=float(tail / sm[i_on]))
    return out


# ==========================================================================================
#  FITS
# ==========================================================================================
def theta_deg(cx, cz):
    return float(np.degrees(np.arctan2(abs(cx), cz)))


def fit_track(log: TipLog, track: TiltTrack, com_x: float, *, x0=(0.4, 0.1),
              free_com_x=False, onset=None, pivot=None, trim_deg=None):
    """Methods A, B and D on each branch for one tilt source.

    com_x   horizontal CoM distance to the pivot (> 0), known.
    onset   detect_onset() output; method D is skipped without an onset index.
    x0      (mass, z_c) initial guess for the least-squares fits.
    trim_deg  press-pull phase trim: keep trim < tilt < (branch max - trim). Note trim 0
              still drops each branch's single most-tilted sample (the press-pull path has
              always done this); None keeps every sample.

    Returns {"contact_mask", "tip_axis", "w_app", "tau", "branches": {name: fits}}; every
    per-sample array is indexed like the log, w_app/tau are NaN outside contact_mask.
    """
    n = len(log.t_epoch)
    pivot = (track.pivot_t if track.pivot_t is not None else log.pivot) if pivot is None else pivot
    rv = track.rotvec
    tilt = rv @ log.tilt_dir
    in_branch = np.isin(log.branch, BRANCHES)
    contact_mask = in_branch & track.valid & np.isfinite(tilt) & (tilt > TILT_DEADBAND)
    if contact_mask.sum() < 10:
        raise ValueError(f"{track.source}: only {int(contact_mask.sum())} tilted samples")
    # CoM sits on the side the object tips away from: for a +y tilt (top toward +x) it is
    # at -x of the pivot, for a -y tilt at +x.
    assert abs(abs(log.tilt_dir[1]) - 1.0) < 1e-9, "tilt must be about the world y axis"
    sx = -float(np.sign(log.tilt_dir[1]))
    com_vec = lambda cx, cz: np.array([sx * cx, 0.0, cz])

    T_B_S = log.sensor_T()
    w_app = np.full((n, 6), np.nan)
    w_app[contact_mask] = pivot_wrench(log, rv, pivot, contact_mask, T_B_S)
    tip_axis = rv[contact_mask].mean(0)
    tip_axis = tip_axis / np.linalg.norm(tip_axis)
    tau = w_app[:, :3] @ tip_axis

    m_on, tau_on = None, None
    if onset is not None and onset.get("i_on") is not None:
        # Onset torque: upright object about the pivot at onset, smoothed like the onset search.
        full = np.isin(log.branch, BRANCHES) & log.contact
        idx = np.arange(n)
        win = (idx >= onset["i_on"]) & (idx < onset["i_on"] + ONSET_WINDOW) & full & (log.branch == "tip")
        if win.any():
            tau_all = np.zeros(n)
            tau_all[full] = pivot_wrench(log, np.where(np.isfinite(rv), rv, 0.0), pivot, full, T_B_S)[:, :3] @ tip_axis
            tau_on = float(uniform_filter1d(tau_all, ONSET_SMOOTH)[win].max())
            m_on = tau_on / (G * com_x)

    branches = {}
    for name in BRANCHES:
        sel = contact_mask & (log.branch == name)
        if trim_deg is not None and sel.any():
            tr = np.deg2rad(trim_deg)
            sel &= (tilt > tr) & (tilt < tilt[sel].max() - tr)
        if sel.sum() < 10:
            continue
        branches[name] = _fit_branch(log, track, w_app, tau, tip_axis, sel, com_x, com_vec,
                                     x0, free_com_x, m_on)
    if not branches:
        raise ValueError(f"{track.source}: no branch has enough tilted samples")
    return {"contact_mask": contact_mask, "tip_axis": tip_axis, "w_app": w_app, "tau": tau,
            "tau_onset": tau_on, "m_onset": m_on, "branches": branches, "source": track.source}


def _fit_branch(log, track, w_app, tau, tip_axis, sel, com_x, com_vec, x0, free_com_x, m_on):
    rv_ph = track.rotvec[sel]
    tau_meas = tau[sel]
    tilt_deg = np.degrees(rv_ph @ log.tilt_dir)
    pitch_deg = np.degrees(rv_ph[:, 1])        # signed, as the press-pull path always used

    def fwd(m, cz, cx=com_x, off=0.0):
        r = rv_ph if off == 0.0 else rv_ph - off * log.tilt_dir
        return model_fwd_wrench(r, com_vec(cx, cz), m)[:, :3] @ tip_axis

    # Method A: linear zero crossing of a signal that vanishes at theta*.
    if log.a_signal == "gx" and "r0" in track.info:
        # g_x = f_x - (r0_x/r0_z) f_z zeros at theta* for any no-slip contact geometry.
        r0 = track.info["r0"]
        sig = w_app[sel, 3] - (r0[0] / r0[2]) * w_app[sel, 5]
    else:
        sig = tau_meas
    a_coeffs = np.polyfit(pitch_deg, sig, 1)
    thA = -a_coeffs[1] / a_coeffs[0]
    czA = com_x / np.tan(np.deg2rad(abs(thA)))
    mA = least_squares(lambda p: (fwd(p[0], czA) - tau_meas).ravel(), x0=[x0[0]],
                       bounds=([1e-6], [np.inf]), method="trf").x[0]

    # Method B: joint (m, z_c) [, com_x].
    if free_com_x:
        rB = least_squares(lambda p: (fwd(p[0], p[1], p[2]) - tau_meas).ravel(),
                           x0=[x0[0], x0[1], com_x], bounds=([1e-6, 1e-3, -0.2], [np.inf, np.inf, 0.2]),
                           method="trf")
        mB, czB, cxB = rB.x
    else:
        rB = least_squares(lambda p: (fwd(p[0], p[1]) - tau_meas).ravel(), x0=list(x0),
                           bounds=([1e-6, 1e-3], [np.inf, np.inf]), method="trf")
        (mB, czB), cxB = rB.x, com_x

    out = {
        "n": int(sel.sum()),
        "tilt_deg": [float(tilt_deg.min()), float(tilt_deg.max())],
        "A": {"m": float(mA), "zc": float(czA), "theta": float(abs(thA)), "cx": com_x,
              "theta_signed": float(thA), "coeffs": a_coeffs},
        "B": {"m": float(mB), "zc": float(czB), "theta": theta_deg(cxB, czB), "cx": float(cxB),
              "rms": float(np.sqrt(np.mean(rB.fun ** 2)))},
        "sel": sel,
    }
    # Method D: mass pinned by the onset torque; z_c and a constant tilt offset fitted.
    if m_on is not None and np.isfinite(m_on) and m_on > 0:
        rD = least_squares(lambda p: fwd(m_on, p[0], off=p[1]) - tau_meas, x0=[x0[1], 0.0],
                           bounds=([1e-3, -np.radians(15)], [np.inf, np.radians(15)]))
        czD, offD = rD.x
        out["D"] = {"m": float(m_on), "zc": float(czD), "theta": theta_deg(com_x, czD), "cx": com_x,
                    "offset_deg": float(np.degrees(offD)), "rms": float(np.sqrt(np.mean(rD.fun ** 2)))}
    return out


def branch_mean(fit: dict, method: str) -> dict:
    """Push/return average of one method (hysteresis cancels); single branch if only one."""
    fs = [f[method] for f in fit["branches"].values() if method in f]
    if not fs:
        return None
    m = float(np.mean([f["m"] for f in fs]))
    zc = float(np.mean([f["zc"] for f in fs]))
    cx = float(np.mean([f["cx"] for f in fs]))
    return {"m": m, "zc": zc, "cx": cx, "theta": float(np.mean([f["theta"] for f in fs])),
            "mgr": m * float(np.hypot(cx, zc)), "n_branches": len(fs)}


# ==========================================================================================
#  TIP VS SLIDE (needs a source that sees the object: vision)
# ==========================================================================================
def classify_motion(log: TipLog, track: TiltTrack, rate_tilt_dps=0.25, rate_slide_mms=0.5,
                    smooth_s=0.5) -> dict:
    """Label each in-contact sample 'static' / 'slide' / 'tip' / 'tip+slide' from the rates
    of a track that measures slide (vision). Returns labels plus summary numbers."""
    if track.slide is None:
        return {}
    t = log.time
    ok = track.valid & np.isfinite(track.slide)
    if ok.sum() < 20:
        return {}
    dt = float(np.median(np.diff(t)))
    k = max(int(round(smooth_s / dt)), 1)
    tilt = np.interp(t, t[ok], uniform_filter1d((track.rotvec @ log.tilt_dir)[ok], k))
    slide = np.interp(t, t[ok], uniform_filter1d(track.slide[ok], k))
    dtilt = np.degrees(np.gradient(tilt, t))
    dslide = np.gradient(slide, t) * 1e3
    tipping = np.abs(dtilt) > rate_tilt_dps
    sliding = np.abs(dslide) > rate_slide_mms
    lab = np.full(len(t), "", dtype=object)
    m = log.contact & ok
    lab[m] = "static"
    lab[m & sliding & ~tipping] = "slide"
    lab[m & tipping & ~sliding] = "tip"
    lab[m & tipping & sliding] = "tip+slide"
    frac = {k_: float(np.mean(lab[m] == k_)) for k_ in ("static", "slide", "tip", "tip+slide")}
    first_tip = np.where(m & tipping & (tilt > TILT_DEADBAND))[0]
    return {"labels": lab, "fractions": frac,
            "slide_total_mm": float((np.nanmax(slide[m]) - slide[m][0]) * 1e3) if m.any() else np.nan,
            "slide_before_tip_mm": float((slide[first_tip[0]] - slide[m][0]) * 1e3) if len(first_tip) else np.nan,
            "max_tilt_deg": float(np.degrees(np.nanmax(tilt[m]))) if m.any() else np.nan,
            "tipped": bool(len(first_tip) and np.degrees(np.nanmax(tilt[m])) > 2.0)}


def compare_tracks(log: TipLog, a: TiltTrack, b: TiltTrack) -> dict:
    """Agreement of two tilt sources where both are valid (e.g. proprio vs vision)."""
    both = a.valid & b.valid & log.contact
    ta, tb = a.rotvec @ log.tilt_dir, b.rotvec @ log.tilt_dir
    both &= np.isfinite(ta) & np.isfinite(tb)
    if both.sum() < 10:
        return {"n": int(both.sum())}
    d = np.degrees(ta[both] - tb[both])
    return {"n": int(both.sum()), "bias_deg": float(np.mean(d)), "rms_deg": float(np.sqrt(np.mean(d ** 2))),
            "max_abs_deg": float(np.max(np.abs(d))),
            "scale": float(np.polyfit(tb[both], ta[both], 1)[0])}


# ==========================================================================================
#  SECOND TILT SOURCE
# ==========================================================================================
def estimate_with_vision(log: TipLog, proprio: TiltTrack, com_x: float, x0, onset, pivot,
                         camera=None, verbose=True, label="", trim_deg=None) -> dict | None:
    """Re-run the same fits on the vision tilt track and report tip-vs-slide.

    Returns None when the log has no episode camera data. The vision fit uses the pivot
    carried along by the vision-measured slide, so a slide needs no separate correction.
    """
    try:
        from irb120_control.estimation.tilt_vision import track_tilt_vision
        vt = track_tilt_vision(log, camera=camera, verbose=verbose)
    except (ValueError, OSError, ImportError) as exc:
        if verbose:
            print(f"  [{label}] vision tilt unavailable: {exc}")
        return {"available": False, "why": str(exc)[:120]}
    if vt is None:
        return {"available": False, "why": "no episode camera data"}
    out = {"available": True, "camera": vt.info["camera"], "sync": vt.info["sync"],
           "tilt_noise_floor_deg": vt.info["tilt_noise_floor_deg"],
           "agreement": compare_tracks(log, proprio, vt) if proprio is not None else {},
           "motion": {k: v for k, v in classify_motion(log, vt).items() if k != "labels"},
           "_track": vt}
    if not out["motion"].get("tipped", False):
        out["fit_error"] = "vision: object did not tip (no torque-vs-tilt curve to fit)"
        if verbose:
            mo = out["motion"]
            print(f"  [{label}] vision ({out['camera']}): did not tip — slide {mo.get('slide_total_mm', np.nan):.1f} mm, "
                  f"max tilt {mo.get('max_tilt_deg', np.nan):.1f}°")
        return out
    try:
        fit = fit_track(log, vt, com_x, x0=x0, onset=onset, trim_deg=trim_deg)
    except ValueError as exc:
        out["fit_error"] = str(exc)[:120]
        return out
    out["fit"] = fit
    for meth in ("B", "D"):
        out[meth] = branch_mean(fit, meth)
    if verbose:
        mo, ag = out["motion"], out["agreement"]
        print(f"  [{label}] vision ({out['camera']}): slide before tip {mo.get('slide_before_tip_mm', np.nan):.1f} mm, "
              f"total {mo.get('slide_total_mm', np.nan):.1f} mm, max tilt {mo.get('max_tilt_deg', np.nan):.1f}°"
              + (f" | vs proprio: bias {ag['bias_deg']:+.2f}° rms {ag['rms_deg']:.2f}° scale {ag['scale']:.3f}"
                 if "bias_deg" in ag else ""))
        for meth in ("B", "D"):
            if out.get(meth):
                print(f"  [{label}] vision {meth}: m={out[meth]['m']:.4f} kg  z_c={out[meth]['zc']*1e3:.1f} mm  "
                      f"θ*={out[meth]['theta']:.2f}°")
    return out


def print_vision_report(outcomes):
    """One row per log: what the cameras saw and the estimate from the vision tilt alone.
    `outcomes` are dicts with "obj" and "vision" (estimate_with_vision output)."""
    if not any(rec.get("vision") for rec in outcomes):
        return
    print("\n  Vision tilt track: what the object did, and the estimate from vision tilt alone")
    print("  (slide→tip = object slide before it started tipping; Δφ = proprio − vision tilt)")
    print(f"  {'trial':<10}{'cam':>6}{'sync s':>8}{'slide→tip':>11}{'slide tot':>11}{'max tilt':>10}"
          f"{'tip/slide/still %':>19}{'Δφ bias':>9}{'Δφ rms':>8}{'scale':>7}"
          f"{'vis D θ*':>10}{'vis D z_c':>11}{'vis B θ*':>10}")
    for rec in outcomes:
        v = rec.get("vision")
        if not v or not v.get("available"):
            print(f"  {rec['obj']:<10}  no vision{(': ' + v.get('why', '')) if v else ''}")
            continue
        mo, ag = v.get("motion", {}), v.get("agreement", {}) or {}
        fr = mo.get("fractions", {})
        sync = v.get("sync", {})
        d, b = v.get("D") or {}, v.get("B") or {}
        pct = "{:.0f}/{:.0f}/{:.0f}".format(100 * (fr.get("tip", 0) + fr.get("tip+slide", 0)),
                                            100 * fr.get("slide", 0), 100 * fr.get("static", 0))
        print(f"  {rec['obj']:<10}{v['camera']:>6}{sync.get('offset_s', np.nan):+8.2f}"
              f"{mo.get('slide_before_tip_mm', np.nan):11.1f}{mo.get('slide_total_mm', np.nan):11.1f}"
              f"{mo.get('max_tilt_deg', np.nan):10.1f}{pct:>19}"
              f"{ag.get('bias_deg', np.nan):+9.2f}{ag.get('rms_deg', np.nan):8.2f}{ag.get('scale', np.nan):7.3f}"
              f"{d.get('theta', np.nan):10.2f}{d.get('zc', np.nan)*1e3:11.1f}{b.get('theta', np.nan):10.2f}")
