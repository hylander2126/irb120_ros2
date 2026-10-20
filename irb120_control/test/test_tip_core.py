"""Shared tipping estimator (tip_core) and the vision tilt source (tilt_vision)."""
import numpy as np
import pytest

from irb120_control.estimation.tip_core import (
    G, TiltTrack, TipLog, branch_mean, classify_motion, compare_tracks, detect_onset, fit_track,
    tilt_noslip)

FT_ORIGIN = 0.08225


def _press_pull_log(m=0.39, cx=0.028, cz=0.094, hz=100.0):
    """No-slip press-pull with known physics: the ball sits on the object's top, rotates
    with it about the pivot (tipping toward -x, a negative y rotation) and pushes with the
    force that balances gravity about the pivot plus a large squash load."""
    pivot = np.array([0.61, 0.0, 0.0])
    r0 = np.array([0.04, 0.0, 0.16])                 # pivot -> ball at contact
    theta_star = np.arctan2(cx, cz)
    phis = np.r_[np.zeros(100), np.linspace(0, theta_star - np.radians(2), 600),
                 np.linspace(theta_star - np.radians(2), 0, 600), np.zeros(100)]
    branch = np.r_[["" ] * 100, ["tip"] * 600, ["return"] * 600, [""] * 100].astype("<U6")
    n = len(phis)
    c, s = np.cos(-phis), np.sin(-phis)
    R = np.zeros((n, 3, 3))
    R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, s, 1, -s, c
    ball = pivot + np.einsum("nij,j->ni", R, r0)
    com = pivot + np.einsum("nij,j->ni", R, np.array([cx, 0, cz]))
    # Gravity torque about the pivot (y component) that the finger force must cancel.
    tau_g = np.cross(com - pivot, np.array([0, 0, -m * G]))[:, 1]
    rb = ball - pivot
    squash = 5.0
    F = np.zeros((n, 3))
    for i in range(n):
        # A vertical squash plus a horizontal pull chosen to balance the torque about y:
        # tau_y = rz*Fx - rx*Fz.
        fz = -squash
        fx = (-tau_g[i] + rb[i, 0] * fz) / rb[i, 2]
        F[i] = [fx, 0.0, fz]
    contact = np.ones(n, bool)
    p_S = ball - np.array([0.0866512, 0.0, 0.0])
    f_meas = -F
    tau_meas = np.cross(ball - p_S, f_meas)
    log = TipLog(primitive="press_pull", path="synthetic", t_epoch=np.arange(n) / hz,
                 f_S=f_meas, tau_S=tau_meas, p_S=p_S, q_S=np.tile([0, 0, 0, 1.0], (n, 1)),
                 p_ball=ball, branch=branch, contact=contact, pivot=pivot,
                 tilt_dir=np.array([0.0, -1.0, 0.0]), push_dir=np.array([-1.0, 0.0, 0.0]),
                 a_signal="gx")
    return log, dict(m=m, cx=cx, cz=cz, theta=np.degrees(theta_star))


def test_press_pull_core_recovers_parameters_with_every_method():
    log, gt = _press_pull_log()
    track = tilt_noslip(log, log.pivot)
    onset = detect_onset(log, log.pivot)
    fit = fit_track(log, track, gt["cx"], x0=(0.3, 0.12), onset=onset, trim_deg=0.0)
    for meth in ("B", "D"):
        est = branch_mean(fit, meth)
        assert abs(est["m"] - gt["m"]) / gt["m"] < 0.02, meth
        assert abs(est["zc"] - gt["cz"]) < 0.002, meth
        assert abs(est["theta"] - gt["theta"]) < 0.3, meth
    for br in fit["branches"].values():
        assert abs(br["A"]["theta"] - gt["theta"]) < 0.5


def test_tilt_offset_biases_b_but_not_d():
    """D's raison d'etre: a wrong tilt zero moves B, but D absorbs it as the offset."""
    log, gt = _press_pull_log()
    track = tilt_noslip(log, log.pivot)
    track.rotvec[log.contact, 1] -= np.radians(1.0)
    onset = detect_onset(log, log.pivot)
    fit = fit_track(log, track, gt["cx"], x0=(0.3, 0.12), onset=onset)
    assert abs(branch_mean(fit, "B")["zc"] - gt["cz"]) > 0.004
    assert abs(branch_mean(fit, "D")["zc"] - gt["cz"]) < 0.003


def _track(n, tilt_deg, slide_mm, valid=None):
    rv = np.zeros((n, 3))
    rv[:, 1] = np.radians(tilt_deg)
    return TiltTrack(rotvec=rv, valid=np.ones(n, bool) if valid is None else valid, source="vision",
                     slide=np.asarray(slide_mm, float) * 1e-3)


def test_classify_motion_separates_slide_from_tip():
    n, hz = 3000, 100.0
    t = np.arange(n) / hz
    slide = np.clip((t - 2) * 5.0, 0, 30.0)               # 5 mm/s for 6 s, then stops
    tilt = np.clip((t - 10) * 1.0, 0, 12.0)               # then tips at 1 deg/s
    log = TipLog(primitive="fwd_tip", path="", t_epoch=t, f_S=np.zeros((n, 3)), tau_S=np.zeros((n, 3)),
                 p_S=np.zeros((n, 3)), q_S=np.tile([0, 0, 0, 1.0], (n, 1)), p_ball=np.zeros((n, 3)),
                 branch=np.full(n, "tip"), contact=t > 1.0, pivot=np.zeros(3),
                 tilt_dir=np.array([0.0, 1.0, 0.0]), push_dir=np.array([1.0, 0.0, 0.0]))
    mo = classify_motion(log, _track(n, tilt, slide))
    lab = mo["labels"]
    assert (lab[(t > 3) & (t < 7)] == "slide").mean() > 0.95
    assert (lab[(t > 11) & (t < 21)] == "tip").mean() > 0.95
    assert abs(mo["slide_before_tip_mm"] - 30.0) < 1.0
    assert mo["tipped"]

    ag = compare_tracks(log, _track(n, tilt * 1.05, slide), _track(n, tilt, slide))
    assert abs(ag["scale"] - 1.05) < 1e-6


# ------------------------------------------------------------------------------- vision
cv2 = pytest.importorskip("cv2")


def _render_face(cam, pivot_b, phi, s, size=(240, 320)):
    """Render a textured planar face (plane y = 0.03 in the object frame) under (phi, s).
    Returns (image, depth). Exact ray/plane intersection, so it is the true warp."""
    from irb120_control.estimation.tilt_vision import _rigid
    H, W = size
    v, u = np.mgrid[0:H, 0:W]
    rays_c = np.stack([(u - cam.K[0, 2]) / cam.K[0, 0], (v - cam.K[1, 2]) / cam.K[1, 1], np.ones_like(u, float)], -1)
    rays_b = rays_c @ cam.R                          # optical -> base directions
    o_b = -cam.R.T @ cam.t                           # camera centre in base
    # Face plane after the motion: point and normal of the y = 0.03 face.
    p0 = _rigid(np.array([[pivot_b[0], 0.03, pivot_b[2]]]), pivot_b, phi, s)[0]
    n_ = np.array([0.0, 1.0, 0.0])                   # rotation about y keeps the normal
    lam = ((p0 - o_b) @ n_) / (rays_b @ n_)
    X = o_b + lam[..., None] * rays_b
    # Back to object coordinates (undo the motion) for the texture lookup.
    c, sn = np.cos(-phi), np.sin(-phi)
    Ry = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
    Xo = (X - pivot_b - np.array([s, 0, 0])) @ Ry.T + pivot_b
    a, b = (Xo[..., 0] - pivot_b[0]) * 400, (Xo[..., 2] - pivot_b[2]) * 400
    tex = 128 + 50 * np.sin(a) * np.cos(1.3 * b) + 30 * np.sin(0.7 * a + 2.1 * b)
    on = (Xo[..., 0] > pivot_b[0] - 0.10) & (Xo[..., 0] < pivot_b[0]) & (Xo[..., 2] > pivot_b[2]) & \
         (Xo[..., 2] < pivot_b[2] + 0.15)
    img = np.where(on, tex, 30.0).astype(np.float32)
    depth = np.where(on, (X - o_b) @ cam.R[2], 0.0)
    return img, depth, X[on]


def test_vision_aligner_recovers_tilt_and_slide():
    from irb120_control.estimation.tilt_vision import _Aligner, _Camera
    # A side-ish camera like cam2: 0.5 m from the face, looking along -y and down.
    ang = np.radians(30)
    R = np.array([[-1, 0, 0], [0, np.sin(ang), -np.cos(ang)], [0, -np.cos(ang), -np.sin(ang)]])
    pivot_b = np.array([0.67, 0.0, -0.02])
    o = np.array([0.62, 0.45, 0.25])
    cam = _Camera({"K": np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]), "R": R, "t": -R @ o})
    ref, depth, _ = _render_face(cam, pivot_b, 0.0, 0.0)
    V, U = np.nonzero(depth > 0)
    keep = (U % 2 == 0) & (V % 2 == 0)
    U, V = U[keep], V[keep]
    X = cam.backproject(U.astype(float), V.astype(float), depth[V, U])
    al = _Aligner(cam, X, ref, pivot_b, U, V)
    p = np.zeros(2)
    truth = [(np.radians(a), s) for a, s in zip(np.linspace(0, 10, 21), np.r_[np.linspace(0, 0.02, 11), np.full(10, 0.02)])]
    for phi, s in truth:
        img, _, _ = _render_face(cam, pivot_b, phi, s)
        p, rms, sd, cov = al.fit(img, p)
    assert abs(np.degrees(p[0]) - 10.0) < 0.3
    assert abs(p[1] - 0.02) < 0.0015


def test_spherical_contact_tilt_recovers_world_fixed_rolling_motion():
    from scipy.spatial.transform import Rotation
    from irb120_control.estimation.tip_core import tilt_rolling_sphere
    from irb120_control.estimation.episode_estimation import fit_first_moments
    log, gt = _press_pull_log()
    phi = tilt_noslip(log, log.pivot).rotvec @ log.tilt_dir
    radius = .01325
    r0 = np.array([-.03, 0., .196])
    rot = Rotation.from_rotvec(phi[:,None]*log.tilt_dir)
    # Integrate the no-slip velocity constraint for a nonrotating sphere.
    log.p_ball = log.pivot + rot.apply(r0+np.c_[radius*phi, np.zeros(len(phi)), np.zeros(len(phi))])
    contact = log.p_ball - radius*rot.apply(np.tile([0.,0.,1.],(len(phi),1)))
    arm = contact-log.pivot
    tau_y = -gt['m']*G*(gt['cx']*np.cos(phi)-gt['cz']*np.sin(phi))
    force = np.c_[(tau_y+arm[:,0]*-5.)/arm[:,2], np.zeros(len(phi)), np.full(len(phi),-5.)]
    log.p_S = log.p_ball-np.array([.0866512,0.,0.])
    log.f_S = -force
    log.tau_S = np.cross(contact-log.p_S, log.f_S)
    corrected = tilt_rolling_sphere(log, log.pivot)
    np.testing.assert_allclose(corrected.rotvec@log.tilt_dir, phi, atol=1e-12)
    estimate = fit_first_moments(log, corrected, gt['cx'])['mean']
    assert estimate['m'] == pytest.approx(gt['m'], abs=1e-10)
    assert estimate['zc'] == pytest.approx(gt['cz'], abs=1e-10)
    naive = fit_first_moments(log, tilt_noslip(log, log.pivot), gt['cx'])['mean']
    assert abs(naive['zc']-gt['cz']) > .005


@pytest.fixture(autouse=True)
def synthetic_cloud_height(monkeypatch):
    """These dynamics fixtures have no perception files; supply a separate 30 cm envelope."""
    import irb120_control.estimation.tip_core as core
    import irb120_control.estimation.episode_estimation as episodes
    bound = lambda log: (.30, 'synthetic independent cloud envelope')
    monkeypatch.setattr(core, 'cloud_height_bound', bound)
    monkeypatch.setattr(episodes, 'cloud_height_bound', bound)
