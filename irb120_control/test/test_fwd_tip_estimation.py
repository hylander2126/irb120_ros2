import json

import numpy as np
import pytest

from irb120_control.estimation.fwd_tip_estimation import (
    BALL_RADIUS, FT_WRENCH_ORIGIN_OFFSET_X, friction_from_fwd_tip, load_fwd_tip,
    run_fwd_tip_estimation, tilt_from_sliding_contact, tip_slide_margin)
from irb120_control.estimation.tip_core import detect_onset

G = 9.81
EE_TO_SENSOR = FT_WRENCH_ORIGIN_OFFSET_X + 0.0866512


def _synthetic_episode(tmp_path, m=0.38, cx=0.030, cz=0.110, slide=0.010, mu_f=0.2,
                       rock=True, probe=False):
    """A forward tip with known physics: optional pre-slide, rocking to near theta*, return.

    probe  the controller's slide probe 10 mm into the push: back off 2 mm, then re-advance.
           Rocking, the object follows the finger back holding only part of its force
           (pivot stiction; the controller needs just follow_fraction of it); sliding, it stays put, the force vanishes and the push ends ('slide').
    """
    pivot0 = np.array([0.67, 0.0, 0.003])
    zb, d0 = 0.175, 0.120 + BALL_RADIUS
    theta_star = np.arctan2(cx, cz)
    hz = 100.0

    rows = []  # (state, ball_x, phi, pivot_x, sign) ; sign +1 push, -1 return, 0 no force

    def seg(state, n, xs, phis, pxs, sign):
        for x, p, px in zip(np.broadcast_to(xs, n), np.broadcast_to(phis, n), np.broadcast_to(pxs, n)):
            rows.append((state, x, p, px, sign))

    x_contact = pivot0[0] - d0
    seg("SEEK", 200, np.linspace(x_contact - 0.02, x_contact, 200), 0.0, pivot0[0], 0)
    n_sl = int(slide / 0.005 * hz)
    if n_sl:
        seg("TIP", n_sl, x_contact + np.linspace(0, slide, n_sl), 0.0,
            pivot0[0] + np.linspace(0, slide, n_sl), 2)
    pivot1 = pivot0[0] + slide
    phis = np.linspace(0.0, theta_star - np.radians(1.5) if rock else 0.0, 600)
    xs = pivot1 - (d0 - zb * np.sin(phis)) / np.cos(phis)
    if rock and probe:
        j = 200
        k = int(np.where(xs[j] - xs[:j] >= 0.002)[0][-1])
        seg("TIP", j, xs[:j], phis[:j], pivot1, 1)
        seg("PROBE", j - k, xs[j:k:-1], phis[j:k:-1], pivot1, -3)
        seg("TIP", j - k, xs[k:j], phis[k:j], pivot1, 1)     # re-advance over the same ground
        seg("TIP", 600 - j, xs[j:], phis[j:], pivot1, 1)
        seg("RETURN", 600, xs[::-1], phis[::-1], pivot1, -1)
    elif rock:
        seg("TIP", 600, xs, phis, pivot1, 1)
        seg("RETURN", 600, xs[::-1], phis[::-1], pivot1, -1)
    elif probe:
        x1 = x_contact + slide + 0.010
        seg("TIP", 200, np.linspace(x_contact + slide, x1, 200), 0.0,
            pivot1 + np.linspace(0, 0.010, 200), 2)
        seg("PROBE", 40, np.linspace(x1, x1 - 0.002, 40), 0.0, rows[-1][3], 0)
    else:
        seg("TIP", 600, x_contact + slide + np.linspace(0, 0.03, 600), 0.0,
            pivot1 + np.linspace(0, 0.03, 600), 2)
    seg("RETURN", 200, np.linspace(xs[0] if rock else rows[-1][1], x_contact - 0.02, 200), 0.0,
        rows[-1][3], 0)

    n = len(rows)
    t = np.arange(n) / hz
    state = [r[0] for r in rows]
    bx = np.array([r[1] for r in rows])
    phi = np.array([r[2] for r in rows])
    px = np.array([r[3] for r in rows])
    sign = np.array([r[4] for r in rows])
    ball = np.c_[bx, np.zeros(n), np.full(n, pivot0[2] + zb)]

    F = np.zeros((n, 3))
    n_in = np.c_[np.cos(phi), np.zeros(n), -np.sin(phi)]
    t_up = np.c_[np.sin(phi), np.zeros(n), np.cos(phi)]
    contact = ball + BALL_RADIUS * n_in
    for i in range(n):
        if sign[i] == 0:
            continue
        if sign[i] == 2:                     # sliding: pure push below the tipping torque
            F[i] = [0.5, 0.0, 0.0]
            continue
        held = 0.4 if sign[i] == -3 else 1.0
        u = n_in[i] - np.sign(sign[i]) * mu_f * t_up[i]
        r = contact[i] - np.array([px[i], 0.0, pivot0[2]])
        tau_unit = r[2] * u[0] - r[0] * u[2]
        tau_req = m * G * (cx * np.cos(phi[i]) - cz * np.sin(phi[i]))
        F[i] = held * u * tau_req / tau_unit
    p_tool = ball - np.array([EE_TO_SENSOR, 0.0, 0.0])
    p_S = p_tool + np.array([FT_WRENCH_ORIGIN_OFFSET_X, 0.0, 0.0])
    f_meas = -F
    tau_meas = np.cross(contact - p_S, f_meas)

    ep = tmp_path / "episodes" / "20260101_000000" / "01_forward_tip"
    ep.mkdir(parents=True)
    ts = "20260101_000100"
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    np.savez(ep / f"fwd_tip_{ts}.npz",
             pose_time_s=t, x=bx, y=np.zeros(n), z=ball[:, 2],
             qx=quat[:, 0], qy=quat[:, 1], qz=quat[:, 2], qw=quat[:, 3],
             ft_time_s=t, fx=f_meas[:, 0], fy=f_meas[:, 1], fz=f_meas[:, 2],
             tx=tau_meas[:, 0], ty=tau_meas[:, 1], tz=tau_meas[:, 2],
             ft_px=p_tool[:, 0], ft_py=p_tool[:, 1], ft_pz=p_tool[:, 2],
             ft_qx=quat[:, 0], ft_qy=quat[:, 1], ft_qz=quat[:, 2], ft_qw=quat[:, 3],
             commanded_pivot=pivot0)
    meta = {"timestamp": ts, "commanded_pivot": pivot0.tolist(), "direction": [1.0, 0.0, 0.0]}
    if probe:
        meta["reversal_reason"] = "force_drop" if rock else "slide"
        if not rock:
            meta.update(failure_reason="slide", slide_force_n=0.5)
    (ep / f"fwd_tip_{ts}.json").write_text(json.dumps(meta))
    (ep / f"fwd_tip_motion_{ts}.json").write_text(json.dumps(
        [{"time_s": float(ti), "state": s, "progress_m": float(x - bx[0])}
         for ti, s, x in zip(t, state, bx)]))
    gt = {"mass": m, "com": np.array([cx, 0.0, cz]), "theta_deg": float(np.degrees(theta_star))}
    return str(ep / f"fwd_tip_{ts}.npz"), gt


def test_tilt_from_sliding_contact_inverts_the_face_geometry():
    pivot = np.array([0.67, 0.0, 0.0])
    zb, d0 = 0.175, 0.133
    phi = np.radians(np.linspace(0, 15, 50))
    d = (d0 - zb * np.sin(phi)) / np.cos(phi)
    ball = np.c_[pivot[0] - d, np.zeros_like(d), np.full_like(d, zb)]
    est, ratio = tilt_from_sliding_contact(ball, pivot, 0)
    np.testing.assert_allclose(est, phi, atol=1e-9)
    assert np.all(ratio <= 1.0 + 1e-12)


@pytest.mark.parametrize("slide", [0.0, 0.010])
def test_method_d_recovers_mass_and_com_height(tmp_path, slide):
    path, gt = _synthetic_episode(tmp_path, slide=slide)
    r = run_fwd_tip_estimation("synthetic", path, gt=gt, verbose=False)
    assert abs(r["slide_mm"] - slide * 1e3) < 1.0
    assert abs(r["mass_est"] - gt["mass"]) / gt["mass"] < 0.03
    assert abs(r["com_z_est"] - gt["com"][2]) < 0.003
    assert abs(r["theta_est_deg"] - gt["theta_deg"]) < 0.5


def test_pure_slide_is_not_estimated(tmp_path):
    path, gt = _synthetic_episode(tmp_path, slide=0.010, rock=False)
    with pytest.raises(ValueError, match="never rocked"):
        run_fwd_tip_estimation("synthetic", path, gt=gt, verbose=False)


def test_tip_slide_margin_reduces_to_classic_cost_without_finger_friction():
    assert tip_slide_margin(0.5, 0.0, 0.3, 0.1, 0.05) == pytest.approx(0.5 * 0.3 / 0.05)
    # Finger friction down the face costs tipping moment AND loads the table: both terms.
    assert tip_slide_margin(0.5, 0.3, 0.3, 0.1, 0.05) == pytest.approx(
        0.5 * (0.3 - 0.3 * 0.1) / (0.05 * (1 - 0.5 * 0.3)))


def test_probe_is_its_own_branch_and_stays_out_of_the_fit(tmp_path):
    path, gt = _synthetic_episode(tmp_path, slide=0.0, probe=True)
    log = load_fwd_tip(path)
    probe = np.where(log.branch == "probe")[0]
    # The back-off and the re-advance over the same 2 mm, one contiguous block.
    assert len(probe) > 0 and np.all(np.diff(probe) == 1)
    assert np.ptp(log.p_ball[probe, 0]) == pytest.approx(0.002, abs=2e-4)
    assert log.branch[probe[-1] + 1] == "tip"
    r = run_fwd_tip_estimation("synthetic", path, gt=gt, verbose=False)
    # Same estimate as the push without a probe: fitting the probe (the object holds only
    # part of its force backing off) would pull z_c ~4 mm low.
    ref_path, _ = _synthetic_episode(tmp_path / "ref", slide=0.0)
    ref = run_fwd_tip_estimation("synthetic", ref_path, gt=gt, verbose=False)
    assert r["mass_est"] == pytest.approx(ref["mass_est"], rel=0.005)
    assert abs(r["com_z_est"] - ref["com_z_est"]) < 0.0005
    assert abs(r["theta_est_deg"] - ref["theta_est_deg"]) < 0.1


def test_probe_slide_is_skipped_but_kept_as_a_friction_sample(tmp_path):
    m = 0.38
    path, gt = _synthetic_episode(tmp_path, m=m, rock=False, probe=True)
    with pytest.raises(ValueError, match="slide probe"):
        run_fwd_tip_estimation("synthetic", path, gt=gt, verbose=False)
    log = load_fwd_tip(path)
    mu = friction_from_fwd_tip(log, detect_onset(log, log.pivot), m)
    assert mu["mu_k"] == pytest.approx(0.5 / (m * G), rel=0.02)
    assert mu["mu_k_probe"] == pytest.approx(0.5 / (m * G))


def test_face_geometry_also_inverts_variable_height_pivot_arc():
    # The face-contact equation uses measured z at each sample; fixed height is
    # unnecessary. Commanded angle is deliberately not used by the inversion.
    pivot = np.array([0.67, 0.0, 0.003])
    phi = np.radians(np.linspace(0, 15, 100))
    d0, z0 = 0.133, 0.175
    d = d0 * np.cos(phi) - z0 * np.sin(phi)
    z = d0 * np.sin(phi) + z0 * np.cos(phi)
    ball = np.c_[pivot[0] - d, np.zeros_like(d), pivot[2] + z]
    est, _ = tilt_from_sliding_contact(ball, pivot, 0)
    np.testing.assert_allclose(est, phi, atol=1e-9)


def test_arc_log_loader_preserves_probe_exclusion(tmp_path):
    from pathlib import Path
    path, _ = _synthetic_episode(tmp_path, slide=0.0, probe=True)
    old = Path(path)
    ref = load_fwd_tip(path)
    for file in list(old.parent.glob('fwd_tip_*')):
        file.rename(file.with_name(file.name.replace('fwd_tip_', 'fwd_arc_', 1)))
    arc = load_fwd_tip(str(old.with_name(old.name.replace('fwd_tip_', 'fwd_arc_', 1))))
    np.testing.assert_array_equal(arc.branch, ref.branch)
    np.testing.assert_allclose(arc.f_S, ref.f_S)
