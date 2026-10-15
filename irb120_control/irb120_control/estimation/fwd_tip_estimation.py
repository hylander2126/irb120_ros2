#!/usr/bin/env python3
"""Parameter estimation from forward-tip episodes (fixed-height push about the far edge).

The forward tip is the same tipping primitive as press-pull and runs through the same
estimator (tip_core): only the loader and the proprioceptive tilt model are its own.

Proprioceptive tilt. The finger pushes horizontally at a FIXED height and orientation, so
the ball slides along the pushed face. For a planar face at horizontal distance d0 (ball
centre to pivot) when the object is upright, a ball centre at horizontal distance d and
height z_b above the pivot touches the face tilted by phi where

    z_b sin(phi) + d cos(phi) = d0                 (tip_core.tilt_sliding_face)

Slip. The object may slide on the table before it starts to rock, and finger travel alone
cannot tell that apart from tipping. Rocking onset is where the pivot torque peaks (it
falls monotonically once the object rocks). The slide up to there comes from the vision
track when the episode has camera data (it measures the object), else from finger travel
between first contact and onset; the pivot is shifted by it. The before/after perception
clouds are reported as a further check.

Vision tilt. tilt_vision fits the object's (tilt, slide) to every video frame; the same
fits are then run on that track as a second, independent, lower-accuracy estimate, and it
labels each sample static / slide / tip / tip+slide.

Methods (com_x known; see tip_core): D primary (mass from onset torque, tilt-zero free),
A and B as cross-checks, each on the push (tip) and return branches.

    python3 -m irb120_control.estimation.estimate_params --primitive fwd_tip --logs-root DIR [--vision]
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

from irb120_control.estimation.com_estimation import model_fwd_wrench
from irb120_control.estimation.helper_fns import quat_to_rotvec, rotvec_to_rot
from irb120_control.estimation.tip_core import (
    F_CONTACT_N, G, TipLog, branch_mean, classify_motion, compare_tracks, detect_onset,
    estimate_with_vision, fit_track, print_vision_report, tilt_sliding_face)

FT_WRENCH_ORIGIN_OFFSET_X = 0.08225   # same NetFT torque origin as estimate_params
BALL_RADIUS = 0.01325                 # util/push_geometry.BALL_RADIUS
PRIMARY_METHOD = "D"

_LPF_CUTOFF_HZ = 6.0

# Ground truth for the forward-tip suite. com_obj is (x, z) in the object's own frame:
# origin at the base's near-bottom edge, +x along the push, +z up. The far base edge at
# x = base_w is the pivot, so the CoM sits base_w - x behind it. Mass weighed 2026-10-15.
FWD_TIP_GROUND_TRUTH = {
    "tipT_a": {"mass": 0.403, "com_obj": (0.030, 0.08467), "base_w": 0.060},
    "tipT_b": {"mass": 0.403, "com_obj": (0.030, 0.1016), "base_w": 0.060},
    "tipT_c": {"mass": 0.403, "com_obj": (0.030, 0.1242), "base_w": 0.060},
    "tipT_d": {"mass": 0.403, "com_obj": (0.0074, 0.1242), "base_w": 0.060},
    "tipT_e": {"mass": 0.403, "com_obj": (0.0526, 0.1242), "base_w": 0.060},
}
for _gt in FWD_TIP_GROUND_TRUTH.values():
    _x, _z = _gt["com_obj"]
    # Same convention as estimate_params.GROUND_TRUTH: com[0] = horizontal distance to pivot.
    _gt["com"] = np.array([_gt["base_w"] - _x, 0.0, _z])
    _gt["theta_deg"] = float(np.degrees(np.arctan2(_gt["com"][0], _gt["com"][2])))

FWD_TIP_OBJECTS = tuple(FWD_TIP_GROUND_TRUTH)


# ==========================================================================================
#  LOADING
# ==========================================================================================
def find_fwd_tip_logs(obj: str, root: str) -> list:
    """Forward-tip logs for `obj`, oldest first, under runtime_logs/ or a copied suite dir."""
    pats = [os.path.join(root, "runtime_logs", obj, "episodes", "*", "*_forward_tip", "fwd_tip_2*.npz"),
            os.path.join(root, obj, "episodes", "*", "*_forward_tip", "fwd_tip_2*.npz")]
    return sorted({f for p in pats for f in glob.glob(p)})


def _episode_dir(path: str) -> Path:
    return Path(path).parent.parent


def _lowpass(t, x):
    """4th-order 6 Hz zero-phase low pass at the stream's own rate (F/T is ~470 Hz)."""
    from scipy.signal import butter, filtfilt
    if len(t) <= 20:
        return x
    fs = 1.0 / float(np.median(np.diff(t)))
    b, a = butter(4, _LPF_CUTOFF_HZ, fs=fs, btype="low")
    return filtfilt(b, a, x, axis=0)


def load_fwd_tip(path: str) -> TipLog:
    """One forward-tip log as a tip_core.TipLog on the EE pose grid, wrench in sensor axes."""
    d = np.load(path)
    if len(d["pose_time_s"]) == 0 or len(d["ft_time_s"]) == 0:
        raise ValueError("Empty log (aborted run).")
    stem = Path(path)
    meta = json.loads(stem.with_suffix(".json").read_text())
    ts = meta.get("timestamp") or stem.stem.split("fwd_tip_")[-1]
    motion_path = stem.parent / f"fwd_tip_motion_{ts}.json"
    if not motion_path.exists():
        raise ValueError(f"No motion log {motion_path.name} (phase labels needed).")
    motion = json.loads(motion_path.read_text())

    time = d["pose_time_s"]
    t_ft = d["ft_time_s"]

    def interp(arr):
        return np.column_stack([np.interp(time, t_ft, arr[:, i]) for i in range(arr.shape[1])])

    f_w = interp(_lowpass(t_ft, np.c_[d["fx"], d["fy"], d["fz"]]))
    tau_w = interp(_lowpass(t_ft, np.c_[d["tx"], d["ty"], d["tz"]]))
    p_tool = interp(np.c_[d["ft_px"], d["ft_py"], d["ft_pz"]])
    q_tool = interp(np.c_[d["ft_qx"], d["ft_qy"], d["ft_qz"], d["ft_qw"]])
    q_tool /= np.linalg.norm(q_tool, axis=1, keepdims=True)
    R_tool = rotvec_to_rot(quat_to_rotvec(q_tool))
    # The adjoint wants the wrench in the sensor frame. Older fwd_tip logs (metadata
    # 'pose_and_wrench_frame') hold it rotated into world axes, so rotate those back; newer
    # ones ('wrench_axes_frame') log it as received, in sensor axes.
    if meta.get("pose_and_wrench_frame"):
        f_S = np.einsum("nji,nj->ni", R_tool, f_w)
        tau_S = np.einsum("nji,nj->ni", R_tool, tau_w)
    else:
        f_S, tau_S = f_w, tau_w
    p_S = p_tool + np.einsum("nij,j->ni", R_tool, np.array([FT_WRENCH_ORIGIN_OFFSET_X, 0.0, 0.0]))

    mt = np.array([r["time_s"] for r in motion])
    ms = np.array([r["state"] for r in motion])
    state = ms[np.clip(np.searchsorted(mt, time, side="right") - 1, 0, len(ms) - 1)]
    branch = np.full(len(time), "", dtype="<U6")
    branch[state == "SEEK"] = "seek"
    branch[state == "TIP"] = "tip"
    branch[state == "RETURN"] = "return"

    direction = np.asarray(meta.get("direction", [1.0, 0.0, 0.0]), float)
    if abs(direction[0]) < 0.999:
        raise ValueError("Push direction is not world +x; the planar tilt model assumes it is.")
    pivot = np.asarray(meta.get("commanded_pivot", d["commanded_pivot"]), float)
    # Contact window: first to last sample with the finger pushing on the object.
    on = np.where((branch != "") & (-(f_w @ direction) > F_CONTACT_N))[0]
    contact = np.zeros(len(time), bool)
    if len(on):
        contact[on[0]:on[-1] + 1] = True
    stage = Path(path).parent
    return TipLog(primitive="fwd_tip", path=path, t_epoch=time, f_S=f_S, tau_S=tau_S, p_S=p_S,
                  q_S=q_tool, p_ball=np.c_[d["x"], d["y"], d["z"]], branch=branch, contact=contact,
                  pivot=pivot, tilt_dir=np.array([0.0, 1.0, 0.0]), push_dir=direction / np.linalg.norm(direction),
                  a_signal="tau", episode_dir=stage.parent, tip_stage=stage.name,
                  extra={"meta": meta, "f_w": f_w})


def tilt_from_sliding_contact(p_ball, pivot, i_ref):
    """Pitch phi (rad, >= 0) of a planar face pushed along +x by a fixed-height ball;
    phi(i_ref) = 0. ratio = d0 / |r| > 1 means no rigid tilt reaches the finger (slid)."""
    d = pivot[0] - p_ball[:, 0]
    zb = p_ball[:, 2] - pivot[2]
    ratio = d[i_ref] / np.hypot(d, zb)
    phi = np.arctan2(zb, d) - np.arccos(np.clip(ratio, -1.0, 1.0))
    phi[ratio > 1.0] = np.nan
    return np.maximum(phi, 0.0), ratio


def cloud_slide_m(path: str) -> float:
    """Object translation along x between the episode's before/after perception clouds."""
    ep = _episode_dir(path)
    stages = sorted(p for p in ep.iterdir() if p.is_dir() and p.name.endswith("_perceive"))
    clouds = [s / "object_cloud.npz" for s in stages if (s / "object_cloud.npz").exists()]
    if len(clouds) < 2:
        return float("nan")
    # The first two perceive stages bracket the forward tip.
    # Centre of the base slice's x extent: a median is pulled around by which side of the
    # base happens to be better sampled, and the lowest ~1 cm is mixed with table returns.
    centre = []
    for c in clouds[:2]:
        X = np.load(c)["xyz"]
        lo = np.percentile(X[:, 2], 1)
        x = X[(X[:, 2] > lo + 0.01) & (X[:, 2] < lo + 0.06), 0]
        if len(x) < 20:
            return float("nan")
        centre.append(0.5 * (np.percentile(x, 5) + np.percentile(x, 95)))
    return float(centre[1] - centre[0])


# ==========================================================================================
#  FRICTION / TIP-SLIDE PREDICTION
# ==========================================================================================
def friction_from_fwd_tip(log: TipLog, onset, mass):
    """Table friction seen during the forward tip.

    Slid: mu_s from the breakaway peak (first ~3 mm of contact), mu_k from the slide plateau.
    Did not slide: the largest tangential/normal ratio reached is a LOWER BOUND on mu_s.
    The finger's vertical force adds to (or relieves) the table normal load.
    """
    if not onset.get("contact") or mass is None:
        return {}
    i0, x = onset["i0"], log.p_ball @ log.push_dir
    tip = log.contact & (log.branch == "tip")
    fx = log.f_push
    normal = mass * G + log.extra["f_w"][:, 2]      # sensor +z reaction = object pushed down
    out = {}
    slid = onset.get("slide_m", 0.0) > 0 or not onset.get("rocking", False)
    if slid:
        brk = tip & (x - x[i0] < 0.003)
        k = np.where(brk)[0][np.argmax(fx[brk])] if brk.any() else onset.get("i_on", i0)
        out["mu_s"] = float(fx[k] / normal[k])
        end = onset.get("i_on") if onset.get("rocking") else np.where(tip)[0][-1]
        plateau = tip & (x - x[i0] > 0.005) & (np.arange(len(x)) < end - 25)
        if plateau.sum() > 20:
            out["mu_k"] = float(np.median(fx[plateau] / normal[plateau]))
    else:
        out["mu_s_lower_bound"] = float(np.max(fx[tip] / normal[tip]))
    return out


def friction_from_push(push_path, mass):
    """Static (breakaway peak) and kinetic (moving plateau) table friction from a planar push."""
    d = np.load(push_path)
    t = d["ft_time_s"]
    f = np.hypot(_lowpass(t, d["fx"]), _lowpass(t, d["fy"]))
    fz = _lowpass(t, d["fz"])
    x = np.interp(t, d["pose_time_s"], d["x"])
    v = np.gradient(_lowpass(t, x), t)
    loaded = f > 0.3 * np.percentile(f, 99)
    if not loaded.any():
        return {}
    i0 = int(np.argmax(loaded))
    moving = loaded & (v > 0.5 * np.percentile(v, 95))
    normal = mass * G + fz
    brk = loaded & (x - x[i0] < 0.003)
    return {"mu_s": float(np.max(f[brk] / normal[brk])),
            "mu_k": float(np.median(f[moving] / normal[moving]))}


def tip_slide_margin(mu_t, mu_f, h, L, d):
    """Generalised CoST for a horizontal push at height h, contact L behind the pivot.

    Finger force on the object at onset: normal N along the push and finger friction
    mu_f*N DOWN the face (the face rises past the finger as the object tips).
        tip   needs  N (h - mu_f L) = m g d
        slide needs  N = mu_t (m g + mu_f N)
    CoST = N_slide / N_tip = mu_t (h - mu_f L) / (d (1 - mu_t mu_f)); > 1 means it tips.
    mu_f = 0 recovers mu_t h / d. A contact on a rounded top edge acts like a larger mu_f:
    use the measured |F_z/F_x| there.
    """
    if d <= 0:
        return np.inf
    denom = d * (1.0 - mu_t * mu_f)
    return float(mu_t * (h - mu_f * L) / denom) if denom > 0 else np.inf


def contact_mechanics(path, obj, mass, mu_t, gt=None):
    """Onset quantities for the tip/slide check, for any log that made contact.

    The onset is rocking onset for a log that rocked and the breakaway peak for one that
    only slid; N_meas is the push force there. Returns None without contact.
    """
    gt = gt or FWD_TIP_GROUND_TRUTH[obj]
    log = load_fwd_tip(path)
    onset = detect_onset(log, log.pivot)
    if not onset.get("contact"):
        return None
    x, i0 = log.p_ball @ log.push_dir, onset["i0"]
    fpush = log.f_push
    if onset.get("rocking"):
        k, outcome = onset["i_on"], "tipped" + (f" after {onset['slide_m']*1e3:.0f} mm slide"
                                                if onset["slide_m"] else "")
        pivot = log.pivot + onset["slide_m"] * log.push_dir
    else:
        brk = log.contact & (log.branch == "tip") & (x - x[i0] < 0.003)
        k = int(np.where(brk)[0][np.argmax(fpush[brk])]) if brk.any() else i0
        outcome, pivot = "slid", log.pivot
    win = slice(max(k - 5, 0), k + 6)   # ±50 ms: a breakaway peak is sharp
    fx = float(np.mean(fpush[win]))
    mu_f = float(np.mean(log.extra["f_w"][win, 2]) / fx)    # sensor +z = object pushed down
    h = float(log.p_ball[k, 2] - pivot[2])
    L = float((pivot - log.p_ball[k]) @ log.push_dir - BALL_RADIUS)
    d = float(gt["com"][0])
    mg = mass * G
    n_tip = mg * d / (h - mu_f * L) if h > mu_f * L else np.inf
    n_slide = mu_t * mg / (1.0 - mu_t * mu_f)
    return dict(d=d, h=h, L=L, mu_f=mu_f, N_meas=fx, N_tip=n_tip, N_slide=n_slide,
                cost0=tip_slide_margin(mu_t, 0.0, h, L, d), cost=tip_slide_margin(mu_t, mu_f, h, L, d),
                outcome=outcome)


# ==========================================================================================
#  ESTIMATION
# ==========================================================================================
def _phase_estimates(fit):
    br = fit["branches"]
    pe = {}
    for name, key in (("tip", "arc"), ("return", "unarc")):
        for meth in ("A", "B"):
            if name in br:
                pe[f"{key}_{meth}"] = {k: br[name][meth][k] for k in ("m", "zc", "theta")}
    return pe


def run_fwd_tip_estimation(obj, path, gt=None, verbose=True, pivot_offset=None,
                           ft_bias_n=0.0, com_x_offset=0.0, slide_override_m=None,
                           vision=False, camera=None):
    """Estimate (m, z_c, theta*) from ONE forward-tip log. Raises ValueError if it never rocked.

    vision  also run the camera tilt track: it supplies the slide before rocking (in place of
            finger travel) and a second, independent estimate under r["vision"].
    The keyword perturbations exist for fwd_tip_uncertainty_budget().
    """
    gt = gt or FWD_TIP_GROUND_TRUTH[obj]
    log = load_fwd_tip(path)
    if ft_bias_n:
        # A stale tare is a constant force offset in the sensor frame.
        log.f_S = log.f_S + np.array([ft_bias_n, 0.0, 0.0])
    pivot0 = log.pivot + (np.zeros(3) if pivot_offset is None else np.asarray(pivot_offset, float))
    log.pivot = pivot0
    onset = detect_onset(log, pivot0)
    name = os.path.basename(path)
    cx = float(gt["com"][0] + com_x_offset)

    vision_res, vt = None, None
    if vision and onset.get("contact"):
        vision_res = estimate_with_vision(log, None, cx, (0.4, 0.1), onset, pivot0, camera=camera,
                                          verbose=verbose, label=obj)
        vt = vision_res.pop("_track", None)

    if not onset["contact"]:
        raise ValueError("no contact above threshold (object fell or was missed)")
    if not onset["rocking"]:
        raise ValueError(f"never rocked: torque stayed at {onset.get('tau_end_frac', 1)*100:.0f}% "
                         f"of its peak (pure slide)")

    i_on = onset["i_on"]
    slide_source, slide = "finger", onset["slide_m"]
    if vt is not None and np.isfinite(vt.slide[i_on]):
        pre = vt.valid & (np.arange(len(vt.valid)) <= onset["i0"])
        if pre.any():
            # Vision measures the object, so it is the authority on how far it slid.
            slide = max(float(vt.slide[i_on] - vt.slide[np.where(pre)[0][-1]]), 0.0)
            slide_source = "vision"
    if slide_override_m is not None:
        slide = slide_override_m
    pivot = pivot0 + slide * log.push_dir           # the pivot travelled with the slide
    track = tilt_sliding_face(log, pivot, i_on)
    cz_gt = float(gt["com"][2])
    fit = fit_track(log, track, cx, x0=(0.4, 0.1), onset=onset, pivot=pivot)

    curves = {"phi": track.rotvec @ log.tilt_dir, "tau": fit["tau"],
              "sel": {ph: f["sel"] for ph, f in fit["branches"].items()}, "cx": cx}
    if vision_res and vt is not None:
        vision_res["agreement"] = compare_tracks(log, track, vt)
        vfit = vision_res.pop("fit", None)
        if vfit is not None:
            curves["vision"] = {"tilt": vt.rotvec @ log.tilt_dir, "tau": vfit["tau"],
                                "mask": vfit["contact_mask"]}

    mean = branch_mean(fit, PRIMARY_METHOD)
    tip_sel = fit["branches"]["tip"]["sel"]
    mu_f = float(np.median(np.abs(log.extra["f_w"][tip_sel, 2] / log.f_push[tip_sel])))
    h = float(np.median(log.p_ball[:, 2]) - pivot[2])
    L = float((pivot - log.p_ball[i_on]) @ log.push_dir - BALL_RADIUS)
    cloud = cloud_slide_m(path)

    if verbose:
        print(f"  [{obj}] {name}: slide {slide*1e3:.1f} mm ({slide_source}; finger travel "
              f"{onset['slide_m']*1e3:.1f}, cloud {cloud*1e3:.1f}), tau_on {fit['tau_onset']*1e3:.1f} mN·m "
              f"-> m {fit['m_onset']:.3f} kg")
        for ph, f in fit["branches"].items():
            print(f"    {ph:6s} n={f['n']:4d} tilt {f['tilt_deg'][0]:.1f}-{f['tilt_deg'][1]:.1f}° | "
                  f"D z_c={f['D']['zc']*1e3:6.1f} mm θ*={f['D']['theta']:5.2f}° off={f['D']['offset_deg']:+.2f}° | "
                  f"B m={f['B']['m']:.3f} z_c={f['B']['zc']*1e3:6.1f} | A θ*={f['A']['theta']:5.2f}°")

    mass_gt = gt.get("mass")
    return {
        "obj": obj,
        "trial": name,
        "path": path,
        "method": PRIMARY_METHOD,
        "mass_est": mean["m"],
        "mass_gt": mass_gt,
        "com_z_est": mean["zc"],
        "com_z_gt": cz_gt,
        "theta_est_deg": mean["theta"],
        "theta_gt_deg": gt["theta_deg"],
        "mgr_est": mean["mgr"],
        "mgr_gt": None if mass_gt is None else mass_gt * np.hypot(cx, cz_gt),
        "com_x_est": cx,
        "com_x_gt": float(gt["com"][0]),
        "mu_est": None,
        "mu_gt": None,
        "pivot_x": float(pivot[0]),
        "pivot_xyz": pivot.tolist(),
        "pivot_source": f"perceived+slide({slide_source})",
        "commanded_pivot_xyz": log.pivot.tolist(),
        "pivot_fit_resid_mm": np.nan,
        "noslip_dev_mm": np.nan,
        "slide_mm": slide * 1e3,
        "slide_source": slide_source,
        "slide_finger_mm": onset["slide_m"] * 1e3,
        "cloud_slide_mm": cloud * 1e3,
        "tau_onset": fit["tau_onset"],
        "push_height": h,
        "contact_L": L,
        "mu_finger": mu_f,
        "method_B": branch_mean(fit, "B"),
        "vision": vision_res,
        "phase_estimates": _phase_estimates(fit),
        "branch_fits": {ph: {k: v for k, v in f.items() if k != "sel"} for ph, f in fit["branches"].items()},
        "_curves": curves,
    }


# ==========================================================================================
#  DRIVER
# ==========================================================================================
def run_fwd_tip_object(obj, root, verbose_trials=False, vision=False):
    """All forward-tip logs for `obj`: per-log estimates and per-log outcome records."""
    files = find_fwd_tip_logs(obj, root)
    print(f"\n{'='*60}\n  OBJECT: {obj}   ({len(files)} forward-tip log(s))\n{'='*60}")
    results, outcomes = [], []
    for path in files:
        name = os.path.basename(path)
        rec = {"obj": obj, "trial": name, "path": path}
        try:
            r = run_fwd_tip_estimation(obj, path, verbose=verbose_trials or vision, vision=vision)
        except (KeyError, ValueError, IndexError) as exc:
            rec["skipped"] = str(exc).split("\n")[0][:90]
            print(f"  SKIPPED  {name:36s} {rec['skipped']}")
            if vision:
                rec["vision"] = _vision_outcome(path)
            outcomes.append(rec)
            continue
        results.append(r)
        rec.update(rocked=True, slide_mm=r["slide_mm"], vision=r["vision"])
        outcomes.append(rec)
        print(f"  trial {len(results):2d}  {name:36s} m={r['mass_est']:.4f} kg  "
              f"z_c={r['com_z_est']*1e3:6.1f} mm  θ*={r['theta_est_deg']:5.2f}° (GT {r['theta_gt_deg']:.2f}°)  "
              f"slide {r['slide_mm']:.1f} mm")
    return results, outcomes


def _vision_outcome(path):
    """What the cameras saw for a log the estimator skipped (slid only, fell, ...)."""
    try:
        log = load_fwd_tip(path)
        from irb120_control.estimation.tilt_vision import track_tilt_vision
        if not log.contact.any():
            # No finger contact window: classify over the whole motion instead.
            log.contact = log.branch != ""
        vt = track_tilt_vision(log, verbose=False)
        if vt is None:
            return {"available": False, "why": "no episode camera data"}
        mo = classify_motion(log, vt)
        return {"available": True, "camera": vt.info["camera"], "sync": vt.info["sync"],
                "motion": {k: v for k, v in mo.items() if k != "labels"}}
    except (ValueError, OSError) as exc:
        return {"available": False, "why": str(exc)[:120]}


def add_friction(all_results, all_outcomes, mass):
    """Static/kinetic friction per log, from the forward tip itself and the episode's planar push."""
    rows = []
    for rec in all_outcomes:
        log = load_fwd_tip(rec["path"])
        onset = detect_onset(log, log.pivot)
        row = {"obj": rec["obj"], "trial": rec["trial"], "source": "fwd_tip"}
        row.update(friction_from_fwd_tip(log, onset, mass))
        rows.append(row)
        pushes = sorted(glob.glob(str(_episode_dir(rec["path"]) / "*_push" / "push_ft_pose_2*.npz")))
        for p in pushes:
            rows.append({"obj": rec["obj"], "trial": os.path.basename(p), "source": "planar_push",
                         **friction_from_push(p, mass)})
    for r in all_results:
        mus = [x["mu_k"] for x in rows if x["obj"] == r["obj"] and x["source"] == "planar_push" and "mu_k" in x]
        r["mu_est"] = float(np.mean(mus)) if mus else None
    return rows


def print_fwd_tip_report(all_results, outcomes, friction_rows, mass):
    print("\n" + "=" * 100)
    print(f"  FORWARD-TIP SUMMARY   (method D primary; batch mass {mass:.3f} kg used for friction)")
    print("=" * 100)
    print(f"  {'trial':<10}{'slide':>8}{'from':>8}{'cloud':>8}{'m':>8}{'z_c':>8}{'z_c GT':>8}{'θ*':>8}{'θ* GT':>8}"
          f"{'off TIP/RET':>14}{'B z_c':>8}{'A θ*':>8}")
    for r in all_results:
        bf = r["branch_fits"]
        a_th = np.mean([f["A"]["theta"] for f in bf.values()])
        print(f"  {r['obj']:<10}{r['slide_mm']:8.1f}{r['slide_source']:>8}{r['cloud_slide_mm']:8.1f}{r['mass_est']:8.3f}"
              f"{r['com_z_est']*1e3:8.1f}{r['com_z_gt']*1e3:8.1f}{r['theta_est_deg']:8.2f}{r['theta_gt_deg']:8.2f}"
              f"{bf['tip']['D']['offset_deg']:+7.2f}/{bf['return']['D']['offset_deg']:+.2f}"
              f"{r['method_B']['zc']*1e3:8.1f}{a_th:8.2f}")
    for rec in outcomes:
        if "skipped" in rec:
            print(f"  {rec['obj']:<10} not estimated: {rec['skipped']}")
    print_vision_report(outcomes)

    print("\n  Friction (table).  mu_s = breakaway, mu_k = sliding plateau, '>=' = lower bound (did not slide)")
    for x in friction_rows:
        parts = [f"mu_s={x['mu_s']:.3f}" if "mu_s" in x else "",
                 f"mu_k={x['mu_k']:.3f}" if "mu_k" in x else "",
                 f"mu_s>={x['mu_s_lower_bound']:.3f}" if "mu_s_lower_bound" in x else ""]
        print(f"  {x['obj']:<10}{x['source']:<13}{'  '.join(p for p in parts if p)}")

    mus_s = [x["mu_s"] for x in friction_rows if "mu_s" in x]
    mu_s = float(np.median(mus_s)) if mus_s else np.nan
    print(f"\n  Tip/slide check at onset (mu_t = median breakaway mu_s = {mu_s:.3f}, m = {mass:.3f} kg)")
    print("  mu_f = |F_z/F_x| at onset: finger friction, or the contact normal's slope on an edge.")
    print("  N_tip = m g d / (h - mu_f L),  N_slide = mu_t m g / (1 - mu_t mu_f),  CoST = N_slide / N_tip")
    print(f"  {'trial':<10}{'d mm':>7}{'h mm':>7}{'L mm':>7}{'mu_f':>7}{'N_tip':>8}{'N_slide':>9}"
          f"{'N meas':>8}{'CoST0':>7}{'CoST':>7}  outcome")
    for rec in outcomes:
        cm = contact_mechanics(rec["path"], rec["obj"], mass, mu_s)
        if cm is None:
            print(f"  {rec['obj']:<10} no contact")
            continue
        print(f"  {rec['obj']:<10}{cm['d']*1e3:7.1f}{cm['h']*1e3:7.0f}{cm['L']*1e3:7.0f}{cm['mu_f']:7.2f}"
              f"{cm['N_tip']:8.2f}{cm['N_slide']:9.2f}{cm['N_meas']:8.2f}{cm['cost0']:7.2f}{cm['cost']:7.2f}"
              f"  {cm['outcome']}")
    print("=" * 100)


# ==========================================================================================
#  UNCERTAINTY BUDGET
# ==========================================================================================
FWD_TIP_UNCERTAINTY_SOURCES = (
    # (name, kwarg, 1-sigma, unit, provenance)
    ("pivot x (edge localisation)", "pivot_dx", 0.002, "m",
     "nominal; RGB/SAM pivot vs cloud stem edge can disagree by several mm"),
    ("pivot z (table plane / edge fillet)", "pivot_dz", 0.002, "m",
     "table-fit p95 residual ~2.2 mm in the perception review"),
    ("F/T zero drift after tare", "ft_bias", 0.010, "N",
     "same 1-sigma as the press-pull budget"),
    ("com_x (base width / GT x)", "com_dx", 0.002, "m",
     "base width assumed 60 mm; the cloud reads 62-68 mm incl. edge noise"),
    ("slide length", "slide_d", 0.003, "m",
     "finger-travel, vision and before/after cloud slides differ by 1-4 mm"),
)


def fwd_tip_uncertainty_budget(result):
    """Central-difference sensitivity of (m, z_c, theta*) to each input, combined in quadrature.
    The slide is held at the value the result used, so the budget does not re-run vision."""
    obj, path = result["obj"], result["path"]

    def est(pivot_dx=0.0, pivot_dz=0.0, ft_bias=0.0, com_dx=0.0, slide_d=0.0):
        r = run_fwd_tip_estimation(obj, path, verbose=False,
                                   pivot_offset=np.array([pivot_dx, 0.0, pivot_dz]),
                                   ft_bias_n=ft_bias, com_x_offset=com_dx,
                                   slide_override_m=max(result["slide_mm"] * 1e-3 + slide_d, 0.0))
        return np.array([r["mass_est"], r["com_z_est"], r["theta_est_deg"]])

    rows = []
    for name, kw, sigma, unit, prov in FWD_TIP_UNCERTAINTY_SOURCES:
        try:
            dp, dm = est(**{kw: +sigma}), est(**{kw: -sigma})
        except ValueError:
            continue
        rows.append({"source": name, "sigma": sigma, "unit": unit, "provenance": prov,
                     "d": 0.5 * np.abs(dp - dm)})
    tot = np.sqrt(sum(r["d"] ** 2 for r in rows)) if rows else np.full(3, np.nan)
    return {"obj": obj, "trial": result["trial"], "rows": rows, "total": tot}


def print_fwd_tip_uncertainty(all_results):
    print("\n" + "=" * 100)
    print("  FORWARD-TIP UNCERTAINTY BUDGET  (each input ±1σ, full estimator re-run)")
    print("=" * 100)
    for r in all_results:
        b = fwd_tip_uncertainty_budget(r)
        print(f"\n  --- {r['obj']} ({b['trial']}) ---")
        print(f"  {'source':<38}{'1σ':>8}{'σ(m) kg':>10}{'σ(z_c) mm':>11}{'σ(θ*) °':>10}")
        for row in b["rows"]:
            print(f"  {row['source']:<38}{row['sigma']:8.3f}{row['d'][0]:10.4f}{row['d'][1]*1e3:11.2f}{row['d'][2]:10.2f}")
        t = b["total"]
        dz = abs(r["com_z_est"] - r["com_z_gt"])
        dth = abs(r["theta_est_deg"] - r["theta_gt_deg"])
        print(f"  {'combined (quadrature)':<38}{'':>8}{t[0]:10.4f}{t[1]*1e3:11.2f}{t[2]:10.2f}")
        print(f"  observed |z_c − GT| = {dz*1e3:.1f} mm, |θ* − GT| = {dth:.2f}°  →  "
              f"{'WITHIN' if dz <= t[1] else 'EXCEEDS'} the 1σ budget")
    print("=" * 100)


# ==========================================================================================
#  FIGURE
# ==========================================================================================
def plot_fwd_tip_fits(all_results, save_dir):
    """Pivot torque vs tilt for every rocking log, with the method-D fits and the GT curve.
    With vision, the torque against the camera tilt is overlaid (hollow markers)."""
    import matplotlib.pyplot as plt
    if not all_results:
        return
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), sharey=True, squeeze=False)
    for ax, r in zip(axes[0], all_results):
        c = r["_curves"]
        phi_deg = np.degrees(c["phi"])
        top = max(r["theta_gt_deg"], r["theta_est_deg"]) + 3
        grid = np.radians(np.linspace(0, top, 200))
        rv = np.c_[np.zeros_like(grid), grid, np.zeros_like(grid)]
        for ph, col in (("tip", "#2a78d6"), ("return", "#52514e")):
            s = c["sel"][ph]
            f = r["branch_fits"][ph]["D"]
            ax.plot(phi_deg[s], c["tau"][s] * 1e3, ".", ms=2, color=col, alpha=0.3)
            rvo = rv.copy()
            rvo[:, 1] -= np.radians(f["offset_deg"])
            fit = model_fwd_wrench(rvo, np.array([-c["cx"], 0, f["zc"]]), f["m"])[:, 1]
            ax.plot(np.degrees(grid), fit * 1e3, color=col, lw=1.6,
                    label=f"{ph} D: z$_c$={f['zc']*1e3:.0f} mm")
        v = c.get("vision")
        if v is not None:
            sel = np.where(v["mask"])[0][::10]
            ax.plot(np.degrees(v["tilt"][sel]), v["tau"][sel] * 1e3, "o", ms=2.5, mfc="none",
                    mec="#eb6834", mew=0.6, alpha=0.7, label="torque vs vision tilt")
        gt = model_fwd_wrench(rv, np.array([-c["cx"], 0, r["com_z_gt"]]), r["mass_est"])[:, 1]
        ax.plot(np.degrees(grid), gt * 1e3, ":", color="#0b0b0b", lw=1.5,
                label=f"GT CoM (m={r['mass_est']:.3f} kg)")
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_title(f"{r['obj']}  θ*={r['theta_est_deg']:.1f}° (GT {r['theta_gt_deg']:.1f}°)\n"
                     f"slide {r['slide_mm']:.0f} mm ({r['slide_source']})", loc="left", fontsize=9)
        ax.set_xlabel("tilt φ [deg]")
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(True, color="#e4e3df")
    axes[0][0].set_ylabel("applied torque about pivot [mN·m]")
    fig.tight_layout()
    out = os.path.join(save_dir, "fwd_tip_fits.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main_fwd_tip(objects, root, out_dir, verbose_trials=False, uncertainty=False, vision=False):
    """Forward-tip counterpart of estimate_params.main(); returns the aggregated records."""
    from irb120_control.estimation.estimate_params import (
        aggregate_trials, print_repeatability_stats, print_results_table, write_per_trial_csv)

    all_results, all_outcomes = [], []
    for obj in objects:
        results, outcomes = run_fwd_tip_object(obj, root, verbose_trials, vision=vision)
        all_results += results
        all_outcomes += outcomes
    if not all_results:
        print("No forward-tip log rocked — nothing to estimate.")
        return []

    # The suite is one body with a moved weight, so one batch mass serves the friction
    # estimate for logs that never rocked (and therefore have no mass of their own).
    mass = float(np.mean([r["mass_est"] for r in all_results]))
    friction_rows = add_friction(all_results, all_outcomes, mass)
    print_fwd_tip_report(all_results, all_outcomes, friction_rows, mass)
    aggs = [aggregate_trials(obj, [r for r in all_results if r["obj"] == obj])
            for obj in objects if any(r["obj"] == obj for r in all_results)]

    os.makedirs(out_dir, exist_ok=True)
    write_per_trial_csv(aggs, out_dir)
    print_results_table(aggs, out_dir)
    print_repeatability_stats(aggs)
    plot_fwd_tip_fits(all_results, out_dir)
    if uncertainty:
        print_fwd_tip_uncertainty(all_results)
    return aggs
