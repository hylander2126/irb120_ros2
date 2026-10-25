#!/usr/bin/env python3
"""Object tilt AND slide from the episode's camera videos (primitive-agnostic).

The proprioceptive tilt sources in tip_core are accurate but assume a contact model, so
they cannot tell tipping from sliding. This source measures the object itself.

Model. The object is rigid and, relative to its pose before contact, has rotated by phi
about the pivot edge (world y axis) and translated by s along world x:

    X(t) = R_y(phi) (X0 - p) + p + s x_hat

That form is exact whether the slide happens before, during or after the rotation (a
rotation about the slid pivot p + s is the same motion). X0 are the object's surface
points, taken from the perceive stage before the tip: that stage's aligned depth image is
back-projected with its camera calibration and kept where it lies on the segmented object
cloud. Each video frame is then fitted by direct (photometric) alignment of those pixels
against the same pixels in a reference frame recorded before contact: a 2-parameter
Gauss-Newton with Huber IRLS weights, coarse-to-fine, warm-started from the previous
frame. Pixels covered by the finger (ball + rod, known from the log) are masked out.

Timing. Newer recordings have a per-frame stamp sidecar (util/video_stamps.py); those
stamps are used directly and the ball track below only reports a residual offset. Older
videos have only the filename (1 s resolution) and a nominal 30 fps; that coarse sync is
refined by tracking the finger ball in the same video and solving for the time offset
that best matches its logged 3D trajectory projected into the image.

Output: a tip_core.TiltTrack on the log's time grid, with phi, the slide, a 1-sigma tilt
that includes the static noise floor, and validity (residual and coverage gates).
"""
import datetime
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from irb120_control.estimation.tip_core import TiltTrack, TipLog
from irb120_control.util.video_stamps import load_frame_stamps

BASE_LINK_IN_WORLD = np.array([0.0, 0.0, 0.021])   # irb120_with_finger.xacro world -> base_link
NOMINAL_FPS = 30.0
CLOUD_TOL_M = 0.006          # depth pixel must lie this close to the object cloud
SYNC_SEARCH_S = 1.5          # +- range searched around the filename time
BALL_RADIUS = 0.01325
FINGER_MASK_M = 0.012        # extra margin around the projected finger
SENSOR_BODY_R = 0.045        # radius of the F/T sensor / flange body around the wrench origin
RMS_GATE = 3.0               # frame invalid if photometric RMS > RMS_GATE x static RMS
COVERAGE_GATE = 0.5          # frame invalid if fewer than this fraction of pixels are usable
# cam2 looks at the object's broad face from the side, so tilt about y and slide along x
# are both in-plane motions there. cam1/cam3 see the push nearly end-on (slide is along
# their viewing ray) and the arm crosses cam1's view, so they are only a fallback.
DEFAULT_CAMERA = "cam2"
_VIDEO_TS = re.compile(r"(\d{8}_\d{6})\.(mp4|avi)$")


# ==========================================================================================
#  ASSETS
# ==========================================================================================
def _stage_number(p: Path) -> int:
    m = re.match(r"(\d+)_", p.name)
    return int(m.group(1)) if m else -1


def find_assets(log: TipLog, camera: str) -> dict | None:
    """Calibration, depth, object cloud (from the perceive stage before the tip) and video."""
    if log.episode_dir is None:
        return None
    ep = Path(log.episode_dir)
    tip_dir = Path(log.path).parent
    n_tip = _stage_number(tip_dir)
    perceive = sorted((p for p in ep.iterdir() if p.is_dir() and p.name.endswith("_perceive")
                       and _stage_number(p) < n_tip), key=_stage_number)
    if not perceive:
        return None
    pd = perceive[-1]
    from irb120_perception.snapshot_layout import snapshot_file
    info_p, depth_p, cloud_p = (snapshot_file(pd, f"{camera}_camera_info.json"),
                                snapshot_file(pd, f"{camera}_aligned_depth.npz"), pd / "object_cloud.npz")
    tag = "" if camera == "cam1" else f"_{camera}"
    videos = sorted(tip_dir.glob(f"camera_hull_overlay{tag}_2*.mp4")) + sorted(
        tip_dir.glob(f"camera_hull_overlay{tag}_2*.avi"))
    if not (info_p.exists() and depth_p.exists() and cloud_p.exists() and videos):
        return None
    info = json.loads(info_p.read_text())
    return {"camera": camera, "perceive": pd.name, "video": str(videos[-1]),
            "K": np.array(info["K"], float),
            "R": np.array(info["base_to_optical"]["R"], float),
            "t": np.array(info["base_to_optical"]["t"], float),
            "depth": np.load(depth_p)["depth_mm"].astype(float) / 1000.0,
            "cloud": np.load(cloud_p)["xyz"]}


def video_start_epoch(video: str) -> float:
    """Epoch of the first frame from the recorder's local-time filename (+-0.5 s)."""
    m = _VIDEO_TS.search(os.path.basename(video))
    if not m:
        raise ValueError(f"no timestamp in {video}")
    return datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp() + 0.5


def _read_gray(video: str):
    cap = cv2.VideoCapture(video)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cap.release()


# ==========================================================================================
#  GEOMETRY
# ==========================================================================================
class _Camera:
    def __init__(self, a):
        self.K, self.R, self.t = a["K"], a["R"], a["t"]

    def to_cam(self, Xb):
        return Xb @ self.R.T + self.t

    def project(self, Xb):
        pc = self.to_cam(Xb)
        return (self.K[0, 0] * pc[:, 0] / pc[:, 2] + self.K[0, 2],
                self.K[1, 1] * pc[:, 1] / pc[:, 2] + self.K[1, 2], pc[:, 2])

    def backproject(self, u, v, z):
        pc = np.c_[(u - self.K[0, 2]) / self.K[0, 0] * z, (v - self.K[1, 2]) / self.K[1, 1] * z, z]
        return (pc - self.t) @ self.R


def object_pixels(a, cam: _Camera):
    """Reference-image pixels on the object and their 3D points (base_link)."""
    depth = a["depth"]
    mask = np.zeros(depth.shape, np.uint8)
    v, u = np.nonzero(depth > 0.1)
    X = cam.backproject(u.astype(float), v.astype(float), depth[v, u])
    near = cKDTree(a["cloud"]).query(X)[0] < CLOUD_TOL_M
    mask[v[near], u[near]] = 1
    # Drop the silhouette rim: those pixels mix object and background once it moves.
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    v, u = np.nonzero(mask)
    return u, v, cam.backproject(u.astype(float), v.astype(float), depth[v, u])


def _rigid(X, pivot_b, phi, s):
    c, sn = np.cos(phi), np.sin(phi)
    Ry = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])
    return (X - pivot_b) @ Ry.T + pivot_b + np.array([s, 0.0, 0.0])


def conditioning(a, pivot_b) -> float:
    """Smallest singular value of d(pixels)/d(phi [rad], s [m]) per pixel: how well this
    camera separates tilt from slide. Used to pick the camera."""
    cam = _Camera(a)
    _u, _v, X = object_pixels(a, cam)
    if len(X) < 200:
        return 0.0
    u0, v0, _ = cam.project(X)
    cols = []
    for dp in ((1e-3, 0.0), (0.0, 1e-3)):
        u1, v1, _ = cam.project(_rigid(X, pivot_b, *dp))
        cols.append(np.r_[u1 - u0, v1 - v0] / 1e-3)
    J = np.c_[cols[0] * 0.1, cols[1]] / np.sqrt(len(X))   # 0.1 rad ~ 100 mm: equal weight
    return float(np.linalg.svd(J, compute_uv=False).min())


def _finger_mask_fn(cam, shape):
    """Pixels covered by the ball, the finger rod and the sensor body at one log sample."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]

    def capsule(p_a, p_b, r_m):
        ua, va, za = cam.project(p_a[None])
        ub, vb, zb = cam.project(p_b[None])
        f = cam.K[0, 0]
        r = r_m * f / min(za[0], zb[0])
        d = np.array([ub[0] - ua[0], vb[0] - va[0]])
        L2 = max(float(d @ d), 1e-9)
        tt = np.clip(((xx - ua[0]) * d[0] + (yy - va[0]) * d[1]) / L2, 0, 1)
        return np.hypot(xx - (ua[0] + tt * d[0]), yy - (va[0] + tt * d[1])) < r

    def fn(p_ball_b, p_S_b):
        m = capsule(p_ball_b, p_ball_b, BALL_RADIUS + FINGER_MASK_M)
        m |= capsule(p_ball_b, p_S_b, 0.006 + FINGER_MASK_M)
        m |= capsule(p_S_b, p_S_b, SENSOR_BODY_R + FINGER_MASK_M)
        return m
    return fn


# ==========================================================================================
#  TIME SYNC (finger ball)
# ==========================================================================================
def refine_sync(log: TipLog, a, cam: _Camera, t_start: float, n_frames: int, patch_r=None,
                frame_times=None) -> dict:
    """Time offset of the video against the log, from the finger ball's image track.
    frame_times: per-frame stamps when the video has a sidecar; then the offset found is
    a residual check (expected ~0), not the sync itself."""
    tf = t_start + np.arange(n_frames) / NOMINAL_FPS if frame_times is None else np.asarray(frame_times)
    p_b = log.p_ball - BASE_LINK_IN_WORLD
    t_log = log.t_epoch
    in_log = (tf > t_log[0] + SYNC_SEARCH_S) & (tf < t_log[-1] - SYNC_SEARCH_S)
    if in_log.sum() < 60:
        return {"offset_s": 0.0, "ok": False, "why": "video barely overlaps the log"}

    def pred(times):
        P = np.column_stack([np.interp(times, t_log, p_b[:, k]) for k in range(3)])
        u, v, z = cam.project(P)
        return np.c_[u, v], z

    uv_all, z_all = pred(tf)
    r_px = int(round((patch_r or BALL_RADIUS) * cam.K[0, 0] / np.median(z_all[in_log]))) + 2
    # Template from the frame where the predicted ball is most central in the log window.
    i_tpl = int(np.where(in_log)[0][len(np.where(in_log)[0]) // 2])
    tracks = np.full((n_frames, 2), np.nan)
    tpl = None
    search = r_px + 25
    for i, img in enumerate(_read_gray(a["video"])):
        if i >= n_frames:
            break
        if not in_log[i]:
            continue
        if tpl is None:
            # Grab the template on the way (frames before i_tpl are revisited below).
            if i < i_tpl:
                continue
            cu, cv_ = np.round(uv_all[i]).astype(int)
            tpl = img[cv_ - r_px:cv_ + r_px + 1, cu - r_px:cu + r_px + 1].copy()
            if tpl.shape != (2 * r_px + 1, 2 * r_px + 1):
                return {"offset_s": 0.0, "ok": False, "why": "ball outside the image"}
        tracks[i] = _match(img, tpl, uv_all[i], search)
    # Second pass for the frames before the template frame.
    for i, img in enumerate(_read_gray(a["video"])):
        if i >= i_tpl:
            break
        if in_log[i]:
            tracks[i] = _match(img, tpl, uv_all[i], search)
    good = np.isfinite(tracks).all(1)
    if good.sum() < 60:
        return {"offset_s": 0.0, "ok": False, "why": "ball not tracked"}
    best = None
    for dt in np.arange(-SYNC_SEARCH_S, SYNC_SEARCH_S + 1e-9, 0.01):
        uv, _ = pred(tf[good] + dt)
        r = tracks[good] - uv
        r -= np.median(r, axis=0)          # calibration bias of the ball centre
        cost = float(np.mean(np.minimum(np.sum(r ** 2, 1), 25.0)))   # truncated: occlusions
        if best is None or cost < best[1]:
            best = (dt, cost)
    uv, _ = pred(tf[good] + best[0])
    motion = float(np.ptp(uv, axis=0).max())
    return {"offset_s": float(best[0]), "ok": motion > 5.0, "rms_px": float(np.sqrt(best[1])),
            "ball_motion_px": motion, "n": int(good.sum()),
            "why": "" if motion > 5.0 else "finger barely moves in this view"}


def _match(img, tpl, uv, search):
    cu, cv_ = np.round(uv).astype(int)
    h = tpl.shape[0] // 2
    y0, x0 = cv_ - search, cu - search
    if y0 < 0 or x0 < 0 or cv_ + search + 1 > img.shape[0] or cu + search + 1 > img.shape[1]:
        return np.nan, np.nan
    win = img[y0:cv_ + search + 1, x0:cu + search + 1]
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _mn, mx, _l, (bx, by) = cv2.minMaxLoc(res)
    if mx < 0.6:
        return np.nan, np.nan

    def sub(c, l_, r_):
        den = l_ - 2 * c + r_
        return 0.0 if abs(den) < 1e-9 else 0.5 * (l_ - r_) / den
    dx = sub(res[by, bx], res[by, bx - 1], res[by, bx + 1]) if 0 < bx < res.shape[1] - 1 else 0.0
    dy = sub(res[by, bx], res[by - 1, bx], res[by + 1, bx]) if 0 < by < res.shape[0] - 1 else 0.0
    return x0 + bx + dx + h, y0 + by + dy + h


# ==========================================================================================
#  PHOTOMETRIC FIT
# ==========================================================================================
def _sample(img, u, v):
    return cv2.remap(img, u.astype(np.float32).reshape(-1, 1), v.astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan).ravel()


class _Aligner:
    """Two-parameter (phi, s) direct alignment against a reference frame."""
    SCALES = ((3.0, 6), (1.0, 10))   # (Gaussian sigma px, max iterations), coarse to fine

    def __init__(self, cam, X, ref_img, pivot_b, U, V):
        self.cam, self.X, self.pivot_b = cam, X, pivot_b
        self.ref = {s: cv2.GaussianBlur(ref_img, (0, 0), s)[V, U] for s, _ in self.SCALES}

    def _uvJ(self, p, eps=1e-4):
        u0, v0, _ = self.cam.project(_rigid(self.X, self.pivot_b, *p))
        J = []
        for k in range(2):
            q = np.array(p, float)
            q[k] += eps
            u1, v1, _ = self.cam.project(_rigid(self.X, self.pivot_b, *q))
            J.append(((u1 - u0) / eps, (v1 - v0) / eps))
        return u0, v0, J

    def fit(self, img, p, keep=None):
        for sigma, iters in self.SCALES:
            im = cv2.GaussianBlur(img, (0, 0), sigma)
            gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            rf = self.ref[sigma]
            for _ in range(iters):
                u, v, J = self._uvJ(p)
                r = _sample(im, u, v) - rf
                Ix, Iy = _sample(gx, u, v), _sample(gy, u, v)
                A = np.c_[Ix * J[0][0] + Iy * J[0][1], Ix * J[1][0] + Iy * J[1][1]]
                ok = np.isfinite(r) & np.isfinite(A).all(1)
                if keep is not None:
                    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
                    inside = (ui >= 0) & (vi >= 0) & (ui < keep.shape[1]) & (vi < keep.shape[0])
                    ok &= inside
                    ok[inside] &= keep[vi[inside], ui[inside]]
                if ok.sum() < 50:
                    return p, np.nan, np.full(2, np.nan), 0.0
                rr, AA = r[ok], A[ok]
                c = 1.4826 * np.median(np.abs(rr)) + 1e-6
                w = np.minimum(1.0, 2.0 * c / (np.abs(rr) + 1e-9))        # Huber IRLS
                H = AA.T @ (AA * w[:, None])
                dp = -np.linalg.solve(H + 1e-9 * np.eye(2), AA.T @ (w * rr))
                p = p + dp
                if abs(dp[0]) < 1e-5 and abs(dp[1]) < 2e-5:
                    break
        dof = max(ok.sum() - 2, 1)
        cov = (np.sum(w * rr ** 2) / dof) * np.linalg.inv(H)
        return p, float(np.sqrt(np.mean(rr ** 2))), np.sqrt(np.diag(cov)), float(ok.mean())


# ==========================================================================================
#  ENTRY POINT
# ==========================================================================================
def choose_camera(log: TipLog, cameras=("cam1", "cam2", "cam3")):
    pivot_b = log.pivot - BASE_LINK_IN_WORLD
    scored = []
    for c in cameras:
        a = find_assets(log, c)
        if a is not None:
            scored.append((conditioning(a, pivot_b), c, a))
    if not scored:
        return None, None, []
    scored.sort(key=lambda x: -x[0])
    return scored[0][1], scored[0][2], [(c, k) for k, c, _ in scored]


def track_tilt_vision(log: TipLog, camera: str | None = None, cache_dir: str | None = None,
                      verbose: bool = True) -> TiltTrack | None:
    """Vision TiltTrack for one log, or None when the episode has no usable camera data.

    camera None uses DEFAULT_CAMERA, or the best-conditioned other camera if that one has
    no data. Results are cached as
    <tip stage>/vision_tilt_<camera>.npz (the fit is ~30 ms/frame)."""
    if camera is None and find_assets(log, DEFAULT_CAMERA) is not None:
        camera = DEFAULT_CAMERA
    if camera is None:
        camera, a, scores = choose_camera(log)
        if camera is None:
            return None
    else:
        a, scores = find_assets(log, camera), []
        if a is None:
            return None
    cache = Path(cache_dir or Path(log.path).parent) / f"vision_tilt_{camera}_{Path(log.path).stem}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        frames = {k: z[k] for k in z.files}
        sync = json.loads(str(frames.pop("sync")))
    else:
        frames, sync = _fit_video(log, a, verbose)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, sync=json.dumps(sync), **frames)
    track = _to_track(log, frames, sync)
    track.info.update(camera=camera, camera_scores=scores, video=os.path.basename(a["video"]),
                      sync=sync, cache=str(cache))
    return track


def _fit_video(log, a, verbose):
    cam = _Camera(a)
    pivot_b = log.pivot - BASE_LINK_IN_WORLD
    U, V, X = object_pixels(a, cam)
    if len(X) < 200:
        raise ValueError(f"only {len(X)} object pixels in {a['camera']}")
    cap = cv2.VideoCapture(a["video"])
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    stamps = load_frame_stamps(a["video"])
    if stamps is not None and len(stamps[1]) >= n_frames:
        # Recorded per-frame stamps: use them directly; the ball track only checks them.
        frame_times = stamps[1][:n_frames]
        sync = refine_sync(log, a, cam, frame_times[0], n_frames, frame_times=frame_times)
        sync.update(source="frame_stamps", residual_offset_s=sync["offset_s"], offset_s=0.0)
        tf = frame_times
    else:
        t_start = video_start_epoch(a["video"])
        sync = refine_sync(log, a, cam, t_start, n_frames)
        if not sync["ok"]:
            sync["offset_s"] = 0.0
        sync["source"] = "filename_and_ball_track"
        tf = t_start + sync["offset_s"] + np.arange(n_frames) / NOMINAL_FPS
    if verbose:
        print(f"    vision {a['camera']}: {len(X)} object px, sync {sync['offset_s']:+.2f} s "
              f"({'ball track' if sync['ok'] else 'filename only: ' + sync['why']})")

    # Reference: the first frame -- the object is untouched and the arm is away from it.
    # The fit then runs forward with a warm start through the whole video.
    finger = _finger_mask_fn(cam, a["depth"].shape)
    p_ball_b = log.p_ball - BASE_LINK_IN_WORLD
    p_S_b = log.p_S - BASE_LINK_IN_WORLD
    aligner, p = None, np.zeros(2)
    out = {k: np.full(n_frames, np.nan) for k in ("phi", "s", "rms", "sd_phi", "sd_s", "coverage")}
    for i, img in enumerate(_read_gray(a["video"])):
        if i >= n_frames:
            break
        if aligner is None:
            aligner = _Aligner(cam, X, img, pivot_b, U, V)
        keep = None
        if log.t_epoch[0] <= tf[i] <= log.t_epoch[-1]:
            k = int(np.clip(np.searchsorted(log.t_epoch, tf[i]), 0, len(log.t_epoch) - 1))
            keep = ~finger(p_ball_b[k], p_S_b[k])
        p, rms, sd, cov = aligner.fit(img, p, keep)
        if not np.isfinite(p).all():
            p = np.zeros(2)
        out["phi"][i], out["s"][i], out["rms"][i] = p[0], p[1], rms
        out["sd_phi"][i], out["sd_s"][i], out["coverage"][i] = sd[0], sd[1], cov
    out["t"] = tf
    return out, sync


def _to_track(log: TipLog, fr: dict, sync: dict) -> TiltTrack:
    t = fr["t"]
    pre = t < log.t_epoch[np.argmax(log.contact)] - 0.5     # untouched object: noise floor
    rms0 = np.nanmedian(fr["rms"][pre]) if pre.sum() > 10 else np.nanmedian(fr["rms"])
    floor_phi = np.nanstd(fr["phi"][pre]) if pre.sum() > 10 else 0.0
    good = (np.isfinite(fr["phi"]) & (fr["coverage"] >= COVERAGE_GATE)
            & (fr["rms"] <= RMS_GATE * max(rms0, 1.0)))
    tl = log.t_epoch
    inside = (tl >= t[0]) & (tl <= t[-1])
    phi = np.interp(tl, t[good], fr["phi"][good]) if good.any() else np.full(len(tl), np.nan)
    s = np.interp(tl, t[good], fr["s"][good]) if good.any() else np.full(len(tl), np.nan)
    sd = np.interp(tl, t, np.hypot(np.nan_to_num(fr["sd_phi"]), floor_phi))
    # A log sample is valid when the nearest frame on each side is good and close.
    j = np.clip(np.searchsorted(t, tl), 1, len(t) - 1)
    near_ok = good[j - 1] & good[j] & (t[j] - t[j - 1] < 2.5 / NOMINAL_FPS)
    valid = inside & near_ok
    phi[~valid], s[~valid] = np.nan, np.nan
    rv = np.zeros((len(tl), 3))
    rv[:, 1] = phi                       # rotation about world +y, as _rigid()
    slide = s * float(np.array([1.0, 0.0, 0.0]) @ log.push_dir)
    pivot_t = log.pivot + np.c_[np.nan_to_num(s), np.zeros(len(tl)), np.zeros(len(tl))]
    return TiltTrack(rotvec=rv, valid=valid, source="vision", slide=slide, sigma=sd, pivot_t=pivot_t,
                     info={"rms_static": float(rms0), "tilt_noise_floor_deg": float(np.degrees(floor_phi)),
                           "frames_good": float(good.mean())})
