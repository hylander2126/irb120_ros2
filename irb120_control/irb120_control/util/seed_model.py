"""Seeded perception: reuse an episode's first full object model instead of re-running SAM.

Within one episode the object is the same rigid body; between interactions it can only
slide and turn on the table (if it tilted or fell, it is no longer the same planar pose and
a full perceive runs instead). So a later snapshot only needs the object's new planar pose:
register the model cloud (the SAM-culled cloud of the episode's first full perceive) to the
fresh DBSCAN cloud with a planar (x, y, yaw) trimmed ICP, then move the model, and the SAM
pivot fits that were made on it, by that transform. Contacts are then re-selected on the
moved model as usual.

The fit is accepted only if most model points land on the new cloud and the object's top
height is unchanged; anything else falls back to the full SAM perceive.
"""
import copy

import numpy as np
from scipy.spatial import cKDTree

INLIER_MM = 5.0             # a model point within this of the new cloud is on it
MIN_INLIER_FRACTION = 0.80  # below this the model does not match: full perceive
MAX_HEIGHT_CHANGE_MM = 10.0  # = tip_cascade.HEIGHT_CHANGE_MM: tilted or fell -> full perceive
ICP_ITERATIONS = 40
AMBIGUOUS_FRACTION = 0.03  # rival fits within this inlier fraction of the best...
AMBIGUOUS_MM = 5.0          # ...but further than this from it make the pose ambiguous
PROBE_MM = 10.0             # observability probe: shift the fit this far along x and y...
MIN_PROBE_DROP = 0.15       # ...and the inlier fraction must drop by at least this much


def _rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _icp(model, tree, cloud, R, t):
    """Trimmed planar ICP from (R, t). Returns (R, t, inlier mask of model points)."""
    gate = 0.03
    for _ in range(ICP_ITERATIONS):
        moved = model @ R.T + t
        dist, idx = tree.query(moved)
        use = dist < gate
        if use.sum() < 10:
            break
        a, b = moved[use, :2], cloud[idx[use], :2]
        ca, cb = a.mean(0), b.mean(0)
        u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        d = np.sign(np.linalg.det(vt.T @ u.T))
        R2 = vt.T @ np.diag([1., d]) @ u.T
        step = np.eye(3)
        step[:2, :2] = R2
        R, t = step @ R, step @ t + np.r_[cb - R2 @ ca, 0.]
        gate = max(INLIER_MM / 1000, 0.7 * gate)
    dist, _ = tree.query(model @ R.T + t)
    return R, t, dist < INLIER_MM / 1000


def register_planar(model, cloud) -> dict:
    """Planar rigid transform (R about z, t in the table plane) taking `model` onto `cloud`.

    Both (N,3) in base_link. Matching runs model -> cloud, so extra points in the
    un-culled DBSCAN cloud do not pull the fit. Returns R, t, yaw_deg, rms_mm,
    inlier_fraction, height_change_mm, ok and reason."""
    model = np.asarray(model, float).reshape(-1, 3)
    cloud = np.asarray(cloud, float).reshape(-1, 3)
    model, cloud = model[np.isfinite(model).all(1)], cloud[np.isfinite(cloud).all(1)]
    if len(model) < 20 or len(cloud) < 20:
        return dict(ok=False, reason="too few points to register")
    tree = cKDTree(cloud)
    # Multi-start: the un-culled cloud's extra points bias any centroid guess by
    # centimetres, so also start from "not moved" and from the guess's x alone (pushes
    # and slides move the object along x).
    shift = np.r_[np.median(cloud[:, :2], 0) - np.median(model[:, :2], 0), 0.]
    fits = [_icp(model, tree, cloud, np.eye(3), t0) for t0 in (np.zeros(3), shift, shift * [1, 0, 0])]
    R, t, inl = max(fits, key=lambda f: f[2].mean())
    # Two different poses fitting about as well (a prism slid along its own axis, a
    # symmetric face) means the cloud does not pin the pose: do not guess.
    rival = [f for f in fits if f[2].mean() >= inl.mean() - AMBIGUOUS_FRACTION
             and (np.linalg.norm(f[1] - t) * 1000 > AMBIGUOUS_MM
                  or abs(np.degrees(np.arctan2(f[0][1, 0], f[0][0, 0]) - np.arctan2(R[1, 0], R[0, 0]))) > 2.)]
    dist, _ = tree.query(model @ R.T + t)
    height = float(np.percentile(cloud[:, 2], 95) - np.percentile(model[:, 2], 95)) * 1000
    out = dict(R=R.tolist(), t=t.tolist(), yaw_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
               rms_mm=float(np.sqrt(np.mean(dist[inl] ** 2)) * 1000) if inl.any() else float("nan"),
               inlier_fraction=float(inl.mean()), height_change_mm=height)
    if out["inlier_fraction"] < MIN_INLIER_FRACTION:
        return dict(out, ok=False, reason=f"only {100 * out['inlier_fraction']:.0f}% of the model "
                                          f"matches the new cloud (< {100 * MIN_INLIER_FRACTION:.0f}%)")
    if rival:
        return dict(out, ok=False, reason=f"ambiguous pose: {len(rival)} other fit(s) match about as well")
    if not abs(height) <= MAX_HEIGHT_CHANGE_MM:
        return dict(out, ok=False, reason=f"top height changed {height:+.0f} mm: tilted or fell")
    # An extrusion (e.g. the tipT, a prism along y) matches equally well slid along its
    # axis: the pose is then not observable from the cloud alone, so do not guess.
    drops = {}
    for name, axis in (("x", np.r_[1., 0, 0]), ("y", np.r_[0, 1., 0])):
        probe = [(tree.query(model @ R.T + t + sgn * PROBE_MM / 1000 * axis)[0] < INLIER_MM / 1000).mean()
                 for sgn in (-1, 1)]
        drops[name] = float(out["inlier_fraction"] - max(probe))
    out["probe_drop"] = drops
    flat = [k for k, v in drops.items() if v < MIN_PROBE_DROP]
    if flat:
        return dict(out, ok=False, reason=f"pose not observable along {'/'.join(flat)} "
                                          f"(shape slides along itself): full perceive")
    return dict(out, ok=True, reason=None)


def apply(fit: dict, points):
    """Move base_link points (N,3) or (3,) by a register_planar fit."""
    p = np.asarray(points, float)
    return p @ np.asarray(fit["R"]).T + np.asarray(fit["t"])


def moved_pivot_report(report: dict, fit: dict, source_stage: str) -> dict:
    """A SAM pivot report (pivot_boundary_review / forward_pivot_review) moved with the
    object. Image-space fields are dropped: they belong to the model's frames."""
    out = copy.deepcopy(report)
    R = np.asarray(fit["R"])
    geometry = out.get("geometry") or {}
    for key in ("candidate_pivot_base_m", "pivot_edge_base_m", "support_samples_base_m"):
        if geometry.get(key):  # None or [] (a rejected review) stays as it is
            geometry[key] = apply(fit, geometry[key]).tolist()
    for key in ("tangent_base",):
        if geometry.get(key):
            geometry[key] = (np.asarray(geometry[key], float) @ R.T).tolist()
    for key in [k for k in geometry if k.endswith("_px")]:
        geometry.pop(key)
    out["geometry"] = geometry
    out["source"] = f"seeded from {source_stage}: {report.get('source')}"
    out["seeded"] = {"model_stage": source_stage, "yaw_deg": fit["yaw_deg"], "t": fit["t"]}
    return out
