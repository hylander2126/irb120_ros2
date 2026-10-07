"""Experimental local top-support gate and wrench scoring; no motion changes."""
import numpy as np
from scipy.spatial import ConvexHull, QhullError


def has_upward_upper_surface(points, normals, upper_quantile=.85, min_normal_z=.7):
    """Whether upward-normal evidence occurs near the object top, not anywhere.

    A single noisy upward normal low on a side or near the table must not suppress
    horizontal-cap inference for an otherwise missing top surface.
    """
    points = np.asarray(points)
    normals = np.asarray(normals)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return False
    upper_floor = np.quantile(points[finite, 2], upper_quantile)
    return bool(np.any(finite & np.isfinite(normals).all(axis=1)
                       & (points[:, 2] >= upper_floor)
                       & (normals[:, 2] >= min_normal_z)))


def infer_horizontal_top(points, spacing=.005, rim_depth=.015, margin=.004):
    """Infer candidate samples on an unobserved flat cap from its upper rim.

    This is used only when fused PCA produces no upward-facing top candidates.
    The returned points are hypotheses and must still be validated against a
    usable camera depth view before execution.
    """
    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 20:
        return np.empty((0, 3)), dict(available=False, reason='insufficient object points')
    top_z = float(np.quantile(points[:, 2], .995))
    rim = points[points[:, 2] >= top_z-rim_depth]
    if len(rim) < 12:
        return np.empty((0, 3)), dict(available=False, reason='insufficient upper-rim support')
    try:
        hull = ConvexHull(rim[:, :2])
    except (ValueError, QhullError):
        return np.empty((0, 3)), dict(available=False, reason='degenerate upper-rim footprint')
    low, high = rim[:, :2].min(0), rim[:, :2].max(0)
    if np.min(high-low) < 2*margin:
        return np.empty((0, 3)), dict(available=False, reason='upper rim too narrow')
    xs = np.arange(low[0], high[0]+spacing*.5, spacing)
    ys = np.arange(low[1], high[1]+spacing*.5, spacing)
    xy = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    signed = xy @ hull.equations[:, :2].T + hull.equations[:, 2]
    xy = xy[np.max(signed, axis=1) <= -margin]
    candidates = np.column_stack((xy, np.full(len(xy), top_z)))
    return candidates, dict(available=bool(len(candidates)), method='upper-rim horizontal-cap inference',
        top_z=top_z, rim_depth_mm=rim_depth*1000, margin_mm=margin*1000,
        rim_samples=len(rim), candidate_count=len(candidates),
        xy_bounds=[low.tolist(), high.tolist()])


def assess(point, normal, points, views, require_local_support=True):
    """Fixed development thresholds; all usable cameras must support the patch.

    A 15-degree allowance around tangency replaces V2's brittle zero sign test.
    It is a hypothesis, not a friction model or a calibrated uncertainty bound.
    """
    top_cos = np.cos(np.radians(30))
    if not np.isfinite(normal).all() or normal[2] < top_cos:
        return 'reject', 'fused surface exceeds 30 degree top slope'
    if require_local_support:
        local = points[np.linalg.norm(points-point, axis=1) <= .012]
        center = point+.01325*normal
        try:
            hull = ConvexHull(local[:, :2])
        except (ValueError, QhullError):
            return 'uncertain', 'insufficient observed local top support'
        if np.min(-(hull.equations[:, :2] @ center[:2]+hull.equations[:, 2])) < 0:
            return 'reject', 'ball center projects outside observed local patch'
    usable = [v for v in views.values() if v['usable']]
    if not usable:
        return 'uncertain', 'no usable camera view'
    for view in usable:
        n = np.asarray(view['normal'])
        if n @ normal < np.cos(np.radians(25)):
            return 'uncertain', 'camera and fused normals disagree'
        if n[2] < top_cos:
            return 'reject', 'camera surface exceeds 30 degree top slope'
        if view['normal_dot_pull'] > np.sin(np.radians(15)):
            return 'reject', 'camera normal exceeds 15 degree pull tangency allowance'
    for a in usable:
        for b in usable:
            if np.dot(a['normal'], b['normal']) < np.cos(np.radians(25)):
                return 'uncertain', 'camera normals disagree'
    return 'pass', 'supported top patch with consistent normals'


def tipping_torque_scores(points, geometry, downward_to_pull_ratio=1.0):
    """Signed moment about the pivot axis from the expected applied wrench.

    ``direction`` is the horizontal force on the object.  The downward press
    opposes tipping when a contact lies inboard of the pivot, so it must be
    included in the same cross product rather than handled as a shape rule.
    """
    force = geometry['direction'] + np.array([0., 0., -downward_to_pull_ratio])
    return np.cross(np.asarray(points)-geometry['pivot'], force) @ geometry['axis']
