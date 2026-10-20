"""Horizontal CoM distance from a filled convex hull of the perceived cloud (GT-free).

This is the paper's stated modelling approximation ("shape is sufficiently complete:
filled convex hull"), i.e. uniform density inside the hull. It is an ASSUMPTION used to
turn the identifiable first moment h_x = m*d into a mass; label results that use it.
"""
import numpy as np
from scipy.spatial import ConvexHull

TABLE_Z_BASE_LINK = -0.021


def hull_centroid(points):
    """Volume centroid of the convex hull, closed down to the table plane."""
    p = np.asarray(points, float); p = p[np.isfinite(p).all(1)]
    floor = p.copy(); floor[:, 2] = TABLE_Z_BASE_LINK      # the unseen bottom rests on the table
    hull = ConvexHull(np.vstack([p, floor]))
    pts = hull.points; c0 = pts[hull.vertices].mean(0)
    vol, acc = 0.0, np.zeros(3)
    for simplex in hull.simplices:
        a, b, c = pts[simplex]
        v = abs(np.dot(a - c0, np.cross(b - c0, c - c0))) / 6
        vol += v; acc += v * (a + b + c + c0) / 4
    return acc / vol, vol


def horizontal_distance(centroid, pivot, tilt_axis):
    """|horizontal offset of the centroid from the pivot, perpendicular to the tilt axis|."""
    lever = np.cross(np.asarray(tilt_axis, float), [0., 0., 1.])
    lever /= np.linalg.norm(lever)
    return float(abs(np.dot(np.asarray(centroid) - np.asarray(pivot), lever)))
