"""Per-camera lower-profile evidence from the expanded support scene. No CAD/GT imports.

For each camera separately:
  1. reject depth-edge (flying) pixels using the saved aligned depth image,
  2. fit that camera's own local table plane in an annulus around the object,
  3. keep the face of the object that camera views head-on (azimuth sector),
  4. bin radial distance vs height above its OWN plane, and
  5. evaluate a chi-square profile over the circular-transition radius R.

The axis (cx, cy) comes from all cameras' wall points; each camera then gets its own
outer radius a_c so a per-view registration offset cannot masquerade as a fillet.
Bin noise = within-bin scatter plus a systematic floor measured from that camera's
own wall bins. Heights below 3 sigma of that camera's table residual, and below each fixed flare
cutoff, are treated as unobserved. Outer radius and wall lean are free per view.
Profile family: straight wall + circular lower transition (same as cloud_geometry).
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares

TABLE_Z_REF = -.021          # base_link z of the configured table
EDGE_JUMP_M = .008           # depth second difference over +-2 px -> flying/mixed pixel
SECTOR_DEG = 40.             # half-width of the head-on azimuth sector per camera
WALL_H = (.040, .120)        # wall band used for axis and per-camera outer radius
BIN_M = .001
PROFILE_TOP = .045
R_GRID = np.linspace(0, .036, 361)
# Depth sensors fill the object/table wedge, so the measured radius flares OUTWARD in the
# lowest few mm. The flare height is not known a priori, so every view is fitted at fixed
# cutoffs and the 95% intervals are unioned (envelope). Frozen before scoring on 2026-10-20.
CUTOFFS_M = (.003, .005, .007, .009)
MAIN_CUTOFF_M = .005


def edge_free(folder, camera_id, uv):
    depth = np.load(Path(folder) / f'cam{camera_id}_aligned_depth.npz')['depth_mm'].astype(float)
    depth[depth <= 0] = np.nan
    # Second difference: a smooth slanted surface (grazing view) passes, a discontinuity
    # between object and background (mixed pixels) does not. Missing neighbours fail.
    pad = np.pad(depth, 2, constant_values=np.nan)
    c = pad[2:-2, 2:-2]
    lap_u = np.abs(pad[2:-2, :-4] + pad[2:-2, 4:] - 2 * c)
    lap_v = np.abs(pad[:-4, 2:-2] + pad[4:, 2:-2] - 2 * c)
    jump = np.fmax(lap_u, lap_v)[uv[:, 1], uv[:, 0]] / 1000
    return np.nan_to_num(jump, nan=np.inf) < EDGE_JUMP_M


def fit_table(xyz, center, rmin, rmax=.12):
    rho = np.linalg.norm(xyz[:, :2] - center, axis=1)
    near = (rho > rmin) & (rho < rmax) & (np.abs(xyz[:, 2] - TABLE_Z_REF) < .015)
    pts = xyz[near]
    if len(pts) < 200:
        return None
    X = np.c_[pts[:, :2] - center, np.ones(len(pts))]
    sol = least_squares(lambda q: X @ q - pts[:, 2], [0, 0, TABLE_Z_REF], loss='soft_l1', f_scale=.001)
    res = sol.fun
    sigma = 1.4826 * np.median(np.abs(res - np.median(res)))
    return dict(plane=sol.x.tolist(), sigma_m=float(sigma), n=int(len(pts)),
                tilt_deg=float(np.degrees(np.arctan(np.hypot(*sol.x[:2])))))


def height_above(xyz, center, plane):
    return xyz[:, 2] - (np.c_[xyz[:, :2] - center, np.ones(len(xyz))] @ np.asarray(plane))


def camera_position(view):
    R = np.array(view['base_to_optical']['R']); t = np.array(view['base_to_optical']['t'])
    return -R.T @ t


def fillet_rho(h, a, R):
    hh = np.clip(h, 0, None)
    out = np.full(np.shape(h), a, float)
    low = hh < R
    out[low] = a - R + np.sqrt(np.maximum(0, R * R - (R - hh[low]) ** 2))
    return out


def chi2_profile(h, rho, sigma):
    """chi2(R) with outer radius a and wall lean k profiled out (linear given R)."""
    out, nuisance = [], []
    W = 1 / sigma
    for R in R_GRID:
        inset = fillet_rho(h, 0., R)              # <= 0 below R, 0 on the wall
        X = np.c_[np.ones_like(h), h - .035] * W[:, None]
        y = (rho - inset) * W
        q = np.linalg.lstsq(X, y, rcond=None)[0]
        out.append(float(np.sum((X @ q - y) ** 2))); nuisance.append(q.tolist())
    return np.array(out), nuisance


def sys_floor_of(bins):
    """Scatter of wall bins about their own line: surface/registration noise per bin."""
    wb = np.array([b for b in bins if b[0] > .025])
    k, a = np.polyfit(wb[:, 0] - .035, wb[:, 1], 1)
    return max(float(1.4826 * np.median(np.abs(wb[:, 1] - a - k * (wb[:, 0] - .035)))), .0003)


def interval(chi2, level):
    ok = R_GRID[chi2 <= chi2.min() + level]
    return [float(ok.min()), float(ok.max())]


def analyze_scene(folder, selector_xyz, cloud_dir=None):
    folder = Path(folder); cloud_dir = Path(cloud_dir or folder)
    meta = json.loads((cloud_dir / 'support_scene_cloud.json').read_text())
    cloud = np.load(cloud_dir / 'support_scene_cloud.npz')
    xyz = cloud['xyz'].astype(float); cam = cloud['camera_id']; uv = cloud['pixel_uv']
    clean = np.zeros(len(xyz), bool)
    for c in np.unique(cam):
        s = cam == c
        clean[s] = edge_free(folder, int(c), uv[s])
    # Axis from selector-cloud median, then refined on all cameras' clean wall points.
    center = np.median(selector_xyz[:, :2], axis=0)
    zwall = xyz[:, 2] - TABLE_Z_REF
    wall = clean & (zwall > WALL_H[0]) & (zwall < WALL_H[1]) & (np.linalg.norm(xyz[:, :2] - center, axis=1) < .07)
    w = xyz[wall]
    sol = least_squares(lambda q: np.linalg.norm(w[:, :2] - q[:2], axis=1) - q[2],
                        [*center, .035], loss='soft_l1', f_scale=.001)
    center = sol.x[:2]; outer_all = float(sol.x[2])
    views = []
    for view in meta['views']:
        c = view['camera_id']
        if view.get('status') != 'recorded':
            views.append(dict(camera=view['camera'], status='missing')); continue
        s = (cam == c) & clean
        table = fit_table(xyz[s], center, outer_all + .015)
        if table is None:
            views.append(dict(camera=view['camera'], status='insufficient table pixels')); continue
        h = height_above(xyz[s], center, table['plane'])
        d = xyz[s, :2] - center
        rho = np.linalg.norm(d, axis=1)
        position = camera_position(view)
        toward = position[:2] - center; horizontal = np.linalg.norm(toward); toward /= horizontal
        elevation = float(np.degrees(np.arctan2(position[2] - TABLE_Z_REF, horizontal)))
        cosang = (d @ toward) / np.maximum(rho, 1e-9)
        sector = cosang > np.cos(np.deg2rad(SECTOR_DEG))
        body = sector & (rho < outer_all + .012) & (rho > .5 * outer_all)
        hw = body & (h > WALL_H[0]) & (h < WALL_H[1])
        if hw.sum() < 100:
            views.append(dict(camera=view['camera'], status='face not visible')); continue
        h_min = max(.002, 3 * table['sigma_m'])
        edges = np.arange(0, PROFILE_TOP + BIN_M, BIN_M)
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            q = body & (h >= lo) & (h < hi)
            if q.sum() >= 5:
                r = rho[q]
                bins.append((lo + BIN_M / 2, float(np.median(r)), 1.4826 * float(np.median(np.abs(r - np.median(r)))), int(q.sum())))
        if len([b for b in bins if b[0] > .025]) < 5:
            views.append(dict(camera=view['camera'], status='wall bins insufficient')); continue
        fits = {}
        for cut in CUTOFFS_M:
            B = np.array([b for b in bins if b[0] >= max(h_min, cut)])
            hB, rB = B[:, 0], B[:, 1]
            sB = np.sqrt(sys_floor_of(bins) ** 2 + (B[:, 2] / np.sqrt(B[:, 3])) ** 2)
            chi2, nuisance = chi2_profile(hB, rB, sB)
            dof = max(len(hB) - 3, 1)
            # Misfit beyond the noise model widens the interval (Birge ratio), never narrows it.
            birge = max(1.0, float(chi2.min() / dof))
            chi2 = chi2 / birge
            j = int(np.argmin(chi2))
            fits[f'{cut * 1000:.0f}mm'] = dict(cutoff_m=float(max(h_min, cut)), best_R_m=float(R_GRID[j]),
                R_68_m=interval(chi2, 1.0), R_95_m=interval(chi2, 3.84), reduced_chi2=birge,
                outer_radius_m=float(nuisance[j][0]), wall_lean_mm_per_10mm=float(10 * nuisance[j][1]),
                chi2_profile=chi2.tolist(), bins_used=int(len(hB)))
        envelope = [min(f['R_95_m'][0] for f in fits.values()), max(f['R_95_m'][1] for f in fits.values())]
        main = fits[f'{MAIN_CUTOFF_M * 1000:.0f}mm']
        views.append(dict(camera=view['camera'], camera_id=int(c), status='ok', azimuth_toward_camera=toward.tolist(),
            elevation_deg=elevation, table=table, table_height_floor_m=float(h_min),
            systematic_floor_m=sys_floor_of(bins), fits=fits, best_R_m=main['best_R_m'],
            outer_radius_m=main['outer_radius_m'], R_95_envelope_m=envelope,
            bins=[dict(h=b[0], rho=b[1], mad=b[2], n=b[3]) for b in bins],
            points=dict(h=h[body].tolist(), rho=rho[body].tolist()),
            edge_rejected=int(((cam == c) & ~clean).sum())))
    return dict(center_xy=center.tolist(), outer_radius_all_m=outer_all, wall_points=int(wall.sum()),
                views=views, R_grid_m=R_GRID.tolist())


def support_view(scene, support_xy):
    """The camera whose head-on sector contains the direction axis -> support."""
    d = np.asarray(support_xy) - np.asarray(scene['center_xy']); d /= np.linalg.norm(d)
    ok = [v for v in scene['views'] if v['status'] == 'ok']
    return max(ok, key=lambda v: float(np.dot(v['azimuth_toward_camera'], d)))


# ------------------------------------------------------------------ shape-agnostic profile
LATERAL_BAND_M = 0.012      # half-width of the strip in front of the pivot (cylinder sagitta <2 mm)
ENVELOPE_Q = 0.95           # outermost surface toward the support in each height bin
MAX_ELEVATION_DEG = 45.     # steeper views cannot see a downward-facing fillet


def envelope_profile(folder, selector_xyz, pivot, push_dir, cloud_dir=None):
    """Support-side outer envelope u(h) per low-elevation camera; no shape family assumed.

    d = the horizontal push direction (the object tips toward it, so it points at the pivot side);
    the strip is the points within LATERAL_BAND_M of the line through the PIVOT along d
    (the selector-cloud median is biased sideways by uneven camera coverage). u is measured
    from c = selector-cloud median along d. For each camera: its own table plane and
    flying-pixel filter. In each 1 mm height bin, u is the ENVELOPE_Q
    quantile of (p - c).d, i.e. the outermost surface toward the support. A vertical face
    (box, tipT, cylinder seen head-on) gives a constant u; a fillet makes u recede near the
    table. The same fillet + lean chi2 machinery as the axisymmetric profile then applies,
    with the flare cutoffs. Returns the scene dict shape used by joint_fit.
    """
    folder = Path(folder); cloud_dir = Path(cloud_dir or folder)
    meta = json.loads((cloud_dir / 'support_scene_cloud.json').read_text())
    cloud = np.load(cloud_dir / 'support_scene_cloud.npz')
    xyz = cloud['xyz'].astype(float); cam = cloud['camera_id']; uv = cloud['pixel_uv']
    c = np.median(selector_xyz[:, :2], axis=0)
    pivot = np.asarray(pivot, float)[:2]
    d = np.asarray(push_dir, float)[:2]; d /= np.linalg.norm(d)
    lat = np.array([-d[1], d[0]])
    # Footprint from the base band only: an overhang (tipT top bar) is not where the object
    # meets the table, and would push the table ring out of the recorded scene.
    hs = selector_xyz[:, 2] - TABLE_Z_REF
    base = selector_xyz[(hs > .020) & (hs < .045)]
    base = base if len(base) >= 50 else selector_xyz
    footprint = float(np.quantile(np.linalg.norm(base[:, :2] - c, axis=1), .95))
    views = []
    for view in meta['views']:
        cid = view['camera_id']
        if view.get('status') != 'recorded':
            views.append(dict(camera=view['camera'], status='missing')); continue
        sel = cam == cid
        clean = edge_free(folder, int(cid), uv[sel])
        p = xyz[sel][clean]
        table = fit_table(p, c, footprint + .015, footprint + .060)
        if table is None:
            views.append(dict(camera=view['camera'], status='insufficient table pixels')); continue
        position = camera_position(view)
        toward = position[:2] - c; horizontal = np.linalg.norm(toward); toward /= horizontal
        elevation = float(np.degrees(np.arctan2(position[2] - TABLE_Z_REF, horizontal)))
        h = height_above(p, c, table['plane'])
        u = (p[:, :2] - c) @ d; w = (p[:, :2] - pivot) @ lat
        strip = (np.abs(w) < LATERAL_BAND_M) & (u > 0) & (u < footprint + .015)
        h_min = max(.002, 3 * table['sigma_m'])
        bins = []
        for lo in np.arange(0, PROFILE_TOP, BIN_M):
            q = strip & (h >= lo) & (h < lo + BIN_M)
            if q.sum() >= 5:
                uq = u[q]; top = uq[uq >= np.quantile(uq, .8)]
                bins.append((lo + BIN_M / 2, float(np.quantile(uq, ENVELOPE_Q)),
                             1.4826 * float(np.median(np.abs(top - np.median(top)))), int(len(top))))
        entry = dict(camera=view['camera'], camera_id=int(cid), azimuth_toward_camera=toward.tolist(),
                     elevation_deg=elevation, table=table, table_height_floor_m=float(h_min),
                     facing=float(np.dot(toward, d)))
        if len([b for b in bins if b[0] > .025]) < 5:
            views.append(dict(entry, status='wall bins insufficient')); continue
        fits = {}
        for cut in CUTOFFS_M:
            B = np.array([b for b in bins if b[0] >= max(h_min, cut)])
            hB, rB = B[:, 0], B[:, 1]
            sB = np.sqrt(sys_floor_of(bins) ** 2 + (B[:, 2] / np.sqrt(B[:, 3])) ** 2)
            chi2, nuisance = chi2_profile(hB, rB, sB)
            dof = max(len(hB) - 3, 1)
            birge = max(1.0, float(chi2.min() / dof))
            chi2 = chi2 / birge
            j = int(np.argmin(chi2))
            fits[f'{cut * 1000:.0f}mm'] = dict(cutoff_m=float(max(h_min, cut)), best_R_m=float(R_GRID[j]),
                R_68_m=interval(chi2, 1.0), R_95_m=interval(chi2, 3.84), reduced_chi2=birge,
                outer_radius_m=float(nuisance[j][0]), wall_lean_mm_per_10mm=float(10 * nuisance[j][1]),
                chi2_profile=chi2.tolist(), bins_used=int(len(hB)))
        views.append(dict(entry, status='ok', fits=fits, best_R_m=fits['5mm']['best_R_m'],
                          outer_radius_m=fits['5mm']['outer_radius_m'],
                          R_95_envelope_m=[min(f['R_95_m'][0] for f in fits.values()),
                                           max(f['R_95_m'][1] for f in fits.values())],
                          bins=[dict(h=b[0], rho=b[1], mad=b[2], n=b[3]) for b in bins]))
    return dict(center_xy=c.tolist(), support_dir=d.tolist(), footprint_m=footprint, views=views,
                R_grid_m=R_GRID.tolist(), profile='envelope')
