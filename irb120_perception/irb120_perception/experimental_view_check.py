"""Temporary MVP: filter press candidates using visible depth views before scoring.

Existing scores, normals, and motion directions are unchanged. Remove this file
and the snapshot hook to discard the experiment.
"""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np


TABLE_Z = -.021


def estimate_table_plane(depth, info, table_z=TABLE_Z, stride=4):
    """Fit the camera's observed table as z=ax+by+c in the base frame."""
    if 'K' not in info or 'base_to_optical' not in info:
        return None
    K = np.asarray(info['K']); tf = info['base_to_optical']
    R, t = np.asarray(tf['R']), np.asarray(tf['t'])
    z = np.asarray(depth)[::stride, ::stride].astype(float) / 1000
    v, u = np.indices(z.shape); u *= stride; v *= stride
    optical = np.stack(((u-K[0, 2])*z/K[0, 0],
                        (v-K[1, 2])*z/K[1, 1], z), axis=-1)
    base = (optical-t) @ R
    q = base[(z > 0) & np.isfinite(base).all(axis=-1)
             & (base[:, :, 2] >= table_z-.05)
             & (base[:, :, 2] <= table_z+.05)]
    if len(q) < 200:
        return None
    for _ in range(5):
        A = np.c_[q[:, :2], np.ones(len(q))]
        coef = np.linalg.lstsq(A, q[:, 2], rcond=None)[0]
        residual = q[:, 2]-A@coef
        median = np.median(residual)
        mad = np.median(np.abs(residual-median))+1e-12
        keep = np.abs(residual-median) < max(.003, 3*1.4826*mad)
        if keep.all() or keep.sum() < 200:
            break
        q = q[keep]
    return np.linalg.lstsq(np.c_[q[:, :2], np.ones(len(q))], q[:, 2], rcond=None)[0]


def check_view(point, direction, depth, info, table_plane=None, table_z=TABLE_Z):
    """Check one aligned uint16-mm depth image; invalid/occluded views abstain."""
    K = np.asarray(info['K']); tf = info['base_to_optical']
    R, t = np.asarray(tf['R']), np.asarray(tf['t'])
    q = R @ point + t
    if q[2] <= 0 or np.any(np.abs(info.get('D', [])) > 1e-8):
        return dict(usable=False, reason='behind camera or unsupported distorted depth')
    uv = np.rint((K @ q)[:2] / q[2]).astype(int)
    if not (0 <= uv[0] < depth.shape[1] and 0 <= uv[1] < depth.shape[0]):
        return dict(usable=False, reason='outside image')
    measured = float(depth[uv[1], uv[0]]) / 1000.
    if measured <= 0:
        return dict(usable=False, reason='missing depth or depth mismatch/occlusion')
    table_offset = 0.
    if table_plane is not None:
        table_offset = float(point[0]*table_plane[0]
                             + point[1]*table_plane[1]+table_plane[2]-table_z)
    # A base-frame vertical correction changes optical depth by R[2,2]
    # times that amount. Apply it before expensive neighborhood recovery so
    # occluded and mismatched views keep the original fast exit.
    residual = float(measured-q[2]-R[2, 2]*table_offset)
    if abs(residual) > .005:
        return dict(usable=False, reason='missing depth or depth mismatch/occlusion',
                    depth_residual_mm=float(residual * 1000),
                    table_offset_mm=table_offset*1000)
    v, u = np.indices(depth.shape); z = depth.astype(float) / 1000.
    optical = np.stack([(u-K[0, 2])*z/K[0, 0], (v-K[1, 2])*z/K[1, 1], z], axis=-1)
    base = (optical-t) @ R
    if table_plane is not None:
        # Remove the same local Z error seen on the known-flat table. This
        # cancels camera depth/extrinsic bias without assuming object geometry.
        # The validation neighborhood is only 12 mm wide, so the plane's
        # offset variation within it is negligible; one local correction is
        # both clearer and much cheaper than correcting the entire image.
        base[:, :, 2] -= table_offset
    distances = np.linalg.norm(base-point, axis=-1)
    mask = ((depth > 0) & (distances <= .012)
            & (np.abs(z-measured) <= .008)).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask, connectivity=4)
    component = labels[uv[1], uv[0]]
    if component == 0:
        return dict(usable=False, reason='no connected neighborhood at projected pixel')
    connected = labels == component
    local = base[connected]
    if len(local) < 20:
        return dict(usable=False, reason='insufficient local support', count=len(local))
    centered = local-local.mean(0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    if values[1] <= 1e-12 or values[0]/values.sum() > .05:
        return dict(usable=False, reason='nonplanar/degenerate neighborhood', count=len(local))
    n = vectors[:, 0]; toward_camera = -R.T @ t-point
    toward_camera /= np.linalg.norm(toward_camera)
    if n @ toward_camera < 0:
        n = -n
    if n @ toward_camera < .3:
        return dict(usable=False, reason='grazing view')
    small = base[connected & (distances <= .008)]
    if len(small) < 20:
        return dict(usable=False, reason='insufficient 8 mm support')
    small = small-small.mean(0)
    sv, vectors = np.linalg.eigh(small.T @ small)
    sn = vectors[:, 0]
    if sn @ toward_camera < 0:
        sn = -sn
    if (sv[1] <= 1e-12 or sv[0]/sv.sum() > .05
            or sn @ n < np.cos(np.radians(20))
            or (sn @ direction > 0) != (n @ direction > 0)):
        return dict(usable=False, reason='normal unstable across 8/12 mm scales')
    return dict(usable=True, normal=n.tolist(), normal_dot_pull=float(n @ direction),
                count=len(local), depth_residual_mm=float(residual*1000),
                table_offset_mm=table_offset*1000)


def decide(views):
    usable = [v for v in views.values() if v['usable']]
    if not usable:
        return 'uncertain', 'no usable camera view'
    for i, a in enumerate(usable):
        for b in usable[i+1:]:
            if np.dot(a['normal'], b['normal']) < np.cos(np.radians(25)):
                return 'uncertain', 'camera normals disagree by more than 25 degrees'
    if any(v['normal_dot_pull'] > 0 for v in usable):
        return 'reject', 'visible normal points along proposed pull'
    return 'pass', 'single-view support' if len(usable) == 1 else 'multi-view agreement'


def evaluate(snapshot, contacts, *, cameras=None, rule='top_support_v1', render_overlays=True):
    """Return an experimental contacts copy and evidence, preserving baseline."""
    snapshot = Path(snapshot); result = copy.deepcopy(contacts)
    from irb120_perception.contact_point_selector import select_contact_points
    if rule not in ('view_check_v2', 'top_support_v1'):
        raise ValueError('Unknown experimental press rule: '+rule)
    report = dict(experiment='experimental_view_check_v2', candidates=[],
                  thresholds=dict(depth_agreement_mm=5, radii_mm=[8,12],
                                  connected_depth_band_mm=8, min_samples=20,
                                  max_surface_variation=.05, min_view_cosine=.3,
                                  max_normal_angle_deg=25, max_scale_angle_deg=20,
                                  max_normal_dot_pull=0))
    load_saved = cameras is None
    cameras = {} if load_saved else cameras
    for cam in (('cam1', 'cam2', 'cam3') if load_saved else ()):
        try:
            info = json.loads((snapshot/f'{cam}_camera_info.json').read_text())
            depth = cv2.imread(str(snapshot/f'{cam}_depth.png'), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
                raise ValueError('missing or non-uint16 depth')
            cameras[cam] = (depth, info)
        except (OSError, KeyError, ValueError):
            continue
    table_planes = {cam: estimate_table_plane(depth, info)
                    for cam, (depth, info) in cameras.items()}
    report['camera_table_planes'] = {
        cam: (plane.tolist() if plane is not None else None)
        for cam, plane in table_planes.items()
    }

    xyz = np.load(snapshot/'object_cloud.npz')['xyz']
    inferred_tree = None
    if rule == 'top_support_v1':
        from scipy.spatial import cKDTree
        from .contact_point_selector import estimate_normals, estimate_pivot
        from .experimental_top_support import (
            assess, has_upward_upper_surface, infer_horizontal_top,
            tipping_torque_scores,
        )
        observed = np.unique(xyz[np.isfinite(xyz).all(axis=1)], axis=0)
        clean = observed
        normals = estimate_normals(observed)
        inferred = np.empty((0, 3))
        inference = dict(available=False, reason='observed upward-facing candidates exist')
        if not has_upward_upper_surface(observed, normals):
            inferred, inference = infer_horizontal_top(observed)
            if len(inferred):
                clean = np.vstack((observed, inferred))
                normals = np.vstack((normals, np.tile([0., 0., 1.], (len(inferred), 1))))
        report['horizontal_top_inference'] = inference
        tree = cKDTree(clean)
        inferred_tree = cKDTree(inferred) if len(inferred) else None
        geometry = estimate_pivot(clean, [-1., 0., 0.], -.021)
        report['experiment'] = 'experimental_local_support_v4'
        report['geometry'] = json.loads(json.dumps(geometry, default=lambda value: value.tolist()))
        report['thresholds'].update(max_top_slope_deg=30, max_camera_fused_angle_deg=25,
            max_normal_dot_pull=float(np.sin(np.radians(15))), local_support_radius_mm=12,
            downward_to_pull_force_ratio=1, minimum_torque_band_m=0.003,
            relative_torque_band=0.05,
            ranking='near-maximum 3-D wrench torque, then pivot-edge centering')

    def filter_candidates(points, direction):
        keep = []
        for point in points:
            views = {}
            for cam, (depth, info) in cameras.items():
                try:
                    views[cam] = check_view(
                        point, direction, depth, info,
                        table_plane=table_planes.get(cam))
                except (KeyError, ValueError) as exc:
                    views[cam] = dict(usable=False, reason=str(exc))
            is_inferred = bool(inferred_tree is not None and inferred_tree.query(point)[0] < 1e-8)
            decision, reason = (assess(point, normals[tree.query(point)[1]], observed, views,
                                       require_local_support=not is_inferred)
                                if rule == 'top_support_v1' else decide(views))
            report['candidates'].append(dict(point=point.tolist(), views=views,
                direction=direction.tolist(), decision=decision, reason=reason,
                inferred_horizontal_top=is_inferred))
            keep.append(decision == 'pass')
        if rule == 'top_support_v1':
            scores = tipping_torque_scores(points, geometry)
            for row, score in zip(report['candidates'][-len(points):], scores):
                row['torque_per_pull_N_m'] = float(score)
            # Do not collapse to the exact maximum here.  The main selector
            # applies its physically meaningful torque band and then centers
            # the survivors along the detected pivot edge.
            return np.asarray(keep, dtype=bool)
        return np.asarray(keep, dtype=bool)

    selection_points = clean if rule == 'top_support_v1' else xyz
    selection_normals = normals if rule == 'top_support_v1' else None
    selected = select_contact_points(selection_points, normals=selection_normals,
                                    min_ball_center_z=.06,
                                    press_score_fraction=.05,
                                    experimental_press_filter=filter_candidates)
    result['press'] = selected['press']
    report['decision'] = 'pass' if result['press']['available'] else 'unavailable'
    report['counts'] = {key: sum(c['decision'] == key for c in report['candidates'])
                        for key in ('pass', 'reject', 'uncertain')}
    report['selected_point'] = result['press'].get('point')
    if report['selected_point'] is not None:
        report['selected_point'] = report['selected_point'].tolist()
    # JSON-friendly output, including newly selected numpy geometry.
    result = json.loads(json.dumps(result, default=lambda value: value.tolist()))
    result['press']['experimental_view_check'] = dict(
        experiment=report['experiment'], decision=report['decision'], counts=report['counts'])
    if rule == 'top_support_v1' and render_overlays:
        from .press_review import write_review
        write_review(snapshot, report, clean, normals, cameras)
    for cam, (_, info) in cameras.items():
        if not render_overlays or not (snapshot/f'{cam}_color.png').exists():
            continue
        img = cv2.imread(str(snapshot/f'{cam}_color.png'))
        if img is None or 'base_to_optical' not in info:
            continue
        tf = info['base_to_optical']; R = np.asarray(tf['R']); t = np.asarray(tf['t'])
        K = np.asarray(info['K'])
        for candidate in report['candidates']:
            q = R @ np.asarray(candidate['point']) + t
            if q[2] <= 0:
                continue
            uv, _ = cv2.projectPoints(q.reshape(1,3), np.zeros(3), np.zeros(3),
                                     K, np.asarray(info.get('D', [])))
            pixel = tuple(np.rint(uv.reshape(2)).astype(int))
            color = {'pass': (0,255,0), 'reject': (0,0,255), 'uncertain': (0,200,255)}[candidate['decision']]
            cv2.circle(img, pixel, 3, color, -1)
            if report['selected_point'] == candidate['point']:
                cv2.circle(img, pixel, 8, (255,0,255), 2)
        cv2.putText(img, 'v2: green pass / red reject / yellow uncertain / magenta selected',
                    (8,20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
        cv2.imwrite(str(snapshot/f'{cam}_experimental_view_check.png'), img)
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--rule', choices=['view_check_v2', 'top_support_v1'], default='top_support_v1')
    args = parser.parse_args()
    baseline = args.snapshot/'contacts_baseline.json'
    if not baseline.exists():
        baseline = args.snapshot/'contacts.json'
    result, report = evaluate(args.snapshot, json.loads(baseline.read_text()), rule=args.rule)
    (args.snapshot/'contacts_experimental_view_check.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.snapshot/'experimental_view_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
