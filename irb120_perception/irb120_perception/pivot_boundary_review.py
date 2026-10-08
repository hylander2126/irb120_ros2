"""Offline experimental lower-mask boundary review; never changes contacts.

Run with python -m irb120_perception.pivot_boundary_review SNAPSHOT.
Requires the saved SAM mask; older episodes cannot reconstruct it from a cloud.
"""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from irb120_perception.contact_point_selector import select_contact_points
from irb120_perception.depth_plane_audit import audit


def contacts_with_candidate_pivot(contacts, report):
    """Return a copy using a reviewable RGB pivot, or fail closed.

    The press point, ball center, normal, direction, and controller axis remain
    exactly as selected from the production cloud.  The current controller
    executes an XZ arc and consumes only geometry.pivot from this addition.
    """
    if report.get('candidate_status') != 'reviewable':
        raise ValueError(f"RGB pivot is not reviewable: {report.get('fallback_reason')}")
    pivot = (report.get('geometry') or {}).get('candidate_pivot_base_m')
    if pivot is None or len(pivot) != 3 or not np.isfinite(pivot).all():
        raise ValueError('RGB pivot candidate is missing or invalid')
    result = copy.deepcopy(contacts)
    press = result.get('press') or {}
    geometry = press.get('geometry') or {}
    if not press.get('available') or geometry.get('pivot') is None:
        raise ValueError('production press contact/pivot is unavailable')
    old = copy.deepcopy(geometry['pivot'])
    geometry['pivot'] = [float(v) for v in pivot]
    press['geometry'] = geometry
    press['rgb_pivot'] = {
        'applied': True,
        'source': report.get('source'),
        'replaced_cloud_pivot_base_m': old,
        'support_type': (report.get('evidence') or {}).get('support_fit', {}).get('support_type'),
    }
    result['press'] = press
    return result


def intersect_table(pixels, info, plane):
    """Intersect calibrated optical rays with z=a*x+b*y+c in base coordinates."""
    K = np.asarray(info['K'], float)
    tf = info['base_to_optical']
    R, t = np.asarray(tf['R']), np.asarray(tf['t'])
    uv = cv2.undistortPoints(np.asarray(pixels, float).reshape(-1, 1, 2),
                             K, np.asarray(info.get('D', []), float)).reshape(-1, 2)
    rays = np.c_[uv, np.ones(len(uv))] @ R
    origin = -t @ R
    normal = np.array([-plane[0], -plane[1], 1.])
    denominator = rays @ normal
    distance = np.divide(plane[2] - origin @ normal, denominator,
                         out=np.full(len(rays), np.nan), where=np.abs(denominator) > 1e-8)
    valid = np.isfinite(distance) & (distance > 0)
    return origin + distance[:, None] * rays, valid


def lower_boundary(mask, cloud, info):
    """Choose the component with the most projected object-cloud votes."""
    _, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    tf = info['base_to_optical']
    cam = cloud @ np.asarray(tf['R']).T + np.asarray(tf['t'])
    cam = cam[np.isfinite(cam).all(axis=1) & (cam[:, 2] > .01)]
    if not len(cam):
        raise ValueError('object cloud is not visible in Cam1')
    uv, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3),
                              np.asarray(info['K'], float), np.asarray(info.get('D', []), float))
    uv = np.rint(uv.reshape(-1, 2)).astype(int)
    h, w = mask.shape
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
    votes = np.bincount(labels[uv[:, 1], uv[:, 0]], minlength=labels.max()+1)
    votes[0] = 0
    if votes.max() < 3:
        raise ValueError('no mask component associated with object cloud')
    component = labels == votes.argmax()
    u = np.flatnonzero(component.any(axis=0))
    v = h - 1 - np.argmax(component[::-1, u], axis=0)
    return np.c_[u, v], int(votes.max())


def boundary_depth_evidence(pixels, depth, info, plane):
    """Probe 3--5 pixels inside/outside; require object above and table below."""
    K = np.asarray(info['K']); tf = info['base_to_optical']
    R, t = np.asarray(tf['R']), np.asarray(tf['t'])
    heights = []
    counts = []
    for offsets in ((-5, -4, -3), (3, 4, 5)):
        samples = []
        for offset in offsets:
            u = pixels[:, 0].astype(int)
            v = pixels[:, 1].astype(int) + offset
            inside = (v >= 0) & (v < depth.shape[0])
            z = depth[np.clip(v, 0, depth.shape[0]-1), u].astype(float)/1000
            optical = np.c_[(u-K[0, 2])*z/K[0, 0], (v-K[1, 2])*z/K[1, 1], z]
            base = (optical-t) @ R
            h = base[:, 2] - base[:, :2] @ np.asarray(plane[:2]) - plane[2]
            samples.append(np.where(inside & (z > 0) & np.isfinite(z), h, np.nan))
        samples = np.asarray(samples).T
        counts.append(np.isfinite(samples).sum(axis=1))
        # Avoid all-NaN median warnings; invalid rows are gated below.
        samples[counts[-1] == 0] = 0
        heights.append(np.nanmedian(samples, axis=1))
    valid = (counts[0] >= 2) & (counts[1] >= 2)
    accepted = valid & (heights[0] >= .002) & (heights[0] <= .035) & (np.abs(heights[1]) <= .003)
    return accepted, dict(inside_height_mm=(heights[0]*1000).tolist(),
                          outside_height_mm=(heights[1]*1000).tolist(),
                          valid_neighborhood=valid.tolist())


def fit_support(points, plane):
    """Robust local front support x(y), with a line or observed convex tangent.

    Only uses metric boundary samples. No object identities or nominal radii.
    A quadratic is a local tangent model, not a claim that the object is circular.
    """
    q = np.asarray(points, float)
    if len(q) < 12 or np.ptp(q[:, 1]) < .015:
        raise ValueError('insufficient connected support span')
    # Remote side lobes are not the -X tipping support. Keep a 15 mm front band.
    q = q[q[:, 0] <= np.quantile(q[:, 0], .10)+.015]
    if len(q) < 12 or np.ptp(q[:, 1]) < .015:
        raise ValueError('insufficient front support span')
    center = np.median(q[:, 1])
    y = q[:, 1] - center

    def fit(degree):
        A = np.vander(y, degree+1, increasing=True)
        keep = np.ones(len(q), bool)
        for _ in range(5):
            coef = np.linalg.lstsq(A[keep], q[keep, 0], rcond=None)[0]
            residual = q[:, 0] - A @ coef
            median = np.median(residual[keep])
            mad = np.median(np.abs(residual[keep]-median))
            new = np.abs(residual-median) <= max(.0015, 3*1.4826*mad)
            if new.sum() < 10 or np.array_equal(new, keep):
                break
            keep = new
        coef = np.linalg.lstsq(A[keep], q[keep, 0], rcond=None)[0]
        return coef, keep, float(np.sqrt(np.mean((q[keep, 0]-A[keep]@coef)**2)))

    line, line_keep, line_rms = fit(1)
    curve, curve_keep, curve_rms = fit(2)
    ymin, ymax = np.quantile(y, [.05, .95])
    sag = curve[2] * (ymax-ymin)**2/4
    vertex = -curve[1]/(2*curve[2]) if curve[2] > 0 else np.inf
    rounded = (line_rms > .0018 and sag > .003 and curve_rms < .7*line_rms
               and ymin <= vertex <= ymax)
    if rounded:
        yp = vertex + center
        xp = float(curve[0]+curve[1]*vertex+curve[2]*vertex**2)
        slope = 0.
        kind, rms, keep = 'rounded_tangent', curve_rms, curve_keep
    else:
        if line_rms > .0025:
            raise ValueError('support is neither a stable line nor an observed convex tangent')
        yp = float(np.median(q[line_keep, 1]))
        xp = float(line[0]+line[1]*(yp-center))
        slope = float(line[1])
        kind, rms, keep = 'flat_edge', line_rms, line_keep
    axis = np.array([slope, 1., plane[0]*slope+plane[1]])
    axis /= np.linalg.norm(axis)
    pivot = np.array([xp, yp, plane[0]*xp+plane[1]*yp+plane[2]])
    halfspan = min(.04, float(np.ptp(q[keep, 1]))/2)
    return dict(pivot=pivot.tolist(), axis=axis.tolist(),
                edge=[(pivot-halfspan*axis).tolist(), (pivot+halfspan*axis).tolist()],
                support_type=kind, rms_mm=rms*1000, samples=int(keep.sum()),
                span_mm=float(np.ptp(q[keep, 1])*1000),
                line_rms_mm=line_rms*1000, quadratic_rms_mm=curve_rms*1000)


def project(points, info):
    tf = info['base_to_optical']
    cam = np.atleast_2d(points) @ np.asarray(tf['R']).T + np.asarray(tf['t'])
    uv, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3),
                              np.asarray(info['K'], float), np.asarray(info.get('D', []), float))
    return np.rint(uv.reshape(-1, 2)).astype(int)


def review_overlay(color, pixels, accepted, info, report):
    """Full frame plus magnified support, baseline and candidate in one artifact.

    Delineates the two pivot selectors (cloud vs. SAM/RGB candidate) and each
    one's corresponding press contact (always chosen by the same pointcloud
    heuristic in contact_point_selector; only the pivot it is scored against
    differs), then labels whichever one is this report's final choice.
    """
    canvas = color.copy()
    for pixel, keep in zip(pixels, accepted):
        cv2.circle(canvas, tuple(np.rint(pixel).astype(int)), 1,
                   (0, 220, 0) if keep else (0, 0, 255), -1)
    baseline = report['physics']['selected_pivot_base_m']
    if baseline is not None:
        cv2.drawMarker(canvas, tuple(project(baseline, info)[0]), (255, 0, 255),
                       cv2.MARKER_CROSS, 7, 1)
    geometry = report['geometry']
    if 'candidate_pivot_base_m' in geometry:
        pivot = project(geometry['candidate_pivot_base_m'], info)[0]
        edge = project(geometry['pivot_edge_base_m'], info)
        cv2.line(canvas, tuple(edge[0]), tuple(edge[1]), (255, 220, 0), 1)
        cv2.drawMarker(canvas, tuple(pivot), (255, 220, 0), cv2.MARKER_TILTED_CROSS, 7, 1)
    comparison = report.get('press_comparison') or {}
    cloud_press = (comparison.get('cloud_pivot') or {}).get('press')
    if cloud_press is not None:
        cv2.drawMarker(canvas, tuple(project(cloud_press['point'], info)[0]), (255, 0, 255),
                       cv2.MARKER_SQUARE, 10, 2)
    sam_press = (comparison.get('sam_pivot') or {}).get('press')
    if sam_press is not None:
        cv2.drawMarker(canvas, tuple(project(sam_press['point'], info)[0]), (0, 255, 0),
                       cv2.MARKER_STAR, 12, 2)
    h, w = canvas.shape[:2]
    if len(pixels):
        low = np.maximum(np.floor(pixels.min(axis=0)).astype(int)-[20, 25], 0)
        high = np.minimum(np.ceil(pixels.max(axis=0)).astype(int)+[20, 35], [w, h])
        # Center crop around accepted support rather than rejected upper side lobes.
        if accepted.any():
            low[1] = max(0, int(pixels[accepted, 1].min())-20)
        crop = canvas[low[1]:high[1], low[0]:high[0]]
        if crop.size:
            crop = cv2.resize(crop, (w, max(80, min(240, int(crop.shape[0]*w/crop.shape[1])))))
            canvas = np.vstack([canvas, crop])
    final_choice = comparison.get('final_choice')
    if final_choice == 'sam_pivot':
        final_line = 'VALIDATED PROPOSAL: green star, SAM pivot + its press point'
    elif final_choice == 'cloud_pivot' and cloud_press is not None:
        final_line = ('VALIDATED PROPOSAL: magenta square, cloud pivot + its press point '
                       f'(SAM candidate unusable: {comparison.get("sam_pivot_error") or "see status line"})')
    else:
        final_line = 'NO VALIDATED PRESS SELECTED (see candidate status)'
    banner = np.full((135, w, 3), 24, np.uint8)
    lines = ['OFFLINE ONLY | green dots: screened | red dots: rejected',
             'cyan: SAM/RGB candidate tangent+pivot | magenta cross: cloud pivot',
             'magenta square: press point via cloud pivot | green star: press point via SAM pivot',
             final_line,
             report.get('candidate_status', 'unavailable') + ': ' + report['fallback_reason']]
    for i, line in enumerate(lines):
        cv2.putText(banner, line[:110], (8, 22+i*25), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([banner, canvas])


def evaluate(snapshot, *, snapshot_record=None, cameras=None, rule='top_support_v1', sam_images=None):
    snapshot = Path(snapshot)
    contacts = json.loads((snapshot/'contacts.json').read_text())
    baseline = ((contacts.get('press') or {}).get('geometry') or {}).get('pivot')
    report = dict(source='cam1_mask_table_intersection', experimental=True,
                  image='cam1_sam_color.png', geometry={},
                  physics=dict(selected_pivot_base_m=baseline, support_type='cloud_fallback', confidence=0.),
                  evidence={}, candidate_status='unavailable',
                  fallback_reason='offline boundary review only; support fitting not validated')
    color = (sam_images[1] if sam_images is not None else
             cv2.imread(str(snapshot/'cam1_sam_color.png')) if (snapshot/'cam1_sam_color.png').exists() else None)
    if sam_images is not None:
        report['image'] = 'in_memory_sam_rgb'
    try:
        mask_path = snapshot/'cam1_sam_mask.png'
        if sam_images is None and not mask_path.exists():
            raise ValueError('missing saved SAM mask; collect a new snapshot')
        mask = sam_images[0] if sam_images is not None else cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or color is None or mask.shape != color.shape[:2]:
            raise ValueError('missing or incompatible SAM mask/color')
        info = json.loads((snapshot/'cam1_camera_info.json').read_text())
        if mask.shape != (info['height'], info['width']):
            raise ValueError('mask and calibration dimensions differ')
        # The existing depth plane fitter assumes rectified aligned depth.
        if np.any(np.abs(info.get('D', [])) > 1e-8):
            raise ValueError('distorted aligned depth is not supported by table fitter')
        cloud = np.load(snapshot/'object_cloud.npz')['xyz']
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        pixels, votes = lower_boundary(mask, cloud, info)
        table = audit(snapshot, camera='cam1')['table']
        report['evidence']['table_fit'] = table
        if not table['available'] or table['p95_abs_mm'] > 6:
            raise ValueError('insufficient or noisy measured table plane')
        plane = table['equation_z_ax_by_c']
        # Pixel cells meet halfway below the last foreground pixel.
        boundary = pixels.astype(float) + [0., .5]
        points, valid = intersect_table(boundary, info, plane)
        distances = np.full(len(points), np.inf)
        distances[valid] = cKDTree(cloud[:, :2]).query(points[valid, :2])[0]
        depth_path = (snapshot/'cam1_sam_depth.npz'
                      if (snapshot/'cam1_sam_depth.npz').exists()
                      else snapshot/'cam1_aligned_depth.npz')
        depth = np.load(depth_path)['depth_mm']
        if depth.shape != mask.shape:
            raise ValueError('depth and mask dimensions differ')
        depth_ok, depth_evidence = boundary_depth_evidence(pixels, depth, info, plane)
        accepted = valid & (distances <= .02) & depth_ok & (pixels[:, 1] < mask.shape[0]-6)
        report.update(image_width=mask.shape[1], image_height=mask.shape[0])
        report['geometry'] = dict(object_table_boundary_px=boundary[accepted].tolist(),
                                  rejected_boundary_px=boundary[~accepted].tolist(),
                                  support_samples_base_m=points[accepted].tolist())
        report['evidence'].update(component_cloud_votes=votes, boundary_count=len(pixels),
                                  accepted_count=int(accepted.sum()), envelope_threshold_m=.02,
                                  depth_boundary_check=depth_evidence)
        provenance_path = snapshot/'replay_provenance.json'
        if not provenance_path.exists():
            provenance_path = snapshot/'cam1_sam_metadata.json'
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        report['evidence']['provenance'] = provenance
        try:
            # Do not bridge disconnected contour runs into an invented edge.
            ids = np.flatnonzero(accepted)
            runs = np.split(ids, np.flatnonzero((np.diff(pixels[ids, 0]) > 2)
                            | (np.abs(np.diff(pixels[ids, 1])) > 3))+1)
            runs = [run for run in runs if len(run) >= 12]
            if not runs:
                raise ValueError('no connected support run of at least 12 pixels')
            run = min(runs, key=lambda r: np.quantile(points[r, 0], .1))
            fit = fit_support(points[run], plane)
            pivot = np.asarray(fit['pivot'])
            report['geometry'].update(candidate_pivot_base_m=fit['pivot'],
                                      pivot_edge_base_m=fit['edge'], tangent_base=fit['axis'])
            report['evidence']['support_fit'] = fit
            # Sensitivity, not a calibrated error bar: perturb each input independently.
            pixel_shift, _ = intersect_table(boundary[run]+[0, 1], info, plane)
            raised_plane = np.asarray(plane)+[0, 0, .001]
            height_shift, _ = intersect_table(boundary[run], info, raised_plane)
            pixel_sensitivity = float(np.median(np.abs(pixel_shift[:, 0]-points[run, 0]))*1000)
            plane_sensitivity = float(np.median(np.abs(height_shift[:, 0]-points[run, 0]))*1000)
            delta = None if baseline is None else (pivot-np.asarray(baseline))*1000
            report['evidence'].update(pivot_delta_from_cloud_mm=None if delta is None else delta.tolist(),
                                      x_shift_per_pixel_mm=pixel_sensitivity,
                                      x_shift_per_mm_table_height_mm=plane_sensitivity)
            reasons = []
            # The object is stationary throughout the perception window. Keep
            # acquisition timing as evidence, not a rejection criterion.
            report['evidence']['timing_policy'] = 'known_static_perception_window'
            # No RGB/cloud disagreement gate: on rounded supports the cloud pivot is
            # the one historically shown to be off by 20-32 mm while RGB stays within
            # a few mm, so a large disagreement is more often evidence against cloud
            # than against the candidate. Every applied candidate is still reviewed
            # visually via final_selection.jpg before trusting the next run.
            if plane_sensitivity*table['p95_abs_mm'] > 10:
                reasons.append('table residual amplified beyond 10 mm in X')
            report['candidate_status'] = 'rejected' if reasons else 'reviewable'
            report['fallback_reason'] = '; '.join(reasons) if reasons else 'candidate needs independent ground truth validation'
        except ValueError as exc:
            report['candidate_status'] = 'rejected'
            report['fallback_reason'] = str(exc)
        # Both pivots feed the SAME press-point heuristic in contact_point_selector;
        # only the pivot each is scored/edge-centered against differs. Recomputing
        # the SAM-pivot press here (rather than reusing contacts_with_candidate_pivot's
        # pivot-only swap) shows the press point actually moving with the pivot.
        comparison = {}
        baseline_contact = contacts.get('press') or {}
        if baseline_contact.get('available') and baseline_contact.get('point') is not None:
            comparison['cloud_pivot'] = dict(
                pivot=baseline,
                press=dict(point=baseline_contact['point'], ball_center=baseline_contact.get('ball_center')))
        if 'candidate_pivot_base_m' in report['geometry']:
            try:
                snapshot_meta = snapshot/'snapshot.json'
                meta = (snapshot_record if snapshot_record is not None else
                        json.loads(snapshot_meta.read_text()) if snapshot_meta.exists() else {})
                min_ball_center_z = meta.get('min_ball_center_z')
                baseline_path = snapshot/'contacts_baseline.json'
                view_check_mode = (meta.get('experimental_view_check') or {}).get('mode')
                if view_check_mode not in (None, 'off') and baseline_path.exists():
                    # contacts.json's press came from this rule (observed + any
                    # inferred top-surface candidates, camera-view-check filtered),
                    # not the plain selector below -- reseed through the same path
                    # or an object whose only valid press needs those extra
                    # candidates (e.g. a rounded cylinder cap with no clean observed
                    # upward normal) spuriously reports no press at all.
                    from irb120_perception.experimental_view_check import evaluate as evaluate_view_check
                    comparison_cameras = {} if cameras is None else cameras
                    for cam in (('cam1', 'cam2', 'cam3') if cameras is None else ()):
                        info_path, depth_path = snapshot/f'{cam}_camera_info.json', snapshot/f'{cam}_aligned_depth.npz'
                        if info_path.exists() and depth_path.exists():
                            comparison_cameras[cam] = (np.load(depth_path)['depth_mm'], json.loads(info_path.read_text()))
                    reseeded_contacts, _ = evaluate_view_check(
                        snapshot, json.loads(baseline_path.read_text()), cameras=comparison_cameras,
                        rule=rule, render_overlays=False, press_pivot_override=pivot.tolist())
                    sam_press = reseeded_contacts.get('press') or {}
                else:
                    reseeded = select_contact_points(cloud, min_ball_center_z=min_ball_center_z,
                                                     press_pivot_override=pivot)
                    sam_press = reseeded.get('press') or {}
                    if sam_press.get('available'):
                        sam_press = dict(sam_press, point=sam_press['point'].tolist(),
                                         ball_center=sam_press['ball_center'].tolist())
                if not sam_press.get('available'):
                    raise ValueError(sam_press.get('reason') or 'press unavailable with SAM pivot')
                comparison['sam_pivot'] = dict(
                    pivot=fit['pivot'],
                    press=dict(point=sam_press['point'], ball_center=sam_press['ball_center']))
            except ValueError as exc:
                comparison['sam_pivot_error'] = str(exc)
        comparison['final_choice'] = ('sam_pivot' if report['candidate_status'] == 'reviewable'
                                      and 'sam_pivot' in comparison else 'cloud_pivot' if 'cloud_pivot' in comparison else 'unavailable')
        report['press_comparison'] = comparison
        color = review_overlay(color, boundary, accepted, info, report)
    except (ValueError, KeyError, OSError, cv2.error) as exc:
        report['fallback_reason'] = str(exc)
    if color is not None:
        if not cv2.imwrite(str(snapshot/'cam1_pivot_boundary_review.png'), color):
            raise OSError('could not write boundary overlay')
    (snapshot/'pivot_boundary_review.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.snapshot), indent=2))


if __name__ == '__main__':
    main()
