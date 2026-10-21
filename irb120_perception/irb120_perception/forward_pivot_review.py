"""Forward-tip pivot from the cam3 SAM mask, and whether that pivot is straight or curved.

The forward tip rotates the object about its far (+X) support edge, which only
cam3 sees (it looks back along -X from beyond the object). This is the cam1
press-pivot fit of pivot_boundary_review mirrored to that edge: lowest cam3
mask pixel per column -> table plane -> screened by depth and cloud distance ->
line or convex-quadratic fit, then pivot_shape(): 'straight' (an extended edge)
or 'curved' (a point-like contact).

Run offline with python -m irb120_perception.forward_pivot_review SNAPSHOT.
Needs cam3_sam_mask.png, which snapshots save when the culler publishes a
cam3 review mask; older episodes have none.
"""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from irb120_perception.depth_plane_audit import audit
from irb120_perception.pivot_boundary_review import (
    fit_sensitivity, fit_support, project, screened_boundary, support_run)

TABLE_Z = -.021  # configured table height in base_link (= experimental_view_check.TABLE_Z)
CAMERA = 'cam3'
# SAM-only fallback, when no edge pixel is within 2 cm of the object cloud: a large object
# whose base the cloud cannot reach (the monitor's thin plate: cam1 cannot see its far half,
# cam3 gets little depth on it). The mask then stands on its own if the cloud clearly voted
# for it, its edge is within SAM_ONLY_ENVELOPE_M of the cloud, and the edge fits one
# straight line; pixels with depth must still pass the depth check.
SAM_ONLY_ENVELOPE_M = .10
# Table-height error assumed for the X-sensitivity check. It was the table fit's p95 point
# residual (2.5-5.5 mm), which is per-point scatter, not the error of a ~900-point fit; with
# cam3's grazing view (3-4 mm of X per mm of height) that rejected nearly every object.
# 1.5 mm ~ cam3's calibration after the 2026-10-20 depth recalibration.
TABLE_HEIGHT_ERROR_MM = 1.5
MAX_X_ERROR_MM = 10.
SAM_ONLY_MIN_VOTES = 50
OUTWARD = 1.  # the forward pivot is the +X edge
# A straight pivot: the line's inliers span at least this much, and this fraction of
# the support band. fit_support's robust line trims a curved support's flanks as
# outliers, so a short chord of a curved base (cylinder_e, ~25 mm) can still pass as
# a low-rms line; straight edges seen so far span 52-92 mm at >= 0.9.
STRAIGHT_MIN_SPAN_MM = 40.
STRAIGHT_MIN_FRACTION = .8


def pivot_shape(fit):
    """('straight' | 'curved', reason) for a fit_support result."""
    if fit['support_type'] == 'rounded_tangent':
        return 'curved', (f"convex tangent (quadratic rms {fit['quadratic_rms_mm']:.1f} mm"
                          f" vs line {fit['line_rms_mm']:.1f} mm)")
    span, band = fit['line_span_mm'], fit['band_span_mm']
    if span < STRAIGHT_MIN_SPAN_MM:
        return 'curved', f'straight run {span:.0f} mm < {STRAIGHT_MIN_SPAN_MM:.0f} mm: point-like contact'
    if span < STRAIGHT_MIN_FRACTION*band:
        return 'curved', f'line covers {span:.0f} of {band:.0f} mm of the support band'
    return 'straight', f'straight run {span:.0f} mm, line rms {fit["line_rms_mm"]:.1f} mm'


def review_overlay(color, boundary, accepted, info, report):
    """Frame with screened/rejected boundary, cloud and SAM pivots, plus a magnified crop."""
    canvas = color.copy()
    for pixel, keep in zip(boundary, accepted):
        cv2.circle(canvas, tuple(np.rint(pixel).astype(int)), 1, (0, 220, 0) if keep else (0, 0, 255), -1)
    if report['cloud_pivot_base_m'] is not None:
        cv2.drawMarker(canvas, tuple(project(report['cloud_pivot_base_m'], info)[0]), (0, 140, 255),
                       cv2.MARKER_CROSS, 9, 1)
    geometry = report['geometry']
    if 'candidate_pivot_base_m' in geometry:
        edge = project(geometry['pivot_edge_base_m'], info)
        cv2.line(canvas, tuple(edge[0]), tuple(edge[1]), (255, 220, 0), 1)
        cv2.drawMarker(canvas, tuple(project(geometry['candidate_pivot_base_m'], info)[0]), (255, 220, 0),
                       cv2.MARKER_TILTED_CROSS, 9, 1)
    h, w = canvas.shape[:2]
    if len(boundary):
        low = np.maximum(np.floor(boundary.min(axis=0)).astype(int)-[20, 25], 0)
        high = np.minimum(np.ceil(boundary.max(axis=0)).astype(int)+[20, 35], [w, h])
        if accepted.any():
            low[1] = max(0, int(boundary[accepted, 1].min())-20)
        crop = canvas[low[1]:high[1], low[0]:high[0]]
        if crop.size:
            crop = cv2.resize(crop, (w, max(80, min(240, int(crop.shape[0]*w/crop.shape[1])))))
            canvas = np.vstack([canvas, crop])
    fit = report['evidence'].get('support_fit') or {}
    lines = [f'{CAMERA} FORWARD-TIP (+X) PIVOT | green dots: screened | red dots: rejected',
             'cyan: SAM/RGB tangent+pivot | orange cross: cloud pivot',
             f"pivot shape: {report['shape'].upper()}"
             + (f" ({fit['support_type']}: {report['shape_reason']})" if report.get('shape_reason') else ''),
             f"{report['candidate_status']}: {report['fallback_reason']}"]
    banner = np.full((22+25*len(lines), w, 3), 24, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(banner, line[:110], (8, 22+i*25), cv2.FONT_HERSHEY_SIMPLEX, .42, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([banner, canvas])


def evaluate(snapshot, contacts=None):
    """Fit the forward pivot from the snapshot's saved cam3 mask; writes
    forward_pivot_review.json and cam3_forward_pivot_review.png. Never changes contacts."""
    snapshot = Path(snapshot)
    if contacts is None:
        contacts = json.loads((snapshot/'contacts.json').read_text())
    baseline = ((contacts.get('forward_tip') or {}).get('geometry') or {}).get('pivot')
    report = dict(source=f'{CAMERA}_mask_table_intersection', camera=CAMERA, geometry={}, evidence={},
                  cloud_pivot_base_m=baseline, candidate_status='unavailable', shape='unknown',
                  fallback_reason='')
    color = cv2.imread(str(snapshot/f'{CAMERA}_sam_color.png'))
    if color is None:
        color = cv2.imread(str(snapshot/f'{CAMERA}_color.png'))
    try:
        mask = cv2.imread(str(snapshot/f'{CAMERA}_sam_mask.png'), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f'missing saved {CAMERA} SAM mask (is the culler publishing its review mask?)')
        if color is None or mask.shape != color.shape[:2]:
            raise ValueError(f'missing or incompatible {CAMERA} color')
        info = json.loads((snapshot/f'{CAMERA}_camera_info.json').read_text())
        if mask.shape != (info['height'], info['width']):
            raise ValueError('mask and calibration dimensions differ')
        if np.any(np.abs(info.get('D', [])) > 1e-8):
            raise ValueError('distorted aligned depth is not supported by table fitter')
        cloud = np.load(snapshot/'object_cloud.npz')['xyz']
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        # +-10 mm about the configured table: the default band also takes in a thin base plate
        # missing from the object cloud (monitor), which fails the noise check below.
        table = audit(snapshot, camera=CAMERA, table_band=(TABLE_Z - .01, TABLE_Z + .01))['table']
        report['evidence']['table_fit'] = table
        if not table['available'] or table['p95_abs_mm'] > 6:
            raise ValueError('insufficient or noisy measured table plane')
        plane = table['equation_z_ax_by_c']
        depth_path = snapshot/f'{CAMERA}_sam_depth.npz'
        if not depth_path.exists():
            depth_path = snapshot/f'{CAMERA}_aligned_depth.npz'
        depth = np.load(depth_path)['depth_mm']
        if depth.shape != mask.shape:
            raise ValueError('depth and mask dimensions differ')
        pixels, boundary, points, accepted, votes, depth_evidence = screened_boundary(
            mask, cloud, info, plane, depth)
        envelope, sam_only = .02, False
        try:
            support_run(pixels, points, accepted, OUTWARD)
        except ValueError as exc:
            if votes >= SAM_ONLY_MIN_VOTES:
                envelope, sam_only = SAM_ONLY_ENVELOPE_M, True
                report['evidence']['strict_screen_failure'] = str(exc)
                pixels, boundary, points, accepted, votes, depth_evidence = screened_boundary(
                    mask, cloud, info, plane, depth, envelope=envelope, depth_optional=True)
        report['support_mode'] = 'sam_only' if sam_only else 'cloud_checked'
        report.update(image_width=mask.shape[1], image_height=mask.shape[0])
        report['geometry'] = dict(object_table_boundary_px=boundary[accepted].tolist(),
                                  rejected_boundary_px=boundary[~accepted].tolist(),
                                  support_samples_base_m=points[accepted].tolist())
        report['evidence'].update(component_cloud_votes=votes, boundary_count=len(pixels),
                                  accepted_count=int(accepted.sum()), envelope_threshold_m=envelope,
                                  depth_boundary_check=depth_evidence)
        try:
            run = support_run(pixels, points, accepted, OUTWARD)
            fit = fit_support(points[run], plane, OUTWARD)
            if sam_only and fit['support_type'] != 'flat_edge':
                raise ValueError(f"SAM-only edge is not one straight line ({fit['support_type']})")
            report['geometry'].update(candidate_pivot_base_m=fit['pivot'],
                                      pivot_edge_base_m=fit['edge'], tangent_base=fit['axis'])
            report['evidence']['support_fit'] = fit
            pixel_sensitivity, plane_sensitivity = fit_sensitivity(boundary[run], points[run], info, plane)
            report['evidence'].update(
                pivot_delta_from_cloud_mm=None if baseline is None else
                ((np.asarray(fit['pivot'])-np.asarray(baseline))*1000).tolist(),
                x_shift_per_pixel_mm=pixel_sensitivity, x_shift_per_mm_table_height_mm=plane_sensitivity)
            report['evidence']['x_error_from_table_height_mm'] = plane_sensitivity*TABLE_HEIGHT_ERROR_MM
            if plane_sensitivity*TABLE_HEIGHT_ERROR_MM > MAX_X_ERROR_MM:
                report['candidate_status'] = 'rejected'
                report['fallback_reason'] = (f'{TABLE_HEIGHT_ERROR_MM} mm table-height error amplified '
                                             f'beyond {MAX_X_ERROR_MM:.0f} mm in X')
            else:
                report['candidate_status'] = 'reviewable'
                report['shape'], report['shape_reason'] = pivot_shape(fit)
                report['fallback_reason'] = 'candidate needs independent ground truth validation'
        except ValueError as exc:
            report['candidate_status'] = 'rejected'
            report['fallback_reason'] = str(exc)
        color = review_overlay(color, boundary, accepted, info, report)
    except (ValueError, KeyError, OSError, cv2.error) as exc:
        report['fallback_reason'] = str(exc)
    if color is not None and not cv2.imwrite(str(snapshot/f'{CAMERA}_forward_pivot_review.png'), color):
        raise OSError('could not write forward pivot overlay')
    (snapshot/'forward_pivot_review.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def contacts_with_forward_pivot(contacts, report, apply=True):
    """Copy of `contacts` whose forward_tip records the pivot shape and, if `apply` and the
    fit is reviewable, uses the SAM pivot point. Returns (contacts, applied, reason).

    Like the press path, only geometry.pivot (and its tangent) moves: the axis and
    push direction stay the cloud's, and the contact point stays as selected (its
    forward score depends on the pivot only through its height).
    """
    result = copy.deepcopy(contacts)
    forward = result.get('forward_tip') or {}
    fit = report.get('evidence', {}).get('support_fit') or {}
    forward['pivot_shape'] = dict(shape=report.get('shape', 'unknown'), support_type=fit.get('support_type'),
                                  reason=report.get('shape_reason') or report.get('fallback_reason'),
                                  source=report.get('source'), status=report.get('candidate_status'))
    result['forward_tip'] = forward
    geometry = forward.get('geometry') or {}
    pivot = (report.get('geometry') or {}).get('candidate_pivot_base_m')
    if not apply:
        return result, False, 'shadow mode'
    if report.get('candidate_status') != 'reviewable' or pivot is None:
        return result, False, f"forward SAM pivot not reviewable: {report.get('fallback_reason')}"
    if not forward.get('available') or geometry.get('pivot') is None:
        return result, False, 'no forward-tip contact/pivot'
    pivot = np.asarray(pivot, float)
    direction = np.asarray(geometry['direction'], float)
    if (pivot - np.asarray(forward['ball_center'], float)) @ direction <= 0:
        return result, False, 'SAM pivot is not beyond the forward contact'
    axis = np.asarray(geometry['axis'], float)
    old = geometry['pivot']
    geometry['pivot'] = pivot.tolist()
    geometry['tangent'] = [(pivot - .04*axis).tolist(), (pivot + .04*axis).tolist()]
    geometry['pivot_source'] = 'rgb_sam_cam3'
    forward['rgb_pivot'] = dict(applied=True, source=report.get('source'),
                                replaced_cloud_pivot_base_m=old, support_type=fit.get('support_type'))
    return result, True, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.snapshot), indent=2))


if __name__ == '__main__':
    main()
