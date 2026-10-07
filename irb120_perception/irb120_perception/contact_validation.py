"""Offline contact review and seeded sensitivity experiments; never writes contacts.

Run with --episodes ROOT --output NEW_DIRECTORY. Camera sensitivity holds the
cloud fixed; cloud sensitivity deliberately omits camera gates to isolate geometry.
Neither experiment measures manipulation success or calibrated uncertainty.
"""
import argparse
import copy
import csv
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .contact_point_selector import estimate_normals, select_contact_points
from .experimental_view_check import check_view, decide

SCENES = ['20261005_160640_full_success',
          '20261005_170325_full_badpressselection', '20261005_184639']
RADIUS = .01325
LIMITS = (46, 30, 20)


def torque(points, geometry, ratio=1.):
    """Signed torque / horizontal pull magnitude, in metres (ratio=P/H)."""
    force = np.asarray(geometry['direction']) + np.array([0., 0., -ratio])
    return np.cross(np.asarray(points) - geometry['pivot'], force) @ geometry['axis']


def slope(normal):
    return float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))


def approach_evidence(points, point, normal):
    """Observed support proxies, NOT a collision check or contact prediction.

    A local XY hull can bridge gaps. No support is inferred outside it; even
    inside it, the evidence cannot certify the ball's vertical descent.
    """
    center = point + RADIUS * normal
    local = points[np.linalg.norm(points - point, axis=1) <= .012]
    margin = None
    try:
        hull = ConvexHull(local[:, :2])
        margin = float(np.min(-(hull.equations[:, :2] @ center[:2]
                                + hull.equations[:, 2])) * 1000)
    except (ValueError, QhullError):
        pass
    return dict(local_samples=len(local), local_radius_mm=12,
                ball_xy_offset_mm=float(np.linalg.norm(center[:2]-point[:2])*1000),
                center_inside_local_xy_hull=margin is not None and margin >= 0,
                local_xy_hull_margin_mm=margin,
                nearest_local_sample_to_descent_axis_mm=(
                    float(np.min(np.linalg.norm(local[:, :2]-center[:2], axis=1))*1000)
                    if len(local) else None),
                interpretation='diagnostic only; observed hull may bridge missing surface')


def candidates(points):
    points = np.unique(points[np.isfinite(points).all(axis=1)], axis=0)
    normals = estimate_normals(points)
    tree = cKDTree(points)
    captured = []

    def capture(p, direction):
        captured.extend(tree.query(p)[1].tolist())
        return np.ones(len(p), dtype=bool)

    result = select_contact_points(points, normals=normals.copy(),
                                   min_ball_center_z=.06,
                                   experimental_press_filter=capture)
    return points, normals, np.asarray(captured, dtype=int), result['press']


def camera_evidence(point, direction, cameras):
    views = {name: check_view(point, direction, depth, info)
             for name, (depth, info) in cameras.items()}
    decision, reason = decide(views)
    return dict(decision=decision, reason=reason, views=views)


def choose(points, normals, allowed):
    tree = cKDTree(points)
    return select_contact_points(
        points, normals=normals.copy(), min_ball_center_z=.06,
        experimental_press_filter=lambda p, d: allowed[tree.query(p)[1]])['press']


def cloud_trials(points, trials, seed):
    """Paired perturbations across slope variants, with normals/pivot recomputed."""
    rng = np.random.default_rng(seed)
    records = []
    base_p, base_n, _, _ = candidates(points)
    references = {limit: choose(base_p, base_n, base_n[:, 2] >= np.cos(np.radians(limit)))
                  for limit in LIMITS}
    for condition in ['jitter_1mm', 'jitter_2mm', 'jitter_4mm',
                      'drop_25pct', 'missing_patch_8mm', 'missing_contact_patch_8mm']:
        for trial in range(trials):
            changed = points.copy()
            if condition.startswith('jitter'):
                sigma = float(condition.split('_')[1][:-2]) / 1000
                changed += rng.normal(0, sigma, changed.shape)
            elif condition == 'drop_25pct':
                changed = changed[rng.random(len(changed)) >= .25]
            else:
                anchor = (np.asarray(references[46]['point']) + rng.normal(0, .003, 3)
                          if condition == 'missing_contact_patch_8mm' and references[46]['available']
                          else changed[rng.integers(len(changed))])
                changed = changed[np.linalg.norm(changed-anchor, axis=1) > .008]
            p, n, _, _ = candidates(changed)
            for limit in LIMITS:
                selected = choose(p, n, n[:, 2] >= np.cos(np.radians(limit)))
                ref = references[limit]
                row = dict(condition=condition, trial=trial, slope_limit_deg=limit,
                           available=selected['available'], reason=selected.get('reason'))
                if selected['available']:
                    row.update(point=selected['point'], ball_center=selected['ball_center'])
                if selected['available'] and ref['available']:
                    row.update(ball_shift_mm=float(np.linalg.norm(selected['ball_center']-
                                                                  ref['ball_center'])*1000),
                               point_shift_mm=float(np.linalg.norm(selected['point']-ref['point'])*1000),
                               pivot_shift_mm=float(np.linalg.norm(selected['geometry']['pivot']-
                                                                   ref['geometry']['pivot'])*1000),
                               direction_change_deg=float(np.degrees(np.arccos(np.clip(
                                   selected['geometry']['direction'] @ ref['geometry']['direction'], -1, 1)))))
                records.append(row)
    return dict(scope='geometry only, camera gate omitted; not end-to-end success',
                seed=seed, trials_per_condition=trials, references=references, records=records)


def changed_cameras(cameras, condition):
    result = copy.deepcopy(cameras)
    if condition.startswith('omit_'):
        result.pop(condition[5:], None)
    elif condition.startswith('depth_'):
        offset = int(condition.split('_')[1])
        for name, (depth, info) in result.items():
            result[name] = (np.where(depth > 0, np.clip(depth.astype(int)+offset, 1, 65535), 0)
                            .astype(np.uint16), info)
    elif condition.startswith('cam2_tx_'):
        result['cam2'][1]['base_to_optical']['t'][0] += float(condition.split('_')[-1]) / 1000
    elif condition.startswith('cam2_yaw_'):
        # Rotate the optical frame, including its translation, about its Z axis.
        angle = np.radians(float(condition.split('_')[-1]))
        rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        tf = result['cam2'][1]['base_to_optical']
        tf['R'] = (rot @ np.asarray(tf['R'])).tolist()
        tf['t'] = (rot @ np.asarray(tf['t'])).tolist()
    return result


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path.write_text(json.dumps(value, default=plain, indent=2, allow_nan=False)+'\n')


def project(points, info):
    tf = info['base_to_optical']
    optical = np.asarray(points) @ np.asarray(tf['R']).T + tf['t']
    uv, _ = cv2.projectPoints(optical.reshape(-1, 3), np.zeros(3), np.zeros(3),
                              np.asarray(info['K']), np.asarray(info.get('D', [])))
    return np.rint(uv.reshape(-1, 2)).astype(int)


def render_geometry(snapshot, output, geometry, cameras):
    from .contact_graphics import contact_graphic
    panels = []
    for name, (depth, info) in cameras.items():
        color = cv2.imread(str(snapshot/f'{name}_color.png'))
        if color is not None:
            panels.append(contact_graphic(color, depth, info, geometry=geometry))
    if panels:
        cv2.imwrite(str(output/'geometry.jpg'), np.hstack(panels))


def render(snapshot, output, option, cameras):
    from .contact_graphics import contact_graphic
    panels = []
    for name, (depth, info) in cameras.items():
        color = cv2.imread(str(snapshot/f'{name}_color.png'))
        if color is None:
            continue
        img = contact_graphic(color, depth, info, option)
        x, y = project([option['point']], info)[0]
        h, w = img.shape[:2]
        x0 = max(0, min(w-160, x-80)); y0 = max(0, min(h-160, y-80))
        crop = cv2.resize(img[y0:y0+160, x0:x0+160], (400, 400))
        cv2.rectangle(crop, (0, 0), (400, 43), (20, 20, 20), -1)
        cv2.putText(crop, f"{option['id']} / {name}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255), 1)
        cv2.putText(crop, 'Flat projection; visibility not inferred', (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        panels.append(crop)
    if panels:
        cv2.imwrite(str(output/f"{option['id']}.jpg"), np.hstack(panels))


def run_scene(snapshot, output, trials, seed):
    output.mkdir()
    raw = np.load(snapshot/'object_cloud.npz')['xyz']
    points, normals, ids, geometric = candidates(raw)
    geometry = geometric['geometry']
    cameras = {}
    for name in ('cam1', 'cam2', 'cam3'):
        depth = cv2.imread(str(snapshot/f'{name}_depth.png'), cv2.IMREAD_UNCHANGED)
        info_path = snapshot/f'{name}_camera_info.json'
        if depth is not None and depth.dtype == np.uint16 and info_path.exists():
            cameras[name] = (depth, json.loads(info_path.read_text()))
    rows = []
    allowed = np.zeros(len(points), dtype=bool)
    for i in ids:
        evidence = camera_evidence(points[i], geometry['direction'], cameras)
        allowed[i] = evidence['decision'] == 'pass'
        rows.append(dict(index=int(i), point=points[i], normal=normals[i],
                         ball_center=points[i]+RADIUS*normals[i], slope_deg=slope(normals[i]),
                         torque_per_pull_N_m=float(torque(points[i], geometry)),
                         approach=approach_evidence(points, points[i], normals[i]), **evidence))
    variants = {str(limit): choose(points, normals, allowed &
                                    (normals[:, 2] >= np.cos(np.radians(limit))))
                for limit in LIMITS}
    recorded = json.loads((snapshot/'contacts.json').read_text())['press']
    options = []
    if recorded.get('available'):
        p = np.asarray(recorded['point']); n = np.asarray(recorded['normal'])
        options.append(dict(id='A', role='recorded contact', point=p, normal=n,
                            ball_center=recorded['ball_center'], slope_deg=slope(n),
                            torque_per_pull_N_m=float(torque(p, geometry)),
                            approach=approach_evidence(points, p, n),
                            **camera_evidence(p, geometry['direction'], cameras)))
    passed = [r for r in rows if r['decision'] == 'pass']
    slope_order = sorted(passed, key=lambda r: (r['slope_deg'], -r['torque_per_pull_N_m']))
    slope_role = 'least-sloped camera-passing alternative'
    for limit in ('20', '30'):
        if variants[limit]['available']:
            winner = variants[limit]['point']
            slope_order.sort(key=lambda r: np.linalg.norm(r['point']-winner))
            slope_role = f'{limit}-degree selector winner'
            break
    for role, ordered in [
        (slope_role, slope_order),
        ('torque alternative', sorted(passed, key=lambda r: -r['torque_per_pull_N_m']))]:
        for row in ordered:
            if all(np.linalg.norm(np.asarray(row['point'])-o['point']) >= .004 for o in options):
                options.append(dict(row, id=chr(65+len(options)), role=role))
                break
    for option in options:
        render(snapshot, output, option, cameras)
    render_geometry(snapshot, output, geometry, cameras)
    diagnostic_mask = np.zeros(len(points), dtype=bool)
    for row in rows:
        diagnostic_mask[row['index']] = row['approach']['center_inside_local_xy_hull']
    support_ablation = {str(limit): choose(points, normals, allowed & diagnostic_mask &
                        (normals[:, 2] >= np.cos(np.radians(limit)))) for limit in LIMITS}
    save_json(output/'review.json', dict(snapshot=str(snapshot.resolve()), geometry=geometry,
              options=options, candidates=rows, variants=variants,
              support_proxy_ablation=support_ablation,
              option_spacing_mm=4, support_rule='diagnostic only; no hard support gate',
              torque_assumption='downward pressure / horizontal pull = 1; not measured'))
    save_json(output/'cloud_trials.json', cloud_trials(points, trials, seed))
    sensitivities = []
    conditions = ['nominal', 'depth_-4', 'depth_-2', 'depth_2', 'depth_4']
    conditions += [f'omit_{cam}' for cam in cameras]
    if 'cam2' in cameras:
        conditions += ['cam2_tx_-2', 'cam2_tx_2', 'cam2_yaw_-0.5', 'cam2_yaw_0.5']
    for condition in conditions:
        altered = changed_cameras(cameras, condition)
        mask = np.zeros(len(points), dtype=bool)
        decisions = []
        for i in ids:
            evidence = camera_evidence(points[i], geometry['direction'], altered)
            mask[i] = evidence['decision'] == 'pass'
            decisions.append(dict(index=int(i), **evidence))
        sensitivities.append(dict(condition=condition, candidates=decisions,
            selections={str(limit): choose(points, normals, mask &
                        (normals[:, 2] >= np.cos(np.radians(limit)))) for limit in LIMITS}))
        print(snapshot.parent.name, condition, 'done', flush=True)
    save_json(output/'camera_trials.json', dict(scope='fixed cloud, normals and pivot; camera evidence sensitivity',
                                              records=sensitivities))
    return options


def summarize(output):
    trial_count = json.loads((output/SCENES[0]/'cloud_trials.json').read_text())['trials_per_condition']
    lines = ['# Offline validation results', '',
             'No robot motion or executable contact changes. The three scenes are development examples; '
             'a fresh held-out pose and your contact judgments are still needed.', '',
             'The existing fused-normal limit is n_z >= 0.7 (45.57 degrees); the 46-degree column '
             'preserves that existing constraint. All nominal variants use the current V2 camera gate. '
             'Torque assumes downward pressure / horizontal pull = 1.', '',
             'Cloud experiments recompute normals, pivot and selection but omit camera gates to isolate geometry. '
             f'Each condition has {trial_count} seeded trials. Gaussian jitter is independent per XYZ coordinate; '
             'it is a stress model, not a calibrated depth-noise model. Random missing patches are centered on '
             'a random cloud point; targeted patches are centered within 3 mm Gaussian jitter of the nominal '
             'geometry-only winner. The ball shifts compare each slope variant against its own nominal selection.', '',
             'Camera experiments hold cloud geometry fixed. Depth bias affects all available cameras; '
             'translation and optical-frame rotation affect camera 2 only. Dropout removes each camera in turn. '
             'These are separate sensitivity tests, not a joint end-to-end uncertainty estimate.', '']
    lines += ['The selector reports `camera_agreement` for the shared offline filter hook; in these '
              'tables that hook also includes the slope and optional support constraints. An unavailable '
              'result at that stage does not necessarily mean camera disagreement.', '']
    csv_rows = []
    for scene in SCENES:
        folder = output/scene
        review = json.loads((folder/'review.json').read_text())
        lines += [f'## {scene}', '', '| Slope limit | V2 selection | With local-hull support proxy |',
                  '|---|---|---|']
        for limit, selection in review['variants'].items():
            def desc(value):
                return ('available at '+str([round(x, 4) for x in value['point']])
                        if value['available'] else 'unavailable: '+value['reason'])
            lines.append(f'| {limit}° | {desc(selection)} | {desc(review["support_proxy_ablation"][limit])} |')
        lines += ['', 'The support proxy is an offline ablation only: the ball center must project inside '
                  'the XY hull of observed points within 12 mm of the contact. This can bridge holes and '
                  'does not certify descent clearance. It is not enabled in the selector.', '',
                  '| Cloud perturbation | Slope | Available / trials | Ball shift median / p95 (mm) |',
                  '|---|---|---|---|']
        cloud = json.loads((folder/'cloud_trials.json').read_text())
        for condition in dict.fromkeys(r['condition'] for r in cloud['records']):
            for limit in LIMITS:
                rows = [r for r in cloud['records'] if r['condition'] == condition and r['slope_limit_deg'] == limit]
                shifts = [r['ball_shift_mm'] for r in rows if 'ball_shift_mm' in r]
                median, p95 = np.percentile(shifts, [50, 95]) if shifts else (None, None)
                available = sum(r['available'] for r in rows)
                shift_text = f'{median:.1f} / {p95:.1f}' if shifts else 'no available reference/pairs'
                lines.append(f'| {condition} | {limit}° | {available}/{len(rows)} | {shift_text} |')
                csv_rows.append([scene, condition, limit, available, len(rows), median, p95])
        lines += ['', '| Camera perturbation | Available slope variants | Ball shift from nominal (mm), by variant |',
                  '|---|---|---|']
        camera = json.loads((folder/'camera_trials.json').read_text())
        reference = camera['records'][0]['selections']
        for row in camera['records']:
            available = [k for k,v in row['selections'].items() if v['available']]
            shifts = [f'{k}°: {np.linalg.norm(np.asarray(v["ball_center"])-reference[k]["ball_center"])*1000:.1f}'
                      for k,v in row['selections'].items() if v['available'] and reference[k]['available']]
            lines.append(f'| {row["condition"]} | {", ".join(available) or "none"} | {"; ".join(shifts) or "—"} |')
        lines += ['', 'Options: '+', '.join(f'{o["id"]}: {o["role"]} ({o["slope_deg"]:.1f}°, {o["decision"]})'
                                           for o in review['options']), '']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    with (output/'cloud_summary.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scene', 'condition', 'slope_deg', 'available', 'trials', 'median_ball_shift_mm', 'p95_ball_shift_mm'])
        writer.writerows(csv_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261006)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    page = ['<!doctype html><meta charset="utf-8"><title>Contact review</title>',
            '<style>body{font:17px system-ui;max-width:1250px;margin:30px auto;background:#fafafa}'
            'img{width:100%}article{background:white;padding:20px;margin:20px 0;border:1px solid #ccc}'
            'td,th{padding:6px;text-align:left}textarea{width:95%;height:65px}</style>',
            '<h1>Contact options for your review</h1><p>Yellow dot: surface sample. Magenta: ball outline and center. '
            'Cyan arrow: vertical descent. Overlays are projected geometry, not proof of visible contact or clearance.</p>',
            '<p>A is the recorded contact; B/C are distinct alternatives, not automatically approved recommendations. '
            'Judge preferred / acceptable / bad / unsure, and give a reason. None acceptable is a valid answer. '
            'Feedback fields are for drafting only; copy your answers into chat or feedback.csv.</p>']
    feedback = []
    for index, scene in enumerate(SCENES):
        snapshot = args.episodes/scene/'00_perceive'
        for path in sorted(snapshot.glob('*')):
            if path.is_file() and path.suffix in ('.json', '.npz', '.png'):
                manifest[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        options = run_scene(snapshot, args.output/scene, args.trials, args.seed+index)
        page.append(f'<h2>{html.escape(scene)}</h2>')
        page.append(f'<img src="{scene}/geometry.jpg" alt="Estimated pivot tangent and pull direction">')
        for option in options:
            ident = option['id']
            page.append(f'<article><h3>{ident}: {option["role"]}</h3>'
                        f'<img src="{scene}/{ident}.jpg"><p>Slope: {option["slope_deg"]:.1f}°; '
                        f'torque / pull magnitude: {option["torque_per_pull_N_m"]:.4f} m; '
                        f'camera decision: {option["decision"]}; '
                        f'ball horizontal offset: {option["approach"]["ball_xy_offset_mm"]:.1f} mm.</p>'
                        '<textarea placeholder="Preferred / acceptable / bad / unsure — reason"></textarea></article>')
            feedback.append([scene, ident, '', ''])
    (args.output/'index.html').write_text('\n'.join(page))
    with (args.output/'feedback.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scene', 'option', 'judgment', 'reason']); writer.writerows(feedback)
    save_json(args.output/'input_hashes.json', manifest)
    save_json(args.output/'run.json', dict(seed=args.seed, trials=args.trials,
              episodes=str(args.episodes.resolve()),
              source_sha256={str(Path(__file__).with_name(name).resolve()):
                  hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                  for name in ('contact_validation.py', 'contact_point_selector.py', 'experimental_view_check.py')}))
    summarize(args.output)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
               for path, digest in manifest.items()), 'Input artifacts changed during run'
    print('Review:', args.output/'index.html', flush=True)


if __name__ == '__main__':
    main()
