"""Estimate episode tipping stages without supplying GT mass or height to fits.

Two gravity first moments are fitted on each branch. Splitting them into mass
and height assumes a known horizontal CoM distance and the selected tilt source.
No fitted empirical angle, gain, contact-couple or lever correction is applied.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from .physical_bounds import cloud_height_bound, bounded_gravity_fit

from .tip_core import (G, TILT_DEADBAND, compare_tracks, detect_onset,
                       pivot_wrench, tilt_noslip, tilt_sliding_face, tilt_rolling_sphere)


def reference_for_primitive(reference, primitive):
    """Copy the scoring reference, expressing CoM relative to the active edge."""
    result = dict(reference)
    result['com'] = np.array(reference['com'], float).copy()
    if primitive == 'press_pull':
        result['com'][0] = reference['com_obj'][0]
    result['theta_deg'] = float(np.degrees(np.arctan2(result['com'][0], result['com'][2])))
    return result


def fit_first_moments(log, track, com_x, *, pivot=None, slide_speed_limit=None):
    """Jointly estimate mass/height from known com_x; no mass/height reference input.

    The model is tau = g*(h_x*cos(phi) - h_z*sin(phi)), h_x=m*d,
    h_z=m*z. Camera tracks carry a moving pivot. Proprioceptive tracks require
    a fixed, independently specified support. Results remain conditional on
    angle zero, scale and support geometry; a small residual does not validate them.
    """
    if not np.isfinite(com_x) or com_x <= 0:
        raise ValueError('horizontal CoM distance must be positive')
    phi = track.rotvec @ log.tilt_dir
    p = track.pivot_t if pivot is None and track.pivot_t is not None else (
        log.pivot if pivot is None else pivot)
    tau = pivot_wrench(log, np.zeros_like(log.p_ball), p)[:, :3] @ log.tilt_dir
    mask = (track.valid & log.contact & np.isfinite(phi) & np.isfinite(tau)
            & (phi > TILT_DEADBAND))
    if slide_speed_limit is not None:
        if track.slide is None:
            raise ValueError('slide-speed gating requires an observed slide track')
        good = track.valid & np.isfinite(track.slide)
        if good.sum() < 20:
            raise ValueError('insufficient observed translation for slide-speed gating')
        dt = float(np.median(np.diff(log.t_epoch)))
        slide = np.interp(log.t_epoch, log.t_epoch[good], track.slide[good])
        speed = np.gradient(gaussian_filter1d(slide, .25 / dt), log.t_epoch)
        mask &= good & (np.abs(speed) < slide_speed_limit)
    height_bound, height_source = cloud_height_bound(log)
    branches = {}
    for branch in ('tip', 'return'):
        sel = mask & (log.branch == branch)
        if sel.sum() < 10:
            continue
        X = np.c_[np.cos(phi[sel]), -np.sin(phi[sel])]
        mass, height, active_bounds = bounded_gravity_fit(phi[sel], tau[sel], com_x, 0., height_bound)
        hx, hz = mass*com_x, mass*height
        branches[branch] = dict(m=float(mass), zc=float(height), active_bounds=active_bounds,
            theta=float(np.degrees(np.arctan2(com_x, height))),
            horizontal_first_moment_kg_m=float(hx), vertical_first_moment_kg_m=float(hz),
            amplitude_kg_m=float(np.hypot(hx, hz)), n=int(sel.sum()),
            tilt_range_deg=np.degrees([phi[sel].min(), phi[sel].max()]).tolist(),
            condition=float(np.linalg.cond(X)),
            rms_mNm=float(np.sqrt(np.mean((G * X @ [hx, hz] - tau[sel]) ** 2))) * 1000)
    if not branches:
        raise ValueError('no branch has enough valid rocking samples above 1 degree')
    keys = ('m', 'zc', 'theta', 'horizontal_first_moment_kg_m',
            'vertical_first_moment_kg_m', 'amplitude_kg_m')
    mean = {k: float(np.mean([q[k] for q in branches.values()])) for k in keys}
    # Independent contact-wrench consistency check. This is diagnostic only:
    # a finite contact patch may carry a couple, and geometry may be inaccurate.
    from scipy.spatial.transform import Rotation
    R = log.sensor_T()[:, :3, :3]
    force = -np.einsum('nij,nj->ni', R, log.f_S)
    moment = -np.einsum('nij,nj->ni', R, log.tau_S)
    normal0 = log.push_dir if log.primitive == 'fwd_tip' else np.array([0., 0., -1.])
    normal = Rotation.from_rotvec(np.nan_to_num(track.rotvec)).apply(np.broadcast_to(normal0, log.p_ball.shape))
    contact_point = log.p_ball + .01325 * normal
    residual = (moment - np.cross(contact_point - log.p_S, force)) @ log.tilt_dir
    diagnostics = {}
    flags = ['parameter_bound_active'] if any(q['active_bounds'] for q in branches.values()) else []
    for b, q in branches.items():
        sel = mask & (log.branch == b)
        diagnostics[b] = dict(contact_wrench_residual_median_mNm=float(np.median(residual[sel])) * 1000,
                              contact_wrench_residual_rms_mNm=float(np.sqrt(np.mean(residual[sel] ** 2))) * 1000)
        if abs(diagnostics[b]['contact_wrench_residual_median_mNm']) > 5:
            flags.append('contact_wrench_disagreement_above_5mNm')
        if np.ptp(q['tilt_range_deg']) < 5:
            flags.append('angular_span_below_5deg')
    if len(branches) < 2:
        flags.append('single_branch_only')
    return dict(height_bound_m=height_bound,height_bound_source=height_source,method='joint_first_moments', tilt_source=track.source, mean=mean,
                branches=branches, diagnostics=diagnostics, quality_flags=sorted(set(flags)))


def episode_paths(root, obj, selection='latest'):
    from irb120_perception.snapshot_layout import episode_jsons
    root = Path(root)
    if (root/'episode.json').exists():  # one episode folder
        return [root/'episode.json']
    paths = sorted(set(episode_jsons(root/'runtime_logs', obj)) | set(episode_jsons(root, obj)),
                   key=lambda p: (p.parent.name, str(p)))
    if selection == 'latest':
        return paths[-1:]
    if selection == 'all':
        return paths
    return [p for p in paths if p.parent.name == selection]


def episode_object(path) -> str:
    """Object name recorded in an episode folder."""
    return json.loads((Path(path)/'episode.json').read_text())['object']


def analyze_episode(path, reference, *, tilt_source='proprio', vision=False, cache_dir=None,
                    press_contact_model='point', press_pivot_offset_x=0., forward_reference='first_contact'):
    from .estimate_params import tiplog_from_press_pull
    from .fwd_tip_estimation import load_fwd_tip, controller_outcome
    from .tilt_vision import track_tilt_vision
    episode = json.loads(Path(path).read_text())
    output = dict(object=episode['object'], episode=Path(path).parent.name,
                  outcome=episode.get('outcome'), stages=[])
    for stage in episode['stages']:
        kind = stage['kind']
        if kind not in ('forward_arc', 'forward_tip', 'press_pull_tip'):
            continue
        fw = kind in ('forward_arc', 'forward_tip')
        folder = Path(path).parent / stage['dir']
        files = sorted(folder.glob('fwd_*_2*.npz' if fw else 'arc_static_2*.npz'))
        rec = dict(stage=stage['dir'], controller_status=stage['status'], kind=kind)
        output['stages'].append(rec)
        if not files:
            rec.update(status='skipped', reason='no timestamped log')
            continue
        rec['path'] = str(files[-1])
        try:
            metadata = json.loads(files[-1].with_suffix('.json').read_text())
            rec['controller_parameters'] = {key: metadata[key] for key in (
                'final_force_ref_n', 'attempts', 'pre_squash', 'arc_center',
                'commanded_pivot', 'direction', 'git_commit', 'git_dirty') if key in metadata}
            log = load_fwd_tip(str(files[-1])) if fw else tiplog_from_press_pull(str(files[-1]))
            rec['commanded_pivot_xyz'] = log.pivot.tolist()
            rec['wrench_origin_convention'] = 'finger_mount_from_logged_ball'
            reason = controller_outcome(log) if fw else None
            rec['controller_outcome'] = reason
            if fw and reason in ('slide', 'too_easy'):
                rec.update(status='skipped', reason=reason)
                continue
            gt = reference_for_primitive(reference, log.primitive)
            d = float(gt['com'][0])
            onset = detect_onset(log, log.pivot)
            rec['onset'] = onset
            pivot = log.pivot.copy()
            selected_source = 'vision' if tilt_source == 'vision' or (fw and tilt_source == 'forward_vision') else 'proprio'
            vt = None
            if vision or selected_source == 'vision':
                try:
                    vt = track_tilt_vision(log, cache_dir=cache_dir, verbose=False)
                except (OSError, ValueError, ImportError) as exc:
                    rec['vision_error'] = str(exc)
            if fw:
                if not onset.get('rocking'):
                    raise ValueError('no torque-drop rocking evidence')
                ion = onset['i_on']
                if vt is not None and vt.slide is not None:
                    pre = vt.valid & np.isfinite(vt.slide) & (np.arange(len(log.t_epoch)) < onset['i0'])
                    if pre.any() and vt.valid[ion] and np.isfinite(vt.slide[ion]):
                        slide = float(vt.slide[ion] - np.median(vt.slide[pre][-50:]))
                        pivot += slide * log.push_dir
                        rec['slide_m'], rec['slide_source'] = slide, 'vision'
                if 'slide_source' not in rec:
                    # Arc travel is not observed table translation.
                    slide = onset['slide_m'] if kind == 'forward_tip' else 0.
                    pivot += slide * log.push_dir
                    rec['slide_m'], rec['slide_source'] = slide, 'finger' if kind == 'forward_tip' else 'unobserved_arc'
                if forward_reference not in ('first_contact', 'torque_peak'):
                    raise ValueError('unknown forward tilt reference')
                iref = onset['i0'] if forward_reference == 'first_contact' else ion
                upright_distance = (float((log.pivot-log.p_ball[iref])@log.push_dir)
                                    if forward_reference == 'first_contact' else None)
                track = tilt_sliding_face(log, pivot, iref, upright_distance=upright_distance)
                rec['forward_reference'] = forward_reference
                rec['upright_distance_m'] = upright_distance
            else:
                pivot[0] += press_pivot_offset_x
                if press_contact_model not in ('point', 'sphere_roll'):
                    raise ValueError('unknown press contact model')
                track = (tilt_rolling_sphere(log, pivot) if press_contact_model == 'sphere_roll'
                         else tilt_noslip(log, pivot))
                rec['press_contact_model'] = press_contact_model
                rec['press_pivot_offset_x_m'] = press_pivot_offset_x
                rec['tilt_diagnostics'] = track.info
            fits = {'proprio': fit_first_moments(log, track, d, pivot=pivot)}
            if vt is not None:
                rec['vision'] = dict(info=vt.info, agreement=compare_tracks(log, track, vt))
                try:
                    fits['vision'] = fit_first_moments(log, vt, d, slide_speed_limit=.001)
                except ValueError as exc:
                    rec['vision_error'] = str(exc)
            if selected_source not in fits:
                raise ValueError('requested vision fit unavailable; no silent fallback')
            rec.update(status='estimated', selected_source=selected_source,
                       horizontal_com_distance_m=d, fits=fits, estimate=fits[selected_source])
            if stage['status'] != 'ok':
                rec['estimate']['quality_flags'].append('controller_stage_not_ok')
            if (episode.get('outcome') or {}).get('result') != 'success':
                rec['estimate']['quality_flags'].append('episode_not_success')
            # Scoring is deliberately outside fit_first_moments. Mass and height GT
            # cannot affect tilt, filtering, branch selection, or fitting.
            q = rec['estimate']['mean']
            rec['score'] = dict(mass_gt=gt['mass'], com_z_gt=float(gt['com'][2]),
                theta_gt_deg=gt['theta_deg'], mass_error_pct=100*(q['m']/gt['mass']-1),
                height_error_pct=100*(q['zc']/gt['com'][2]-1))
        except (ValueError, KeyError, IndexError, OSError) as exc:
            rec.update(status='skipped', reason=str(exc))
    return output


def main_episodes(objects, root, out_dir, *, selection='latest', tilt_source='proprio',
                  vision=False, cache_dir=None, press_contact_model='point', press_pivot_offset_x=0.,
                  forward_reference='first_contact'):
    from irb120_models.experiment_gt import episode_reference
    references = {obj: episode_reference(obj) for obj in objects}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results, table = [], []
    for obj in objects:
        for path in episode_paths(root, obj, selection):
            episode = analyze_episode(path, references[obj], tilt_source=tilt_source,
                                      vision=vision, cache_dir=cache_dir or str(out/'camera_cache'),
                                      press_contact_model=press_contact_model,
                                      press_pivot_offset_x=press_pivot_offset_x,
                                      forward_reference=forward_reference)
            results.append(episode)
            for stage in episode['stages']:
                if stage['status'] != 'estimated':
                    print(f"{obj} {episode['episode']} {stage['stage']}: skipped ({stage['reason']})")
                    continue
                q, score = stage['estimate']['mean'], stage['score']
                row = dict(object=obj, episode=episode['episode'], stage=stage['stage'],
                    source=stage['selected_source'], mass_g=q['m']*1000, height_mm=q['zc']*1000,
                    mass_error_pct=score['mass_error_pct'], height_error_pct=score['height_error_pct'],
                    theta_deg=q['theta'],
                    final_press_force_n=stage['controller_parameters'].get('final_force_ref_n'),
                    controller_attempts=stage['controller_parameters'].get('attempts'),
                    flags=';'.join(stage['estimate']['quality_flags']))
                table.append(row)
                print(f"{obj} {row['episode']} {row['stage']} {row['source']}: {row['mass_g']:.1f} g "
                      f"({row['mass_error_pct']:+.1f}%), {row['height_mm']:.1f} mm "
                      f"({row['height_error_pct']:+.1f}%) {row['flags']}")
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    (out/'episodes.json').write_text(json.dumps(results, indent=2, default=encode)+'\n')
    if table:
        with (out/'estimates.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return results
