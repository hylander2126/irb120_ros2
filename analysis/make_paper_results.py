"""Regenerate every results table and figure from one data root. Never writes into raw data.

    python3 make_paper_results.py --data-root ~/Documents/irb120_data/official_v1/raw \
        --out ~/Documents/irb120_data/official_v1/results [--vision]

Every tip stage of every episode is processed with each method. A method is a support model
x a CoM source:

  support models
    fixed    production estimator (episode_estimation.analyze_episode): torque about the
             perceived/commanded pivot, proprioceptive tilt. All primitives.
    motion   shape-free instantaneous support recovered from ball motion (orange.py).
             Press-pull only.
    joint    cloud prior + motion posterior support geometry with a 95% band and gates
             (joint_fit.py). Press-pull with a support_scene_cloud only.
  CoM sources
    supplied d_env from irb120_models: CoM distance from the OUTER base edge, i.e. what a
             table-edge balance measures (the paper's "planar CoM supplied" assumption).
             Its 1-sigma propagates as mass_rel_sigma_from_d.
    hull     filled convex hull of the pre-action cloud (no supplied CoM; uniform density).
    cloud    the cloud axis (joint only; axisymmetric assumption).
  diagnostics (never paper inputs)
    fixed_oracle  fixed pivot with the fillet-aware support distance from the object record;
                  differs from fixed_supplied only for filleted bases (cylinders B-E).
    fixed_supplied_vision  fixed pivot, camera tilt instead of proprioceptive (--vision).

The identifiable first moments h_x = m*d and h_z = m*z are reported for the fixed methods.
Per stage, from the pre-contact prior alone: predicted_mass_rel_sigma_registration =
(camera registration sigma) / (hull arm d).

Supplied d is an input; mass/height references are looked up and used only after all
estimates for an episode exist. Each episode's endpoint is its last processed tip stage
(the cascade's final attempt); all stages are kept in stages.csv. Episodes recorded from a
dirty checkout or without --official are flagged, not dropped.
"""
import argparse
import csv
import datetime
import json
import subprocess
import traceback
from pathlib import Path

import numpy as np

import _paths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from irb120_control.estimation.episode_estimation import analyze_episode  # noqa: E402
from irb120_control.estimation.estimate_params import tiplog_from_press_pull  # noqa: E402
from irb120_control.estimation.fwd_tip_estimation import load_fwd_tip  # noqa: E402
from irb120_control.util.registration_check import check_folder  # noqa: E402
import hull_com  # noqa: E402
import joint_fit  # noqa: E402
import orange  # noqa: E402
import references  # noqa: E402
from scene_profile import analyze_scene  # noqa: E402

TIP_KINDS = ('forward_arc', 'forward_tip', 'press_pull_tip')
METHODS = ('fixed_supplied', 'fixed_hull', 'motion_supplied', 'motion_hull', 'joint_supplied', 'joint_cloud',
           'fixed_oracle', 'fixed_supplied_vision')
PAPER_CANDIDATES = METHODS[:6]


def stage_number(name):
    return int(name.split('_')[0])


def stage_log(ep_dir, stage):
    fw = stage['kind'] != 'press_pull_tip'
    files = sorted((ep_dir / stage['dir']).glob('fwd_*_2*.npz' if fw else 'arc_static_2*.npz'))
    return files[-1] if files else None


def perceive_before(ep_dir, episode, stage_dir):
    n = stage_number(stage_dir)
    found = [s for s in episode['stages'] if s['kind'] == 'perceive' and stage_number(s['dir']) < n
             and (ep_dir / s['dir'] / 'object_cloud.npz').exists()]
    return ep_dir / found[-1]['dir'] if found else None


def mean_estimate(fit):
    q = fit['mean']
    return dict(m=q.get('m'), zc=q.get('zc'), hx=q['horizontal_first_moment_kg_m'],
                hz=q['vertical_first_moment_kg_m'], flags=';'.join(fit['quality_flags']))


def fixed_fits(ep_dir, stage_dir, reference, vision_dir=None):
    """Production estimator for one stage: {'proprio': ..., 'vision': ...} or {'error': ...}."""
    rec = next(s for s in analyze_episode(ep_dir / 'episode.json', reference, vision=vision_dir is not None,
                                          cache_dir=vision_dir)['stages'] if s['stage'] == stage_dir)
    if rec['status'] != 'estimated':
        return dict(error=rec.get('reason'))
    return {k: mean_estimate(v) for k, v in rec['fits'].items()}


# Base band of the detector cloud used for the outer base edge: above the outward flare near
# the table (~20 mm) and below any overhang (the tipT top bar starts at 100 mm and overhangs
# the base by 60 mm; the whole cloud would put the "edge" there).
BASE_BAND_M = (0.020, 0.045)


def near_envelope_x(cloud, centroid, pivot):
    """World x of the object's outer BASE edge on the pivot side: 98th percentile of the
    cloud's extent toward the pivot, using only the base band (BASE_BAND_M)."""
    direction = np.asarray(pivot[:2], float) - centroid[:2]
    direction /= np.linalg.norm(direction)
    h = cloud[:, 2] - hull_com.TABLE_Z_BASE_LINK
    wall = cloud[(h > BASE_BAND_M[0]) & (h < BASE_BAND_M[1])]
    extent = np.quantile((wall[:, :2] - centroid[:2]) @ direction, .98)
    return float(centroid[0] + direction[0] * extent), direction


def joint_summary(r):
    s = joint_fit.summarize(r); post = r['geometry']['posterior']
    return dict(m=s['mass_g'] / 1000, m_lo=s['mass_g_95'][0] / 1000, m_hi=s['mass_g_95'][1] / 1000,
                zc=s['height_mm'] / 1000, R_mm=1000 * post['best_R_m'], R95_mm=[1000 * v for v in post['R_95_m']],
                status=r['gates']['status'],
                flags=';'.join(r['gates']['flags'] + [f'{b}:{f}' for b, fl in r['gates']['branch_flags'].items() for f in fl]))


def analyze_stage(ep_dir, episode, stage, sig_reg, vision_dir, ref):
    """All methods for one tip stage. `ref` supplies only d (an input); scoring is separate."""
    row = dict(stage=stage['dir'], kind=stage['kind'], controller_status=stage['status'])
    path = stage_log(ep_dir, stage)
    if path is None:
        return dict(row, status='no log')
    fw = stage['kind'] != 'press_pull_tip'
    mode = 'forward' if fw else 'press_pull'
    row['log'] = str(path)
    try:
        log = load_fwd_tip(str(path)) if fw else tiplog_from_press_pull(str(path))
    except Exception as exc:
        return dict(row, status=f'log unreadable: {exc}')
    perceive = perceive_before(ep_dir, episode, stage['dir'])
    row['perceive'] = perceive.name if perceive else None
    results = {}

    def attempt(name, fn):
        try:
            results[name] = fn()
        except Exception as exc:
            results[name] = dict(error=f'{type(exc).__name__}: {exc}')

    d_env = ref['d_env'][mode] if ref else None
    row['supplied_d_mm'] = 1000 * d_env if d_env is not None else None
    if ref is not None and d_env is not None:
        fits = fixed_fits(ep_dir, stage['dir'], references.estimator_reference(ref, key='d_env'), vision_dir)
        results['fixed_supplied'] = fits.get('proprio', fits)
        if 'vision' in fits:
            results['fixed_supplied_vision'] = fits['vision']
        d_support = ref['d'][mode]
        if d_support is not None and abs(d_support - d_env) > 1e-9:
            attempt('fixed_oracle', lambda: fixed_fits(ep_dir, stage['dir'],
                                                         references.estimator_reference(ref, key='d')).get('proprio'))
    if perceive is None:
        row['results'] = results
        row['status'] = 'processed'
        return row
    reg = check_folder(perceive)
    row['camera_agreement'] = reg['status']
    row['camera_spread_mm'] = reg.get('spread_m', np.nan) * 1000
    cloud = np.load(perceive / 'object_cloud.npz')['xyz'].astype(float)
    centroid, volume = hull_com.hull_centroid(cloud)
    d_hull = hull_com.horizontal_distance(centroid, log.pivot, log.tilt_dir)
    row.update(hull_d_mm=1000 * d_hull, hull_volume_cm3=volume * 1e6)
    if sig_reg:
        # Known before contact: camera registration error relative to the prior arm (mass ~ 1/d).
        row['predicted_mass_rel_sigma_registration'] = sig_reg / d_hull
    attempt('fixed_hull', lambda: fixed_fits(ep_dir, stage['dir'], references.estimator_reference(None, d_hull))
            .get('proprio') or dict(error='no fit'))
    if fw:
        row['results'] = results
        row['status'] = 'processed'
        return row
    has_scene = (perceive / 'support_scene_cloud.npz').exists()
    try:
        # The per-camera scene profile locates the wall to ~1 mm; the fused detector cloud
        # only to ~4 mm (5 mm voxels, merged cameras). Prefer the scene when recorded.
        env_x, direction = joint_fit.outer_face_x(perceive, log.pivot) if has_scene else \
            near_envelope_x(cloud, centroid, log.pivot)
        row['envelope_source'] = 'scene_profile' if has_scene else 'detector_cloud'
    except Exception:
        env_x, direction = near_envelope_x(cloud, centroid, log.pivot)
        row['envelope_source'] = 'detector_cloud'
    row['near_envelope_x'] = env_x

    def motion(com_x):
        o = orange.estimate(log, com_x)
        return dict(m=o['mass_kg'], zc=o['height_m'], split=o['branch_mass_split'])
    attempt('motion_hull', lambda: motion(float(centroid[0])))
    if d_env is not None:
        attempt('motion_supplied', lambda: motion(env_x - direction[0] * d_env))
    if has_scene:
        track = None
        if vision_dir is not None:
            try:
                from irb120_control.estimation.tilt_vision import track_tilt_vision
                track = track_tilt_vision(log, cache_dir=str(vision_dir), verbose=False)
            except Exception:
                track = None
        attempt('joint_cloud', lambda: joint_summary(joint_fit.run(episode['object'], path, perceive, sig_reg,
                                                                   vision_track=track)[0]))
        if d_env is not None:
            attempt('joint_supplied', lambda: joint_summary(joint_fit.run(episode['object'], path, perceive, sig_reg,
                                                                          vision_track=track, envelope_d=d_env)[0]))
    row['results'] = results
    row['status'] = 'processed'
    return row


def score(rows, ref):
    """After every estimate of the episode exists: errors against mass/height, where known."""
    for row in rows:
        mode = 'press_pull' if row['kind'] == 'press_pull_tip' else 'forward'
        d, sd = ref['d_env'].get(mode), ref['d_sigma'].get(mode)
        for method, q in row.get('results', {}).items():
            if not isinstance(q, dict):
                continue
            if method.endswith('supplied') and d and sd and q.get('m') is not None:
                q['mass_rel_sigma_from_d'] = sd / d       # mass ~ h_x / d
            if q.get('m') is not None and np.isfinite(q['m']) and ref.get('mass'):
                q['mass_error_pct'] = 100 * (q['m'] / ref['mass'] - 1)
            if q.get('zc') is not None and np.isfinite(q['zc']) and ref.get('height'):
                q['height_error_pct'] = 100 * (q['zc'] / ref['height'] - 1)


def process_episode(ep_file, data_root, sig_reg, vision_dir, label=None):
    """Rows for every tip stage, the endpoint row (or None), and crash records.
    `label` overrides the episode's object label for the input/reference lookup."""
    ep = json.loads(Path(ep_file).read_text())
    label = label or ep['object']
    ref = references.lookup(label)
    rows, failures = [], []
    for stage in ep['stages']:
        if stage['kind'] not in TIP_KINDS:
            continue
        try:
            row = analyze_stage(Path(ep_file).parent, ep, stage, sig_reg, vision_dir, ref)
        except Exception:
            row = dict(stage=stage['dir'], kind=stage['kind'], status='analysis crashed', results={})
            failures.append(dict(episode=str(ep_file), stage=stage['dir'], traceback=traceback.format_exc()))
        row.update(object=label, recorded_label=ep['object'], episode=Path(ep_file).parent.name,
                   set=str(Path(ep_file).parent.parent.parent.relative_to(data_root)),
                   episode_outcome=(ep.get('outcome') or {}).get('result', 'unfinished'),
                   git_commit=ep.get('git_commit'), git_dirty=ep.get('git_dirty'), official='official' in ep,
                   tier=ref['tier'] if ref else 2, reference_source=ref['source'] if ref else None)
        rows.append(row)
    if ref:
        score(rows, ref)
    done = [r for r in rows if r.get('status') == 'processed']
    return rows, (done[-1] if done else None), failures


def scene_folders(episodes):
    out = []
    for ep_file in episodes:
        ep = json.loads(Path(ep_file).read_text())
        out += [Path(ep_file).parent / s['dir'] for s in ep['stages'] if s['kind'] == 'perceive'
                and (Path(ep_file).parent / s['dir'] / 'support_scene_cloud.npz').exists()]
    return out


def pooled_registration(scene_folders):
    sig = []
    for folder in scene_folders:
        try:
            s = analyze_scene(folder, np.load(folder / 'object_cloud.npz')['xyz'])
            q = joint_fit.registration_sigma([v for v in s['views'] if v['status'] == 'ok'
                                              and v['elevation_deg'] <= joint_fit.MAX_ELEVATION_DEG])
            if q is not None:
                sig.append(q)
        except Exception:
            continue
    return float(np.median(sig)) if sig else None


def flatten(rows):
    flat = []
    for r in rows:
        base = {k: v for k, v in r.items() if k != 'results'}
        for method in METHODS:
            q = r.get('results', {}).get(method) or {}
            for k, v in q.items():
                base[f'{method}.{k}'] = v if not isinstance(v, list) else json.dumps(v)
        flat.append(base)
    return flat


def write_tables(rows, out):
    (out / 'stages.json').write_text(json.dumps(rows, indent=1, default=float) + '\n')
    flat = flatten(rows)
    keys = sorted({k for r in flat for k in r}, key=lambda k: (k.count('.'), k))
    with open(out / 'stages.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(flat)


def git_state():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_paths.REPO, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=_paths.REPO, text=True).strip())
        return dict(analysis_commit=commit, analysis_dirty=dirty)
    except Exception:
        return dict(analysis_commit=None, analysis_dirty=None)


def figures(endpoints, out, methods=PAPER_CANDIDATES):
    tier1 = [r for r in endpoints if r['tier'] == 1]
    if tier1:
        objects = sorted({r['object'] for r in tier1})
        fig, axes = plt.subplots(2, 1, figsize=(max(9, 1.2 * len(objects)), 8), sharex=True, layout='constrained')
        for ax, key, label in zip(axes, ('mass_error_pct', 'height_error_pct'), ('Mass error (%)', 'CoM height error (%)')):
            for j, method in enumerate(methods):
                xs, ys = [], []
                for i, obj in enumerate(objects):
                    for r in tier1:
                        q = r.get('results', {}).get(method) or {}
                        if r['object'] == obj and q.get(key) is not None:
                            xs.append(i + (j - (len(methods) - 1) / 2) * 0.12); ys.append(q[key])
                ax.plot(xs, ys, 'o', ms=4, alpha=.8, label=method)
            ax.axhline(0, color='gray', ls=':'); ax.set_ylabel(label); ax.grid(alpha=.2)
            ax.set_ylim(max(ax.get_ylim()[0], -80), min(ax.get_ylim()[1], 80))
        axes[1].set_xticks(range(len(objects)), objects, rotation=45, ha='right')
        axes[0].legend(fontsize=8, ncol=3)
        axes[0].set_title('Tier 1: one point per trial endpoint (clipped at ±80%)')
        fig.savefig(out / 'tier1_errors.png', dpi=150); fig.savefig(out / 'tier1_errors.pdf'); plt.close(fig)
    tier2 = [r for r in endpoints if r['tier'] == 2]
    if tier2:
        objects = sorted({r['object'] for r in tier2})
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
        for ax, key, label, f in zip(axes, ('m', 'hx', 'hz'), ('Mass (g): supplied d (o), hull (x)',
                                     'h_x = m d (g mm), identifiable', 'h_z = m z (g mm), identifiable'), (1e3, 1e6, 1e6)):
            for i, obj in enumerate(objects):
                for method, marker in (('fixed_supplied', 'o'), ('fixed_hull', 'x')):
                    vals = [r['results'][method][key] * f for r in tier2 if r['object'] == obj
                            and (r.get('results', {}).get(method) or {}).get(key) is not None]
                    ax.plot([i] * len(vals), vals, marker, alpha=.8, color='C0' if marker == 'o' else 'C1')
            ax.set_xticks(range(len(objects)), objects, rotation=45, ha='right'); ax.set_title(label, fontsize=9); ax.grid(alpha=.2)
        fig.suptitle('Tier 2: estimates only (no reference); spread across trials is repeatability')
        fig.savefig(out / 'tier2_estimates.png', dpi=150); fig.savefig(out / 'tier2_estimates.pdf'); plt.close(fig)


def summary_table(endpoints, methods=METHODS):
    lines = ['| Object | Tier | Method | Estimated/trials | Mass (g) mean ± sd | Mass err (%) mean ± sd | Height (mm) mean ± sd | Height err (%) mean ± sd |',
             '|---|---|---|---|---|---|---|---|']
    fmt = lambda v: f'{np.mean(v):.1f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.1f}' if v else '—'
    for obj in sorted({r['object'] for r in endpoints}):
        rows = [r for r in endpoints if r['object'] == obj]
        for method in methods:
            qs = [r.get('results', {}).get(method) or {} for r in rows]
            m = [q['m'] * 1000 for q in qs if q.get('m') is not None and np.isfinite(q['m'])]
            if not m:
                continue
            me = [q['mass_error_pct'] for q in qs if 'mass_error_pct' in q]
            z = [q['zc'] * 1000 for q in qs if q.get('zc') is not None and np.isfinite(q['zc'])]
            ze = [q['height_error_pct'] for q in qs if 'height_error_pct' in q]
            lines.append(f'| {obj} | {rows[0]["tier"]} | {method} | {len(m)}/{len(rows)} | {fmt(m)} | {fmt(me)} | {fmt(z)} | {fmt(ze)} |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, default=_paths.OFFICIAL_ROOT / 'raw')
    p.add_argument('--out', type=Path, default=None, help='default: <data root>/../results')
    p.add_argument('--vision', action='store_true', help='also compute camera tilt tracks (slow; cached in <out>/vision_cache)')
    p.add_argument('--limit', type=int, default=None, help='process at most this many episodes (smoke test)')
    a = p.parse_args()
    root = a.data_root.expanduser()
    out = (a.out or root.parent / 'results').expanduser(); out.mkdir(parents=True, exist_ok=True)
    vision_dir = (out / 'vision_cache') if a.vision else None
    if vision_dir:
        vision_dir.mkdir(exist_ok=True)
    episodes = sorted(root.glob('**/episodes/*/episode.json'))[:a.limit]
    sig_reg = pooled_registration(scene_folders(episodes))
    print(f'{len(episodes)} episodes under {root}; pooled registration sigma {sig_reg}')
    rows, endpoints, failures = [], [], []
    for ep_file in episodes:
        ep_rows, endpoint, fails = process_episode(ep_file, root, sig_reg, vision_dir)
        rows += ep_rows; failures += fails
        if endpoint:
            endpoints.append(endpoint)
        print(f"  {ep_rows[0]['object'] if ep_rows else '-':<22} {ep_file.parent.name:<22} {len(ep_rows)} tip stage(s)")
    write_tables(rows, out)
    figures(endpoints, out)
    provenance = dict(generated=datetime.datetime.now().isoformat(timespec='seconds'), data_root=str(root),
                      episodes=len(episodes), tip_stages=len(rows), vision=a.vision, pooled_registration_sigma_m=sig_reg,
                      dirty_episodes=sorted({r['episode'] for r in rows if r.get('git_dirty')}),
                      non_official_episodes=sorted({r['episode'] for r in rows if not r.get('official')}),
                      analysis_failures=len(failures), **git_state())
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=1) + '\n')
    (out / 'analysis_failures.json').unlink(missing_ok=True)      # never leave a stale one
    if failures:
        (out / 'analysis_failures.json').write_text(json.dumps(failures, indent=1) + '\n')
    (out / 'summary.md').write_text(
        f"# Results summary\n\nGenerated {provenance['generated']} from `{root}` by `make_paper_results.py` "
        f"(analysis commit {str(provenance['analysis_commit'])[:10]}{' DIRTY' if provenance['analysis_dirty'] else ''}).\n\n"
        f"Endpoints: last tip stage per episode. Methods: see the script docstring.\n\n"
        f"{summary_table(endpoints)}\n\n"
        f"{len(provenance['dirty_episodes'])} episode(s) recorded from a dirty checkout; "
        f"{len(provenance['non_official_episodes'])} not recorded with --official; {len(failures)} analysis failure(s).\n")
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
