"""Regenerate every results table and figure from one data root. Never writes into raw data.

    python3 make_paper_results.py --data-root ~/Documents/irb120_data/official_v1/raw \
        --out ~/Documents/irb120_data/official_v1/results [--vision]

Every tip stage of every episode under the root is processed with each method:

  reference_d  production estimator (episode_estimation.analyze_episode) with the measured
               horizontal CoM distance. Tier 1 only; this is where errors come from.
  hull_d       the same estimator with d from the filled convex hull of the pre-action cloud
               (GT-free; uniform-density ASSUMPTION). Its first moments h_x = m*d and
               h_z = m*z are identifiable regardless of d.
  orange       shape-free instantaneous support (press-pull only), CoM on the hull centroid.
  joint        cloud prior + interaction posterior support geometry, then mass with an
               uncertainty band and gates (press-pull with a support_scene_cloud only).

Also per stage, from the pre-contact prior alone: predicted_mass_rel_sigma_registration =
(camera registration sigma) / (hull arm d). Short-arm objects (cylinder E: ~10 mm) are
flagged as poorly identifiable before any interaction runs.

Scoring references are looked up only after all estimates for an episode exist. Each
episode's endpoint is its last tip stage (the cascade's final attempt); all stages are kept
in stages.csv. Episodes recorded from a dirty checkout are flagged, not dropped.
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
METHODS = ('reference_d', 'hull_d', 'orange', 'joint')


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


def mean_estimate(rec):
    q = rec['estimate']
    return dict(m=q['mean'].get('m'), zc=q['mean'].get('zc'), hx=q['mean']['horizontal_first_moment_kg_m'],
                hz=q['mean']['vertical_first_moment_kg_m'], flags=';'.join(q['quality_flags']))


def analyze_stage(ep_dir, episode, stage, sig_reg, vision_dir):
    row = dict(stage=stage['dir'], kind=stage['kind'], controller_status=stage['status'])
    path = stage_log(ep_dir, stage)
    if path is None:
        return dict(row, status='no log')
    fw = stage['kind'] != 'press_pull_tip'
    row['log'] = str(path)
    try:
        log = load_fwd_tip(str(path)) if fw else tiplog_from_press_pull(str(path))
    except Exception as exc:
        return dict(row, status=f'log unreadable: {exc}')
    perceive = perceive_before(ep_dir, episode, stage['dir'])
    row['perceive'] = perceive.name if perceive else None
    if perceive is not None:
        reg = check_folder(perceive)
        row['camera_agreement'] = reg['status']
        row['camera_spread_mm'] = reg.get('spread_m', np.nan) * 1000
    # GT-free: d from the filled hull of the pre-action cloud.
    results = {}
    if perceive is not None:
        try:
            centroid, volume = hull_com.hull_centroid(np.load(perceive / 'object_cloud.npz')['xyz'])
            d = hull_com.horizontal_distance(centroid, log.pivot, log.tilt_dir)
            row.update(hull_d_mm=1000 * d, hull_volume_cm3=volume * 1e6)
            if sig_reg:
                # Known before contact: camera registration error relative to the prior arm.
                # mass ~ 1/d, so this is the relative mass uncertainty from registration alone.
                row['predicted_mass_rel_sigma_registration'] = sig_reg / d
            rec = next(s for s in analyze_episode(ep_dir / 'episode.json', references.estimator_reference(None, d),
                                                  vision=vision_dir is not None, cache_dir=vision_dir)['stages']
                       if s['stage'] == stage['dir'])
            results['hull_d'] = mean_estimate(rec) if rec['status'] == 'estimated' else dict(error=rec.get('reason'))
            if not fw:
                o = orange.estimate(log, float(centroid[0]))
                results['orange'] = dict(m=o['mass_kg'], zc=o['height_m'], split=o['branch_mass_split'])
        except Exception as exc:
            results.setdefault('hull_d', dict(error=str(exc)))
    if not fw and perceive is not None and (perceive / 'support_scene_cloud.npz').exists():
        try:
            track = None
            if vision_dir is not None:
                from irb120_control.estimation.tilt_vision import track_tilt_vision
                track = track_tilt_vision(log, cache_dir=str(vision_dir), verbose=False)
            r, _, _ = joint_fit.run(episode['object'], path, perceive, sig_reg, vision_track=track)
            s = joint_fit.summarize(r); post = r['geometry']['posterior']
            results['joint'] = dict(m=s['mass_g'] / 1000, m_lo=s['mass_g_95'][0] / 1000, m_hi=s['mass_g_95'][1] / 1000,
                                    zc=s['height_mm'] / 1000, R_mm=1000 * post['best_R_m'],
                                    R95_mm=[1000 * v for v in post['R_95_m']], status=r['gates']['status'],
                                    flags=';'.join(r['gates']['flags'] + [f'{b}:{f}' for b, fl in r['gates']['branch_flags'].items() for f in fl]))
        except Exception as exc:
            results['joint'] = dict(error=str(exc))
    row['results'] = results
    row['status'] = 'processed'
    return row


def score_episode(ep_file, rows, ref, vision_dir):
    """After all estimates exist: production estimator with the reference d, then errors."""
    try:
        scored = {s['stage']: s for s in analyze_episode(ep_file, references.estimator_reference(ref),
                                                         vision=vision_dir is not None, cache_dir=vision_dir)['stages']}
    except Exception as exc:
        scored = {}
        for row in rows:
            row.setdefault('results', {})['reference_d'] = dict(error=str(exc))
    for row in rows:
        if 'results' not in row:
            continue
        rec = scored.get(row['stage'])
        if rec is not None:
            row['results']['reference_d'] = (mean_estimate(rec) if rec['status'] == 'estimated'
                                             else dict(error=rec.get('reason')))
        for method, q in row['results'].items():
            if q.get('m') is not None and np.isfinite(q['m']):
                q['mass_error_pct'] = 100 * (q['m'] / ref['mass'] - 1)
            if q.get('zc') is not None and np.isfinite(q['zc']) and ref.get('height'):
                q['height_error_pct'] = 100 * (q['zc'] / ref['height'] - 1)


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
            q = r.get('results', {}).get(method, {})
            for k, v in q.items():
                base[f'{method}.{k}'] = v if not isinstance(v, list) else json.dumps(v)
        flat.append(base)
    return flat


def git_state():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=_paths.REPO, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=_paths.REPO, text=True).strip())
        return dict(analysis_commit=commit, analysis_dirty=dirty)
    except Exception:
        return dict(analysis_commit=None, analysis_dirty=None)


def figures(endpoints, out):
    tier1 = [r for r in endpoints if r['tier'] == 1]
    if tier1:
        objects = sorted({r['object'] for r in tier1})
        fig, axes = plt.subplots(2, 1, figsize=(max(8, 1.1 * len(objects)), 8), sharex=True, layout='constrained')
        for ax, key, label in zip(axes, ('mass_error_pct', 'height_error_pct'), ('Mass error (%)', 'CoM height error (%)')):
            for j, method in enumerate(METHODS):
                xs, ys = [], []
                for i, obj in enumerate(objects):
                    for r in tier1:
                        q = r.get('results', {}).get(method, {})
                        if r['object'] == obj and q.get(key) is not None:
                            xs.append(i + (j - 1.5) * 0.15); ys.append(q[key])
                ax.plot(xs, ys, 'o', ms=5, alpha=.8, label=method)
            ax.axhline(0, color='gray', ls=':'); ax.set_ylabel(label); ax.grid(alpha=.2)
        axes[1].set_xticks(range(len(objects)), objects, rotation=45, ha='right')
        axes[0].legend(fontsize=8, ncol=4)
        axes[0].set_title('Tier 1: one point per trial endpoint (repeats shown individually)')
        fig.savefig(out / 'tier1_errors.png', dpi=150); fig.savefig(out / 'tier1_errors.pdf'); plt.close(fig)
    tier2 = [r for r in endpoints if r['tier'] == 2]
    if tier2:
        objects = sorted({r['object'] for r in tier2})
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
        for ax, key, label, f in zip(axes, ('m', 'hx', 'hz'), ('Mass, hull-centroid assumption (g)',
                                     'h_x = m d (g mm), identifiable', 'h_z = m z (g mm), identifiable'), (1e3, 1e6, 1e6)):
            for i, obj in enumerate(objects):
                vals = [r['results']['hull_d'][key] * f for r in tier2 if r['object'] == obj
                        and r.get('results', {}).get('hull_d', {}).get(key) is not None]
                ax.plot([i] * len(vals), vals, 'o', alpha=.8)
            ax.set_xticks(range(len(objects)), objects, rotation=45, ha='right'); ax.set_title(label, fontsize=9); ax.grid(alpha=.2)
        fig.suptitle('Tier 2: estimates only (no reference); spread across trials is repeatability')
        fig.savefig(out / 'tier2_estimates.png', dpi=150); fig.savefig(out / 'tier2_estimates.pdf'); plt.close(fig)


def summary_table(endpoints):
    lines = ['| Object | Tier | Trials | Method | Mass (g) mean ± sd | Mass err (%) mean ± sd | Height (mm) mean ± sd | Height err (%) mean ± sd |',
             '|---|---|---|---|---|---|---|---|']
    for obj in sorted({r['object'] for r in endpoints}):
        rows = [r for r in endpoints if r['object'] == obj]
        for method in METHODS:
            qs = [r.get('results', {}).get(method, {}) for r in rows]
            m = [q['m'] * 1000 for q in qs if q.get('m') is not None and np.isfinite(q['m'])]
            if not m:
                continue
            fmt = lambda v: f'{np.mean(v):.1f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.1f}' if v else '—'
            me = [q['mass_error_pct'] for q in qs if 'mass_error_pct' in q]
            z = [q['zc'] * 1000 for q in qs if q.get('zc') is not None and np.isfinite(q['zc'])]
            ze = [q['height_error_pct'] for q in qs if 'height_error_pct' in q]
            lines.append(f'| {obj} | {rows[0]["tier"]} | {len(m)}/{len(rows)} | {method} | {fmt(m)} | {fmt(me)} | {fmt(z)} | {fmt(ze)} |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, default=_paths.OFFICIAL_ROOT / 'raw')
    p.add_argument('--out', type=Path, default=None, help='default: <data root>/../results')
    p.add_argument('--vision', action='store_true', help='also compute camera tilt tracks (slow; cached in <out>/vision_cache)')
    p.add_argument('--limit', type=int, default=None, help='process at most this many episodes (smoke test)')
    a = p.parse_args()
    out = (a.out or a.data_root.parent / 'results').expanduser(); out.mkdir(parents=True, exist_ok=True)
    vision_dir = (out / 'vision_cache') if a.vision else None
    if vision_dir:
        vision_dir.mkdir(exist_ok=True)
    episodes = sorted(a.data_root.expanduser().glob('**/episodes/*/episode.json'))[:a.limit]
    print(f'{len(episodes)} episodes under {a.data_root}')
    scenes = []
    for ep_file in episodes:
        ep = json.loads(ep_file.read_text())
        scenes += [ep_file.parent / s['dir'] for s in ep['stages'] if s['kind'] == 'perceive'
                   and (ep_file.parent / s['dir'] / 'support_scene_cloud.npz').exists()]
    sig_reg = pooled_registration(scenes)
    print(f'{len(scenes)} scene clouds; pooled registration sigma {sig_reg}')
    rows, endpoints, failures = [], [], []
    for ep_file in episodes:
        ep = json.loads(ep_file.read_text())
        ep_rows = []
        for stage in ep['stages']:
            if stage['kind'] not in TIP_KINDS:
                continue
            try:
                row = analyze_stage(ep_file.parent, ep, stage, sig_reg, vision_dir)
            except Exception:
                row = dict(stage=stage['dir'], kind=stage['kind'], status='analysis crashed')
                failures.append(dict(episode=str(ep_file), stage=stage['dir'], traceback=traceback.format_exc()))
            row.update(object=ep['object'], episode=ep_file.parent.name, set=str(ep_file.parent.parent.parent.relative_to(a.data_root.expanduser())),
                       episode_outcome=(ep.get('outcome') or {}).get('result', 'unfinished'),
                       git_commit=ep.get('git_commit'), git_dirty=ep.get('git_dirty'), official='official' in ep)
            ep_rows.append(row)
        # Scoring only now, after every estimate for this episode exists.
        ref = references.lookup(ep['object'])
        for row in ep_rows:
            row['tier'] = 1 if ref else 2
            row['reference_source'] = ref['source'] if ref else None
        if ref:
            score_episode(ep_file, ep_rows, ref, vision_dir)
        rows += ep_rows
        done = [r for r in ep_rows if r.get('status') == 'processed']
        if done:
            endpoints.append(done[-1])
        print(f"  {ep['object']:<22} {ep_file.parent.name:<22} {len(ep_rows)} tip stage(s)")
    (out / 'stages.json').write_text(json.dumps(rows, indent=1, default=float) + '\n')
    flat = flatten(rows)
    keys = sorted({k for r in flat for k in r}, key=lambda k: (k.count('.'), k))
    with open(out / 'stages.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(flat)
    figures(endpoints, out)
    provenance = dict(generated=datetime.datetime.now().isoformat(timespec='seconds'), data_root=str(a.data_root),
                      episodes=len(episodes), tip_stages=len(rows), vision=a.vision, pooled_registration_sigma_m=sig_reg,
                      dirty_episodes=sorted({r['episode'] for r in rows if r.get('git_dirty')}),
                      non_official_episodes=sorted({r['episode'] for r in rows if not r.get('official')}),
                      analysis_failures=len(failures), **git_state())
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=1) + '\n')
    if failures:
        (out / 'analysis_failures.json').write_text(json.dumps(failures, indent=1) + '\n')
    (out / 'summary.md').write_text(
        f"# Results summary\n\nGenerated {provenance['generated']} from `{a.data_root}` by `make_paper_results.py` "
        f"(analysis commit {str(provenance['analysis_commit'])[:10]}{' DIRTY' if provenance['analysis_dirty'] else ''}).\n\n"
        f"Endpoints: last tip stage per episode. Methods: {', '.join(METHODS)} (see the script docstring). "
        f"`hull_d` and `orange` masses rest on the filled-hull uniform-density assumption.\n\n"
        f"{summary_table(endpoints)}\n\n"
        f"{len(provenance['dirty_episodes'])} episode(s) recorded from a dirty checkout; "
        f"{len(provenance['non_official_episodes'])} not recorded with --official; {len(failures)} analysis failure(s).\n")
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
