"""Compare every estimation method on all development data, to choose the headline method.

    python3 compare_methods.py [--out DIR] [--vision]

Data (read-only):
  ~/Documents/irb120_data/dev_2026-10/{tipT_suite, tipT_full_suite_old, tipT_c_debugging,
                                       cylinder_suite, cylinder_a_pivot_test}
  ~/Documents/github/press-pull-tipping/runtime_logs/{box,heart,flashlight,monitor}/arc_squash
      the 40 published press-pull trials (no perception clouds, legacy pivot convention)

Methods are those of make_paper_results.py. Legacy trials have no clouds, so only the
support models are compared there: fixed (logged pivot) and motion (support from ball motion),
both with the supplied d. The height cap is set to 1 m since no cloud bound exists.

Repeat labels are mapped to the configuration they were run as (ALIASES, from the audits:
c1-c3 and the arc runs are repeats of tipT_c; cylinder_a_pivot_test is cylinder_a).
One endpoint per episode (the cascade's final stage), as the paper would report.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

import _paths
import make_paper_results as mp
import orange
from irb120_control.estimation.estimate_params import GROUND_TRUTH, tiplog_from_press_pull
from irb120_control.estimation.physical_bounds import bounded_gravity_fit
from irb120_control.estimation.tip_core import TILT_DEADBAND, pivot_wrench, tilt_noslip

SETS = ('tipT_suite', 'tipT_full_suite_old', 'tipT_c_debugging', 'cylinder_suite', 'cylinder_a_pivot_test')
ALIASES = {'tipT_c1': 'tipT_c', 'tipT_c2': 'tipT_c', 'tipT_c3': 'tipT_c', 'tipT_c_arc': 'tipT_c',
           'tipT_c_arc_2': 'tipT_c', 'tipT_c_arc_3': 'tipT_c', 'cylinder_a_pivot_test': 'cylinder_a'}
LEGACY = Path.home() / 'Documents/github/press-pull-tipping/runtime_logs'
LEGACY_OBJECTS = ('box', 'heart', 'flashlight', 'monitor')
# Published Table 2 (relative pivot mode, 10 trials each), for context only.
PUBLISHED_MASS_KG = dict(box=0.7069, heart=0.242, flashlight=0.396, monitor=5.274)
LEGACY_CAP_M = 1.0


def family(row):
    obj = row['object']
    if obj.startswith('tipT'):
        return 'tipT forward' if row['kind'] != 'press_pull_tip' else 'tipT press-pull'
    if obj.startswith('cylinder'):
        return 'cylinder press-pull'
    return f'legacy {obj} press-pull'


def seed_vision_cache(cache):
    """Copy archived camera tracks (plain tracker only; never the CAD-informed 'round' variant)."""
    seen = {}
    for f in sorted(mp._paths.DEV_DATA.rglob('vision_tilt_*.npz')):
        if 'round_vision' in str(f):
            continue
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        if f.name in seen and seen[f.name] != digest:
            seen[f.name] = None          # conflicting copies: recompute instead of guessing
            (cache / f.name).unlink(missing_ok=True)
            continue
        if f.name not in seen:
            seen[f.name] = digest
            if not (cache / f.name).exists():
                shutil.copyfile(f, cache / f.name)     # contents only: the archive is read-only
    return sum(v is not None for v in seen.values())


def legacy_fixed(log, d):
    """fit_first_moments' core with the logged pivot, without the cloud height bound."""
    track = tilt_noslip(log, log.pivot)
    phi = track.rotvec @ log.tilt_dir
    tau = pivot_wrench(log, np.zeros_like(log.p_ball), log.pivot)[:, :3] @ log.tilt_dir
    mask = track.valid & log.contact & np.isfinite(phi) & np.isfinite(tau) & (phi > TILT_DEADBAND)
    fits = []
    for b in ('tip', 'return'):
        sel = mask & (log.branch == b)
        if sel.sum() >= 10:
            fits.append(bounded_gravity_fit(phi[sel], tau[sel], d, 0., LEGACY_CAP_M))
    if not fits:
        raise ValueError('no branch with enough rocking samples')
    return dict(m=float(np.mean([f[0] for f in fits])), zc=float(np.mean([f[1] for f in fits])),
                flags=';'.join(sorted({a for f in fits for a in f[2]})))


def legacy_rows():
    rows = []
    for obj in LEGACY_OBJECTS:
        gt = GROUND_TRUTH[obj]; d = float(gt['com'][0])
        for f in sorted((LEGACY / obj / 'arc_squash').glob('arc_static*_2*.npz')):   # not most_recent.npz (a copy)
            row = dict(object=obj, recorded_label=obj, episode=f.stem, stage='arc_squash', kind='press_pull_tip',
                       set='published_2026-08', tier=1, results={})
            try:
                log = tiplog_from_press_pull(str(f))
            except Exception as exc:
                row['status'] = f'log unreadable: {exc}'
                continue                                   # empty/aborted logs are not trials
            for name, fn in (('fixed_supplied', lambda: legacy_fixed(log, d)),
                             ('motion_supplied', lambda: (lambda o: dict(m=o['mass_kg'], zc=o['height_m'], split=o['branch_mass_split']))(
                                 orange.estimate(log, float(log.pivot[0] + d), height_cap=LEGACY_CAP_M)))):
                try:
                    row['results'][name] = fn()
                except Exception as exc:
                    row['results'][name] = dict(error=f'{type(exc).__name__}: {exc}')
            for q in row['results'].values():
                if q.get('m') is not None:
                    q['mass_error_pct'] = 100 * (q['m'] / gt['mass'] - 1)
                    q['height_error_pct'] = 100 * (q['zc'] / gt['com'][2] - 1)
            row['status'] = 'processed'
            rows.append(row)
    return rows


def stats(values):
    v = np.asarray(values, float)
    if not len(v):
        return None
    return dict(n=len(v), median_abs=float(np.median(np.abs(v))), rms=float(np.sqrt(np.mean(v ** 2))),
                max_abs=float(np.max(np.abs(v))), within_10=int(np.sum(np.abs(v) <= 10)), mean=float(np.mean(v)))


def aggregate(endpoints):
    table = {}
    for fam in sorted({family(r) for r in endpoints}):
        rows = [r for r in endpoints if family(r) == fam]
        for method in mp.METHODS:
            qs = [(r.get('results') or {}).get(method) or {} for r in rows]
            if not any(q for q in qs):
                continue
            mass = [q['mass_error_pct'] for q in qs if 'mass_error_pct' in q]
            height = [q['height_error_pct'] for q in qs if 'height_error_pct' in q]
            accepted = None
            if method.startswith('joint'):
                accepted = sum(q.get('status') == 'accepted' for q in qs)
            table[(fam, method)] = dict(trials=len(rows), estimated=len(mass), mass=stats(mass), height=stats(height),
                                        joint_accepted=accepted)
    return table


def report(table, endpoints, out, n_vision):
    fmt = lambda s, k: '—' if s is None else f"{s[k]:.1f}"
    lines = ['| Family | Method | Estimated / trials | Mass median abs % | Mass RMS % | Mass max abs % | Within 10% | Height median abs % | Height RMS % |',
             '|---|---|---|---|---|---|---|---|---|']
    for (fam, method), q in table.items():
        lines.append(f"| {fam} | {method} | {q['estimated']}/{q['trials']} | {fmt(q['mass'], 'median_abs')} | "
                     f"{fmt(q['mass'], 'rms')} | {fmt(q['mass'], 'max_abs')} | "
                     f"{'—' if q['mass'] is None else q['mass']['within_10']} | {fmt(q['height'], 'median_abs')} | {fmt(q['height'], 'rms')} |")
    per_object = mp.summary_table(endpoints)
    (out / 'COMPARISON_TABLES.md').write_text(
        '# Method comparison tables (development data)\n\nGenerated by `analysis/compare_methods.py`. '
        f'Camera tracks reused from the archive: {n_vision}.\n\n## By geometry family\n\n' + '\n'.join(lines) +
        '\n\n## Per object (endpoints)\n\n' + per_object + '\n\nPublished Table 2 masses (relative pivot): ' +
        ', '.join(f'{k} {v * 1000:.1f} g' for k, v in PUBLISHED_MASS_KG.items()) + '\n')


def figure(endpoints, out):
    fams = sorted({family(r) for r in endpoints})
    methods = [m for m in mp.METHODS if any(((r.get('results') or {}).get(m) or {}).get('mass_error_pct') is not None for r in endpoints)]
    fig, axes = mp.plt.subplots(2, 1, figsize=(15, 9), sharex=True, layout='constrained')
    for ax, key, label in zip(axes, ('mass_error_pct', 'height_error_pct'), ('Mass error (%)', 'CoM height error (%)')):
        for j, method in enumerate(methods):
            for i, fam in enumerate(fams):
                v = [((r.get('results') or {}).get(method) or {}).get(key) for r in endpoints if family(r) == fam]
                v = np.clip([x for x in v if x is not None], -100, 100)
                x = i + (j - (len(methods) - 1) / 2) * 0.1
                ax.plot(np.full(len(v), x), v, 'o', ms=3, alpha=.7, color=f'C{j}', label=method if i == 0 else None)
        ax.axhline(0, color='gray', ls=':'); ax.axhspan(-10, 10, color='gray', alpha=.08)
        ax.set_ylabel(label + ', clipped ±100'); ax.grid(alpha=.2)
    axes[1].set_xticks(range(len(fams)), fams, rotation=20, ha='right')
    axes[0].legend(fontsize=8, ncol=4)
    axes[0].set_title('Every method on every development trial (one point per episode endpoint; shaded ±10%)')
    fig.savefig(out / 'method_comparison.png', dpi=150); fig.savefig(out / 'method_comparison.pdf'); mp.plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=_paths.DATA / 'method_comparison_dev')
    p.add_argument('--vision', action='store_true')
    a = p.parse_args()
    out = a.out.expanduser(); out.mkdir(parents=True, exist_ok=True)
    vision_dir, n_vision = None, 0
    if a.vision:
        vision_dir = out / 'vision_cache'; vision_dir.mkdir(exist_ok=True)
        n_vision = seed_vision_cache(vision_dir)
    root = _paths.DEV_DATA
    episodes = [e for s in SETS for e in sorted((root / s).glob('**/episodes/*/episode.json'))]
    sig_reg = mp.pooled_registration(mp.scene_folders(episodes))
    rows, endpoints, failures = [], [], []
    for ep_file in episodes:
        label = json.loads(ep_file.read_text())['object']
        ep_rows, endpoint, fails = mp.process_episode(ep_file, root, sig_reg, vision_dir, label=ALIASES.get(label, label))
        rows += ep_rows; failures += fails
        if endpoint:
            endpoints.append(endpoint)
        print(f'{label:<24} {ep_file.parent.name:<22} {len(ep_rows)} stage(s)', flush=True)
    legacy = legacy_rows()
    rows += legacy; endpoints += legacy
    print(f'legacy trials: {len(legacy)}', flush=True)
    mp.write_tables(rows, out)
    table = aggregate(endpoints)
    (out / 'comparison.json').write_text(json.dumps({f'{k[0]} | {k[1]}': v for k, v in table.items()}, indent=1) + '\n')
    (out / 'analysis_failures.json').unlink(missing_ok=True)      # never leave a stale one
    if failures:
        (out / 'analysis_failures.json').write_text(json.dumps(failures, indent=1) + '\n')
    report(table, endpoints, out, n_vision)
    figure(endpoints, out)
    print(f'wrote {out}; {len(failures)} failure(s)')


if __name__ == '__main__':
    main()
