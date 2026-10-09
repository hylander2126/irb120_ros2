"""Run offline O0 checks without changing the estimator or source logs.

Run from the control package with ``python3 -m
irb120_control.estimation.uncertainty_baseline --help``.
"""

import argparse
from contextlib import redirect_stdout
import gzip
import json
from pathlib import Path

import numpy as np

from irb120_control.estimation import estimate_params as estimator


PUBLISHED_OBJECTS = ('box', 'heart', 'flashlight', 'monitor')
RECENT_OBJECTS = ('cylinder_e', 'flashlight', 'heart_a', 'tipT_a')


def json_ready(value):
    """Keep output valid JSON, including failed/non-finite diagnostics."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def fit_trials(obj, paths, mode):
    """Use the same pivot preparation, fits and aggregation as run_object."""
    pivots = estimator._relative_pivots(obj, paths) if mode == 'relative' else {}
    fits, skipped = [], []
    for index, path in enumerate(paths):
        try:
            fit = estimator._run_estimation(
                obj, str(path.parent), str(path), None, verbose=False,
                trial_label=str(path), pivot_mode=mode,
                pivot_override=pivots.get(index))
            fits.append(fit)
        except (KeyError, ValueError, IndexError) as exc:
            skipped.append({'path': str(path), 'reason': str(exc)})
    return {
        'object': obj, 'pivot_mode': mode,
        'inputs': [str(path) for path in paths], 'skipped': skipped,
        'trials': fits,
        'summary': estimator.aggregate_trials(obj, fits) if fits else None,
    }


def verify_csv_mapping(paths, csv_dir):
    """Verify the plan's filename-order mapping against every exported fx row."""
    rows = []
    usable = []
    for path in paths:
        with np.load(path) as data:
            if data['fx'].size and data['pose_time_s'].size:
                usable.append(path)
    csv_paths = sorted(csv_dir.glob('trial_*_ft.csv.gz'))
    if len(usable) != 10 or len(csv_paths) != 10:
        raise ValueError(f'Expected 10 raw and CSV trials: {csv_dir}')
    for path, csv_path in zip(usable, csv_paths):
        with gzip.open(csv_path, 'rt') as stream:
            exported = np.genfromtxt(stream, delimiter=',', names=True)
        with np.load(path) as data:
            # The export rounds channels to five decimal places.
            np.testing.assert_allclose(data['fx'], exported['fx'], rtol=0, atol=5.00001e-6)
            rows.append({'npz': str(path), 'csv': str(csv_path),
                         'rows': len(data['fx'])})
    return rows


def main():
    """Write a self-contained baseline report and machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published-root', type=Path, required=True,
                        help='press-pull-tipping repository (read only)')
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New experiment directory beneath runtime_logs/uncertainty_checks')
    parser.add_argument('--recent-date', default='20261008')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    published = args.published_root.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(workspace / 'runtime_logs' / 'uncertainty_checks'):
        parser.error('--output must be beneath workspace/runtime_logs/uncertainty_checks')
    output.mkdir(parents=True, exist_ok=False)
    results, mapping, unsupported = [], {}, []
    with (output / 'estimator_stdout.txt').open('w') as transcript, redirect_stdout(transcript):
        for obj in PUBLISHED_OBJECTS:
            paths = [Path(p).resolve() for p in estimator.find_trial_logs(obj, str(published))]
            mapping[obj] = verify_csv_mapping(paths, published / 'code' / 'data' / obj)
            for mode in ('relative', 'trajectory', 'fixed'):
                result = fit_trials(obj, paths, mode)
                result['dataset'] = 'published'
                results.append(result)
        for obj in RECENT_OBJECTS:
            paths = sorted((workspace / 'runtime_logs' / obj / 'episodes').glob(
                f'*/*_press_pull_tip/arc_static_{args.recent_date}_*.npz'))
            paths = [path.resolve() for path in paths]
            if obj not in estimator.GROUND_TRUTH:
                unsupported.append({'object': obj, 'inputs': [str(p) for p in paths],
                                    'reason': 'Known CoM-x required; no GROUND_TRUTH entry.'})
                continue
            result = fit_trials(obj, paths, 'fixed')
            result['dataset'] = args.recent_date
            results.append(result)
    payload = {'estimator': str(Path(estimator.__file__).resolve()),
               'script': str(Path(__file__).resolve()),
               'csv_mapping': mapping, 'results': results, 'unsupported': unsupported}
    (output / 'results.json').write_text(json.dumps(json_ready(payload), indent=2, allow_nan=False) + '\n')
    lines = ['# O0 baseline', '',
             'Workspace estimator; mean ± sample standard deviation across usable trials.',
             'For N=1, the estimator reports zero spread; repeatability is not measured.',
             'Published headline: relative. Recent headline: fixed. No friction refits.', '',
             '| Dataset | Object | Pivot | N | Mass (kg) | CoM z (cm) | Skipped |',
             '|---|---|---|---:|---:|---:|---:|']
    for result in results:
        summary = result['summary']
        if summary:
            mass, z = summary['spread']['mass'], summary['spread']['com_z']
            lines.append(f"| {result['dataset']} | {result['object']} | {result['pivot_mode']} | "
                         f"{summary['n_trials']} | {mass['mean']:.4f} ± {mass['std']:.4f} | "
                         f"{100*z['mean']:.2f} ± {100*z['std']:.2f} | {len(result['skipped'])} |")
        else:
            lines.append(f"| {result['dataset']} | {result['object']} | {result['pivot_mode']} | 0 | — | — | {len(result['skipped'])} |")
    lines += ['', 'All 40 usable published logs match CSV fx values (within export rounding) and row counts.',
              'Full absolute input paths, per-trial fits and skip reasons: `results.json`.',
              'Estimator output: `estimator_stdout.txt`.', '',
              'Unsupported recent objects (no ground-truth entries were invented):']
    lines += [f"- {row['object']}: {len(row['inputs'])} files; {row['reason']}" for row in unsupported]
    lines += ['', f"Script: `{Path(__file__).resolve()}`",
              f"Estimator: `{Path(estimator.__file__).resolve()}`", '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    print(output / 'REPORT.md')


if __name__ == '__main__':
    main()
