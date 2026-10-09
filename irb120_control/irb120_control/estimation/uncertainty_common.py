"""Small shared helpers for read-only source-log uncertainty analyses."""

import argparse
import json
from pathlib import Path

import numpy as np

from irb120_control.estimation import estimate_params as est
from irb120_control.estimation.uncertainty_baseline import json_ready


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    baseline = args.baseline.resolve()
    # Keep all derived data beside the baseline, never in source log directories.
    if not args.output.is_relative_to(baseline.parents[1]):
        parser.error('output must be beneath the baseline offline directory')
    args.output.mkdir(parents=True, exist_ok=True)
    args.baseline = baseline
    return args


def trials(baseline):
    data = json.loads(Path(baseline).read_text())
    return [trial for result in data['results']
            if result['dataset'] == 'published' and result['pivot_mode'] == 'relative'
            for trial in result['trials']]


def streams(trial):
    with np.load(trial['trial']) as data:
        return {key: data[key] for key in data.files}


def fit_copy(trial, data, path):
    """Fit a derived NPZ using the original relative pivot, without estimator edits."""
    np.savez(path, **data)
    return est._run_estimation(
        trial['obj'], str(path.parent), str(path), None, verbose=False,
        trial_label=trial['trial'], pivot_mode='relative',
        pivot_override=np.asarray(trial['pivot_xyz']))


def gaps(fit):
    phases = fit['phase_estimates']
    return np.array([phases['arc_B'][key] - phases['unarc_B'][key]
                     for key in ('m', 'zc')])


def save(path, data):
    path.write_text(json.dumps(json_ready(data), indent=2, allow_nan=False) + '\n')


def stats(values):
    values = np.asarray(values)
    return {'mean': values.mean(axis=0), 'sd': values.std(axis=0, ddof=1)}


def provenance(args, script):
    return {'script': str(Path(script).resolve()), 'baseline': str(args.baseline),
            'estimator': str(Path(est.__file__).resolve()),
            'pivot_mode': 'original per-trial relative pivot (held constant)'}
