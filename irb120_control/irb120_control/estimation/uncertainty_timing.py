"""O1: sweep F/T timestamp offsets, preserving geometry and relative pivots."""

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from irb120_control.estimation.uncertainty_common import (
    arguments, fit_copy, gaps, provenance, save, streams, trials,
)


def shifted_streams(data, shift):
    """Positive shift means a wrench is associated with a later pose."""
    result = dict(data)
    result['ft_time_s'] = data['ft_time_s'] + shift
    # Existing data has no finite sensor poses. Refuse to shift those by accident.
    if 'ft_px' in data and np.isfinite(data['ft_px']).any():
        raise ValueError('Finite sensor-pose stream requires separate time handling')
    return result


def main():
    args = arguments(__doc__)
    all_trials = trials(args.baseline)
    results = {}
    with tempfile.TemporaryDirectory(prefix='derived_', dir=args.output) as scratch:
        scratch = Path(scratch)
        for obj in sorted({r['obj'] for r in all_trials}):
            selected = [r for r in all_trials if r['obj'] == obj]
            datasets = [streams(r) for r in selected]
            by_shift = {}

            def evaluate(shift):
                shift = round(float(shift), 6)
                if shift in by_shift:
                    return
                rows = []
                for trial, data in zip(selected, datasets):
                    changed = shifted_streams(data, shift)
                    # No endpoint clamping is allowed on either sweep.
                    mask = np.isin(data['controller_state_id'], [3, 4])
                    sweep_times = data['pose_time_s'][mask]
                    if sweep_times.min() < changed['ft_time_s'][0] or sweep_times.max() > changed['ft_time_s'][-1]:
                        raise ValueError(f'Insufficient F/T coverage: {trial["trial"]}, {shift}')
                    fit = fit_copy(trial, changed, scratch / 'shifted.npz')
                    if shift == 0:
                        np.testing.assert_allclose(gaps(fit), gaps(trial), atol=1e-9, rtol=1e-7)
                    rows.append({'source': trial['trial'], 'gap': gaps(fit),
                                 'phase_estimates': fit['phase_estimates']})
                by_shift[shift] = rows

            coarse = np.round(np.arange(-1, 1.001, .05), 6)
            for shift in coarse:
                evaluate(shift)
            gt = selected[0]
            scale = np.array([gt['mass_gt'], gt['com_z_gt']])

            def mean_gap(shift):
                return np.mean([row['gap'] for row in by_shift[shift]], axis=0)

            # Refine near individual mass/height minima and a joint normalized score.
            coarse_gaps = np.array([mean_gap(float(s)) for s in coarse])
            scores = np.linalg.norm(coarse_gaps / scale, axis=1)
            centers = [coarse[np.argmin(scores)]]
            centers += [coarse[np.argmin(abs(coarse_gaps[:, k]))] for k in range(2)]
            for center in centers:
                for shift in np.arange(max(-1, center - .05), min(1, center + .05) + .001, .005):
                    evaluate(shift)
            shifts = sorted(by_shift)
            means = np.array([mean_gap(shift) for shift in shifts])
            scores = np.linalg.norm(means / scale, axis=1)
            best = int(np.argmin(scores))
            plausible = [i for i, shift in enumerate(shifts) if abs(shift) <= .100001]
            local = min(plausible, key=lambda i: scores[i])
            results[obj] = {
                'zero_gap': mean_gap(0), 'joint_best_shift_s': shifts[best],
                'joint_best_gap': means[best], 'plausible_best_shift_s': shifts[local],
                'plausible_best_gap': means[local],
                'mass_best_shift_s': shifts[int(np.argmin(abs(means[:, 0])))],
                'height_best_shift_s': shifts[int(np.argmin(abs(means[:, 1])))],
                'grid': [{'shift_s': shift, 'mean_gap': mean_gap(shift),
                          'trials': by_shift[shift]} for shift in shifts]}
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            for k, label in enumerate(('ARC − UNARC mass (kg)', 'ARC − UNARC CoM z (m)')):
                for j in range(len(selected)):
                    axes[k].plot(shifts, [by_shift[s][j]['gap'][k] for s in shifts], alpha=.2)
                axes[k].plot(shifts, means[:, k], 'k-', label='Mean of 10 trials')
                axes[k].axhline(0, color='gray', lw=1)
                axes[k].axvspan(-.1, .1, color='green', alpha=.12)
                axes[k].set_ylabel(label)
            axes[0].legend()
            axes[-1].set_xlabel('Added F/T timestamp offset (s)')
            fig.suptitle(obj)
            fig.tight_layout()
            fig.savefig(args.output / f'{obj}_timing.png', dpi=160)
            plt.close(fig)
            save(args.output / 'results.json', {**provenance(args, __file__),
                 'positive_shift': 'ft_time_s += shift; wrench associated with later pose',
                 'derived_npz': 'Temporary copies under output, removed after fitting; original streams unchanged.',
                 'objective': 'norm(mean ARC−UNARC mass/GT mass, mean height/GT height)',
                 'results': results})
            print(f'{obj}: joint optimum {shifts[best]:+.3f} s; gap {means[best]}', flush=True)


if __name__ == '__main__':
    main()
