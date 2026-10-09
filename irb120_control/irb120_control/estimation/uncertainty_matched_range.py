"""O5: refit sweeps on their common tilt interval after the logged ramp."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from irb120_control.estimation.uncertainty_common import (
    arguments, est, fit_copy, gaps, provenance, save, stats, streams, trials,
)


def matched_states(time, states, theta, ramp_sec):
    """Drop ramp samples, then use the intersection of the two tilt ranges."""
    masks = []
    for phase in (est.STATE_ARC, est.STATE_UNARC):
        phase_mask = states == phase
        if not phase_mask.any():
            raise ValueError('Missing sweep')
        mask = phase_mask & (time >= time[phase_mask][0] + ramp_sec) & (theta > np.deg2rad(1))
        if mask.sum() < 3:
            raise ValueError('Insufficient post-ramp sweep samples')
        masks.append(mask)
    lower = max(theta[m].min() for m in masks)
    upper = min(theta[m].max() for m in masks)
    if lower >= upper:
        raise ValueError('No common post-ramp tilt range')
    keep = (masks[0] | masks[1]) & (theta >= lower) & (theta <= upper)
    result = states.copy()
    # Retain LULL and RETRACT, preserving the estimator's original tilt reference.
    result[np.isin(states, [est.STATE_ARC, est.STATE_UNARC]) & ~keep] = 0
    if any(np.count_nonzero(result == phase) < 3 for phase in (3, 4)):
        raise ValueError('Insufficient common-range samples')
    return result, np.rad2deg([lower, upper])


def main():
    args = arguments(__doc__)
    rows = []
    for trial in trials(args.baseline):
        data = streams(trial)
        metadata = json.loads(Path(trial['trial']).with_suffix('.json').read_text())
        ramp = float(metadata.get('ARC_TANGENTIAL_RAMP_SEC', 2.0))
        ramp_source = ('sidecar' if 'ARC_TANGENTIAL_RAMP_SEC' in metadata
                       else 'assumed 2 s: batch sidecar omits controller constants')
        p = np.column_stack([data[k] for k in ('x', 'y', 'z')])
        states = data['controller_state_id']
        arm = p - np.asarray(trial['pivot_xyz'])
        first = np.argmax(np.isin(states, [2, 3, 4, 5]))
        theta = -est.rotvec_between(arm[first], arm)[:, 1]
        changed = dict(data)
        changed['controller_state_id'], interval = matched_states(data['pose_time_s'], states, theta, ramp)
        path = args.output / f"{trial['obj']}_{Path(trial['trial']).name}"
        fit = fit_copy(trial, changed, path)
        save(path.with_suffix('.json'), {'source': trial['trial'], 'ramp_sec': ramp,
             'common_theta_deg': interval, 'pivot_override_m': trial['pivot_xyz'],
             'ramp_source': ramp_source,
             'transformation': 'Only excluded ARC/UNARC state IDs set to 0; other streams unchanged.'})
        rows.append({'object': trial['obj'], 'source': trial['trial'], 'derived': str(path),
                     'ramp_sec': ramp, 'ramp_source': ramp_source, 'theta_interval_deg': interval,
                     'samples_arc_unarc': [int(np.sum(changed['controller_state_id'] == s)) for s in (3, 4)],
                     'baseline_gap': gaps(trial), 'matched_gap': gaps(fit),
                     'phase_estimates': fit['phase_estimates']})
    summary = {obj: {key: stats([r[key] for r in rows if r['object'] == obj])
                    for key in ('baseline_gap', 'matched_gap')}
               for obj in sorted({r['object'] for r in rows})}
    save(args.output / 'results.json', {**provenance(args, __file__), 'rows': rows, 'summary': summary})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k, label in enumerate(('Mass gap (kg)', 'Height gap (m)')):
        for i, (obj, values) in enumerate(summary.items()):
            axes[k].plot([i-.15, i+.15], [values['baseline_gap']['mean'][k], values['matched_gap']['mean'][k]], 'o-')
        axes[k].set_xticks(range(len(summary)), summary.keys())
        axes[k].set_ylabel(label)
        axes[k].axhline(0, color='gray', lw=1)
    fig.suptitle('ARC − UNARC: original (left) → matched range after ramp (right)')
    fig.tight_layout()
    fig.savefig(args.output / 'matched_range.png', dpi=160)
    plt.close(fig)
    print(summary)


if __name__ == '__main__':
    main()
