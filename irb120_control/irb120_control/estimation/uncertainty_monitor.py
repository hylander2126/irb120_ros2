"""O6: inspect monitor trajectory shape and available commanded-pivot evidence."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from irb120_control.estimation.uncertainty_common import (
    arguments, est, provenance, save, stats, streams, trials,
)
from irb120_control.estimation.uncertainty_pivots import sweep_pivot


def main():
    args = arguments(__doc__)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for trial in trials(args.baseline):
        if trial['obj'] != 'monitor':
            continue
        data = streams(trial)
        metadata = json.loads(Path(trial['trial']).with_suffix('.json').read_text())
        logged = data.get('commanded_pivot', metadata.get('arc_center', metadata.get('ARC_CENTER')))
        commanded = est.commanded_pivot_from_log(trial['trial'])
        p = np.column_stack([data[k] for k in ('x', 'y', 'z')])
        states = data['controller_state_id']
        mask = np.isin(states, [3, 4])
        radius = np.linalg.norm(p - commanded, axis=1)
        row = {'source': trial['trial'], 'logged_commanded_pivot_m': logged,
               'estimator_commanded_pivot_m': commanded,
               'commanded_source': 'logged' if logged is not None else 'estimator fallback; not logged',
               'radius_ptp_mm': np.ptp(radius[mask]) * 1000}
        for name, phase, color in [('arc', 3, 'tab:blue'), ('unarc', 4, 'tab:orange')]:
            selected = states == phase
            pivot, fitted_radius, residual = sweep_pivot(p, states, phase, commanded[2])
            row[name] = {'pivot_x_minus_commanded_mm': (pivot[0] - commanded[0]) * 1000,
                         'rms_mm': residual * 1000, 'radius_m': fitted_radius}
            t = data['pose_time_s'][selected] - data['pose_time_s'][mask][0]
            axes[0].plot(p[selected, 0] * 1000, p[selected, 2] * 1000, color=color, alpha=.35)
            axes[1].plot(t, (radius[selected] - radius[mask][0]) * 1000, color=color, alpha=.35)
            fitted_r = np.hypot(p[selected, 0] - pivot[0], p[selected, 2] - pivot[2])
            axes[2].plot(t, (fitted_r - fitted_radius) * 1000, color=color, alpha=.35)
        rows.append(row)
    summary = {'radius_ptp_mm': stats([r['radius_ptp_mm'] for r in rows])}
    for phase in ('arc', 'unarc'):
        summary[phase] = {key: stats([r[phase][key] for r in rows])
                          for key in ('pivot_x_minus_commanded_mm', 'rms_mm')}
    axes[0].set(xlabel='Ball x (mm)', ylabel='Ball z (mm)')
    axes[1].set(xlabel='Time from ARC start (s)', ylabel='Radius change about nominal pivot (mm)')
    axes[2].set(xlabel='Time from ARC start (s)', ylabel='Residual about fitted sweep circle (mm)')
    fig.suptitle('Monitor: ARC blue, UNARC orange; all 10 trials')
    fig.tight_layout()
    fig.savefig(args.output / 'monitor_trajectory.png', dpi=180)
    plt.close(fig)
    save(args.output / 'results.json', {**provenance(args, __file__), 'rows': rows, 'summary': summary,
         'conclusion': 'Direction-dependent effective pivots contradict a single fixed-circle explanation. '
                       'Ball trajectory alone cannot distinguish slip, rolling contact or a rounded base. '
                       'Missing logged commanded pivots prevent independent verification of ARC_CENTER.'})
    print(summary)


if __name__ == '__main__':
    main()
