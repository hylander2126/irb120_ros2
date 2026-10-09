"""O3: pre-contact offset and within-trial drift versus trial index and time."""

from datetime import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from irb120_control.estimation.uncertainty_common import (
    arguments, provenance, save, streams, trials,
)


def main():
    args = arguments(__doc__)
    # Reuse the independently reviewed window selection; no canonical fits run here.
    project = Path('/home/hylander/Documents/github/mujoco_rl')
    sys.path.insert(0, str(project))
    from uncertainty.hardware_check import windows

    rows = []
    for trial in trials(args.baseline):
        data = streams(trial)
        pose = np.zeros((len(data['pose_time_s']), 9))
        pose[:, 0], pose[:, 8] = data['pose_time_s'], data['controller_state_id']
        pre, post = windows(data['ft_time_s'], pose)
        if not pre.any() or not post.any():
            raise ValueError(f'Missing offset/drift window: {trial["trial"]}')
        wrench = np.column_stack([data[k] for k in ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')])
        offset = wrench[pre].mean(0)
        drift = wrench[post].mean(0) - offset
        date = datetime.strptime('_'.join(Path(trial['trial']).stem.split('_')[-2:]), '%Y%m%d_%H%M%S')
        rows.append({'object': trial['obj'], 'source': trial['trial'],
                     'wall_clock_local': date.isoformat(),
                     'pre_samples': int(pre.sum()), 'post_samples': int(post.sum()),
                     'offset': offset, 'drift': drift,
                     'offset_force_n': np.linalg.norm(offset[:3]),
                     'offset_torque_mnm': 1000 * np.linalg.norm(offset[3:]),
                     'drift_force_n': np.linalg.norm(drift[:3]),
                     'drift_torque_mnm': 1000 * np.linalg.norm(drift[3:])})
    summary = {}
    metrics = ('offset_force_n', 'drift_force_n', 'offset_torque_mnm', 'drift_torque_mnm')
    for obj in sorted({r['object'] for r in rows}):
        selected = [r for r in rows if r['object'] == obj]
        dates = [datetime.fromisoformat(r['wall_clock_local']) for r in selected]
        minutes = np.array([(date - dates[0]).total_seconds() / 60 for date in dates])
        index = np.arange(1, len(selected) + 1)
        summary[obj] = {}
        fig, axes = plt.subplots(4, 2, figsize=(10, 10))
        for i, metric in enumerate(metrics):
            values = np.array([r[metric] for r in selected])
            regression = linregress(minutes, values)
            summary[obj][metric] = {'mean': values.mean(), 'sd': values.std(ddof=1),
                                    'slope_per_min': regression.slope,
                                    'slope_stderr': regression.stderr, 'p_value': regression.pvalue}
            axes[i, 0].plot(index, values, '.-')
            axes[i, 1].plot(dates, values, '.-')
            axes[i, 0].set_ylabel(metric)
            axes[i, 1].tick_params(axis='x', rotation=30)
        axes[-1, 0].set_xlabel('Trial index')
        axes[-1, 1].set_xlabel('Filename wall-clock time (local)')
        fig.suptitle(obj)
        fig.tight_layout()
        fig.savefig(args.output / f'{obj}_drift.png', dpi=150)
        plt.close(fig)
    save(args.output / 'results.json', {**provenance(args, __file__),
         'window_helper': str(project / 'uncertainty/hardware_check.py'),
         'units': 'offset/drift vectors: fx fy fz in N; tx ty tz in N m',
         'rows': rows, 'summary': summary})
    print(summary)


if __name__ == '__main__':
    main()
