"""O4: independently fit ARC and UNARC pivots using the workspace estimator."""

import matplotlib.pyplot as plt
import numpy as np

from irb120_control.estimation.uncertainty_common import (
    arguments, est, provenance, save, stats, streams, trials,
)


def sweep_pivot(positions, states, phase, z):
    # pivot_from_trajectory only selects STATE_ARC, including for UNARC input.
    selected = positions[states == phase]
    return est.pivot_from_trajectory(
        selected, np.full(len(selected), est.STATE_ARC), z_fixed=z)


def main():
    args = arguments(__doc__)
    rows = []
    for trial in trials(args.baseline):
        data = streams(trial)
        positions = np.column_stack([data[k] for k in ('x', 'y', 'z')])
        states = data['controller_state_id']
        commanded = est.commanded_pivot_from_log(trial['trial'])
        row = {'object': trial['obj'], 'source': trial['trial'],
               'commanded_pivot_m': commanded}
        for name, phase in [('arc', est.STATE_ARC), ('unarc', est.STATE_UNARC)]:
            pivot, radius, residual = sweep_pivot(positions, states, phase, commanded[2])
            if pivot is None:
                raise ValueError(f'Insufficient {name} samples: {trial["trial"]}')
            row[name] = {'pivot_m': pivot, 'radius_m': radius, 'rms_mm': residual * 1000}
        row['unarc_minus_arc_x_mm'] = (row['unarc']['pivot_m'][0] - row['arc']['pivot_m'][0]) * 1000
        rows.append(row)
    summary = {obj: stats([r['unarc_minus_arc_x_mm'] for r in rows if r['object'] == obj])
               for obj in sorted({r['object'] for r in rows})}
    save(args.output / 'results.json', {**provenance(args, __file__), 'rows': rows, 'summary': summary})
    fig, ax = plt.subplots(figsize=(8, 4))
    for obj in summary:
        values = [r['unarc_minus_arc_x_mm'] for r in rows if r['object'] == obj]
        ax.plot(range(1, len(values) + 1), values, '.-', label=obj)
    ax.axhline(0, color='gray', lw=1)
    ax.set(xlabel='Trial index', ylabel='UNARC − ARC fitted pivot x (mm)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'pivot_shift.png', dpi=160)
    plt.close(fig)
    print(summary)


if __name__ == '__main__':
    main()
