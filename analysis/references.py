"""Scoring references for an episode's object label. Used ONLY after estimation.

Tier 1 (scored): tipT configurations, cylinder configurations, and objects with a measured
ground truth in irb120_models (e.g. heart). Everything else is tier 2 (estimate only).
"""
import numpy as np

import _paths  # noqa: F401
from irb120_models import list_objects, load_object
from irb120_models.configurations import list_configurations, load_configuration
from irb120_models.experiment_gt import forward_tip_ground_truth


def lookup(label: str) -> dict | None:
    """Reference in the estimator's convention, or None (tier 2).

    Returns dict(source, mass, com (forward: from the far edge), com_obj (press-pull: from
    the near edge) when known, height). Unknown fields are absent, never guessed.
    """
    if label in list_configurations('tipT'):
        gt = forward_tip_ground_truth()[label]
        return dict(source='tipT configuration', mass=gt['mass'], com=np.asarray(gt['com'], float),
                    com_obj=np.array([gt['com_obj'][0], 0., gt['com_obj'][1]]), height=float(gt['com'][2]))
    if label in list_configurations('cylinder'):
        record = load_configuration(label)
        d = record['initial_support_distances']['press_pull']
        z = float(record['com_obj'][-1])
        ref = dict(source='cylinder configuration', mass=record['mass'], height=z, radius=record.get('fillet_radius'))
        if d is not None:
            ref['com_obj'] = np.array([d, 0., z]); ref['com'] = ref['com_obj'].copy()
        return ref
    base = label if label in list_objects() else label.split('_')[0]
    if base in list_objects():
        try:
            gt = load_object(base).ground_truth()
        except ValueError:          # broken object record (e.g. a missing mesh): no reference
            return None
        if gt.get('mass') is None:
            return None
        ref = dict(source=f'measured ground truth ({base})', mass=gt['mass'], height=gt['com_z'])
        if gt.get('com_x') is not None:
            # Measured com_x is from the forward-tip edge; the press-pull edge distance is
            # not recorded, so press-pull stages get no reference d (not guessed).
            ref['com'] = np.array([gt['com_x'], 0., gt['com_z']])
        return ref
    return None


def estimator_reference(ref: dict | None, d: float | None = None) -> dict:
    """Reference dict for episode_estimation.analyze_episode.

    d given: GT-free run with that horizontal distance for both primitives; mass/height NaN
    so no score is produced. d None: the scoring reference (missing edges -> KeyError ->
    that stage is reported as 'no reference d')."""
    if d is not None:
        return dict(mass=np.nan, com=np.array([d, 0., np.nan]), com_obj=np.array([d, 0., np.nan]))
    out = dict(mass=ref['mass'])
    for key in ('com', 'com_obj'):
        if key in ref:
            out[key] = np.asarray(ref[key], float)
    if 'com' not in out and 'com_obj' in out:
        out['com'] = np.array([np.nan, 0., ref['height']])
    return out
