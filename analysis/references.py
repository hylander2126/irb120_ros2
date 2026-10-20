"""Per-object inputs and scoring references, looked up by episode label.

Two different things, kept apart:
  d_env      horizontal CoM distance from the OUTER BASE EDGE on the pivot side, per mode
             (forward / press_pull). What a sharp-table-edge balance measures, so it is the
             estimator INPUT for the paper's "planar CoM is supplied" assumption; 1-sigma
             d_sigma when recorded.
  d          distance from the actual flat-table SUPPORT (fillet-aware for cylinders).
             Equal to d_env for sharp-edged objects. For rounded bases it needs the support
             geometry, so it is a DIAGNOSTIC ceiling ("oracle"), never a paper input.
  mass, height  SCORING references, used only after estimation. None when unknown or
             sealed (everyday objects until the estimates are frozen).

Tier 1 = mass and height both known (tipT, cylinder, heart). Tier 2 = everything else.
"""
import numpy as np

import _paths  # noqa: F401
from irb120_models import list_objects, load_object
from irb120_models.configurations import list_configurations, load_configuration
from irb120_models.experiment_gt import forward_tip_ground_truth

# Family records whose ground truth is a documented placeholder (cylinder.yaml notes); sweep
# variants have their own configurations. Never use these.
PLACEHOLDER_RECORDS = {'cylinder', 'tipT'}
MODES = ('forward', 'press_pull')


def _ref(source, mass, height, d, d_sigma=None, d_env=None, **extra):
    d = {m: d.get(m) for m in MODES}
    d_env = d if d_env is None else {m: d_env.get(m) for m in MODES}
    sigma = {m: (d_sigma or {}).get(m) for m in MODES}
    tier = 1 if mass is not None and height is not None else 2
    return dict(source=source, mass=mass, height=height, d=d, d_env=d_env, d_sigma=sigma, tier=tier, **extra)


def envelope_distances(record):
    """Outer-base-edge distances for a configuration record (see module docstring).
    Explicit support_distances (table-edge measurements) win; otherwise geometry."""
    w, x = record.get('base_w'), record['com_obj'][0]
    out = dict(forward=None, press_pull=None)
    if w is not None and x is not None:
        if record.get('frame') == 'bottom_center_x_along_push_z_up':
            out.update(press_pull=w / 2 + x, forward=w / 2 - x)
        else:
            out.update(press_pull=x, forward=w - x)
    elif x is not None:
        out['press_pull'] = x
    for mode, value in (record.get('support_distances') or {}).items():
        if value is not None:
            out[mode] = value
    return out


def lookup(label: str) -> dict | None:
    """Inputs/references for an episode label, or None when nothing is recorded."""
    if label in list_configurations('tipT', include_temporary=True):
        gt = forward_tip_ground_truth(include_temporary=True)[label]
        # forward: from the far edge (gt['com'][0]); press-pull: from the near edge.
        return _ref('tipT configuration', gt['mass'], float(gt['com'][2]),
                    dict(forward=float(gt['com'][0]), press_pull=float(gt['com_obj'][0])))
    if label in list_configurations(include_temporary=True):
        record = load_configuration(label)
        unc = (record.get('uncertainty') or {}).get('support_distances') or {}
        return _ref(f"{record['family']} configuration", record.get('mass'), record['com_obj'][-1],
                    record['initial_support_distances'], unc, d_env=envelope_distances(record),
                    radius=record.get('fillet_radius'), placement=record.get('placement'))
    base = label if label in list_objects() else label.split('_')[0]
    if base in list_objects() and base not in PLACEHOLDER_RECORDS:
        try:
            gt = load_object(base).ground_truth()
        except ValueError:          # broken object record (e.g. a missing mesh): nothing
            return None
        if gt.get('mass') is None:
            return None
        # The measured com_x is the press-pull pivot distance: the published press-pull
        # estimator (estimate_params) uses it as d. The forward edge is not recorded.
        return _ref(f'measured ground truth ({base})', gt['mass'], gt['com_z'],
                    dict(press_pull=gt.get('com_x')))
    return None


def estimator_reference(ref: dict | None, d: float | None = None, key: str = 'd_env') -> dict:
    """Reference dict for episode_estimation.analyze_episode.

    d given: that horizontal distance for both primitives (GT-free hull run). Otherwise the
    supplied per-mode distances; a missing mode makes that stage's fit fail and it is
    reported as unavailable, never filled in. Mass/height are NaN: scoring happens outside.
    """
    if d is not None:
        return dict(mass=np.nan, com=np.array([d, 0., np.nan]), com_obj=np.array([d, 0., np.nan]))
    nan, dd = np.nan, ref[key]
    return dict(mass=nan, com=np.array([dd['forward'] if dd['forward'] is not None else nan, 0., nan]),
                com_obj=np.array([dd['press_pull'] if dd['press_pull'] is not None else nan, 0., nan]))
