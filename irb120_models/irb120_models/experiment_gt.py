"""Adapters for experiment scoring; authoritative values live beside object meshes."""
import numpy as np
from .objects import load_object
from .configurations import list_configurations, load_configuration, com_from_near_edge, planar_com


def forward_tip_ground_truth(*, include_temporary=False):
    result = {}
    for name in list_configurations('tipT', include_temporary=include_temporary):
        record = load_configuration(name)
        gt = dict(record)
        x, z = com_from_near_edge(record)
        # The legacy estimator adapter expects both near/far edge coordinates.
        gt['com_obj'] = [x, z]
        gt['com'] = np.array([gt['base_w'] - x, 0.0, z])
        gt['theta_deg'] = float(np.degrees(np.arctan2(gt['com'][0], z)))
        result[name] = gt
    return result


def episode_reference(name):
    """Scoring reference for any configured object (tipT: forward_tip_ground_truth).

    com[0] is the CoM distance from the forward (far) support, com_obj[0] from the
    press-pull (near) support, z the CoM height. Unknown values are NaN, never guessed:
    an estimate needing one is skipped, a score needing one is NaN. Without a measured
    forward distance it is base_w - press_pull (recorded in forward_distance_source).
    """
    if name in list_configurations('tipT', include_temporary=True):
        return forward_tip_ground_truth(include_temporary=True)[name]
    record = load_configuration(name)
    gt = dict(record)
    d = record['initial_support_distances']
    near, far, source = d['press_pull'], d['forward'], 'measured'
    if far is None and near is not None and record.get('base_w') is not None:
        far, source = record['base_w'] - near, 'base_w_minus_press_pull'
    nan = lambda v: np.nan if v is None else float(v)  # noqa: E731
    z = nan(planar_com(record)[1])
    gt.update(mass=nan(record.get('mass')), com_obj=[nan(near), z], com=np.array([nan(far), 0.0, z]),
              forward_distance_source=source if far is not None else None,
              theta_deg=float(np.degrees(np.arctan2(nan(far), z))))
    return gt


def cylinder_sweep_references():
    """Analysis adapter: read canonical SI YAML, omit unknown measurements.

    d uses the declared initial-support geometry (outer_radius - bottom fillet),
    with explicit measured distances taking precedence. This is the upright
    support reference; subsequent support migration remains a model limitation.
    """
    result = {}
    for name in list_configurations('cylinder'):
        record = load_configuration(name)
        ref = {}
        for key, value, factor in (
                ('radius_mm', record.get('fillet_radius'), 1000),
                ('mass_kg', record.get('mass'), 1),
                ('height_mm', planar_com(record)[1], 1000)):
            if value is not None:
                ref[key] = value*factor
        d = record['initial_support_distances']['press_pull']
        if d is not None:
            ref['d_mm'] = d*1000
        result[name] = ref
    return result


def measured_ground_truth(names):
    result = {}
    for name in names:
        gt = load_object(name).ground_truth()
        result[name] = dict(mass=gt['mass'], com=np.array([gt['com_x'], 0., gt['com_z']]),
                            theta_deg=gt['theta_star_deg'], theta_cad_deg=gt['theta_star_cad_deg'])
    return result
