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
