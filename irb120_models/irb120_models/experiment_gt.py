"""Adapters for experiment scoring; authoritative values live beside object meshes."""
import numpy as np
import yaml
from .objects import load_object, objects_dir


def forward_tip_ground_truth(*, include_temporary=False):
    records = yaml.safe_load((objects_dir() / 'tipT' / 'configurations.yaml').read_text())['configurations']
    result = {}
    for name, record in records.items():
        if record.get('temporary', False) and not include_temporary:
            continue
        gt = dict(record)
        x, z = gt['com_obj']
        gt['com'] = np.array([gt['base_w'] - x, 0.0, z])
        gt['theta_deg'] = float(np.degrees(np.arctan2(gt['com'][0], z)))
        result[name] = gt
    return result


def measured_ground_truth(names):
    result = {}
    for name in names:
        gt = load_object(name).ground_truth()
        result[name] = dict(mass=gt['mass'], com=np.array([gt['com_x'], 0., gt['com_z']]),
                            theta_deg=gt['theta_star_deg'], theta_cad_deg=gt['theta_star_cad_deg'])
    return result
