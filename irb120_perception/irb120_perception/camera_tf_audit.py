"""Check captured camera poses against the configured handeye calibration.

The optional cam2 base-Z correction is recorded separately from calibration so
reconstruction can apply a requested total correction exactly once.
"""
import ast
import hashlib
import re
from pathlib import Path

import numpy as np


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Invalid camera TF quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def calibration_references(handeye_share):
    """Read the calibration files selected by bringup_cams, without launching ROS."""
    launch = Path(handeye_share) / 'launch'
    tree = ast.parse((launch / 'bringup_cams.launch.py').read_text())
    cameras = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'CAMERAS' for t in node.targets))
    references = {}
    for camera, (_, _, filename) in cameras.items():
        path = launch / filename
        content = path.read_text()
        args = dict(re.findall(r'"(--[a-z-]+)",\s*"([^"]+)"', content))
        references[camera] = dict(
            path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            parent=args['--frame-id'], child=args['--child-frame-id'],
            translation=[float(args['--'+k]) for k in ('x', 'y', 'z')],
            quaternion=[float(args['--'+k]) for k in ('qx', 'qy', 'qz', 'qw')])
    return references


def audit_camera_pose(camera, rotation, translation, reference, observed_statics=()):
    """Reject conflicting/stale TFs; allow and record cam2's intentional Z shift.

    rotation/translation map camera_link into base_link. observed_statics are
    all latched transforms received for that child, not just TF Buffer's winner.
    """
    rotation, translation = np.asarray(rotation), np.asarray(translation)
    nominal_r = quaternion_matrix(reference['quaternion'])
    nominal_t = np.asarray(reference['translation'])
    for sample in observed_statics:
        if (sample['parent'] != reference['parent']
                or not np.allclose(quaternion_matrix(sample['quaternion']), rotation, atol=1e-5, rtol=0)
                or not np.allclose(sample['translation'], translation, atol=1e-5, rtol=0)):
            raise ValueError(f'{camera}: conflicting static TF publishers for {reference["child"]}; '
                             'stop stale camera TF publishers and restart camera bringup')
    delta = translation - nominal_t
    translation_error = delta[:2] if camera == 'cam2' else delta
    if (np.max(np.abs(rotation - nominal_r)) > 1e-4
            or np.max(np.abs(translation_error)) > .0001):
        raise ValueError(f'{camera}: live TF does not match configured calibration '
                         f'{reference["path"]}; stop stale camera TF publishers and restart camera bringup')
    return dict(reference=reference, camera_link_to_base=dict(R=rotation.tolist(), t=translation.tolist()),
                base_z_offset_m=float(delta[2]) if camera == 'cam2' else 0.,
                status='matches_calibration', static_samples=len(observed_statics))


def resolve_cam2_z_offset(camera_calibration, extra_offset_m=None, target_offset_m=.004):
    """Return additional reconstruction shift and its provenance.

    Explicit extra offsets preserve legacy replay semantics. Automatic mode
    requires an audited capture; it never guesses a historical TF's correction.
    """
    audit = camera_calibration.get('tf_audit', {})
    captured = audit.get('base_z_offset_m') if audit.get('status') == 'matches_calibration' else None
    if extra_offset_m is None:
        if captured is None:
            raise ValueError('Capture lacks audited cam2 TF offset. For a legacy replay specify '
                             '--cam2-z-offset-mm explicitly (0 for no extra shift, 6 for a known unshifted capture).')
        extra_offset_m = target_offset_m - captured
        mode = 'target_total'
    else:
        mode = 'explicit_extra'
    if not np.isfinite(extra_offset_m):
        raise ValueError('cam2 Z offset must be finite')
    return float(extra_offset_m), dict(mode=mode, captured_tf_offset_m=captured,
                                      additional_offset_m=float(extra_offset_m),
                                      effective_total_offset_m=None if captured is None else captured+extra_offset_m)
