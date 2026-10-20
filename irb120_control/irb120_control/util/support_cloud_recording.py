"""Record an analysis-only local scene cloud without detector ROI/table removal.

Never used for contact selection. Contains table, object and possibly clutter;
it is deliberately NOT named object_cloud. All points are in base_link.
"""
import json
from pathlib import Path
import numpy as np


def reconstruct_view(depth_mm, info, xy_bounds, z_bounds):
    depth = np.asarray(depth_mm, dtype=float) / 1000.
    if depth.ndim != 2:
        raise ValueError('Expected aligned depth image')
    K = np.asarray(info['K'], float)
    tf = info['base_to_optical']
    R, t = np.asarray(tf['R'], float), np.asarray(tf['t'], float)
    if K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,):
        raise ValueError('Invalid camera calibration shape')
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError('Nonfinite camera transform')
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0))
    uv = np.c_[u, v]
    normalized = np.c_[(u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1]]
    distortion = np.asarray(info.get('D', []), float)
    if distortion.size and np.any(distortion):
        import cv2
        normalized = cv2.undistortPoints(uv.astype(float).reshape(-1, 1, 2), K, distortion).reshape(-1, 2)
    optical = np.c_[normalized, np.ones(len(u))] * depth[v, u, None]
    base = (optical-t) @ R
    low, high = np.asarray(xy_bounds[0]), np.asarray(xy_bounds[1])
    keep = ((base[:, :2] >= low).all(1) & (base[:, :2] <= high).all(1)
            & (base[:, 2] >= z_bounds[0]) & (base[:, 2] <= z_bounds[1]))
    return base[keep], uv[keep].astype(np.int32)


def record_support_scene(folder, *, xy_padding=.04, table_z=-.021, out_dir=None):
    """Add independent artifacts using existing snapshot depth, TF and object XY ROI.

    Object XY extent only scopes the scene. No object z-min crop, table subtraction,
    segmentation, voxelization, or detector-input modification is performed.
    Missing camera views are recorded explicitly; an empty object ROI fails closed.
    out_dir: where to write the two files (default: the snapshot folder itself). Used to
    rebuild clouds for read-only archived snapshots without touching them.
    """
    folder = Path(folder)
    out = Path(out_dir) if out_dir is not None else folder
    out.mkdir(parents=True, exist_ok=True)
    with np.load(folder/'object_cloud.npz') as data:
        obj = np.array(data['xyz'], dtype=float, copy=True)
    obj = obj[np.isfinite(obj).all(1)]
    if len(obj) < 3:
        raise ValueError('Support-scene recording needs a nonempty object XY ROI')
    low, high = obj[:, :2].min(0)-xy_padding, obj[:, :2].max(0)+xy_padding
    z_bounds = [table_z-.02, max(float(obj[:, 2].max())+.02, table_z+.02)]
    points, pixels, ids, views = [], [], [], []
    for number, cam in enumerate(('cam1', 'cam2', 'cam3'), start=1):
        info_path, depth_path = folder/f'{cam}_camera_info.json', folder/f'{cam}_aligned_depth.npz'
        view = dict(camera=cam, camera_id=number, depth_file=depth_path.name, calibration_file=info_path.name)
        try:
            info = json.loads(info_path.read_text())
            with np.load(depth_path) as data:
                xyz, uv = reconstruct_view(data['depth_mm'], info, [low, high], z_bounds)
                stamp = float(data['stamp']) if 'stamp' in data else None
            points.append(xyz);pixels.append(uv);ids.append(np.full(len(xyz), number, np.uint8))
            view.update(status='recorded', points=len(xyz), depth_stamp=stamp,
                        base_to_optical=info['base_to_optical'])
        except (OSError, ValueError, KeyError) as exc:
            view.update(status='unavailable', reason=str(exc))
        views.append(view)
    if not points or not sum(map(len, points)):
        raise ValueError('No usable depth view for support-scene recording')
    xyz = np.concatenate(points)
    metadata = dict(schema_version=1, frame='base_link', purpose='analysis_only_not_contact_selection',
        semantics='local scene: object, table and possible clutter; not segmented object geometry',
        source_object_cloud='object_cloud.npz', roi_xy_m=[low.tolist(), high.tolist()],
        roi_z_m=z_bounds, table_z_reference_m=table_z, table_removed=False, voxelized=False,
        points=len(xyz), views=views, synchronization='per-view depth stamps retained; not claimed simultaneous',
        below_legacy_roi_points=int(np.sum(xyz[:, 2] < -.01)))
    np.savez_compressed(out/'support_scene_cloud.npz', xyz=xyz.astype(np.float32),
                        camera_id=np.concatenate(ids), pixel_uv=np.concatenate(pixels))
    (out/'support_scene_cloud.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return dict(available=True, file='support_scene_cloud.npz', metadata_file='support_scene_cloud.json',
                points=len(xyz), below_legacy_roi_points=metadata['below_legacy_roi_points'],
                views_recorded=sum(v['status']=='recorded' for v in views))
