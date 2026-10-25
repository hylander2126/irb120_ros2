"""One sampled, cleaned observed cloud shared by contact and motion consumers."""
import errno
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree

SCHEMA = 1
SAMPLE_VOXEL_M = .003
NEIGHBOR_RADIUS_M = .010


def prepare_object_cloud(xyz, camera_id):
    """Keep actual per-camera observations, then reject isolated sampled points once."""
    xyz = np.asarray(xyz, dtype=float)
    camera_id = np.asarray(camera_id)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz)
            or not np.isfinite(xyz).all() or camera_id.shape != (len(xyz),)
            or not np.isin(camera_id, [1, 2, 3]).all()):
        raise ValueError('Object cloud requires finite Nx3 points and camera IDs 1/2/3')
    keys = np.c_[np.floor(xyz / SAMPLE_VOXEL_M).astype(np.int64), camera_id]
    _, ids = np.unique(keys, axis=0, return_index=True)
    points = xyz[ids]
    distances, _ = cKDTree(points).query(points, k=2)
    keep = distances[:, 1] <= NEIGHBOR_RADIUS_M
    if not keep.any():
        raise ValueError('Object cloud has no supported surface; acquire a fresh snapshot')
    accepted = ids[keep]
    cloud = dict(xyz=xyz[accepted], camera_id=camera_id[accepted], source_index=accepted,
                 object_cloud_schema=np.array(SCHEMA))
    audit = dict(schema=SCHEMA, frame='base_link', input_points=len(xyz),
                 sampled_points=len(points), retained_points=len(accepted),
                 sample_voxel_m=SAMPLE_VOXEL_M, neighbor_radius_m=NEIGHBOR_RADIUS_M,
                 removed_points=int((~keep).sum()), removed_source_index=ids[~keep].tolist(),
                 removed_camera_id=camera_id[ids[~keep]].tolist(),
                 removed_xyz=points[~keep].tolist(),
                 bounds_m=[cloud['xyz'].min(0).tolist(), cloud['xyz'].max(0).tolist()])
    return cloud, audit


def load_object_cloud(path):
    """Validate the published artifact; consumers never re-filter its geometry."""
    with np.load(path) as data:
        if 'object_cloud_schema' not in data or int(data['object_cloud_schema']) != SCHEMA:
            raise ValueError('Object cloud predates shared-cloud validation; take a fresh snapshot')
        cloud = {key: data[key] for key in data.files}
    xyz, camera_id = cloud['xyz'], cloud['camera_id']
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz)
            or not np.isfinite(xyz).all() or camera_id.shape != (len(xyz),)
            or not np.isin(camera_id, [1, 2, 3]).all()
            or cloud['source_index'].shape != (len(xyz),)):
        raise ValueError('Invalid shared object cloud')
    return cloud


def publish_object_cloud(source, output):
    """Expose the same cloud at the handoff root, without a second data copy on disk."""
    source, output = Path(source), Path(output)
    cloud = load_object_cloud(source / 'object_cloud.npz')
    for name in ('object_cloud.npz', 'object_cloud.json'):
        try:
            os.link(source / name, output / name)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(source / name, output / name)
    return cloud
