"""Clean an observed TSDF surface and optionally form a closed convex envelope."""

import numpy as np
import open3d as o3d


def clean_mesh(mesh, voxel_m, completion='convex-hull', smooth_iterations=10):
    """Reject detached/noisy samples before smoothing and convex completion.

    Assumes one scanned object. Hull geometry is inferred, not observed; colors
    on its vertices come from the retained surface samples.
    """
    if voxel_m <= 0 or smooth_iterations < 0:
        raise ValueError('voxel_m must be positive and smooth_iterations nonnegative')
    if completion not in ('convex-hull', 'none'):
        raise ValueError('Unknown completion mode')
    mesh = o3d.geometry.TriangleMesh(mesh)
    stats = {'input_vertices': len(mesh.vertices), 'input_triangles': len(mesh.triangles),
             'completion': completion, 'smooth_iterations': smooth_iterations}
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) < 4 or not len(mesh.triangles):
        raise ValueError('Too little surface geometry to reconstruct')

    def largest_component():
        labels, _, areas = mesh.cluster_connected_triangles()
        if not len(areas):
            raise ValueError('No connected surface remains after outlier rejection')
        mesh.remove_triangles_by_mask(np.asarray(labels) != np.argmax(areas))
        mesh.remove_unreferenced_vertices()

    largest_component()
    stats['vertices_after_component_filter'] = len(mesh.vertices)
    cloud = o3d.geometry.PointCloud()
    cloud.points = mesh.vertices
    _, indices = cloud.remove_statistical_outlier(
        nb_neighbors=min(20, len(cloud.points) - 1), std_ratio=1.5)
    keep = np.zeros(len(mesh.vertices), dtype=bool)
    keep[indices] = True
    mesh.remove_vertices_by_mask(~keep)
    largest_component()
    stats['vertices_after_outlier_filter'] = len(mesh.vertices)
    if len(mesh.vertices) < 4:
        raise ValueError('Too few inliers remain for reconstruction')
    if smooth_iterations:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
    if completion == 'convex-hull':
        observed = mesh
        mesh, indices = observed.compute_convex_hull()
        if observed.has_vertex_colors():
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.asarray(observed.vertex_colors)[np.asarray(indices)])
        if not mesh.is_watertight():
            raise ValueError('Convex completion failed to produce a watertight mesh')
    mesh.compute_vertex_normals()
    stats.update(output_vertices=len(mesh.vertices), output_triangles=len(mesh.triangles),
                 watertight=mesh.is_watertight())
    return mesh, stats
