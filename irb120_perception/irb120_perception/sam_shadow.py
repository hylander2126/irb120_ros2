"""ROS-free SAM-first snapshot analysis. No production publishers or control calls.

Single stationary target in the workspace, arm clear (or explicit robot masks).
Joint mode selects one prompt-consistent RGB mask; legacy automatic/guided modes
union accepted proposals after 3-D clipping.
This deliberately does not claim general multi-object instance association.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class View:
    camera: str
    rgb: np.ndarray
    depth: np.ndarray  # metres
    K: np.ndarray
    R: np.ndarray  # base -> optical
    t: np.ndarray
    robot_mask: np.ndarray | None = None

    def geometry(self):
        v, u = np.indices(self.depth.shape)
        z = self.depth
        optical = np.stack(((u-self.K[0, 2])*z/self.K[0, 0],
                            (v-self.K[1, 2])*z/self.K[1, 1], z), -1)
        return (optical-self.t) @ self.R


def workspace_support(view, bounds, table_z=-.021, clearance=.004):
    """Clip pixels, never reject an entire mask for crossing the workspace."""
    xyz = view.geometry()
    valid = np.isfinite(view.depth) & (view.depth > 0)
    valid &= np.all((xyz >= bounds[0]) & (xyz <= bounds[1]), axis=-1)
    # Fit near the known table height, independent of the active object cloud.
    candidates = xyz[valid & (abs(xyz[..., 2]-table_z) < .015)]
    coef = np.array([0., 0., table_z])
    status = 'fallback'
    if len(candidates) >= 100:
        rng = np.random.default_rng(17)
        candidates = candidates[rng.choice(len(candidates), min(6000, len(candidates)), replace=False)]
        A = np.c_[candidates[:, :2], np.ones(len(candidates))]
        best = np.zeros(len(A), bool)
        for _ in range(120):
            ids = rng.choice(len(A), 3, replace=False)
            c = np.linalg.lstsq(A[ids], candidates[ids, 2], rcond=None)[0]
            if np.linalg.norm(c[:2]) > .15:
                continue
            inliers = abs(A@c-candidates[:, 2]) < .0025
            if inliers.sum() > best.sum():
                best = inliers
        if best.sum() >= 100:
            c = np.linalg.lstsq(A[best], candidates[best, 2], rcond=None)[0]
            center = np.mean(bounds, axis=0)
            if np.linalg.norm(c[:2]) <= .15 and abs(np.r_[center[:2], 1]@c-table_z) <= .015:
                coef, status = c, 'fit'
    height = (xyz[..., 2]-xyz[..., :2]@coef[:2]-coef[2])/np.sqrt(1+sum(coef[:2]**2))
    valid &= height > clearance
    if view.robot_mask is not None:
        valid &= ~view.robot_mask.astype(bool)
    return xyz, valid, dict(status=status, coefficients=coef.tolist())


def guided_points(support, count=8):
    """Bounded, deterministic image-space farthest-point prompts from depth ROI."""
    vu = np.argwhere(support)
    if not len(vu):
        return np.empty((0, 2))
    vu = vu[::max(1, len(vu)//4000)]
    chosen = [int(np.argmin(np.sum((vu-np.median(vu, axis=0))**2, axis=1)))]
    distance = np.full(len(vu), np.inf)
    for _ in range(min(count, len(vu))-1):
        distance = np.minimum(distance, np.sum((vu-vu[chosen[-1]])**2, axis=1))
        chosen.append(int(np.argmax(distance)))
    return vu[chosen, ::-1].astype(float)


def accept_proposal(mask, support, workspace_pixels):
    """Reject table-dominated proposals using only their in-workspace pixels."""
    clipped = mask & support
    denominator = int((mask & workspace_pixels).sum())
    return clipped if clipped.sum() >= 30 and clipped.sum() >= .2*denominator else None


def prompt_box(support, padding=12):
    """Use substantial image components only to prompt, never to clip output."""
    from scipy.ndimage import label
    labels, count = label(support)
    if not count:
        return None
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    selected = sizes >= max(30, .1*sizes.max())
    v, u = np.nonzero(selected[labels])
    if not len(u):
        return None
    h, w = support.shape
    return np.array([max(0, u.min()-padding), max(0, v.min()-padding),
                     min(w-1, u.max()+padding), min(h-1, v.max()+padding)])


def joint_prompts(seed, support, workspace_pixels, padding=12):
    """Refine the prompt using a seed RGB outline, never clipping final geometry.

    Select the seed image component with most above-table support to avoid
    detached table markings. Zero depth is unknown, never a negative prompt.
    """
    from scipy.ndimage import label, binary_erosion, binary_dilation
    labels, count = label(seed)
    if not count or not np.any(seed & support):
        return None, np.empty((0, 2)), np.empty(0, int)
    evidence = np.bincount(labels[support], minlength=count+1); evidence[0] = 0
    outline = labels == int(np.argmax(evidence))
    box = prompt_box(outline, padding)
    positive = binary_erosion(outline, iterations=3) & support
    if positive.sum() < 30:
        positive = outline & support
    pos = guided_points(positive, 8)
    # Known table/background, well outside the RGB silhouette, near the box.
    negative = workspace_pixels & ~support & ~binary_dilation(outline, iterations=15)
    if box is not None:
        y, x = np.indices(seed.shape)
        negative &= (x >= box[0]-30) & (x <= box[2]+30) & (y >= box[1]-30) & (y <= box[3]+30)
    neg = guided_points(negative, 4)
    return box, np.concatenate((pos, neg)), np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]


def select_joint_candidate(masks, scores, points, labels):
    """Choose one prompt-consistent RGB mask, not a union of object parts."""
    u, v = np.rint(points).astype(int).T
    positive = labels == 1
    ranking = []
    for mask, score in zip(masks, scores):
        hits = mask[v, u]
        fg = float(hits[positive].mean()) if positive.any() else 1.
        bg = float(hits[~positive].mean()) if (~positive).any() else 0.
        ranking.append(float(score) + .5*fg - .5*bg if fg >= .75 and bg <= .25 else -np.inf)
    return int(np.argmax(ranking)) if np.isfinite(ranking).any() else None


def segment(view, support, model, mode='joint', prompts=8, workspace_pixels=None,
            diagnostics=None):
    from mobile_sam import SamPredictor, SamAutomaticMaskGenerator
    started = time.perf_counter()
    if mode == 'automatic':
        generator = SamAutomaticMaskGenerator(model, points_per_side=8, points_per_batch=64,
                    pred_iou_thresh=.86, stability_score_thresh=.92, min_mask_region_area=400)
        proposals = generator.generate(view.rgb)
    elif mode in ('guided', 'box', 'joint'):
        predictor = SamPredictor(model)
        proposals = []
        points = guided_points(support, prompts) if mode == 'guided' else np.empty((0, 2))
        box = prompt_box(support) if mode in ('box', 'joint') else None
        if len(points) or box is not None:
            predictor.set_image(view.rgb)  # one embedding per camera
            if box is not None:
                masks, scores, _ = predictor.predict(box=box, multimask_output=False)
                if mode == 'joint':
                    initial_box = box.copy()
                    refined_box, points, point_labels = joint_prompts(masks[0], support,
                        workspace_pixels if workspace_pixels is not None else support)
                    fallback = True
                    if refined_box is not None and len(points):
                        candidates, confidence, _ = predictor.predict(box=refined_box,
                            point_coords=points, point_labels=point_labels, multimask_output=True)
                        best = select_joint_candidate(candidates, confidence, points, point_labels)
                        if best is not None:
                            masks, scores = candidates[best:best+1], confidence[best:best+1]
                            box = refined_box
                            fallback = False
                    if diagnostics is not None:
                        diagnostics['initial_box'] = initial_box.tolist()
                        diagnostics['point_labels'] = point_labels.tolist()
                        diagnostics['seed_fallback'] = fallback
                proposals.append(dict(segmentation=masks[0], predicted_iou=float(scores[0])))
            for point in points if mode == 'guided' else []:
                masks, scores, _ = predictor.predict(point_coords=point[None],
                                    point_labels=np.ones(1), multimask_output=True)
                # Retain the most confident proposal per prompt; parts may overlap.
                best = int(np.argmax(scores))
                if scores[best] >= .86:
                    proposals.append(dict(segmentation=masks[best], predicted_iou=float(scores[best])))
        if diagnostics is not None:
            diagnostics['prompt_points'] = points.tolist()
            diagnostics['prompt_box'] = box.tolist() if box is not None else None
    else:
        raise ValueError(f'Unknown segmentation mode: {mode}')
    keep = np.zeros(support.shape, bool)
    raw = np.zeros(support.shape, bool)
    accepted = 0
    for proposal in proposals:
        clipped = accept_proposal(proposal['segmentation'], support,
                                  workspace_pixels if workspace_pixels is not None else support)
        if clipped is not None:
            keep |= clipped
            raw |= proposal['segmentation']
            accepted += 1
    if diagnostics is not None:
        diagnostics['raw_selected_mask'] = raw
        diagnostics['support_mask'] = support.copy()
    return keep, dict(seconds=time.perf_counter()-started, proposals=len(proposals),
                      accepted=accepted, pixels=int(keep.sum()), raw_selected_pixels=int(raw.sum()))


def surface_normals(xyz, depth):
    """Oriented toward camera; unreliable depth discontinuities get zero normals."""
    dx = np.zeros_like(xyz); dy = np.zeros_like(xyz)
    dx[:, 1:-1] = xyz[:, 2:]-xyz[:, :-2]
    dy[1:-1] = xyz[2:]-xyz[:-2]
    n = np.cross(dx, dy)
    length = np.linalg.norm(n, axis=-1)
    good = (length > 1e-10) & (np.linalg.norm(dx, axis=-1) < .015) & (np.linalg.norm(dy, axis=-1) < .015)
    good &= depth > 0
    return np.divide(n, length[..., None], out=np.zeros_like(n), where=good[..., None])


def priority_merge(views, masks, tolerance=.004):
    """Keep exact preferred samples; suppress cam2 only on matching visible faces.

    Missing masks, out-of-view and occluded samples never constitute rejection.
    Oppositely oriented thin faces remain separate. Visible free-space conflicts
    are quarantined rather than averaging them into preferred observations.
    """
    geometry = {v.camera: v.geometry() for v in views}
    normals = {}
    for v in views:
        n = surface_normals(geometry[v.camera], v.depth)
        toward = -v.t@v.R - geometry[v.camera]
        n[np.sum(n*toward, axis=-1) < 0] *= -1
        normals[v.camera] = n
    xyz, source = [], []
    points = geometry['cam2'][masks['cam2']]
    ns = normals['cam2'][masks['cam2']]
    suppressed = np.zeros(len(points), bool)
    conflicts = np.zeros(len(points), bool)
    for v in views:
        if v.camera == 'cam2':
            continue
        preferred = geometry[v.camera][masks[v.camera]]
        xyz.append(preferred); source.append(np.full(len(preferred), int(v.camera[-1]), np.uint8))
        optical = points@v.R.T+v.t
        z = optical[:, 2]
        uv = optical[:, :2]/np.maximum(z[:, None], 1e-9)
        uv = np.rint(uv*np.array([v.K[0, 0], v.K[1, 1]])+v.K[:2, 2]).astype(int)
        h, w = v.depth.shape
        inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ids = np.flatnonzero(inside)
        u, y = uv[ids].T
        measured = v.depth[y, u]
        good = masks[v.camera][y, u] & (measured > 0)
        agreement = np.sum(ns[ids]*normals[v.camera][y, u], axis=-1) > .8
        distance = np.linalg.norm(points[ids]-geometry[v.camera][y, u], axis=-1)
        suppressed[ids] |= good & agreement & (abs(z[ids]-measured) <= tolerance) & (distance <= tolerance)
        # In front of a trusted face with agreeing orientation is disagreement.
        conflicts[ids] |= good & agreement & (z[ids] < measured-tolerance)
    keep = ~(suppressed | conflicts)
    xyz.append(points[keep]); source.append(np.full(keep.sum(), 2, np.uint8))
    result_masks = {key: value.copy() for key, value in masks.items()}
    result_masks['cam2'][masks['cam2']] = keep
    return np.concatenate(xyz), np.concatenate(source), result_masks, dict(
        corroborating_cam2=int(suppressed.sum()), conflicting_cam2=int(conflicts.sum()),
        gap_cam2=int(keep.sum()), preferred_displacement_m=0.,
        conflict_points=points[conflicts])


<<<<<<< HEAD
def _tsdf_volume(voxel):
    import open3d as o3d
    return o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel,
        sdf_trunc=4*voxel, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


def _integrate(volume, v, depth):
    import open3d as o3d
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(v.rgb),
        o3d.geometry.Image(depth.astype(np.float32)), depth_scale=1., depth_trunc=3.,
        convert_rgb_to_intensity=False)
    h, w = depth.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, v.K[0, 0], v.K[1, 1], v.K[0, 2], v.K[1, 2])
    E = np.eye(4); E[:3, :3] = v.R; E[:3, 3] = v.t
    volume.integrate(rgbd, intrinsic, E)


def _extract(volume):
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def protected_meshes(views, masks, output, voxel=.003, gap_guard_voxels=6):
    """One shared TSDF volume for the actual output mesh -- cam1+cam3 integrate first, then cam2,
    with any cam2 pixel masked to zero (treated as no depth there, exactly as if it were outside
    its SAM mask) if it sits within gap_guard_voxels*voxel of preferred geometry AND the surface
    normals there agree (same face, same 0.8-cosine threshold priority_merge uses for point-level
    corroboration). That keeps the original safety guarantee -- cam2 can never add or revise a
    sample near an already-trusted cam1/cam3 surface, so it cannot shift it -- while letting
    cam2's genuinely new information (a differently oriented nearby face, or anywhere cam1/cam3
    never observed at all) integrate into the *same* voxel grid, so marching cubes extracts one
    continuous surface there instead of two independently-triangulated meshes that only
    coincidentally meet. The previous design built cam1+cam3 and cam2 as fully separate volumes
    and concatenated the extracted meshes; even where cam2 had real, unpruned data, concatenation
    never welds the two patches, so a genuine strip could still go missing at the seam no matter
    how the gap guard was tuned (confirmed on real captures, see STEREO_OFFLINE.md). 0 disables
    cam2 depth masking entirely (cam2 integrates everywhere, same as 0 did before).

    Still builds and exports preferred.ply (cam1+cam3 only) and gaps.ply (cam2 only) from their
    own fully isolated volumes, unaffected by anything above -- unchanged from before, for the
    SAM-shadow-ablation review tooling and as the untouched-preferred regression test's reference.
    combined.ply is the new single jointly-integrated mesh, not preferred+gaps concatenated.
    """
    preferred_volume = _tsdf_volume(voxel)
    for v in views:
        if v.camera in ('cam1', 'cam3'):
            _integrate(preferred_volume, v, np.where(masks[v.camera], v.depth, 0))
    preferred = _extract(preferred_volume)

    gaps_volume = _tsdf_volume(voxel)
    cam2 = next(v for v in views if v.camera == 'cam2')
    cam2_depth = np.where(masks['cam2'], cam2.depth, 0).astype(np.float32)
    _integrate(gaps_volume, cam2, cam2_depth)
    gaps = _extract(gaps_volume)

    joint_volume = _tsdf_volume(voxel)
    for v in views:
        if v.camera in ('cam1', 'cam3'):
            _integrate(joint_volume, v, np.where(masks[v.camera], v.depth, 0))
    masked_cam2_depth = cam2_depth
    if gap_guard_voxels > 0 and len(preferred.vertices) and cam2_depth.any():
        geometry = cam2.geometry()
        normals = surface_normals(geometry, cam2.depth)
        toward = -cam2.t@cam2.R - geometry
        normals[np.sum(normals*toward, axis=-1) < 0] *= -1
        pv = np.asarray(preferred.vertices); pn = np.asarray(preferred.vertex_normals)
        valid = cam2_depth > 0
        distance, nearest = cKDTree(pv).query(geometry[valid])
        unreliable = np.linalg.norm(normals[valid], axis=-1) < 1e-9
        # Safety-conservative: an unreliable (depth-discontinuity) cam2 normal is never grounds
        # to *exempt* a close pixel from masking -- that exemption requires positive evidence of
        # a different face. Falls back to the plain distance guard there, same as it would if no
        # normal check existed at all.
        same_face = (np.einsum('ij,ij->i', normals[valid], pn[nearest]) > .8) | unreliable
        redundant = (distance < gap_guard_voxels*voxel) & same_face
        redundant_pixels = np.zeros(cam2_depth.shape, bool)
        redundant_pixels[valid] = redundant
        masked_cam2_depth = np.where(redundant_pixels, 0, cam2_depth)
    _integrate(joint_volume, cam2, masked_cam2_depth)
    combined = _extract(joint_volume)

    for name, mesh in [('preferred', preferred), ('gaps', gaps), ('combined', combined)]:
        if len(mesh.triangles):
            import open3d as o3d
            o3d.io.write_triangle_mesh(str(output/f'{name}.ply'), mesh)
    return dict(preferred_triangles=len(preferred.triangles), gap_triangles=len(gaps.triangles),
                combined_triangles=len(combined.triangles),
                cam2_integration_into_preferred_volume=False, gap_guard_m=gap_guard_voxels*voxel)
=======
def protected_meshes(views, masks, output, voxel=.003):
    """Separate TSDF volumes prevent *all* cam2 updates to preferred voxels.

    Export preferred and gap meshes plus a conservative concatenation. Cam2
    triangles within the truncation guard of preferred mesh vertices are omitted;
    no welding, blending or invented closure. Seams/holes are expected.
    """
    import open3d as o3d
    meshes = []
    for group in (('cam1', 'cam3'), ('cam2',)):
        volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel,
            sdf_trunc=4*voxel, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        for v in views:
            if v.camera not in group:
                continue
            d = np.where(masks[v.camera], v.depth, 0).astype(np.float32)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(v.rgb),
                o3d.geometry.Image(d), depth_scale=1., depth_trunc=3., convert_rgb_to_intensity=False)
            h, w = d.shape
            intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, v.K[0, 0], v.K[1, 1], v.K[0, 2], v.K[1, 2])
            E = np.eye(4); E[:3, :3] = v.R; E[:3, 3] = v.t
            volume.integrate(rgbd, intrinsic, E)
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    preferred, gaps = meshes
    if len(preferred.vertices) and len(gaps.triangles):
        distance = cKDTree(np.asarray(preferred.vertices)).query(np.asarray(gaps.vertices))[0]
        gaps.remove_triangles_by_mask(np.any(distance[np.asarray(gaps.triangles)] < 6*voxel, axis=1))
        gaps.remove_unreferenced_vertices()
    for name, mesh in [('preferred', preferred), ('gaps', gaps), ('combined', preferred+gaps)]:
        if len(mesh.triangles):
            o3d.io.write_triangle_mesh(str(output/f'{name}.ply'), mesh)
    return dict(preferred_triangles=len(preferred.triangles), gap_triangles=len(gaps.triangles),
                cam2_integration_into_preferred_volume=False, gap_guard_m=6*voxel)
>>>>>>> 2607c08e5a6cea5dbb6f19d1da1d380fe31ce5c9
