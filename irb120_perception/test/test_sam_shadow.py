"""Geometry and isolation regressions; no ROS graph or model checkpoint needed."""
import numpy as np
from irb120_perception.sam_shadow import View, workspace_support, priority_merge, guided_points


def view(camera, depth=1., translation=0.):
    return View(camera, np.zeros((9, 9, 3), np.uint8), np.full((9, 9), depth),
                np.array([[1000., 0, 4], [0, 1000., 4], [0, 0, 1]]),
                np.eye(3), np.array([0., 0., translation]))


def test_workspace_clips_pixels_and_removes_robot():
    v = view('cam1')
    v.robot_mask = np.zeros((9, 9), bool); v.robot_mask[4, 4] = True
    _, support, _ = workspace_support(v, np.array([[0, -.1, .5], [.1, .1, 1.5]]), table_z=0.)
    assert support[:, :4].sum() == 0
    assert support[:, 4:].sum() == 44
    assert not support[4, 4]


def test_exact_preferred_samples_and_cam2_corroboration():
    views = [view('cam1'), view('cam2'), view('cam3')]
    masks = {v.camera: np.ones((9, 9), bool) for v in views}
    points, ids, retained, report = priority_merge(views, masks)
    assert report['corroborating_cam2'] == 49  # edges have unreliable normals
    for v in (views[0], views[2]):
        np.testing.assert_array_equal(points[ids == int(v.camera[-1])], v.geometry().reshape(-1, 3))
    assert masks['cam2'].all()  # input not mutated
    for cam in ('cam1', 'cam2', 'cam3'):
        assert retained[cam].sum() == (ids == int(cam[-1])).sum()


def test_occluded_cam2_is_not_rejected():
    views = [view('cam1'), view('cam2', depth=1.02), view('cam3')]
    masks = {v.camera: np.ones((9, 9), bool) for v in views}
    _, _, retained, report = priority_merge(views, masks)
    assert retained['cam2'].all()
    assert report['conflicting_cam2'] == 0


def test_missing_mask_is_no_evidence():
    views = [view('cam1'), view('cam2'), view('cam3')]
    masks = {v.camera: np.full((9, 9), v.camera == 'cam2') for v in views}
    _, _, retained, _ = priority_merge(views, masks)
    assert retained['cam2'].all()


def test_visible_conflict_is_quarantined():
    views = [view('cam1'), view('cam2', depth=.98), view('cam3')]
    masks = {v.camera: np.ones((9, 9), bool) for v in views}
    _, _, _, report = priority_merge(views, masks)
    assert report['conflicting_cam2'] == 49


def test_opposite_thin_face_is_retained():
    a, b, c = view('cam1'), view('cam2'), view('cam3')
    # Rotate cam2 around y: observes opposite face at z=1.002, 2 mm apart.
    b.R = np.diag([-1., 1., -1.]); b.t = np.array([0., 0., 2.002])
    views = [a, b, c]; masks = {v.camera: np.ones((9, 9), bool) for v in views}
    _, _, retained, report = priority_merge(views, masks)
    assert retained['cam2'].all()
    assert report['corroborating_cam2'] == 0


def test_priority_merge_drops_isolated_cam1_cam3_noise():
    views = [view('cam1'), view('cam2'), view('cam3')]
    masks = {v.camera: np.ones((9, 9), bool) for v in views}
    outlier_xyz = views[0].geometry()[0, 0].copy()
    views[0].depth[0, 0] = 5.  # one pixel far from the rest of cam1's dense flat plane
    points, ids, retained, report = priority_merge(views, masks)
    assert report['noise_filtered'] == {'cam1': 1, 'cam3': 0, 'cam2': 0}
    assert not retained['cam1'][0, 0]
    assert retained['cam1'].sum() == (ids == 1).sum()
    cam1_points = points[ids == 1]
    assert len(cam1_points) == 9*9 - 1
    assert not np.any(np.all(np.isclose(cam1_points, outlier_xyz), axis=1))


def test_priority_merge_noise_filter_disabled_by_default_zero():
    views = [view('cam1'), view('cam2'), view('cam3')]
    masks = {v.camera: np.ones((9, 9), bool) for v in views}
    views[0].depth[0, 0] = 5.
    points, ids, _, report = priority_merge(views, masks, noise_min_neighbors=0)
    assert report['noise_filtered'] == {'cam1': 0, 'cam3': 0, 'cam2': 0}
    assert len(points[ids == 1]) == 9*9


def test_guided_prompts_are_bounded_and_inside_support():
    support = np.zeros((100, 100), bool); support[20:80, 40:60] = True
    points = guided_points(support)
    assert len(points) == 8
    assert all(support[int(y), int(x)] for x, y in points)
    assert guided_points(np.zeros_like(support)).shape == (0, 2)


def test_cam2_cannot_shift_preferred_tsdf(tmp_path):
    import pytest
    o3d = pytest.importorskip('open3d')
    from irb120_perception.sam_shadow import protected_meshes
    views = [view('cam1'), view('cam2'), view('cam3')]
    masks = {v.camera: np.ones((9, 9), bool) for v in views}
    first = tmp_path/'first'; first.mkdir()
    second = tmp_path/'second'; second.mkdir()
    protected_meshes(views, masks, first, voxel=.001)
    views[1].depth[:] = .8
    protected_meshes(views, masks, second, voxel=.001)
    a = o3d.io.read_triangle_mesh(str(first/'preferred.ply'))
    b = o3d.io.read_triangle_mesh(str(second/'preferred.ply'))
    assert len(a.vertices) > 0
    np.testing.assert_array_equal(np.asarray(a.vertices), np.asarray(b.vertices))
    np.testing.assert_array_equal(np.asarray(a.triangles), np.asarray(b.triangles))


def test_protected_meshes_drops_isolated_cam1_cam3_noise(tmp_path):
    import pytest
    pytest.importorskip('open3d')
    from irb120_perception.sam_shadow import protected_meshes
    views = [view('cam1'), view('cam2'), view('cam3')]
    masks = {v.camera: np.ones((9, 9), bool) for v in views}
    views[0].depth[0, 0] = 5.  # isolated cam1 pixel, far from the rest of the flat plane
    out = tmp_path/'out'; out.mkdir()
    mesh = protected_meshes(views, masks, out, voxel=.001)
    assert mesh['noise_filtered_pixels'] == {'cam1': 1, 'cam2': 0, 'cam3': 0}


def test_protected_meshes_noise_filter_disabled_by_zero(tmp_path):
    import pytest
    pytest.importorskip('open3d')
    from irb120_perception.sam_shadow import protected_meshes
    views = [view('cam1'), view('cam2'), view('cam3')]
    masks = {v.camera: np.ones((9, 9), bool) for v in views}
    views[0].depth[0, 0] = 5.
    out = tmp_path/'out'; out.mkdir()
    mesh = protected_meshes(views, masks, out, voxel=.001, noise_min_neighbors=0)
    assert mesh['noise_filtered_pixels'] == {'cam1': 0, 'cam2': 0, 'cam3': 0}


def test_combined_cannot_be_shifted_by_disagreeing_nearby_cam2(tmp_path):
    import pytest
    o3d = pytest.importorskip('open3d')
    from irb120_perception.sam_shadow import protected_meshes
    views = [view('cam1'), view('cam2'), view('cam3')]
    masks = {v.camera: np.ones((9, 9), bool) for v in views}
    first = tmp_path/'first'; first.mkdir()
    protected_meshes(views, masks, first, voxel=.001)
    views[1].depth[:] = .997  # 3mm closer: within sdf_trunc and the default 6mm gap guard
    second = tmp_path/'second'; second.mkdir()
    protected_meshes(views, masks, second, voxel=.001)
    a = o3d.io.read_triangle_mesh(str(first/'combined.ply'))
    b = o3d.io.read_triangle_mesh(str(second/'combined.ply'))
    assert len(a.vertices) > 0
    np.testing.assert_array_equal(np.asarray(a.vertices), np.asarray(b.vertices))
    np.testing.assert_array_equal(np.asarray(a.triangles), np.asarray(b.triangles))


def test_combined_mesh_is_one_continuous_patch_where_only_cam2_has_coverage(tmp_path):
    """Regression for the missing-strip bug: cam1/cam3 only cover half the object, cam2 covers
    all of it. The old design (two independently-extracted TSDF volumes, concatenated) could
    leave a seam between the two patches even with the gap guard fully disabled. The shared
    single-volume design should extract one continuous surface instead."""
    import pytest
    o3d = pytest.importorskip('open3d')
    from irb120_perception.sam_shadow import protected_meshes
    views = [view('cam1'), view('cam2'), view('cam3')]
    masks = {v.camera: np.zeros((9, 9), bool) for v in views}
    masks['cam1'][:, :4] = True  # cam1/cam3 only ever see the left half
    masks['cam2'][:, :] = True  # cam2 sees the whole plane
    out = tmp_path/'out'; out.mkdir()
    # gap_guard_voxels=0: at this synthetic scale (1mm voxels) the default 6mm guard would mask
    # even cam2's genuinely-new right-half columns, since they sit only ~1mm from the preferred
    # boundary -- a real effect of the guard's distance margin, not what this test is checking.
    mesh = protected_meshes(views, masks, out, voxel=.001, gap_guard_voxels=0)
    assert mesh['gap_triangles'] > 0  # cam2 has real candidate geometry on the right half
    preferred = o3d.io.read_triangle_mesh(str(out/'preferred.ply'))
    combined = o3d.io.read_triangle_mesh(str(out/'combined.ply'))
    assert len(combined.triangles) > len(preferred.triangles)  # cam2 extended real coverage
    _, counts, _ = combined.cluster_connected_triangles()
    assert len(counts) == 1  # one continuous surface, not a preferred patch glued to a separate cam2 patch


def test_table_dominated_proposal_rejected_without_rejecting_outside_roi():
    from irb120_perception.sam_shadow import accept_proposal
    mask = np.ones((100, 100), bool)
    support = np.zeros_like(mask); support[:10, :10] = True
    assert accept_proposal(mask, support, np.ones_like(mask)) is None
    # Most of this mask lies outside the workspace. Its in-workspace support
    # is object geometry, so the whole-mask/out-of-workspace area is irrelevant.
    np.testing.assert_array_equal(accept_proposal(mask, support, support), support)


def test_box_prompt_ignores_tiny_islands_but_not_output():
    from irb120_perception.sam_shadow import prompt_box
    support = np.zeros((100, 100), bool)
    support[20:70, 30:50] = True
    support[0, 99] = True
    np.testing.assert_array_equal(prompt_box(support), [18, 8, 61, 81])
    assert prompt_box(np.zeros_like(support)) is None


def test_raw_sam_diagnostics_preserve_holes_as_depth_filtering(monkeypatch):
    import sys
    from types import SimpleNamespace
    from irb120_perception.sam_shadow import segment
    raw = np.ones((9, 9), bool)
    support = raw.copy(); support[3:6, 3:6] = False
    class Predictor:
        def __init__(self, model):
            pass
        def set_image(self, rgb):
            pass
        def predict(self, **kwargs):
            return raw[None], np.array([.99]), None
    monkeypatch.setitem(sys.modules, 'mobile_sam', SimpleNamespace(
        SamPredictor=Predictor, SamAutomaticMaskGenerator=None))
    diagnostics = {}
    filtered, report = segment(view('cam1'), support, object(), mode='box',
                               workspace_pixels=raw, diagnostics=diagnostics)
    np.testing.assert_array_equal(filtered, support)
    np.testing.assert_array_equal(diagnostics['raw_selected_mask'], raw)
    assert report['raw_selected_pixels'] == 81
    assert report['pixels'] == 72


def test_joint_negative_prompts_never_use_missing_depth():
    from irb120_perception.sam_shadow import joint_prompts
    seed = np.zeros((100, 100), bool); seed[30:70, 40:60] = True
    support = seed.copy()
    valid_workspace = np.ones_like(seed); valid_workspace[:20] = False
    box, points, labels = joint_prompts(seed, support, valid_workspace)
    assert box is not None and (labels == 1).any() and (labels == 0).any()
    for (u, v), label in zip(points.astype(int), labels):
        assert valid_workspace[v, u]
        if label:
            assert support[v, u]
        else:
            assert not seed[v, u]


def test_joint_selection_rejects_conflicting_high_model_score():
    from irb120_perception.sam_shadow import select_joint_candidate
    points = np.array([[1., 1.], [3., 3.]])
    labels = np.array([1, 0])
    bad = np.ones((5, 5), bool)
    good = np.zeros_like(bad); good[1, 1] = True
    assert select_joint_candidate(np.array([bad, good]), [.99, .85], points, labels) == 1
    assert select_joint_candidate(np.array([bad]), [.99], points, labels) is None


def test_joint_empty_seed_has_no_fabricated_prompts():
    from irb120_perception.sam_shadow import joint_prompts
    empty = np.zeros((30, 30), bool)
    box, points, labels = joint_prompts(empty, empty, ~empty)
    assert box is None and len(points) == len(labels) == 0


def test_workspace_retry_removes_depth_fringe_without_accepting_seed():
    from irb120_perception.sam_shadow import retry_workspace_prompts, select_joint_candidate
    support = np.zeros((100, 100), bool); support[20:80, 30:70] = True
    seed = support.copy(); support[20:80, 27:30] = True  # stereo fringe outside RGB
    points, labels = retry_workspace_prompts(np.array([seed]), [.95], support, np.ones_like(seed))
    pos = points[labels == 1].astype(int)
    assert len(pos) and seed[pos[:, 1], pos[:, 0]].all()
    # Refined prompts still reject an inconsistent prediction, even at high score.
    assert select_joint_candidate(np.ones((1, 100, 100), bool), [.99], points, labels) is None


def test_workspace_retry_rejects_partial_object_seed():
    from irb120_perception.sam_shadow import retry_workspace_prompts
    support = np.zeros((100, 100), bool); support[10:90, 30:70] = True
    part = np.zeros_like(support); part[10:30, 30:70] = True
    points, labels = retry_workspace_prompts(np.array([part]), [.99], support, np.ones_like(support))
    assert len(points) == len(labels) == 0


def test_joint_reuses_embedding_and_makes_one_joint_refinement(monkeypatch):
    import sys
    from types import SimpleNamespace
    from irb120_perception.sam_shadow import segment
    seed = np.ones((9, 9), bool)
    calls = []
    class Predictor:
        def __init__(self, model): pass
        def set_image(self, rgb): calls.append('embedding')
        def predict(self, **kwargs):
            calls.append(kwargs)
            return seed[None], np.array([.99]), None
    monkeypatch.setitem(sys.modules, 'mobile_sam', SimpleNamespace(SamPredictor=Predictor, SamAutomaticMaskGenerator=None))
    diagnostics = {}
    mask, report = segment(view('cam1'), seed, object(), mode='joint', workspace_pixels=seed, diagnostics=diagnostics)
    assert calls[0] == 'embedding' and len(calls) == 3
    assert 'box' in calls[2] and len(calls[2]['point_labels']) > 1
    assert report['proposals'] == 1 and not diagnostics['seed_fallback']
    np.testing.assert_array_equal(mask, seed)
