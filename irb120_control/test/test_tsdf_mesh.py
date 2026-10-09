"""Geometry checks for single-object TSDF cleanup."""
import numpy as np
import pytest

# A module-level pytest.importorskip here skipped the WHOLE control test
# session (every other file reported nothing), so skip per test instead.
try:
    import open3d as o3d
    from irb120_control.tsdf_mesh import clean_mesh
except ImportError:
    o3d = None
pytestmark = pytest.mark.skipif(o3d is None, reason='open3d not installed')


def test_rejects_flying_fragment_and_closes_missing_surface():
    surface = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=20)
    surface.paint_uniform_color([0.2, 0.4, 0.6])
    # Simulate an unobserved cap and a small disconnected depth artifact.
    surface.remove_vertices_by_mask(np.asarray(surface.vertices)[:, 2] > 0.035)
    flying = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
    flying.translate([0.2, 0, 0])
    flying.paint_uniform_color([1, 0, 0])
    raw = surface + flying
    original_count = len(raw.vertices)
    cleaned, stats = clean_mesh(raw, 0.005)
    assert cleaned.is_watertight()
    assert cleaned.get_max_bound()[0] < 0.06
    assert cleaned.has_vertex_colors()
    assert np.allclose(np.asarray(cleaned.vertex_colors), [0.2, 0.4, 0.6])
    assert stats['vertices_after_component_filter'] < original_count
    assert len(raw.vertices) == original_count
    observed, _ = clean_mesh(raw, 0.005, completion='none', smooth_iterations=0)
    assert not observed.is_watertight()


def test_empty_mesh_fails_explicitly():
    with pytest.raises(ValueError, match='Too little surface'):
        clean_mesh(o3d.geometry.TriangleMesh(), 0.005)
