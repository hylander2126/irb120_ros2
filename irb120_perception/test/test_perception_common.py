"""Check FrameAccumulator's voxel-occupancy-consensus fusion without ROS."""

import numpy as np
import pytest

from irb120_perception.perception_common import (
    FrameAccumulator, fit_dominant_horizontal_plane, remove_plane,
)


def test_horizontal_table_plane_is_removed_without_removing_object_points():
    grid = np.stack(np.meshgrid(np.linspace(0.15, 0.8, 30),
                                np.linspace(-0.25, 0.25, 30)), axis=-1).reshape(-1, 2)
    table = np.column_stack((grid, -0.02 + 0.01 * grid[:, 0]))
    object_pts = np.array([[0.45, 0.0, 0.08], [0.46, 0.01, 0.10]])
    points = np.concatenate((table, object_pts))
    plane = fit_dominant_horizontal_plane(points, distance=0.003)
    assert plane is not None
    kept = remove_plane(points, plane, 0.003)
    np.testing.assert_allclose(kept, object_pts, atol=1e-6)


def test_warms_up_before_returning_a_result():
    acc = FrameAccumulator(n_frames=3, voxel_size=0.01, min_hits=2)
    assert acc.add(np.array([[0.0, 0.0, 0.0]])) is None
    assert acc.add(np.array([[0.0, 0.0, 0.0]])) is None
    # Only the third add() fills the window and returns a fused cloud
    fused = acc.add(np.array([[0.0, 0.0, 0.0]]))
    assert fused is not None
    assert len(fused) == 1


def test_persistent_point_survives_transient_noise_is_dropped():
    # A real surface point recurs at the same spot on every frame; a noise
    # point wanders to a different voxel each frame and never repeats.
    real = np.array([0.50, 0.00, 0.10])
    acc = FrameAccumulator(n_frames=5, voxel_size=0.005, min_hits=3)
    rng = np.random.default_rng(0)
    fused = None
    for _ in range(5):
        noise = real + rng.uniform(0.5, 1.0, size=3)  # far away, different each frame
        frame = np.stack([real, noise])
        fused = acc.add(frame)
    assert fused is not None
    np.testing.assert_allclose(fused, [real], atol=1e-6)


def test_min_hits_rejects_a_voxel_seen_in_too_few_frames():
    acc = FrameAccumulator(n_frames=4, voxel_size=0.01, min_hits=3)
    pt = np.array([[0.2, 0.2, 0.05]])
    empty = np.zeros((0, 3))
    fused = None
    # pt appears in only 2 of 4 frames -> below min_hits=3
    for frame in (pt, empty, pt, empty):
        fused = acc.add(frame)
    assert fused is not None
    assert len(fused) == 0


def test_averages_jitter_within_a_voxel_across_frames():
    voxel_size = 0.01
    acc = FrameAccumulator(n_frames=2, voxel_size=voxel_size, min_hits=2)
    # Two frames, same voxel, slightly different positions (sensor jitter)
    acc.add(np.array([[0.100, 0.000, 0.000]]))
    fused = acc.add(np.array([[0.102, 0.000, 0.000]]))
    assert fused is not None
    np.testing.assert_allclose(fused, [[0.101, 0.0, 0.0]], atol=1e-6)


def test_reset_clears_the_window():
    acc = FrameAccumulator(n_frames=2, voxel_size=0.01, min_hits=1)
    acc.add(np.array([[0.0, 0.0, 0.0]]))
    acc.reset()
    # After reset, a single add() should not complete the window again
    assert acc.add(np.array([[0.0, 0.0, 0.0]])) is None


def test_empty_frames_only_returns_empty_cloud():
    acc = FrameAccumulator(n_frames=2, voxel_size=0.01, min_hits=1)
    acc.add(np.zeros((0, 3)))
    fused = acc.add(np.zeros((0, 3)))
    assert fused is not None
    assert fused.shape == (0, 3)
