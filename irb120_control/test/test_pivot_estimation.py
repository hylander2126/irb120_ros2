import json

import numpy as np

from irb120_control.estimation.estimate_params import (
    STATE_ARC,
    commanded_pivot_from_log,
    pivot_from_trajectory,
)


def test_commanded_pivot_prefers_npz_and_reads_legacy_sidecar(tmp_path):
    path = tmp_path / "arc_static_1.npz"
    np.savez(path, commanded_pivot=np.array([0.58, 0.01, -0.02]))
    (tmp_path / "arc_static_1.json").write_text(
        json.dumps({"arc_center": [0.61, 0.0, 0.0]}))
    np.testing.assert_allclose(commanded_pivot_from_log(str(path)), [0.58, 0.01, -0.02])

    legacy = tmp_path / "arc_static_2.npz"
    np.savez(legacy, x=np.array([1.0]))
    (tmp_path / "arc_static_2.json").write_text(
        json.dumps({"arc_center": [0.59, -0.01, 0.0]}))
    np.testing.assert_allclose(commanded_pivot_from_log(str(legacy)), [0.59, -0.01, 0.0])


def test_pivot_from_trajectory_recovers_arc_center_x():
    center = np.array([0.584, 0.0, -0.021])
    angle = np.linspace(0.25, 0.55, 200)
    radius = 0.18
    positions = np.column_stack((center[0] + radius * np.cos(angle),
                                 np.zeros_like(angle),
                                 center[2] + radius * np.sin(angle)))
    pivot, fitted_radius, residual = pivot_from_trajectory(
        positions, np.full(len(angle), STATE_ARC), z_fixed=center[2])
    np.testing.assert_allclose(pivot, center, atol=1e-9)
    assert abs(fitted_radius - radius) < 1e-9
    assert residual < 1e-9
