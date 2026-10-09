"""Tests for analysis transformations that could silently mix sweep semantics."""

import numpy as np
import pytest

from irb120_control.estimation.uncertainty_matched_range import matched_states
from irb120_control.estimation.uncertainty_pivots import sweep_pivot
from irb120_control.estimation.uncertainty_timing import shifted_streams


def test_unarc_pivot_uses_unarc_positions_only():
    angle = np.linspace(.1, .4, 100)
    arc = np.column_stack((.61 + .2 * np.sin(angle), np.zeros(100), .2 * np.cos(angle)))
    unarc = arc + [.012, 0, 0]
    points = np.vstack((arc, unarc))
    states = np.repeat([3, 4], 100)
    p_arc, _, _ = sweep_pivot(points, states, 3, 0)
    p_unarc, _, _ = sweep_pivot(points, states, 4, 0)
    assert p_unarc[0] - p_arc[0] == pytest.approx(.012, abs=1e-9)


def test_matched_range_removes_ramps_but_preserves_reference_and_input():
    states = np.array([1, 2] + [3] * 20 + [4] * 20 + [5])
    original = states.copy()
    time = np.arange(len(states), dtype=float)
    theta = np.deg2rad([0, 0] + list(np.linspace(0, 20, 20)) + list(np.linspace(18, 0, 20)) + [0])
    changed, bounds = matched_states(time, states, theta, 2)
    np.testing.assert_array_equal(states, original)
    np.testing.assert_array_equal(changed[[0, 1, -1]], [1, 2, 5])
    for phase in (3, 4):
        keep = changed == phase
        assert time[keep].min() >= time[states == phase].min() + 2
        assert np.rad2deg(theta[keep]).min() >= bounds[0] - 1e-10
        assert np.rad2deg(theta[keep]).max() <= bounds[1] + 1e-10


def test_no_matched_range_is_rejected():
    states = np.repeat([3, 4], 20)
    theta = np.deg2rad(np.r_[np.linspace(2, 5, 20), np.linspace(10, 15, 20)])
    with pytest.raises(ValueError, match='No common'):
        matched_states(np.arange(40), states, theta, 2)


def test_timing_shift_does_not_shift_pose_or_modify_source():
    data = {'ft_time_s': np.arange(4.), 'pose_time_s': np.arange(4.),
            'fx': np.arange(4.), 'ft_px': np.full(4, np.nan)}
    shifted = shifted_streams(data, .05)
    np.testing.assert_allclose(shifted['ft_time_s'], np.arange(4.) + .05)
    np.testing.assert_array_equal(data['ft_time_s'], np.arange(4.))
    np.testing.assert_array_equal(shifted['pose_time_s'], data['pose_time_s'])
    np.testing.assert_array_equal(shifted['fx'], data['fx'])
    data['ft_px'][0] = 0
    with pytest.raises(ValueError, match='Finite sensor-pose'):
        shifted_streams(data, .05)
