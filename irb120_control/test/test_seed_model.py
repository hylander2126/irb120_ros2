"""Seeded snapshots: moving a saved pivot report with the object; no hardware."""
import numpy as np

from irb120_control.util.seed_model import moved_pivot_report


def test_rejected_review_with_empty_samples_moves_without_error():
    fit = dict(R=np.eye(3).tolist(), t=[.05, .033, 0.], yaw_deg=-3.6)
    report = dict(candidate_status='rejected', source='cam3',
                  geometry=dict(support_samples_base_m=[], object_table_boundary_px=[],
                                candidate_pivot_base_m=[.6, 0., 0.]))
    moved = moved_pivot_report(report, fit, '00_perceive')
    assert moved['geometry']['support_samples_base_m'] == []
    np.testing.assert_allclose(moved['geometry']['candidate_pivot_base_m'], [.65, .033, 0.])
    assert 'object_table_boundary_px' not in moved['geometry']
