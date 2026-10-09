"""Tip choice (press-pull vs forward tip) and the press-pull slip check, without hardware."""

import numpy as np

from irb120_control.orchestrator import NOSLIP_TOL_MM, choose_tip, press_pull_slip_mm


def contacts(press=True, forward=True):
    return {'press': {'available': press, 'reason': 'no top'},
            'forward_tip': {'available': forward, 'reason': 'no side'}}


def test_first_tip_prefers_press_pull():
    assert choose_tip(contacts())[0] == 'press_pull_tip'
    assert choose_tip(contacts(press=False))[0] == 'forward_tip'
    # Nothing usable: still press-pull, so the gate reports its missing contact.
    assert choose_tip(contacts(press=False, forward=False))[0] == 'press_pull_tip'


def test_forward_tip_follows_failed_or_slipped_press_pull():
    assert choose_tip(contacts(), press_result={'status': 'ok', 'slip_mm': 1.0})[0] is None
    assert choose_tip(contacts(), press_result={'status': 'ok', 'slip_mm': None})[0] is None
    slipped = {'status': 'ok', 'slip_mm': NOSLIP_TOL_MM + 0.1}
    assert choose_tip(contacts(), press_result=slipped)[0] == 'forward_tip'
    assert choose_tip(contacts(), press_result={'status': 'failed', 'slip_mm': None})[0] == 'forward_tip'
    kind, reason = choose_tip(contacts(forward=False), press_result=slipped)
    assert kind is None and 'no forward-tip contact' in reason


def test_forced_mode_runs_one_tip():
    assert choose_tip(contacts(), 'forward')[0] == 'forward_tip'
    assert choose_tip(contacts(), 'press')[0] == 'press_pull_tip'
    failed = {'status': 'failed', 'slip_mm': None}
    assert choose_tip(contacts(), 'press', failed)[0] is None


def _write_log(path, radii, states):
    pivot = np.array([0.6, 0.0, 0.0])
    angle = np.linspace(0, -0.3, len(radii))
    np.savez(path, commanded_pivot=pivot, controller_state_id=np.asarray(states, float),
             x=pivot[0] - radii * np.sin(angle), y=np.zeros(len(radii)), z=radii * np.cos(angle))


def test_slip_is_radius_spread_over_last_attempt(tmp_path):
    assert press_pull_slip_mm(tmp_path) is None
    # Failed first attempt at a different radius, then a clean second attempt: only the
    # second attempt's ARC/UNARC counts.
    radii = np.r_[[0.25] * 3, [0.23] * 3, [0.2] * 2, [0.2, 0.201, 0.202, 0.201], [0.22]]
    states = [1, 1, 3, 3, 3, 5, 1, 2, 3, 3, 4, 4, 5]
    _write_log(tmp_path / 'arc_static_20261009_000000.npz', radii[:len(states)], states)
    assert abs(press_pull_slip_mm(tmp_path) - 2.0) < 1e-6
