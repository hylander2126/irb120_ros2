"""Tip cascade: outcome of each attempt, and the attempt that follows it; no hardware."""

import numpy as np
import pytest

from irb120_control.util.tip_cascade import (
    FORWARD, MAX_ATTEMPTS, PRESS, PRESS_FORCE_MAX_N, attempt, classify, first_attempt,
    next_attempt, object_change,
)


def press(x, available=True):
    return dict(available=available, point=[x, 0., .1])


def contacts(shape='curved', alternatives=(.56, .54), forward=True, press_ok=True):
    return {'press': press(.58, press_ok),
            'press_alternatives': [dict(press(x), rank=k) for k, x in enumerate(alternatives, 1)],
            'forward_tip': {'available': forward, 'pivot_shape': {'shape': shape, 'reason': 't'}}}


def run(plan, outcome, why='w', point=None, force=None):
    return dict(attempt=plan, outcome=outcome, why=why, press_point=point, final_force=force)


def test_press_outcomes():
    assert classify(PRESS, {'completed': True, 'balance_reached': True})[0] == 'success'
    # The arc limit before the balance point still counts as a tip.
    outcome, why = classify(PRESS, {'completed': True, 'balance_reached': False})
    assert outcome == 'success' and 'balance point' in why
    assert classify(PRESS, {'completed': True}, slip_mm=6.)[0] == 'slip'
    assert classify(PRESS, {'completed': False, 'force_cap_reached': True, 'final_force_ref_n': 12.2})[0] == 'slip'
    assert classify(PRESS, {'completed': False, 'contact_missed': True})[0] == 'no_contact'
    assert classify(PRESS, {'completed': False, 'abort_reason': 'ft_stale'})[0] == 'sensor_fault'
    assert classify(PRESS, {'completed': False, 'status': 'failed'})[0] == 'failed'
    assert classify(PRESS, None)[0] == 'failed'


def test_forward_outcomes():
    assert classify(FORWARD, {'completed': True})[0] == 'success'
    assert classify(FORWARD, {'reversal_reason': 'contact_not_found'})[0] == 'no_contact'
    assert classify(FORWARD, {'failure_reason': 'force_limit'})[0] == 'force_limit'
    assert classify(FORWARD, {'failure_reason': 'height_lateral_or_orientation_drift'})[0] == 'drift'
    assert classify(FORWARD, {'reversal_reason': 'travel_limit'})[0] == 'no_tip'
    assert classify(FORWARD, {'failure_reason': 'stale_force'})[0] == 'sensor_fault'


def test_only_a_tilted_or_fallen_object_ends_the_cascade():
    shifted = dict(shift_mm=15., height_change_mm=0.)
    assert classify(PRESS, {'completed': True}, change=shifted)[0] == 'success'
    outcome, why = classify(PRESS, {'completed': False, 'status': 'failed'}, change=shifted)
    assert outcome == 'failed' and 'still upright' in why
    assert classify(FORWARD, {'reversal_reason': 'travel_limit'}, change=shifted)[0] == 'slid'
    fell = dict(shift_mm=2., height_change_mm=-60.)
    assert classify(FORWARD, {'completed': True}, change=fell)[0] == 'moved'
    assert classify(FORWARD, {'completed': True}, change=dict(shift_mm=3., height_change_mm=2.))[0] == 'success'


def test_object_change():
    cloud = np.random.default_rng(0).uniform([0, 0, 0], [.05, .05, .1], (200, 3))
    change = object_change(cloud, cloud + [.012, 0., 0.])
    assert change['shift_mm'] == pytest.approx(12.) and change['height_change_mm'] == pytest.approx(0.)
    assert object_change(cloud[:2], cloud) is None


def test_slipping_press_goes_riskier_then_forward():
    c = contacts()
    first, _ = first_attempt(c, force_ref=5.)
    assert first == attempt(PRESS, 0, 5.)
    history = [run(first, 'slip', point=[.58, 0., .1], force=PRESS_FORCE_MAX_N)]
    # The new snapshot's conservative contact is the one that slipped: riskier #1, back at
    # the default force (arc_static escalates again from there).
    plan, _ = next_attempt(history, c, force_ref=5.)
    assert plan == attempt(PRESS, 1, 5.)
    history.append(run(plan, 'slip', point=[.56, 0., .1], force=PRESS_FORCE_MAX_N))
    plan, _ = next_attempt(history, c, force_ref=5.)
    assert plan['press_candidate'] == 2
    history.append(run(plan, 'failed', point=[.54, 0., .1]))
    plan, reason = next_attempt(history, c, force_ref=5.)
    assert plan == attempt(FORWARD) and 'no untried press contact' in reason


def test_forward_runs_on_a_curved_pivot_but_only_once():
    history = [run(attempt(PRESS, 0, 5.), 'failed', point=[.58, 0., .1])]
    c = contacts(alternatives=())
    plan, _ = next_attempt(history, c)
    assert plan['kind'] == FORWARD
    history.append(run(plan, 'force_limit'))
    plan, reason = next_attempt(history, c)
    assert plan is None and 'no untried press contact' in reason


def test_failed_forward_tip_goes_to_press_pull():
    c = contacts(shape='straight')
    first, _ = first_attempt(c)
    assert first['kind'] == FORWARD
    for outcome in ('force_limit', 'drift', 'no_tip', 'failed'):
        plan, _ = next_attempt([run(first, outcome)], c, force_ref=6.)
        assert plan == attempt(PRESS, 0, 6.)


def test_missed_contact_retries_once_and_sensor_fault_retries():
    first = attempt(PRESS, 0, 5.)
    plan, _ = next_attempt([run(first, 'no_contact', point=[.58, 0., .1])], contacts())
    assert plan == dict(first, retry=True)
    plan, _ = next_attempt([run(first, 'no_contact', point=[.58, 0., .1]),
                            run(dict(first, retry=True), 'no_contact', point=[.58, 0., .1])], contacts())
    assert plan['press_candidate'] == 1  # then treated as a failure
    plan, _ = next_attempt([run(first, 'sensor_fault', point=[.58, 0., .1])], contacts())
    assert plan == dict(first, retry=True)


def test_stops_on_success_moved_and_attempt_limit():
    first = attempt(PRESS, 0, 5.)
    assert next_attempt([run(first, 'success', 'tipped')], contacts()) == (None, 'tipped')
    assert next_attempt([run(first, 'moved', 'object moved 20 mm')], contacts())[0] is None
    history = [run(first, 'sensor_fault')] * MAX_ATTEMPTS
    plan, reason = next_attempt(history, contacts())
    assert plan is None and f'{MAX_ATTEMPTS} attempts' in reason


def test_forced_modes_keep_to_one_tip():
    history = [run(attempt(PRESS, 0, 5.), 'slip', point=[.58, 0., .1], force=8.)]
    c = contacts(alternatives=())
    assert next_attempt(history, c, mode='press')[0] is None
    assert next_attempt(history, c, mode='auto')[0]['kind'] == FORWARD
    forward_first = [run(attempt(FORWARD), 'force_limit')]
    assert next_attempt(forward_first, c, mode='forward')[0] is None


def test_forward_slide_and_too_easy():
    slid = {'completed': False, 'failure_reason': 'slide', 'reversal_reason': 'slide',
            'slide_force_n': .54, 'slide_travel_m': .01}
    # The probe saw the slide: the shift it caused is the point, not a reason to stop.
    outcome, why = classify(FORWARD, slid, change=dict(shift_mm=12., height_change_mm=1.))
    assert outcome == 'slid' and '0.54 N over 10 mm' in why
    assert classify(FORWARD, slid)[0] == 'slid'
    # tipT_d: no probe, full travel, the object 67 mm further on at the same height.
    full = {'completed': False, 'failure_reason': 'travel_limit', 'reversal_reason': 'travel_limit'}
    assert classify(FORWARD, full, change=dict(shift_mm=67., height_change_mm=1.))[0] == 'slid'
    assert classify(FORWARD, full, change=dict(shift_mm=3., height_change_mm=0.))[0] == 'no_tip'
    # tipT_e: fell over, whatever the step says.
    assert classify(FORWARD, slid, change=dict(shift_mm=95., height_change_mm=-18.))[0] == 'moved'
    easy = {'completed': False, 'failure_reason': 'too_easy', 'reversal_reason': 'too_easy',
            'contact_force_n': .086}
    outcome, why = classify(FORWARD, easy, change=dict(shift_mm=1., height_change_mm=0.))
    assert outcome == 'too_easy' and '0.09 N' in why
    assert classify(FORWARD, easy, change=dict(shift_mm=95., height_change_mm=-18.))[0] == 'moved'


@pytest.mark.parametrize('outcome', ['slid', 'too_easy'])
def test_slid_or_too_easy_forward_tip_goes_to_press_pull(outcome):
    c = contacts(shape='straight')
    first, _ = first_attempt(c)
    plan, _ = next_attempt([run(first, outcome)], c, force_ref=5.)
    assert plan == attempt(PRESS, 0, 5.)
