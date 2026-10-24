"""No robot or ROS graph needed for candidate selection tests."""
import copy

import pytest

from irb120_control.util.shove_check import check_candidates, validate_request


def request():
    samples = [dict(time_s=t, position=[.4+x, 0., .1],
                    orientation_xyzw=[0., 0., 0., 1.])
               for t, x in [(0., 0.), (.1, .02), (.15, .025)]]
    return dict(joint_names=['joint_6'], initial_joints=[0.],
                candidates=[dict(id='nominal', release_index=1, samples=samples),
                            dict(id='shorter_contact', release_index=1,
                                 samples=copy.deepcopy(samples))])


def test_fallback_and_preserve_requested_timing():
    data = request()
    original = copy.deepcopy(data)
    seen = []

    def planner(req, candidate):
        seen.append(candidate['id'])
        return dict(geometric_path_found=candidate['id'] == 'shorter_contact')

    result = check_candidates(data, planner)
    assert seen == ['nominal', 'shorter_contact']
    assert result['selected_candidate'] == 'shorter_contact'
    assert not result['execution_ready'] and not result['timing_verified']
    assert result['request'] == original == data


def test_stop_after_first_complete_plan():
    result = check_candidates(request(), lambda *_: dict(geometric_path_found=True))
    assert len(result['attempts']) == 1


def test_all_failed_is_no_plan():
    result = check_candidates(request(), lambda *_: dict(geometric_path_found=False))
    assert result['selected_candidate'] is None


@pytest.mark.parametrize('mutation', [
    lambda r: r['initial_joints'].__setitem__(0, 2.),
    lambda r: r['candidates'][0].__setitem__('release_index', 2),
    lambda r: r['candidates'][0]['samples'][1].__setitem__('time_s', 0.),
    lambda r: r['candidates'][0]['samples'][1].__setitem__('position', [float('nan'), 0., 0.]),
    lambda r: r['candidates'][0]['samples'][0].__setitem__('orientation_xyzw', [0., 0., 0., 0.]),
])
def test_invalid_requests(mutation):
    data = request()
    mutation(data)
    with pytest.raises(ValueError):
        validate_request(data)
