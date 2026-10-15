"""run_pipeline's tip phase end to end, with the motion steps and snapshots faked."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from irb120_control import orchestrator


def test_slip_then_riskier_contact_succeeds(monkeypatch):
    contacts = {'press': dict(available=True, point=[.58, 0., .1]),
                'press_alternatives': [dict(available=True, rank=1, point=[.56, 0., .1])],
                'forward_tip': {'available': True, 'pivot_shape': {'shape': 'curved', 'reason': 't'}}}
    cloud = np.random.default_rng(0).uniform([.55, -.03, -.02], [.62, .03, .12], (300, 3))
    results = iter([{'completed': False, 'force_cap_reached': True, 'final_force_ref_n': 13.},
                    {'completed': True, 'balance_reached': True}])
    commands = []

    def run_step(episode, kind, command):
        commands.append(command)
        return dict(dir=f'0{len(commands)}_{kind}', status='ok', results=next(results)), True

    monkeypatch.setattr(orchestrator, 'show_snapshot', lambda episode, open_images: contacts)
    monkeypatch.setattr(orchestrator, 'latest_object_cloud', lambda episode: cloud)
    monkeypatch.setattr(orchestrator, 'press_pull_slip_mm', lambda path: None)
    monkeypatch.setattr(orchestrator, 'run_step', run_step)
    episode = Mock(path=orchestrator.Path('/tmp/episode'))
    args = SimpleNamespace(tip='auto', force_ref=5., open_images=False, auto=True,
                           object='heart_a', quality='h264')

    assert orchestrator.run_tip_cascade(None, episode, args) is None
    assert commands[0][:6] == ['arc_static', 'heart_a', '--force-ref', '5.000', '--press-candidate', '0']
    assert commands[1][:6] == ['arc_static', 'heart_a', '--force-ref', '13.000', '--press-candidate', '1']
    events = [c.args[0] for c in episode.add_event.call_args_list]
    assert any('tip attempt 1: slip' in e for e in events)
    assert any('tip attempt 2: success' in e for e in events)
    episode.set_outcome.assert_not_called()


def test_auto_mode_aborts_when_nothing_is_left(monkeypatch):
    contacts = {'press': dict(available=True, point=[.58, 0., .1]), 'press_alternatives': [],
                'forward_tip': {'available': False, 'pivot_shape': {'shape': 'curved', 'reason': 't'}}}
    monkeypatch.setattr(orchestrator, 'show_snapshot', lambda episode, open_images: contacts)
    monkeypatch.setattr(orchestrator, 'latest_object_cloud', lambda episode: np.zeros((0, 3)))
    monkeypatch.setattr(orchestrator, 'press_pull_slip_mm', lambda path: 9.)
    monkeypatch.setattr(orchestrator, 'run_step', lambda episode, kind, command: (
        dict(dir='01_press_pull_tip', status='ok', results={'completed': True}), True))
    episode = Mock(path=orchestrator.Path('/tmp/episode'))
    args = SimpleNamespace(tip='auto', force_ref=5., open_images=False, auto=True,
                           object='heart_a', quality='h264')
    assert orchestrator.run_tip_cascade(None, episode, args) is episode
    assert episode.set_outcome.call_args.args[0] == 'aborted'


def test_forward_slide_is_the_slide_interaction_then_press_pull(monkeypatch):
    contacts = {'press': dict(available=True, point=[.6, 0., .16]), 'press_alternatives': [],
                'forward_tip': {'available': True, 'pivot_shape': {'shape': 'straight', 'reason': 't'}}}
    cloud = np.random.default_rng(0).uniform([.55, -.03, -.01], [.73, .03, .17], (300, 3))
    clouds = iter([cloud, cloud + [.01, 0., 0.], cloud + [.01, 0., 0.], cloud + [.01, 0., 0.]])
    results = iter([{'completed': False, 'failure_reason': 'slide', 'reversal_reason': 'slide',
                     'slide_force_n': .54, 'slide_travel_m': .01},
                    {'completed': True, 'balance_reached': True}])
    commands = []

    def run_step(episode, kind, command):
        commands.append(command)
        return dict(dir=f'0{2 * len(commands) - 1}_{kind}', status='ok', results=next(results)), True

    monkeypatch.setattr(orchestrator, 'show_snapshot', lambda episode, open_images: contacts)
    monkeypatch.setattr(orchestrator, 'latest_object_cloud', lambda episode: next(clouds))
    monkeypatch.setattr(orchestrator, 'press_pull_slip_mm', lambda path: None)
    monkeypatch.setattr(orchestrator, 'run_step', run_step)
    episode = Mock(path=orchestrator.Path('/tmp/episode'))
    args = SimpleNamespace(tip='auto', force_ref=5., open_images=False, auto=True,
                           object='tipT_d', quality='h264')
    slides = []
    assert orchestrator.run_tip_cascade(None, episode, args, slides) is None
    assert [c[0] for c in commands] == ['fwd_tip', 'arc_static']
    assert slides == ['01_forward_tip']
    events = [c.args[0] for c in episode.add_event.call_args_list]
    assert any('tip attempt 1: slid' in e for e in events)
    assert any('slide interaction: 01_forward_tip' in e for e in events)
    # --auto: the push phase is skipped; the slide was the push.
    monkeypatch.setattr(orchestrator, 'run_step', Mock(side_effect=AssertionError('pushed')))
    assert orchestrator.run_push(None, episode, args, slides[-1]) is None
    assert 'push: skipped' in episode.add_event.call_args.args[0]
