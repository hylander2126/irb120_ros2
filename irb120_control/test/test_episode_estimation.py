"""Episode fitting: direction, force-sign contact and scoring independence."""
import json
import numpy as np
import pytest

from irb120_control.estimation.episode_estimation import (
    analyze_episode, episode_paths, fit_first_moments, reference_for_primitive)
from irb120_control.estimation.fwd_tip_estimation import find_fwd_tip_logs
from irb120_control.estimation.tip_core import detect_onset, tilt_noslip
from test_tip_core import _press_pull_log


def test_joint_first_moments_recover_mass_and_height_without_anchor():
    log, truth = _press_pull_log()
    fit = fit_first_moments(log, tilt_noslip(log, log.pivot), truth['cx'])
    assert fit['mean']['m'] == pytest.approx(truth['m'], abs=1e-10)
    assert fit['mean']['zc'] == pytest.approx(truth['cz'], abs=1e-10)
    assert len(fit['branches']) == 2


def test_loaded_press_contact_does_not_require_force_along_arc():
    log, _ = _press_pull_log()
    # Move contact over the outside of the support and preserve the required
    # gravity moment. A vertical load now needs a force opposing the arc.
    torque_y = -np.cross(log.p_ball-log.pivot, log.f_S)[:, 1]
    log.p_ball[:, 0] -= .1
    r = log.p_ball - log.pivot
    force = np.c_[(torque_y + r[:, 0]*-5.)/r[:, 2], np.zeros(len(r)), np.full(len(r), -5.)]
    log.p_S = log.p_ball - np.array([.0866512, 0, 0])
    log.f_S = -force
    log.tau_S = np.cross(log.p_ball-log.p_S, log.f_S)
    assert not (log.f_push > .12).any()
    onset = detect_onset(log, log.pivot)
    assert onset['contact'] and onset['contact_source'] == 'controller'
    assert 'i_on' in onset
    assert onset['slide_m'] == 0


def test_opposite_edge_reference_does_not_mutate_forward_reference():
    ref = dict(mass=.404, com=np.array([.0526, 0., .122]), com_obj=[.0074, .122])
    backward = reference_for_primitive(ref, 'press_pull')
    assert backward['com'][0] == .0074
    assert backward['theta_deg'] == pytest.approx(3.4710651, abs=1e-5)
    assert ref['com'][0] == .0526


def test_gt_mass_and_height_only_change_scores(tmp_path, monkeypatch):
    import irb120_control.estimation.estimate_params as loaders
    log, _ = _press_pull_log()
    monkeypatch.setattr(loaders, 'tiplog_from_press_pull', lambda path: log)
    stage = tmp_path/'03_press_pull_tip'
    stage.mkdir()
    (stage/'arc_static_20261016_161229.npz').touch()
    (stage/'arc_static_20261016_161229.json').write_text('{}')
    ep = tmp_path/'episode.json'
    ep.write_text(json.dumps(dict(object='tipT_d', stages=[dict(kind='press_pull_tip',
        dir=stage.name, status='ok')], outcome=dict(result='success'))))
    ref = dict(mass=.404, com=np.array([.032, 0., .094]), com_obj=[.028, .094])
    a = analyze_episode(ep, ref)['stages'][0]
    ref['mass'] = 9.
    ref['com'][2] = .8
    ref['com_obj'][1] = .8
    b = analyze_episode(ep, ref)['stages'][0]
    assert a['status'] == b['status'] == 'estimated'
    assert a['estimate'] == b['estimate']
    assert a['score'] != b['score']


def test_discovery_finds_arc_and_straight_without_most_recent(tmp_path):
    for kind, filename in [('forward_arc', 'fwd_arc_20261016_161229.npz'),
                           ('forward_tip', 'fwd_tip_20261015_161229.npz')]:
        folder = tmp_path/'runtime_logs/tipT_a/episodes/20261016_150000'/('01_'+kind)
        folder.mkdir(parents=True)
        (folder/filename).touch()
        (folder/'most_recent.npz').touch()
    assert len(find_fwd_tip_logs('tipT_a', str(tmp_path))) == 2
    assert len(find_fwd_tip_logs('tipT_a', str(tmp_path/'runtime_logs'))) == 2
    for stamp in ('20261015_120000', '20261016_160000'):
        ep = tmp_path/'runtime_logs/tipT_a/episodes'/stamp
        ep.mkdir()
        (ep/'episode.json').write_text('{}')
    assert episode_paths(tmp_path, 'tipT_a')[0].parent.name == '20261016_160000'


def test_slide_stage_is_preserved_without_fitting(tmp_path, monkeypatch):
    import irb120_control.estimation.fwd_tip_estimation as loader
    log, _ = _press_pull_log()
    log.extra['meta'] = dict(reversal_reason='slide')
    monkeypatch.setattr(loader, 'load_fwd_tip', lambda path: log)
    stage = tmp_path/'01_forward_arc'
    stage.mkdir()
    (stage/'fwd_arc_20261016_161229.npz').touch()
    (stage/'fwd_arc_20261016_161229.json').write_text('{}')
    ep = tmp_path/'episode.json'
    ep.write_text(json.dumps(dict(object='tipT_d', stages=[dict(kind='forward_arc',
        dir=stage.name, status='failed')], outcome=dict(result='success'))))
    result = analyze_episode(ep, {})['stages'][0]
    assert result['status'] == 'skipped' and result['reason'] == 'slide'
    assert 'estimate' not in result


@pytest.mark.parametrize('slide', [0., .010])
def test_first_contact_reference_handles_slide_without_rezeroing_at_peak(tmp_path, slide):
    from pathlib import Path
    from test_fwd_tip_estimation import _synthetic_episode
    path, gt = _synthetic_episode(tmp_path, slide=slide)
    ep = Path(path).parent.parent/'episode.json'
    ep.write_text(json.dumps(dict(object='tipT_a', outcome=dict(result='success'),
        stages=[dict(kind='forward_tip', dir=Path(path).parent.name, status='ok')])) )
    result = analyze_episode(ep, gt)['stages'][0]
    assert result['status'] == 'estimated'
    assert result['forward_reference'] == 'first_contact'
    assert result['estimate']['mean']['m'] == pytest.approx(gt['mass'], rel=.02)
    assert result['estimate']['mean']['zc'] == pytest.approx(gt['com'][2], abs=.003)


@pytest.fixture(autouse=True)
def synthetic_cloud_height(monkeypatch):
    """These dynamics fixtures have no perception files; supply a separate 30 cm envelope."""
    import irb120_control.estimation.tip_core as core
    import irb120_control.estimation.episode_estimation as episodes
    bound = lambda log: (.30, 'synthetic independent cloud envelope')
    monkeypatch.setattr(core, 'cloud_height_bound', bound)
    monkeypatch.setattr(episodes, 'cloud_height_bound', bound)
