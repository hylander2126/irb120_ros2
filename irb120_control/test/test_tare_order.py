"""Execute proposed startup/loop code with mocked nodes; never import ROS."""

import ast
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest


SOURCE_DIR = Path(__file__).parent / 'irb120_control'
if not SOURCE_DIR.exists():
    SOURCE_DIR = Path(__file__).parents[1] / 'irb120_control'


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('failure', [None, 'move', 'tare'])
def test_tare_order_and_failures(legacy, failure):
    filename = 'arc_static.py'
    source = SOURCE_DIR / filename
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    body = next(n for n in main.body if isinstance(n, ast.Try)).body
    start = next(i for i, n in enumerate(body)
                 if isinstance(n, ast.If) and ast.unparse(n.test).startswith('parsed.tare_at_start'))
    fn = ast.parse('def startup():\n    pass').body[0]
    fn.body = body[start:]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    order = []
    node = Mock()
    node._force_ctrl.reference = 5.
    node._completed = True
    node._abort_reason = None
    node._ft_capture = NS(raw=[], metadata=lambda: {}, reset=lambda: None)  # C2 logging
    for attr in ('_object', '_pre_squash_pos', '_arc_center'):
        setattr(node, attr, 'test')
    node.move_to_pre_squash.side_effect = lambda: order.append('move') or failure != 'move'
    node.tare_with_metadata.side_effect = lambda: order.append('tare') or failure != 'tare'
    node.pause_servo.side_effect = lambda: order.append('pause') or failure != 'pause'
    node._operator_confirm.side_effect = lambda message: order.append('confirm') or True
    node.resume_servo.side_effect = lambda: order.append('resume')
    stage = Mock(path=Path('/tmp/mock_stage'))
    episode = Mock()
    episode.begin_stage.return_value = stage
    env = dict(node=node, episode=episode, stage=stage,
               parsed=NS(tare_at_start=legacy, trials=3, quality='h264'),
               ensure_egm_active=lambda n: True, start_recording=lambda *a, **k: True,
               stop_recording=Mock(), save_ft_pose_log=Mock(), save_run_metadata=Mock(),
               module_constants=lambda g: {}, arc_static=NS(), datetime=datetime, LOG_PREFIX='batch',
               run_adaptive_press=lambda *a, **k: order.append('run') or (True, 5., 1))
    # These are used only by the unchanged single-run diagnostic f-string.
    for name in ('FORCE_HARD_LIMIT_N', 'KP_FORCE', 'KI_FORCE', 'FORCE_DEADBAND_N',
                 'FORCE_FILTER_ALPHA', 'MAX_NORMAL_SPEED', 'DESCEND_SPEED',
                 'ARC_MAX_ANGLE_DEG', 'ARC_TANGENTIAL_RAMP_SEC',
                 'UNARC_FORCE_AUGMENT_SPEED', 'UNARC_FORCE_AUGMENT_SOFTNESS_N'):
        env[name] = 1.
    exec(compile(module, str(source), 'exec'), env)
    env['startup']()
    if failure:
        assert 'resume' not in order
        assert 'run' not in order
        if failure == 'move' and not legacy:
            assert 'tare' not in order
        return
    count = 1
    assert order.count('tare') == (1 if legacy else count)
    assert order.count('run') == count
    assert order.index('confirm') < order.index('resume')
    if legacy:
        assert order.index('tare') < order.index('move')
    else:
        previous = -1
        for resume in [i for i, item in enumerate(order) if item == 'resume']:
            segment = order[previous + 1:resume]
            assert segment.index('move') < segment.index('tare')
            assert 'pause' in segment
            previous = resume
        assert order.index('tare') < order.index('confirm')


def test_metadata_uses_only_recent_wrench_and_records_failure():
    source = SOURCE_DIR / 'arc_static.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ArcStatic')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'tare_with_metadata')
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    env = {'np': np, 'tare_netft': lambda n: False}
    exec(compile(module, str(source), 'exec'), env)
    node = NS(_tare_events=[], _now_s=Mock(side_effect=[10., 12.]),
              _ft_transformed_log=[[9., *([99.] * 6)], [9.9, *([2.] * 6)], [9.95, *([4.] * 6)]])
    assert env['tare_with_metadata'](node) is False
    event = node._tare_events[0]
    assert event['pre_tare_sample_count'] == 2
    assert event['pre_tare_mean_wrench'] == [3.] * 6
    assert event['requested_time_s'] == 10.
    assert event['completed_time_s'] == 12.
    assert event['success'] is False


def test_failed_pause_prevents_tare_and_resume():
    test_tare_order_and_failures(False, 'pause')
