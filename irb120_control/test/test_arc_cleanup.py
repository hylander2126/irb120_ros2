"""Run actual cleanup branch against failed/verified home outcomes, without motion."""
import ast
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest


@pytest.mark.parametrize('pause_ok,home_ok',[(True,False),(True,True),(False,False)])
def test_home_gates_snapshot_and_stage_success(tmp_path,pause_ok,home_ok):
    source=Path(__file__).parents[1]/'irb120_control/arc_static.py'
    tree=ast.parse(source.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    cleanup=next(n for n in main.body if isinstance(n,ast.Try)).finalbody
    fn=ast.parse('def cleanup():\n pass').body[0];fn.body=cleanup
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    order=[]
    node=Mock()
    node.pause_servo.return_value=pause_ok
    node.move_to_home.side_effect=lambda: order.append('home') or home_ok
    node._completed=True;node._pre_squash_pos=[0,0,0];node._arc_center=[0,0,0]
    node._last_joint_goal_verification={'verified':home_ok}
    node._tracking_log=[];node._force_diagnostics=[]
    node._ft_capture=NS(raw=[], metadata=lambda: {});node._abort_reason=None
    stage=Mock(path=tmp_path);snapshot=Mock()
    env=dict(node=node,stage=stage,episode=Mock(),motion_started=True,home_ok=None,
        rclpy=Mock(),stop_recording=lambda n:order.append('stop_recording'),
        save_ft_pose_log=Mock(),save_run_metadata=Mock(),module_constants=lambda g:{},
        datetime=datetime,Path=Path,json=json,force_ref=5.,attempt=1,
        deactivate_egm=Mock(),take_snapshot=snapshot,
        parsed=NS(press_candidate=0,press_backup=0,tare_at_start=False))
    exec(compile(module,str(source),'exec'),env);env['cleanup']()
    assert stage.end.call_args.args[0]==('ok' if pause_ok and home_ok else 'failed')
    assert snapshot.call_count==int(pause_ok and home_ok)
    if pause_ok: assert order==['home','stop_recording']
    else: node.move_to_home.assert_not_called()
