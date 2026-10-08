import math
from types import SimpleNamespace as NS
import pytest
from irb120_control.controllers.moveit_single_shot import validate_cable_trajectory, cable_path_constraints


def trajectory(angles):
    return NS(joint_trajectory=NS(joint_names=['joint_6'],points=[NS(positions=[a]) for a in angles]))


def test_fixed_neutral_constraint():
    c=cable_path_constraints().joint_constraints[0]
    assert c.position == 0
    assert c.tolerance_above == pytest.approx(math.pi/3)
    assert c.tolerance_below == pytest.approx(math.pi/3)


@pytest.mark.parametrize('angles', [[0,.2,.5],[-math.pi/3,0,math.pi/3]])
def test_allowed_path(angles):
    assert validate_cable_trajectory(trajectory(angles))


@pytest.mark.parametrize('angles', [[0,2*math.pi,0],[0,-2*math.pi],[1.05],[float('nan')],[],[-1.,1.]])
def test_full_turn_out_of_bounds_or_large_jump_rejected(angles):
    assert not validate_cable_trajectory(trajectory(angles))


def test_missing_wrist_or_positions_rejected():
    t=trajectory([0]);t.joint_trajectory.joint_names=['joint_5']
    assert not validate_cable_trajectory(t)
    t=trajectory([0]);t.joint_trajectory.points[0].positions=[]
    assert not validate_cable_trajectory(t)
