import math
from irb120_control.controllers.joint_verification import JointGoalVerifier


def test_requires_measured_target_dwell_with_joint_name_mapping():
    v=JointGoalVerifier(['a','b'],[0.,1.],started=10.)
    assert not v.observe(10.01,10.02,['b','a'],[1.,.5])
    assert not v.observe(10.1,10.11,['b','a'],[1.,0.])
    assert not v.observe(10.3,10.31,['b','a'],[1.,0.])
    assert v.observe(10.5,10.51,['b','a'],[1.,0.])
    assert v.report['max_error_rad']==0


def test_stale_duplicate_missing_and_nan_cannot_verify():
    for names,positions,stamp,now in [(['a'],[0],9.,10.),(['a'],[0],10.1,11.),
            (['b'],[0],10.1,10.11),(['a'],[math.nan],10.1,10.11),
            (['a'],[0],11.,10.11)]:
        v=JointGoalVerifier(['a'],[0],10.)
        assert not v.observe(stamp,now,names,positions)
    v=JointGoalVerifier(['a'],[0],10.)
    for _ in range(100): assert not v.observe(10.1,10.11,['a'],[0])


def test_gap_and_excursion_reset_dwell():
    v=JointGoalVerifier(['a'],[0],10.)
    assert not v.observe(10.1,10.1,['a'],[0])
    assert not v.observe(10.5,10.5,['a'],[0])
    assert not v.observe(10.6,10.6,['a'],[.02])
    assert not v.observe(10.7,10.7,['a'],[0])
    assert not v.observe(10.9,10.9,['a'],[0])
    assert v.observe(11.1,11.1,['a'],[0])
