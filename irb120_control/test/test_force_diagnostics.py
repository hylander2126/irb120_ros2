from irb120_control.controllers.force_controller import PIDForceController


def controller():
    return PIDForceController(kp=.00035,ki=.000005,force_ref_n=5.,max_normal_speed=.006,
        control_hz=100.,deadband_n=.25,measurement_filter_alpha=.12,output_slew_rate=.02)


def test_diagnostics_do_not_change_control_and_integral_is_bounded():
    a,b=controller(),controller()
    for force in [0.]*500+[8.]*500+[4.2]*100:
        expected=b.update(force)
        assert a.update(force)==expected
        before=dict(a.__dict__)
        d=a.diagnostics
        assert a.__dict__==before
        assert abs(d['integral_velocity'])<=.00001
        assert d['output_velocity']==expected
    # Under-load control pushes inward (caller negates this positive output).
    assert a.diagnostics['output_velocity']>0
