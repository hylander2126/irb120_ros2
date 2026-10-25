"""Press approach feasibility against the object's padded bounding box."""
from irb120_perception.press_targets import press_approachable


def test_pre_press_must_clear_the_box_top():
    top = .262  # toy_robot cloud top (base_link)
    assert not press_approachable([.622, 0., .2405], top)   # head contact: planning failed
    assert not press_approachable([.651, 0., .207], top)    # backup B2: planning failed
    assert press_approachable([.573, 0., .270], top)        # head top: clears the box
    assert press_approachable([.609, 0., .1737], .1605)     # tipT_c5 conservative press that ran
