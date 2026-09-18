"""Check selector integration without nodes, service calls, or robot motion."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from rclpy.time import Time

from irb120_control.util import press_point_check as check


@pytest.mark.parametrize('available', [True, False])
def test_new_selector_drives_check(monkeypatch, available):
    data = np.array([(0.6, 0., 0.2, 0)],
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'i4')])
    monkeypatch.setattr(check.point_cloud2, 'read_points', Mock(return_value=data))
    press = {'available': available, 'point': np.array([0.6, 0., 0.221]),
             'reason': 'No candidates after inset filter', 'candidate_counts': {'inset': 0}}
    selector = Mock(return_value={'press': press})
    monkeypatch.setattr(check, 'select_contact_points', selector)
    for name in ('_report', '_publish_press_point_marker', '_clear_press_point_marker',
                 '_set_perception_active'):
        monkeypatch.setattr(check, name, Mock())
    msg = SimpleNamespace(header=SimpleNamespace(frame_id='base_link'))
    node = Mock()
    node.get_clock.return_value.now.return_value = Time()
    node.create_subscription.side_effect = lambda _, topic, callback, qos: callback(msg)

    def transform(dst, src, _):
        dz = 0.021 if dst == 'world' else -0.021
        return SimpleNamespace(transform=SimpleNamespace(
            translation=SimpleNamespace(x=0., y=0., z=dz),
            rotation=SimpleNamespace(x=0., y=0., z=0., w=1.)))
    node._tf_buffer.lookup_transform.side_effect = transform
    assert check.check_press_point(node, [0.6, 0., 0.271]) is available
    np.testing.assert_allclose(selector.call_args.args[0], [[0.6, 0., 0.221]])
    assert selector.call_args.kwargs == {'table_z': 0.0}
    assert node._press_point_check_result['selector'] == 'contact_point_selector'
    assert node._press_point_check_result['ok'] is available
    assert check._set_perception_active.call_args.args == (node, False)
    check._clear_press_point_marker.assert_called_once_with(node)
    if available:
        np.testing.assert_allclose(node._press_point_check_result['computed_xyz'], [0.6, 0., 0.271])
        np.testing.assert_allclose(check._publish_press_point_marker.call_args.args[1], [0.6, 0., 0.25])
    else:
        check._publish_press_point_marker.assert_not_called()
