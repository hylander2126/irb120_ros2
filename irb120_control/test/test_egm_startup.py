"""Exercise startup ordering without constructing ROS nodes or contacting RWS."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from irb120_control.util.egm_handler import EGMHandler


def handler(stop=1, settings=1, start=1, services=True, jtc=True):
    return SimpleNamespace(
        _ready_pub=Mock(),
        _wait_for_startup_services=Mock(return_value=services),
        _call_trigger=Mock(side_effect=[(stop, ''), (start, '')]),
        _set_egm_settings=Mock(return_value=(settings, '')),
        _wait_for_jtc=Mock(return_value=jtc),
        get_logger=Mock(return_value=Mock()),
        egm_stop_srv='stop', egm_start_srv='start', startup_service_timeout_sec=30.,
    )


def test_success_announces_ready_without_a_trajectory():
    fake = handler()
    assert EGMHandler.startup_sequence(fake) is True
    assert [call.args[0].data for call in fake._ready_pub.publish.call_args_list] == [False, True]
    assert [call.args[0] for call in fake._call_trigger.call_args_list] == ['stop', 'start']
    # The fake intentionally has no trajectory action client or send-hold method.


@pytest.mark.parametrize('failure', ['services', 'stop', 'settings', 'start', 'jtc'])
def test_failure_never_announces_ready(failure):
    fake = handler(**{failure: False if failure in ('services', 'jtc') else 3005})
    assert EGMHandler.startup_sequence(fake) is False
    assert [call.args[0].data for call in fake._ready_pub.publish.call_args_list] == [False]
    if failure == 'services':
        fake._call_trigger.assert_not_called()
    if failure == 'stop':
        fake._set_egm_settings.assert_not_called()
    if failure in ('stop', 'settings'):
        assert fake._call_trigger.call_count == 1
    if failure != 'jtc':
        fake._wait_for_jtc.assert_not_called()


def test_failed_settings_read_is_not_written_back():
    response = SimpleNamespace(result_code=3005, message='RAPID unavailable')
    future = Mock()
    future.result.return_value = response
    get_client, set_client = Mock(), Mock()
    get_client.call_async.return_value = future
    fake = SimpleNamespace(
        create_client=Mock(side_effect=[get_client, set_client]),
        _wait_for_service=Mock(return_value=True),
        _spin_future=Mock(return_value=True),
        get_settings_srv='get', set_settings_srv='set', task='T_ROB1',
    )
    assert EGMHandler._set_egm_settings(fake) == (3005, 'RAPID unavailable')
    set_client.call_async.assert_not_called()
