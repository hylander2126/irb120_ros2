"""Capture helpers no longer import any retired detector/comparison module."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip('rclpy')
from irb120_perception.stereo_capture_utils import _decode_image, _lookup


def test_decode_native_ir_with_row_padding():
    message = NS(encoding='mono8', height=2, width=2, step=3,
                 data=bytes([1, 2, 99, 3, 4, 99]))
    np.testing.assert_array_equal(_decode_image(message), [[1, 2], [3, 4]])


def test_lookup_preserves_named_transform_direction():
    buffer = Mock()
    buffer.lookup_transform.return_value = NS(transform=NS(
        translation=NS(x=.1, y=.2, z=.3), rotation=NS(x=0., y=0., z=0., w=1.)))
    rotation, translation = _lookup(buffer, 'camera', 'base_link', Mock())
    np.testing.assert_allclose(rotation, np.eye(3))
    np.testing.assert_allclose(translation, [.1, .2, .3])
    assert buffer.lookup_transform.call_args.args[:2] == ('camera', 'base_link')
