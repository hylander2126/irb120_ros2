"""Image decoding and TF lookup for the supported native-stereo capture."""
import time
import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformException

def _quat_matrix(x, y, z, w) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

def _decode_image(message) -> np.ndarray:
    """Decode a sensor_msgs/Image by hand (mono8/16UC1/32FC1/rgb8/bgr8) — no cv_bridge."""
    encoding = message.encoding.lower()
    if encoding in ('mono8', '8uc1'):
        dtype, channels = np.uint8, 1
    elif encoding in ('16uc1', 'mono16'):
        dtype, channels = np.uint16, 1
    elif encoding == '32fc1':
        dtype, channels = np.float32, 1
    elif encoding in ('rgb8', 'bgr8'):
        dtype, channels = np.uint8, 3
    else:
        raise ValueError(f'unsupported encoding {message.encoding!r}')
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, :message.width * channels * np.dtype(dtype).itemsize].view(dtype)
    image = image.reshape(message.height, message.width, channels) if channels == 3 \
        else image.reshape(message.height, message.width)
    if encoding == 'rgb8':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def _lookup(buffer: Buffer, target: str, source: str, node, timeout_sec: float = 5.0):
    """(R, t) with p_target = R @ p_source + t. Spins the node while waiting."""
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            tf = buffer.lookup_transform(target, source, rclpy.time.Time(), timeout=Duration(seconds=0.1))
            t, q = tf.transform.translation, tf.transform.rotation
            return _quat_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])
        except TransformException:
            if time.monotonic() > deadline:
                raise
            rclpy.spin_once(node, timeout_sec=0.05)

