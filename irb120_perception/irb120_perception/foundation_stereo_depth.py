"""
Fast-FoundationStereo depth node.
==================================
Runs NVIDIA Fast-FoundationStereo (github.com/NVlabs/Fast-FoundationStereo)
on one camera's own rectified IR1/IR2 pair and republishes a dense depth
estimate alongside — not instead of — the RealSense driver's on-chip depth.
Purely additive: no existing topic is touched, subscribed to, or modified.

One node instance per camera, by design (see this package's
`launch/foundation_stereo.launch.py`, parameterized by `camera_name`, meant to
be run once per camera in its own terminal next to `irb120_handeye`'s
`bringup_cams.launch.py`, which must publish infra1/infra2 for this node to
have anything to subscribe to).

Interpreter: like `object_detector_sam`, this node needs torch, which lives
in the `irb_venv` virtualenv, not in `irb120_perception`'s normal
system-Python build. Launch it through `prefix=[VENV_PYTHON]` the way
`perception_sam.launch.py` already does for SAM — see that file's docstring
("Interpreter", "CPU budget") for why `irb120_perception` itself must stay
built under the system interpreter, and why this process's BLAS thread
pools must be capped (`OMP_NUM_THREADS` etc.) next to the 250 Hz EGM loop.

Frame note: this node's output depth lives in the **left-IR optical frame**
(same frame/intrinsics as the input `infra1` stream), not the
`aligned_depth_to_color` frame the rest of this workspace's recorded
snapshots use. That's a deliberate, honest default — this node does not
invent a color-aligned reprojection it hasn't verified. Feeding this into
the existing TSDF/snapshot tooling will need the infra1-optical extrinsics
composed in at that point; this node only publishes what it actually
computed.

Topics (all parameterized — see `declare_parameter` calls below for exact
names/defaults):
  Subscribes: infra1, infra2 (sensor_msgs/Image, mono8, rectified), and the
    matching camera_info (sensor_msgs/CameraInfo) for the left-IR intrinsics.
  Publishes:  <output_image_topic> (sensor_msgs/Image, 32FC1, metres) and
              <output_camera_info_topic> (sensor_msgs/CameraInfo, passthrough
              of the input camera_info, same stamp/frame as the depth).

No cv_bridge dependency, matching the rest of this package (see
`object_detector_sam.py`'s `_image_to_rgb`/`_image_to_depth_m`) — decodes
sensor_msgs/Image by hand instead.
"""
import copy
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
import message_filters
from sensor_msgs.msg import CameraInfo, Image

_FFS_IMPORT_ERROR = None


def _image_to_mono8(msg: Image) -> np.ndarray:
    """Decode a sensor_msgs/Image (mono8) to an HxW uint8 array."""
    if msg.encoding not in ('mono8', '8UC1'):
        raise ValueError(f"unsupported IR encoding '{msg.encoding}' (expected mono8)")
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return np.ascontiguousarray(buf.reshape(msg.height, msg.step)[:, :msg.width])


def _depth_to_image_msg(depth_m: np.ndarray, header) -> Image:
    """Encode an HxW float32 metres array as a 32FC1 sensor_msgs/Image."""
    msg = Image()
    msg.header = header
    msg.height, msg.width = depth_m.shape
    msg.encoding = '32FC1'
    msg.is_bigendian = 0
    msg.step = msg.width * 4
    msg.data = np.ascontiguousarray(depth_m, dtype=np.float32).tobytes()
    return msg


class FoundationStereoDepthNode(Node):

    def __init__(self):
        super().__init__('foundation_stereo_depth')

        default_repo = os.path.expanduser('~/Documents/irb120_ws/third_party/Fast-FoundationStereo')
        default_ckpt = os.path.join(default_repo, 'weights', '23-36-37', 'model_best_bp2_serialize.pth')

        self.declare_parameter('infra1_topic', '/realsense/infra1/image_rect_raw')
        self.declare_parameter('infra2_topic', '/realsense/infra2/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/realsense/infra1/camera_info')
        self.declare_parameter('right_camera_info_topic', '/realsense/infra2/camera_info')
        self.declare_parameter('output_image_topic', '/realsense/depth_ffs/image_raw')
        self.declare_parameter('output_camera_info_topic', '/realsense/depth_ffs/camera_info')
        self.declare_parameter('checkpoint_path', default_ckpt)
        self.declare_parameter('ffs_repo_path', default_repo)
        self.declare_parameter('baseline_m', 0.0)  # zero: derive per-device baseline from rectified CameraInfo.P
        self.declare_parameter('valid_iters', 8)
        self.declare_parameter('max_disp', 192)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('min_disparity_px', 0.5)
        self.declare_parameter('min_depth_m', 0.2)
        self.declare_parameter('max_depth_m', 1.4)
        self.declare_parameter('sync_slop_sec', 0.01)

        p = self.get_parameter
        self.infra1_topic = p('infra1_topic').value
        self.infra2_topic = p('infra2_topic').value
        self.camera_info_topic = p('camera_info_topic').value
        self.output_image_topic = p('output_image_topic').value
        self.output_camera_info_topic = p('output_camera_info_topic').value
        self.checkpoint_path = p('checkpoint_path').value
        self.ffs_repo_path = p('ffs_repo_path').value
        self.baseline_m = float(p('baseline_m').value)
        self.valid_iters = int(p('valid_iters').value)
        self.max_disp = int(p('max_disp').value)
        self.device = p('device').value
        self.min_disparity_px = float(p('min_disparity_px').value)
        self.min_depth_m = float(p('min_depth_m').value)
        self.max_depth_m = float(p('max_depth_m').value)

        self._model = None
        self._torch = None
        self._padder_cls = None
        self._amp_dtype = None
        self._camera_info = None
        self._right_camera_info = None
        self._busy = False
        self._warned_no_info = False
        self._frame_count = 0

        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            self.get_logger().error(
                f"'checkpoint_path' ({self.checkpoint_path!r}) does not exist — this node will "
                f"subscribe but never publish depth until a valid Fast-FoundationStereo .pth "
                f"checkpoint is given. See third_party/Fast-FoundationStereo/readme.md.")
        else:
            self._load_model()

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_cb, sensor_qos)
        self.create_subscription(CameraInfo, p('right_camera_info_topic').value,
                                 self._right_info_cb, sensor_qos)
        infra1_sub = message_filters.Subscriber(self, Image, self.infra1_topic, qos_profile=sensor_qos)
        infra2_sub = message_filters.Subscriber(self, Image, self.infra2_topic, qos_profile=sensor_qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [infra1_sub, infra2_sub], queue_size=3,
            slop=float(p('sync_slop_sec').value))
        self._sync.registerCallback(self._pair_cb)
        # Keep strong references — message_filters subscriptions can
        # otherwise be garbage-collected despite registered callbacks.
        self._subs = (infra1_sub, infra2_sub)

        self._depth_pub = self.create_publisher(Image, self.output_image_topic, sensor_qos)
        self._info_pub = self.create_publisher(CameraInfo, self.output_camera_info_topic, sensor_qos)

        self.get_logger().info(
            f'foundation_stereo_depth ready — {self.infra1_topic} + {self.infra2_topic} -> '
            f'{self.output_image_topic}' +
            ('' if self._model is not None else ' (model NOT loaded — see error above)'))

    # -------------------------------------------------------------------------

    def _load_model(self):
        global _FFS_IMPORT_ERROR
        if self.ffs_repo_path not in sys.path:
            sys.path.insert(0, self.ffs_repo_path)
        try:
            import torch
            from core.utils.utils import InputPadder
            from Utils import AMP_DTYPE
        except ImportError as exc:  # pragma: no cover - exercised only without the extra deps installed
            _FFS_IMPORT_ERROR = exc
            self.get_logger().error(
                f"Fast-FoundationStereo not importable from ffs_repo_path={self.ffs_repo_path!r} "
                f"({exc}); this node will subscribe but never publish depth.")
            return

        self._torch = torch
        self._padder_cls = InputPadder
        self._amp_dtype = AMP_DTYPE
        model = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        model.args.valid_iters = self.valid_iters
        model.args.max_disp = self.max_disp
        model.to(self.device).eval()
        self._model = model
        self.get_logger().info(
            f'Loaded Fast-FoundationStereo device={self.device} checkpoint={self.checkpoint_path}')

    def _camera_info_cb(self, msg):
        self._camera_info = msg

    def _right_info_cb(self, msg):
        self._right_camera_info = msg

    def _pair_cb(self, infra1_msg, infra2_msg):
        if self._model is None:
            return
        if self._camera_info is None:
            if not self._warned_no_info:
                self.get_logger().warn(
                    f"no camera_info received yet on {self.camera_info_topic!r} — waiting.")
                self._warned_no_info = True
            return
        if self.baseline_m <= 0 and self._right_camera_info is None:
            return
        if self._busy:
            return  # inference (~100ms) can't keep up with 30Hz IR; drop instead of queueing
        self._busy = True
        try:
            self._infer_and_publish(infra1_msg, infra2_msg)
        finally:
            self._busy = False

    def _infer_and_publish(self, infra1_msg, infra2_msg):
        torch = self._torch
        ir1 = _image_to_mono8(infra1_msg)
        ir2 = _image_to_mono8(infra2_msg)
        info = self._camera_info
        if (info.width, info.height) != (ir1.shape[1], ir1.shape[0]) or ir1.shape != ir2.shape:
            self.get_logger().warning('IR/calibration dimensions differ; skipping pair')
            return
        fx = info.p[0]
        baseline = self.baseline_m
        cx_right = info.p[2]
        if baseline <= 0:
            right = self._right_camera_info
            if (right.width, right.height) != (info.width, info.height) or abs(right.p[0]-fx) > .01:
                return
            baseline = info.p[3]/fx-right.p[3]/right.p[0]
            cx_right = right.p[2]
        if not .01 < baseline < .2:
            self.get_logger().error('Invalid calibrated baseline; refusing depth publication')
            return

        img0 = np.tile(ir1[..., None], (1, 1, 3))
        img1 = np.tile(ir2[..., None], (1, 1, 3))
        H, W = img0.shape[:2]
        t0 = torch.as_tensor(img0).to(self.device).float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(img1).to(self.device).float()[None].permute(0, 3, 1, 2)
        padder = self._padder_cls(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)

        t_start = time.monotonic()
        with torch.no_grad(), torch.amp.autocast(self.device, enabled=(self.device == 'cuda'), dtype=self._amp_dtype):
            disp = self._model.forward(
                t0, t1, iters=self.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W).clip(0, None)
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        from .stereo_geometry import disparity_depth
        depth = disparity_depth(disp, fx, baseline, info.p[2], cx_right,
                                self.min_depth_m, self.max_depth_m, self.min_disparity_px)

        self._depth_pub.publish(_depth_to_image_msg(depth, infra1_msg.header))
        info_out = copy.deepcopy(self._camera_info)
        info_out.header = infra1_msg.header
        self._info_pub.publish(info_out)

        self._frame_count += 1
        if self._frame_count == 1:
            self.get_logger().info(f'first depth frame published: {elapsed_ms:.1f} ms, {W}x{H}')
        self.get_logger().debug(f'inference {elapsed_ms:.1f} ms', throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = FoundationStereoDepthNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
