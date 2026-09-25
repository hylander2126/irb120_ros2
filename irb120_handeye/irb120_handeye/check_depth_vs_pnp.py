"""Compare a camera's stereo depth against the ChArUco board's PnP pose.

Both measurements are in the camera's own color optical frame, so the hand-eye
extrinsic plays no part: this isolates the RealSense depth itself. For each
frame the board is detected in the color image (solvePnP -> a plane), the
aligned depth pixels inside the board's inner-corner outline are back-projected,
and each pixel's depth is compared to where its ray meets the PnP plane.

  bias  -- median (depth - PnP) along the optical axis. A consistent offset
           here is a depth error the hand-eye calibration cannot see or fix.
  tilt  -- angle between a plane fit to the depth points and the PnP plane.
  noise -- robust std. dev. of the per-pixel difference (depth noise on a flat
           surface).

Repeat at two or three board distances: a bias that grows in proportion to
distance is a depth-scale error; a constant one is an offset. Either is fixed
in the RealSense Viewer (On-Chip Calibration, then Tare Calibration). Tare
measures the center of the image: put the board over it and enter the printed
"Tare ground truth".

PnP distance is only as good as the board's square length: a printer that
scaled the board by 0.5% shifts PnP distance by 0.5%. Set
SQUARE_LENGTH_M_OVERRIDE in run_handeye_calibration.py to the caliper-measured
value if it hasn't been.

Needs depth on, i.e. the normal camera profile (not calibration:=true):
  ros2 run irb120_handeye check_depth_vs_pnp --camera realsense2
"""

import argparse
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from irb120_handeye.run_handeye_calibration import (
    BOARD_TYPE,
    BOARD_YAML_OVERRIDE,
    CAMERA_LINK_FRAMES,
    DEFAULT_MIN_CHARUCO_CORNERS,
    MARKER_LENGTH_M_OVERRIDE,
    MARKER_SEPARATION_M_OVERRIDE,
    SQUARE_LENGTH_M_OVERRIDE,
    _detect_board_pose,
    _load_board,
)

BIAS_OK_MM = 2.0        # |median depth - PnP| below this (and below BIAS_OK_PCT) -> depth agrees
BIAS_OK_PCT = 0.4
TILT_OK_DEG = 1.5
MIN_BOARD_PIXELS = 500  # valid depth pixels inside the board outline needed to use a frame


class DepthPnpNode(Node):
    """Caches the latest color frame and processes each aligned depth frame against it."""

    def __init__(self, ns: str):
        super().__init__('check_depth_vs_pnp')
        self.bridge = CvBridge()
        self.color = None
        self.info = None
        self.depth_frames = []
        self.create_subscription(Image, f'/{ns}/color/image_raw', self._on_color, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, f'/{ns}/color/camera_info', self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, f'/{ns}/aligned_depth_to_color/image_raw', self._on_depth,
                                 qos_profile_sensor_data)

    def _on_color(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

    def _on_info(self, msg):
        self.info = msg

    def _on_depth(self, msg):
        if self.color is None or self.info is None:
            return
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth_m = depth.astype(np.float64) * (0.001 if depth.dtype == np.uint16 else 1.0)
        self.depth_frames.append((self.color.copy(), depth_m))


def _measure(gray, depth_m, board, K, D, min_features):
    """One frame -> dict of stats, or a string saying why it was skipped."""
    det = _detect_board_pose(gray, board, K, D, min_features)
    if det is None:
        return 'board not detected'

    # Outline of the inner chessboard corners (inset from the board's printed
    # edge, so edge pixels that mix board and background depth are excluded).
    cc = board.board.chessboardCorners
    (x0, y0), (x1, y1) = cc[:, :2].min(axis=0), cc[:, :2].max(axis=0)
    outline = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(det.R)
    px, _ = cv2.projectPoints(outline, rvec, det.t, K, D)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(px.reshape(-1, 2)).astype(np.int32)], 1)

    # Aligned depth normally matches the color resolution; scale if it doesn't.
    if depth_m.shape != gray.shape:
        mask = cv2.resize(mask, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
    sy, sx = depth_m.shape[0] / gray.shape[0], depth_m.shape[1] / gray.shape[1]
    vs, us = np.nonzero(mask & (depth_m > 0))
    if len(us) < MIN_BOARD_PIXELS:
        return f'only {len(us)} valid depth pixels on the board'

    # Rays through each pixel (in color-image pixel coordinates), then the PnP
    # plane's depth along each ray: z = (n . t) / (n . ray) for a ray with z=1.
    pix = np.column_stack(((us + 0.5) / sx - 0.5, (vs + 0.5) / sy - 0.5)).reshape(-1, 1, 2)
    rays = np.column_stack((cv2.undistortPoints(pix, K, D).reshape(-1, 2), np.ones(len(us))))
    n = det.R[:, 2]
    z_pnp = (n @ det.t) / (rays @ n)
    z_depth = depth_m[vs, us]
    diff = z_depth - z_pnp

    pts = rays * z_depth[:, None]
    centroid = pts.mean(axis=0)
    n_depth = np.linalg.svd(pts - centroid, full_matrices=False)[2][-1]
    tilt = np.degrees(np.arccos(min(1.0, abs(float(n_depth @ n)))))

    med = float(np.median(diff))
    return dict(range_m=float(np.median(z_pnp)), center_m=float((n @ det.t) / n[2]), bias_mm=med * 1000,
                bias_pct=100 * float(np.median(diff / z_pnp)),
                noise_mm=1.4826 * float(np.median(np.abs(diff - med))) * 1000,
                tilt_deg=tilt, pixels=len(us))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--camera', required=True, choices=sorted(CAMERA_LINK_FRAMES))
    p.add_argument('--frames', type=int, default=30, help='Depth frames to average over (default 30).')
    p.add_argument('--timeout', type=float, default=20.0)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    board_yaml = BOARD_YAML_OVERRIDE or os.path.normpath(
        os.path.join(here, '..', 'calibrations', f'{BOARD_TYPE}_board.yaml'))
    board = _load_board(BOARD_TYPE, board_yaml, SQUARE_LENGTH_M_OVERRIDE, MARKER_LENGTH_M_OVERRIDE,
                        MARKER_SEPARATION_M_OVERRIDE)
    if board.kind != 'charuco':
        print('Only the ChArUco board is supported.')
        return 1

    rclpy.init()
    node = DepthPnpNode(args.camera)
    results, skipped = [], {}
    deadline = time.time() + args.timeout
    try:
        while rclpy.ok() and len(results) < args.frames and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            while node.depth_frames:
                gray, depth_m = node.depth_frames.pop(0)
                K = np.array(node.info.k).reshape(3, 3)
                D = np.array(node.info.d)
                r = _measure(gray, depth_m, board, K, D, DEFAULT_MIN_CHARUCO_CORNERS)
                if isinstance(r, str):
                    skipped[r] = skipped.get(r, 0) + 1
                else:
                    results.append(r)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not results:
        print(f'[{args.camera}] no usable frames in {args.timeout:.0f}s.')
        if node.info is None:
            print(f'  Nothing received on /{args.camera}/color/camera_info -- is the camera up?')
        elif not skipped:
            print(f'  No aligned depth received -- is depth on (not the calibration:=true profile)?')
        for why, count in skipped.items():
            print(f'  {count} frame(s): {why}')
        return 1

    col = {k: np.array([r[k] for r in results]) for k in results[0]}
    bias, pct, tilt = np.median(col['bias_mm']), np.median(col['bias_pct']), np.median(col['tilt_deg'])
    print(f'[{args.camera}] {len(results)} frames, board at {np.median(col["range_m"]):.3f} m, '
          f'~{int(np.median(col["pixels"]))} depth px on the board per frame'
          + (f' ({sum(skipped.values())} frames skipped)' if skipped else ''))
    print(f'  depth - PnP bias : {bias:+.1f} mm  ({pct:+.2f} % of range)   '
          f'frame-to-frame std {np.std(col["bias_mm"]):.1f} mm')
    print(f'  depth plane tilt : {tilt:.2f} deg vs PnP plane')
    print(f'  depth noise      : {np.median(col["noise_mm"]):.1f} mm (robust std on the flat board)')
    print(f'  Tare ground truth: {np.median(col["center_m"]) * 1000:.1f} mm (board plane along the image-center ray;'
          ' valid only if the board covers the image center)')
    ok_bias = abs(bias) < BIAS_OK_MM and abs(pct) < BIAS_OK_PCT
    ok_tilt = tilt < TILT_OK_DEG
    if ok_bias and ok_tilt:
        print('  -> depth agrees with PnP. Any error you see in base_link is the extrinsic, not depth.')
    else:
        if not ok_bias:
            print(f'  -> depth reads {"FAR" if bias > 0 else "NEAR"} by {abs(bias):.1f} mm. Repeat at another '
                  'distance: bias proportional to range = scale error, constant = offset. '
                  'RealSense Viewer: On-Chip Calibration, then Tare Calibration.')
        if not ok_tilt:
            print('  -> depth plane is tilted relative to PnP: On-Chip Calibration in the RealSense Viewer.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
