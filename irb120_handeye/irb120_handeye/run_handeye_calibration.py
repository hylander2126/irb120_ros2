#!/usr/bin/env python3
"""Headless, dual-camera eye-to-hand calibration — no MoveIt HandEyeCalibration
panel, no per-camera manual runs.

Drives the arm through one shared set of joint poses (reusing the same
FollowJointTrajectory motion as run_calibration_poses.py) and, at every
settled pose, automatically detects the target board in whichever cameras
currently see it — there is no requirement that a pose be visible to both
cameras at once. Each camera accumulates its own independent sample set and
gets its own solve, so a pose set tailored to one camera's FOV (e.g. cam2's
steep overhead view) costs nothing for the other.

--board-type defaults to 'aruco': the existing grid board (irb_target_image.png)
already printed and mounted, used as a stopgap until a printer and a rigid
flat backing are available to produce+mount the ChArUco board this pipeline
otherwise prefers (see generate_charuco_target.py and "Future work" in the
README). Pass --board-type charuco once that board exists.

Solves per camera with cv2.calibrateHandEye() in ITS eye-to-hand mode: feed
gripper<-base (inverted from the natural TF direction) instead of the usual
gripper2base, leave target<-camera as OpenCV's PnP gives it, and the
returned "cam2gripper" output IS base_link -> camera directly (this is
OpenCV's own documented recipe for a static camera + gripper-mounted
target, not a manual before/after inversion trick layered on top).

Usage:
  ros2 run irb120_handeye run_handeye_calibration
  ros2 run irb120_handeye run_handeye_calibration --cameras realsense2 --samples-per-pose 5

Prerequisites: same bringup as run_calibration_poses.py (abb_control +
robot_state_publisher for base_link/tool0 TF) PLUS both realsense drivers
running (bringup_cam1.launch.py / bringup_cam2.launch.py, or
bringup_handeye.launch.py) so image_raw/camera_info/link->optical TF are
all live. The MoveIt/RViz stack is NOT needed for this script.
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, LookupException, TransformListener

from irb120_handeye.run_calibration_poses import (
    HandeyePoseRunner,
    _load_pose_yaml,
    _resolve_pose_path,
)

# TSAI and DANIILIDIS are the textbook-standard choices (DANIILIDIS is what
# cam1's existing MoveIt-panel calibration used) but on this OpenCV build
# (4.6.0) both are numerically unreliable: fed perfect noiseless synthetic
# ground truth, TSAI came back up to 144 degrees off and DANIILIDIS up to
# 7 degrees / 40mm off, repeatably, across many random trials. PARK, HORAUD
# and ANDREFF all recovered ground truth to machine precision every time.
# Default to PARK; the other two are kept selectable but --method tsai/
# daniilidis should not be trusted without re-verifying against this build.
METHODS = {
    'tsai': cv2.CALIB_HAND_EYE_TSAI,
    'park': cv2.CALIB_HAND_EYE_PARK,
    'horaud': cv2.CALIB_HAND_EYE_HORAUD,
    'andreff': cv2.CALIB_HAND_EYE_ANDREFF,
    'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
}

# camera_name (rs_launch.py) -> the link frame its eye-to-hand transform is
# published against. Matches bringup_cam1.launch.py / bringup_cam2.launch.py.
# The optical frame each camera's images/PnP are actually expressed in is
# derived from this as f'{name}_color_optical_frame' (realsense2_camera's
# own naming convention when camera_name is overridden).
CAMERA_LINK_FRAMES = {'realsense': 'realsense_link', 'realsense2': 'realsense2_link'}

BASE_FRAME = 'base_link'
GRIPPER_FRAME = 'tool0'


# --- SE(3) helpers -----------------------------------------------------

def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Same formula as irb120_control/util/press_point_check.py's _to_frame."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Shepperd's method; returns [x, y, z, w]."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw, qx, qy, qz = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw, qx, qy, qz = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw, qx, qy, qz = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([qx, qy, qz, qw])


def _transform_to_Rt(t: TransformStamped) -> Tuple[np.ndarray, np.ndarray]:
    q, p = t.transform.rotation, t.transform.translation
    return _quat_to_matrix(q.x, q.y, q.z, q.w), np.array([p.x, p.y, p.z])


def _inv(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Rt = R.T
    return Rt, -Rt @ t


def _compose(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """p' = R1 (R2 p + t2) + t1 -- i.e. apply (R2,t2) then (R1,t1)."""
    return R1 @ R2, R1 @ t2 + t1


# --- Target board (ChArUco, or the ArUco grid board as a stopgap) -----
#
# FUTURE WORK: --board-type defaults to 'aruco' right now only because
# there's no printer/rigid backing on hand yet to produce and mount the
# ChArUco board this pipeline otherwise prefers (see generate_charuco_target.py
# and this package's README "Future work" note). A GridBoard's pose comes
# from marker corners alone, with no checkerboard-corner refinement, so it's
# noisier and less occlusion-tolerant than ChArUco. Switch the default back
# to 'charuco' once a ChArUco board is printed and mounted.

class BoardSpec:
    def __init__(self, kind: str, board, dictionary):
        self.kind = kind  # 'charuco' or 'aruco'
        self.board = board
        self.dictionary = dictionary


def _load_board(board_type: str, board_yaml_path: str, square_length_m: Optional[float],
                marker_length_m: Optional[float], marker_separation_m: Optional[float]) -> BoardSpec:
    with open(board_yaml_path, 'r', encoding='utf-8') as f:
        spec = yaml.safe_load(f)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec['dictionary']))

    if board_type == 'charuco':
        square = square_length_m if square_length_m is not None else spec['square_length_m']
        marker = marker_length_m if marker_length_m is not None else spec['marker_length_m']
        if square_length_m is not None:
            print(f'Using measured square length {square * 1000:.2f}mm '
                  f'(design value was {spec["square_length_m"] * 1000:.2f}mm)')
        board = cv2.aruco.CharucoBoard_create(spec['squares_x'], spec['squares_y'], square, marker, dictionary)
    else:
        marker = marker_length_m if marker_length_m is not None else spec['marker_length_m']
        separation = marker_separation_m if marker_separation_m is not None else spec['marker_separation_m']
        if marker_length_m is not None:
            print(f'Using measured marker length {marker * 1000:.2f}mm '
                  f'(design value was {spec["marker_length_m"] * 1000:.2f}mm)')
        board = cv2.aruco.GridBoard_create(spec['markers_x'], spec['markers_y'], marker, separation, dictionary)
    return BoardSpec(board_type, board, dictionary)


def _detect_board_pose(gray: np.ndarray, board_spec: BoardSpec, camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray, min_features: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    corners, ids, _ = cv2.aruco.detectMarkers(gray, board_spec.dictionary)
    if ids is None or len(ids) == 0:
        return None

    if board_spec.kind == 'charuco':
        count, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board_spec.board, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
        if count < min_features:
            return None
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, board_spec.board, camera_matrix, dist_coeffs, None, None)
    else:
        if len(ids) < min_features:
            return None
        n_used, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners, ids, board_spec.board, camera_matrix, dist_coeffs, None, None)
        ok = n_used > 0
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


# --- Camera + TF node ------------------------------------------------------

class HandEyeCameraNode(Node):
    """Caches the latest frame/intrinsics per camera namespace, plus TF."""

    def __init__(self, namespaces: List[str]):
        super().__init__('handeye_camera_node')
        self.bridge = CvBridge()
        self._images: Dict[str, np.ndarray] = {}
        self._info: Dict[str, CameraInfo] = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        for ns in namespaces:
            self.create_subscription(
                Image, f'/{ns}/color/image_raw',
                lambda msg, ns=ns: self._on_image(ns, msg), qos_profile_sensor_data)
            self.create_subscription(
                CameraInfo, f'/{ns}/color/camera_info',
                lambda msg, ns=ns: self._info.setdefault(ns, msg), qos_profile_sensor_data)

    def _on_image(self, ns: str, msg: Image) -> None:
        self._images[ns] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

    def latest_image(self, ns: str) -> Optional[np.ndarray]:
        return self._images.get(ns)

    def intrinsics(self, ns: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        info = self._info.get(ns)
        if info is None:
            return None
        return np.array(info.k).reshape(3, 3), np.array(info.d)

    def spin_for(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for(self, predicate, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return predicate()

    def lookup(self, target_frame: str, source_frame: str) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        except LookupException:
            return None


# --- Calibration solve + residual check -------------------------------

def _pairwise_axxb_residual(R_g2b: List[np.ndarray], t_g2b: List[np.ndarray],
                            R_t2c: List[np.ndarray], t_t2c: List[np.ndarray],
                            R_x: np.ndarray, t_x: np.ndarray,
                            max_pairs: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Residual of (g2b_j)^-1 (g2b_i) X == X (t2c_j)(t2c_i)^-1 over pose pairs.

    Not the same quantity as the MoveIt panel's reprojection error (that's a
    pixel/mm reprojection metric; this is the algebraic AX=XB consistency of
    the solved calibration), but it's a real, independently meaningful
    measure of how well a single rigid X explains every pose pair.
    """
    n = len(R_g2b)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(0)
        pairs = [pairs[k] for k in rng.choice(len(pairs), size=max_pairs, replace=False)]
    rot_deg, trans_mm = [], []
    for i, j in pairs:
        Rgb_j_inv, tgb_j_inv = _inv(R_g2b[j], t_g2b[j])
        R_A, t_A = _compose(Rgb_j_inv, tgb_j_inv, R_g2b[i], t_g2b[i])
        Rtc_i_inv, ttc_i_inv = _inv(R_t2c[i], t_t2c[i])
        R_B, t_B = _compose(R_t2c[j], t_t2c[j], Rtc_i_inv, ttc_i_inv)
        R_lhs, t_lhs = _compose(R_A, t_A, R_x, t_x)
        R_rhs, t_rhs = _compose(R_x, t_x, R_B, t_B)
        R_err = R_lhs.T @ R_rhs
        rot_deg.append(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))))
        trans_mm.append(np.linalg.norm(t_lhs - t_rhs) * 1000)
    return np.array(rot_deg), np.array(trans_mm)


def _write_launch_file(out_dir: str, ns: str, link_frame: str, xyz: np.ndarray, quat: np.ndarray,
                       residual_mm: float, n_samples: int, method_name: str) -> str:
    path = os.path.join(out_dir, f'cam_tf_{ns}_{residual_mm:.0f}mm.launch.py')
    content = f'''"""Static transform acquired via run_handeye_calibration.py (headless).

EYE-TO-HAND: base_link -> {link_frame}. Solved with cv2.calibrateHandEye
(method={method_name}), {n_samples} samples, mean pairwise AX=XB translation
residual ~{residual_mm:.1f}mm (algebraic solve-consistency metric, not the
MoveIt panel's pixel reprojection error -- see run_handeye_calibration.py).

Not wired into bringup automatically. Point the matching
bringup_camN.launch.py's cameraN_tf.launch.py include at this file (or copy
the values in) once you trust the result, and keep the previous
cam_tf_*.launch.py around as a fallback per this package's README.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id", "base_link",
                "--child-frame-id", "{link_frame}",
                "--x", "{xyz[0]:.6f}",
                "--y", "{xyz[1]:.6f}",
                "--z", "{xyz[2]:.6f}",
                "--qx", "{quat[0]:.6f}",
                "--qy", "{quat[1]:.6f}",
                "--qz", "{quat[2]:.6f}",
                "--qw", "{quat[3]:.6f}",
            ],
        ),
    ]
    return LaunchDescription(nodes)
'''
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


# --- Main --------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pose-file', default='joints_5_6mm.yaml',
                   help='Pose YAML under share/irb120_handeye/calibrations/. NOTE: this set was '
                        'tuned for cam1s FOV -- design a combined/cam2-aware set before trusting '
                        'cam2s result (see package README).')
    p.add_argument('--pose-path', default=None)
    p.add_argument('--cameras', default='realsense,realsense2',
                   help='Comma-separated camera_name values (rs_launch.py), e.g. "realsense2" alone.')
    p.add_argument('--move-time', type=float, default=3.0, help='Minimum seconds per move.')
    p.add_argument('--max-joint-speed', type=float, default=0.5,
                   help='rad/s cap: each move takes at least (largest joint delta / this), so long '
                        'moves are automatically slowed down. Moves are straight joint-space '
                        'interpolation with NO collision checking.')
    p.add_argument('--step', action='store_true',
                   help='Print the planned joint move and wait for Enter before every pose '
                        '(Ctrl+C to abort). Use for first runs of a new pose set.')
    p.add_argument('--dry-run', action='store_true',
                   help='Move and report per-camera detections only; skip solving and file output.')
    p.add_argument('--settle-time', type=float, default=3.0)
    p.add_argument('--max-spread-mm', type=float, default=3.0,
                   help='Reject a pose for a camera if repeat captures of the board differ by more than this.')
    p.add_argument('--samples-per-pose', type=int, default=5,
                   help='Detections captured per pose per camera; the one closest to the '
                        'per-pose median translation is kept (cheap jitter rejection).')
    p.add_argument('--sample-delay', type=float, default=0.2, help='Seconds between repeat captures.')
    p.add_argument('--board-type', choices=['charuco', 'aruco'], default='aruco',
                   help='aruco (default, TEMPORARY): the grid board already printed and mounted -- '
                        'see "Future work" in the README. Switch to charuco once that board is '
                        'printed on a rigid backing (generate_charuco_target.py); its pose estimate '
                        'is more accurate and more occlusion-tolerant.')
    p.add_argument('--min-features', type=int, default=None,
                   help='Minimum interpolated ChArUco corners / detected ArUco markers to accept a '
                        'detection. Defaults to 6 for charuco, 3 for aruco (of the boards 12 markers).')
    p.add_argument('--min-samples', type=int, default=8,
                   help='Warn (not abort) if a camera ends up with fewer accepted poses than this; '
                        'calibrateHandEye itself needs >= 3.')
    p.add_argument('--board-yaml', default=None,
                   help='Defaults to calibrations/{charuco,aruco}_board.yaml next to this package, '
                        'matching --board-type.')
    p.add_argument('--square-length-m', type=float, default=None,
                   help='charuco only: override the boards square length with a caliper-measured value.')
    p.add_argument('--marker-length-m', type=float, default=None,
                   help='Override the boards marker length with a caliper-measured value.')
    p.add_argument('--marker-separation-m', type=float, default=None,
                   help='aruco only: override the boards marker separation with a caliper-measured value.')
    p.add_argument('--method', choices=sorted(METHODS), default='park',
                   help='park (default): verified to recover ground truth to machine precision '
                        'in this OpenCV build. tsai/daniilidis are unreliable here -- see METHODS '
                        'comment in this file -- and should not be used without re-verifying.')
    p.add_argument('--out-dir', default=None, help='Defaults to your home directory.')
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    board_yaml = args.board_yaml or os.path.normpath(
        os.path.join(here, '..', 'calibrations', f'{args.board_type}_board.yaml'))
    min_features = args.min_features if args.min_features is not None else (6 if args.board_type == 'charuco' else 3)
    out_dir = args.out_dir or os.path.expanduser('~')
    cameras = [c.strip() for c in args.cameras.split(',') if c.strip()]
    for ns in cameras:
        if ns not in CAMERA_LINK_FRAMES:
            print(f'Unknown camera {ns!r}; known cameras: {sorted(CAMERA_LINK_FRAMES)}')
            return 1

    if not os.path.isfile(board_yaml):
        print(f'Board spec not found: {board_yaml}.'
              + (' Run generate_charuco_target first.' if args.board_type == 'charuco' else ''))
        return 1
    board = _load_board(args.board_type, board_yaml, args.square_length_m, args.marker_length_m,
                        args.marker_separation_m)
    print(f'Board: {args.board_type} ({board_yaml})')

    pose_path = _resolve_pose_path(args.pose_path, args.pose_file)
    if not os.path.isfile(pose_path):
        print(f'Pose file not found: {pose_path}')
        return 1
    joint_names, joint_values = _load_pose_yaml(pose_path)

    rclpy.init()
    pose_runner = HandeyePoseRunner(joint_names)
    cam_node = HandEyeCameraNode(cameras)

    try:
        print(f'Loaded {len(joint_values)} poses from {pose_path}')
        if not pose_runner.wait_for_joint_states(timeout_sec=10.0):
            print('No joint states received -- is abb_control bringup running?')
            return 2

        active = []
        for ns in cameras:
            link = CAMERA_LINK_FRAMES[ns]
            optical = f'{ns}_color_optical_frame'
            if not cam_node.wait_for(lambda ns=ns: cam_node.intrinsics(ns) is not None, 10.0):
                print(f'ERROR [{ns}]: no camera_info within 10s -- is its driver running? Not moving the robot.')
                return 3
            if not cam_node.wait_for(lambda: cam_node.lookup(link, optical) is not None, 10.0):
                print(f'ERROR [{ns}]: no TF {link} <- {optical} within 10s. Not moving the robot.')
                return 3
            link_opt_tf = cam_node.lookup(link, optical)
            R_link_opt, t_link_opt = _transform_to_Rt(link_opt_tf)
            active.append((ns, link, optical, R_link_opt, t_link_opt))

        samples: Dict[str, Dict[str, List[np.ndarray]]] = {
            ns: {'R_g2b': [], 't_g2b': [], 'R_t2c': [], 't_t2c': []} for ns, *_ in active
        }
        missed: Dict[str, int] = {ns: 0 for ns, *_ in active}

        total = len(joint_values)
        for i, target in enumerate(joint_values, start=1):
            current = pose_runner._current_positions()
            max_delta = max(abs(a - b) for a, b in zip(target, current))
            move_time = max(0.5, args.move_time, max_delta / max(args.max_joint_speed, 1e-3))
            print(f'Pose {i}/{total}: largest joint delta {np.degrees(max_delta):.1f} deg, '
                  f'move time {move_time:.1f}s')
            if args.step:
                print('  joint deltas (deg): '
                      + ', '.join(f'{np.degrees(b - a):+.1f}' for a, b in zip(current, target)))
                input('  Press Enter to move (Ctrl+C to abort)...')
            if not pose_runner.move_to(target, move_time_sec=move_time):
                print(f'  move failed, skipping pose {i}')
                continue
            cam_node.spin_for(max(0.0, args.settle_time))

            g2b_tf = cam_node.lookup(GRIPPER_FRAME, BASE_FRAME)  # deliberately inverted: see module docstring
            if g2b_tf is None:
                print(f'  no TF {GRIPPER_FRAME} <- {BASE_FRAME} at pose {i}, skipping')
                continue
            R_g2b, t_g2b = _transform_to_Rt(g2b_tf)

            for ns, link, optical, R_link_opt, t_link_opt in active:
                hits = []
                for _ in range(max(1, args.samples_per_pose)):
                    cam_node.spin_for(args.sample_delay)
                    gray = cam_node.latest_image(ns)
                    K, D = cam_node.intrinsics(ns)
                    if gray is None:
                        continue
                    det = _detect_board_pose(gray, board, K, D, min_features)
                    if det is not None:
                        hits.append(det)
                if not hits:
                    print(f'  [{ns}] no detection at pose {i}')
                    missed[ns] += 1
                    continue
                tvecs = np.array([t for _, t in hits])
                spread_mm = float(np.linalg.norm(tvecs - tvecs.mean(axis=0), axis=1).max() * 1000)
                if spread_mm > args.max_spread_mm:
                    print(f'  [{ns}] pose {i} rejected: board position spread {spread_mm:.1f}mm across '
                          f'{len(hits)} captures (still settling, or noisy detection)')
                    missed[ns] += 1
                    continue
                median_t = np.median(tvecs, axis=0)
                best = hits[int(np.argmin(np.linalg.norm(tvecs - median_t, axis=1)))]
                R_opt_target, t_opt_target = best
                # Compose into the *_link frame so the solve (and its output)
                # lands directly in the same frame the existing cam_tf launch
                # files publish, without a second composition step after solving.
                R_link_target, t_link_target = _compose(R_link_opt, t_link_opt, R_opt_target, t_opt_target)
                samples[ns]['R_g2b'].append(R_g2b)
                samples[ns]['t_g2b'].append(t_g2b)
                samples[ns]['R_t2c'].append(R_link_target)
                samples[ns]['t_t2c'].append(t_link_target)
                print(f'  [{ns}] sample {len(samples[ns]["R_g2b"])} captured (spread {spread_mm:.1f}mm)')

        if args.dry_run:
            print('\nDry run summary (accepted detections per camera):')
            for ns, *_ in active:
                print(f'  [{ns}] {len(samples[ns]["R_g2b"])}/{total} poses, {missed[ns]} missed')
            return 0

        print('\nSolving...')
        os.makedirs(out_dir, exist_ok=True)
        for ns, link, optical, *_ in active:
            s = samples[ns]
            n = len(s['R_g2b'])
            print(f'\n[{ns}] {n} accepted poses, {missed[ns]} missed')
            if n < 3:
                print(f'  not enough samples to solve (need >= 3) -- check framing/board visibility for this camera')
                continue
            if n < args.min_samples:
                print(f'  WARNING: only {n} samples (< --min-samples {args.min_samples}); '
                      'solve will run but is likely poorly conditioned -- add more poses for this camera')

            R_sol, t_sol = cv2.calibrateHandEye(
                s['R_g2b'], s['t_g2b'], s['R_t2c'], s['t_t2c'], method=METHODS[args.method])
            t_sol = t_sol.reshape(3)
            quat = _matrix_to_quat(R_sol)

            rot_res, trans_res = _pairwise_axxb_residual(s['R_g2b'], s['t_g2b'], s['R_t2c'], s['t_t2c'], R_sol, t_sol)
            print(f'  base_link -> {link}:')
            print(f'    xyz = [{t_sol[0]:.6f}, {t_sol[1]:.6f}, {t_sol[2]:.6f}]')
            print(f'    quat(xyzw) = [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]')
            print(f'    AX=XB residual over {len(rot_res)} pose pairs: '
                  f'rotation mean={rot_res.mean():.3f} deg max={rot_res.max():.3f} deg, '
                  f'translation mean={trans_res.mean():.2f} mm max={trans_res.max():.2f} mm')

            path = _write_launch_file(out_dir, ns, link, t_sol, quat, float(trans_res.mean()), n, args.method)
            print(f'  wrote {path}')

        return 0
    except KeyboardInterrupt:
        print('Interrupted by user.')
        return 130
    finally:
        pose_runner.destroy_node()
        cam_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
