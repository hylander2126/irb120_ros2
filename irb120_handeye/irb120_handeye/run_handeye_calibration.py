#!/usr/bin/env python3
"""Headless, multi-camera eye-to-hand calibration -- no MoveIt HandEyeCalibration
panel, no manual "Take Sample" clicks.

Drives the arm through a shared pose set and, at each settled pose, detects
the target board in whichever requested camera(s) currently see it. A pose
taught (via record_calibration_pose.py) as not seen by any requested camera
is skipped -- no move. Each camera gets its own independent sample set and
its own cv2.calibrateHandEye() solve, in its documented eye-to-hand mode.

Every move is planned and executed by move_group, so it is collision-checked
against the robot model (table and bracket included). Before any motion, the
whole pose sequence is planned once as a check; --plan-only stops after it.

Per-run knobs live in --cameras/--pose-file/--step/--dry-run/--plan-only below; anything
else (motion speed, sampling, board type/measurements, solver method, output
dir -- see the "Tunables" block) is fixed for this rig and meant to be edited
here directly rather than passed as a flag.

Usage:
  ros2 run irb120_handeye run_handeye_calibration
  ros2 run irb120_handeye run_handeye_calibration --cameras realsense3

Prerequisites: abb_control bringup (base_link/tool0 TF), bringup_stack with
calibration:=true (move_group with the ChArUco-mount model, and the cameras).
"""

import argparse
import os
import select
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.optimize import least_squares
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from tf2_ros import Buffer, LookupException, TransformListener

from irb120_handeye.run_calibration_poses import (
    HandeyePoseRunner,
    _load_pose_yaml,
    _resolve_pose_path,
    trajectory_duration,
)

# PARK is the only solver verified reliable on this OpenCV build (4.6.0) --
# fed noiseless synthetic ground truth, TSAI/DANIILIDIS came back tens of
# degrees off, repeatably; PARK/HORAUD/ANDREFF recovered it to machine
# precision. Kept as a dict (rather than just calling cv2.CALIB_HAND_EYE_PARK
# directly) so METHOD below can still be swapped to one of these if re-verified.
METHODS = {
    'tsai': cv2.CALIB_HAND_EYE_TSAI,
    'park': cv2.CALIB_HAND_EYE_PARK,
    'horaud': cv2.CALIB_HAND_EYE_HORAUD,
    'andreff': cv2.CALIB_HAND_EYE_ANDREFF,
    'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
}

# camera_name (rs_launch.py) -> its eye-to-hand link frame (bringup_cam1/2/3.launch.py).
# Optical frame is derived as f'{name}_color_optical_frame'.
CAMERA_LINK_FRAMES = {'realsense': 'realsense_link', 'realsense2': 'realsense2_link', 'realsense3': 'realsense3_link'}

BASE_FRAME = 'base_link'
GRIPPER_FRAME = 'tool0'

# --- Tunables for this rig ----------------------------------------------
# Fixed for this setup; edit here rather than passing flags. Only
# --cameras/--pose-file/--step/--dry-run vary run-to-run (see _build_arg_parser).
VELOCITY_SCALING = 0.15        # MoveIt scaling of joint_limits.yaml: joints 1-3 ~0.65 rad/s, joint 6 ~1.1 rad/s,
ACCELERATION_SCALING = 0.15    # all under egm_handler's max_speed_dev_rad (1.5 rad/s)
SETTLE_TIME_SEC = 1.0          # wait after each move before sampling; MAX_SPREAD_MM rejects a pose still moving
SAMPLES_PER_POSE = 3           # repeat detections per pose per camera; median-closest kept (jitter rejection)
SAMPLE_DELAY_SEC = 0.1         # seconds between repeat captures
EGM_WINDOW_SEC = 180.0         # RobotWare ends the EGM session this long after it starts (egm_handler cond_time)
EGM_MARGIN_SEC = 10.0          # pause for an EGM restart unless the next move ends this long before the window does
EGM_READY_TOPICS = ('/egm_handler_startup/ready', '/egm_handler/ready')  # egm_handler in bringup_stack / ros2 run
MAX_SPREAD_MM = 3.0            # reject a pose/camera if repeat captures disagree by more than this
MIN_SAMPLES_WARN = 8           # warn (not abort) if a camera ends up with fewer accepted poses than this
BOARD_TYPE = 'charuco'         # physical ChArUco target; dimensions live in calibrations/charuco_board.yaml
BOARD_YAML_OVERRIDE = None     # None -> calibrations/{BOARD_TYPE}_board.yaml next to this package
SQUARE_LENGTH_M_OVERRIDE = None    # charuco only: caliper-measured square length, if remeasured
MARKER_LENGTH_M_OVERRIDE = None    # caliper-measured marker length, if remeasured
MARKER_SEPARATION_M_OVERRIDE = None  # aruco only: caliper-measured marker separation, if remeasured
MIN_FEATURES = None            # None -> DEFAULT_MIN_CHARUCO_CORNERS (3 ArUco markers for legacy grid-board runs)
DEFAULT_MIN_CHARUCO_CORNERS = 12   # partial board views are fine, but too few corners gives a weak pose
MIN_CORNER_SPREAD_SQUARES = 0.6    # charuco only: detected corners' std. dev. across the board's narrower
                                   # direction, in squares. Rejects near-collinear sets (a single row or a
                                   # diagonal line), whose pose is poorly constrained; 2 rows ~0.5, 3x3 ~0.82.
METHOD = 'park'                # see METHODS comment above; seeds the reprojection refinement
OUTLIER_RATIO = 3.0            # flag a pose whose reprojection RMS is > this x the set's median...
OUTLIER_MIN_PX = 2.0           # ...and above this floor (robot/intrinsics error alone gives ~0.5-1.5px)
OUT_DIR = os.path.expanduser('~')


# --- SE(3) helpers -----------------------------------------------------

def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Unit quaternion (x, y, z, w) -> 3x3 rotation matrix."""
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
# See BOARD_TYPE above re: which one's in use and why.

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


def _load_pose_detections(pose_path: str, n_poses: int) -> List[Optional[Dict[str, bool]]]:
    """The per-pose 'detections' list record_calibration_pose.py writes (which
    camera(s) saw the board when that pose was taught), or [None] * n_poses
    when the pose file predates that field / wasn't taught interactively
    (e.g. joints_5_6mm.yaml) or its length doesn't match joint_values."""
    with open(pose_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    detections = data.get('detections')
    if not detections or len(detections) != n_poses:
        return [None] * n_poses
    return detections


def _charuco_corner_spread(board, ch_ids: np.ndarray) -> float:
    """Detected corners' spread across the board's narrower direction, in squares.

    The smallest principal std. dev. of the corners' board-plane positions, so a
    straight line of corners scores ~0 whatever its direction on the board.
    """
    pts = board.chessboardCorners[ch_ids.reshape(-1), :2]
    square = board.chessboardCorners[1, 0] - board.chessboardCorners[0, 0]
    cov = np.cov((pts - pts.mean(axis=0)).T)
    return float(np.sqrt(max(np.linalg.eigvalsh(cov)[0], 0.0))) / square


class BoardDetection(NamedTuple):
    """Board pose in the camera optical frame, plus the corner correspondences
    it came from (needed for the reprojection-error solve)."""
    R: np.ndarray        # board -> optical rotation
    t: np.ndarray        # board origin in optical frame (m)
    obj_pts: np.ndarray  # (N, 3) corner positions in the board frame (m)
    img_pts: np.ndarray  # (N, 2) detected corner pixels


def _detect_board_pose(gray: np.ndarray, board_spec: BoardSpec, camera_matrix: np.ndarray,
                       dist_coeffs: np.ndarray, min_features: int) -> Optional[BoardDetection]:
    corners, ids, _ = cv2.aruco.detectMarkers(gray, board_spec.dictionary)
    if ids is None or len(ids) == 0:
        return None

    if board_spec.kind == 'charuco':
        count, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board_spec.board, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
        if count < min_features or _charuco_corner_spread(board_spec.board, ch_ids) < MIN_CORNER_SPREAD_SQUARES:
            return None
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, board_spec.board, camera_matrix, dist_coeffs, None, None)
        obj_pts = board_spec.board.chessboardCorners[ch_ids.reshape(-1)]
        img_pts = ch_corners.reshape(-1, 2)
    else:
        if len(ids) < min_features:
            return None
        n_used, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners, ids, board_spec.board, camera_matrix, dist_coeffs, None, None)
        ok = n_used > 0
        obj_pts, img_pts = cv2.aruco.getBoardObjectAndImagePoints(board_spec.board, corners, ids)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return BoardDetection(R, tvec.reshape(3), np.asarray(obj_pts, dtype=float).reshape(-1, 3),
                          np.asarray(img_pts, dtype=float).reshape(-1, 2))


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


# --- Calibration solve + reprojection-error diagnostic ----------------
#
# cv2.calibrateHandEye (METHOD) gives the camera pose X; the board-in-tool0
# pose Y then follows from each sample and is averaged. Both are refined
# together by minimizing the pixel error of every detected corner projected
# through base -> tool0 -> Y -> board and base -> X -> camera. Per-pose RMS
# reprojection error is the diagnostic: unlike pairwise AX=XB residuals, it
# doesn't grow with how far a pose is from the others, so an informative pose
# isn't penalized for being different -- only for disagreeing with the solve.

SAMPLE_KEYS = ('pose_idx', 'spread_mm', 'R_g2b', 't_g2b', 'R_t2c', 't_t2c',
               'obj_pts', 'img_pts', 'K', 'D', 'R_link_opt', 't_link_opt')


def _new_samples() -> Dict[str, List]:
    return {key: [] for key in SAMPLE_KEYS}


def _append_sample(s: Dict[str, List], pose_idx: int, spread_mm: Optional[float],
                   R_g2b: np.ndarray, t_g2b: np.ndarray, det: BoardDetection,
                   K: np.ndarray, D: np.ndarray, R_link_opt: np.ndarray, t_link_opt: np.ndarray) -> None:
    # Compose into the *_link frame so the solve (and its output) lands directly
    # in the frame the cam_tf launch files publish.
    R_link_target, t_link_target = _compose(R_link_opt, t_link_opt, det.R, det.t)
    for key, value in (('pose_idx', pose_idx), ('spread_mm', spread_mm), ('R_g2b', R_g2b), ('t_g2b', t_g2b),
                       ('R_t2c', R_link_target), ('t_t2c', t_link_target), ('obj_pts', det.obj_pts),
                       ('img_pts', det.img_pts), ('K', np.asarray(K, dtype=float)),
                       ('D', np.asarray(D, dtype=float)), ('R_link_opt', R_link_opt), ('t_link_opt', t_link_opt)):
        s[key].append(value)


class HandEyeSolution(NamedTuple):
    R_x: np.ndarray            # camera link in base_link (what the cam_tf launch file publishes)
    t_x: np.ndarray
    R_y: np.ndarray            # board in tool0
    t_y: np.ndarray
    pose_rms_px: np.ndarray    # per sample
    pose_max_px: np.ndarray
    pose_rms_mm: np.ndarray    # per sample, pixel error scaled to the board's distance
    rms_px: float              # over all corners
    rms_mm: float
    init_delta_mm: float       # refined vs. METHOD's closed-form camera pose
    init_delta_deg: float
    outliers: List[int]        # pose_idx values flagged (empty below 6 samples)


def _rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _mean_rotation(Rs: List[np.ndarray]) -> np.ndarray:
    U, _, Vt = np.linalg.svd(sum(Rs))
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return U @ Vt


def _board_in_optical(s: Dict[str, List], k: int, R_x: np.ndarray, t_x: np.ndarray,
                      R_y: np.ndarray, t_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R, t = _compose(*_inv(s['R_g2b'][k], s['t_g2b'][k]), R_y, t_y)   # board in base (g2b is base in tool0)
    R, t = _compose(*_inv(R_x, t_x), R, t)                            # board in camera link
    return _compose(*_inv(s['R_link_opt'][k], s['t_link_opt'][k]), R, t)


def _reprojection_errors(s: Dict[str, List], k: int, R_x, t_x, R_y, t_y) -> np.ndarray:
    R, t = _board_in_optical(s, k, R_x, t_x, R_y, t_y)
    px, _ = cv2.projectPoints(s['obj_pts'][k], cv2.Rodrigues(R)[0], t, s['K'][k], s['D'][k])
    return px.reshape(-1, 2) - s['img_pts'][k]


def _solve_handeye(s: Dict[str, List], method_key: str) -> HandEyeSolution:
    n = len(s['R_g2b'])
    R_x0, t_x0 = cv2.calibrateHandEye(s['R_g2b'], s['t_g2b'], s['R_t2c'], s['t_t2c'],
                                      method=METHODS[method_key])
    t_x0 = t_x0.reshape(3)
    # g2b is base in tool0, so board in tool0 = g2b * X * (board in camera link).
    Ys = [_compose(*_compose(s['R_g2b'][k], s['t_g2b'][k], R_x0, t_x0), s['R_t2c'][k], s['t_t2c'][k])
          for k in range(n)]
    R_y0, t_y0 = _mean_rotation([R for R, _ in Ys]), np.mean([t for _, t in Ys], axis=0)

    def unpack(x):
        return cv2.Rodrigues(x[0:3])[0], x[3:6], cv2.Rodrigues(x[6:9])[0], x[9:12]

    def residuals(x):
        R_x, t_x, R_y, t_y = unpack(x)
        return np.concatenate([_reprojection_errors(s, k, R_x, t_x, R_y, t_y).ravel() for k in range(n)])

    x0 = np.concatenate([cv2.Rodrigues(R_x0)[0].ravel(), t_x0, cv2.Rodrigues(R_y0)[0].ravel(), t_y0])
    # Huber: corners more than ~1px off get linear weight, so one bad pose can't drag the solve.
    result = least_squares(residuals, x0, loss='huber', f_scale=1.0, x_scale='jac')
    R_x, t_x, R_y, t_y = unpack(result.x)

    rms_px, max_px, rms_mm, sq_px, sq_mm, n_pts = [], [], [], 0.0, 0.0, 0
    for k in range(n):
        err = np.linalg.norm(_reprojection_errors(s, k, R_x, t_x, R_y, t_y), axis=1)
        depth = _board_in_optical(s, k, R_x, t_x, R_y, t_y)[1][2]
        mm_per_px = depth / s['K'][k][0, 0] * 1000
        rms_px.append(float(np.sqrt(np.mean(err ** 2))))
        max_px.append(float(err.max()))
        rms_mm.append(rms_px[-1] * mm_per_px)
        sq_px += float(np.sum(err ** 2))
        sq_mm += float(np.sum((err * mm_per_px) ** 2))
        n_pts += len(err)
    rms_px, max_px, rms_mm = np.array(rms_px), np.array(max_px), np.array(rms_mm)

    outliers = []
    if n >= 6:
        threshold = max(OUTLIER_RATIO * float(np.median(rms_px)), OUTLIER_MIN_PX)
        outliers = [s['pose_idx'][k] for k in range(n) if rms_px[k] > threshold]

    return HandEyeSolution(R_x, t_x, R_y, t_y, rms_px, max_px, rms_mm,
                           float(np.sqrt(sq_px / n_pts)), float(np.sqrt(sq_mm / n_pts)),
                           float(np.linalg.norm(t_x - t_x0) * 1000), _rot_angle_deg(R_x0.T @ R_x), outliers)


def _print_solution_summary(link: str, sol: HandEyeSolution, method_key: str) -> None:
    quat = _matrix_to_quat(sol.R_x)
    print(f'  base_link -> {link}:')
    print(f'    xyz = [{sol.t_x[0]:.6f}, {sol.t_x[1]:.6f}, {sol.t_x[2]:.6f}]')
    print(f'    quat(xyzw) = [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]')
    print(f'    reprojection RMS {sol.rms_px:.2f} px (~{sol.rms_mm:.2f} mm at the board); '
          f'refinement moved the {method_key} solution {sol.init_delta_mm:.1f} mm / {sol.init_delta_deg:.2f} deg')


def _print_reprojection_report(ns: str, sol: HandEyeSolution, s: Dict[str, List]) -> None:
    n = len(s['pose_idx'])
    print(f'\n  [{ns}] per-pose reprojection error over {n} poses '
          f'(median {np.median(sol.pose_rms_px):.2f} px):')
    print(f'    {"pose#":>6}  {"RMS":>8}  {"~at board":>10}  {"max":>8}  {"corners":>7}  {"capture spread":>15}')
    for k in range(n):
        spread = s['spread_mm'][k]
        spread_str = f'{spread:.2f} mm' if spread is not None else 'n/a'
        flag = '  <-- outlier' if s['pose_idx'][k] in sol.outliers else ''
        print(f'    {s["pose_idx"][k]:>6}  {sol.pose_rms_px[k]:>5.2f} px  {sol.pose_rms_mm[k]:>7.2f} mm  '
              f'{sol.pose_max_px[k]:>5.2f} px  {len(s["img_pts"][k]):>7}  {spread_str:>15}{flag}')
    if n < 6:
        print(f'  (outlier flagging needs >= 6 poses, have {n})')
    elif sol.outliers:
        pose_list = ','.join(str(p) for p in sol.outliers)
        print(f'  Pose(s) {pose_list} disagree with the rest (RMS > {OUTLIER_RATIO:g}x median and > '
              f'{OUTLIER_MIN_PX:g} px): misdetection, board moved, or arm not settled. Drop and re-solve '
              f'(no robot needed): diagnose_handeye_samples --exclude-poses "{pose_list}".')
    elif sol.rms_px > OUTLIER_MIN_PX:
        print(f'  No single pose stands out, but {sol.rms_px:.2f} px overall is high -- the error is systematic: '
              'check the board YAML against the real board (calipers), board flatness/rigidity, and intrinsics.')
    else:
        print('  No outlier poses.')


def _rt_to_dict(R: np.ndarray, t: np.ndarray) -> dict:
    q = _matrix_to_quat(R)
    return {'xyz': [float(v) for v in t], 'quat_xyzw': [float(v) for v in q]}


def _samples_from_dicts(rows: List[dict]) -> Dict[str, List]:
    if rows and 'img_pts' not in rows[0]:
        raise ValueError('samples file predates reprojection data (no corner pixels saved) -- '
                         're-run run_handeye_calibration to regenerate it')
    s = _new_samples()
    for row in rows:
        s['pose_idx'].append(row['pose_idx'])
        s['spread_mm'].append(row.get('spread_mm'))
        for key, name in (('g2b', 'g2b'), ('t2c', 't2c'), ('link_opt', 'link_opt')):
            qx, qy, qz, qw = row[name]['quat_xyzw']
            s[f'R_{key}'].append(_quat_to_matrix(qx, qy, qz, qw))
            s[f't_{key}'].append(np.array(row[name]['xyz'], dtype=float))
        s['obj_pts'].append(np.array(row['obj_pts'], dtype=float).reshape(-1, 3))
        s['img_pts'].append(np.array(row['img_pts'], dtype=float).reshape(-1, 2))
        s['K'].append(np.array(row['K'], dtype=float).reshape(3, 3))
        s['D'].append(np.array(row['D'], dtype=float))
    return s


def _write_samples_yaml(path: str, ns: str, link: str, board_type: str, method: str, s: Dict[str, List]) -> None:
    """Dump the raw per-pose samples (transforms + corner correspondences)
    behind a solve, so a run can be re-diagnosed or re-solved with poses
    excluded later without re-driving the robot. See diagnose_handeye_samples.py."""
    rows = [{
        'pose_idx': s['pose_idx'][k],
        'spread_mm': s['spread_mm'][k],
        'g2b': _rt_to_dict(s['R_g2b'][k], s['t_g2b'][k]),
        't2c': _rt_to_dict(s['R_t2c'][k], s['t_t2c'][k]),
        'link_opt': _rt_to_dict(s['R_link_opt'][k], s['t_link_opt'][k]),
        'K': s['K'][k].ravel().tolist(),
        'D': s['D'][k].ravel().tolist(),
        'obj_pts': s['obj_pts'][k].tolist(),
        'img_pts': s['img_pts'][k].tolist(),
    } for k in range(len(s['R_g2b']))]
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump({'camera': ns, 'link_frame': link, 'board_type': board_type, 'method': method,
                        'samples': rows}, f, sort_keys=False)


def _write_launch_file(out_dir: str, ns: str, link_frame: str, sol: HandEyeSolution,
                       n_samples: int, method_name: str) -> str:
    xyz, quat = sol.t_x, _matrix_to_quat(sol.R_x)
    path = os.path.join(out_dir, f'cam_tf_{ns}_{sol.rms_px:.1f}px.launch.py')
    content = f'''"""Static transform acquired via run_handeye_calibration.py (headless).

EYE-TO-HAND: base_link -> {link_frame}. Seeded with cv2.calibrateHandEye
(method={method_name}), then refined by minimizing board-corner reprojection
error; {n_samples} samples, RMS reprojection error {sol.rms_px:.2f} px
(~{sol.rms_mm:.2f} mm at the board).

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


# --- EGM session window ---------------------------------------------------

class EgmWindow:
    """Tracks time left in the current EGM session so moves never straddle its end.

    RobotWare stops EGM EGM_WINDOW_SEC after it starts, and a move cut off
    mid-trajectory leaves the arm stuck. Before each move, fits() says whether
    the move ends with EGM_MARGIN_SEC to spare; if not, wait_for_restart()
    parks at the current pose until the user re-enables EGM and presses Enter.
    """

    def __init__(self, node: Node):
        self._node = node
        self._started: Optional[float] = None
        self._ready_at: Optional[float] = None
        for topic in EGM_READY_TOPICS:
            node.create_subscription(Bool, topic, self._on_ready, 10)

    def _on_ready(self, msg: Bool) -> None:
        if msg.data:
            self._ready_at = time.monotonic()

    def remaining(self) -> float:
        return EGM_WINDOW_SEC - (time.monotonic() - self._started)

    def fits(self, move_time: float) -> bool:
        return self.remaining() >= move_time + EGM_MARGIN_SEC

    def wait_for_restart(self, reason: str) -> None:
        """Block (spinning ROS) until Enter. The window restarts from egm_handler's
        ready signal if one arrived while waiting, else from the keypress."""
        print(f'\n*** {reason}')
        print('*** Re-enable EGM now, then press Enter to continue (Ctrl+C to abort).')
        waiting_since = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if select.select([sys.stdin], [], [], 0)[0]:
                if not sys.stdin.readline():
                    raise KeyboardInterrupt  # stdin closed: no one can confirm the restart, so don't move
                break
        ready_seen = self._ready_at is not None and self._ready_at > waiting_since
        self._started = self._ready_at if ready_seen else time.monotonic()
        print(f'  EGM window restarted ({"from egm_handler ready signal" if ready_seen else "from Enter"}): '
              f'{self.remaining():.0f}s available')


def _plan_or_ask(pose_runner: HandeyePoseRunner, target: List[float], pose_no: int):
    """Plan to target from the current state; on failure ask whether to retry or skip.
    Never falls back to an unchecked move."""
    while True:
        trajectory, reason = pose_runner.plan(target, VELOCITY_SCALING, ACCELERATION_SCALING)
        if trajectory is not None:
            return trajectory
        answer = input(f'  Pose {pose_no}: {reason}. [r]etry planning, Enter to skip this pose, '
                       'Ctrl+C to abort: ').strip().lower()
        if answer != 'r':
            print(f'  skipping pose {pose_no} (robot not moved)')
            return None


def _check_sequence(pose_runner: HandeyePoseRunner, joint_values: List[List[float]], skip_idxs: set) -> List[int]:
    """Plan every move in order (each from the previous target) without moving.
    Returns the 1-based pose numbers MoveIt can't reach collision-free."""
    start = pose_runner._current_positions()
    failed, total_sec = [], 0.0
    for i, target in enumerate(joint_values, start=1):
        if (i - 1) in skip_idxs:
            continue
        trajectory, reason = pose_runner.plan(target, VELOCITY_SCALING, ACCELERATION_SCALING, start_positions=start)
        if trajectory is None:
            print(f'  pose {i}: {reason}')
            failed.append(i)
            continue
        total_sec += trajectory_duration(trajectory)
        start = target
    print(f'  {len(joint_values) - len(skip_idxs) - len(failed)} moves plan collision-free '
          f'(~{total_sec / 60:.1f} min of motion); {len(failed)} failed')
    return failed


# --- Main --------------------------------------------------------------

def _resolve_pose_source(pose: str) -> str:
    """--pose-file as either an existing path, or a filename under
    share/irb120_handeye/calibrations/ (the _resolve_pose_path convention
    shared with run_calibration_poses.py / record_calibration_pose.py)."""
    if os.path.isfile(pose):
        return os.path.abspath(pose)
    return _resolve_pose_path(None, pose)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pose-file', default='joints_5_6mm.yaml',
                   help='Pose YAML: an existing path (e.g. ~/joints_custom.yaml), or a filename under '
                        'share/irb120_handeye/calibrations/.')
    p.add_argument('--cameras', default=None,
                   help='Comma-separated camera_name values, e.g. "realsense3". Defaults to all of: '
                        + ', '.join(sorted(CAMERA_LINK_FRAMES)) + '. A pose taught '
                        '(record_calibration_pose.py) as not seen by any of these is skipped -- no move.')
    p.add_argument('--step', action='store_true',
                   help='Print the planned joint move and wait for Enter before every pose '
                        '(Ctrl+C to abort). Use for first runs of a new pose set.')
    p.add_argument('--dry-run', action='store_true',
                   help='Move and report per-camera detections only; skip solving and file output.')
    p.add_argument('--plan-only', action='store_true',
                   help='Plan every move with MoveIt (collision-checked) and report; do not move the robot.')
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    board_yaml = BOARD_YAML_OVERRIDE or os.path.normpath(
        os.path.join(here, '..', 'calibrations', f'{BOARD_TYPE}_board.yaml'))
    min_features = MIN_FEATURES if MIN_FEATURES is not None else (DEFAULT_MIN_CHARUCO_CORNERS if BOARD_TYPE == 'charuco' else 3)
    out_dir = OUT_DIR
    cameras = ([c.strip() for c in args.cameras.split(',') if c.strip()]
               if args.cameras else sorted(CAMERA_LINK_FRAMES))
    for ns in cameras:
        if ns not in CAMERA_LINK_FRAMES:
            print(f'Unknown camera {ns!r}; known cameras: {sorted(CAMERA_LINK_FRAMES)}')
            return 1

    if not os.path.isfile(board_yaml):
        print(f'Board spec not found: {board_yaml}.'
              + (' Run generate_charuco_target first.' if BOARD_TYPE == 'charuco' else ''))
        return 1
    board = _load_board(BOARD_TYPE, board_yaml, SQUARE_LENGTH_M_OVERRIDE, MARKER_LENGTH_M_OVERRIDE,
                        MARKER_SEPARATION_M_OVERRIDE)
    print(f'Board: {BOARD_TYPE} ({board_yaml})')

    pose_path = _resolve_pose_source(args.pose_file)
    if not os.path.isfile(pose_path):
        print(f'Pose file not found: {pose_path}')
        return 1
    joint_names, joint_values = _load_pose_yaml(pose_path)
    pose_detections = _load_pose_detections(pose_path, len(joint_values))
    skip_idxs = {idx for idx, det in enumerate(pose_detections)
                 if det is not None and not any(det.get(ns, False) for ns in cameras)}

    rclpy.init()
    pose_runner = HandeyePoseRunner(joint_names)
    cam_node = HandEyeCameraNode(cameras)

    try:
        print(f'Loaded {len(joint_values)} poses from {pose_path}')
        if skip_idxs:
            print(f'  {len(skip_idxs)}/{len(joint_values)} poses were not recorded as seen by any of '
                  f'{cameras} when taught -- skipping them (no move).')
        if not pose_runner.wait_for_joint_states(timeout_sec=10.0):
            print('No joint states received -- is abb_control bringup running?')
            return 2
        if not pose_runner.wait_for_move_group(timeout_sec=10.0):
            print('move_group not available -- is bringup_stack (calibration:=true) running?')
            return 2

        print('Planning every move with MoveIt before moving (collision check, no motion)...')
        unplannable = _check_sequence(pose_runner, joint_values, skip_idxs)
        if args.plan_only:
            return 0 if not unplannable else 4
        if unplannable:
            input(f'  Pose(s) {unplannable} could not be planned from the previous pose; at run time each is '
                  're-planned from where the arm actually is, and you will be asked if it fails again.\n'
                  '  Press Enter to start moving, Ctrl+C to abort...')

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

        samples: Dict[str, Dict[str, List]] = {ns: _new_samples() for ns, *_ in active}
        missed: Dict[str, int] = {ns: 0 for ns, *_ in active}

        egm = EgmWindow(cam_node)
        egm.wait_for_restart(f'Starting: the EGM window ({EGM_WINDOW_SEC:.0f}s) is timed from here.')

        total = len(joint_values)
        for i, target in enumerate(joint_values, start=1):
            if (i - 1) in skip_idxs:
                continue
            trajectory = _plan_or_ask(pose_runner, target, i)
            if trajectory is None:
                continue
            move_time = trajectory_duration(trajectory)
            print(f'Pose {i}/{total}: planned move {move_time:.1f}s')
            if args.step:
                current = pose_runner._current_positions()
                print('  joint deltas (deg): '
                      + ', '.join(f'{np.degrees(b - a):+.1f}' for a, b in zip(current, target)))
                input('  Press Enter to move (Ctrl+C to abort)...')
            if not egm.fits(move_time):
                egm.wait_for_restart(f'Pausing at the current pose: {egm.remaining():.0f}s left in the EGM '
                                     f'window, next move needs {move_time:.1f}s + {EGM_MARGIN_SEC:.0f}s margin.')
            if not pose_runner.execute(trajectory):
                # Most likely EGM ended early (window started before we were told). The arm may have
                # stopped mid-path, so re-plan from where it is after the restart.
                egm.wait_for_restart(f'Move to pose {i} failed -- EGM may have stopped.')
                trajectory = _plan_or_ask(pose_runner, target, i)
                if trajectory is None or not pose_runner.execute(trajectory):
                    print(f'  move failed again, skipping pose {i}')
                    continue
            cam_node.spin_for(max(0.0, SETTLE_TIME_SEC))

            g2b_tf = cam_node.lookup(GRIPPER_FRAME, BASE_FRAME)  # deliberately inverted: see module docstring
            if g2b_tf is None:
                print(f'  no TF {GRIPPER_FRAME} <- {BASE_FRAME} at pose {i}, skipping')
                continue
            R_g2b, t_g2b = _transform_to_Rt(g2b_tf)

            for ns, link, optical, R_link_opt, t_link_opt in active:
                hits = []
                for _ in range(max(1, SAMPLES_PER_POSE)):
                    cam_node.spin_for(SAMPLE_DELAY_SEC)
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
                tvecs = np.array([det.t for det in hits])
                spread_mm = float(np.linalg.norm(tvecs - tvecs.mean(axis=0), axis=1).max() * 1000)
                if spread_mm > MAX_SPREAD_MM:
                    print(f'  [{ns}] pose {i} rejected: board position spread {spread_mm:.1f}mm across '
                          f'{len(hits)} captures (still settling, or noisy detection)')
                    missed[ns] += 1
                    continue
                median_t = np.median(tvecs, axis=0)
                best = hits[int(np.argmin(np.linalg.norm(tvecs - median_t, axis=1)))]
                _append_sample(samples[ns], i, spread_mm, R_g2b, t_g2b, best, K, D, R_link_opt, t_link_opt)
                print(f'  [{ns}] sample {len(samples[ns]["R_g2b"])} captured (spread {spread_mm:.1f}mm)')

        if args.dry_run:
            attempted = total - len(skip_idxs)
            print(f'\nDry run summary (accepted detections per camera, {attempted}/{total} poses attempted '
                  f'-- {len(skip_idxs)} skipped as not recorded for these cameras):')
            for ns, *_ in active:
                print(f'  [{ns}] {len(samples[ns]["R_g2b"])}/{attempted} poses, {missed[ns]} missed')
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
            if n < MIN_SAMPLES_WARN:
                print(f'  WARNING: only {n} samples (< MIN_SAMPLES_WARN={MIN_SAMPLES_WARN}); '
                      'solve will run but is likely poorly conditioned -- add more poses for this camera')

            sol = _solve_handeye(s, METHOD)
            _print_solution_summary(link, sol, METHOD)
            _print_reprojection_report(ns, sol, s)

            path = _write_launch_file(out_dir, ns, link, sol, n, METHOD)
            print(f'  wrote {path}')
            samples_path = os.path.join(out_dir, f'handeye_samples_{ns}.yaml')
            _write_samples_yaml(samples_path, ns, link, BOARD_TYPE, METHOD, s)
            print(f'  wrote {samples_path} (raw per-pose samples -- re-diagnose or re-solve with poses '
                  f'excluded, no robot needed: ros2 run irb120_handeye diagnose_handeye_samples --in {samples_path})')

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
