#!/usr/bin/env python3
"""Headless, multi-camera eye-to-hand calibration -- no MoveIt HandEyeCalibration
panel, no manual "Take Sample" clicks.

Drives the arm through a shared pose set and, at each settled pose, detects
the target board in whichever requested camera(s) currently see it. A pose
taught (via record_calibration_pose.py) as not seen by any requested camera
is skipped -- no move. Each camera gets its own independent sample set and
its own cv2.calibrateHandEye() solve, in its documented eye-to-hand mode.

Per-run knobs live in --cameras/--pose-file/--step/--dry-run below; anything
else (motion speed, sampling, board type/measurements, solver method, output
dir -- see the "Tunables" block) is fixed for this rig and meant to be edited
here directly rather than passed as a flag.

Usage:
  ros2 run irb120_handeye run_handeye_calibration
  ros2 run irb120_handeye run_handeye_calibration --cameras realsense3

Prerequisites: abb_control bringup (base_link/tool0 TF) + the realsense
driver(s) for every camera being calibrated. MoveIt/RViz not needed.
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
MOVE_TIME_SEC = 3.0            # minimum seconds per move
MAX_JOINT_SPEED_RAD_S = 0.5    # long moves are slowed to stay under this (rad/s, no collision checking)
SETTLE_TIME_SEC = 3.0          # wait after each move before sampling
SAMPLES_PER_POSE = 5           # repeat detections per pose per camera; median-closest kept (jitter rejection)
SAMPLE_DELAY_SEC = 0.2         # seconds between repeat captures
MAX_SPREAD_MM = 3.0            # reject a pose/camera if repeat captures disagree by more than this
MIN_SAMPLES_WARN = 8           # warn (not abort) if a camera ends up with fewer accepted poses than this
BOARD_TYPE = 'aruco'           # the grid board on hand; switch to 'charuco' once that board is printed (see README)
BOARD_YAML_OVERRIDE = None     # None -> calibrations/{BOARD_TYPE}_board.yaml next to this package
SQUARE_LENGTH_M_OVERRIDE = None    # charuco only: caliper-measured square length, if remeasured
MARKER_LENGTH_M_OVERRIDE = None    # caliper-measured marker length, if remeasured
MARKER_SEPARATION_M_OVERRIDE = None  # aruco only: caliper-measured marker separation, if remeasured
MIN_FEATURES = None            # None -> 6 for charuco, 3 for aruco (of the board's 12 markers)
METHOD = 'park'                # see METHODS comment above
OUT_DIR = os.path.expanduser('~')


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


def _leave_one_out_diagnosis(s: Dict[str, List], method_key: str, max_pairs: int = 300):
    """Solve with the full sample set, then re-solve once per sample with that
    one sample left out, to see how much each pose is dragging on the result.

    A pose that's actually bad (misdetection, board slipped, robot still
    settling, ...) pulls the AX=XB residual up; solving without it should
    make the residual noticeably *better*. A pose that's fine barely moves
    the residual either way when dropped, or makes it worse. This is a
    standard leave-one-out (LOO) outlier check, just applied to hand-eye
    calibration's own AX=XB consistency metric instead of a prediction error.

    Returns (R_sol, t_sol, rot_res, trans_res, ranked) where the first four
    are the full-set solve and its residual (same quantities the caller
    already printed pre-diagnosis), and ranked is a list of
    (pose_idx, rot_deg_without_it, trans_mm_without_it, trans_mm_improvement)
    sorted with the single biggest offender (most improvement from removal)
    first. ranked is empty when there are too few samples (< 6) for the
    leave-one-out solves to still be well-conditioned (each needs >= 3 left).
    """
    R_g2b, t_g2b, R_t2c, t_t2c = s['R_g2b'], s['t_g2b'], s['R_t2c'], s['t_t2c']
    pose_idx = s['pose_idx']
    n = len(R_g2b)

    R_sol, t_sol = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=METHODS[method_key])
    t_sol = t_sol.reshape(3)
    rot_res, trans_res = _pairwise_axxb_residual(R_g2b, t_g2b, R_t2c, t_t2c, R_sol, t_sol, max_pairs)

    ranked = []
    if n >= 6:
        baseline_trans_mean = float(trans_res.mean())
        for k in range(n):
            idxs = [j for j in range(n) if j != k]
            sub_R_g2b = [R_g2b[j] for j in idxs]
            sub_t_g2b = [t_g2b[j] for j in idxs]
            sub_R_t2c = [R_t2c[j] for j in idxs]
            sub_t_t2c = [t_t2c[j] for j in idxs]
            R_xk, t_xk = cv2.calibrateHandEye(sub_R_g2b, sub_t_g2b, sub_R_t2c, sub_t_t2c, method=METHODS[method_key])
            rot_k, trans_k = _pairwise_axxb_residual(
                sub_R_g2b, sub_t_g2b, sub_R_t2c, sub_t_t2c, R_xk, t_xk.reshape(3), max_pairs)
            trans_k_mean = float(trans_k.mean())
            ranked.append((pose_idx[k], float(rot_k.mean()), trans_k_mean, baseline_trans_mean - trans_k_mean))
        ranked.sort(key=lambda r: -r[3])

    return R_sol, t_sol, rot_res, trans_res, ranked


def _print_loo_report(ns: str, baseline_rot_mean: float, baseline_trans_mean: float,
                      ranked: List[Tuple[int, float, float, float]], s: Dict[str, List]) -> None:
    n = len(s['pose_idx'])
    spread_by_pose = dict(zip(s['pose_idx'], s['spread_mm']))
    print(f'\n  [{ns}] leave-one-out diagnostic over {n} poses '
          f'(full-set baseline: rot mean={baseline_rot_mean:.3f} deg, trans mean={baseline_trans_mean:.2f} mm):')
    print(f'    {"pose#":>6}  {"trans w/o it":>13}  {"Δ vs baseline":>14}  {"rot w/o it":>11}  {"capture spread":>15}')
    for pose_no, rot_m, trans_m, delta in ranked:
        flag = '  <-- biggest single-pose offender' if (pose_no, rot_m, trans_m, delta) == ranked[0] and delta > 0 else ''
        spread = spread_by_pose.get(pose_no)
        spread_str = f'{spread:.2f} mm' if spread is not None else 'n/a'
        print(f'    {pose_no:>6}  {trans_m:>10.2f} mm  {delta:>+11.2f} mm  {rot_m:>9.3f} deg  {spread_str:>15}{flag}')
    offenders = [r for r in ranked if r[3] > 0.25]  # >0.25mm improvement from dropping it
    if offenders:
        offender_list = ",".join(str(r[0]) for r in offenders)
        print(f'  Removing pose(s) {", ".join(str(r[0]) for r in offenders)} would meaningfully reduce the '
              'residual -- drop them and re-solve (no robot needed): '
              f'diagnose_handeye_samples --exclude-poses "{offender_list}".')
        # "capture spread" (repeat-detection disagreement within that one pose) is a direct read on
        # whether THAT capture was noisy (oblique angle, motion blur, still settling) as opposed to the
        # taught joint pose itself being bad -- a flagged pose with unremarkable spread more likely means
        # the joint pose genuinely conflicts with the others (re-teach or drop it); a flagged pose that
        # also had high spread was probably just a noisy detection (retake it, or fix settle-time/angle).
        median_spread = float(np.median(list(spread_by_pose.values())))
        noisy = [r[0] for r in offenders if (spread_by_pose.get(r[0]) or 0) > 2 * median_spread]
        if noisy:
            print(f'  Pose(s) {", ".join(str(p) for p in noisy)} also had a capture spread well above the '
                  f'{median_spread:.2f}mm median for this set -- that alone points at a noisy detection there '
                  '(oblique angle / motion blur / still settling) rather than a bad taught pose; worth a '
                  'retake before assuming that joint configuration itself is the problem.')
    else:
        print('  No single pose stands out; the error looks evenly spread across poses -- suspect the mounted '
              'target measurement, camera intrinsics, or extrinsic frame instead of any one pose.')


def _rt_to_dict(R: np.ndarray, t: np.ndarray) -> dict:
    q = _matrix_to_quat(R)
    return {'xyz': [float(v) for v in t], 'quat_xyzw': [float(v) for v in q]}


def _samples_from_dicts(rows: List[dict]) -> Dict[str, List]:
    s: Dict[str, List] = {'R_g2b': [], 't_g2b': [], 'R_t2c': [], 't_t2c': [], 'pose_idx': [], 'spread_mm': []}
    for row in rows:
        qx, qy, qz, qw = row['g2b']['quat_xyzw']
        s['R_g2b'].append(_quat_to_matrix(qx, qy, qz, qw))
        s['t_g2b'].append(np.array(row['g2b']['xyz'], dtype=float))
        qx, qy, qz, qw = row['t2c']['quat_xyzw']
        s['R_t2c'].append(_quat_to_matrix(qx, qy, qz, qw))
        s['t_t2c'].append(np.array(row['t2c']['xyz'], dtype=float))
        s['pose_idx'].append(row['pose_idx'])
        s['spread_mm'].append(row.get('spread_mm'))
    return s


def _write_samples_yaml(path: str, ns: str, link: str, board_type: str, method: str, s: Dict[str, List]) -> None:
    """Dump the raw per-pose AX/AB samples behind a solve, so a run can be
    re-diagnosed or re-solved with poses excluded later without re-driving
    the robot. See diagnose_handeye_samples.py."""
    rows = [{
        'pose_idx': s['pose_idx'][k],
        'spread_mm': s['spread_mm'][k],
        'g2b': _rt_to_dict(s['R_g2b'][k], s['t_g2b'][k]),
        't2c': _rt_to_dict(s['R_t2c'][k], s['t_t2c'][k]),
    } for k in range(len(s['R_g2b']))]
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump({'camera': ns, 'link_frame': link, 'board_type': board_type, 'method': method,
                        'samples': rows}, f, sort_keys=False)


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
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    board_yaml = BOARD_YAML_OVERRIDE or os.path.normpath(
        os.path.join(here, '..', 'calibrations', f'{BOARD_TYPE}_board.yaml'))
    min_features = MIN_FEATURES if MIN_FEATURES is not None else (6 if BOARD_TYPE == 'charuco' else 3)
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

        samples: Dict[str, Dict[str, List]] = {
            ns: {'R_g2b': [], 't_g2b': [], 'R_t2c': [], 't_t2c': [], 'pose_idx': [], 'spread_mm': []}
            for ns, *_ in active
        }
        missed: Dict[str, int] = {ns: 0 for ns, *_ in active}

        total = len(joint_values)
        for i, target in enumerate(joint_values, start=1):
            if (i - 1) in skip_idxs:
                continue
            current = pose_runner._current_positions()
            max_delta = max(abs(a - b) for a, b in zip(target, current))
            move_time = max(0.5, MOVE_TIME_SEC, max_delta / max(MAX_JOINT_SPEED_RAD_S, 1e-3))
            print(f'Pose {i}/{total}: largest joint delta {np.degrees(max_delta):.1f} deg, '
                  f'move time {move_time:.1f}s')
            if args.step:
                print('  joint deltas (deg): '
                      + ', '.join(f'{np.degrees(b - a):+.1f}' for a, b in zip(current, target)))
                input('  Press Enter to move (Ctrl+C to abort)...')
            if not pose_runner.move_to(target, move_time_sec=move_time):
                print(f'  move failed, skipping pose {i}')
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
                tvecs = np.array([t for _, t in hits])
                spread_mm = float(np.linalg.norm(tvecs - tvecs.mean(axis=0), axis=1).max() * 1000)
                if spread_mm > MAX_SPREAD_MM:
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
                samples[ns]['pose_idx'].append(i)
                samples[ns]['spread_mm'].append(spread_mm)
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

            R_sol, t_sol, rot_res, trans_res, loo_ranked = _leave_one_out_diagnosis(s, METHOD)
            quat = _matrix_to_quat(R_sol)

            print(f'  base_link -> {link}:')
            print(f'    xyz = [{t_sol[0]:.6f}, {t_sol[1]:.6f}, {t_sol[2]:.6f}]')
            print(f'    quat(xyzw) = [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]')
            print(f'    AX=XB residual over {len(rot_res)} pose pairs: '
                  f'rotation mean={rot_res.mean():.3f} deg max={rot_res.max():.3f} deg, '
                  f'translation mean={trans_res.mean():.2f} mm max={trans_res.max():.2f} mm')
            if loo_ranked:
                _print_loo_report(ns, float(rot_res.mean()), float(trans_res.mean()), loo_ranked, s)
            else:
                print(f'  (skipping leave-one-out diagnostic: need >= 6 accepted poses, have {n})')

            path = _write_launch_file(out_dir, ns, link, t_sol, quat, float(trans_res.mean()), n, METHOD)
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
