#!/usr/bin/env python3
"""Teach calibration poses: move the arm with MoveIt/RViz, press a key to save.

Run alongside the full stack (hardware + MoveIt + all cameras). Drag the
tool0 interactive marker in RViz and Plan & Execute as usual; this node shows,
live, whether each camera currently sees the board (and at what distance and
viewing angle), and saves the arm's current joint values on a keypress.

Keys (no Enter needed):
  s / Space   save the current joint state as a new pose
  u           undo (delete) the last saved pose
  q / Ctrl-C  quit

The output YAML has the same joint_names/joint_values layout as the other
pose sets, plus a parallel `detections` list recording what each camera saw
at save time. Re-running with the same --out appends to it. Test the result
with:
  ros2 run irb120_handeye run_handeye_calibration --pose-path <file> --step --dry-run

Wait for the arm to finish moving before saving; the status line reflects the
latest camera frame, not a settled pose.
"""

import argparse
import os
import select
import sys
import termios
import time
import tty
from typing import List, Optional

import numpy as np
import rclpy
import yaml
from sensor_msgs.msg import JointState

from irb120_handeye.run_calibration_poses import _load_pose_yaml, _resolve_pose_path
from irb120_handeye.run_handeye_calibration import (
    BASE_FRAME,
    CAMERA_LINK_FRAMES,
    DEFAULT_MIN_CHARUCO_CORNERS,
    GRIPPER_FRAME,
    METHOD,
    HandEyeCameraNode,
    _append_sample,
    _detect_board_pose,
    _load_board,
    _new_samples,
    _solve_handeye,
    _transform_to_Rt,
)

WARN_ANGLE_DEG = 60.0  # beyond this the board is viewed too obliquely for good corner accuracy


class PoseRecorder(HandEyeCameraNode):
    def __init__(self, cameras: List[str], joint_names: List[str]):
        super().__init__(cameras)
        self.joint_names = joint_names
        self._joints = {}
        self.create_subscription(JointState, '/joint_states', self._on_joints, 20)

    def _on_joints(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self._joints[name] = float(pos)

    def current_joints(self) -> Optional[List[float]]:
        if not all(j in self._joints for j in self.joint_names):
            return None
        return [self._joints[j] for j in self.joint_names]


def _view_status(det) -> str:
    R, t = det.R, det.t
    dist = float(np.linalg.norm(t))
    to_cam = -t / dist
    angle = float(np.degrees(np.arccos(np.clip(abs(R[:, 2] @ to_cam), 0.0, 1.0))))
    flag = ' (oblique!)' if angle > WARN_ANGLE_DEG else ''
    return f'OK {dist:.2f}m {angle:.0f}deg{flag}'


def _default_out_path() -> str:
    return os.path.expanduser('~/joints_custom.yaml')


def _write(path: str, joint_names, joint_values, detections) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump({'joint_names': list(joint_names), 'joint_values': joint_values,
                        'detections': detections}, f, sort_keys=False)


def _report_live_solution(ns: str, samples: dict) -> None:
    """Print incremental solve quality after a newly saved sample."""
    n = len(samples["R_g2b"])
    if n < 3:
        print(f"  [{ns}] {n}/3 samples: collecting the minimum solve set")
        return
    try:
        sol = _solve_handeye(samples, METHOD)
    except Exception as exc:
        print(f"  [{ns}] live solve unavailable: {exc}")
        return
    print(f"  [{ns}] live {n}-sample solve: reprojection RMS {sol.rms_px:.2f} px "
          f"(~{sol.rms_mm:.2f} mm at the board)")
    latest_pose = samples["pose_idx"][-1]
    median = float(np.median(sol.pose_rms_px))
    newest = (f"newest pose {latest_pose}: {sol.pose_rms_px[-1]:.2f} px "
              f"({sol.pose_rms_px[-1] / median:.1f}x median), {len(samples['img_pts'][-1])} corners")
    if n < 6:
        print(f"  [{ns}] {newest} -- provisional: add {6 - n} more samples before outlier flagging")
        return
    print(f"  [{ns}] {newest} -- {'OUTLIER, consider undo' if latest_pose in sol.outliers else 'OK'}")
    earlier = [p for p in sol.outliers if p != latest_pose]
    if earlier:
        print(f"  [{ns}] earlier pose(s) {earlier} now flagged as outliers")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=None, help='Pose YAML to create/append to (default: ~/joints_custom.yaml).')
    p.add_argument('--cameras', default='realsense,realsense2,realsense3')
    p.add_argument('--board-type', choices=['charuco', 'aruco'], default='charuco',
                   help='Calibration target type (default: the calibrated ChArUco board).')
    p.add_argument('--board-yaml', default=None)
    p.add_argument('--marker-length-m', type=float, default=None)
    p.add_argument('--marker-separation-m', type=float, default=None)
    p.add_argument('--square-length-m', type=float, default=None)
    p.add_argument('--min-features', type=int, default=None)
    # p.add_argument('--joint-names-from', default='joints_5_6mm.yaml',
                #    help='Existing pose file to copy the joint name order from.')
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    board_yaml = args.board_yaml or os.path.normpath(
        os.path.join(here, '..', 'calibrations', f'{args.board_type}_board.yaml'))
    min_features = args.min_features if args.min_features is not None else (DEFAULT_MIN_CHARUCO_CORNERS if args.board_type == 'charuco' else 3)
    cameras = [c.strip() for c in args.cameras.split(',') if c.strip()]
    for ns in cameras:
        if ns not in CAMERA_LINK_FRAMES:
            print(f'Unknown camera {ns!r}; known cameras: {sorted(CAMERA_LINK_FRAMES)}')
            return 1
    board = _load_board(args.board_type, board_yaml, args.square_length_m, args.marker_length_m,
                        args.marker_separation_m)

    joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']# _load_pose_yaml(_resolve_pose_path(None, args.joint_names_from))
    out_path = os.path.abspath(args.out) if args.out else _default_out_path()
    joint_values: List[List[float]] = []
    detections: List[dict] = []
    if os.path.isfile(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            existing = yaml.safe_load(f) or {}
        if existing.get('joint_names') == joint_names:
            joint_values = existing.get('joint_values', [])
            detections = existing.get('detections', [None] * len(joint_values))
            print(f'Appending to {out_path} ({len(joint_values)} existing poses)')
        else:
            print(f'{out_path} exists with different joint names; refusing to overwrite.')
            return 1
    else:
        print(f'Saving to {out_path}')

    rclpy.init()
    node = PoseRecorder(cameras, joint_names)
    fd = sys.stdin.fileno()
    old_tty = termios.tcgetattr(fd)
    counts = {ns: sum(1 for d in detections if d and d.get(ns)) for ns in cameras}
    # Raw transforms captured in this teaching session for incremental solves.
    # Older pose YAML files contain only visibility booleans and are intentionally
    # not mixed with these measured AX=XB samples.
    live_samples = {ns: _new_samples() for ns in cameras}
    print('Keys: s/Space = save pose, u = undo last, q = quit\n')
    try:
        tty.setcbreak(fd)
        last_status = 0.0
        latest = {ns: None for ns in cameras}
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.time()
            if now - last_status > 0.5:
                last_status = now
                parts = []
                for ns in cameras:
                    gray, intr = node.latest_image(ns), node.intrinsics(ns)
                    det = _detect_board_pose(gray, board, *intr, min_features) if gray is not None and intr else None
                    latest[ns] = det
                    parts.append(f'{ns}: ' + (_view_status(det) if det else 'no board'))
                saved = ' '.join(f'{ns}={counts[ns]}' for ns in cameras)
                sys.stdout.write(f'\r\x1b[K{" | ".join(parts)}   [saved {len(joint_values)}: {saved}]')
                sys.stdout.flush()
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key in ('q', '\x03'):
                    break
                if key in ('s', ' '):
                    joints = node.current_joints()
                    if joints is None:
                        sys.stdout.write('\r\x1b[Kno /joint_states yet\n')
                        continue
                    seen = {ns: latest[ns] is not None for ns in cameras}
                    joint_values.append([float(v) for v in joints])
                    detections.append(seen)
                    for ns in cameras:
                        counts[ns] += int(seen[ns])
                    pose_no = len(joint_values)
                    g2b_tf = node.lookup(GRIPPER_FRAME, BASE_FRAME)
                    if g2b_tf is None:
                        print("  live solve skipped: no tool0 <- base_link TF")
                    else:
                        R_g2b, t_g2b = _transform_to_Rt(g2b_tf)
                        for ns in cameras:
                            if latest[ns] is None:
                                continue
                            link = CAMERA_LINK_FRAMES[ns]
                            optical = f"{ns}_color_optical_frame"
                            link_opt_tf = node.lookup(link, optical)
                            if link_opt_tf is None:
                                print(f"  [{ns}] live solve skipped: no {link} <- {optical} TF")
                                continue
                            R_link_opt, t_link_opt = _transform_to_Rt(link_opt_tf)
                            _append_sample(live_samples[ns], pose_no, None, R_g2b, t_g2b, latest[ns],
                                           *node.intrinsics(ns), R_link_opt, t_link_opt)
                            _report_live_solution(ns, live_samples[ns])
                    _write(out_path, joint_names, joint_values, detections)
                    sys.stdout.write(f'\r\x1b[Ksaved pose {len(joint_values)} '
                                     f'(seen by: {[ns for ns in cameras if seen[ns]] or "no camera"})\n')
                elif key == "u" and joint_values:
                    removed_pose_no = len(joint_values)
                    joint_values.pop()
                    removed = detections.pop()
                    for s in live_samples.values():
                        if s["pose_idx"] and s["pose_idx"][-1] == removed_pose_no:
                            for values in s.values():
                                values.pop()
                    for ns in cameras:
                        counts[ns] -= int(bool(removed and removed.get(ns)))
                    _write(out_path, joint_names, joint_values, detections)
                    sys.stdout.write(f'\r\x1b[Kundid; {len(joint_values)} poses remain\n')
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_tty)
        print(f'\n{len(joint_values)} poses in {out_path}')
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
