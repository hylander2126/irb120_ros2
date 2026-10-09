#!/usr/bin/env python3
"""Calibrate FT sensor gravity-load model — interactive stdin version.

Usage
-----
    ros2 run irb120_control calibrate_ft_sensor

The node runs in the same terminal as your input. Commands:

    <Enter>  capture a pose at the current robot orientation
    s        solve using all captured poses (need ≥3 distinct orientations)
    r        reset (clear all captured poses)
    p        print status (how many poses captured so far)
    q        quit

Workflow:
    1. Move robot to pose 1, wait for it to settle. Press Enter.
    2. Move robot to pose 2 (rotate ~90° about a different axis), settle. Press Enter.
    3. Move robot to pose 3 (rotate ~90° about yet another axis), settle. Press Enter.
    4. Type 's' to solve and save a JSON artifact. Review it, then copy it to
       irb120_control/config/ft_calibration.json to activate it.

This utility only samples; it never commands robot motion.

Model (same as the service-based version):
    raw_sensor = electrical_zero_sensor + R_cur^T @ g_world

Each pose contributes 3 force equations. At least 3 poses spanning multiple rotation axes are required for the 6 unknowns
(3 electrical_zero, 3 g_world). Two poses always leave an unobservable direction.
"""

import sys
import json
from datetime import datetime
from pathlib import Path
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener
from irb120_control.util.ft_frames import WRENCH_AXES_FRAME
from irb120_control.util.ft_calibration import (
    default_calibration_path, solve_force_calibration, solve_torque_calibration)
from irb120_control.util.runtime_log_dir import runtime_log_dir, git_commit_info


INPUT_TOPIC      = "/netft_data"
BASE_FRAME       = "world"
SENSOR_FRAME     = WRENCH_AXES_FRAME
SAMPLES_PER_POSE = 200
SETTLE_TIMEOUT_S = 5.0
G_MAGNITUDE      = 9.81


class CalibrateFTSensor(Node):
    def __init__(self):
        super().__init__("calibrate_ft_sensor")

        default_output = str(runtime_log_dir(
            f"uncertainty_checks/{datetime.now():%Y%m%d}/calibration"))
        self._output_dir = Path(self.declare_parameter('output_dir', default_output).value)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self._sub         = self.create_subscription(WrenchStamped, INPUT_TOPIC, self._on_wrench, qos)
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._poses        = []
        self._accumulating = False
        self._acc_f        = np.zeros(3)
        self._acc_t        = np.zeros(3)
        self._acc_count    = 0
        self._acc_R        = None
        self._acc_done     = threading.Event()

    # Wrench hot path — accumulates only while triggered
    def _on_wrench(self, msg: WrenchStamped):
        if not self._accumulating:
            return

        f = np.array([msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z])
        t = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        if not (np.isfinite(f).all() and np.isfinite(t).all()):
            return

        self._acc_f += f
        self._acc_t += t
        self._acc_count += 1
        if self._acc_count >= SAMPLES_PER_POSE:
            inv = 1.0 / self._acc_count
            self._poses.append({
                "R":     self._acc_R,
                "raw_f": self._acc_f * inv,
                "raw_t": self._acc_t * inv,
                "capture_time_s": self._capture_time_s,
                "tf_stamp_s": self._capture_tf_stamp_s,
                "samples": self._acc_count,
            })
            self._accumulating = False
            self._acc_done.set()

    # Commands
    def capture_pose(self):
        """Snapshot R from TF, accumulate SAMPLES_PER_POSE wrench samples, store."""
        if self._accumulating:
            print("[!] Still accumulating previous pose.")
            return

        try:
            tf = self._tf_buffer.lookup_transform(BASE_FRAME, SENSOR_FRAME, rclpy.time.Time())
        except TransformException as e:
            print(f"[!] TF lookup failed: {e}")
            return

        self._capture_time_s = self.get_clock().now().nanoseconds * 1e-9
        self._capture_tf_stamp_s = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        if self._capture_tf_stamp_s == 0 or not -0.1 <= self._capture_time_s - self._capture_tf_stamp_s <= 0.5:
            print("[!] TF is stale; capture refused.")
            return
        q = tf.transform.rotation
        self._acc_R     = self._quat_to_rot(q.x, q.y, q.z, q.w)
        self._acc_f     = np.zeros(3)
        self._acc_t     = np.zeros(3)
        self._acc_count = 0
        self._acc_done.clear()
        self._accumulating = True

        pose_num = len(self._poses) + 1
        print(f"[pose {pose_num}] accumulating {SAMPLES_PER_POSE} samples "
              f"(~{SAMPLES_PER_POSE/500:.1f}s @ 500Hz)...", end="", flush=True)

        if not self._acc_done.wait(timeout=SETTLE_TIMEOUT_S):
            self._accumulating = False
            print(f" TIMEOUT after {SETTLE_TIMEOUT_S}s.")
            return

        p = self._poses[-1]
        rf, rt = p["raw_f"], p["raw_t"]
        print(f" done.\n   raw_f = ({rf[0]:+.4f}, {rf[1]:+.4f}, {rf[2]:+.4f}) N")
        print(  f"   raw_t = ({rt[0]:+.4f}, {rt[1]:+.4f}, {rt[2]:+.4f}) N.m")
        print(  f"   total poses captured: {len(self._poses)}")

    def solve(self):
        n = len(self._poses)
        if self._accumulating:
            print("[!] Wait for capture to finish before solving.")
            return
        try:
            electrical_zero, g_world, sv, residuals = solve_force_calibration(self._poses)
            torque_zero, tool_com, _, torque_residuals = solve_torque_calibration(
                self._poses, g_world)
        except ValueError as exc:
            print(f"[!] {exc}")
            return

        cond = sv[0] / sv[-1] if sv[-1] > 1e-12 else float("inf")

        gmag = np.linalg.norm(g_world)
        ghat = g_world / gmag if gmag > 1e-6 else np.zeros(3)
        mass = gmag / G_MAGNITUDE
        tilt_deg = np.degrees(np.arccos(max(-1.0, min(1.0, -ghat[2]))))

        bar = "=" * 70
        print(f"\n{bar}\nCALIBRATION SOLVE - {n} poses\n{bar}")
        print(f"electrical_zero_sensor (N) = "
              f"({electrical_zero[0]:+.4f}, {electrical_zero[1]:+.4f}, {electrical_zero[2]:+.4f})")
        print(f"g_world                (N) = "
              f"({g_world[0]:+.4f}, {g_world[1]:+.4f}, {g_world[2]:+.4f})")
        print(f"  magnitude               = {gmag:.4f} N  ->  mass = {mass:.4f} kg")
        print(f"  direction (unit)        = ({ghat[0]:+.3f}, {ghat[1]:+.3f}, {ghat[2]:+.3f})")
        print(f"  tilt from -z_world      = {tilt_deg:.2f} deg  "
              + ("(pure gravity)" if tilt_deg < 5.0
                 else "(WARNING: load not purely gravity-aligned)"))
        print()
        print(f"design matrix conditioning: cond = {cond:.1f}  "
              "(want <50; >100 = redo with more-different poses)")
        print("per-pose residuals (raw - predicted), want < 0.05 N:")
        for i, r in enumerate(residuals):
            print(f"  pose {i+1}: ({r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f})  "
                  f"|.|={np.linalg.norm(r):.4f} N")
        print()
        print(f"torque_zero (N*m)      = ({torque_zero[0]:+.5f}, {torque_zero[1]:+.5f}, "
              f"{torque_zero[2]:+.5f})")
        print(f"tool CoM (m, sensor)   = ({tool_com[0]:+.4f}, {tool_com[1]:+.4f}, {tool_com[2]:+.4f})"
              "  (the finger hangs along the tool axis; expect a few cm along it)")
        print("per-pose torque residuals, want < 0.002 N*m:")
        for i, r in enumerate(torque_residuals):
            print(f"  pose {i+1}: ({r[0]:+.5f}, {r[1]:+.5f}, {r[2]:+.5f})  "
                  f"|.|={np.linalg.norm(r):.5f} N*m")
        print("Large force residuals with a sensible |g| usually mean the sensor axes are not")
        print(f"the {SENSOR_FRAME} axes (NetFT tool transform); do not activate the file then.")
        print(bar)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / f"ft_calibration_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        artifact = {
            'schema_version': 2, 'axes_frame': SENSOR_FRAME, 'base_frame': BASE_FRAME,
            'electrical_zero': electrical_zero.tolist(), 'g_world': g_world.tolist(),
            'torque_zero': torque_zero.tolist(), 'tool_com_m': tool_com.tolist(),
            'torque_model': 'torque_zero + tool_com x (R^T g_world)',
            'condition_number': float(cond), 'residuals_n': np.asarray(residuals).tolist(),
            'torque_residuals_nm': np.asarray(torque_residuals).tolist(),
            'gravity_tilt_deg': float(tilt_deg), **git_commit_info(),
            'poses': [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in p.items()}
                      for p in self._poses],
        }
        path.write_text(json.dumps(artifact, indent=2) + '\n')
        print(f"Saved {path} (not activated).")
        print(f"To activate after review: cp {path} {default_calibration_path()}")
        print("then restart the netft_preprocessor (relaunch bringup_stack).")

    def reset(self):
        if self._accumulating:
            print("[!] Wait for capture to finish before resetting.")
            return
        n = len(self._poses)
        self._poses = []
        print(f"[reset] cleared {n} pose(s).")

    def print_status(self):
        n = len(self._poses)
        if self._accumulating:
            print(f"[status] accumulating pose {n+1} ({self._acc_count}/{SAMPLES_PER_POSE})")
        else:
            print(f"[status] {n} pose(s) captured. "
                  + ("Ready to solve." if n >= 3 else "Need >=3 poses."))

    @staticmethod
    def _quat_to_rot(x, y, z, w):
        """Quaternion -> 3x3 rotation matrix; R takes sensor-frame vectors to world."""
        x2, y2, z2 = x*x, y*y, z*z
        return np.array([
            [1 - 2*(y2+z2),   2*(x*y - w*z),  2*(x*z + w*y)],
            [2*(x*y + w*z),   1 - 2*(x2+z2),  2*(y*z - w*x)],
            [2*(x*z - w*y),   2*(y*z + w*x),  1 - 2*(x2+y2)],
        ])


def _print_help():
    print(
        "\nCommands (then Enter):\n"
        "  <Enter>  capture pose at current orientation\n"
        "  s        solve (need >=3 distinct orientations)\n"
        "  r        reset all captured poses\n"
        "  p        print status\n"
        "  h, ?     show this help\n"
        "  q        quit\n"
    )


def _input_loop(node, stop_event):
    print("=" * 70)
    print("FT Sensor Calibration - interactive mode")
    print("=" * 70)
    print(f"Subscribing to {INPUT_TOPIC}")
    print(f"TF: {BASE_FRAME} -> {SENSOR_FRAME}")
    print(f"Samples per pose: {SAMPLES_PER_POSE} (~{SAMPLES_PER_POSE/500:.1f}s)")
    _print_help()

    while not stop_event.is_set():
        try:
            line = input(">> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            stop_event.set()
            break

        if line == "":
            node.capture_pose()
        elif line == "s":
            node.solve()
        elif line == "r":
            node.reset()
        elif line == "p":
            node.print_status()
        elif line in ("h", "?"):
            _print_help()
        elif line == "q":
            stop_event.set()
            break
        else:
            print(f"[?] Unknown command '{line}'. Type 'h' for help.")


def main(args=None):
    rclpy.init(args=args)
    node = CalibrateFTSensor()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    stop_event = threading.Event()
    try:
        _input_loop(node, stop_event)
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()