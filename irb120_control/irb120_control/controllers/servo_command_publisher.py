#!/usr/bin/env python3
"""Helper for publishing Servo twist commands with diagnostics logging."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from geometry_msgs.msg import TwistStamped


def _resolve_workspace_root() -> Path:
    file_path = Path(__file__).resolve()
    for parent in file_path.parents:
        if (parent / "runtime_logs").exists():
            return parent
        if (parent / "src").is_dir() and (parent / "build").is_dir() and (parent / "install").is_dir():
            return parent
    # Fallback for unusual layouts.
    return file_path.parents[0]


class ServoCommandPublisher:
    """Publishes validated Twist commands and logs command/force correlation."""

    def __init__(self, node, topic: str, frame_id: str, log_name_prefix: str = "squash_pull_log") -> None:
        self._node = node
        self._frame_id = frame_id
        self._pub = node.create_publisher(TwistStamped, topic, 10)

        workspace_root = _resolve_workspace_root()
        log_dir = workspace_root / "runtime_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = log_dir / f"{log_name_prefix}_{timestamp}.csv"
        self._log_file = self._log_path.open("w")
        self._log_file.write("timestamp_s,state,force_z_N,cmd_vx,cmd_vy,cmd_vz\n")
        self._log_file.flush()
        self._log_start_time = None
        self._last_nonfinite_warn = 0.0

    @property
    def log_path(self):
        return self._log_path

    def publish_twist(self, vx: float, vy: float, vz: float, state: str, force_z: float) -> None:
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
            now = self._node.get_clock().now().nanoseconds * 1e-9
            if now - self._last_nonfinite_warn > 1.0:
                self._last_nonfinite_warn = now
                self._node.get_logger().error(
                    "Refusing to publish non-finite twist command (NaN/Inf). Sending zero instead."
                )
            vx, vy, vz = 0.0, 0.0, 0.0

        now_ns = self._node.get_clock().now().nanoseconds
        if self._log_start_time is None:
            self._log_start_time = now_ns
        elapsed_s = (now_ns - self._log_start_time) * 1e-9
        self._log_file.write(f"{elapsed_s:.6f},{state},{force_z:.4f},{vx:.6f},{vy:.6f},{vz:.6f}\n")
        self._log_file.flush()

        msg = TwistStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        self._pub.publish(msg)

    def publish_zero(self, state: str, force_z: float) -> None:
        self.publish_twist(0.0, 0.0, 0.0, state, force_z)

    def close(self) -> None:
        if not self._log_file.closed:
            self._log_file.close()
