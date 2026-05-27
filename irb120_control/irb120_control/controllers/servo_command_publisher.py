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
    """Publishes validated MoveIt Servo twist commands."""

    def __init__(
        self,
        node,
        topic: str,
        frame_id: str,
        log_name_prefix: str | None = None,
    ) -> None:
        self._node = node
        self._topic = topic
        self._frame_id = frame_id
        self._pub = node.create_publisher(TwistStamped, topic, 10)

        self._log_path = None
        self._log_file = None
        if log_name_prefix is not None:
            workspace_root = _resolve_workspace_root()
            log_dir = workspace_root / "runtime_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_path = log_dir / f"{log_name_prefix}_{timestamp}.csv"
            self._log_file = self._log_path.open("w")
            self._log_file.write("timestamp_s,state,force_z_N,cmd_vx,cmd_vy,cmd_vz,cmd_wy\n")
            self._log_file.flush()
        self._log_start_time = None
        self._last_nonfinite_warn = 0.0
        self._missing_subscriber_logged = False

    @property
    def log_path(self):
        return self._log_path

    def has_subscribers(self) -> bool:
        return self._node.count_subscribers(self._topic) > 0

    def publish_twist(
        self,
        vx: float,
        vy: float,
        vz: float,
        wy: float = 0.0,
        state: str = "",
        force_z: float = 0.0,
    ) -> bool:
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz) and math.isfinite(wy)):
            now = self._node.get_clock().now().nanoseconds * 1e-9
            if now - self._last_nonfinite_warn > 1.0:
                self._last_nonfinite_warn = now
                self._node.get_logger().error(
                    "Refusing to publish non-finite twist command (NaN/Inf). Sending zero instead."
                )
            vx, vy, vz, wy = 0.0, 0.0, 0.0, 0.0

        if not self.has_subscribers():
            if not self._missing_subscriber_logged:
                self._missing_subscriber_logged = True
                self._node.get_logger().error(f"No subscribers on {self._topic}; refusing to publish twist")
            return False

        if self._log_file is not None:
            now_ns = self._node.get_clock().now().nanoseconds
            if self._log_start_time is None:
                self._log_start_time = now_ns
            elapsed_s = (now_ns - self._log_start_time) * 1e-9
            self._log_file.write(
                f"{elapsed_s:.6f},{state},{force_z:.4f},"
                f"{vx:.6f},{vy:.6f},{vz:.6f},{wy:.6f}\n"
            )
            self._log_file.flush()

        msg = TwistStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.y = wy
        self._pub.publish(msg)
        return True

    def publish_zero(self, state: str = "", force_z: float = 0.0) -> bool:
        return self.publish_twist(0.0, 0.0, 0.0, state=state, force_z=force_z)

    def close(self) -> None:
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.close()
