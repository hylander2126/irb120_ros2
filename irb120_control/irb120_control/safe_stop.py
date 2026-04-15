#!/usr/bin/env python3
"""Best-effort forced stop for RAPID + EGM via rws_client services."""

import sys
import time

import rclpy
from rclpy.node import Node

from abb_robot_msgs.srv import TriggerWithResultCode


class SafeStop(Node):
    def __init__(self) -> None:
        super().__init__("safe_stop")
        self.declare_parameter("rws_service_prefix", "/rws_client")
        self.declare_parameter("attempts", 6)
        self.declare_parameter("timeout_sec", 2.0)
        self.declare_parameter("sleep_between", 0.25)

        prefix = str(self.get_parameter("rws_service_prefix").value).rstrip("/")
        self.attempts = int(self.get_parameter("attempts").value)
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.sleep_between = float(self.get_parameter("sleep_between").value)

        self.stop_rapid_srv = f"{prefix}/stop_rapid"
        self.stop_egm_srv = f"{prefix}/stop_egm"

    def _call_trigger(self, service_name: str) -> tuple[bool, str]:
        client = self.create_client(TriggerWithResultCode, service_name)
        if not client.wait_for_service(timeout_sec=self.timeout_sec):
            return False, "service unavailable"

        future = client.call_async(TriggerWithResultCode.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout_sec)
        if not future.done() or future.result() is None:
            return False, "call failed"

        res = future.result()
        return res.result_code == 1, f"code={res.result_code}, msg='{res.message}'"

    def _retry(self, service_name: str) -> bool:
        for i in range(1, self.attempts + 1):
            ok, detail = self._call_trigger(service_name)
            self.get_logger().info(f"{service_name} attempt {i}/{self.attempts}: {detail}")
            if ok:
                return True
            time.sleep(self.sleep_between)
        return False

    def run(self) -> int:
        # Stop RAPID, stop EGM, stop RAPID again for a stronger idle transition.
        ok1 = self._retry(self.stop_rapid_srv)
        ok2 = self._retry(self.stop_egm_srv)
        ok3 = self._retry(self.stop_rapid_srv)

        if ok1 and ok2 and ok3:
            self.get_logger().info("safe_stop completed successfully")
            return 0

        self.get_logger().error("safe_stop could not fully confirm stop state")
        return 2


def main() -> int:
    rclpy.init()
    node = SafeStop()
    rc = 1
    try:
        rc = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
