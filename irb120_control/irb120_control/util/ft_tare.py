"""Helpers for starting a NetFT session tare before an experiment."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import SetBool


def tare_netft(
    node,
    service_name: str = "/netft_preprocessor/set_tare",
    timeout_sec: float = 3.0,
    settle_sec: float = 1.5,
) -> bool:
    """Request a NetFT session tare and wait while samples are collected."""
    latest_wrench: WrenchStamped | None = None

    def _on_wrench(msg: WrenchStamped) -> None:
        nonlocal latest_wrench
        latest_wrench = msg

    wrench_sub = node.create_subscription(WrenchStamped, "/netft_data_transformed", _on_wrench, 10)
    client = node.create_client(SetBool, service_name)
    if not client.wait_for_service(timeout_sec=timeout_sec):
        node.get_logger().error(f"{service_name} service not available; cannot tare F/T sensor")
        node.destroy_subscription(wrench_sub)
        return False

    node.get_logger().info("Starting NetFT session tare. Keep the finger unloaded and stationary.")
    future = client.call_async(SetBool.Request(data=True))
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    result = future.result() if future.done() else None
    if result is None or not result.success:
        node.get_logger().error(f"NetFT tare request failed: {result.message if result else 'no response'}")
        node.destroy_subscription(wrench_sub)
        return False

    node.get_logger().info(f"NetFT tare requested: {result.message}")
    end_time = node.get_clock().now().nanoseconds * 1e-9 + settle_sec
    while rclpy.ok() and node.get_clock().now().nanoseconds * 1e-9 < end_time:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.get_logger().info("NetFT tare wait complete")
    if latest_wrench is None:
        node.get_logger().warn("No transformed F/T sample received after tare")
    else:
        f = latest_wrench.wrench.force
        t = latest_wrench.wrench.torque
        node.get_logger().info(
            "Post-tare transformed wrench: "
            f"f=({f.x:+.4f}, {f.y:+.4f}, {f.z:+.4f}) N  "
            f"t=({t.x:+.5f}, {t.y:+.5f}, {t.z:+.5f}) N*m"
        )
    node.destroy_subscription(wrench_sub)
    return True
