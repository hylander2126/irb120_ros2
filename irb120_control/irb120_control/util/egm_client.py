#!/usr/bin/env python3
"""Shared EGM start/stop helpers for motion scripts (push, squash_pull, etc.).

EGM is the underlying hardware interface that enables ALL robot motion,
including JTC-based MoveIt trajectories. Scripts must call ensure_egm_active()
before any motion and deactivate_egm() when done.

Typical usage in a script's main():

    from irb120_control.util.egm_client import ensure_egm_active, deactivate_egm

    if not ensure_egm_active(node):
        return 1
    try:
        ... do motions ...
    finally:
        deactivate_egm(node)

Safety contract
---------------
ensure_egm_active() guarantees that when it returns True:
  1. The JTC command stream has drained for at least cmd_timeout + margin so
     the hardware interface wakes up with zero commanded velocity.
  2. EGM has been started and the hardware interface has had time to register.

The robot will not move until the calling script explicitly sends a trajectory.
"""

from __future__ import annotations

import time

import rclpy
from abb_robot_msgs.srv import TriggerWithResultCode

_EGM_START_SRV = "/rws_client/start_egm_joint"
_EGM_STOP_SRV  = "/rws_client/stop_egm"

# Must be > cmd_timeout in irb120_controllers.yaml (0.1 s) so the JTC drops
# its output before EGM reconnects.  3× margin.
_JTC_DRAIN_SEC = 0.35


def _call_egm_service(node, service_name: str, timeout_sec: float) -> tuple[bool, str]:
    """Call a TriggerWithResultCode service. Returns (success, message)."""
    client = node.create_client(TriggerWithResultCode, service_name)
    if not client.wait_for_service(timeout_sec=timeout_sec):
        msg = f"{service_name} not available after {timeout_sec:.0f}s"
        node.get_logger().error(msg)
        return False, msg

    future = client.call_async(TriggerWithResultCode.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if not future.done() or future.result() is None:
        msg = f"{service_name} call timed out or returned no result"
        node.get_logger().error(msg)
        return False, msg

    res = future.result()
    success = (res.result_code == 1)
    return success, res.message


def _drain_jtc(node) -> None:
    """Wait for the JTC command stream to go silent before activating EGM.

    cmd_timeout in irb120_controllers.yaml is 0.1 s — sleeping longer than
    that guarantees the JTC has stopped sending position commands to the
    hardware interface before EGM connects, so the robot stays still.
    """
    node.get_logger().info(f"Draining JTC command stream ({_JTC_DRAIN_SEC:.2f}s)...")
    time.sleep(_JTC_DRAIN_SEC)


def ensure_egm_active(node, timeout_sec: float = 10.0, settle_sec: float = 1.5) -> bool:
    """Start EGM safely — robot will not move until the caller sends a command.

    Sequence:
      1. Wait _JTC_DRAIN_SEC for any in-flight JTC command stream to go silent.
      2. Call start_egm_joint.
      3. Wait settle_sec for the hardware interface to register the connection.

    Returns True on success, False if EGM is already idling/unavailable (abort).
    If EGM has timed out and the robot is idling, the operator must restart it
    via egm_handler before running motion scripts.
    """
    node.get_logger().info("Preparing EGM start: draining JTC command stream...")
    _drain_jtc(node)

    node.get_logger().info("Starting EGM (required for all robot motion)...")
    ok, msg = _call_egm_service(node, _EGM_START_SRV, timeout_sec)
    if not ok:
        node.get_logger().error(
            f"EGM start failed: {msg}\n"
            "If the robot is showing 'Idling' on the FlexPendant, EGM has timed out.\n"
            "Run the egm_handler node (or bringup_stack) to re-establish EGM before "
            "running motion scripts."
        )
        return False

    node.get_logger().info(f"EGM active ({msg}). Settling {settle_sec:.1f}s for hardware interface...")
    time.sleep(settle_sec)
    return True


def deactivate_egm(node, timeout_sec: float = 5.0) -> bool:
    """Stop EGM cleanly. Always call this in a finally block after motion.

    Returns True if stop succeeded. A False return is non-fatal but logged
    as a warning since EGM may already be inactive.
    """
    node.get_logger().info("Stopping EGM...")
    ok, msg = _call_egm_service(node, _EGM_STOP_SRV, timeout_sec)
    if ok:
        node.get_logger().info(f"EGM stopped: {msg}")
    else:
        node.get_logger().warn(f"EGM stop returned unexpected result: {msg}")
    return ok
