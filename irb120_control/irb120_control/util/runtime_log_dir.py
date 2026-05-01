"""Utilities for resolving, configuring, and writing runtime log directories."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters


def resolve_workspace_root() -> Path:
    file_path = Path(__file__).resolve()
    for parent in file_path.parents:
        if (parent / "runtime_logs").exists():
            return parent
        if (parent / "src").is_dir() and (parent / "build").is_dir() and (parent / "install").is_dir():
            return parent
    return file_path.parents[0]


def runtime_log_dir(subdir: str) -> Path:
    """Return workspace_root/runtime_logs/<subdir>, creating it if needed."""
    path = resolve_workspace_root() / "runtime_logs" / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_recorder_output_dir(node, subdir: str, recorder_node_name: str = "camera_hull_recorder", timeout_sec: float = 3.0) -> bool:
    """Set the output_dir parameter on the recorder node before recording starts."""
    target_dir = str(runtime_log_dir(subdir))
    client = node.create_client(SetParameters, f"/{recorder_node_name}/set_parameters")
    if not client.wait_for_service(timeout_sec=timeout_sec):
        node.get_logger().warn(f"set_parameters not available on {recorder_node_name} — video will save to default dir")
        return False

    param = Parameter()
    param.name = "output_dir"
    param.value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=target_dir)

    req = SetParameters.Request()
    req.parameters = [param]
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if not future.done() or future.result() is None:
        node.get_logger().warn("set_parameters call timed out — video will save to default dir")
        return False

    node.get_logger().info(f"Recorder output dir set to: {target_dir}")
    return True


def save_ft_log(ft_log: list, subdir: str, prefix: str) -> None:
    """Save a collected F/T buffer to a timestamped .npz file.

    ft_log: list of [timestamp_s, fx, fy, fz, tx, ty, tz] rows.
    subdir: subdirectory under runtime_logs/ (e.g. "push").
    prefix: filename prefix (e.g. "push_ft").
    """
    if not ft_log:
        print(f"[{prefix}] No F/T data collected — skipping npz save")
        return
    data = np.array(ft_log, dtype=np.float64)
    log_dir = runtime_log_dir(subdir)
    npz_path = log_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez_compressed(
        npz_path,
        time_s=data[:, 0],
        fx=data[:, 1], fy=data[:, 2], fz=data[:, 3],
        tx=data[:, 4], ty=data[:, 5], tz=data[:, 6],
    )
    print(f"F/T log written to {npz_path}")


def save_ft_pose_log(ft_log: list, pose_log: list, subdir: str, prefix: str) -> None:
    """Save collected F/T buffer and end-effector pose buffer to a timestamped .npz file.

    ft_log: list of [timestamp_s, fx, fy, fz, tx, ty, tz] rows.
    pose_log: list of [timestamp_s, x, y, z, qx, qy, qz, qw] rows.
    subdir: subdirectory under runtime_logs/ (e.g. "push").
    prefix: filename prefix (e.g. "push_ft_pose").
    """
    if not ft_log and not pose_log:
        print(f"[{prefix}] No F/T or pose data collected — skipping npz save")
        return

    ft_arr = np.array(ft_log, dtype=np.float64) if ft_log else np.empty((0, 7), dtype=np.float64)
    pose_arr = np.array(pose_log, dtype=np.float64) if pose_log else np.empty((0, 8), dtype=np.float64)

    log_dir = runtime_log_dir(subdir)
    npz_path = log_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez_compressed(
        npz_path,
        # F/T columns
        ft_time_s=ft_arr[:, 0] if ft_arr.size else np.array([]),
        fx=ft_arr[:, 1] if ft_arr.size else np.array([]),
        fy=ft_arr[:, 2] if ft_arr.size else np.array([]),
        fz=ft_arr[:, 3] if ft_arr.size else np.array([]),
        tx=ft_arr[:, 4] if ft_arr.size else np.array([]),
        ty=ft_arr[:, 5] if ft_arr.size else np.array([]),
        tz=ft_arr[:, 6] if ft_arr.size else np.array([]),
        # EE pose columns
        pose_time_s=pose_arr[:, 0] if pose_arr.size else np.array([]),
        x=pose_arr[:, 1] if pose_arr.size else np.array([]),
        y=pose_arr[:, 2] if pose_arr.size else np.array([]),
        z=pose_arr[:, 3] if pose_arr.size else np.array([]),
        qx=pose_arr[:, 4] if pose_arr.size else np.array([]),
        qy=pose_arr[:, 5] if pose_arr.size else np.array([]),
        qz=pose_arr[:, 6] if pose_arr.size else np.array([]),
        qw=pose_arr[:, 7] if pose_arr.size else np.array([]),
    )
    print(f"F/T + EE pose log written to {npz_path}")
