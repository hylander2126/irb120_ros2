"""Utilities for resolving, configuring, and writing runtime log directories."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters


_OBJECT_PARAMS_PATH = Path(__file__).resolve().parents[1] / "object_params.json"
VALID_OBJECTS = {"box", "heart", "flashlight", "monitor", "soda"}


def load_object_params(object_name: str) -> dict:
    """Load and return the params dict for a named object from object_params.json.

    Raises ValueError for unknown objects, FileNotFoundError if JSON is missing.
    """
    if object_name not in VALID_OBJECTS:
        raise ValueError(
            f"Unknown object '{object_name}'. Must be one of {sorted(VALID_OBJECTS)}."
        )
    with open(_OBJECT_PARAMS_PATH) as f:
        data = json.load(f)
    return data["objects"][object_name]


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


def save_ft_pose_log(
    ft_log: list,
    pose_log: list,
    subdir: str,
    prefix: str,
    obj_pose_log: list | None = None,
    ft_raw_log: list | None = None,
) -> None:
    """Save collected F/T, EE pose, and optional object pose buffers to a timestamped .npz file.
    Also writes a 'most_recent.npz' in the same directory for easy downstream access.

    ft_log: list of [timestamp_s, fx, fy, fz, tx, ty, tz] rows — bias-corrected transformed wrench;
            optional columns 7-13 are ft_link pose [px, py, pz, qx, qy, qz, qw].
    pose_log: list of [timestamp_s, x, y, z, qx, qy, qz, qw] rows (EE pose);
              optional columns 8, 9, and 10 are arc_angle_rad, wrist_pitch_rad, and controller_state_id.
    obj_pose_log: list of [timestamp_s, x, y, z, qx, qy, qz, qw] rows (object pose from detector);
                  optional column 8 is obj_pitch_rad.
    ft_raw_log: list of [timestamp_s, fx, fy, fz, tx, ty, tz, ft_px, ft_py, ft_pz, ft_qx, ft_qy, ft_qz, ft_qw] rows.
    subdir: subdirectory under runtime_logs/ (e.g. "push").
    prefix: filename prefix (e.g. "push_ft_pose").
    """
    if not ft_log and not pose_log and not obj_pose_log:
        print(f"[{prefix}] No F/T or pose data collected — skipping npz save")
        return

    ft_arr = np.array(ft_log, dtype=np.float64) if ft_log else np.empty((0, 7), dtype=np.float64)
    pose_arr = np.array(pose_log, dtype=np.float64) if pose_log else np.empty((0, 8), dtype=np.float64)
    obj_arr = np.array(obj_pose_log, dtype=np.float64) if obj_pose_log else np.empty((0, 8), dtype=np.float64)
    raw_arr = np.array(ft_raw_log, dtype=np.float64) if ft_raw_log else np.empty((0, 14), dtype=np.float64)
    ft_pose_arr = ft_arr if ft_arr.size and ft_arr.shape[1] > 13 else raw_arr

    log_dir = runtime_log_dir(subdir)
    npz_path = log_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    save_kwargs = dict(
        # F/T columns — bias-corrected, transformed to world frame
        ft_time_s=ft_arr[:, 0] if ft_arr.size else np.array([]),
        fx=ft_arr[:, 1]        if ft_arr.size else np.array([]),
        fy=ft_arr[:, 2]        if ft_arr.size else np.array([]),
        fz=ft_arr[:, 3]        if ft_arr.size else np.array([]),
        tx=ft_arr[:, 4]        if ft_arr.size else np.array([]),
        ty=ft_arr[:, 5]        if ft_arr.size else np.array([]),
        tz=ft_arr[:, 6]        if ft_arr.size else np.array([]),
        # EE pose columns
        pose_time_s=pose_arr[:, 0] if pose_arr.size else np.array([]),
        x=pose_arr[:, 1] if pose_arr.size else np.array([]),
        y=pose_arr[:, 2] if pose_arr.size else np.array([]),
        z=pose_arr[:, 3] if pose_arr.size else np.array([]),
        qx=pose_arr[:, 4] if pose_arr.size else np.array([]),
        qy=pose_arr[:, 5] if pose_arr.size else np.array([]),
        qz=pose_arr[:, 6] if pose_arr.size else np.array([]),
        qw=pose_arr[:, 7] if pose_arr.size else np.array([]),
        arc_angle_rad=pose_arr[:, 8] if pose_arr.size and pose_arr.shape[1] > 8 else np.array([]),
        wrist_pitch_rad=pose_arr[:, 9] if pose_arr.size and pose_arr.shape[1] > 9 else np.array([]),
        controller_state_id=pose_arr[:, 10] if pose_arr.size and pose_arr.shape[1] > 10 else np.array([]),
        controller_state_names=np.array(["UNKNOWN", "SQUASH", "LULL", "ARC", "UNARC", "RETRACT"]),
        # Object pose columns (from detector)
        obj_time_s=obj_arr[:, 0] if obj_arr.size else np.array([]),
        obj_x=obj_arr[:, 1] if obj_arr.size else np.array([]),
        obj_y=obj_arr[:, 2] if obj_arr.size else np.array([]),
        obj_z=obj_arr[:, 3] if obj_arr.size else np.array([]),
        obj_qx=obj_arr[:, 4] if obj_arr.size else np.array([]),
        obj_qy=obj_arr[:, 5] if obj_arr.size else np.array([]),
        obj_qz=obj_arr[:, 6] if obj_arr.size else np.array([]),
        obj_qw=obj_arr[:, 7] if obj_arr.size else np.array([]),
        obj_pitch_rad=obj_arr[:, 8] if obj_arr.size and obj_arr.shape[1] > 8 else np.array([]),
        # Legacy raw sensor frame F/T
        ft_raw_time_s=raw_arr[:, 0]  if raw_arr.size else np.array([]),
        fx_raw=raw_arr[:, 1]         if raw_arr.size else np.array([]),
        fy_raw=raw_arr[:, 2]         if raw_arr.size else np.array([]),
        fz_raw=raw_arr[:, 3]         if raw_arr.size else np.array([]),
        tx_raw=raw_arr[:, 4]         if raw_arr.size else np.array([]),
        ty_raw=raw_arr[:, 5]         if raw_arr.size else np.array([]),
        tz_raw=raw_arr[:, 6]         if raw_arr.size else np.array([]),
        # ft_link pose in base/world frame. New logs store this alongside transformed F/T;
        # legacy logs stored it alongside raw F/T.
        ft_px=ft_pose_arr[:, 7]       if ft_pose_arr.size else np.array([]),
        ft_py=ft_pose_arr[:, 8]       if ft_pose_arr.size else np.array([]),
        ft_pz=ft_pose_arr[:, 9]       if ft_pose_arr.size else np.array([]),
        ft_qx=ft_pose_arr[:, 10]      if ft_pose_arr.size else np.array([]),
        ft_qy=ft_pose_arr[:, 11]      if ft_pose_arr.size else np.array([]),
        ft_qz=ft_pose_arr[:, 12]      if ft_pose_arr.size else np.array([]),
        ft_qw=ft_pose_arr[:, 13]      if ft_pose_arr.size else np.array([]),
    )
    np.savez_compressed(npz_path, **save_kwargs)
    print(f"F/T + EE pose + object pose log written to {npz_path}")

    most_recent_path = log_dir / "most_recent.npz"
    np.savez_compressed(most_recent_path, **save_kwargs)
    print(f"most_recent.npz updated at {most_recent_path}")
