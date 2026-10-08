"""Read-only approach evidence, flushed before motion and throughout execution."""
import json
import math
import time
from pathlib import Path

import numpy as np


def position_error_mm(position, target):
    delta = (np.asarray(position, dtype=float)-np.asarray(target, dtype=float))*1000
    return dict(actual_minus_target_mm=delta.tolist(), distance_mm=float(np.linalg.norm(delta)))


def json_finite(value):
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_finite(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


class ApproachDiagnostics:
    def __init__(self, node, path, target, frame, link):
        self.node, self.path = node, Path(path)
        self.target, self.frame, self.link = list(target), frame, link
        self.timer = None

    def write(self, event, **fields):
        row = dict(event=event, unix_time_s=time.time(), ros_time_s=self.node._now_s(), **fields)
        with self.path.open('a') as stream:
            stream.write(json.dumps(json_finite(row), allow_nan=False, default=lambda v: v.tolist())+'\n')

    def sample(self, event='pose_sample'):
        from rclpy.time import Time
        row = {}
        try:
            tf = self.node._tf_buffer.lookup_transform(self.frame, self.link, Time())
            p, q = tf.transform.translation, tf.transform.rotation
            stamp = tf.header.stamp.sec+tf.header.stamp.nanosec*1e-9
            row['tf'] = dict(position=[p.x,p.y,p.z], quaternion_xyzw=[q.x,q.y,q.z,q.w],
                             stamp_s=stamp, age_s=self.node._now_s()-stamp,
                             **position_error_mm([p.x,p.y,p.z], self.target))
        except Exception as exc:
            row['tf'] = dict(available=False, reason=str(exc))
        # Preserve measured joints and the controller's reference/error separately.
        for source in ('joint_states', 'controller_state'):
            latest = next((r for r in reversed(self.node._tracking_log) if r['source']==source), None)
            row[source] = latest
        self.write(event, **row)
        return row

    def measured_fk(self, event):
        """Use MoveIt's own robot model when the EE TF stream is unavailable."""
        import rclpy
        from moveit_msgs.srv import GetPositionFK
        latest = next((r for r in reversed(self.node._tracking_log)
                       if r['source']=='joint_states'), None)
        if latest is None:
            self.write(event, available=False, reason='No measured joint state received')
            return
        client = self.node.create_client(GetPositionFK, '/compute_fk')
        future = None
        try:
            if not client.wait_for_service(timeout_sec=.5):
                self.write(event, available=False, reason='/compute_fk unavailable')
                return
            request = GetPositionFK.Request()
            request.header.frame_id = self.frame
            request.fk_link_names = [self.link]
            request.robot_state.joint_state.name = latest['names']
            request.robot_state.joint_state.position = latest['positions']
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=1.)
            if not future.done() or future.result() is None:
                self.write(event, available=False, reason='/compute_fk timeout')
                return
            result = future.result()
            if result.error_code.val != 1 or not result.pose_stamped:
                self.write(event, available=False, error_code=result.error_code.val)
                return
            pose = result.pose_stamped[0].pose
            position = [pose.position.x,pose.position.y,pose.position.z]
            error = position_error_mm(position, self.target)
            self.write(event, available=True, frame=result.pose_stamped[0].header.frame_id,
                       position=position, quaternion_xyzw=[pose.orientation.x,pose.orientation.y,
                       pose.orientation.z,pose.orientation.w], measured_joints=latest,
                       joint_receipt_age_s=self.node._now_s()-latest['time'], **error)
            self.node.get_logger().info(
                f'[approach diagnostic] {event}: actual-target XYZ mm '
                f'{np.round(error["actual_minus_target_mm"], 2).tolist()}, '
                f'distance {error["distance_mm"]:.2f} mm (FK from measured joints)')
        except Exception as exc:
            self.write(event, available=False, reason=str(exc))
        finally:
            if future is not None and not future.done():
                future.cancel()
            self.node.destroy_client(client)

    def start(self):
        import rclpy
        rclpy.spin_once(self.node, timeout_sec=.05)
        self.sample('before_approach')
        self.measured_fk('before_approach_fk')
        self.timer = self.node.create_timer(.2, self.sample)

    def finish(self, success, action):
        if self.timer is not None:
            self.node.destroy_timer(self.timer)
        self.sample('after_approach')
        self.measured_fk('after_approach_fk')
        self.write('approach_result', moveit_reported_success=success, action=action)
