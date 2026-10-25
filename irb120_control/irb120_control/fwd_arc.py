#!/usr/bin/env python3
"""Forward tipping along a pivot-axis arc, with fixed finger orientation.

ros2 run irb120_control fwd_arc OBJECT --episode last --max-angle 30
The arc reduces face sliding; its commanded angle is not measured object tilt.
"""
import math
from dataclasses import replace
import time
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import TransformException
from irb120_control.fwd_tip import (
    ForwardTip, main as run_primitive, rotate, BASE_FRAME, EE_LINK,
    SENSOR_TIMEOUT_SEC, LINE_TOLERANCE_M, ORIENTATION_TOLERANCE_RAD,
)
from irb120_control.util.stall_guard import StallGuard
from irb120_control.util.ft_frames import WRENCH_AXES_FRAME
from irb120_control.util.forward_arc_geometry import PivotArc, pivot_torque


class ForwardArc(ForwardTip):
    def __init__(self, object_name, config):
        super().__init__(object_name, config, node_name='fwd_arc')
        self.max_angle = math.radians(30.)
        self._arc = None
        self._torque = np.zeros(3)
        self._wrench_origin = np.zeros(3)
        self._arc_pub = self.create_publisher(Marker, '/fwd_arc/trajectory',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.trajectory_metadata = dict(trajectory='pivot_arc', orientation='fixed',
                                        reversal_quantity='pivot_axis_torque')

    def _on_wrench(self, msg):
        serial = self._force_serial
        super()._on_wrench(msg)
        if self._force_serial == serial:
            return
        try:
            pose = self._transform(WRENCH_AXES_FRAME)
            # Match the estimator: NetFT moment origin is +82.25mm along tool0 X.
            self._wrench_origin = pose[:3] + rotate(pose[3:], [.08225, 0., 0.])
            t = msg.wrench.torque
            moment = np.array([t.x, t.y, t.z])
            if msg.header.frame_id != BASE_FRAME:
                axes_pose = pose if msg.header.frame_id in ('ft_link', WRENCH_AXES_FRAME) else self._transform(msg.header.frame_id)
                moment = rotate(axes_pose[3:], moment)
            self._torque = moment
        except (TransformException, ValueError):
            self._force_received = None  # invalidate this sample as a complete wrench

    def set_targets(self, contacts, cloud):
        super().set_targets(contacts, cloud)
        axis = self._world_from_servo @ np.asarray(contacts['forward_tip']['geometry']['axis'])
        self._nominal_arc = PivotArc(self._pivot, axis, self._ball, self._direction)
        # max_angle is the only cap; the travel limit sits just past it so it never fires first
        # (a fixed 80 mm was 9 deg on the monitor's 0.5 m radius, 20261025_155858).
        self.config = replace(self.config, max_travel=self._nominal_arc.radius * self.max_angle + .01)
        self.trajectory_metadata.update(pivot_axis=self._nominal_arc.axis.tolist(),
            selected_ball=self._ball.tolist(), radius_m=self._nominal_arc.radius,
            max_angle_rad=self.max_angle)
        self._draw_arc(self._nominal_arc)
        self.get_logger().info('Arc preview: /fwd_arc/trajectory (world); fixed orientation')

    def _draw_arc(self, arc):
        marker = Marker()
        marker.header.frame_id = BASE_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id = 'forward_arc', 0
        marker.type, marker.action = Marker.LINE_STRIP, Marker.ADD
        marker.pose.orientation.w = 1.
        marker.scale.x = .002
        marker.color.r, marker.color.g, marker.color.a = 1., .55, 1.
        points = [arc.point(a) for a in np.linspace(0., self.max_angle, 101)]
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
        self._arc_pub.publish(marker)
        self.trajectory_metadata['arc_preview_world'] = [p.tolist() for p in points]

    def _tick(self):
        if self.motion.done:
            self._servo_cmd.publish_zero()
            return
        late = self._stall_guard.tick()
        now = time.monotonic()
        force_age = float('inf') if self._force_received is None else now - self._force_received
        if force_age > SENSOR_TIMEOUT_SEC:
            if StallGuard.defer(late, force_age):
                self._hold_after_stall(f'force {force_age * 1000:.0f} ms old')
                return
            self._stop('stale_force')
            return
        try:
            pose = self._transform(EE_LINK)
            pose_tf_stamp = self._last_tf_stamp   # this lookup's stamp, before others overwrite it
        except (TransformException, ValueError):
            if late:  # TF is queued behind the freeze too; a real dropout fails the next tick
                self._hold_after_stall('pose missing or stale')
                return
            self._stop('missing_or_stale_pose')
            return
        delta = pose[:3] - self._start_pose[:3]
        progress = float(delta @ self._direction)
        tangent = self._direction
        target = self._start_pose[:3] + progress * self._direction
        arc_angle = 0.
        if self._arc is not None:
            arc_angle = self._arc.angle(pose[:3])
            if arc_angle >= 0:
                progress = self._arc_start + self._arc.radius * arc_angle
                tangent = self._arc.tangent(arc_angle)
                target = self._arc.point(arc_angle)
        line_error = np.linalg.norm(pose[:3] - target)
        q_dot = abs(float(pose[3:] @ self._start_pose[3:]))
        angle_error = 2. * math.acos(float(np.clip(q_dot, 0., 1.)))
        if line_error > LINE_TOLERANCE_M or angle_error > ORIENTATION_TOLERANCE_RAD:
            self._stop('arc_path_or_orientation_drift')
            return
        axial = float(self._force @ self._direction)
        before = self.motion.state
        torque = pivot_torque(self._force, self._torque, self._wrench_origin,
                              self._pivot, self._nominal_arc.axis)
        if self._arc is not None and arc_angle >= self.max_angle and self.motion.state != 'RETURN':
            self.motion.reverse(now, 'angle_limit')
        speed = self.motion.step(now, progress, axial, float(np.linalg.norm(self._force)),
                                 fresh=self._force_serial != self._used_serial, reversal_signal=torque)
        self._used_serial = self._force_serial
        if self._arc is None and self.motion.contact_progress is not None:
            self._arc = PivotArc(self._pivot, self._nominal_arc.axis, pose[:3], self._direction)
            self._arc_start = progress
            self.trajectory_metadata['actual_contact_ball'] = pose[:3].tolist()
            self._draw_arc(self._arc)
        if self.motion.state == 'SEEK' and self.motion.contact_progress is None:
            self._arc = None
        self.trajectory_metadata['peak_pivot_torque_nm'] = self.motion.peak_signal
        correction = 2. * (target - pose[:3]) if speed else np.zeros(3)
        velocity = self._world_from_servo.T @ (speed * tangent + correction)
        if not self._servo_cmd.publish_twist(*velocity, state=self.motion.state):
            self._stop('servo_subscriber_lost')
        # Columns as arc_static: ..., arc angle, wrist pitch, state id (0 = none), TF stamp.
        self._pose_log.append([self._now_s(), *pose, np.nan, np.nan, 0, pose_tf_stamp])
        self._motion_log.append(dict(time_s=self._now_s(), state=self.motion.state,
            progress_m=progress, arc_angle_rad=arc_angle, pivot_torque_nm=torque,
            peak_pivot_torque_nm=self.motion.peak_signal, axial_force_n=axial, peak_force_n=self.motion.peak_force,
            speed_m_s=speed, height_error_m=float(delta[2]),
            line_error_m=float(line_error), orientation_error_rad=angle_error))
        if before == 'PROBE' and self.motion.state != 'PROBE':
            p = self.motion.probe
            self.get_logger().warn(
                f"Slide probe: {p['held']:.2f} N of {p['before']:.2f} N stayed on backing off -> "
                + ('tilted, tipping on' if p['followed'] else 'the object SLID, returning'))
        if before != self.motion.state or now - self._last_report >= 1.:
            self._last_report = now
            self.get_logger().info(
                f'{self.motion.state}: travel={progress * 1000:.1f}mm, '
                f'force={axial:.2f}N, peak={self.motion.peak_force:.2f}N, '
                f'torque={torque:.4f}Nm, reverse at {self.config.reverse_fraction * self.motion.peak_signal:.4f}Nm '
                f'({self.motion.reversal_reason or self.motion.failure_reason or "running"})')


def main(args=None):
    return run_primitive(args, node_type=ForwardArc, primitive='fwd_arc')


if __name__ == '__main__':
    raise SystemExit(main())
