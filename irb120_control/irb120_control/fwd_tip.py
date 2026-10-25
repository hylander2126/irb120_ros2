#!/usr/bin/env python3
"""Fixed-height, fixed-wrist forward tipping normal to the saved pivot edge.

Seek contact, push until axial force falls to 10% of its experienced peak,
then reverse along the same line to the pre-tip pose. Gravity supplies the
restoring object motion; this primitive does not verify that the object untips.

It also backs off early when the push will not give a usable tip
(util/forward_tip_motion.py): 'too_easy' when the contact force stays under
TipConfig.tip_force (the object is close to toppling, or too light to see the
force drop), and 'slide' when, after probe_travel, the object does not follow
the finger back (it slid instead of tilting; the slide force is kept).

    ros2 run irb120_control fwd_tip monitor --episode last
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import sys
import time

import numpy as np
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import WrenchStamped
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener

from irb120_control.controllers.moveit_single_shot import (
    plan_and_execute_joint_goal, plan_and_execute_pose_goal,
)
from irb120_control.controllers.servo_command_publisher import ServoCommandPublisher
from irb120_control.util.egm_client import ensure_egm_active, deactivate_egm
from irb120_control.util.episode import Episode
from irb120_control.util.forward_tip_motion import TIP_STANDOFF, TipConfig, TipMotion, tip_targets
from irb120_control.util.ft_capture import FTCapture
from irb120_control.util.ft_frames import WRENCH_AXES_FRAME, WRENCH_HEADER_FRAME
from irb120_control.util.ft_tare import tare_netft
from irb120_control.util.object_obstacle import object_obstacle
from irb120_control.util.perception_snapshot import (
    begin_motion_stage, latest_object_cloud, lookup, take_snapshot,
)
from irb120_control.util.stall_guard import StallGuard, gc_paused
from irb120_control.util.runtime_log_dir import (
    module_constants, save_ft_pose_log, save_run_metadata, start_recording, stop_recording,
)

BASE_FRAME = 'world'
SERVO_FRAME = 'base_link'
EE_LINK = 'finger_ball_center'
TIP_ORIENTATION = (0., 0., 0., 1.)
HOME_JOINT_POSITIONS = (0.,) * 6
CONTROL_HZ = 100.
SENSOR_TIMEOUT_SEC = .3
LINE_TOLERANCE_M = .003
ORIENTATION_TOLERANCE_RAD = math.radians(3.)
STANDOFF_M = TIP_STANDOFF


def rotate(q, vector):
    """Rotate a vector by an xyzw unit quaternion."""
    q = np.asarray(q, dtype=float)
    v = np.asarray(vector, dtype=float)
    return v + 2. * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)


class ForwardTip(Node):
    def __init__(self, object_name, config, node_name='fwd_tip'):
        super().__init__(node_name)
        self._object = object_name
        self.config = config
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._servo_cmd = ServoCommandPublisher(self, '/servo_node/delta_twist_cmds', SERVO_FRAME)
        self._move_group_client = ActionClient(self, MoveGroup, '/move_action')
        self._pause_client = self.create_client(SetBool, '/servo_node/pause_servo')
        self._wrench_sub = self.create_subscription(
            WrenchStamped, '/netft_data_transformed', self._on_wrench, 10)
        self._force = np.zeros(3)
        self._force_received = None
        self._force_serial = 0
        self._used_serial = 0
        self.motion = None
        self._ft_log = []  # [time_s, f(3), tau(3) in sensor axes, tool0 pose(7), msg stamp, TF stamp]
        self._pose_log = []
        self._motion_log = []
        # Same proprioception as arc_static: raw F/T for the whole run, and joint
        # states + controller state tagged with the phase (approach / tip / home).
        self._ft_capture = FTCapture(self, self._now_s)
        self._tracking_log = []
        self._phase = 'approach'
        self._last_tf_stamp = float('nan')
        self.create_subscription(JointState, '/joint_states', self._record_joints, 10)
        self.create_subscription(JointTrajectoryControllerState,
                                 '/joint_trajectory_controller/controller_state',
                                 self._record_controller, 10)
        self._recording = False
        self._pre = self._ball = self._pivot = self._direction = None
        self._object_xyz = None
        self._world_from_servo = None
        self.failure_reason = None
        self.motion_started = False
        self.pre_ready = False
        self.egm_requested = False
        self._last_report = 0.
        self._stall_guard = StallGuard()  # tells our own freezes from sensor dropouts

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _motion_state(self):
        return self.motion.state if self.motion is not None else None

    def _record_joints(self, msg):
        self._tracking_log.append(dict(time=self._now_s(), source='joint_states',
            stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            phase=self._phase, state=self._motion_state(), names=list(msg.name),
            positions=list(msg.position), velocities=list(msg.velocity)))

    def _record_controller(self, msg):
        row = dict(time=self._now_s(), source='controller_state',
            stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            phase=self._phase, state=self._motion_state(), names=list(msg.joint_names))
        for name in ('reference', 'feedback', 'error', 'output'):
            value = getattr(msg, name)
            row[name] = dict(positions=list(value.positions), velocities=list(value.velocities))
        self._tracking_log.append(row)

    def _transform(self, source):
        tf = self._tf_buffer.lookup_transform(BASE_FRAME, source, rclpy.time.Time())
        stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        if stamp and abs(self._now_s() - stamp) > SENSOR_TIMEOUT_SEC:
            raise ValueError(f'Stale TF for {source}')
        self._last_tf_stamp = stamp
        tr, q = tf.transform.translation, tf.transform.rotation
        pose = np.array([tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w])
        if not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1.) > .01:
            raise ValueError(f'Invalid TF for {source}')
        return pose

    def _on_wrench(self, msg):
        try:
            if not msg.header.frame_id:
                return
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if stamp and abs(self._now_s() - stamp) > SENSOR_TIMEOUT_SEC:
                return
            f, t = msg.wrench.force, msg.wrench.torque
            force = np.array([f.x, f.y, f.z])
            torque = np.array([t.x, t.y, t.z])
            # Logged as received (sensor axes, tool0) with tool0's pose, as arc_static logs
            # it and estimate_params expects; only the controller uses the rotated force.
            logged = np.r_[force, torque]
            # The preprocessor labels its output 'ft_link', a frame the URDF no longer has;
            # the axes are tool0's (util/ft_frames.py). Looking up 'ft_link' dropped every
            # sample, so the freshness watchdog stopped every run with stale_force.
            ft_pose = self._transform(WRENCH_AXES_FRAME)
            tf_stamp = self._last_tf_stamp
            if msg.header.frame_id != BASE_FRAME:
                axes = (WRENCH_AXES_FRAME if msg.header.frame_id == WRENCH_HEADER_FRAME
                        else msg.header.frame_id)
                source_pose = ft_pose if axes == WRENCH_AXES_FRAME else self._transform(axes)
                force = rotate(source_pose[3:], force)
                torque = rotate(source_pose[3:], torque)
            if not np.isfinite(np.r_[force, torque]).all():
                self.failure_reason = 'nonfinite_wrench'
                if self.motion is not None:
                    self.motion.stop(self.failure_reason)
                self._servo_cmd.publish_zero()
                return
        except (TransformException, ValueError):
            return  # Freshness watchdog stops motion if transformed samples cease.
        self._force = force
        self._force_received = time.monotonic()
        self._force_serial += 1
        if self._recording:
            self._ft_log.append([self._now_s(), *logged, *ft_pose, stamp, tf_stamp])

    def set_targets(self, contacts, cloud):
        contact = contacts.get('forward_tip') or {}
        ball, direction, pivot, pre = tip_targets(contact, cloud, STANDOFF_M)
        # Fallback pre-pose clearing the whole bounding box, if the close one does not plan.
        box_pre = tip_targets(contact, cloud, STANDOFF_M, clearance='box')[3]
        transform = lookup(self, BASE_FRAME, SERVO_FRAME, timeout_sec=10.)
        if transform is None:
            raise ValueError('No world <- base_link transform')
        rotation, translation = transform
        self._world_from_servo = rotation
        self._ball, self._pivot, self._pre, self._box_pre = [
            rotation @ p + translation for p in (ball, pivot, pre, box_pre)]
        self.pre_clearance = 'slabs'

        self._direction = rotation @ direction
        if abs(self._direction[2]) > 1e-3:
            raise ValueError('Tipping-edge normal is not horizontal in world')
        self._direction[2] = 0.
        self._direction /= np.linalg.norm(self._direction)
        self._object_xyz = cloud
        self.get_logger().info(
            f'Forward tip: pre={self._pre.round(4)}, direction={self._direction.round(4)}, '
            f'pivot={self._pivot.round(4)}; fixed height and orientation')

    def set_servo_paused(self, paused):
        if not self._pause_client.wait_for_service(timeout_sec=2.):
            self.get_logger().error('Servo pause service unavailable')
            return False
        future = self._pause_client.call_async(SetBool.Request(data=paused))
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.)
        result = future.result() if future.done() else None
        return result is not None and result.success

    def approach(self):
        """To the close pre-pose against the sliced object; if that does not plan (e.g. the
        F/T sensor would sit in a lower, wider slice), to the whole-box pre-pose against the
        whole box, as before."""
        diagnostics = {}
        with object_obstacle(self, self._object_xyz, slabs=True):
            if plan_and_execute_pose_goal(
                    self, self._move_group_client, base_frame=BASE_FRAME, ee_link=EE_LINK,
                    target_position=self._pre.tolist(), target_orientation=TIP_ORIENTATION,
                    velocity_scale=.1, acceleration_scale=.1, diagnostics=diagnostics):
                return True
        # Only a plan that failed falls back; a move that was attempted and failed does not.
        if 'execution_success' in diagnostics or np.linalg.norm(self._box_pre - self._pre) < 1e-4:
            return False
        self.get_logger().warn(
            f'Close pre-tip {self._pre.round(4)} did not plan; using the whole-box pre-tip '
            f'{self._box_pre.round(4)} ({1000 * np.linalg.norm(self._box_pre - self._ball):.0f} mm back)')
        self._pre, self.pre_clearance = self._box_pre, 'box'
        with object_obstacle(self, self._object_xyz):
            return plan_and_execute_pose_goal(
                self, self._move_group_client, base_frame=BASE_FRAME, ee_link=EE_LINK,
                target_position=self._pre.tolist(), target_orientation=TIP_ORIENTATION,
                velocity_scale=.1, acceleration_scale=.1)

    def home(self):
        with object_obstacle(self, self._object_xyz):
            return plan_and_execute_joint_goal(
                self, self._move_group_client, joint_positions=HOME_JOINT_POSITIONS,
                velocity_scaling_factor=.1, acceleration_scaling_factor=.1)

    def _stop(self, reason):
        self.failure_reason = reason
        if self.motion is not None:
            self.motion.stop(reason)
        self._servo_cmd.publish_zero()
        self.get_logger().error(f'Forward tip stopped: {reason}')

    def _hold_after_stall(self, what):
        # This tick ran late: the process froze and the sensor messages are still queued.
        self.get_logger().warn(f'Control loop stalled in {self.motion.state}: {what} -- holding one tick')
        self._servo_cmd.publish_zero()

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
        line_error = np.linalg.norm(delta - progress * self._direction)
        q_dot = abs(float(pose[3:] @ self._start_pose[3:]))
        angle_error = 2. * math.acos(float(np.clip(q_dot, 0., 1.)))
        if line_error > LINE_TOLERANCE_M or angle_error > ORIENTATION_TOLERANCE_RAD:
            self._stop('height_lateral_or_orientation_drift')
            return
        axial = float(self._force @ self._direction)
        before = self.motion.state
        speed = self.motion.step(now, progress, axial, float(np.linalg.norm(self._force)),
                                 fresh=self._force_serial != self._used_serial)
        self._used_serial = self._force_serial
        # Fixed world height, fixed orientation: no commanded Z or angular motion.
        velocity = self._world_from_servo.T @ (speed * self._direction)
        if not self._servo_cmd.publish_twist(*velocity, state=self.motion.state):
            self._stop('servo_subscriber_lost')
        # Columns as arc_static: ..., arc angle, wrist pitch, state id (0 = none), TF stamp.
        self._pose_log.append([self._now_s(), *pose, np.nan, np.nan, 0, pose_tf_stamp])
        self._motion_log.append(dict(time_s=self._now_s(), state=self.motion.state,
            progress_m=progress, axial_force_n=axial, peak_force_n=self.motion.peak_force,
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
                f'reverse at {self.config.reverse_fraction * self.motion.peak_force:.2f}N '
                f'({self.motion.reversal_reason or self.motion.failure_reason or "running"})')

    def run_motion(self):
        self._start_pose = self._transform(EE_LINK)
        if (np.linalg.norm(self._start_pose[:3] - self._pre) > LINE_TOLERANCE_M
                or 2. * math.acos(float(np.clip(abs(self._start_pose[3:] @ TIP_ORIENTATION), 0., 1.)))
                > ORIENTATION_TOLERANCE_RAD):
            raise ValueError('Measured pre-tip pose does not match the approach target')
        expected = float((self._ball - self._start_pose[:3]) @ self._direction)
        self.motion = TipMotion(self.config, expected, time.monotonic())
        self._used_serial = self._force_serial  # only post-tare, fresh samples arm contact
        self._phase = 'tip'
        self._recording = True
        self._stall_guard.reset()
        timer = self.create_timer(1. / CONTROL_HZ, self._tick)
        try:
            with gc_paused():
                while rclpy.ok() and not self.motion.done:
                    rclpy.spin_once(self, timeout_sec=.05)
        finally:
            self.destroy_timer(timer)
            self._servo_cmd.publish_zero()
            self._recording = False

    def execute(self):
        """Approach and run; cleanup and episode persistence are owned by main."""
        self.egm_requested = True
        if not ensure_egm_active(self):
            raise RuntimeError('EGM startup failed')
        if not self.set_servo_paused(True):
            raise RuntimeError('Servo pause unconfirmed; not approaching')
        self.motion_started = True
        if not self.approach():
            raise RuntimeError('Pre-tip approach failed')
        self.pre_ready = True
        self.get_logger().warn('Check the finger at the pre-tip pose before forward tipping.')
        try:
            reply = input("Press Enter to tip, or type 'q' to abort: ").strip().lower()
        except EOFError:
            reply = 'q'
        if reply == 'q':
            self.failure_reason = 'operator_aborted'
            return
        if not tare_netft(self):
            raise RuntimeError('F/T tare failed')
        self._force_received = None
        deadline = time.monotonic() + 3.
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.05)
            if self._force_received is not None and self._servo_cmd.has_subscribers():
                break
        if self._force_received is None or not self._servo_cmd.has_subscribers():
            raise RuntimeError('No fresh transformed force or Servo subscriber')
        if not self.set_servo_paused(False):
            raise RuntimeError('Servo resume failed')
        self.run_motion()


def main(args=None, *, node_type=ForwardTip, primitive="fwd_tip"):
    raw = list(sys.argv[1:] if args is None else args)
    ros_index = raw.index('--ros-args') if '--ros-args' in raw else len(raw)
    parser = argparse.ArgumentParser(description=(__doc__ if primitive == 'fwd_tip' else
        'Forward pivot-axis arc with fixed orientation; return at 10% of peak pivot torque.'))
    parser.add_argument('object', help='Object/episode label; no ground-truth parameters required')
    parser.add_argument('--episode', default='new', help='new, last, or episode path')
    parser.add_argument('--quality', choices=['h264', 'lossless'], default='h264')
    parser.add_argument('--speed', type=float, default=.005, help='Load/tip/return speed in m/s; the contact seek runs at 1.5x')
    parser.add_argument('--max-travel', type=float, default=.080, help='Travel cap after contact, metres')
    parser.add_argument('--reverse-fraction', type=float, default=.10, help='Fraction of peak pivot torque' if primitive == 'fwd_arc' else 'Fraction of peak axial force')
    parser.add_argument('--force-limit', type=float, default=20., help='Resultant force stop limit, N')
    parser.add_argument('--tip-force', type=float, default=.30,
                        help="Weakest contact force worth tipping on; below it back off as 'too_easy', N")
    if primitive == 'fwd_arc':
        parser.add_argument('--max-angle', type=float, default=30., help='Maximum arc angle, degrees')
    parsed = parser.parse_args(raw[:ros_index])
    if primitive == 'fwd_arc' and (not math.isfinite(parsed.max_angle) or not 0 < parsed.max_angle <= 90):
        parser.error('--max-angle must be in (0, 90] degrees')
    try:
        config = TipConfig(speed=parsed.speed, seek_speed=1.5 * parsed.speed, max_travel=parsed.max_travel,
                           reverse_fraction=parsed.reverse_fraction, force_limit=parsed.force_limit,
                           tip_force=parsed.tip_force)
    except ValueError as exc:
        parser.error(str(exc))
    rclpy.init(args=raw[ros_index:])
    node = node_type(parsed.object, config)
    if primitive == 'fwd_arc':
        node.max_angle = math.radians(parsed.max_angle)
    episode = stage = None
    home_ok = False
    recording_requested = False
    try:
        episode = Episode.resolve(parsed.episode, parsed.object, node, allow_unconfigured_object=True)
        stage, contacts = begin_motion_stage(node, episode, 'forward_arc' if primitive == 'fwd_arc' else 'forward_tip')
        if stage is None:
            raise RuntimeError('No usable perception contacts; not moving')
        node.set_targets(contacts, latest_object_cloud(episode))
        recording_requested = True
        if not start_recording(node, str(stage.path), quality=parsed.quality):
            raise RuntimeError('Camera recording failed')
        node.execute()
    except KeyboardInterrupt:
        node._stop('interrupted')
    except Exception as exc:
        node._stop(str(exc))
    finally:
        try:
            node._servo_cmd.publish_zero()
            if node.motion_started and rclpy.ok():
                paused = node.set_servo_paused(True)
                # Never plan home from unresolved contact or a failed approach.
                clear_to_home = node.pre_ready and (node.motion is None or node.motion.returned)
                if paused and clear_to_home:
                    node._phase = 'home'
                    home_ok = node.home()
                else:
                    node.get_logger().warn('Home skipped: return to pre-tip not verified or Servo pause failed.')
        except Exception as exc:
            node.failure_reason = f'cleanup: {exc}'
            node.get_logger().error(node.failure_reason)
        finally:
            try:
                if node.egm_requested and rclpy.ok():
                    deactivate_egm(node)
            finally:
                node._servo_cmd.close()
        try:
            if recording_requested:
                stop_recording(node)
            completed = bool(node.motion and node.motion.completed)
            success = completed and home_ok and node.failure_reason is None
            if stage is not None:
                def vector(value):
                    return None if value is None else value.tolist()
                result = dict(completed=completed, home_ok=home_ok,
                    failure_reason=node.failure_reason or (node.motion.failure_reason if node.motion else None),
                    reversal_reason=node.motion.reversal_reason if node.motion else None,
                    peak_axial_force_n=node.motion.peak_force if node.motion else 0.,
                    contact_force_n=node.motion.contact_force if node.motion else None,
                    slide_force_n=node.motion.slide_force if node.motion else None,
                    slide_travel_m=node.motion.slide_travel if node.motion else None,
                    slide_probe=node.motion.probe if node.motion else None,
                    pre_tip=vector(node._pre), pre_tip_clearance=getattr(node, 'pre_clearance', None),
                    direction=vector(node._direction),
                    commanded_pivot=vector(node._pivot),
                    home_verification=getattr(node, '_last_joint_goal_verification', None))
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                try:
                    save_ft_pose_log(node._ft_log, node._pose_log, str(stage.path), primitive,
                                     ft_raw_log=node._ft_capture.raw,
                                     timestamp=ts, commanded_pivot=node._pivot)
                    (stage.path / f'{primitive}_motion_{ts}.json').write_text(
                        json.dumps(node._motion_log, indent=2) + '\n')
                    (stage.path / f'{primitive}_diagnostics_{ts}.json').write_text(
                        json.dumps(dict(tracking=node._tracking_log), separators=(',', ':')) + '\n')
                    save_run_metadata(str(stage.path), primitive, ts, {
                        **module_constants(globals()), **asdict(config), **result,
                        **node._ft_capture.metadata(),
                        'object': parsed.object, 'pose_frame': BASE_FRAME,
                        'completion_means': 'load-drop reversal and finger return; object restoration unverified',
                        **getattr(node, 'trajectory_metadata', {}),
                    })
                except Exception as exc:
                    success = False
                    result['log_error'] = str(exc)
                    node.get_logger().error(f'Failed saving forward-tip logs: {exc}')
                stage.end('ok' if success else 'failed' if node.motion_started else 'aborted', **result)
                if home_ok and rclpy.ok():
                    take_snapshot(node, episode, label='after forward tip')
            if episode is not None:
                print(episode.summary())
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
