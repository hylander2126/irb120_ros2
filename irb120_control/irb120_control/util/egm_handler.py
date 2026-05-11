#!/usr/bin/env python3
import math
import signal
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from abb_rapid_sm_addin_msgs.srv import GetEGMSettings, SetEGMSettings
from abb_robot_msgs.srv import TriggerWithResultCode

_JTC_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


class EGMHandler(Node):
    def __init__(self):
        super().__init__("egm_handler")

        self.declare_parameter("rws_service_prefix", "/rws_client")
        self.declare_parameter("task", "T_ROB1")
        self.declare_parameter("startup_service_timeout_sec", 30.0)
        self.declare_parameter("shutdown_service_timeout_sec", 5.0)

        self.declare_parameter("max_speed_dev_rad", 1.5)
        self.declare_parameter("comm_timeout", 5.0)
        self.declare_parameter("ramp_in_time", 2.0)
        self.declare_parameter("ramp_out_time", 0.25)
        self.declare_parameter("pos_corr_gain", 1.0)

        self.rws_prefix = self.get_parameter("rws_service_prefix").value.rstrip("/")
        self.task = self.get_parameter("task").value
        self.startup_service_timeout_sec = float(self.get_parameter("startup_service_timeout_sec").value)
        self.shutdown_service_timeout_sec = float(self.get_parameter("shutdown_service_timeout_sec").value)
        self._shutdown_done = False
        self._executor: SingleThreadedExecutor | None = None

        self.egm_stop_srv = f"{self.rws_prefix}/stop_egm"
        self.egm_start_srv = f"{self.rws_prefix}/start_egm_joint"
        self.get_settings_srv = f"{self.rws_prefix}/get_egm_settings"
        self.set_settings_srv = f"{self.rws_prefix}/set_egm_settings"

        self._ready_pub = self.create_publisher(Bool, "~/ready", 1)
        self._jtc_client = ActionClient(
            self, FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory"
        )
        self.get_logger().info(f"Using RWS service prefix: {self.rws_prefix}")



    def _spin_future(self, future, timeout_sec: float) -> bool:
        end = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < end:
            if not rclpy.ok():
                break
            if self._executor is not None:
                self._executor.spin_once(timeout_sec=0.05)
            else:
                rclpy.spin_once(self, timeout_sec=0.05)
        return future.done()
    def _wait_for_service(self, client, service_name, timeout_sec):
        end_time = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < end_time:
            if client.wait_for_service(timeout_sec=0.2):
                return True
        self.get_logger().warn(f"Service unavailable: {service_name}")
        return False

    def _call_trigger(self, service_name, timeout_sec=5.0, sleep_after=0.5):
        client = self.create_client(TriggerWithResultCode, service_name)
        if not self._wait_for_service(client, service_name, timeout_sec):
            return None, "service unavailable"

        future = client.call_async(TriggerWithResultCode.Request())
        if not self._spin_future(future, timeout_sec):
            self.get_logger().warn(f"Service call timed out: {service_name}")
            return None, "call timed out"
        if future.result() is None:
            self.get_logger().warn(f"Service call failed: {service_name}")
            return None, "call failed"

        response = future.result()
        if sleep_after > 0.0:
            time.sleep(sleep_after)
        return response.result_code, response.message

    def _call_trigger_retry(self, service_name, attempts=3, timeout_sec=5.0, sleep_after=0.25):
        last_code, last_msg = None, "not called"
        for attempt in range(1, attempts + 1):
            code, msg = self._call_trigger(service_name, timeout_sec=timeout_sec, sleep_after=sleep_after)
            last_code, last_msg = code, msg
            if code == 1:
                return code, msg
            self.get_logger().warn(
                f"{service_name} attempt {attempt}/{attempts} failed: code={code}, msg='{msg}'"
            )
            time.sleep(0.2)
        return last_code, last_msg

    def _wait_for_startup_services(self):
        required = [
            (TriggerWithResultCode, self.egm_stop_srv),
            (TriggerWithResultCode, self.egm_start_srv),
            (GetEGMSettings, self.get_settings_srv),
            (SetEGMSettings, self.set_settings_srv),
        ]
        for srv_type, name in required:
            client = self.create_client(srv_type, name)
            if not self._wait_for_service(client, name, self.startup_service_timeout_sec):
                return False
        return True




    def _set_egm_settings(self):
        get_client = self.create_client(GetEGMSettings, self.get_settings_srv)
        set_client = self.create_client(SetEGMSettings, self.set_settings_srv)

        if not self._wait_for_service(get_client, self.get_settings_srv, 5.0):
            return None, "get_egm_settings unavailable"
        if not self._wait_for_service(set_client, self.set_settings_srv, 5.0):
            return None, "set_egm_settings unavailable"

        get_req = GetEGMSettings.Request()
        get_req.task = self.task
        get_future = get_client.call_async(get_req)
        if not self._spin_future(get_future, 5.0) or get_future.result() is None:
            return None, "get_egm_settings call failed"

        settings = get_future.result().settings
        settings.activate.max_speed_deviation = math.degrees(float(self.get_parameter("max_speed_dev_rad").value))
        settings.setup_uc.comm_timeout = float(self.get_parameter("comm_timeout").value)
        settings.run.ramp_in_time = float(self.get_parameter("ramp_in_time").value)
        settings.stop.ramp_out_time = float(self.get_parameter("ramp_out_time").value)
        settings.run.pos_corr_gain = float(self.get_parameter("pos_corr_gain").value)

        set_req = SetEGMSettings.Request()
        set_req.task = self.task
        set_req.settings = settings

        set_future = set_client.call_async(set_req)
        if not self._spin_future(set_future, 5.0) or set_future.result() is None:
            return None, "set_egm_settings call failed"

        response = set_future.result()
        return response.result_code, response.message




    def shutdown_sequence(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True

        print("[egm_handler] Shutdown requested. Stopping EGM and RAPID.", flush=True)

        if not rclpy.ok():
            print("[egm_handler] rclpy context already shut down — cannot send stop commands.", flush=True)
            return

        shutdown_executor = SingleThreadedExecutor()
        shutdown_executor.add_node(self)
        self._executor = shutdown_executor
        try:
            # Leave RAPID running — the StateMachine is always-on and must not
            # be stopped from ROS. Just close the EGM session cleanly.
            code, msg = self._call_trigger_retry(
                self.egm_stop_srv,
                attempts=4,
                timeout_sec=self.shutdown_service_timeout_sec,
                sleep_after=0.5,
            )
            print(f"[egm_handler] stop_egm -> code={code}, msg='{msg}'", flush=True)
        finally:
            self._executor = None
            shutdown_executor.shutdown()

    def _drain_jtc_commands(self, drain_sec: float = 0.35) -> None:
        """Wait for the JTC command stream to go silent before activating EGM.

        cmd_timeout in irb120_controllers.yaml is 0.1 s — this must exceed it
        so the hardware interface wakes up with no commands in-flight.
        """
        self.get_logger().info(f"Draining JTC command stream ({drain_sec:.2f}s)...")
        time.sleep(drain_sec)

    def _wait_for_jtc(self, timeout_sec: float = 30.0) -> bool:
        """Block until the JTC action server is live.

        The JTC spawner races against egm_handler.  We must wait for the JTC
        to finish activating before we send any hold trajectory.
        """
        self.get_logger().info(f"Waiting for JTC action server (up to {timeout_sec:.0f}s)...")
        end = time.monotonic() + timeout_sec
        while time.monotonic() < end:
            if self._jtc_client.wait_for_server(timeout_sec=1.0):
                self.get_logger().info("JTC action server is ready")
                return True
            if not rclpy.ok():
                return False
        self.get_logger().error("JTC action server never became available")
        return False

    def _send_hold_trajectory(self, timeout_sec: float = 5.0) -> None:
        """Send a zero-displacement trajectory to seed the JTC from actual state.

        On first activation the JTC may not have a commanded position yet.
        Sending a hold-in-place goal forces it to read actual joint positions
        from hardware state and lock onto them, preventing any phantom move.
        """
        # Grab one JointState message to read actual positions
        js: JointState | None = None
        deadline = time.monotonic() + timeout_sec

        def _js_cb(msg: JointState) -> None:
            nonlocal js
            js = msg

        sub = self.create_subscription(JointState, "/joint_states", _js_cb, 1)
        while js is None and time.monotonic() < deadline:
            if self._executor is not None:
                self._executor.spin_once(timeout_sec=0.05)
            else:
                rclpy.spin_once(self, timeout_sec=0.05)
        self.destroy_subscription(sub)

        if js is None:
            self.get_logger().warn("Could not read /joint_states — skipping hold trajectory")
            return

        # Build a position map from the JointState message
        pos_map = dict(zip(js.name, js.position))
        positions = [pos_map.get(j, 0.0) for j in _JTC_JOINTS]

        traj = JointTrajectory()
        traj.joint_names = _JTC_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.velocities = [0.0] * len(_JTC_JOINTS)
        pt.accelerations = [0.0] * len(_JTC_JOINTS)
        # 0.5 s gives the JTC time to accept the goal and latch the command
        pt.time_from_start = Duration(sec=0, nanosec=500_000_000)
        traj.points = [pt]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info("Sending hold-in-place trajectory to seed JTC command state...")
        send_future = self._jtc_client.send_goal_async(goal)
        self._spin_future(send_future, timeout_sec)
        if send_future.done() and send_future.result() is not None:
            self.get_logger().info("Hold trajectory accepted — JTC command state seeded from hardware")
        else:
            self.get_logger().warn("Hold trajectory send failed — proceeding anyway")

    def startup_sequence(self):
        if not self._wait_for_startup_services():
            self.get_logger().error("Required startup services unavailable. Skipping EGM startup sequence.")
            return

        # The StateMachine RAPID program starts automatically when the IRC5
        # boots and sits idle — we must never stop/restart it. Doing so while
        # the controller is in auto mode crashes the FlexPendant.
        # All we need to do is close any stale EGM session then activate a
        # fresh one on the already-running StateMachine.
        code, msg = self._call_trigger(self.egm_stop_srv, sleep_after=2.0)
        self.get_logger().info(f"stop_egm (cleanup) -> code={code}, msg='{msg}'")

        code, msg = self._set_egm_settings()
        self.get_logger().info(f"set_egm_settings -> code={code}, msg='{msg}'")

        # Start EGM — the hardware interface connects to the robot here.
        # The JTC spawner (in abb_control launch) cannot activate until EGM
        # is live, so this ordering is correct: EGM first, then JTC activates.
        code, msg = self._call_trigger(self.egm_start_srv, sleep_after=1.0)
        self.get_logger().info(f"start_egm_joint -> code={code}, msg='{msg}'")

        # Wait for the JTC to finish activating on the now-live hardware interface,
        # then immediately send a hold-in-place trajectory.  This seeds the JTC's
        # internal command state from actual joint positions before any motion
        # script runs, preventing a phantom move to an uninitialized target.
        if self._wait_for_jtc(timeout_sec=self.startup_service_timeout_sec):
            self._send_hold_trajectory()
        else:
            self.get_logger().error(
                "JTC never became ready after EGM start — robot may move unexpectedly on first command"
            )

        self._ready_pub.publish(Bool(data=True))
        self.get_logger().info("EGM handler startup completed. Waiting for Ctrl+C to shutdown cleanly.")


def main(args=None):
    rclpy.init(args=args)

    executor = SingleThreadedExecutor()
    node = EGMHandler()
    node._executor = executor
    executor.add_node(node)

    # Install our SIGINT handler AFTER rclpy.init() so we override rclpy's
    # default handler. rclpy's handler calls rclpy.shutdown() immediately,
    # which would kill the context before shutdown_sequence() can send stop
    # commands. Our handler instead just sets a flag and lets main() drive
    # the shutdown in the correct order: stop_egm → stop_rapid → destroy → shutdown.
    shutdown_requested = [False]

    def _sigint_handler(*_):
        shutdown_requested[0] = True

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    try:
        node.startup_sequence()
        while rclpy.ok() and not shutdown_requested[0]:
            executor.spin_once(timeout_sec=0.1)
    finally:
        node.shutdown_sequence()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
