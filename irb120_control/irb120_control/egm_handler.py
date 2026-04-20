#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from abb_rapid_sm_addin_msgs.srv import GetEGMSettings, SetEGMSettings
from abb_robot_msgs.srv import TriggerWithResultCode


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
        self.declare_parameter("pos_corr_gain", 0.0)

        self.rws_prefix = self.get_parameter("rws_service_prefix").value.rstrip("/")
        self.task = self.get_parameter("task").value
        self.startup_service_timeout_sec = float(self.get_parameter("startup_service_timeout_sec").value)
        self.shutdown_service_timeout_sec = float(self.get_parameter("shutdown_service_timeout_sec").value)
        self._shutdown_done = False
        self._executor: SingleThreadedExecutor | None = None

        self.rapid_stop_srv = f"{self.rws_prefix}/stop_rapid"
        self.pp_to_main_srv = f"{self.rws_prefix}/pp_to_main"
        self.rapid_start_srv = f"{self.rws_prefix}/start_rapid"
        self.egm_stop_srv = f"{self.rws_prefix}/stop_egm"
        self.egm_start_srv = f"{self.rws_prefix}/start_egm_joint"
        self.get_settings_srv = f"{self.rws_prefix}/get_egm_settings"
        self.set_settings_srv = f"{self.rws_prefix}/set_egm_settings"

        self.get_logger().info(f"Using RWS service prefix: {self.rws_prefix}")

    def _spin_future(self, future, timeout_sec: float) -> bool:
        """Spin the node's executor until the future completes or timeout."""
        end = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < end:
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
            (TriggerWithResultCode, self.rapid_stop_srv),
            (TriggerWithResultCode, self.pp_to_main_srv),
            (TriggerWithResultCode, self.rapid_start_srv),
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

        self.get_logger().info("Shutdown requested. Stopping EGM and RAPID.")

        # Use a fresh executor so service calls work even after KeyboardInterrupt
        # tore down the main spin loop.
        shutdown_executor = SingleThreadedExecutor()
        shutdown_executor.add_node(self)
        self._executor = shutdown_executor
        try:
            # Signal EGM to stop FIRST — this lets RAPID's EGM loop call EGMStop/EGMReset
            # cleanly before we kill the RAPID task. Stopping RAPID first leaves EGM
            # connections dangling and causes error 41828 ("too many EGM instances") on
            # the next launch.
            code, msg = self._call_trigger_retry(
                self.egm_stop_srv,
                attempts=4,
                timeout_sec=self.shutdown_service_timeout_sec,
                sleep_after=0.0,
            )
            self.get_logger().info(f"stop_egm -> code={code}, msg='{msg}'")

            # Give RAPID time to process EGMStop and run through its cleanup loop
            # before we kill the task.
            time.sleep(1.5)

            # Now stop RAPID — EGM connection is already closed.
            code, msg = self._call_trigger_retry(
                self.rapid_stop_srv,
                attempts=4,
                timeout_sec=self.shutdown_service_timeout_sec,
                sleep_after=0.5,
            )
            self.get_logger().info(f"stop_rapid -> code={code}, msg='{msg}'")
        finally:
            self._executor = None
            shutdown_executor.shutdown(wait=False)

    def startup_sequence(self):
        if not self._wait_for_startup_services():
            self.get_logger().error("Required startup services unavailable. Skipping EGM startup sequence.")
            return

        # Best-effort stop_egm first: if RAPID is still running from a previous session,
        # this lets it clean up EGM connections before we kill the task.
        # If RAPID is already dead (code=3005), this is a no-op — that's fine.
        code, msg = self._call_trigger(self.egm_stop_srv, sleep_after=1.0)
        self.get_logger().info(f"stop_egm (cleanup) -> code={code}, msg='{msg}'")

        # Now kill RAPID unconditionally — ensures we start from a clean slate.
        code, msg = self._call_trigger(self.rapid_stop_srv, sleep_after=1.0)
        self.get_logger().info(f"stop_rapid -> code={code}, msg='{msg}'")

        code, msg = self._call_trigger(self.pp_to_main_srv, sleep_after=0.5)
        self.get_logger().info(f"pp_to_main -> code={code}, msg='{msg}'")

        code, msg = self._call_trigger(self.rapid_start_srv, sleep_after=5.0)
        self.get_logger().info(f"start_rapid -> code={code}, msg='{msg}'")


        ## Disabling EGM settings adjustment to see if that causes FlexPendant crashes and reboots.
        # for attempt in range(1, 4):
        #     code, msg = self._set_egm_settings()
        #     self.get_logger().info(f"set_egm_settings (attempt {attempt}) -> code={code}, msg='{msg}'")
        #     if code == 1:
        #         break
        #     if code == 4003:
        #         self.get_logger().warn("SM Add-In not ready yet, waiting 2s before retry...")
        #         time.sleep(2.0)
        # time.sleep(0.5)

        code, msg = self._call_trigger(self.egm_start_srv, sleep_after=0.5)
        self.get_logger().info(f"start_egm_joint -> code={code}, msg='{msg}'")
        time.sleep(1.0)

        self.get_logger().info("EGM remains active until shutdown_sequence() or a manual stop request.")


def main(args=None):
    rclpy.init(args=args)
    executor = SingleThreadedExecutor()
    node = EGMHandler()
    node._executor = executor
    executor.add_node(node)

    try:
        node.startup_sequence()
        node.get_logger().info("EGM handler startup completed. Waiting for Ctrl+C to shutdown cleanly.")
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_sequence()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
