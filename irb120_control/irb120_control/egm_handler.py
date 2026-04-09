#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node

from abb_rapid_sm_addin_msgs.srv import GetEGMSettings, SetEGMSettings
from abb_robot_msgs.srv import TriggerWithResultCode


class EGMHandler(Node):
    def __init__(self):
        super().__init__("egm_handler")

        self.declare_parameter("rws_service_prefix", "/rws_client")
        self.declare_parameter("task", "T_ROB1")
        self.declare_parameter("stop_egm_after_startup", True)
        self.declare_parameter("one_shot", True)
        self.declare_parameter("startup_service_timeout_sec", 30.0)
        self.declare_parameter("shutdown_on_exit", False)

        self.declare_parameter("max_speed_dev_rad", 1.5)
        self.declare_parameter("comm_timeout", 5.0)
        self.declare_parameter("ramp_in_time", 2.0)
        self.declare_parameter("ramp_out_time", 0.25)
        self.declare_parameter("pos_corr_gain", 0.0)

        self.rws_prefix = self.get_parameter("rws_service_prefix").value.rstrip("/")
        self.task = self.get_parameter("task").value
        self.stop_egm_after_startup = bool(self.get_parameter("stop_egm_after_startup").value)
        self.one_shot = bool(self.get_parameter("one_shot").value)
        self.startup_service_timeout_sec = float(self.get_parameter("startup_service_timeout_sec").value)
        self.shutdown_on_exit = bool(self.get_parameter("shutdown_on_exit").value)

        self.rapid_stop_srv = f"{self.rws_prefix}/stop_rapid"
        self.pp_to_main_srv = f"{self.rws_prefix}/pp_to_main"
        self.rapid_start_srv = f"{self.rws_prefix}/start_rapid"
        self.egm_stop_srv = f"{self.rws_prefix}/stop_egm"
        self.egm_start_srv = f"{self.rws_prefix}/start_egm_joint"
        self.get_settings_srv = f"{self.rws_prefix}/get_egm_settings"
        self.set_settings_srv = f"{self.rws_prefix}/set_egm_settings"

        self.get_logger().info(f"Using RWS service prefix: {self.rws_prefix}")

    def _wait_for_service(self, client, service_name, timeout_sec):
        end_time = time.time() + timeout_sec
        while rclpy.ok() and time.time() < end_time:
            if client.wait_for_service(timeout_sec=0.2):
                return True
        self.get_logger().warn(f"Service unavailable: {service_name}")
        return False

    def _call_trigger(self, service_name, timeout_sec=5.0, sleep_after=0.5):
        client = self.create_client(TriggerWithResultCode, service_name)
        if not self._wait_for_service(client, service_name, timeout_sec):
            return None, "service unavailable"

        future = client.call_async(TriggerWithResultCode.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done() or future.result() is None:
            self.get_logger().warn(f"Service call failed: {service_name}")
            return None, "call failed"

        response = future.result()
        if sleep_after > 0.0:
            time.sleep(sleep_after)
        return response.result_code, response.message

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
        rclpy.spin_until_future_complete(self, get_future, timeout_sec=5.0)

        if not get_future.done() or get_future.result() is None:
            return None, "get_egm_settings call failed"

        current = get_future.result()
        settings = current.settings

        settings.activate.max_speed_deviation = math.degrees(float(self.get_parameter("max_speed_dev_rad").value))
        settings.setup_uc.comm_timeout = float(self.get_parameter("comm_timeout").value)
        settings.run.ramp_in_time = float(self.get_parameter("ramp_in_time").value)
        settings.stop.ramp_out_time = float(self.get_parameter("ramp_out_time").value)
        settings.run.pos_corr_gain = float(self.get_parameter("pos_corr_gain").value)

        set_req = SetEGMSettings.Request()
        set_req.task = self.task
        set_req.settings = settings

        set_future = set_client.call_async(set_req)
        rclpy.spin_until_future_complete(self, set_future, timeout_sec=5.0)

        if not set_future.done() or set_future.result() is None:
            return None, "set_egm_settings call failed"

        response = set_future.result()
        return response.result_code, response.message

    def shutdown_sequence(self):
        self.get_logger().info("Shutdown requested. Stopping EGM and RAPID.")

        code, msg = self._call_trigger(self.egm_stop_srv, timeout_sec=2.0, sleep_after=0.25)
        self.get_logger().info(f"stop_egm -> code={code}, msg='{msg}'")

        code, msg = self._call_trigger(self.rapid_stop_srv, timeout_sec=2.0, sleep_after=0.25)
        self.get_logger().info(f"stop_rapid -> code={code}, msg='{msg}'")

    def startup_sequence(self):
        if not self._wait_for_startup_services():
            self.get_logger().error("Required startup services unavailable. Skipping EGM startup sequence.")
            return

        code, msg = self._call_trigger(self.egm_stop_srv, sleep_after=0.5)
        self.get_logger().info(f"stop_egm -> code={code}, msg='{msg}'")

        code, msg = self._call_trigger(self.rapid_stop_srv, sleep_after=0.5)
        self.get_logger().info(f"stop_rapid -> code={code}, msg='{msg}'")

        code, msg = self._call_trigger(self.pp_to_main_srv, sleep_after=0.5)
        self.get_logger().info(f"pp_to_main -> code={code}, msg='{msg}'")

        code, msg = self._call_trigger(self.rapid_start_srv, sleep_after=1.0)
        self.get_logger().info(f"start_rapid -> code={code}, msg='{msg}'")

        code, msg = self._set_egm_settings()
        self.get_logger().info(f"set_egm_settings -> code={code}, msg='{msg}'")
        time.sleep(0.5)

        code, msg = self._call_trigger(self.egm_start_srv, sleep_after=0.5)
        self.get_logger().info(f"start_egm_joint -> code={code}, msg='{msg}'")

        if self.stop_egm_after_startup:
            code, msg = self._call_trigger(self.egm_stop_srv, sleep_after=0.25)
            self.get_logger().info(f"post_startup stop_egm -> code={code}, msg='{msg}'")


def main(args=None):
    rclpy.init(args=args)
    node = EGMHandler()

    try:
        node.startup_sequence()
        if node.one_shot:
            node.get_logger().info("EGM handler one-shot completed. Exiting.")
        else:
            node.get_logger().info("EGM handler completed startup sequence and is now idle.")
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.shutdown_on_exit:
            node.shutdown_sequence()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
