"""
Keyboard Jog Node
=================
Reads arrow keys from the terminal and publishes TwistStamped commands to
MoveIt Servo, producing smooth Cartesian end-effector motion.

  Right arrow  ->  +X (forward)
  Left  arrow  ->  -X (backward)
  Up    arrow  ->  +Z (upward)
  Down  arrow  ->  -Z (downward)
  Q / Ctrl-C   ->  quit

Hold a key to move continuously; release to stop.
Velocity is zeroed automatically if no key arrives within KEY_TIMEOUT seconds
(terminals don't send key-release events, so we use a timeout instead).

Usage (in a dedicated terminal after launching with keyboard_jog:=true):
  ros2 run irb120_control keyboard_jog
"""

import sys
import time
import tty
import termios
import threading
import select

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

# ANSI escape sequences emitted by arrow keys: ESC [ <letter>
_ESC = '\x1b'
_ARROWS = {
    '[A': (0.0,  1.0),   # Up    -> +Z
    '[B': (0.0, -1.0),   # Down  -> -Z
    '[C': (1.0,  0.0),   # Right -> +X
    '[D': (-1.0, 0.0),   # Left  -> -X
}

PUBLISH_RATE  = 50.0   # Hz — match servo publish_period
KEY_TIMEOUT   = 0.12   # seconds — zero target if no key seen (2x publish_period)
RAMP_UP       = 4.0    # units/sec — how fast velocity ramps up to target
RAMP_DOWN     = 8.0    # units/sec — how fast velocity decays to zero on release


class KeyboardJog(Node):

    def __init__(self):
        super().__init__('keyboard_jog')
        self.declare_parameter('frame_id', 'base_link')
        self._frame = self.get_parameter('frame_id').value

        self._pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)

        self._vx = 0.0        # actual commanded velocity (ramped)
        self._vz = 0.0
        self._target_vx = 0.0  # target from key press
        self._target_vz = 0.0
        self._last_key_time = 0.0
        self._lock = threading.Lock()

        self._timer = self.create_timer(1.0 / PUBLISH_RATE, self._publish_cb)

        self._running = True
        self._key_thread = threading.Thread(target=self._read_keys, daemon=True)
        self._key_thread.start()

        self.get_logger().info(
            '\n'
            '=========================================\n'
            '  Keyboard Jog  --  IRB120 Cartesian\n'
            '=========================================\n'
            '  Up/Down    : +Z / -Z  (up / down)\n'
            '  Left/Right : -X / +X  (back / forward)\n'
            '  Q          : quit\n'
            '  Hold key to move; release to stop.\n'
            '=========================================\n'
        )

    def _publish_cb(self):
        dt = 1.0 / PUBLISH_RATE
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame
        with self._lock:
            # Key release: zero the target if no key seen recently
            if time.monotonic() - self._last_key_time > KEY_TIMEOUT:
                self._target_vx = 0.0
                self._target_vz = 0.0
            # Ramp actual velocity toward target
            self._vx = self._ramp(self._vx, self._target_vx, dt)
            self._vz = self._ramp(self._vz, self._target_vz, dt)
            msg.twist.linear.x = self._vx
            msg.twist.linear.z = self._vz
        self._pub.publish(msg)

    @staticmethod
    def _ramp(current: float, target: float, dt: float) -> float:
        rate = RAMP_UP if abs(target) > abs(current) else RAMP_DOWN
        step = rate * dt
        if abs(target - current) <= step:
            return target
        return current + step * (1.0 if target > current else -1.0)

    def _set_velocity(self, vx: float, vz: float):
        with self._lock:
            self._target_vx = vx
            self._target_vz = vz
            self._last_key_time = time.monotonic()

    def _read_keys(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue

                ch = sys.stdin.read(1)

                if ch in ('q', 'Q', '\x03'):   # Q or Ctrl-C
                    self._running = False
                    self._set_velocity(0.0, 0.0)
                    break

                if ch == _ESC:
                    more = sys.stdin.read(2) if select.select([sys.stdin], [], [], 0.05)[0] else ''
                    if more in _ARROWS:
                        vx, vz = _ARROWS[more]
                        self._set_velocity(vx, vz)
                    # ignore other escape sequences (e.g. function keys)
                # ignore all other keys

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self._set_velocity(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardJog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        node.destroy_node()
        rclpy.shutdown()
