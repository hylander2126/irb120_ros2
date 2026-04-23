# irb120_ros2

ROS2 driver and application stack for the ABB IRB120 with IRC5 controller, RealSense D400 camera, and ATI net/ft sensor.

---

## Architecture overview

Three terminal groups run at different lifetimes:

| Terminal | What runs | Lifetime |
|---|---|---|
| **T1 — RWS client** | `abb_rws_client` | Always-on. Never kill unless rebooting the controller. |
| **T2 — ABB hardware** | `abb_control` (ros2_control + EGM handler) | Always-on during a session. |
| **T3 — MoveIt stack** | `bringup_stack` (move_group, perception, Servo, etc.) | Restart freely during development. |

**T1 and T2 must stay up** so that EGM shutdown is sent cleanly on Ctrl+C. Killing T1 (rws_client) before T2 means `stop_egm` cannot reach the IRC5, which causes the FlexPendant to crash/reboot.

---

## Bringup sequence

### Terminal 1 — RWS client (run once, leave running)

```bash
ros2 launch abb_bringup abb_rws_client.launch.py \
    robot_ip:=192.168.125.1 \
    robot_nickname:=IRB120
```

Leave this running indefinitely. It manages the Robot Web Services connection to the IRC5 and must outlive everything else so EGM can be stopped cleanly on shutdown.

### Terminal 2 — ABB hardware bringup (run once per session)

```bash
ros2 launch irb120_control abb_control.launch.py
```

This starts:
- `ros2_control_node` with the ABB hardware interface
- `robot_state_publisher`
- `joint_state_broadcaster`
- `egm_handler` — clears any stale EGM session, applies EGM settings, starts a fresh EGM session, then gates the JTC spawner until EGM is confirmed live

The terminal will show `EGM handler startup completed` when the robot is ready to receive joint trajectory commands. **Do not kill this terminal** while the robot is powered — always Ctrl+C cleanly so `egm_handler` can send `stop_egm` before the hardware interface drops the UDP socket.

### Terminal 3 — MoveIt stack (restart freely)

```bash
ros2 launch irb120_control bringup_stack.launch.py
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `start_servo` | `false` | Start MoveIt Servo for Cartesian jogging |
| `debug_perception` | `false` | Launch perception debugger + debug RViz config |
| `perception_method` | `dbscan` | Segmentation backend: `dbscan` or `sam` |

Example with Servo enabled:
```bash
ros2 launch irb120_control bringup_stack.launch.py start_servo:=true
```

This is the terminal you kill and relaunch during iteration. T1 and T2 remain untouched.

---

## Keyboard jogging (requires `start_servo:=true`)

```bash
ros2 run irb120_control keyboard_jog
```

- `↑` / `↓` — +Z / −Z (up/down)
- `←` / `→` — −X / +X (forward/back)

---

## Hand-eye calibration

### Bringup

```bash
ros2 launch irb120_handeye bringup_handeye.launch.py
```

Starts the hardware stack with the handeye MoveIt config, RViz handeye plugin, and RealSense camera.

### Run calibration poses

```bash
ros2 run irb120_handeye run_calibration_poses
```

Options:
```
--pose-file   YAML filename under share/irb120_handeye/calibrations/  (default: joints_20_14mm.yaml)
--pose-path   Absolute path to a pose YAML (overrides --pose-file)
--move-time   Seconds per move  (default: 4.0)
--settle-time Seconds to settle after each move  (default: 1.5)
--auto-continue  Skip Enter prompts between poses
```

In RViz, use the HandEye Calibration panel to take samples manually at each pose.

### Calibration results

Calibration TF launches live in `share/irb120_handeye/launch/`:
- `cam_tf_12mm.launch.py` — current result (12mm lens, eye-to-hand: `base_link` → `realsense_link`)

---

## Squash-pull

Requires T1 + T2 + T3 with `start_servo:=true`.

```bash
ros2 launch irb120_control bringup_stack.launch.py start_servo:=true
# then in another terminal:
ros2 run irb120_control squash_pull
```

The node:
1. Uses MoveIt to plan to the pre-squash pose (position + orientation)
2. Prompts for operator confirmation
3. Descends with force feedback (halves speed on first contact)
4. Pulls laterally while maintaining contact force via PI control
5. Retracts to clearance height

Tune constants at the top of [squash_pull.py](irb120_control/irb120_control/squash_pull.py).

---

## EGM session recovery (occasional timeout)

After a very long idle period, the IRC5 may time out the EGM session. The robot motors will stop whining (brakes engage). Recover without restarting anything:

```bash
# Stop the stale session (safe to call even if already stopped)
ros2 service call /rws_client/stop_egm abb_robot_msgs/srv/TriggerWithResultCode '{}'

# Start a fresh session
ros2 service call /rws_client/start_egm_joint abb_robot_msgs/srv/TriggerWithResultCode '{}'
```

Expected response: `result_code: 1` (success). The robot motors will resume humming within a few seconds.

If the JTC also needs to be respawned after recovery:
```bash
ros2 run controller_manager spawner joint_trajectory_controller -c /controller_manager
```

---

## Package layout

| Package | Purpose |
|---|---|
| `irb120_control` | Hardware bringup, controllers, EGM handler, application nodes (keyboard jog, squash-pull, net/ft) |
| `irb120_moveit_config` | MoveIt config (SRDF, kinematics, OMPL, Servo, joint limits) |
| `irb120_perception` | Object detection (DBSCAN / SAM2), robot mask filter, perception debugger |
| `irb120_handeye` | Hand-eye calibration bringup, pose runner, calibration data files, camera TF |
| `irb120_abb_hardware_interface` | Custom ros2_control hardware plugin for ABB EGM |

---

## Network addresses

| Device | IP |
|---|---|
| IRC5 controller (RWS) | `192.168.125.1` |
| ATI net/ft sensor | `192.168.126.125` |