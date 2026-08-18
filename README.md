# irb120_ros2

ROS2 driver and application stack for the ABB IRB120 with IRC5 controller, two RealSense D400 cameras, and ATI net/ft sensor.

---

## Install guide

These instructions assume Ubuntu 24.04 with ROS 2 Jazzy installed and that only this
Git repository was downloaded. Do not copy `build/`, `install/`, or `log/` from
another computer; those directories contain generated files and must be recreated
locally.

The expected workspace layout is:

```text
irb120_ws/
└── src/
    ├── irb120_ros2/               # this repository
    └── ...                        # dependency repositories cloned below
```

### 1. Clone this repository

```bash
mkdir -p ~/Documents/irb120_ws/src
cd ~/Documents/irb120_ws/src
git clone https://github.com/hylander2126/irb120_ros2.git
```

If the repository is already downloaded, place it at
`~/Documents/irb120_ws/src/irb120_ros2`, or adjust the workspace path in the
remaining commands.

### 2. Clone the source dependencies

Clone these repositories beside `irb120_ros2`, not inside it:

```bash
cd ~/Documents/irb120_ws/src

git clone -b rolling https://github.com/PickNikRobotics/abb_ros2.git
git clone -b rolling https://github.com/gbartyzel/abb_ros2_msgs.git
git clone https://github.com/ros-industrial/abb_egm_rws_managers.git
git clone https://github.com/ros-industrial/abb_libegm.git
git clone https://github.com/ros-industrial/abb_librws.git
git clone -b ros2 https://github.com/UTNuclearRoboticsPublic/netft_utils.git
git clone -b ros2 https://github.com/ros-planning/moveit_calibration.git
```

`abb_ros2` and `abb_ros2_msgs` use their `rolling` branches in this Jazzy
workspace.

### 3. Install dependencies

```bash
source /opt/ros/jazzy/setup.bash
cd ~/Documents/irb120_ws
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
sudo apt install python3-sklearn
```

`python3-sklearn` is required by the DBSCAN perception node. It is listed
separately because it is not currently represented by a rosdep dependency in
`irb120_perception`.

### 4. Build and source the workspace

```bash
cd ~/Documents/irb120_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Source both setup files in every new terminal before running this stack:

```bash
source /opt/ros/jazzy/setup.bash
source ~/Documents/irb120_ws/install/setup.bash
```

Optionally add those two lines to `~/.bashrc`.

## Architecture overview

Three terminal groups run at different lifetimes:

| Terminal | What runs | Lifetime |
|---|---|---|
| **T1 — RWS client** | `irb120_control/abb_rws.launch.py` | Always-on. Never kill unless rebooting the controller. |
| **T2 — ABB hardware** | `irb120_control/abb_control.launch.py` (ros2_control + EGM handler) | Always-on during a session. |
| **T3 — MoveIt stack** | `irb120_control/bringup_stack.launch.py` (move_group, perception, Servo, etc.) | Restart freely during development. |

**T1 and T2 must stay up** so that EGM shutdown is sent cleanly on Ctrl+C. Killing T1 (rws_client) before T2 means `stop_egm` cannot reach the IRC5, which causes the FlexPendant to crash/reboot.

The launch files in this repository are the supported entry points for the robot. They include and configure components from ABB and other third-party ROS 2 packages internally; those packages are still runtime dependencies, but their launch files should not be invoked directly for this setup.

---

## Bringup sequence

### Terminal 1 — RWS client (run once, leave running)

```bash
ros2 launch irb120_control abb_rws.launch.py
```

This local launch file configures the Robot Web Services connection for the IRC5 at `192.168.125.1` and includes the required ABB RWS implementation. Leave it running indefinitely; it must outlive everything else so EGM can be stopped cleanly on shutdown.

### Terminal 2 — ABB hardware bringup (run once per session)

```bash
ros2 launch irb120_control abb_control.launch.py
```

This local launch file is the entry point for the cell-specific hardware configuration. It includes the required ABB bringup components with this repository's URDF, controller configuration, MoveIt configuration, and hardware settings.

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
| `start_servo` | `false` | Start MoveIt Servo for Cartesian motions |
| `debug_perception` | `false` | Launch perception debugger + debug RViz config |
| `perception_method` | `dbscan` | Segmentation backend: `dbscan` or `sam` |

Example with Servo enabled:
```bash
ros2 launch irb120_control bringup_stack.launch.py start_servo:=true
```

This is the terminal you kill and relaunch during iteration. T1 and T2 remain untouched.

This also brings up both RealSense cameras — `irb120_handeye`'s
`bringup_cam1.launch.py` and `bringup_cam2.launch.py`, each pairing a
camera driver pinned to its serial number with its calibrated TF. See
[Calibration results](#calibration-results) below for which camera is which.

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

Each camera has a driver bringup paired with its solved TF, both under
`share/irb120_handeye/launch/`:
- `bringup_cam1.launch.py` + `camera_1_tf.launch.py` — primary camera
  (`realsense`, serial `243522072478`), eye-to-hand `base` → `realsense_link`,
  solved via MoveIt hand-eye calibration (6mm reprojection error, current
  result)
- `bringup_cam2.launch.py` + `camera_2_tf.launch.py` — second camera
  (`realsense2`, serial `750612071219`), `base` → `realsense2_link`, solved
  via ICP calibration

Both pairs are included automatically by `bringup_stack.launch.py`. Launch a
`bringup_camN.launch.py` on its own only to verify a single camera in
isolation (see the header comment in each file).

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
| `irb120_handeye` | Hand-eye calibration bringup, pose runner, calibration data files, and per-camera bringup (driver + solved TF) for both RealSense cameras |
| `irb120_abb_hardware_interface` | Custom ros2_control hardware plugin for ABB EGM |

---


## Networking

The IRC5 uses two different network paths for robot bringup:

- Robot Web Services (RWS) is TCP communication with the controller at
  `192.168.125.1:80`.
- Externally Guided Motion (EGM) is UDP traffic sent **from the controller to
  the robot-facing network adapter on the ROS computer** at port `6511`.

Configure the robot network as follows:

| Endpoint | Address |
|---|---|
| IRC5 controller | `192.168.125.1/24` |
| ROS computer's robot-facing Ethernet adapter | `192.168.125.208/24` |
| IRC5 `ROB_1` UDPUC remote address | `192.168.125.208` |
| IRC5 `ROB_1` UDPUC remote port | `6511` |

The adapter and the `ROB_1` UDPUC remote address must match. The UDPUC address
is the destination ROS computer, **not** the controller's own address. A gateway
and DNS server are not needed on this dedicated robot connection.

After configuring the adapter, verify the address and controller connectivity:

```bash
ip -brief address
ip route
ping -c 3 192.168.125.1
```

Successful RWS communication alone does not prove that EGM is configured
correctly. If bringup identifies the controller but repeatedly prints
`ABBSystemHardware: Not connected to robot...`, check whether EGM packets are
arriving:

```bash
sudo tcpdump -ni <robot-interface> udp port 6511
```

Replace `<robot-interface>` with the adapter name shown by `ip -brief address`.
If no packets arrive after EGM is started, recheck the UDPUC remote address. If
packets arrive but ROS does not connect, check the host firewall and allow UDP
port `6511` from `192.168.125.1`.

The ATI Net/F/T sensor is on a separate subnet:

| Device | Address |
|---|---|
| ATI Net/F/T sensor | `192.168.126.125` |

The computer needs an interface or route on `192.168.126.0/24` to receive
`/netft_data`.
