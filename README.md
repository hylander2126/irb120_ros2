# irb120_ros2

ROS2 driver and application stack for the ABB IRB120 with IRC5 controller, three RealSense D400 cameras, and ATI net/ft sensor.

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

### 2. Import the pinned source dependencies

The committed vcstool manifest records the exact dependency revisions tested by
this project. Import them beside `irb120_ros2`, not inside it:

```bash
sudo apt install python3-vcstool
cd ~/Documents/irb120_ws
vcs import src < src/irb120_ros2/irb120_jazzy.repos
```

The manifest intentionally uses the project fork of `abb_egm_rws_managers`.
Commit `18d6a94` preserves valid joint velocity feedforward below the vendor's
former per-joint 0.5 degree/s (or mm/s) deadband. The regression test records a
coordinated command for which applying that deadband independently to each
joint reverses the Cartesian feedforward direction.

### 3. Install dependencies

```bash
source /opt/ros/jazzy/setup.bash
cd ~/Documents/irb120_ws
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
```


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

## Supported workflow

| Component | Entry point |
|---|---|
| T1: RWS, once per session | `ros2 launch irb120_control abb_rws.launch.py` |
| T2: hardware/controller stack, once per session | `ros2 launch irb120_control abb_control.launch.py` |
| T3: MoveIt, Servo, cameras, recording and the warm FFS worker | `ros2 launch irb120_control bringup_stack.launch.py` |
| Perception or interaction | `ros2 run irb120_control episode snapshot OBJECT` or `ros2 run irb120_control run_pipeline OBJECT` |

Keep T1/T2 running when restarting T3. For calibration, use
`bringup_handeye.launch.py` **instead of** production T3; see
[handeye instructions](irb120_handeye/README.md). Do not duplicate the hardware stack.

Production T3 defaults: `start_servo:=true`, `cameras:=cam1,cam2,cam3`,
`camera_profile:=offline_stereo`. The alternative `raw_depth` profile also publishes
aligned depth/cloud for sensor diagnostics. Live DBSCAN/SAM detector launches,
filtered-cloud perception and the old RealSense-only TSDF CLI are retired.

Perception-only work needs no T1/T2/T3: start cameras with
`ros2 launch irb120_handeye bringup_cams.launch.py profile:=offline_stereo`,
then run `stereo_offline_pipeline.launch.py object_name:=OBJECT`. Do not start
those drivers alongside an existing camera session.

## Documentation map

| Guide | Owns |
|---|---|
| [Control](irb120_control/README.md) | Primitives, automatic FFS episode snapshots, imports, recording and official data |
| [Perception](irb120_perception/README.md) | Supported FFS path and environment contract |
| [FFS workflow](irb120_perception/docs/STEREO_OFFLINE.md) | Capture/reconstruction/review commands and diagnostics |
| [Contact selection](irb120_perception/CONTACT_SELECTION.md) | Current contact/pivot policy and limitations |
| [Handeye](irb120_handeye/README.md) | Camera profiles, calibration procedure and result promotion |
| [Models](irb120_models/README.md) | Robot/object frames, registry and generated assets |
| [Analysis](analysis/README.md) | Methods, scoring and paper artifacts |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Durable cell diagnostics |
| [Backlog](docs/BACKLOG.md) | Remaining work, with links to dated evidence |
| [Archive](docs/archive/README.md) | Superseded plans and experiment history |

The local `irb120_abb_hardware_interface` plugin is required for measured-state
seeding and EGM guards. The six upstream dependencies remain pinned in
`irb120_jazzy.repos`; their own documentation stays with their checkouts.

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
