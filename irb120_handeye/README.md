# irb120_handeye

Three serial-pinned cameras, their active eye-to-hand transforms, ChArUco target
and headless calibration tools. Active camera wiring lives in the `CAMERAS` table
in `launch/bringup_cams.launch.py`; do not infer which solve is active from an
older filename mentioned in an experiment report.

## Camera profiles

```bash
ros2 launch irb120_handeye bringup_cams.launch.py profile:=offline_stereo
ros2 launch irb120_handeye bringup_cams.launch.py cameras:=cam2 profile:=raw_depth
```

| Profile | Purpose |
|---|---|
| `offline_stereo` | Native 1280x720 IR/RGB capture for FFS, without continuous pointcloud/aligned-depth outputs |
| `raw_depth` | Same native stereo streams plus unfiltered aligned RealSense depth/cloud, for sensor diagnostics |
| `calibration` | Color-only ChArUco calibration, 1280x720 at 30 Hz |

The `filtered` profile and old per-camera bringup files are retired. Default
`bringup_cams` profile remains `raw_depth` for diagnostic compatibility; production
`bringup_stack` explicitly selects `offline_stereo`. `cameras` defaults to all
three; missing/drivers exiting report errors without terminating the other cameras.
Never start another driver against a camera already owned by a running stack.
FFS capture audits the live camera-link TFs against the selected calibration and
rejects stale/conflicting static publishers. The cam2 Z compensation defaults to
+4 mm with the current calibration; reconstruction accounts for the captured
shift so it is applied exactly once. Restart camera bringup after calibration changes.

## Calibration procedure

T1 RWS and T2 `abb_control.launch.py` must already be running. Stop production T3
and start calibration T3 instead:

```bash
ros2 launch irb120_handeye bringup_handeye.launch.py cameras:=cam2 egm_cond_time:=1200.0
```

This uses the ChArUco model/`charuco_center` tip and calibration joint limits,
pushes the matching robot description to T2's robot_state_publisher, starts
MoveIt/RViz/cameras and the shared EGM handler. It does not start another hardware
controller stack. Calibration's explicit joint-6 bounds differ from the
production +/-60 degree cable window; do not copy calibration limits into production.

The board definition is `calibrations/charuco_board.yaml`: DICT_5X5_250,
5x7 squares, 32 mm squares and 24 mm markers. Keep the printed target tied to
these measured dimensions. The default shared pose set is `joints_custom.yaml`;
a single-camera session can use its own pose file:

```bash
ros2 run irb120_handeye handeye calibrate --cameras realsense2 \
  --pose-file /path/to/cam2_poses.yaml --plan-only
ros2 run irb120_handeye handeye calibrate --cameras realsense2 \
  --pose-file /path/to/cam2_poses.yaml --step
```

`--plan-only` plans the pose sequence without moving. The subsequent calibration
command moves the robot. `--step` shows the next move and waits before executing.
`--dry-run` still moves and detects but does not solve/write results. Headless
`calibrate` performs sampling/solve itself; it does not require manual RViz
“Take Sample” clicks. Restricting `--cameras` selects cameras to solve, not a new
pose geometry. Teach camera-specific poses if the shared set does not span its view.

## Results and promotion

New solves and raw sample evidence are saved in separate timestamped directories:
`runtime_logs/calibration/handeye/<timestamp>/`. Existing home-directory results
and active transforms are untouched. The outputs include per-camera sample YAML
and generated `cam_tf_<camera>_<RMS>px.launch.py` files. Each run also retains
the board/pose input copies and `provenance.json` with input and solver hashes. Preserve sample provenance;
RMS in a filename is a diagnostic, not sufficient version identity.

```bash
ros2 run irb120_handeye handeye diagnose --in /path/to/handeye_samples_realsense2.yaml
```

Review residuals and stability, then copy the reviewed generated launch into this
package's `launch/` and update only that camera's `CAMERAS` entry. Retain the old
solve in Git/history or the calibration run archive, not as another active driver.
Rebuild/source and restart the camera session to use the promoted transform.

A lower color reprojection RMS does not establish stereo-depth accuracy. Cam2's
1.2 px replacement did not remove its FFS table-plane offset in drill_test3.
For depth/PnP diagnosis use `handeye check-depth-pnp --camera realsense2` with the
board and `profile:=raw_depth` (aligned depth required); compare multiple distances
before changing extrinsics again.

## Tool map

`ros2 run irb120_handeye handeye --help` lists `calibrate`, `run-poses`,
`record-pose`, `generate-target`, `diagnose` and `check-depth-pnp`. Each subcommand
accepts `--help`. Use `record-pose --out PATH` to keep newly taught poses in a
named file; promote reviewed sets deliberately rather than overwriting the
shipped shared set. Older panel-based instructions are in Git and the development
archive, not the current headless procedure.
