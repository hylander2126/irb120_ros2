# irb120_handeye

Hand-eye calibration tools for the IRB120 + RealSense. Produces the static
`base_link -> realsense_link` transform (eye-to-hand: the camera is fixed in
the world, not mounted on the arm) that's consumed by
[`bringup_stack.launch.py`](../irb120_control/launch/bringup_stack.launch.py)
via the `handeye_to_realsense_tf` include.

## Prerequisites

- **Terminal 1** (`ros2 launch irb120_control abb_rws.launch.py`) must already
  be running, same as any other bringup on this robot — see the top-level
  [irb120_ros2 README](../README.md#bringup-sequence). This package's launch
  file starts its own hardware bringup (equivalent to Terminal 2) and its own
  MoveIt/RViz stack (equivalent to Terminal 3), so you do **not** also need
  `abb_control.launch.py` or `bringup_stack.launch.py` running — those would
  conflict with this package's own `move_group`/RViz instance.
- The rigidly mounted, caliper-validated ChArUco target specified by
  [`calibrations/charuco_board.yaml`](calibrations/charuco_board.yaml) and
  [`generate_charuco_target.py`](irb120_handeye/generate_charuco_target.py).
  Mount it at/near the end effector (`tool0`) — since calibration is
  eye-to-hand, the target moves with the arm through each pose while the
  camera stays fixed.

## 1. Bringup

```bash
ros2 launch irb120_handeye bringup_handeye.launch.py
```

This starts:
- `abb_control.launch.py` (ros2_control + EGM handler — the hardware stack)
- `move_group`, using the calibration-specific SRDF (`irb120_handeye.srdf.xacro`)
- A dedicated RViz instance (`moveit_handeye.rviz`) with the `HandEyeCalibration`
  panel and a `Camera` display already pointed at
  `/handeye_calibration/target_detection`
- The RealSense (`rs_launch.py`), at `848x480` depth / `640x480` color,
  `align_depth` off, all post-processing filters off, `pointcloud.enable=true`,
  `clip_distance=2.2` — this profile is independent of (and doesn't need to
  match) whatever resolution `bringup_stack.launch.py` runs in production,
  since extrinsic calibration doesn't depend on image resolution.

## 2. HandEyeCalibration panel parameters

These are already saved in `moveit_handeye.rviz`, but if you ever need to
rebuild the panel from scratch, set:

| Field | Value |
|---|---|
| `target_type` | `HandEyeTarget/Charuco` |
| `ArUco dictionary` | `DICT_5X5_250` |
| `squares, X` / `squares, Y` | `5` / `7` |
| `square size (px)` / `marker size (px)` | `320` / `240` |
| `margin size (px)` | `0` |
| `marker border (bits)` | `1` |
| `longest board side (m)` | `0.224` |
| `measured marker size (m)` | `0.024` |
| `image_topic` | `/realsense/color/image_raw` |
| `sensor_mount_type` | `0` (Eye-to-hand) |
| `sensor` | `realsense_color_optical_frame` |
| `base` | **`base_link`** |
| `eef` | `tool0` |
| `group` | `manipulator` |
| `object` | `handeye_target` |
| `solver` | `OpenCV/Daniilidis1998` |

The longest board side and marker size must match the caliper-validated target:
224 mm and 24 mm. If the target changes, remeasure and update both fields or
the solved transform will be systematically off.

## 3. Run calibration poses

```bash
ros2 run irb120_handeye run_calibration_poses
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--pose-file` | `joints_5_6mm.yaml` | YAML filename under `share/irb120_handeye/calibrations/` |
| `--pose-path` | *(none)* | Absolute/relative path to a pose YAML — overrides `--pose-file` |
| `--move-time` | `4.0` | Seconds per move |
| `--settle-time` | `1.5` | Seconds to wait after each move before prompting |
| `--auto-continue` | off | Skip the Enter-key prompt between poses (use once you trust the target stays in view) |

For each pose the script: moves the arm, waits `--settle-time`, then (unless
`--auto-continue`) prompts you to press Enter. **Before** pressing Enter,
confirm the ChArUco board is detected in RViz's `Camera` view and click **Take
Sample** in the `HandEyeCalibration` panel — then press Enter to advance.

Once all poses are sampled, click **Solve** in the panel. It reports the
solved `base_link -> realsense_link` transform and a reprojection error.

## 4. Saving the result

The panel doesn't write a file for you. Copy the 7 solved values (x, y, z,
qx, qy, qz, qw) into a new static-transform launch file, named after the
**reprojection error you got, in mm — not a lens spec** (the D435's lens is
fixed; `_6mm`/`_12mm`/`_14mm` in these filenames are calibration quality, not
optics). Lower is better.

```python
# launch/cam_tf_<error>mm.launch.py
Node(
    package="tf2_ros",
    executable="static_transform_publisher",
    arguments=[
        "--frame-id", "base_link",      # NOT "base" — see gotcha below
        "--child-frame-id", "realsense_link",
        "--x", "...", "--y", "...", "--z", "...",
        "--qx", "...", "--qy", "...", "--qz", "...", "--qw", "...",
    ],
)
```

Then point `handeye_to_realsense_tf` in
[`bringup_stack.launch.py`](../irb120_control/launch/bringup_stack.launch.py)
at the new file. Keep the previous `cam_tf_*.launch.py` around rather than
deleting it — it's your fallback if the new result turns out worse in
practice than its reprojection error suggested.

## Calibration pose sets (`calibrations/`)

Naming convention: `joints_<N>_<error>mm.yaml` — `N` is the pose count in
that set, `<error>` is the reprojection error achieved the last time that set
was run, kept in the filename so a pose set and the transform file it
produced are easy to pair up.

- `joints_5_6mm.yaml` — current default, 5 poses.
- `OLD_joints_20_14mm.yaml` — previous 20-pose set, kept for reference; not
  used by default anymore.

## Headless recalibration (`run_handeye_calibration.py`)

`ros2 run irb120_handeye run_handeye_calibration` replaces steps 1-4 above
with a scripted OpenCV solve — no MoveIt/RViz, no manual "Take Sample"
clicks, and all cameras (the default; pass `--cameras realsense3` etc. to
calibrate a subset) solved from one shared pose set instead of separate
per-camera runs. If the pose file was taught with
`record_calibration_pose.py`, a pose is automatically skipped (no move)
when none of `--cameras` was recorded as seeing the board when that pose
was taught — so `--cameras realsense3` against a pose set that mixes
cam1/cam2 poses with cam3 poses only drives through the cam3 ones. There's
no requirement that every remaining pose be visible to every requested
camera; each camera accumulates its own sample set and gets its own
independent solve. Most other knobs (motion speed, sampling, board
type/measurements, solver method, output dir) are hardcoded constants near
the top of the module rather than flags — edit them there. See the module
docstring for the full recipe (`cv2.calibrateHandEye` in its documented
eye-to-hand mode) and the calibrated ChArUco target definition in
`generate_charuco_target.py`.

Bring up the camera(s) you're calibrating first — either the per-camera
`bringup_cam1.launch.py` / `bringup_cam2.launch.py` / `bringup_cam3.launch.py`
(each also republishes that camera's existing TF, harmless to leave running
during calibration) or the full `bringup_stack.launch.py`, plus
`abb_control.launch.py`/`abb_rws.launch.py` for the arm itself.

`record_calibration_pose.py` (the interactive MoveIt/RViz pose teacher) also
defaults to all three cameras and shows a live per-camera detection status
line while you jog the arm.

### Finding poses that are dragging down the solve

Each camera's solve starts from `cv2.calibrateHandEye` (Park), then refines
the camera pose and the board-in-`tool0` pose together by minimizing the
**reprojection error** of every detected board corner. The report under each
camera's solved transform lists every pose's RMS reprojection error in pixels
(and roughly in mm at the board's distance). A pose is flagged as an outlier
when its error is above 3x the set's median and above 2 px: misdetection,
board slipped, arm not fully settled, a TF glitch. Unlike the old pairwise
AX=XB residual, this doesn't grow with how different a pose is from the
others, so informative poses aren't penalized. `record_calibration_pose`
prints the same per-pose check live after each save.

As a rough guide, an overall RMS below ~1 px is good. If it's high with no
single outlier, the error is systematic: check the board YAML against the
real board, board flatness, and intrinsics.

Every run also writes `handeye_samples_<camera>.yaml` (raw per-pose
transforms and corner pixels) next to the `cam_tf_*.launch.py` it produces. That file lets
you re-run the diagnostic, try dropping specific poses, and re-solve —
**without moving the robot again**:

```bash
ros2 run irb120_handeye diagnose_handeye_samples --in ~/handeye_samples_realsense3.yaml
ros2 run irb120_handeye diagnose_handeye_samples --in ~/handeye_samples_realsense3.yaml \
    --exclude-poses "4,9" --write-launch
```

`--exclude-poses` takes the "Pose i/total" number printed live for that
pose (i.e. its 1-based index into the `--pose-file` used for that run), not
a position in the accepted-samples list — some poses may have been
missed/rejected for a given camera and so don't consume a "pose#" at all
for it. Quote the list (`"4,9"`, not `4,9`) so the shell passes it through
as a single argument. If dropping the flagged pose(s) doesn't bring the
residual down to a reasonable level, the error is probably not any single
pose — suspect the
mounted target's measured size, the camera intrinsics, or the extrinsic
frame convention (see the "Gotcha" section below) instead.

### Recalibrate all cameras

The prior camera extrinsics are invalid because the cameras have new poses.
Teach a fresh ChArUco-visible pose set, then run the headless calibrator with
its default three-camera selection:

```bash
ros2 run irb120_handeye record_calibration_pose --out ~/joints_charuco_2026.yaml
ros2 run irb120_handeye run_handeye_calibration --pose-path ~/joints_charuco_2026.yaml
```

This independently solves `realsense`, `realsense2`, and `realsense3`. Review
each generated transform/diagnostic before replacing the three active camera
TF launch files.

### Future work

- **Design a combined/multi-camera-aware pose set.** `joints_5_6mm.yaml` was
  tuned for cam1's FOV; cam2 (steep/overhead) and cam3 likely each need
  their own pose subset (board presented closer to face-on to that camera's
  viewing angle, arm held/positioned accordingly) rather than reusing cam1's
  poses as-is.

## Gotcha: check the parent frame before trusting a new result

`cam_tf_6mm.launch.py` (the current result, already wired into
`bringup_stack.launch.py`) publishes `--frame-id base`, but this robot's URDF
has **no link named `base`** — only `base_link` exists at the root of the
kinematic chain (confirmed in `irb120_3_58_macro.xacro`). The panel's own
`base` field is saved as `base_link`, and the older `cam_tf_12mm.launch.py`
correctly used `base_link` too. As written, `cam_tf_6mm.launch.py`'s static
transform is not connected to the rest of the TF tree — worth fixing (just
the `--frame-id` string, the solved x/y/z/quaternion values are unaffected)
before relying on this result.
