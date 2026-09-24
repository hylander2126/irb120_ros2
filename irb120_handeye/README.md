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
- A printed copy of the ArUco target board: [`calibrations/irb_target_image.png`](calibrations/irb_target_image.png).
  Mount it rigidly at/near the end effector (`tool0`) — since calibration is
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
| `target_type` | `HandEyeTarget/Aruco` |
| `ArUco dictionary` | `DICT_5X5_250` |
| `markers, X` / `markers, Y` | `3` / `4` |
| `marker size (px)` / `marker separation (px)` | `200` / `20` |
| `marker border (bits)` | `1` |
| `measured marker size (m)` | `0.034` |
| `measured separation (m)` | `0.0034` |
| `image_topic` | `/realsense/color/image_raw` |
| `sensor_mount_type` | `0` (Eye-to-hand) |
| `sensor` | `realsense_color_optical_frame` |
| `base` | **`base_link`** |
| `eef` | `tool0` |
| `group` | `manipulator` |
| `object` | `handeye_target` |
| `solver` | `OpenCV/Daniilidis1998` |

The measured marker size/separation must match your actual printout — if you
reprint the target at a different scale, remeasure with calipers and update
these two fields, or the solved transform will be systematically off.

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
confirm the ArUco board is detected in RViz's `Camera` view and click **Take
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
eye-to-hand mode) and `generate_charuco_target.py` for the print-ready
ChArUco target it prefers.

Bring up the camera(s) you're calibrating first — either the per-camera
`bringup_cam1.launch.py` / `bringup_cam2.launch.py` / `bringup_cam3.launch.py`
(each also republishes that camera's existing TF, harmless to leave running
during calibration) or the full `bringup_stack.launch.py`, plus
`abb_control.launch.py`/`abb_rws.launch.py` for the arm itself.

`record_calibration_pose.py` (the interactive MoveIt/RViz pose teacher) also
defaults to all three cameras and shows a live per-camera detection status
line while you jog the arm.

### Finding poses that are dragging down the solve

Every camera's solve also runs a **leave-one-out (LOO) diagnostic**: for
each accepted pose it re-solves with just that one pose left out and
reports how much the AX=XB residual improves. A pose that's actually bad
(misdetection, board slipped, arm not fully settled, a TF glitch) drags the
residual up, so dropping it makes the residual noticeably better; a fine
pose barely moves it either way. The report is printed live, ranked
worst-offender-first, right under each camera's solved transform.

Every run also writes `handeye_samples_<camera>.yaml` (raw per-pose
AX/AB data) next to the `cam_tf_*.launch.py` it produces. That file lets
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

### Future work

- **Switch `run_handeye_calibration.py`'s `BOARD_TYPE` constant back to
  `'charuco'` (its long-term default) once a printer and a rigid flat
  backing (acrylic/Dibond/plywood, not foam-core) are available.** It's
  currently `'aruco'` — the existing grid board (`irb_target_image.png`)
  already printed and mounted — purely because that's the only board on
  hand right now. A GridBoard's pose comes from marker corners alone with
  no checkerboard-corner refinement, so it's noisier and less
  occlusion-tolerant than ChArUco; once the ChArUco target is printed
  (`generate_charuco_target.py`, printed at 100% and mounted rigidly per
  its header comment) and mounted, switch the constant and prefer that
  result.
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
