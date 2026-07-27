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
