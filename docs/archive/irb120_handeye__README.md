> Archived 2026-10-24; current procedure is in the package README.

# irb120_handeye

Hand-eye calibration tools for the IRB120 + RealSense. Produces the static
`base_link -> realsense_link` transform (eye-to-hand: the camera is fixed in
the world, not mounted on the arm) that's consumed by
[`bringup_stack.launch.py`](../irb120_control/launch/bringup_stack.launch.py)
via the `handeye_to_realsense_tf` include.

## Prerequisites

- **Terminal 1** (`ros2 launch irb120_control abb_rws.launch.py`) and
  **Terminal 2** (`ros2 launch irb120_control abb_control.launch.py`) must already
  be running, same as any other bringup on this robot — see the top-level
  [irb120_ros2 README](../README.md#bringup-sequence). `bringup_handeye.launch.py`
  is Terminal 3 in place of `bringup_stack.launch.py`; don't run both, since they
  would conflict over `move_group`, RViz, EGM and the cameras.
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
- The EGM startup handler (`irb120_control/launch/egm_handler.launch.py`, shared with `bringup_stack`)
- `move_group` with the ChArUco board + mount in place of the finger
  (`irb120_with_charuco.xacro` + `irb120_handeye.srdf.xacro`), so collision checking
  uses the calibration end effector, and `config/joint_limits_calibration.yaml`
  (joint_6 widened to ABB's full ±400° instead of the ±60° cable limit). The same ChArUco URDF is pushed to abb_control's
  `robot_state_publisher`, so TF and RViz's RobotModel match.
- A dedicated RViz instance (`moveit_handeye.rviz`) with the `HandEyeCalibration`
  panel and `Camera` displays for all three cameras
  (`/realsense{,2,3}/color/image_raw`)
- All three cameras via `bringup_cam{1,2,3}.launch.py calibration:=true`
  (serial-pinned, with their static TFs; `config/realsense_calibration.yaml`:
  1280x720 color, depth off). Extrinsics don't depend on resolution, so results
  stay valid at production resolution.

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
ros2 run irb120_handeye handeye run-poses
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--pose-file` | `joints_custom.yaml` | YAML filename under `share/irb120_handeye/calibrations/` |
| `--pose-path` | *(none)* | Absolute/relative path to a pose YAML — overrides `--pose-file` |
| `--velocity-scaling` | `0.15` | MoveIt max velocity scaling factor |
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

Older sets used the naming convention `joints_<N>_<error>mm.yaml` (pose
count + the reprojection error achieved last time that set was run); both
`joints_5_6mm.yaml` and `OLD_joints_20_14mm.yaml` it describes were replaced
and deleted. The current default is:

- `joints_custom.yaml` — the live shared pose set (38 poses as of this
  writing), used by `run_handeye_calibration.py --pose-file` and
  `run_calibration_poses.py --pose-file` when neither flag is overridden.
  Taught incrementally with `record_calibration_pose.py` as cameras were
  added (cam1 first, cam2 and cam3 later) — see "Recalibrate ONE camera"
  below for why that history matters if one camera's residual won't improve.

## Headless recalibration (`run_handeye_calibration.py`)

`ros2 run irb120_handeye handeye calibrate` replaces steps 1-4 above
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

Run `abb_rws.launch.py` (T1), `abb_control.launch.py` (T2) and
**`bringup_handeye.launch.py`** (T3), which brings up EGM and the cameras
(`bringup_cams.launch.py profile:=calibration`, each also republishing that camera's
existing TF, harmless during calibration) itself; don't run those separately. Pass `cameras:=cam2` to start only the camera
you're calibrating. Do **not** use `bringup_stack.launch.py` for this —
it's the production/experiments launch, has no calibration mode, and keeps
the full `±60°` joint_6 FT-cable-protection limit active with no way to turn
it off (see "Recalibrate ONE camera" below for what that limit is and why
calibration needs `bringup_handeye.launch.py`'s own relaxed
`joint_limits_calibration.yaml` instead).

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
ros2 run irb120_handeye handeye diagnose --in ~/handeye_samples_realsense3.yaml
ros2 run irb120_handeye handeye diagnose --in ~/handeye_samples_realsense3.yaml \
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
ros2 run irb120_handeye handeye record-pose --out ~/joints_charuco_2026.yaml
ros2 run irb120_handeye handeye calibrate --pose-file ~/joints_charuco_2026.yaml
```

This independently solves `realsense`, `realsense2`, and `realsense3`. Review
each generated transform/diagnostic before replacing the three active camera
TF launch files.

### Recalibrate ONE camera (e.g. cam2)

`--cameras` restricts a `calibrate` run to just that camera's solve/output,
e.g. `ros2 run irb120_handeye handeye calibrate --cameras realsense2` (camera
names: `realsense`=cam1, `realsense2`=cam2, `realsense3`=cam3). **That alone
is usually not the fix if one camera keeps coming back worse than the
others** — restricting `--cameras` only changes which camera gets solved and
written out, not which poses are driven through; it still walks the same
shared `--pose-file` (`joints_custom.yaml` by default).

`joints_custom.yaml`'s poses were originally taught against cam1's FOV (cam1
came first; cam2/cam3 were added to the SAME pose set later, see
"Calibration pose sets" above). cam2 is mounted steeply/overhead, a visibly
different angle from cam1 and cam3 — so even though cam2 gets its own
independent solve, it's only ever fed a narrower, more oblique subset of
poses that happened to also be visible to it, not poses chosen for its own
geometry. Re-running `--cameras realsense2` against this same pose file
repeats that same ceiling every time, which is the likely reason cam2's
residual (2.12px / 18 samples, the worst of the three — see
`cam_tf_realsense2_2.1px.launch.py`) hasn't improved across repeated
attempts, even though the solver itself (`cv2.calibrateHandEye` Park seed +
joint reprojection refinement with Huber loss + automatic outlier-pose
flagging; validated against synthetic ground truth, see the `METHODS`
comment in `run_handeye_calibration.py`) is sound and the same for every
camera.

**The actual fix is to teach cam2 its own pose set**, watching only cam2's
live detection status while jogging so every saved pose is one where cam2
specifically has a good, close-to-face-on, well-spread view of the board —
not a pose that happened to also catch cam2's edge of frame while chasing
cam1's geometry:

```bash
ros2 run irb120_handeye handeye record-pose --cameras realsense2 \
    --out ~/joints_realsense2.yaml
ros2 run irb120_handeye handeye calibrate --cameras realsense2 \
    --pose-file ~/joints_realsense2.yaml
```

Then sanity-check the factory color intrinsics weren't themselves the issue
(informational only — does not change the written transform):

```bash
ros2 run irb120_handeye handeye diagnose \
    --in ~/handeye_samples_realsense2.yaml --fit-focal
```

Once you trust the new `cam_tf_realsense2_<err>px.launch.py`, it's wired in
at exactly one place: the `cam2` entry of `CAMERAS` in
`irb120_handeye/launch/bringup_cams.launch.py`, which every bringup (production,
calibration, and the FFS offline capture — see `scripts/STEREO_OFFLINE.md`) uses.

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
