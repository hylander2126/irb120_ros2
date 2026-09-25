# irb120_control

This package contains the ROS 2 bringup and control wiring for the real ABB IRB120 setup in this workspace.

## Important: local copy of vendor hardware interface

This setup depends on a local package named `irb120_abb_hardware_interface`, which is a copy/fork of the vendor package `abb_hardware_interface`.

The local copy is required for this project-specific behavior:

- Custom RWS credential pass-through from launch and xacro parameters.
- Local plugin identity used by this package (`irb120_abb_hardware_interface/ABBSystemHardware`).
- Avoiding direct edits to vendor packages in `src/abb_ros2/...`.

If `irb120_abb_hardware_interface` is missing, not built, or not sourced, real robot bringup from this package will fail.

## Notes

- This package intentionally keeps vendor packages unmodified.
- Bringup launch files in this package are tailored to this robot cell configuration.

## Arc press behavior

`arc_static` uses the selected object's `force_ref_n` as its initial press
force. If radial contact is lost during ARC or UNARC, it retracts, returns to
the pre-squash pose, multiplies the force by 1.25, and retries up to the 13 N
adaptive ceiling. Other failures retract and stop without increasing force.

```bash
ros2 run irb120_control arc_static monitor
```

The ROS parameter form remains supported for launch files and other ROS-native
configuration, but it is not required for normal command-line use.

## Running experiments: episodes

Every run is an *episode*: one folder with everything it produced plus a
summary, `episode.json` (git commit, camera extrinsics, each stage's
status/results/files, notes, outcome):

```
runtime_logs/<object>/episodes/<YYYYmmdd_HHMMSS>/
  episode.json
  00_perceive/        cloud (DBSCAN -> SAM cull), contacts.json, per-camera color/depth/overlay PNGs, short bag
  01_push/            F/T + pose .npz, metadata .json, cam1/2/3 videos
  02_perceive/
  03_press_pull_tip/
  04_perceive/
```

There are no hardcoded poses: each motion script takes its targets from the
episode's latest perception snapshot (taken first if needed) and takes another
after returning home. `push` pushes at the detected `planar_push` contact;
`arc_static` squashes at the detected `press` contact and arcs about its pivot.
The approach moves plan around the detected object (its padded bounding box is
a MoveIt collision object during the approach only; `util/object_obstacle.py`).

```bash
ros2 run irb120_control episode snapshot flashlight           # perception only, no motion
ros2 run irb120_control run_pipeline flashlight --trials 10   # push + press-pull tip, asks for the outcome
ros2 run irb120_control push flashlight                        # or step by step, into a new episode...
ros2 run irb120_control arc_static flashlight --episode last   # ...and continue it
ros2 run irb120_control arc_static_batch flashlight --trials 10
ros2 run irb120_control episode note "object slid ~5 mm sideways during push"
ros2 run irb120_control episode outcome success --reason "tipped past theta* and settled back"
ros2 run irb120_control episode show
```

`estimate_params` reads episode logs as well as the older
`<object>/push` and `<object>/arc_squash` folders.
