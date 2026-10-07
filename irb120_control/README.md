# irb120_control

This package contains the ROS 2 bringup and control wiring for the real ABB IRB120 setup in this workspace.

## Important: local copy of vendor hardware interface

This setup depends on a local package named `irb120_abb_hardware_interface`, which is a copy/fork of the vendor package `abb_hardware_interface`.

The local copy is required for this project-specific behavior:

- Custom RWS credential pass-through from launch and xacro parameters.
- Local plugin identity used by this package (`irb120_abb_hardware_interface/ABBSystemHardware`).
- Command output remains disabled until a complete, finite EGM feedback sample
  seeds every commanded joint from its measured position.
- EGM loss or session restart latches command output off; a recovered session is
  rearmed only after the controller command again matches measured position.
- Recoverable EGM loss keeps joint-state and TF reporting alive while dropping
  writes to the robot.
- Avoiding direct edits to vendor packages in `src/abb_ros2/...`.

If `irb120_abb_hardware_interface` is missing, not built, or not sourced, real robot bringup from this package will fail.

## Notes

- This package intentionally keeps vendor packages unmodified.
- Bringup launch files in this package are tailored to this robot cell configuration.
- The plugin still uses RWS on the physical controller to discover its mechanical
  units by default; it does not require RobotStudio. The alternate
  `configure_via_rws=false` path is not the supported IRC5 configuration.

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
# Offline CPU TSDF reconstruction

With the normal camera/perception stack running and the object stationary:

```bash
ros2 run irb120_control offline_tsdf --duration 3
```

It is pseudo-online: the command captures live RGB-D frames for a short window,
masks them from the active SAM-culled cloud, fuses a CPU TSDF, then exits. No
bag, episode, or saved snapshot is read. The wrapper uses the Open3D virtualenv
only for this separate process, not the live control nodes.

Outputs are saved to `runtime_logs/tsdf/<timestamp>/`: `object_mesh_colored.ply`
(vertex colour), `object_mesh.stl` (geometry-only for basic STL viewers),
`object_surface_colored.ply`, and `report.json`. Use `--output-dir`,
`--voxel-m`, or `--max-frames-per-camera` to override defaults.

Mesh export keeps the largest connected surface (by area), rejects statistical
outliers, and applies 10 Taubin smoothing iterations before constructing a
watertight convex hull. This assumes a single object and fills **all** concavities,
including real ones; occluded geometry and its interpolated colors are estimates.
Use `--completion none` for the cleaned observed surface, and
`--smooth-iterations 0` to disable smoothing. `object_mesh_raw.ply` preserves the
unprocessed TSDF mesh; `object_surface_colored.ply` remains the raw observed
point cloud. `report.json` records filter counts and final watertightness.


## Flashlight MVP rerun

Push remains along +X. Candidates must have inward normals within 20 degrees
of +X; among candidates in the lowest 3 mm height band, prefer alignment.
A missing candidate means reposition and take a new snapshot. Saved contacts
are checked again before motion. This reduces oblique pushing but does not
guarantee zero yaw. Return-arc augmentation is disabled; the ordinary radial
PID and contact-loss handling remain active.

1. Source `install/setup.bash`, place the flashlight near the centerline with
   a side facing the robot, and keep the usual camera/Servo stack running.
2. Run `ros2 run irb120_control episode snapshot flashlight`; inspect the
   contact overlays and `contacts.json` in the new episode.
3. Run `ros2 run irb120_control arc_static flashlight --episode last`.
   Check return-arc hunting and that the object settles back.
4. Inspect the automatically saved post-tip contacts before running
   `ros2 run irb120_control push flashlight --episode last`.
5. Record the outcome with `ros2 run irb120_control episode outcome success
   --reason "your observations"` (choose the appropriate outcome).

Use these staged commands for tip-then-push; `run_pipeline` still pushes first.
If the object is manually repositioned, start a fresh snapshot/episode before
using contacts again. Do not reuse a snapshot from before repositioning.

## Camera-validated top-support selection (default)

Snapshots use the top-support candidate search and individual camera depth
validation by default. Push and forward-tip contacts remain unchanged. No
environment variables are required for normal operation.

For rollback/debugging, `IRB120_EXPERIMENTAL_VIEW_CHECK=off` restores baseline
selection, while `shadow` records the camera-validated result without applying it.
`IRB120_EXPERIMENTAL_PRESS_RULE=view_check_v2` selects the older camera-only rule.

```bash
export IRB120_EXPERIMENTAL_VIEW_CHECK=shadow  # optional diagnostic override
ros2 run irb120_control episode snapshot flashlight
```

Review `experimental_view_check.json`, `contacts_baseline.json`, and
`contacts_experimental_view_check.json` in the new perception stage. Baseline
`contacts.json` is unchanged in shadow mode. In apply mode it contains the gated
selection, and saved overlays are regenerated accordingly. Overrides affect only
new snapshots and do not retroactively change an existing episode.

Gate: valid depth within 5 mm of predicted surface; 12 mm local PCA neighborhood,
at least 20 samples, surface variation <= .05, and view cosine >= .3. Only usable
views participate. Any normal dot proposed pull > 0 rejects; pairwise normal
disagreement above 25 degrees or no usable view abstains (unavailable in apply).
One usable view can pass, explicitly labeled single-view support. Each camera's
depth is corrected by its locally fitted table plane before the 5 mm comparison.
Per-camera timestamp agreement is not checked; use a stationary object. Camera
checks use the saved proposed direction, not a new robot trajectory.

Offline replay (writes only experimental artifacts, not contacts.json):

```bash
python3 -m irb120_perception.experimental_view_check PATH_TO_PERCEIVE_STAGE
```

The files retain their historical `experimental_*` names, but the top-support
path is now the runtime default. No controller modifications are involved.

## RGB boundary pivot trial

New snapshots run the Cam1 RGB boundary estimator in shadow mode by default.
The normal `contacts.json` remains the cloud-pivot path. When the RGB candidate
passes its timing, depth, table-plane, connectivity, fit, and disagreement
gates, the stage also contains:

- `pivot_boundary_review.json` — both pivots and all gate evidence;
- `cam1_pivot_boundary_review.png` — magnified comparison overlay;
- `contacts_rgb_pivot.json` — the same selected press and ball center, with only
  the arc pivot replaced by the RGB candidate.

Collect and inspect a motion-free snapshot first:

```bash
ros2 run irb120_control episode snapshot flashlight
```

If the review JSON says `candidate_status: reviewable` and the overlay is
physically sensible, run the tip in that same episode explicitly:

```bash
ros2 run irb120_control arc_static flashlight --episode last --pivot rgb
```

The command fails closed before motion if `contacts_rgb_pivot.json` is absent.
Use `--pivot cloud` (the default) for the original pivot. If the object moves
after the snapshot, take a new snapshot instead of reusing either contact file.
Set `IRB120_RGB_PIVOT=off` before a new snapshot to disable these artifacts.

### V2 update (supersedes the v1 gate above)

The default path filters all geometrically feasible press candidates
before scoring. Relaxed/topmost fallbacks are disabled only in experimental mode.
Connected 8/12 mm depth neighborhoods must give consistent normal signs and
angles. Candidate evidence and color overlays are saved for every camera.
See `../EXPERIMENT_RESUME.md` for implementation details, the live shadow test,
remaining execution limitations, and tomorrow's resume instructions.
