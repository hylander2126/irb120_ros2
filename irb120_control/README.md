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

## Forward tip (MVP)

```bash
ros2 run irb120_control fwd_tip monitor
ros2 run irb120_control fwd_tip monitor --episode last
```

Requires the normal bringup with MoveIt Servo enabled (`start_servo:=true`).
Uses the saved `forward_tip` contact and pushes horizontally normal to its
pivot edge, with fixed finger height and orientation. After confirmed contact,
it reverses when the absolute force along the push direction falls to **10% of
its running peak** for five fresh samples, after at least 2 mm of tipping travel.

It tells a tip from a slide or a near-topple from force against travel
(`util/forward_tip_motion.py`, closed-loop checks in `test/test_forward_tip_physics.py`):

- **Contact** is the median of five fresh samples at 0.05 N (noise is ~0.012 N).
  The force must then reach `--tip-force` (0.3 N) within 1.5 mm. If it holds a
  real but smaller force, the object is close to toppling forward (or too light
  for the force drop to stand out of the noise): it backs off as `too_easy`.
  tipT_e toppled under 0.09 N; the replay backs off about 2 mm before it fell.
- **Slide probe**: at 10 mm of travel it backs off 2 mm once. A tilted object
  rests on its edge and follows the finger back under gravity (tipT_c kept ~60%
  of its force on the way back). A slid one stays put and the force vanishes
  within 0.2 mm (tipT_d). Under 30% left: `slide`, and the median force while
  sliding is saved as `slide_force_n` (the friction sample) with `slide_travel_m`.
  A slow tip (balance far away) looks flat at first but passes the probe.
It returns along the same line to the measured pre-tip pose, then goes home.
Gravity must restore the object; completing the finger return does not prove
that an object which slid or passed its balance point has untipped.

Defaults: 5 mm/s, 80 mm maximum travel after contact, 20 N resultant force limit.
Options: `--speed` (m/s), `--max-travel` (m), `--reverse-fraction`, `--force-limit`
(N), `--episode new|last|PATH`, and `--quality h264|lossless`.
The normal pre-motion confirmation and unloaded F/T tare run at the pre-tip pose.

Travel/forward-time limits reverse and mark the attempt failed. Excess force,
stale feedback, or height/orientation/lateral drift stop motion; home is skipped
unless the finger's return to the pre-tip pose was verified. Runs save a
`forward_tip` episode stage, F/T and pose NPZ, motion diagnostics, metadata and
videos, followed by a snapshot after verified home. `run_pipeline` chooses it
when there is no press contact, or after a press-pull that failed or slipped
(see "Which tip" below).

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
The move home plans around the same box (stretched `PUSH_DISTANCE` in +X after a
push). The push pre-pose clears the padded box by only 10 mm, so OMPL's
collision-check spacing (`longest_valid_segment_fraction` in
`irb120_moveit_config/config/ompl_planning.yaml`, 0.001 ≈ 6 mm of finger travel)
must stay well below that; at the old 0.005 (~30 mm) paths clipped the box and
MoveIt's post-plan validation rejected them (`error_code=99999`,
`INVALID_MOTION_PLAN`). As a backstop, pose and joint goals
(`moveit_single_shot.py`) replan up to `plan_retries=3` times on a planning
failure; this is plan-only and never moves the robot. These failures are MoveIt
planning, not EGM: each motion script starts its own EGM session
(`egm_cond_time`, default 360 s).

```bash
ros2 run irb120_control episode snapshot flashlight           # perception only, no motion
ros2 run irb120_control run_pipeline heart_a --trials 10      # tip then push, pauses between steps, asks for the outcome
ros2 run irb120_control push flashlight                        # or step by step, into a new episode...
ros2 run irb120_control arc_static flashlight --episode last   # ...and continue it
ros2 run irb120_control arc_static_batch flashlight --trials 10
ros2 run irb120_control episode note "object slid ~5 mm sideways during push"
ros2 run irb120_control episode outcome success --reason "tipped past theta* and settled back"
ros2 run irb120_control episode show
```

### Step order: tip first, then push

`run_pipeline` tips before it pushes by default (`--order push-tip` for the
other way round). Reasons:

- The tip is non-destructive by design: it returns the object to rest, so the
  push afterwards still sees roughly the original pose. The push moves the
  object 80 mm and can yaw it, and the press contact and pivot are selected
  for the pose the object started in. Non-destructive measurements first.
- The push can tip the object instead of sliding it: its lowest allowed ball
  center is 81 mm above the table (`MIN_BALL_CENTER_Z`), high relative to the
  CoM of small objects.
- The push measures the sliding force mu*m*g. Turning that into mu needs the
  mass, which the tip estimates.

The case for pushing first is that it might show which tip suits the object
(forward/top-push vs press-pull). It gives a one-sided hint at best. A
forward push at height h tips the object (rather than sliding it) when
mu > x_c / h, which doesn't depend on mass and gets easier the higher the
contact. So if the low push already tips the object, a forward tip from the
higher forward contact would too. If the push slides, though, that says
nothing about the higher contact, and the sliding force alone can't be turned
into mu without the mass.

### Which tip: decided by the forward pivot's shape

`choose_tip()` in `orchestrator.py` picks the first tip from the forward-tip
pivot alone. That pivot is the far (+X) support edge, which only cam3 sees. Each
snapshot fits it from a cam3 SAM mask (`irb120_perception/forward_pivot_review.py`)
and records `forward_tip.pivot_shape` in `contacts.json`:

- **straight** (an extended edge; a line fits the support over >= 40 mm and >= 80%
  of it): forward tip first. Its fixed-pivot push matches that edge.
- **curved** (a convex tangent, or a straight run too short: a point contact) or
  **unknown** (no cam3 mask, or no fit): press-pull first. It has the estimator and
  a controlled return arc, and its pivot is the near edge that cam1/cam2 see.

Either way, the other tip runs first when the preferred one has no contact.
Every snapshot also makes this first-tip choice itself (`util/tip_choice.py`) and
records it as `tip_choice` in `snapshot.json`; `episode snapshot` prints it.
`cam3_forward_pivot_review.png` shows the fit; `camN_forward_contact_review.png`
shows the forward contact, its pivot, and the pre-tip and contact fingertip spheres.
The fit always runs; when it is not reviewable the shape is unknown and the forward
pivot stays the cloud estimate.

Whether press-pull works depends on the mass, which is unknown until it has run:
the finger slips when the pull needed exceeds mu_finger times the press force
(the monitor). So slip is detected afterwards, not predicted. After a press-pull,
the orchestrator takes the spread of |ball - commanded pivot| over the last
attempt's ARC/UNARC. Without slip the ball stays on a circle about the pivot. If
the spread is over 4 mm (`estimate_params.NOSLIP_TOL_MM`), that counts as a slip.

After a failed tip, the tip cascade (`util/tip_cascade.py`; design notes in
`runtime_logs/ADAPTIVE_PRESS_IDEAS.md`) picks the next attempt from why it failed,
each on the snapshot taken after the one before, up to 4 attempts:

- press-pull slipped (lost contact at the 13 N cap, or spread > 4 mm) or failed:
  a riskier press contact from the snapshot's `press_alternatives` (more torque per
  pull, at most 15 mm sideways), starting at the force the slipping attempt ended
  on; then a forward tip (also on a curved pivot); then stop.
- forward tip hit its force limit, drifted, pushed its full travel without tipping,
  slid the object (`slid`), was `too_easy`, or failed: press-pull, conservative
  contact first, then riskier ones; then stop. A slid forward tip is the episode's
  slide interaction (event `slide interaction: <stage>`); the push phase is then
  skipped under `--auto`, or offered as skippable (`s`) in step mode. A full-travel
  push after which the upright object shifted more than 10 mm also counts as `slid`.
- contact missed: the same attempt once more. Sensor fault: the same attempt again.
- object moved more than 10 mm (other than a forward-tip slide) or its top height
  changed more than 10 mm: stop.

An arc that ends before the tangent-force flip (arc limit, or the tangent force
collapsing) counts as a tip; the summary says which (`arc_exit` in the stage results). In step mode
only the first attempt waits at the gate. `--tip press|forward` keeps the cascade
to that tip. `arc_static --press-candidate K` runs the K-th press alternative.

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

`run_pipeline` runs this tip-then-push sequence (`--order push-tip` for the old order).
If the object is manually repositioned, start a fresh snapshot/episode before
using contacts again. Do not reuse a snapshot from before repositioning.

## Camera-validated top-support selection (default)

Snapshots use the top-support candidate search and individual camera depth
validation by default. Push and forward-tip contacts remain unchanged. No
environment variables are required for normal operation.

There are no on/off switches: every snapshot runs this selector. Its evidence is
in `experimental_view_check.json` (the historical file name).

Gate: valid depth within 5 mm of predicted surface; 12 mm local PCA neighborhood,
at least 20 samples, surface variation <= .05, and view cosine >= .3. Only usable
views participate. Any normal dot proposed pull > 0 rejects; pairwise normal
disagreement above 25 degrees or no usable view abstains (unavailable in apply).
One usable view can pass, explicitly labeled single-view support. Each camera's
depth is corrected by its locally fitted table plane before the 5 mm comparison.
Per-camera timestamp agreement is not checked; use a stationary object. Camera
checks use the saved proposed direction, not a new robot trajectory.

Performance: the view check runs once per snapshot. Each
`check_view` call back-projects only a pixel window around the candidate's
projection that provably contains every pixel within 12 mm (plus any table-plane
correction), instead of the whole depth image. Results are identical to the
full-image computation. This cut a perceive stage from ~55 s to ~15 s; the
remainder is mostly DBSCAN settling and the short bag. Derivation and
measurements: `../NOTES.md` (2026-10-09).

Offline replay (writes contacts_experimental_view_check.json and
experimental_view_check.json, not contacts.json):

```bash
python3 -m irb120_perception.experimental_view_check PATH_TO_PERCEIVE_STAGE
```

The files retain their historical `experimental_*` names, but the top-support
path is now the runtime default. No controller modifications are involved.

## RGB boundary pivot (press)

Every snapshot first fits the press pivot where the cam1 SAM mask meets the table
(`pivot_boundary_review.json`, `cam1_pivot_boundary_review.png`). When that fit is
`reviewable`, the press is selected once, against it; otherwise against the cloud
estimate. `snapshot.json` `rgb_pivot.applied` says which, and the press's
`geometry.pivot_source` is `rgb_sam_override` or `cloud_estimate`. There is one
contacts file, `contacts.json`, and `arc_static` / `episode preview` read it.

### V2 update (supersedes the v1 gate above)

The default path filters all geometrically feasible press candidates
before scoring. Relaxed/topmost fallbacks are disabled only in experimental mode.
Connected 8/12 mm depth neighborhoods must give consistent normal signs and
angles. Candidate evidence and color overlays are saved for every camera.
See `../RESUME_EXPERIMENT_AGENTS.md` for implementation details, the live shadow test,
remaining execution limitations, and tomorrow's resume instructions.

### Preview the saved press before motion

`ros2 run irb120_control episode preview --episode <episode-folder>` publishes
the latest successful saved perception stage to the existing RViz MarkerArray
`/contact_point_selector/markers`. It performs no capture or robot motion.
Cyan is the pre-press ball (30 mm above the contact ball), magenta is the
contact ball at its actual radius, yellow is the surface contact, and orange
is the pivot. The label identifies the episode, stage and contacts file.
It reads `contacts.json`, the same file arc uses. Stop preview
with Ctrl-C before running arc; arc publishes its own transformed command
targets on the same topic. Markers expire when publication stops.

Preview and arc share target calculation; the final-selection camera image
now explicitly shows both the cyan PRE ball and magenta contact ball. The
image is a projection using saved camera calibration, not independent 3D
ground truth. Contact estimation and the 30 mm approach offset are unchanged.
Arc reuses a snapshot only if perceive is the episode's last stage; otherwise
it captures a new snapshot. To inspect a target and then execute that exact
snapshot, preview a newly captured episode and pass the same episode to arc.

Approach diagnostics are written automatically to the motion stage's
`approach_diagnostics.jsonl`. Records include the exact contacts file and
preview/calibration file hashes, saved contact, base-to-world transform, target
frame/link/orientation, timestamped EE TF and its age, measured joints and
controller reference/feedback at 5 Hz, and MoveIt's action status plus planned
and returned executed trajectories. Before/after `/compute_fk` queries use
measured joints and MoveIt's model, independently of availability of the EE TF.
Signed actual-minus-target XYZ error is printed in millimetres. Missing TF/FK
evidence is explicitly logged; action success is not labeled pose verification.
Logs are appended throughout approach, before any later home attempt. This
adds diagnostics only; target selection and motion commands are unchanged.

### Cable-neutral wrist limit

Joint 6 is hardcoded to [-60, +60] degrees about zero in the robot URDF,
MoveIt joint limits, and approach/home path constraints. Pose approaches also
constrain tool orientation throughout the path within 60 degrees per rotation-vector
component of the requested final orientation. Free-space moves plan first and
check every returned joint-6 waypoint before execution; out-of-range angles,
missing wrist data, and adjacent wrist jumps over 60 degrees are rejected.
Angles are never wrapped or recentered. Starts outside the wrist window fail.
Collision avoidance remains active; no clearance padding was changed.
Restart the running bringup/MoveIt/Servo stack after rebuilding to load the new
robot model. These are software planning bounds, not ABB safety-rated limits.

### Analyze complete tipT episodes

The episode estimator reads the stage manifest, including forward arcs and
press-pull fallbacks. It fits the two gravity first moments separately on the
tip/return branches and jointly derives mass and CoM height, assuming the
configured horizontal CoM distance. GT mass and height are used only for
scoring, never as fit inputs, initialization, or corrections. Press-pull uses
the opposite support edge when looking up horizontal CoM distance.

From the workspace root, using the source package directly:

```bash
PYTHONPATH=src/irb120_ros2/irb120_control:src/irb120_ros2/irb120_models \
python3 -m irb120_control.estimation.estimate_params --primitive episodes \
  --episode latest --vision --out-dir runtime_logs/estimation_episodes
```

`--episode all` includes aborted episodes and preserves failed/skipped stages;
a timestamp selects one episode. `--logs-root` also accepts a copied suite
containing `<object>/episodes/`. `--object tipT_d` restricts the configuration.
JSON preserves branch fits, first moments, diagnostics, scoring, controller
outcomes and quality flags; CSV contains the primary estimates.

`--tilt-source proprio` is the default. `--vision` additionally fits camera tilt;
`--tilt-source vision` explicitly selects it for every fitted primitive, and
`--tilt-source forward_vision` selects it only for forward stages. An unavailable
requested camera fit is reported, without silently substituting proprioception.
`--vision-cache-dir` can reuse saved tracks. Camera fitting uses its moving pivot
and a 1 mm/s slide-speed gate; an arc's finger travel is never treated as measured
table translation. Without vision its translation is explicitly unobserved.

A small torque-fit residual does not validate angle zero, effective support or
wrench calibration. Quality flags report angular spans below 5 degrees and
median spherical-contact wrench disagreements above 5 mN m; these are diagnostic
thresholds, not calibrated acceptance limits. The measured torque is retained.
No empirical gain/angle/contact correction or GT-mass-conditioned height is
applied. Existing `press_pull` and `fwd_tip` modes remain available; the latter
now discovers forward-arc files as well as straight-tip files.

The episode forward estimator now uses **first contact** as the upright face
reference. A later torque maximum is not an upright observation; reproduce that
older assumption with `--forward-reference torque_peak`. When pre-tip translation
is observed/estimated, the upright face distance comes from the original contact
geometry while the torque reference follows the translated support.

Both tipping loaders now recover the distal-face wrench reference from the
logged ball pose minus the rigid 86.6512 mm finger length. The fixed historical
82.25 mm tool0 offset did not match the current CAD adapter stack (85.8244 mm
plus its transverse offset). Wrench axes and recorded torque channels are
unchanged. This also preserves the old stack when replaying old ball/tool poses.

`--press-contact-model sphere_roll` enables an **experimental** planar model
for a world-fixed sphere rolling without slip on an initially horizontal top.
The ball centre is not a material point on the object, so its polar angle is
not exactly the object's angle. The default remains the older point model;
curved/sloped contacts and compliance need separate treatment.
`--press-pivot-offset-x METRES` supplies an explicit experimental support offset;
it is recorded in the JSON and is never fitted to GT mass/height. An offset
learned from one object configuration is not a global robot calibration.
See `runtime_logs/tipT_mechanics_audit_20261016/REPORT.md` for simulation,
3/5 N training, withheld 4 N evaluation and the remaining D limitation.
