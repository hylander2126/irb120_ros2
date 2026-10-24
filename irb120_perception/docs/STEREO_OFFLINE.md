# Offline FFS workflow

The supported path uses three calibrated native IR stereo pairs and RGB, one
stationary object and an arm clear of the views. FFS supplies depth; SAM segments
observed surfaces; contact selection is described in [CONTACT_SELECTION](../CONTACT_SELECTION.md).

## Camera and execution layout

Source ROS Jazzy and the workspace. For robot work use the repository T1/T2/T3
layout; production T3 starts cameras with `camera_profile:=offline_stereo`.
For perception alone, with no other camera drivers running:

```bash
ros2 launch irb120_handeye bringup_cams.launch.py profile:=offline_stereo
ros2 launch irb120_perception stereo_offline_pipeline.launch.py object_name:=flashlight
```

Use the second command from another sourced terminal. It runs capture,
reconstruction, report and contacts in sequence. `timeout:=45` controls capture;
noise/gap-guard options are exposed by `--show-args`. Output lives under
`runtime_logs/<object>/<timestamp>/`. Each step stops the chain on nonzero exit.
No controller targets are published by this standalone launch.

## Individual steps

```bash
ros2 run irb120_perception stereo_offline capture /tmp/capture --stationary-arm-clear
ros2 run irb120_perception stereo_offline reconstruct /tmp/capture --output /tmp/recon
ros2 run irb120_perception stereo_offline report /tmp/recon
ros2 run irb120_perception stereo_offline contacts /tmp/recon --output /tmp/contacts
```

Use fresh directories. The bridge executes the corresponding workspace scripts
under `~/.virtualenvs/irb_venv/bin/python3`; scripts currently reside in
`~/Documents/irb120_ws/scripts`, and Fast-FoundationStereo dependencies/checkpoints
in `third_party/`. Preserve that workspace layout. A fresh clone of src alone is
not a complete FFS installation. The venv needs torch, Open3D, OpenCV, SAM and the
existing stereo dependencies. The optional `stereo_offline worker --help` exposes
the retained warm-worker interface.

Camera acquisition is 1280x720 at 6 Hz; acquisition rate is not inference rate.
Defaults use the existing best-accuracy checkpoint, 16 iterations, 384-pixel
disparity search, 0.2–1.4 m depth and 3 mm TSDF. Check script help for tunable
parameters. `raw_depth` is retained for sensor diagnostics requiring aligned
RealSense depth; the filtered live-cloud backend has been retired.

## Artifacts and episodes

`capture/` retains matched IR/RGB samples, calibrated intrinsics/extrinsics and
source timestamps. `recon/` retains FFS observed geometry, aligned view evidence,
pivot diagnostics, mesh previews/viewers and its report. Its pivot-only
`contacts.json` is not the robot contact selection. `contacts/` contains the actual
push/tip/press selections, per-view evidence, source-point indices, candidate
funnel, multi-camera review and optional contact mesh/viewer.

The reports embed images and link actual mesh/viewer files. Smooth meshes do not
certify point bounds, calibration, hidden support or reachability.

`episode snapshot OBJECT` and pipeline snapshots now invoke these four fresh
steps automatically. Raw runs and logs live under `episode/ffs_runs/`; each
perceive stage receives a frozen contact/review bundle and capture hash.
No cached DBSCAN update or pre-motion geometry is reused. Failed steps create
failed stages and block the next contact motion. The normal operator and MoveIt
motion gates remain responsible for execution; a contact `available` flag alone
is insufficient. An explicitly reviewed unchanged standalone run can also be
imported with `episode snapshot --ffs-run ... --reviewed-unchanged`.

## Diagnostics

[Dated diagnostic history](FFS_DIAGNOSTICS.md) preserves the gap-guard/noise
ablations, press candidate funnel, mesh seam and cam2 calibration investigations.
Per-camera density filtering removes points; it does not invent closure. Gap
guarding affects mesh integration independently from the cloud used for targets.
The 2026-10-24 cam2 extrinsic change did not resolve its table-depth bias; do not
repeat the earlier “recalibrate handeye to fix depth” conclusion.

**cam2 +6 mm Z correction.** cam2's FFS depth reads object-top surfaces about 6 mm
deep (fits a ~0.46 px disparity bias: f·B = 32.4 px·m at ~0.65 m; the same bias
predicts the ~10 mm low table at ~0.9 m). Most objects are about drill height and
cam2 is the only top view, so `bringup_cams.launch.py` raises cam2's static TF by
`cam2_z_offset_mm` (default 6.0; 0 publishes the raw calibration, which stays
untouched). With cam2 corrected, reconstruction keeps cam2 points that cam1/cam3
corroborate or conflict with (`cam2_cross_checks`, default off) and the gap guard
defaults to 0; per-camera bounds, the 4 mm table cut and the noise filter stay on.
The pipeline's own `cam2_z_offset_mm` defaults to 0 so the shift isn't applied
twice. drill_test3-5 (`runtime_logs/`) document how this was found; on drill_test5's
capture the new defaults give press (650.9, -4.3, 209.9) mm, push y -15.4 mm and
no cam2 points below the table.

These are calibrated-rig measurements with learned stereo/SAM priors, not proof
of absolute accuracy on every new object. Keep unresolved uncertainty explicit.
