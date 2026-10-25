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
reconstruction and contacts in sequence. `timeout:=45` controls capture;
`review:=true` additionally builds meshes and reconstruction reports;
noise/gap-guard options are exposed by `--show-args`. Output lives under
`runtime_logs/<object>/<timestamp>/`. Each step stops the chain on nonzero exit.
No controller targets are published by this standalone launch.

## Individual steps

```bash
ros2 run irb120_perception stereo_offline capture /tmp/capture --stationary-arm-clear
ros2 run irb120_perception stereo_offline reconstruct /tmp/capture --output /tmp/recon
ros2 run irb120_perception stereo_offline contacts /tmp/recon --output /tmp/contacts
```

Use fresh directories. The bridge executes the corresponding workspace scripts
under `~/.virtualenvs/irb_venv/bin/python3`; scripts currently reside in
`~/Documents/irb120_ws/scripts`, and Fast-FoundationStereo dependencies/checkpoints
in `third_party/`. Preserve that workspace layout. A fresh clone of src alone is
not a complete FFS installation. The venv needs torch, Open3D, OpenCV, SAM and the
existing stereo dependencies.

`bringup_stack.launch.py` starts the warm worker (`ffs_worker:=true`, default): it
loads FFS+SAM once and the pipeline's perceive steps run through it (about 12 s per
perceive plus the capture, against ~21 s with fresh processes). Without it the
pipeline warns and falls back to fresh subprocesses; `snapshot.json` records which
ran (`ffs_runner`). Run it alone with `ros2 run irb120_perception stereo_offline worker`;
its interface is in `scripts/stereo_offline_worker.py`.

Camera acquisition is 1280x720 at 6 Hz; acquisition rate is not inference rate.
Defaults use the existing best-accuracy checkpoint, 16 iterations, 384-pixel
disparity search, 0.2–1.4 m depth and 3 mm TSDF. Check script help for tunable
parameters. `raw_depth` is retained for sensor diagnostics requiring aligned
RealSense depth; the filtered live-cloud backend has been retired.

## Artifacts and episodes

`capture/` retains matched IR/RGB samples, calibrated intrinsics/extrinsics and
source timestamps. `recon/` retains FFS observed geometry, aligned view evidence,
pivot evidence and timing/provenance JSON. Meshes and rendered reconstruction
reports are optional (`reconstruct --mesh`, then `report`). Its pivot-only
`contacts.json` is not the robot contact selection. `contacts/` contains the actual
push/tip/press selections, per-view evidence, source-point indices and the
multi-camera contact image. Press candidate diagnostics are no longer generated.
Contact mesh rendering requires `contacts --review-mesh` and an existing mesh.

Reconstruction publishes one shared `object_cloud.npz`: actual per-camera 3 mm
samples, with isolated samples (no other sample within 10 mm) removed once.
`object_cloud.json` records rejected points, camera/source indices and bounds.
The handoff links the same files without another data copy on the same filesystem.
Contacts, pivots and motion collision shapes use these accepted observations;
motion primitives do not run independent cloud cleanup. Old unvalidated snapshots
require a fresh perceive. Restart the warm worker after updating reconstruction code.

The reports embed images and link actual mesh/viewer files. Smooth meshes do not
certify point bounds, calibration, hidden support or reachability.

`episode snapshot OBJECT` and pipeline snapshots now invoke these three fresh
steps automatically, writing into the perceive stage itself: `capture/`, `recon/`,
`logs/`, and the frozen contact/review bundle and capture hash at the stage root.
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
**cam2 calibration and correction.** Capture checks every camera-link TF against
the calibration selected in `bringup_cams.launch.py`, records the actual cam2 Z
shift, and rejects conflicting or stale static publishers. On 2026-10-24, the T
capture was found to contain the older cam2 calibration and no intended +6 mm
shift: an orphaned publisher was still active. Earlier conclusions about the
new calibration's effect on those captures were therefore not established.

The MVP uses **+4 mm relative to the current cam2 calibration**. The new calibration
itself raises the tested T top roughly 2 mm relative to the old calibration, so
adding the old +6 mm again would overcorrect it. This is a measured local-height
compensation, not a verified disparity calibration for all ranges.
`bringup_cams cam2_z_offset_mm` defaults to 4.0. Reconstruction's
`cam2_z_offset_mm` defaults to `auto`: it subtracts the shift recorded in the
capture from the desired +4 mm total. For example, a TF already shifted +4 mm
receives zero additional shift; a TF shifted +6 mm receives −2 mm. A numeric
reconstruction argument remains an explicit **additional** shift for experiments.
Legacy captures without TF-audit metadata require that explicit numeric argument.

Restart camera bringup after changing calibration. Do not leave standalone camera
static publishers running alongside it. The capture error names stale/conflicting
frames instead of silently choosing whichever publisher arrived last.
The normal run command remains:

```bash
ros2 launch irb120_perception stereo_offline_pipeline.launch.py object_name:=tipT_test3
```

The bounds, table-height cut and noise filters remain active; cross-camera point
suppression and mesh gap guarding remain disabled. FFS press validation uses the
same observed coordinates as reconstruction and the same small-tilt tolerance as
the geometric selector. See [contact policy](../CONTACT_SELECTION.md).

These are calibrated-rig measurements with learned stereo/SAM priors, not proof
of absolute accuracy on every new object. Keep unresolved uncertainty explicit.
