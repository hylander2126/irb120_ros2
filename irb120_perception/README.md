# irb120_perception

The supported perception path is stationary native-stereo capture → Fast
FoundationStereo (FFS) depth → SAM segmentation → observed object cloud →
contact/pivot selection and review. The live DBSCAN, standalone live SAM detector,
streaming stereo prototype and RealSense-only TSDF tools have been retired.
Their implementations remain in Git history; see [the archive](../docs/archive/README.md).
SAM segmentation and TSDF fusion used by offline FFS remain active.

## Run perception

For perception without robot bringup, start the three calibrated cameras:

```bash
ros2 launch irb120_handeye bringup_cams.launch.py profile:=offline_stereo
```

With one stationary object and the arm clear, in another sourced terminal:

```bash
ros2 launch irb120_perception stereo_offline_pipeline.launch.py object_name:=flashlight
```

The robot's `bringup_stack.launch.py` also starts this camera profile by default;
do not start a second camera driver session. Results are written beneath
`runtime_logs/<object>/<timestamp>/{capture,recon,contacts}/`. Review the embedded
the contact image in `contacts/README.md`. Meshes and reconstruction HTML
reports are optional (`review:=true`); routine press diagnostics are disabled.

The `stereo_offline` CLI also supports individual `capture`, `reconstruct`,
`report`, `contacts`, and `worker` steps. [The workflow guide](docs/STEREO_OFFLINE.md)
covers commands, requirements, checkpoint settings and diagnostics.

## Episodes and motion

`episode snapshot OBJECT` and the motion pipeline now capture fresh FFS runs
automatically. They retain `snapshot.json`, `contacts.json`, `object_cloud.npz`,
calibrated image evidence, selection graphics and source capture identity in each
perceive stage, with the raw capture, reconstruction and step logs in its `capture/`,
`recon/` and `logs/` subfolders (written in place, never copied). No DBSCAN node or perception activation service is needed.

A snapshot does not command motion. Motion primitives still apply their existing
contact checks, MoveIt approach/collision checks and operator gates. Frozen contact
geometry becomes invalid after motion; subsequent snapshots acquire new stereo
frames. Explicit imports of existing reviewed runs are also supported; see
[control usage](../irb120_control/README.md).

## Geometry and diagnostics

[Contact selection](CONTACT_SELECTION.md) defines the current observed-surface
selection policy and limitations. Shared pure Python modules remain useful for
recorded clouds and analysis: `contact_point_selector`, `sam_shadow`,
`stereo_geometry`, `pivot_boundary_review`, `forward_pivot_review`,
`experimental_view_check`, and `contact_graphics`.

FFS contact selection uses `view_check_v2`; it does not use inferred horizontal
caps from older top-support experiments. A reviewable RGB/table pivot is not
independent physical ground truth. Camera disagreement and partial support
visibility remain material uncertainties.

The command bridge currently expects workspace scripts in
`~/Documents/irb120_ws/scripts`, model dependencies under `third_party/`, and
`~/.virtualenvs/irb_venv/bin/python3`. This is a workspace deployment, not a
self-contained pip package. Missing scripts/interpreter fail the perceive stage
and leave logs; no fallback detector is started.
