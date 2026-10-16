# irb120_models

Models for the IRB120 cell: the robot description, and ground-truth models of the
objects it presses, pulls and tips.

```text
urdf/irb120_with_finger.xacro   # robot + F/T stack + finger (use_charuco_mount:=true for calibration)
urdf/*.xacro, urdf/*.urdf       # ABB IRB120 macro, ros2_control block, sensor/adapter and finger assemblies
meshes/                         # robot, sensor and finger meshes
objects/<name>/                 # one ground-truth object per folder (below)
```

The robot description moved here from `irb120_control/urdf` and `irb120_control/meshes` on
2026-10-09. Its expanded URDF is unchanged apart from the package name in `package://` URIs.
Consumers find it with `get_package_share_directory('irb120_models')`:

- `bringup_stack.launch.py`;
- `abb_control.launch.py` (`description_package`);
- the hand-eye bringup;
- MoveIt's `.setup_assistant`;
- the perception robot mask filter, which loads the collision meshes from `meshes/irb120_3_58/collision`.

## Objects

The object models replace `irb120_control/object_params.json`.

**Ground truth is the measured mass and CoM.** Most of the everyday objects in these
experiments have no CAD model, and never will. So nothing depends on CAD:

- The mesh is optional.
- θ\* is *derived* from the measured CoM: the sharp-edge balance angle
  atan(com_x / com_z) about the tip edge.
- Where a CAD model exists (soda, monitor, flashlight), its values go in a `cad:` block.
  They are informational only.

```text
objects/<name>/
├── <name>.yaml          # the record you edit: measured mass and CoM, tip edges, sources
├── <name>.urdf          # GENERATED (only when there is a mesh or full inertial data)
└── meshes/              # optional
    ├── <name>.gltf      # CAD export (metres), the mesh source
    └── <name>.stl       # GENERATED from the glTF
```

## Why YAML plus a generated URDF

A URDF has a link with mass, CoM, inertia and a mesh, and every tool here reads it. It
has no place for:

- **Uncertainty and provenance.** `object_params.json` stored bare numbers. Its θ\*
  values came from CAD and don't follow from its measured CoM values (0.3–1.5° apart),
  which is why they are now kept apart.
- **How the object rests on the table** (`rest_pose`).
- **Which edge it tips about** (`tip_edges`). The estimator takes `com_x`, the
  horizontal distance from the tip edge to the CoM, as *known*, so the edge is part of
  the ground truth.
- **Friction.**

So the YAML is the authored record, and the URDF and STL are derived from it. `check`
fails when they are stale.

## Frames

- **Body frame:** for an object with a mesh, the mesh's CAD frame. For one without
  (`body_frame: tip_edge`, the normal case), the origin is on the default tip edge at
  table height, with x horizontal toward the CoM and y along the edge. The measured
  `com` is then [com_x, 0, com_z] directly.
- **Support frame:** the table plane at z = 0, with z up. `rest_pose` (xyz plus URDF rpy)
  maps body to support for the pose the experiments use.
- **Tip edges:** horizontal lines in the support frame. `ground_truth(edge)` returns
  `com_x`, `com_z` and the sharp-edge θ\* = atan(com_x / com_z) for that edge.

For a meshed object whose CoM was measured against a tip edge rather than in the CAD
frame, the `recorded` block holds that measurement. `ground_truth()` returns it with
`source: recorded`.

## Workflow

To add an object without CAD (the usual case), copy `objects/box/box.yaml` to
`objects/<name>/<name>.yaml`. Fill in the measured mass and `com` = [com_x, 0, com_z]
relative to the edge it will tip about, with sources. No mesh or URDF is needed.

With a CAD mesh:

```bash
source /opt/ros/jazzy/setup.bash && source ~/Documents/irb120_ws/install/setup.bash
ros2 run irb120_models object_models new mug --mesh ~/Downloads/mug.gltf --label "Mug"
#   ...edit objects/mug/mug.yaml: mass, com, rest_pose, tip_edges, sources...
ros2 run irb120_models object_models show mug      # compare with the uniform-density centroid
ros2 run irb120_models object_models generate mug
ros2 run irb120_models object_models check         # also run by the package tests
```

Run these against the source tree. With `colcon build --symlink-install`, the installed
module resolves to `src/`, so `generate` writes there. Rebuild after adding an object,
so that its files are installed.

`show` prints the CoM a uniform-density solid would have, and the com_x and θ\* that
CoM would give. A hollow or weighted object will legitimately differ. A centimetre-scale
gap or a sign flip usually means the CoM was entered in the wrong frame.

Set `status: complete` once the mass, the CoM and `tip_edges.default` are measured. A
mesh is not required. The loader enforces this.

## Using it from Python

```python
from irb120_models import load_object
m = load_object('flashlight')
m.ground_truth('default')   # {'mass', 'com_x', 'com_z', 'theta_star_deg', 'source', ...}
m.urdf, m.stl, m.mesh       # paths
```

Outside ROS, add `src/irb120_ros2/irb120_models` to `sys.path`. The loader then finds
`objects/` next to the module. `irb120_control.util.runtime_log_dir.load_object_params()`
wraps this and returns the old `object_params.json` keys, so `arc_static` and the
episode records are unchanged. Per-object controller settings (`force_ref_n`) are in
`irb120_control/config/object_control.yaml`, not here: they are tuning, not ground truth.

## State on 2026-10-09

Every object is `provisional`.

| Object | What's there | What's missing |
|---|---|---|
| box, monitor, soda | Measured mass and CoM in the tip-edge frame; CAD θ\* | Monitor and soda have CAD, but their meshes aren't in this package. Box has no CAD. |
| heart, flashlight | Measured mass and edge-relative CoM, plus a mesh (from `object_meshes_testing`) | `rest_pose` and `tip_edges` for the mesh are placeholders |
| cylinder | A mesh | Its measured values are placeholders (ground truth in progress) |
| tipT | A mesh | Everything else |

The monitor is a known press-pull failure case: the finger slips during the arc. That
motivates forward tipping (`fwd_tip.py`) for it.

**Not done yet:** `estimate_params.py` still keeps its own `GROUND_TRUTH` table, which
holds the same measured mass and CoM values as these models. Its θ\* is derived from the
measured CoM, as here.

Forward-tip experimental configurations are defined in `objects/tipT/configurations.yaml`, with an explicit experiment frame separate from the unregistered CAD frame. `irb120_models.experiment_gt` supplies estimator scoring references. Temporary configurations are opt-in and excluded from the default suite.
