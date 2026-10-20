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

## Experimental families and variants (TipT and cylinder)

**Edit sweep values in these files, not in figures or analysis JSON exports:**

- `objects/tipT/configurations.yaml`: `tipT_a` through `tipT_e`, plus opt-in temporary variants.
- `objects/cylinder/configurations.yaml`: `cylinder_a` through `cylinder_e`.

Each file has shared `defaults` and per-variant `configurations`. Variant fields
override defaults; uncertainty mappings can be overridden field by field. Values
use **kg and metres** throughout. `null` means unknown, including uncertainty;
it never means zero. Existing TipT measurements were preserved. Cylinder radii
are populated (0, .008, .016, .024, .030 m); enter the remaining measurements.

Example cylinder entry (numbers below illustrate the format, not measurements):

```yaml
  cylinder_b:
    fillet_radius: 0.008
    mass: 0.364
    com_obj: [0.000, 0.000, 0.080]  # x,y,z from bottom center, metres
    source: "Scale and Onshape export, YYYY-MM-DD; describe print/inserts"
    mesh: null             # optional: meshes/cylinder_b.gltf once exported
    uncertainty:
      mass: null           # optional 1-sigma, kg
      com_obj: [null, null, null] # optional metadata; unused in fixed-GT paper scoring
```

**Frames:** each family declares its frame explicitly:

- Cylinder: `bottom_center_x_along_push_z_up`. The origin is the cylinder axis
  at table height, +x along the push, +z up. Enter Onshape CoM directly when its
  frame matches this convention. For an axisymmetric mass distribution, enter
  `com_obj: [0.0, 0.0, z]`; z is the actual CoM height, not necessarily half the height.
- TipT: `near_outer_base_edge_x_along_push_z_up`. The near edge is the press-pull
  edge; the far edge is `x=base_w`. Existing TipT coordinates are unchanged.

Both files accept `[x,y,z]` in metres (and legacy `[x,z]`); planar analysis uses x,z. The adapter translates centered x into the
near-envelope frame as `x_near = x_center + base_w/2`; z is unchanged. CAD exports
must use the declared origin/axes; assigning a frame name does not transform the
mesh or register an arbitrary CAD origin. `load_configuration()` preserves the
authored coordinates.

**Cylinder support:** the family defaults declare the shared outer diameter
`base_w: 0.080` and `support_model: circular_base_fillet`. Each variant only needs
its fillet radius, mass and CoM height. With bottom-center CoM coordinate x, the
upright flat-support radius is `a = base_w/2 - fillet_radius`; horizontal distances
to CoM are `d_press_pull = a + x` and `d_forward = a - x`. For symmetric x=0,
the five variants give 40, 32, 24, 16 and 10 mm. The loader exposes the results
as `initial_support_distances`, and the suite adapter uses the press-pull distance.
An explicit `support_distances` entry can override a geometric distance if needed.
These are initial support boundaries; migration during rolling is still a model
validity issue. Fillet radii larger than half the outer diameter are rejected;
zero support distance is retained as zero, not converted into a positive lever.

**Meshes:** TipT variants share `meshes/tipT.gltf` because their exterior is the
same. Their mass/CoM still differ. Cylinder exteriors differ, so CAD visualization
or geometry-based evaluation should use the corresponding per-variant export:
place it at e.g. `objects/cylinder/meshes/cylinder_b.gltf`, then set that variant's
`mesh` field. Leave it null until then. Ground-truth entry and numerical scoring
do not require a mesh. The legacy `cylinder.gltf` is not automatically assigned
to any variant because its fillet identity has not been verified.

```python
from irb120_models import list_configurations, load_configuration, configuration_mesh
list_configurations('cylinder')
c = load_configuration('cylinder_b')
c['mass'], c['com_obj'], c['uncertainty'], c['source']
configuration_mesh('cylinder_b')  # absolute Path, or None
```

The existing `load_object('cylinder')` API and `object_models generate` still
refer to the legacy family model; they do not generate per-variant URDFs. Use
`load_configuration()` for experiment IDs. No variant URDF or synthetic mesh is
required for this registry. Newly added meshes/YAMLs are installed by the existing
package setup; rebuild the package before relying on installed share files.

`scripts/analyze_property_suites.py` now reads these YAMLs by default through
`irb120_models.experiment_gt`; rerun it after edits. Its `--references` argument
remains an explicit optional override for comparison experiments. Figures/tables
are generated outputs and must be regenerated after changing references.

For the current paper, object GT and nominal CAD inputs are fixed references;
GT uncertainty metadata is not propagated. The measurement-only collection plan
is `runtime_logs/suite_analysis/UNCERTAINTY_COLLECTION.md` at workspace root.

## Everyday objects (`objects/everyday/configurations.yaml`)

Tier-2 objects: no CAD and no full ground truth. Each is one configuration, named exactly as
its episode label. The file's header comment is the procedure; the essentials:

- **Frame:** the same as tipT, `near_outer_base_edge_x_along_push_z_up`. Fix a placement first (which face points at the robot; write it in `placement` and mark the object).
  - x runs along the push, away from the robot.
  - x = 0 is the near outer extreme of the base (the press-pull pivot side); x = `base_w` is the far one (forward tip).
  - y is to the robot's left, from the footprint's lateral mid-line.
- **What to measure:** the planar CoM, by balancing on a sharp table edge from both sides along the push axis:
  - `support_distances.press_pull`: near outer extreme to the table edge at balance;
  - `support_distances.forward`: the same from the far side;
  - `base_w` with calipers.

  `support_closure()` returns press_pull + forward − base_w, a direct check on the measurement. Record the 1-sigma in `uncertainty.support_distances`.
- **Mass may stay `null`** (sealed until the estimates are frozen); `com_obj` z stays `null` (estimated).
- **These are outer-edge distances.** For a sharp-edged base that is the pivot. For a rounded base the actual support on a flat table is inset; the analysis treats that inset as something to estimate, not something supplied.

```python
from irb120_models.configurations import load_configuration, support_closure
r = load_configuration('mug_blue')
r['support_distances'], support_closure(r)
```
