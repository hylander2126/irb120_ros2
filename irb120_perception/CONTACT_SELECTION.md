# Contact point selection

`contact_point_selector.py` is an independent geometry selector for one segmented
object. It leaves the legacy press selector, launches and motion controllers alone.
No motion is commanded. The pure Python function needs NumPy and SciPy, not ROS.

## Current camera coverage

Both installed cameras are on the robot side of the object. The far-workspace
camera has **not** been added, so the far support boundary used for forward
tipping is not directly visible. Fusing the two current views does not recover
that hidden boundary.

The script currently closes a convex hull around the **observed** low points.
Its far-facing hull edge can therefore be an artificial boundary of the partial
cloud, not the object's actual far tipping edge. Treat the orange pivot/axis as
an unverified geometric hypothesis. `available: true` means a candidate passed
the implemented filters; it does not establish visibility or pivot accuracy.
There is no visibility/confidence flag or hidden-surface reconstruction yet.

The high, robot-facing forward-push contact may still be useful even when the
far pivot is wrong. For the horizontal directions and `axis = up × direction`
used here, the PDF's moment score simplifies to contact height above the pivot.
Consequently, a plausible high contact does **not** validate the pivot's XY
location. Do not use this far-axis estimate to define an executed tipping arc
without additional geometry evidence. The near-side press/pull pivot is more
observable, but still depends on adequate points close to the table.

```python
from irb120_perception.contact_point_selector import select_contact_points

result = select_contact_points([camera1_object_xyz, camera2_object_xyz], table_z=-0.021)
if result['planar_push']['available']:
    contact = result['planar_push']['point']
    fingertip_center = result['planar_push']['ball_center']
```

Clouds must describe the same stationary object in one common Z-up frame, in
metres. Register and segment them before calling; do not concatenate independent
objects or observations from before and after object motion. Duplicate points are
removed. Optional `normals=` accepts outward normals in concatenated input order.
Otherwise local PCA estimates them and orients them away from the cloud median.
This orientation assumes a reasonably observed, approximately convex object;
concavities and strongly partial views may need externally supplied normals.

After building/sourcing `irb120_perception`:

```bash
ros2 run irb120_perception contact_point_selector                    # one message from /object_detector/object_points
ros2 run irb120_perception contact_point_selector cloud1.npy cloud2.npy   # or Nx3 .npy files already in base_link
```

Markers are always published (snapshot, until Ctrl-C):

In RViz, use **Add → By topic → /contact_point_selector/markers → MarkerArray**.
Use `base_link` as the Fixed Frame, or `world` with the stack's TF available.
Green is planar push, orange is forward tip, and magenta is press. Spheres mark
the **surface contact**, not the fingertip center. Arrows at contacts show the
push/pull direction (the press arrow shows the subsequent pull); arrows at the
table show the signed tipping axes, lifted 8 mm for visibility only. Axes remain
visible even when their contact is unavailable and are labeled accordingly.
The JSON reports the rejecting filter and candidate counts after each stage.

This displays one **snapshot**, republished once per second until Ctrl-C, so
RViz can be opened later. Rerun after moving the object. Markers expire within
three seconds after stopping. No robot motion is commanded.

The topic path reads one message and prints JSON, then publishes markers. It does
not enable the perception stack. Use `--object-id N` for a labeled cloud containing
multiple objects. Incoming frames must be `base_link`; no silent TF conversion
occurs. File inputs must be Nx3 `.npy` arrays in that frame. All tuning values
(table height -0.021, finger radius, bands, etc.) are the fixed defaults of
`select_contact_points()`; call it from Python to change them.

## Heuristics

The tipping score follows *Probing for Properties*, supplied as `main.pdf`,
Section IV, equations (2)–(7): screen by `direction · normal <= epsilon`, then
maximize `axis · ((point - pivot) × direction)`. Planar-push clearance, footprint
estimation, side/top filtering and the inset are implementation heuristics.
The PDF assumes an expected pivot; this script's partial-cloud hull is an
approximation used to supply one. The PDF's interaction-mode selection in
Section IV-C is not implemented: the script reports candidates for all modes.

- **Planar push:** inward-facing side for +X, within 5 mm of y=0, minimum contact
  height. Exact height ties prefer y closest to zero, then minimum x. Points remain
  actual sensed points; y is not forced to zero off the observed surface.
- **Pivot and direction:** build a 2D convex hull using only points within 20 mm
  above the table (3 mm below is allowed for noise). The facing hull edge nearest
  +X supplies the forward-tip pivot and outward direction; repeat toward -X for
  press-and-pull. A facing edge at least 30 mm long and within 45 degrees of the
  preferred direction is an extended edge. Otherwise use an extreme support
  vertex and a tangent perpendicular to the preferred direction. The signed axis
  is `up × direction`. These are geometric estimates, not measured pivots.
- **Forward tip:** maximize the PDF's `axis · ((point - pivot) × direction)` on
  inward-facing side surfaces. This retains the PDF's zero normal tolerance and
  additionally excludes tangential/top contacts that cannot provide a side push.
- **Press:** use the same score for the estimated pull geometry, normal tolerance
  0.1, and upward-facing surfaces. Inset the feasible top region's XY convex hull
  by 5 mm, then retain points within 3 mm of the best remaining moment arm and
  pick the observed point nearest their median. The inset is a simple heuristic
  for contact-patch room.

Every mode returns `available`, `point`, `normal`, `ball_center`, and `score` when
successful. Tip and press also return `geometry` with `pivot`, `axis`, `direction`,
`kind` and edge endpoints (or null for isolated pivots). Scores have metre units
per unit applied force; the planar score is negative height above the table.
Failures return `available: false` and a reason rather than inventing a contact.
No mode is automatically executed or chosen from the three results.

## Clearance and limits

The fingertip radius is 13.25 mm from `irb120_with_finger.xacro`. For each candidate,
`ball_center = contact + radius * outward_normal` must be at least radius + 3 mm
above the table. `--min-ball-center-z` can raise this floor to a known safe tool
height for a chosen orientation. Radius, buffer, bands and tolerances are CLI
options / function keywords; see `--help`.

This checks the spherical fingertip only. It does **not** calculate home/downward
orientation, IK, wrist/FT-body clearance, reachability or approach-path collisions.
Those depend on the robot configuration and must be checked by MoveIt when this
selector is connected to motion. A downward shaft orientation can allow a lower
contact, but is not assumed to make the full robot collision-free.

The low observed band approximates the support footprint; sloped bases, feet,
occlusion and outliers can bias it. Insufficient or degenerate low-band points
make tipping selections unavailable. A partial but nondegenerate band can still
produce an incorrect pivot, particularly on the hidden far side described above.
There is no fallback to the object's full silhouette. Convex hulls do not model
holes or concave top boundaries. A 5 mm inset
is not a contact mechanics or finite-patch guarantee. Refresh observations after
the initial planar push before using a tipping contact or pivot.

## Legacy migration status

The active calibration check now uses `select_contact_points(...)["press"]`:

```text
arc_static.py / arc_static_batch.py
  → util/press_point_check.py
    → contact_point_selector.select_contact_points()
```

The check preserves perception activation, prominent-object selection, the 5 cm
vertical standoff, 5 cm 3D comparison tolerance, CSV/run metadata, and the
`/press_point_marker` recorder topic. It transforms the whole cloud into `world`
before selection with table z=0. The comparison uses surface contact plus vertical
standoff, not ball center plus standoff. The calibrated controller motion target
is unchanged. An unavailable press selection fails the check with its reason and
candidate counts; there is no fallback to the old heuristic.

The legacy `press_point_selector.py` and executable remain available for explicit
compatibility use, but are no longer imported by the active press check. Their
nearest-X preference differs from the new inset/median choice. Existing calibrated
poses may therefore pass or fail differently, and missing support/top observations
can now fail the check rather than produce a height-only estimate. This migration
changes the estimator, not the check into a motion-target generator.

The new forward-tip axis remains an unverified partial-cloud hypothesis with the
current camera layout. No far-axis estimate is used to command motion by this check.
