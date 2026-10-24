# Contact and pivot selection

This document describes the supported FFS snapshot policy. Older selection
studies and rollout chronology live in [the development archive](../docs/archive/README.md).

## Input contract

One stationary segmented object, observed points in metres in `base_link`,
calibrated camera transforms, color-grid metric depth and SAM masks. The nominal
level table is z = -0.021 m; ball radius is 13.25 mm. Camera registration and depth
quality must support the requested contact accuracy. Do not mix observations from
before/after movement or from independent objects.

`sam_shadow` segments calibrated FFS views and builds the observed cloud.
`select_stereo_contacts.py` keeps an actual observation per 3 mm voxel, retaining
source indices. Contacts do not come from mesh closure or averaged invented
surfaces. TSDF mesh rendering is a review artifact; it does not establish accuracy
of the cloud bounds used by a motion primitive.

## Selection

`select_contact_points` supplies planar-push, forward-tip and press candidates.
Each mode records `available` and rejection reasons. Press selection screens
clearance, direction, surface normal, signed moment and footprint inset before
camera agreement and final score/edge-centering decisions. The contact report
shows the candidate funnel spatially and numerically.

The FFS adapter calls `experimental_view_check.evaluate(..., rule='view_check_v2')`.
Each press must be supported by usable local depth-view evidence; unusable or
conflicting views can abstain/reject. The adapter intentionally excludes inferred
horizontal top caps from `top_support_v1`. The underlying older rule helpers remain
for analysis/replay; they are not production backends.

The near press pivot can be replaced by a reviewable RGB-mask/table-plane boundary
fit (`pivot_boundary_review`); the far pivot uses cam3 evidence
(`forward_pivot_review`). If evidence is unavailable, geometry retains its stated
fallback rather than silently claiming a validated physical edge. Pivot geometry
and straight/curved classification feed the tip-choice/cascade helpers.

## Geometry and motion contract

The surface contact `point` differs from the ball-center target `ball_center`.
`press_targets.pre_press_center` and `forward_tip_motion.tip_targets` determine
pre-contact centers. Frame transforms must be explicit; `world` and `base_link`
differ by the robot base/table installation offset.

Green denotes planar push, orange forward tip and magenta press. Contact reports
embed the multi-camera comparison and optional mesh viewer, plus diagnostic JSON.
`available: true` means the implemented filters found a candidate. It does not
certify reachability, complete swept-volume collision clearance, visibility of
hidden support, or no slip.

The pipeline uses the forward arc primitive (`fwd_arc`) for its forward-tip step.
Its shared implementation lives in `fwd_tip.py`; that module remains necessary.
A failed/partial MoveIt plan must not be treated as an executable contact motion.
The separate shove feasibility check has its own timing/scene limitations.

## Remaining limits

- Low grazing views, occlusion, SAM boundary error, depth bias and table-plane
  error can shift pivots. Repeatability and reprojection RMS alone do not prove
  metric support accuracy. The new cam2 extrinsic did not fix its observed FFS
  table-depth bias; see the superseding update in the workflow diagnostics.
- A convex hull closes partial observations; its extreme is not necessarily the
  physical support edge. Rounded bases can migrate during tipping.
- An observed flat contact, geometric torque score or smooth mesh is not evidence
  of successful manipulation or absolute object tilt.
- Preserve acquisition timestamps, source capture identity and the review packet.
  Any object motion invalidates the frozen geometry; reacquire before another
  interaction. Explicit unchanged-scene imports do not waive motion gates.
