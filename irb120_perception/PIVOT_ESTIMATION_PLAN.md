# Pivot Estimation Plan

## Goal

Estimate the physical tipping pivot accurately for previously unseen objects without CAD models or object-specific geometry rules.

The main opportunity is to use RGB semantics where depth cameras are weakest: at the discontinuity between the object's bottom and the table. Metric geometry must still come from camera calibration and the measured table plane, not from an image annotation or language model directly.

## Proposed Architecture

Use three complementary estimates rather than asking one sensor or heuristic to solve the entire problem:

1. **Cam1 RGB silhouette/table boundary — primary online pivot seed**
   - Use the existing MobileSAM object mask.
   - Extract the lower object contour where the mask borders the table.
   - Back-project this image-space boundary through the calibrated Cam1 model.
   - Intersect its rays with the fitted table plane to obtain metric 3-D support geometry.
   - Fit either a line for a flat support edge or a tangent point and lateral tangent for a rounded support.
   - Select the support nearest the robot in the requested `-X` tipping direction.

2. **Fused 3-D cloud — geometric corroboration and fallback**
   - Confirm that the inferred boundary lies near the observed object envelope.
   - Reject disconnected image artifacts and gross RGB/depth disagreement.
   - Retain the current robust support-envelope pivot when the RGB boundary is unavailable or uncertain.

3. **Executed end-effector arc — post-motion pivot correction**
   - Recover pivot X independently from the measured ball-center trajectory.
   - Pin pivot Z to the commanded table-plane pivot because the short arc poorly constrains the radial direction.
   - Gate recovery using the existing circularity/no-slip test.
   - Fall back to the logged commanded pivot when the trajectory is not trustworthy.

Do not jointly optimize pivot and CoM in the torque fit. The torque data primarily observes CoM relative to the pivot, so fitting both together creates a poorly identifiable degree of freedom.

## Why Cam1 Is the Primary View

Cam1 directly observes the object/table interface and therefore has good leverage on the support boundary. Cam2 remains valuable for top-surface and press-point geometry, but an overhead camera often cannot see the physical bottom contact because the object itself occludes it.

Camera contributions should therefore be selected by observability rather than weighted equally for every geometric quantity.

## MVP Algorithm

1. Obtain the prompted MobileSAM mask already used by the DBSCAN/SAM culler.
2. Keep only the mask component associated with the selected 3-D object cluster.
3. Extract the lower mask contour.
4. Score each contour segment using generic evidence:
   - object depth immediately inside/above the contour;
   - table depth immediately outside/below the contour;
   - connection to the main object mask;
   - proximity to the calibrated table-plane projection;
   - agreement with the fused 3-D object envelope.
5. Back-project accepted pixels and intersect their rays with the table plane.
6. Robustly fit the resulting 3-D support samples:
   - a supported line indicates a flat tipping edge;
   - a compact support region indicates a rounded/point contact, with its tangent oriented laterally.
7. Choose the support extreme in the requested tipping direction.
8. Emit the pivot, tangent/edge, support type, confidence, and diagnostic evidence.
9. Fall back to the existing 3-D support-envelope estimate if confidence is insufficient.

The image-space line must never simply be assumed horizontal in the world. Calibration and table-plane intersection handle camera roll, perspective, and object orientation.

## Generic Failure Rejection

The estimator should explicitly reject or down-weight:

- shadows mistaken for object pixels;
- tape or table markings;
- masks that bleed into the table;
- missing or invalid depth around the boundary;
- boundaries disconnected from the selected object;
- RGB/depth disagreement beyond a metric threshold;
- support estimates that disagree strongly across adjacent frames.

These checks should produce a confidence value and evidence record rather than object-name conditionals.

## Diagnostic Output

Save a LabelMe-like JSON record with image geometry separated from inferred physics:

```json
{
  "image": "cam1_color.png",
  "image_width": 640,
  "image_height": 480,
  "geometry": {
    "object_table_boundary_px": [[308, 365], [340, 376], [380, 367]],
    "pivot_edge_base_m": [[0.592, -0.041, -0.021], [0.592, 0.038, -0.021]]
  },
  "physics": {
    "selected_pivot_base_m": [0.592, 0.0, -0.021],
    "support_type": "rounded_tangent",
    "confidence": 0.89
  },
  "source": "cam1_mask_table_intersection"
}
```

Also render a review image showing:

- the accepted object/table contour;
- rejected contour segments;
- the inferred image-space tangent;
- the projected 3-D pivot and edge;
- confidence and fallback reason.

## Validation

Use the available meshes only as ground-truth benchmarks, never as production perception inputs.

For each known object:

1. Compare the RGB-derived pivot with the mesh-derived ground truth.
2. Compare it with the fused-cloud pivot from the same snapshot.
3. For completed robot trials, compare both online estimates with the trajectory-recovered pivot.
4. Report errors in pivot X, lateral centering, edge orientation, and downstream press selection.
5. Include repeatability across new snapshots, lighting changes, tape/table clutter, and filtered versus raw depth.

## Implementation Order

1. Save the Cam1 MobileSAM mask with each perception snapshot.
2. Implement an offline boundary extractor and review overlay for existing episodes.
3. Benchmark it against mesh ground truth and trajectory estimates.
4. Add confidence/fallback logic.
5. Promote the estimator to the default online pivot seed only after the offline benchmark demonstrates improvement across the geometry set.

## Current Status

Experimental capture and offline review are implemented (2026-10-07). Production
pivot/contact selection remains unchanged. Both ROS packages have been rebuilt;
a running culler needs a restart before it emits the new diagnostics.

### Capture and review

- The culler publishes `~/sam_mask` and `~/sam_color`, associated with the input
  cloud timestamp, plus `~/sam_metadata` preserving the actual RGB acquisition
  timestamp. These do not change the culled cloud.
- Snapshots save a matching `cam1_sam_mask.png`, `cam1_sam_color.png`, and optional
  `cam1_sam_metadata.json`. Older cullers still support normal snapshots.
- The offline reviewer associates the mask component with projected cloud points,
  extracts bottommost pixels, and intersects their half-pixel transitions with
  the measured table plane. It screens against the cloud envelope, probes depth
  above/below the boundary, and requires a connected run of supporting samples.
- Robust line / local convex-quadratic fitting produces a candidate pivot and
  tangent. Timing, cloud disagreement, and table-fit sensitivity are recorded.
- `pivot_boundary_review.json` separates the candidate from the retained cloud
  pivot. Physical confidence remains zero: `reviewable` means the experimental
  gates passed and the candidate is exposed for an explicit operator-reviewed
  trial; it is not a calibrated confidence.
- `cam1_pivot_boundary_review.png` shows screened/rejected contour samples,
  the candidate tangent/pivot in cyan, the original cloud pivot in magenta,
  and a magnified boundary panel. No `contacts.json` changes are made.
- New snapshots run this shadow path by default and write
  `contacts_rgb_pivot.json` only for a reviewable result. `arc_static --pivot
  rgb` opts one press-and-pull trial into that file; `--pivot cloud` remains the
  default and uses `contacts.json`. The RGB variant changes only the arc pivot,
  preserving the selected press, ball center, normal, direction, and fixed-XZ
  controller axis.

From the workspace root:

```bash
PYTHONPATH=src/irb120_ros2/irb120_perception python3 -m irb120_perception.pivot_boundary_review /path/to/00_perceive
```

Older episodes need explicitly labeled offline SAM replay from RGB; the cloud
alone cannot reconstruct the silhouette. Replay scripts and evidence are in
`runtime_logs/pivot_boundary_validation_20261007/` at the workspace root.

### Validation outcome — do not promote online

See [the visual report](../../../runtime_logs/pivot_boundary_validation_20261007/REPORT.md).

- 18 real bag-frame mask replays, 6 episodes / 4 geometries; 17 RGB-D pairs within
  40 ms. Within-episode candidate X ranges are 0.01–0.50 mm. Repeatability is
  not accuracy: cylinder candidates differ from the cloud by about 32 mm.
- Four ideal mesh silhouettes at known nominal poses: RGB/cloud absolute pivot-X
  errors in mm are cylinder 2.94/19.73, flashlight 0.25/2.17, T 0.05/0.50,
  and heart 4.46/0.30. The heart also has a large lateral error. This uses exact
  lowest-support vertices, not the legacy cloud selector as ground truth.
- An older flashlight trajectory passes the existing no-slip gate, but its
  recovered X differs from RGB by 9.74 mm. Its historical commanded Z differs
  from the RGB table Z; trajectory recovery is a conditional comparison, not
  an independently measured contact location.
- 69 perception/pivot tests pass, including culler timestamp/payload preservation.
  Five injected-failure checks pass: missing depth/mask, table bleed, stale
  depth, and disconnected artifacts. Two ROS packages build successfully.
- No live camera capture, robot motion, or online promotion was performed.

The first screening-only implementation accepted stray side-contour samples and
occasionally called the flat T rounded. Depth probes and a stricter curvature
criterion addressed those observed failures. These thresholds were adjusted on
this dataset and still need held-out validation.

### Remaining work

Independent support annotations or registered mesh poses are needed for real
snapshots, especially the cylinder and heart. Visible bottom silhouettes can
belong to rounded/occluding undersides above the actual contact. Short-term
repeatability cannot resolve that bias. Cam1 also amplifies a 1-pixel contour
shift into approximately 3–3.5 mm in X, and a 1 mm table-height shift into
approximately 3.7–4 mm in X.

Still pending: arbitrary-roll boundary extraction, online temporal consistency,
calibrated confidence, additional poses/lighting/clutter, raw-versus-filtered
depth comparisons, and validated downstream contact selection. The lower-per-
column extractor and vertical depth probes assume the current Cam1 view.
Distorted depth is rejected because the reused table fitter assumes rectified
aligned depth. Angled diagnostic edges must not be handed to the fixed-XZ
controller without resolving that geometric mismatch.
