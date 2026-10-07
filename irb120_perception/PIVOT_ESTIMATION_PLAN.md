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

This document describes proposed work. The existing production path still uses the robust 3-D support-envelope estimate for the online pivot seed, while the parameter estimator defaults to trajectory-based pivot recovery after an arc trial.
