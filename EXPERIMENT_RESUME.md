# Resume: experimental camera-filtered press selection, 2026-10-05

## Implemented

Temporary experimental_view_check_v2; off by default. Same environment switch:
IRB120_EXPERIMENTAL_VIEW_CHECK=shadow|apply|off. Shadow saves proposed contacts
without changing executable contacts.json. Apply writes the experimental press
selection there. Existing snapshots are not retroactively changed.

V2 calls experimental_press_filter from select_contact_points AFTER geometric
feasibility/inset and BEFORE the unchanged score. Only press is changed. Camera
filter excludes uncertain/rejected points. No relaxed-normal retry or topmost
fallback is allowed in experimental mode. No logistic regression or controller
changes. Original fused normal/ball-center calculation remains unchanged.

Each camera uses a 4-connected neighborhood around the candidate's projected
pixel, within 12 mm of its 3-D location and 8 mm of pixel depth. Fits at 8 and
12 mm must each have 20 samples and pass planarity; normal angles must agree
within 20 degrees and dot-product signs must match. Existing 5 mm visibility,
25 degree intercamera agreement, and strict dot <= 0 rules remain. One usable
view can pass; no usable views or conflicting views abstain. These thresholds
are experimental, not calibrated uncertainty estimates.

Outputs: contacts_baseline.json, contacts_experimental_view_check.json,
experimental_view_check.json (all candidate decisions and evidence), and
camN_experimental_view_check.png (green pass, red reject, yellow uncertain,
magenta selected). Normal contact overlays match applied contacts in apply mode.

## Live test performed

New perception-only episode runtime_logs/flashlight/episodes/20261005_184639.
Ran shadow mode; no robot motion. 810 object points; snapshot succeeded.
15 candidates: 10 pass, 0 reject, 5 uncertain. Baseline winner was uncertain due
to camera disagreement. Proposed point moved 6.51 mm from baseline, to
[0.6347687244, 0.0004975633, 0.1717733145] in base_link metres.
Selected point: camera 2 only, 337 samples, depth residual -0.20 mm,
normal dot proposed pull -0.1701. Cameras 1 and 3 abstain on depth mismatch.
Inspected cam2 candidate overlay; points lie around cap. Not a manipulation
success result. First sandbox-blocked attempt left failed episode 20261005_184628;
ignore it for geometry evaluation (ROS socket access unavailable).

22 selector/experiment tests pass. Both ROS packages rebuilt. Tests verify
filtering precedes scoring, all-rejected cannot fall back, default selectors
remain valid, depth occlusion abstention, and normal disagreement/rejection.

## Tomorrow

1. Review 184639/00_perceive experimental overlays and candidate evidence.
2. Stay in shadow until comfortable with proposed point/ball center. To collect:
   source install/setup.bash
   export IRB120_EXPERIMENTAL_VIEW_CHECK=shadow
   ros2 run irb120_control episode snapshot flashlight
3. Apply requires a NEW snapshot with switch set to apply; do not assume shadow
   contacts are used for motion. Do not reuse contacts after moving the object.
4. Execution limitation remains: arc_static is fixed XZ, while selector pull
   direction can rotate with tangent. Resolve before claiming geometric
   consistency or executing a rotated trajectory. No hardware test done here.
5. Evaluate multiple poses and failures, not just this snapshot. Camera fits can
   still mix curved surfaces; fused ball offset may still be wrong even when
   camera checks pass. Saved images are not guaranteed synchronized.

Discard experiment by removing experimental_view_check.py, its tests, the
experimental_press_filter hook, and the marked snapshot experiment block.
Default behavior remains unchanged when no switch is set.
