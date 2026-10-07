# Offline contact validation

## Latest: local-support wrench optimization (2026-10-07)

The default snapshot path now runs `experimental_local_support_v4` in apply mode;
the historical environment switches remain available only for rollback and
comparison. Candidates must pass top-slope, local projected
ball-center support, and usable-camera consistency checks. Selection scores the
3-D force `pull + downward press` about the estimated pivot, retains candidates
within the larger of 3 mm absolute or 5% relative score loss, then centers along
the pivot edge. This replaces the compact-object central-top ellipse, allowing
the same geometry-driven rule to handle both flat and rounded tops.

When fused-cloud PCA yields no upward-facing samples, V4 may infer a horizontal
cap from a well-supported upper rim. These inferred candidates are never accepted
from geometry alone: each still needs a usable depth view with a consistent local
top normal. Isolated/curved pivot tangents use the robust lateral center of the
full observed support footprint instead of one or two extreme samples; a strongly
supported straight edge continues to override that estimate with its midpoint.

Each enabled top-support evaluation now emits `PRESS_REVIEW.md`, JSON evidence,
a CSV feedback template, up to three camera montages (at least 2 mm separation),
and pivot/pull geometry. Review preferred / acceptable / bad / unsure, or none
acceptable. Feedback preserves geometry IDs and does not apply contacts; this is
not an interactive blocking picker. Graphics are flat projections. No new depth
PNGs are written. Shadow mode keeps executable contacts at the baseline.

The superseded five-scene central-ellipse replay and its limitations remain at:
`runtime_logs/flashlight/central_top_20261006/FINDINGS.md`.
Camera disagreement and candidate loss from invalid normals remain unresolved.

The sections below describe earlier study stages; this update supersedes their
statements about dotted rendering and absence of automatic candidate packets.

Current update: dotted occlusion rendering was withdrawn after side-view depth
disagreement made the preferred successful contact confusing. All review and
snapshot contact graphics now use flat projection. New snapshots emit no depth
PNG or depth-preview PNG; camera checks consume in-memory metric depth and the
bag retains raw topics. Historical images are not deleted.

Top-support v1 applies slope, local support, and camera
consistency gates before signed torque ranking. Its replay and 120 noise trials
are in `runtime_logs/flashlight/top_support_20261006/RESULTS.md`. It recovers
full_success A but still chooses badpressselection B instead of preferred C;
held-out episode 20261006_144117_newrule has a proposal awaiting human review.
This is not a validated replacement for execution.

From the workspace root, generate review options and sensitivity evidence:

```bash
PYTHONPATH=src/irb120_ros2/irb120_perception python3 -m irb120_perception.contact_validation \
  --episodes runtime_logs/flashlight/episodes \
  --output runtime_logs/flashlight/contact_validation_new \
  --trials 20
```

The output directory must not exist. This module does not launch ROS, move the
robot, modify saved episodes, or change production selector behavior. It reads
the three October 5 development scenes listed in `SCENES`. Input artifact hashes
are checked at completion. Saved raw metric depth is read for the camera checks;
no new depth images are emitted.

Open `index.html` in a browser for up to three options per scene. A is the recorded
contact. B prefers the 20-degree or 30-degree selector winner when available;
otherwise it is the least-sloped camera-passing alternative. C favors torque.
Alternatives must be at least 4 mm apart. They are review examples, not necessarily
safe or better than A. A missing alternative means no distinct passing candidate
was available. Image panels show projected contact, ball silhouette/center, and
vertical descent; occluded geometry is still drawn.

Return judgments as `scene, option, preferred/acceptable/bad/unsure, reason`, using
`feedback.csv` or chat. The HTML text fields do not persist or submit responses.
All options may be bad. The selector cannot be scored for acceptability until
these labels exist. Selected-point motion is reported as sensitivity, not failure.

`review.json` contains all candidate evidence, slope variants, and an offline
support-proxy ablation. The proxy checks whether the ball center projects inside
the local observed XY hull; it can bridge holes and is not a collision check.
Neither this proxy nor the slope restrictions are enabled in production.

`RESULTS.md` and `cloud_summary.csv` summarize repeated cloud perturbations and
fixed-cloud camera perturbations. Detailed trials include recomputed pivot and
direction changes. Cloud and camera perturbations are separate experiments, not
an end-to-end sensor simulator. Thresholds and stress magnitudes are uncalibrated.
The existing 0.7 upward-normal cutoff is preserved by the nominal 46-degree
variant; tighter variants use 30 and 20 degrees. Torque ratio remains 1.

The first completed review is `runtime_logs/flashlight/contact_validation_20261006_v2`.
Updated human rankings after viewing pivot geometry: full_success A;
badpressselection C; 184639 tentatively C with B also a plausible choice.
These are preferences, not a validated new selector or motion result.

Future contact geometry drawings use depth-aware styling: solid when visible,
dotted behind measured surfaces, and dim dotted where depth is unavailable.
The 3 mm depth tolerance is a visualization choice, not a calibrated confidence
interval. `contact_graphics.py` implements this without rerunning selection.
Perceive now emits `camN_contact_review.png` with the final selected press,
ball, descent, pivot tangent and pull; snapshot.json records filenames and timing.
In shadow mode these depict executable baseline contacts, not experimental
proposals; in apply mode they depict applied contacts. This does not automatically
generate the three alternative options or run the offline noise suite per snapshot.
Offline render/read/write time was about 0.05 s for three cameras. Both packages
rebuilt; renderer tests and saved-frame snapshot overlay replay passed. Live
capture has not been tested with this addition.

It finds a regression case requiring human review: V2 rejects the recorded contact
from the successful episode, and its remaining candidates disappear under the
20/30-degree limits or the support proxy. Do not promote these constraints based
only on improvements to the bad-selection scene. Review labels, then tune one
rule at a time and evaluate a fresh held-out pose. Rotated selector directions
still differ from the controller's fixed-XZ arc; these are perception-only tests.
