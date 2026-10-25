# Forward arc tipping

Build `irb120_control`, source the workspace, then run:

```bash
ros2 run irb120_control fwd_arc monitor --episode last --max-angle 30
```

Use your object label instead of `monitor`. The approach, operator check, tare,
contact seek, weak-contact rejection, slide probe, force/staleness/drift stops,
camera and wrench/joint logging, EGM cleanup, and verified return before homing
are shared with `fwd_tip`. The robot is commanded only when you run the primitive.

Add a Marker display for `/fwd_arc/trajectory` in RViz (fixed frame `world`).
The retained preview and run metadata contain the arc points. The initial arc
uses the selected forward-tip ball center and estimated pivot edge; after touch,
the measured ball center anchors the circle to avoid a position discontinuity.
The arc lies in the plane normal to the pivot axis. The finger holds its initial
orientation. It retraces the circle to touch, then the approach line to pre-tip.

Default speed is 5 mm/s along the path. `--max-angle` caps the arc after contact
(default 30 degrees); there is no separate travel cap (fwd_tip's `--max-travel` is
not offered). Reaching the cap returns with an unsuccessful result. `--force-limit`,
`--tip-force`, `--quality`, and `--episode` retain their forward-tip meanings.

A successful reversal requires five fresh samples at or below 10% of the
experienced peak absolute pivot-axis torque, after the existing minimum tip
travel and contact loading checks. `--reverse-fraction` changes this fraction.
Torque uses the full wrench, including sensor moment and the sensor-to-pivot
moment arm. The current arc controller still uses the historical +82.25 mm
tool0-X wrench-origin offset; the estimator now derives its reference from the
ball-to-mount geometry. That discrepancy needs a separate mechanical review
before interpreting their torque values as identical.
`fwd_arc_*` logs include pivot torque, peak torque, arc angle, actual contact
anchor, path error, and trajectory geometry in a `forward_arc` episode stage.
The recorded arc angle describes finger motion; it is not independently measured
object tilt. Existing estimation pipelines that select `forward_tip` stages do
not automatically consume these separate comparison runs.

This fixed-orientation arc reduces ideal sliding along the object face, but
retains ball-surface slip. It assumes a stationary pivot and does not certify
object restoration or eliminate the contact/angle biases discussed in
`runtime_logs/tipT_bias_audit/CONTACT_LOCATION_AND_ARC_TIP.md`.
