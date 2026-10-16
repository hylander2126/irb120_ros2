# Engineering Notes

Running log of non-obvious findings, decisions, and open questions. Newest entry first.
Things that are obvious from the code do not belong here — this is for the things that
cost time to discover and would cost time again.

---

## 2026-10-09 — RViz hangs on the splash logo: remove `Transformation:` from .rviz files

### Symptom

RViz (from `bringup_stack.launch.py`) shows the splash logo and never opens the main
window. Intermittent. The last RViz log line is the MoveIt panel's
`MoveGroup namespace changed: / -> . Reloading params`, which looks like the culprit
but isn't. 0% CPU; `move_group` is healthy and answers services. RViz may later die
with exit -11 (segfault).

### Cause (rviz2 Jazzy 14.1.23 / MoveIt 2.12.4)

A deadlock on the main thread while the config is loading, captured by running RViz
under gdb (`ptrace_scope=1` blocks attaching to a running RViz, but gdb can start it):

    main: VisualizationFrame::loadDisplayConfig → VisualizationManager::load
          → TransformationManager::load → setTransformer → transformerChanged
          → (Camera display) message_filters::Signal1<Image>::removeCallback
          → pthread_mutex_lock   ← blocked forever; no other thread holds it
    moveit: PlanningSceneDisplay::loadRobotModel → waitForAllMainLoopJobs ← waits on main

The `.rviz` block

      Transformation:
        Current:
          Class: rviz_default_plugins/TF

re-selects the default TF transformer, but `TransformationManager::load` runs it
*after* every display has already subscribed. Each tf-filtered display (Camera,
PointCloud2, MarkerArray) then detaches its message-filter callback. If data arrives
on those topics at the wrong moment, that lock never frees. It's a timing race:
the same config hung on some starts and opened 9/9 on others.

### Fix

Delete that 3-line block. Then `setTransformer` runs only once, in the
`TransformationManager` constructor, before any display exists (checked with gdb
breakpoints: 2 calls → 1). Removed from `moveit.rviz`, `moveit_handeye.rviz`,
`moveit_handeye_NEW.rviz` and `record_depth_both.rviz`.

**RViz's File → Save Config writes the block back.** If the hang returns, check
first: `grep -n -A2 "Transformation:" */rviz/*.rviz`.

### Debugging tips if it's something else next time

- Don't debug RViz from the VS Code terminal: the snap environment breaks it
  (`symbol lookup error: /snap/core20/.../libpthread.so.0`). Use a normal terminal.
- Getting a backtrace of the hang: `gdb -batch -ex run -ex "thread apply all bt"
  --args rviz2 -d <cfg> --ros-args --params-file <moveit params>`, then send
  SIGINT to rviz2 once it hangs.
- To bisect, start only RViz against the live stack with the same MoveIt params as
  bringup (`MoveItConfigsBuilder(...).to_dict()`). Without `robot_description_semantic`
  the MoveIt panel never loads the robot and the hang can't happen, so standalone
  `rviz2 -d moveit.rviz` tests prove nothing.

---

## 2026-10-09 — Perceive ~55 s → ~15 s; push "EGM bail" was a MoveIt plan failure

### Perceive time was `check_view`, not the cameras

File mtimes in a perceive stage give the real timeline. In episode
`tipT_a/20261009_162353/00_perceive` (55 s total): ~6.5 s DBSCAN settle + SAM
cull (`wait_s`), ~4.5 s short bag, ~2 s save, then **~42 s of post-processing**:
view check 15 s → pivot boundary review 14 s → view-check reseed 13 s.

All three are the same `experimental_view_check.evaluate()` (the pivot review's
SAM-pivot comparison and the RGB-pivot reseed each run it again), and almost all of
its time was `check_view()`. For every press candidate × camera (~700 calls per
pass) that passed the 5 mm depth gate, it back-projected the **entire 640×480 depth
image** into base_link just to find the pixels within 12 mm of the candidate.
Offline it is ~5 s per pass; live, with the cameras/SAM/MoveIt sharing the CPU, ~13 s.

Fix (`irb120_perception/experimental_view_check.py`, `check_view`): crop to a pixel
window around the projected candidate before back-projecting. With `q` the
candidate in the optical frame and `rho = 12 mm + |table_offset|` (the table
correction shifts base z by a constant), any point within `rho` of `q` satisfies

    |X/Z − qx/qz| ≤ rho·(qz + |qx|) / ((qz − rho)·qz)      (same for Y)

so the half-window is `ceil(f · that) + 2` px (+2 covers pixel rounding),
clipped to the image. Every pixel that can pass the 12 mm test lies inside the
window, so the connected component, the 8/12 mm PCA fits, and every returned
value are **identical** to the full-image computation (pixel order is preserved,
so even the floating-point sums match). Gates and thresholds are unchanged.

Verified: on both perceive stages of that episode, with and without the pivot
override, old vs new `evaluate()` give byte-identical contacts and per-candidate
evidence; one pass went 5.3 s → 0.13 s. Regression test:
`test_cropped_neighborhood_matches_full_image` (image borders, a depth step, and a
table-plane correction against a full-image reference).

What remains (~13–15 s) is mostly fixed cost: the DBSCAN accumulator settle
(`SETTLE_CLOUDS`), `ros2 bag record` start-up for the 1 s bag
(`BAG_SECONDS = 0` in `perception_snapshot.py` saves ~4 s if the bag isn't needed),
and the detector on/off service calls.

### The push "EGM went idle" abort was a rejected home plan

The run aborted with `MoveGroup joint goal failed: error_code=99999` →
"no usable snapshot after it". `move_group` log at that moment:
`ValidateSolution … INVALID_MOTION_PLAN`, contact between `detected_object` and
`ball` at waypoint 9/64. The "EGM feedback lost" line 60 ms later is push's own
`deactivate_egm()`. EGM had been up 43 s; `egm_cond_time` defaults to 360 s and
every motion script starts its own EGM session, so perceive time does not count
against it.

Same failure as 2026-09-25 below: home is planned from the pre-push pose, close to
the object box stretched `PUSH_DISTANCE` in +X; OMPL accepts a path its coarser
checking thinks is clear, the denser post-plan validation rejects it. RRTConnect is
random, so a replan usually passes. `plan_and_execute_joint_goal` now replans up to
`plan_retries=3` times (plan-only requests, so a failed plan never moves the
robot). This covers every home move (`push`, `arc_static`, `fwd_tip`).

**Same day, second run:** the push *approach* (a pose goal, not covered by the
above) failed the same way: `ValidateSolution` contact `detected_object`/`ball` at
waypoint 49/58, right before the pre-push pose. Root cause of both: OMPL's
`longest_valid_segment_fraction: 0.005` × the joint-space extent (~10.5 rad from the
URDF limits) = 0.053 rad between collision checks, ~30 mm of finger travel at full
reach. The pre-push ball deliberately clears the padded box by only 10 mm
(`push_geometry.APPROACH_CLEARANCE`), so OMPL routinely returned paths that clipped
the box between its checks, and the denser post-plan validation rejected them.
Fixes: `irb120_moveit_config/config/ompl_planning.yaml` now uses 0.001 (~6 mm
spacing, well under the 10 mm clearance; needs a bringup/move_group restart), and
pose goals share the replan helper (`_plan_with_retries`) as a backstop. Watch
`Calling Planner 'OMPL'` → `AddTimeOptimalParameterization` in the move_group log for
planning time (5x the collision checks per edge; the push approach already took
~4.6 s at 0.005 with its orientation path constraint, `allowed_planning_time` 10 s).

### Test suite note

`irb120_control/test/test_tsdf_mesh.py` used a module-level
`pytest.importorskip('open3d')`; without open3d that skipped the **entire** control
test session ("1 skipped", nothing else reported), which hid stale mocks in
`test_egm_startup.py` and `test_joint_goal_result.py`. It now skips per test.
`test_flake8`/`test_pep257` still fail on package-wide pre-existing style issues.

---

## 2026-10-05 — Proposed confidence-scored digital twin (pending prior-work review)

The current RViz twin is only a rendered view of the latest SAM-culled cloud and
its bounding box. The intended next step is a lightweight, CPU-friendly,
continual representation for a stationary object: transform segmented clouds
into `base_link`, merge them into a 3–5 mm voxel/surfel map, and retain per
voxel centroid, observation confidence, recency, and optionally colour.

Repeated support should strengthen stable surfaces; one-frame points should
decay away. RViz should distinguish current observation, confidence-coloured
accumulated surface, and optional editable CAD/mesh overlay. Object motion must
freeze/save then reset the map (or expose reset explicitly), rather than merge
pre- and post-manipulation geometry. A coarse mesh is a later optional output,
not a substitute for presenting confidence and unobserved regions honestly.

**Disclaimer:** this is a proposed design only. Before implementation, review
prior CPU-feasible work and existing ROS/PCL/Open3D capabilities so we do not
needlessly recreate established TSDF, surfel, probabilistic occupancy, or
multi-view registration tooling. BundleSDF-like learning pipelines are not the
default candidate because this machine has no GPU and the intended offline
turnaround is well under several minutes.

---

## 2026-09-25 — Push test: failed return home, object swirl (open, to discuss)

### Failed return home after the push

Not timing. `move_group` found a path home (0.04 s) but `ValidateSolution` rejected it:
"Invalid states at index 6 of 54 ... contact between 'detected_object' and 'ball'". The
home move starts at the pre-push pose, only ~14 mm from the object's padded collision box;
OMPL's coarser collision checking accepted a path that clips the box corner, and the finer
post-plan validation caught it. Failed safe (no motion).

**Update 2026-10-09:** recurred (home and the push approach) and aborted pipeline
episodes. Root cause was OMPL's coarse collision-check spacing (~30 mm); fixed in
`ompl_planning.yaml`, with replanning as a backstop. See the entry above. The
geometric ideas below remain the fallback if that is not enough.

Ideas:
- Retreat straight back in -X (Cartesian, e.g. an extra 50 mm on the return) before planning home.
- `arc_static`'s move home: fixed. The timed RETRACT (8 mm/s × 3 s, nominally 24 mm) only rose
  ~13 mm on tipT_d 2026-10-16 (episode 20261016_155329), leaving the ball ~4.5 mm inside the box;
  MoveIt rejected the start state (error 99999) and the episode aborted. RETRACT now lifts closed
  loop until the ball is BALL_RADIUS + APPROACH_CLEARANCE above the box top.

### Object swirls (yaws) during the push

The push line is off the object's center. The selector's planar_push only requires the
contact within ±5 mm of world y=0 (`y_band`); on the test run the ball center was at
y=+9 mm while the flashlight's bounding box was centered at y≈-4 mm, a ~13 mm offset.

Ideas:
- Push along the object's center line: pre-push y = bounding-box middle in y (≈ CoM
  projection for roughly symmetric objects), keeping x/z from the detected contact.
- If it still yaws: friction asymmetry under the base, or the ball sliding on a curved side;
  a flat pusher plate resists both better than a single ball.

---

## 2026-09-18 — EGM startup command gate

The 11:44 bringup logs showed EGM hardware activation about 0.96 s before the
startup handler's automatic hold goal. The vendor EGM manager can reject an
initial sequence-zero packet as a duplicate; the old hardware activation ignored
the read result and copied default-zero state into joint commands. Subsequent
valid feedback did not repair those command targets. Packet/target captures were
not available to prove this was the exact trigger in that run.

The local hardware driver now waits up to 100 s for successfully updated,
complete, finite, in-limit six-joint feedback on its single EGM channel. It seeds
position commands from that feedback, zeroes commanded velocity, and gates writes
until initialized. Read/write errors, channel loss, a sequence regression, or a
transition out of running EGM disable writes and return hardware ERROR. This is
latched until explicit hardware activation; it does not automatically resume an
old controller trajectory. Vendor packages are unchanged.

**Compatibility consequence:** intentional `stop_egm` / `start_egm_joint` cycles
(including motion scripts using `egm_client`) may now require restarting hardware
and controllers. Do not assume their existing drain/sleep logic authorizes a safe
reconnect. This change prioritizes refusing stale targets over seamless reconnect.

The EGM handler no longer submits a startup hold trajectory. It aborts on failed
startup service results and leaves readiness false; cleanup still attempts to
stop EGM while leaving RAPID running. Its JTC check means action-server availability,
not proof of controller activation. The hardware gate provides command protection.

Validation is offline: command-gate tests cover skipped first packets, invalid
feedback, measured-position initialization, channel loss and session resets;
mocked handler tests cover service failures and absence of a startup trajectory.
No powered-robot test or live EGM session was initiated during implementation.

---

## 2026-09-04 — Multi-trial estimation (10 trials × 4 objects)

Reviewers asked for more than a single trial per object. `estimate_params.py` now consumes
all trials per object and reports mean / median / SD / CV / 95% CI, with error bars.

### Data blockers found (these prevented ANY of the 40 trials from being processed)

**1. `ft_link` no longer exists as a TF frame.** A URDF refactor replaced the old
`link_6 -> ft_link -> finger_link -> finger_ball_center` chain with
`tool0 -> root_sensor -> ... -> sensor_body -> root_finger -> finger_ball_center`, but
`arc_static.py` and `push.py` still look up `ft_link`. Every `ft_p*` / `ft_q*` sample in
every log is NaN as a result.

Reconstructed instead from the logged EE pose. The old `ft_link` was `tool0`:
- URDF static chain gives `tool0 -> root_finger` = **0.08225 m** along tool0's +X, which is
  exactly the `FT_WRENCH_ORIGIN_OFFSET_X` already hardcoded in `estimate_params.py`.
- The chain also puts `finger_ball_center` at the *same orientation* as tool0, offset
  `0.08225 + 0.0866512 = 0.1689012 m` along +X.
- So `R_ft_B = R_ee_B` and `p_ft_B = p_ee_B - R_ee_B @ [0.1689012, 0, 0]`.

Independently confirmed against hardware: ATI's stated recording frame is the flange where
the finger base connects, i.e. `root_finger` — exactly where the pipeline lands the wrench
origin after the existing +0.08225 shift.

**2. The `obj_*` (vision) stream is empty in all batch logs** — the detector wasn't running.
It was also completely unused by the estimator (loaded, interpolated, returned, never read),
so it is now optional rather than required.

**3. Aborted runs.** `runtime_logs/box/arc_squash` holds 12 logs, 2 of which are zero-length
aborts. Trial discovery skips them with a reason. `most_recent.npz` is a byte-identical copy
of the newest trial and is excluded, or it would double-weight that trial and shrink every SD.

### The dominant error mechanism

`tau(theta) = m*g*|r_com|*sin(theta* - theta)` identifies only an **amplitude** and a
**phase**. The phase is set by the tilt reference, which is defined by the assumed pivot
(`PIVOT_DEFAULT = [0.61, 0, 0]`, one value for all four objects).

**A 1° tilt-reference error moves m and z_c by ~11%, and the fit residual is flat in it to
five significant figures** (heart 0.00300 across ±1.5°, monitor 0.06142). This is structural,
not numerical: goodness-of-fit cannot detect it, so it must be propagated explicitly rather
than inferred from the fit. Splitting m from z_c is the ill-conditioned step; `m*|r_com|`
is the well-conditioned combination (CV 0.14–2.05% vs 0.16–8.63% for m).

Consequence: `corr(theta_max, z_c_est)` = −0.97 (box), −0.95 (heart). The estimate is
largely a function of how far that trial happened to tip.

### Contact model — the ball radius cancels

Worth recording because the intuition points the wrong way. With no slip and a non-rotating
finger ball, the contact point `c` is a fixed material point of the object, so
`p_ball = c + R_ball*n(theta)` and therefore `p_ball - p_pivot = R(theta) * u0`. The **ball
centre is rigidly carried by the object** — radius and contact-patch migration cancel
exactly, and using the ball centre directly is correct.

Implementing the flat-face tangency alternative instead made everything much worse
(heart z_c error 15.6% → 45%), which is the expected signature of a wrong model.

Two things fall out of this:
- `|p_ball - p_pivot|` must be constant. Its peak-to-peak spread is a free, assumption-free
  slip metric: box 1.8, heart 1.4, flashlight 2.1, **monitor 8.4 mm** (the monitor flexes).
- The ball centre sweeps a circle centred on the pivot, so **circle-fitting the EE arc
  recovers the pivot from proprioception alone** — no vision, nothing to occlude.
  Circle residuals are 0.15–0.46 mm, confirming the model.

Gate any use of that recovery on the no-slip metric, **not** on the circle fit's own standard
error — the formal SE is a misleadingly tiny 0.04–0.2 mm even when the path isn't circular,
because a bent path still fits *some* circle tightly. `NOSLIP_TOL_MM = 4.0`.

### Object creep (the heart)

The heart physically slides away from the robot between back-to-back trials. Circle-fitting
each trial's arc sees it independently: **+0.9 mm/trial**, with a −3.5 mm reset between
trials 5 and 6 — which is the sawtooth visible in the raw estimates.

```
heart pivot_x - 0.610 (mm):  -2.57 -1.41 -0.82 +0.06 +0.81 │ -2.68 -1.79 -0.91 +0.14 +1.07
box   pivot_x - 0.610 (mm):  -0.94 -0.98 -0.99 -1.01 -0.91 -1.01 -0.96 -0.98 -0.95 -1.03
```

Box is stable to 0.10 mm total and serves as the control.

`--pivot relative` applies only each trial's **deviation from that object's own mean**
recovered pivot, leaving the mean at `PIVOT_DEFAULT`. Accuracy therefore cannot move by
construction (≤0.5 pp measured on every object), while creep-driven scatter is removed:

| heart | error | CV |
|---|---|---|
| m     | 1.18% → 1.13%   | 8.63% → **2.39%** |
| z_c   | 15.59% → 15.06% | 8.09% → **2.38%** |
| θ*    | 9.43% → 9.45%   | 7.19% → **2.16%** |

This is explicitly **not** fitted to ground truth. Box (no creep) is essentially unaffected —
a knob that only ever helps would be suspicious. Monitor is auto-rejected by the no-slip
gate; without it, the correction made the monitor *worse*.

`--pivot trajectory` is the more aggressive variant that also moves the mean. It buys extra
precision but costs accuracy when `com_x` stays fixed, so it is not the default.

### Uncertainty budget (`--uncertainty-budget`)

The heart's z_c is a *designed* value (3D printed), so the ~15% gap is measurement
uncertainty, not method error. Each input is perturbed by its 1σ and the full estimator
re-run (true nonlinear sensitivity, not a linearisation):

| source | 1σ | → σ(z_c) | % of z_c |
|---|---|---|---|
| pivot z (edge radius / base compliance) | 5 mm | 12.95 mm | **12.95%** |
| pivot x (object creep) | 1.8 mm | 8.06 mm | 8.06% |
| F/T zero drift between tares | 0.010 N | 1.81 mm | 1.81% |
| combined (quadrature) | | 15.36 mm | **15.36%** |

Observed: 15.59%. **~99% of it is pivot geometry.** Caveat: the pivot-z σ of 5 mm is an
estimate, not a measurement — it is the largest term and the one worth pinning down.

### ARC exit criterion

The arc stops when the tangent force decays toward zero — but *that is θ\* itself*, so this
threshold decides how close to θ\* the sweep gets, and the fit extrapolates the rest.

A single absolute threshold can't serve objects whose tangential force differs ~24×
(torque RMS: heart 0.075 vs monitor 1.79 N·m). At the old fixed 0.1 N the heart stopped
3.4–7.6° short of θ\* while the box got within 0.8° — and the box had 0.65% z_c error
against the heart's 15.6%. Replaced with `ARC_FX_LOW_FRACTION = 0.08` of the trial's own
peak tangent force, floored at `ARC_FX_LOW_FLOOR_N = 0.1`.

### Frame mismatch: perception publishes in `base_link`, control uses `world`

`world -> base_link` is the 21 mm bracket the robot sits on. The perception stack is
internally consistent in `base_link` (`perception.launch.py`), but `press_point_check` was
comparing its output against a `world`-frame hardcoded pose with no conversion, silently
understating the z deviation by 21 mm (it passed only because the tolerance is 50 mm).

Fixed at both consumers rather than in the perception stack (which would ripple through
masking and extrinsics): `press_point_check` now transforms into `COMPARE_FRAME = 'world'`,
and `arc_static._on_detection` transforms detections before logging. Both are no-ops when
the frames already match.

**This matters most for the object-pose stream**: it was empty, so nothing was wrong yet,
but enabling the detector without this fix would have put object poses 21 mm low in z
relative to everything else in the log.

Verified separately that the bracket does **not** bleed into the estimates — that path is
entirely in `world`, and the gap between EE-z at contact and the nominal press point is
8.5 / 8.6 / 15.8 / 20.4 mm across objects, i.e. squash penetration, not a constant 21 mm.

### Decisions taken

- **Do not re-tare per trial.** Tried and reverted — the NetFT is high quality and it risks
  disturbing the batch loop. The single pre-batch tare stands.
- **`com_x` stays an assumption.** Freeing it helps precision but shifts z_c accuracy and
  would require rewording the paper. `--free-com-x` exists as a dormant ablation, off by
  default.
- **`m*|r_com|` is not in the paper table.** Less intuitive for readers than raw m / z_c.
  Kept in the text tables and per-trial CSV as a diagnostic; excluded from the figures via
  `FIGURE_EXCLUDED_PARAMS`.
- Figures and CSV are written to `figs/`, not the workspace root.

### Open

- **μ_t cannot be regenerated** — there is no `push/` directory under `runtime_logs/`, so
  the friction column is N/A everywhere. Push trials not yet recorded.
- **Monitor GT tolerance** (CoM ±1 cm ≈ ±4.3%, θ\* ±0.2°) is comparable to its reported
  errors (5.3% z_c, 4.8% θ\*). Should be stated in the paper or that row overstates method error.
- **Trials are serially correlated** (drift check flags heart and flashlight), so SD and the
  95% CI on the uncorrected numbers assume independence they don't have.
- Box used `arc_static` (10 separate invocations, so tared and press-checked per run);
  the other three used `arc_static_batch` (once per batch). Box shows no trial-order drift,
  the batched objects do. Worth keeping in mind when comparing box against the rest.
