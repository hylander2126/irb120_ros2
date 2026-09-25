# Engineering Notes

Running log of non-obvious findings, decisions, and open questions. Newest entry first.
Things that are obvious from the code do not belong here — this is for the things that
cost time to discover and would cost time again.

---

## 2026-09-25 — Push test: failed return home, object swirl (open, to discuss)

### Failed return home after the push

Not timing. `move_group` found a path home (0.04 s) but `ValidateSolution` rejected it:
"Invalid states at index 6 of 54 ... contact between 'detected_object' and 'ball'". The
home move starts at the pre-push pose, only ~14 mm from the object's padded collision box;
OMPL's coarser collision checking accepted a path that clips the box corner, and the finer
post-plan validation caught it. Failed safe (no motion).

Ideas:
- Retreat straight back in -X (Cartesian, e.g. an extra 50 mm on the return) before planning home.
- `arc_static`'s move home likely has the same problem: RETRACT (8 mm/s × 3 s = 24 mm) leaves
  the ball ~14 mm above the box. A longer RETRACT (e.g. 6 s ≈ 48 mm) would clear it, but changes
  tip timing.

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
