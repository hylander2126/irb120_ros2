# Prior → posterior support geometry, joint cloud + motion fit, D/E vision check

2026-10-20. Offline analysis only; no robot motion, no live-code changes, no CAD/GT in
inference (scoring references are loaded only in `joint_figures.py`).

## What was done (the three Codex follow-ups)

1. **Per-camera lower geometry from the expanded scene** — `scene_profile.py`
2. **Cloud and motion in one fit, with uncertainty and rejection gates** — `joint_fit.py`
3. **D/E branch disagreement checked against recorded visual motion** — `vision_check` in `joint_fit.py`

Figures (archive): `joint_geometry.png/pdf`, `joint_mass.png/pdf`, `vision_branch_check.png/pdf`.
Data: `joint_fit.json` (inference only), `joint_fit_scored.json` (with scoring references).
Tests: `test_joint_fit.py` (5 synthetic) + existing `test_cloud_support.py` (8): 13 pass.

## Results

| Trial | Status | R prior 95% (camera, pre-contact) | R from interaction 95% | R posterior (95%) | Mass (95%) | Orange | Measured |
|---|---|---|---|---|---|---|---|
| A | accepted | 0–13.0 | 0–3.5 | 0.0 (0–3.5) | 387 (353–431) | 384 | 365 |
| B | accepted | 0–12.9 | 3.1–11.5 | 7.3 (3.1–11.2) | 385 (340–436) | 377 | 362 |
| C | accepted | 0–19.0 | 7.7–16.9 | 13.2 (8.0–15.3) | 379 (319–458) | 364 | 358 |
| D | **borderline** | 15.7–27.2 | 16.3–24.2 | 21.6 (16.8–24.4) | 422 (341–549) | 324 (unstable) | 352 |
| E | rejected | 19.4–31.5 | 20.6–28.4 | 25.4 (20.8–28.4) | 240 (172–379) | 85 (bound) | 346 |
| A repeat | accepted | 0–13.6 | 0–5.7 | 1.3 (0–5.7) | 399 (364–446) | 388 | 365 |

R in mm, mass in g. Scoring only: nominal radii 0/8/16/24/30/0 mm.
- Posterior R 95% contains the nominal radius for A, B, D and A repeat. It misses C by 0.7 mm and E by 1.6 mm.
- The mass 95% band contains the measured mass for all six trials. **The band covers only geometry and camera registration.** It excludes F/T, angle-source and model-form error.
- Point masses on A–C are +6.0/+6.2/+5.9%, and A repeat is +9.4%. The bias is nearly identical across R. A has R ≈ 0, so the bias is common-mode in the torque/arm chain, not geometry.
- **Orange (shape-free support) remains as good or better on A–C point mass.** The joint fit's gain is geometry plus honest uncertainty, not a better point mass.

The repeat-A failure of the old red method (428 g from a spurious 13.8 mm fillet) is gone. The camera now says R ∈ 0–13.6 mm (unresolved), and the interaction pins it at 0–5.7 mm.

## 1. Per-camera lower profile (`scene_profile.py`)

Each camera is processed independently.
- **Flying-pixel rejection:** second difference of the saved aligned depth over ±2 px > 8 mm. This passes smooth grazing surfaces and rejects object/background discontinuities.
- **Table plane:** each camera fits its own robust local plane in a 15–120 mm annulus. Camera 2's table sits ~8 mm above cameras 1/3, and cameras 1 and 3 differ by ~2.5 mm, so a single shared plane would be wrong.
- **Head-on sector:** points within ±40° of the camera's azimuth. That face is seen near-normal, and silhouette rims are avoided.
- **Profile model:** 1 mm height bins of radius above that camera's plane. The fillet family is fitted with outer radius and wall lean as free nuisance parameters, giving χ²(R). Bin noise = within-bin scatter ⊕ wall-bin scatter. The Birge ratio widens, never narrows.
- **Wedge-fill flare:** the sensor fills the object/table wedge, so the measured radius flares outward in the bottom 3–8 mm (+5 mm on A/B). The flare height is unknown a priori, so fits run at cutoffs of 3/5/7/9 mm. The camera prior is min over cutoffs of Δχ², i.e. the union of intervals. The fitted R rises with the cutoff while misfit falls, consistent with the flare explanation.
- **Visibility rule:** views steeper than 45° (camera 2, ~68°) cannot see a downward-facing lower transition. They are reported but not used.
- **Support view:** the low-elevation camera facing the support. That is camera 1 (−x) on all trials.

**Important for the paper:** at tilts ≤ 20° the rolling contact lies within the bottom R(1−cos θ) ≤ 0.06R of the transition, under 2 mm here. Cameras cannot observe that band (wedge fill, table noise). Pre-contact perception can only *extrapolate* the contact geometry through a shape family, and the interaction measures it.

## 2. Joint fit (`joint_fit.py`)

One fit over R, with χ²_joint(R) = χ²_camera(R) + χ²_motion(R).
- **Motion term:** the shape-free instantaneous support path level per branch (mean support x, mean tilt). The family predicts mean_P = x_face + R(1 − mean_θ).
- **Shared registration error:** x_face (support-side wall at table height, robot frame via camera extrinsics) carries an error common to both branches. It enters as a 2×2 covariance, sized from cross-camera table-height disagreement (~1.7–1.9 mm). When a scene has only one usable low-elevation view, the pooled median is used.
- **σ of branch levels:** spread over 12 smoother settings (window 0.3/0.6/1.2 s × degree 2–5) ⊕ block-bootstrap SE.
- **Slope dP/dθ (= −R under the family):** this is a second derivative of measured position. It is used only as the diagnostic `support_path_not_circular_rolling` (3σ), never as evidence.
- **Mass/CoM:** the existing rolling-support torque model at posterior percentiles of R (2.5/16/50/84/97.5). The support p₀ comes from GLS of the branch levels plus the registration prior. The arm is cloud axis − p₀, as in replay.py.
- **Mass band:** quadrature of the R spread and arm ± 1.96 σ_reg. Registration dominates on short arms (D: ~15 mm arm → ±12% per 1.9 mm).

**Gates (fixed before scoring):**
- *Rejected* on any of: cloud–motion conflict χ² > 6.63 (p = .01); branch mass split > 20%; an active parameter bound at the median; torque RMS > 5 mN m on both branches.
- *Borderline* on any soft flag: branch mass split > 10%, slope diagnostic, branch level disagreement, vision asymmetry.

## 3. D/E against recorded visual motion

The camera-2 photometric tilt tracks from the earlier audit (`cylinder_fillet_audit/*_vision.npz`) give, per branch, the ratio k = vision angle / motion angle:

| | A | B | C | D | E |
|---|---|---|---|---|---|
| k tip / return | .976/.962 | .811/.809 | .775/.786 | **.870/.853** | .729/.727 |
| residual rms (deg) | .18/.80 | .23/.26 | .58/.51 | .12/.11 | .21/.25 |

**D:** the object moved the same way on both branches, with equal k and ~0.1° residuals. The old 28.6/103.2 mm branch height split therefore comes from the derivative-based support reconstruction, not the physics. Under the joint fit, D's branch mass split drops to 8.4%, and the cloud and motion agree (conflict χ² 0.6). The tip branch still trips the slope diagnostic, hence **borderline**: geometry consistent, mass band wide (341–549 g) because its arm is short.

**E:** also visually symmetric, but it reaches only ~4.4° (vision) / 5.5° (motion) of tilt. Branch mass split is 30%, and the height bound is active at low-R percentiles. **Rejected** as a low-excitation failure case.

Vision caveats:
- Vision angle is 3–27% below motion angle on every object. This scale disagreement is unresolved; neither source is validated GT.
- Vision "slide" absorbs warp-pivot misplacement, so it is not used as a slip gate.
- No vision track exists for A repeat (not generated here).

## What changed during development (disclosure)

The design was iterated on these six trials, and the scoring geometry was visible to me in `figure_selection.json` throughout. Changes made after seeing outputs:
1. **Edge filter:** 5×5 range → second difference. The range filter removed camera 2's grazing wall.
2. **Truncation rule:** single data-adaptive truncation (two variants, both unstable) → fixed cutoffs with union.
3. **Wall lean:** fixed detrend → free nuisance parameter in the fit.
4. **Prior union bug:** min of raw χ² → min of Δχ². Raw χ² is not comparable across cutoffs.
5. **Motion term:** free-slope intercept → model-consistent path level. The intercept imported the slope, which the doc already called unreliable.
6. **Arm bug:** a − R → cloud axis − p₀, matching replay.py.
7. **Registration in the mass band:** added.

Items 4 and 6 were bugs against the stated design. Items 1, 2, 3, 5 and 7 are design changes. **Treat these six trials as development data.** The next trials (frozen code) are the real test.

## Remaining error, ranked by uncertainty (pipeline steps, not GT)

1. **Camera ↔ robot registration of the cloud axis (σ ≈ 1.9 mm):** dominates the mass band, especially on fillet objects (short arm).
2. **Common-mode +6% mass bias on A–C:** independent of R, so it sits in torque at zero tilt (F/T bias/scale, finger contact moment) or a shared arm offset. Not resolved here.
3. **Vision vs motion angle scale (k = 0.73–0.98):** affects the CoM height more than the mass. Unresolved.
4. **Wall-lean extrapolation to table height:** up to 3 mm on E; not propagated.
5. **Circular-transition family:** the slope diagnostic flags D's tip branch as non-circular rolling.

## Reproduce

The development run (figures, joint_fit.json) lives in the archive:
`~/Documents/irb120_data/dev_2026-10/cylinder_cloud_support/`. The code now lives here
(`joint_fit.py`, `scene_profile.py`, ...) and runs on any data root through
`make_paper_results.py`. Tests: `python3 -m pytest -q tests` in this folder.
