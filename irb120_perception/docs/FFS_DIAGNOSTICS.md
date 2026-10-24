# FFS diagnostic history

These dated investigations preserve parameter sweeps and evidence. Earlier
conclusions are hypotheses unless supported by later checks. In particular,
cam2's new 1.2 px extrinsic did **not** fix its table-depth bias in drill_test3;
the earlier recommendation that handeye recalibration would resolve that bias
was incorrect. Check depth against board PnP before another extrinsic change.
Current commands are in [the workflow guide](STEREO_OFFLINE.md).

## Cam2/top-mesh gap guard (fixed: normal-aware, not just a distance threshold)

`sam_shadow.protected_meshes()` keeps cam1+cam3 and cam2 as separate TSDF volumes. A cam2-only
triangle used to be discarded whenever it was within `gap_guard_voxels*voxel` (default 6 voxels =
18mm) of *any* cam1/cam3 geometry — deliberate, tested (`test_cam2_cannot_shift_preferred_tsdf`),
but blind to *which* face it was near. Cam2 is the only camera that sees object tops, and
cam1/cam3's incomplete upper-edge geometry sits close enough (in plain Euclidean distance) to the
top that the old rule discarded cam2's top-face triangles too, even though they're a *different*
face (top cap vs. side wall), not a duplicate of anything cam1/cam3 actually captured.

Fixed by adding a surface-normal check alongside the existing distance check: a cam2 triangle is
now only pruned if it's close to cam1/cam3 geometry **and** the normals agree (same face —
reusing `priority_merge`'s own 0.8 cosine-similarity threshold for "same surface", so it's not a
new arbitrary number). A nearby but differently-oriented cam2 surface — the top cap next to the
side wall — survives even inside the guard distance, since it's new information, not a duplicate.
`gap_guard_voxels` (default 6/18mm, also `reconstruct_stereo_offline.py --gap-guard-voxels` /
the `stereo_offline_pipeline.launch.py gap_guard_voxels:=` launch arg) still controls the distance
component; 0 disables pruning entirely as before.

Tested on tipT's capture (reusing one warm model across repeated `reconstruct()` calls, no need to
rerun FFS/SAM each time):

| Variant | Gap triangles | Top-20mm-band triangles |
|---|---:|---:|
| 18mm, distance-only (old) | 3,935 | 2,528 |
| 6mm, distance-only | 5,939 | 3,038 (+20%) |
| 0mm, disabled entirely | 6,684 | 3,185 (+26%) |
| **18mm, normal-aware (shipped)** | **6,489** | **3,165 (+25%)** |

Normal-aware at the *original* 18mm distance recovers 99.4% of what fully disabling the guard
would give (3,165 of 3,185), while keeping the full 18mm protection for genuinely redundant
same-face surfaces — strictly better than picking a single looser distance number, which was
the original ask. Checked for the seam/double-surface risk this could reintroduce (the reason the
guard existed) with a render at both a side and a near-top-down angle: clean, no visible
duplicate-surface artifacts. This is now the default behavior; no parameter change needed.

## Forward-tip pre-contact outlier robustness (fixed)

`forward_tip_motion.tip_targets()`'s padded-box clearance calculation used true `cloud.min()`/
`cloud.max()`, not robust to outliers. On tipT, 2 of 6,346 object-cloud points (0.03%) sat
~350-400mm from the main cluster (same elevated Z as the object's top, possibly related to the
gap-guard finding above) and single-handedly inflated the computed pre-tip standoff from the
normal 20mm to ~377mm. Fixed with `_robust_bounds()`: a point only counts toward the box if it
has at least 2 others within 10mm (real surface points are never isolated like noise is —
confirmed clean separation on real data: outliers measured 1 neighbor, the 1st percentile of real
points measured 4+, at every radius checked 10-30mm). Falls back to the plain min/max on small
clouds (also keeps existing tests passing unchanged). Verified on tipT: 377mm -> 32mm, a sensible
standoff. This is shared with the live robot's actual forward-tip approach (`fwd_tip.py`), not
just this visualization — regression test added
(`test_isolated_outliers_do_not_inflate_the_pre_tip_clearance` in `test_forward_tip_motion.py`).

## Cam2 seam: a real strip band missing even with the gap guard fully disabled (open, not yet fixed)

On a cylindrical flashlight capture, the combined mesh still has a visible vertical strip of
missing geometry down one side, even at `--gap-guard-voxels 0` (the normal-aware guard above
disabled entirely, so it cannot be the cause). Root-caused with an instrumented replay of
`priority_merge()`/`protected_meshes()` on a real capture
(`runtime_logs/flashlight_FFS_full_pipeline_test4/20261023_215543/`):

- Binning native (pre-merge) object points by azimuth around the object's vertical axis at
  mid-height shows cam1 and cam3 both have a genuine ~30-40° blind wedge (steep/grazing angle for
  both — neither camera observes that side of the cylinder directly). Cam2's own points *do*
  populate that exact wedge (confirmed non-zero, `priority_merge` keeps nearly all of them —
  `conflicting_cam2` was 0 on this capture, so point-level suppression is not the cause either).
- But `protected_meshes()` builds cam2's contribution as its *own*, fully separate TSDF volume
  (`ScalableTSDFVolume`, cam2-only), extracted independently via marching cubes, then
  **concatenates** `preferred + gaps` with no welding — this is a pre-existing, documented
  property of the function ("No welding, blending or invented closure otherwise. Seams/holes are
  still expected"), not new behavior. Two consequences, both visible in the renders below even
  with the guard off:
  1. Cam2's own single-view TSDF patch in that wedge is itself sparse/jagged (grazing-angle FFS
     noise + thin single-camera sampling), so it doesn't fully close on its own.
  2. Even where it does have triangles, they're a geometrically separate surface from
     cam1+cam3's — nothing snaps their boundary edges together, so the join is a ragged crack
     wherever the two independently-extracted meshes don't happen to coincide.
- This is why tuning `gap_guard_voxels` down to 0 "didn't help much": that parameter only
  controls whether cam2 triangles get *deleted* near preferred geometry. It was never in the
  business of welding the two surfaces together, and the missing strip here isn't caused by
  deletion — the cam2 data survives, it just doesn't mesh cleanly against cam1/cam3's patch.

Gray = cam1+cam3 ("preferred"), orange = cam2-only ("gaps", guard disabled, nothing pruned). The
jagged gray/orange boundary and the white gaps between them are real absence of triangles, not a
rendering artifact (confirmed by rendering the textured `combined.ply` from the same angle, second
image, and by rendering a second, independent capture from a different angle, third image — same
pattern both times):

![cam1+cam3 vs cam2-only ownership, guard disabled](stereo_offline_assets/cam2_seam_20261023/colored_ownership_wedge.png)

![Same view, actual textured combined.ply](stereo_offline_assets/cam2_seam_20261023/combined_textured_wedge.png)

![Second, independent capture — same seam pattern from a different angle](stereo_offline_assets/cam2_seam_20261023/combined_textured_second_capture.png)

**Fixed: one shared TSDF volume instead of two.** User chose this over welding two separate
volumes after the fact. `protected_meshes()` now integrates cam1+cam3 into a single volume first,
then integrates cam2 into the *same* volume — but with any cam2 pixel masked to zero (treated as
no depth there, same as outside its SAM mask) if it's within `gap_guard_voxels*voxel` of
already-integrated geometry **and** the surface normals agree (same 0.8-cosine same-face test as
before, now applied to cam2's raw per-pixel depth/normals before integration instead of to the
`gaps` mesh's triangles after extraction). Marching cubes then extracts one continuous surface
from that single volume, so a cam2-only patch and a cam1/cam3 patch that happen to border each
other come out as the same connected mesh, not two independently-triangulated pieces that only
coincidentally meet.

`preferred.ply` and `gaps.ply` are still written from their own fully isolated volumes, exactly as
before (unaffected by anything above) — for the SAM-shadow-ablation review tooling and as
`test_cam2_cannot_shift_preferred_tsdf`'s reference. `combined.ply` is the new single
jointly-integrated mesh, no longer `preferred + gaps` concatenated. Two new regression tests cover
the design directly: `test_combined_cannot_be_shifted_by_disagreeing_nearby_cam2` (the safety
invariant: a disagreeing nearby cam2 sample still can't move the combined mesh's geometry where
cam1/cam3 already established it) and `test_combined_mesh_is_one_continuous_patch_where_only_cam2_has_coverage`
(the actual bug: cam2 extending real coverage now stays one connected surface, not a seam). One
subtlety found while writing the first test: a cam2 pixel with an *unreliable* normal (zero
vector, from `surface_normals()`'s own depth-discontinuity handling, e.g. at an image border) used
to never count as "same face" and so was never masked, which is fine for `priority_merge`'s
corroboration — unmasked points there just stay separate observations — but wrong here, where
"not masked" means directly integrated alongside trusted geometry in the same volume. Fixed by
treating an unreliable normal as "agrees" for this guard specifically: no normal evidence is
grounds to *exempt* a close pixel, only to confirm it's a genuinely different face. `irb120_perception`
rebuilt; full relevant suite (`test_sam_shadow.py` + the rest of `irb120_perception/test` +
`irb120_control`'s forward-tip/snapshot tests): 166/166 pass.

Re-ran the flashlight capture end-to-end through the real CLI (not just the diagnostic replay
above) to confirm on the actual pipeline, not just the isolated function. At `--gap-guard-voxels 0`
the strip is now solid — no visible gap from the same wedge angle that showed daylight through the
old concatenated mesh:

![Fixed: single volume, guard disabled -- solid where the old design still showed daylight](stereo_offline_assets/cam2_seam_20261023/fixed_single_volume_guard0.png)

The *default* `gap_guard_voxels=6` (18mm) still leaves the strip open on this capture, though —
masking now happens pre-integration against raw pixels rather than post-hoc against extracted
triangles, which turns out to be meaningfully stricter per mm than before, so the old default
distance is too conservative for this real gap's physical width:

![Default guard (18mm) on the new design: still open -- the default needs retuning, not just the architecture](stereo_offline_assets/cam2_seam_20261023/still_open_single_volume_default_guard6_18mm.png)

`--gap-guard-voxels 1` (3mm, one voxel) closes it almost completely (one small pinhole near the
base) while still keeping a real protection margin against exact-duplicate integration:

![guard=1 (3mm): nearly fully closed, still has a real protection margin](stereo_offline_assets/cam2_seam_20261023/fixed_single_volume_guard1_3mm.png)

**Not yet decided:** whether to ship a new default `gap_guard_voxels` (1 looks like a good
candidate from this one capture) or leave 6 as the default and rely on the caller passing a lower
value explicitly. Left the default at 6 pending that call — `reconstruct_stereo_offline.py
--gap-guard-voxels` / `stereo_offline_pipeline.launch.py gap_guard_voxels:=` already expose it.

## Per-camera isolated-point noise filter (fixed — cam1/cam3 included, see update below)

On a real drill capture (`runtime_logs/drill_test1/20261024_132231/`) the user flagged three
linked symptoms: a visibly noisy/frayed top band where cam1/cam3 don't see (cam2-only territory,
same region the gap guard above deliberately never prunes since it's genuinely new information),
the press contact landing further +X than expected, and the forward-tip pre-tip standoff looking
"massive". Root cause for the top band: nothing in `protected_meshes()` or `priority_merge()`
ever rejected a cam2 pixel for being noisy in isolation — only for being redundant with cam1/cam3
(gap guard) or disagreeing with another camera (corroboration/conflict checks in `priority_merge`).
Neither check fires on a pixel in territory only cam2 observes, which is exactly the top band, so
cam2's own depth noise there reached both the mesh and the object cloud unfiltered.

Fixed with the same "a point only counts if it isn't alone" reasoning already used for this sensor
in `forward_tip_motion._robust_bounds` (see above), applied at the source in two places that need
it independently, since they're separate pipelines starting from the same raw per-camera depth:
- `protected_meshes()`: drops a pixel from TSDF integration (every camera, not just cam2) if it
  has fewer than `noise_min_neighbors` (default 4) other surviving SAME-camera pixels within
  `noise_radius_voxels*voxel` (default 3 voxels = 9mm). Mesh-only; affects `combined.ply` (and
  now `preferred.ply`/`gaps.ply`, since the filter is strictly per-camera and never compares
  across cameras, so it can't interact with the cross-camera gap guard above).
- `priority_merge()`: the same check (`noise_radius_m`/`noise_min_neighbors`, default 9mm/4),
  applied to cam1/cam3 up front (before they're ever treated as "preferred") and to cam2 after
  corroboration/conflict filtering, before the result is concatenated into the returned
  `xyz`/`object_cloud.npz` — this is the cloud `select_contact_points()` and
  `forward_tip_motion.tip_targets()` actually score over, not the mesh.
Both only ever drop a pixel; neither adds, moves, or interpolates one, so neither can invent
closure the sensor didn't report. Both default to 0 = disabled. Exposed end to end:
`reconstruct_stereo_offline.py --noise-radius-voxels`/`--noise-min-neighbors` and the matching
`stereo_offline_pipeline.launch.py` launch args.

Shipped in two steps, both verified by rerunning `reconstruct()` against the saved drill capture
(no new camera/robot time needed) — this is deliberately a per-camera filter, so it needed
verifying on both sides of the "preferred" boundary before it could be called done:

**Step 1 (cam2 only).** 27 cam2 pixels dropped from the mesh integration (36 fewer combined
triangles, 17,374 → 17,338) and 53 cam2 points dropped from the object cloud (`gap_cam2` 8,907 →
8,854). But this did **not** change the drill's selected forward-tip or press contact point at
all (bit-identical before/after). Traced why: the points actually driving this capture's 177mm
forward-tip pre-tip clearance (`tip_targets()`'s padded-box bound) were all `camera_id == 3`
(cam3), sitting far from the object body (X 0.16-0.48m vs. the drill's actual 0.55-0.77m
footprint) — not cam2 at all. `priority_merge()`'s original docstring and its
`test_exact_preferred_samples_and_cam2_corroboration` test made "cam1/cam3 pass through
completely unfiltered and unchallenged" the architecture's deliberate, tested contract at the
time — only cam2 was ever checked against anything.

**Step 2 (extended to cam1/cam3), after explicit user go-ahead** to touch that "exact preferred
samples" guarantee: `priority_merge`/`protected_meshes` now filter every camera independently
(params renamed `cam2_noise_*` → `noise_*` to reflect this — no longer cam2-specific). On the
same drill capture this found real noise in the previously-untouched "preferred" cameras too
(`{'cam1': 48, 'cam3': 101, 'cam2': 53}` points/pixels filtered from the object cloud this time;
mesh-side counts differ slightly since `protected_meshes` runs its own independent copy of the
filter). **This is what actually fixed the forward-tip pre-arrow:** recomputing
`tip_targets()` with the cleaned object cloud took the pre-tip clearance from 177mm to 42.5mm —
a sensible standoff bound derived from the drill's real geometry, not a 120mm-distant outlier.
The press contact still did not move (bit-identical) even with cam1/cam3 filtered too, so the
press drift the user also reported on this capture is apparently not an outlier/noise effect —
it reflects the real `_press_band` double-scoring (height-above-pivot vs. edge-proximity, see
`contact_point_selector.py`) on genuine top-surface geometry, not something this filter addresses;
not investigated further here.

All 17 `test_sam_shadow.py` tests, 2 new `priority_merge` noise-filter regression tests, 2 new
`protected_meshes` noise-filter regression tests, and all 34 `test_forward_tip_motion.py` tests
pass (55/55 total; synthetic test clouds are single dense flat planes — never isolated, so the
filter is a correct no-op on them except where a test deliberately injects one outlier pixel).

## cam2's pivot/marker overlays are visibly biased (+X / "left") -- root cause: calibration, not code

User observation (2026-10-24): pivots/markers reproject correctly onto cam1's and cam3's images
but look consistently offset in cam2's -- "been this way for a while." `contact_graphics.py`'s
`project()` reprojects a 3-D point into a camera's 2-D image using that camera's OWN
`base_to_optical` R/t from its `camera_info.json` (plus `cv2.projectPoints` with that camera's K/D
-- D is `[0,0,0,0,0]` for all three cameras here, so this isn't a distortion-modeling gap specific
to cam2). Since the SAME 3-D point reprojects correctly via cam1/cam3's transforms but not cam2's,
the 3-D point itself (and cam1/cam3's calibration) isn't the problem -- cam2's own calibrated pose
is.

Confirmed two ways:
1. **The calibration's own documented residual.** The static transform actually wired into
   `bringup_cams.launch.py` (its `cam2` entry) is literally named by it:
   `cam_tf_realsense2_2.1px.launch.py` -- "18 samples, RMS reprojection error 2.12 px (~1.60mm at
   the board)". Cam1's is `cam_tf_realsense_1.7px.launch.py` (21 samples, 1.71px, ~0.61mm) and
   cam3's is `cam_tf_realsense3_1.6px.launch.py` (19 samples, 1.58px, ~1.15mm). Cam2 has the
   fewest calibration samples and the worst residual of the three.
2. **A direct, repeatable geometric check on real data**, not just trusting a stale docstring
   number: every FFS capture independently fits a table plane per camera (same real, flat table
   each time). Across all 9 FFS captures in `runtime_logs/` with all three table fits available,
   cam2 measures the table **11-13mm lower** (more negative base-link Z) than the cam1/cam3
   average, every single time, while cam1 and cam3 agree with each other to within ~2mm:

   | Capture | cam1 Z | cam2 Z | cam3 Z | cam2 - avg(cam1,cam3) |
   |---|---:|---:|---:|---:|
   | tipT_FFS_test | -23.77mm | -38.19mm | -25.83mm | -13.39mm |
   | flashlight_test | -23.89mm | -38.39mm | -26.16mm | -13.36mm |
   | drill_test1 | -23.86mm | -37.80mm | -25.65mm | -13.04mm |
   | flashlight_test3 (215250) | -23.88mm | -36.10mm | -25.77mm | -11.28mm |
   | flashlight_test3 (215734) | -23.89mm | -36.25mm | -25.79mm | -11.41mm |
   | flashlight_test4 | -23.86mm | -36.05mm | -25.77mm | -11.23mm |
   | flashlight_test2 | -23.94mm | -36.40mm | -26.15mm | -11.35mm |
   | flashlight_test5 | -23.79mm | -35.80mm | -25.90mm | -10.95mm |
   | drill_test2 | -23.91mm | -36.04mm | -25.83mm | -11.17mm |

   That's a real, physical, ~12mm-scale disagreement about where the same flat table actually is,
   isolated to cam2 and reproducible across 9 independent captures over 2 days -- consistent with
   a genuine extrinsic (hand-eye) calibration bias, not sensor noise (noise wouldn't reproduce this
   consistently) and not a code bug in this pipeline (table-plane fitting and reprojection are two
   independent code paths that both single out cam2 the same way).

**Update 2026-10-24 (drill_test3): the extrinsic explanation below is wrong.** cam2 was recalibrated with
its own 12-pose set (1.17 px, `cam_tf_realsense2_1.2px.launch.py`); the new pose is within 2.7 mm / 0.29 deg of
the old one, and drill_test3 still shows cam2 about 10 mm low at the drill (-13.2 mm at the origin), same as before.
The bias is in cam2's stereo depth, not its hand-eye pose. Next step: `handeye check-depth-pnp --camera
realsense2`, then RealSense On-Chip + Tare calibration if it confirms. Table in
`runtime_logs/drill_test3/20261024_165442/recon/README.md`. Original (superseded) reasoning follows.

**Not fixed here** -- this is a calibration artifact, not a bug in `reconstruct_stereo_offline.py`,
`sam_shadow.py`, or `contact_graphics.py`. The actual fix is a redone hand-eye calibration for
cam2 (`irb120_handeye calibrate` / `run_handeye_calibration.py`) with more samples spread across a
wider range of board poses, aiming to match cam1/cam3's ~1.6-1.7px residual; 18 samples is the
fewest of the three and is plausibly the whole story. Until re-calibrated, treat cam2's own
reprojected overlays (and any metric read directly off cam2, like its table-plane fit) as
offset by roughly this much; cam1/cam3-anchored pivots (which this pipeline already prefers --
see "Pivot and localization diagnostics" above) are not affected, since cam2 never contributes to
either pivot's RGB-boundary fit.

## Press candidate funnel diagnostic (added 2026-10-24)

User asked for more visibility into *why* a point was or wasn't selected for press, not just the
final point and the `candidate_counts` tally already in `contacts.json`. `select_contact_points()`
(`contact_point_selector.py`) gained an opt-in `diagnostics` dict parameter (`None` by default --
zero cost/behavior change for every existing caller): when given, it's filled with a full-length
boolean survivor mask per funnel stage (clearance/direction/surface/positive_moment/inset/
camera-agreement/best-score-band/edge-centering/...) over the deduplicated point cloud, plus the
selected point's index. `experimental_view_check.evaluate()` forwards this through to its own
internal `select_contact_points()` call -- the one that actually applies the camera-agreement/
view-check gate via `experimental_press_filter` -- so the captured funnel is the real, complete
one the shipped pipeline uses, not a simplified stand-in.

`select_stereo_contacts.py` now classifies every candidate point by the LAST stage it survived
(`classify_press_candidates`) and renders it as a colored 3-D scatter -- `contacts/
press_diagnostics.png` (static, 2 viewpoints, embedded in `contacts/README.md`) and `contacts/
press_diagnostics.html` (interactive Plotly, hover for stage + mm coordinates) -- plus a funnel
table of point counts per final stage. On the drill capture: 4,591 points never cleared the basic
fingertip-clearance/normal gate (e.g. inside the object or table-adjacent), 2,391 cleared that but
faced the wrong way for a pull (`direction`), 2,286 had a non-vertical-enough normal
(`positive_moment`, i.e. tilted sides/edges), 783 were on the feasible top region but outside the
convex-hull inset, only 13 of those passed geometric scoring but failed the multi-camera
view-check, and 43 passed everything but lost the final edge-centering tiebreak to the 1 selected
point. 3 new unit tests (`contact_point_selector`) check the stage masks are internally consistent
(monotonically shrinking, counts matching `candidate_counts`, selected index resolves to the
reported point) and 3 more (`scripts/test_select_stereo_contacts.py`) check the classifier itself.

## Cam2-fusion hyperparameter ablation (2026-10-24): confirms the defaults are a real sweet spot

User asked whether `gap_guard_voxels`/`noise_radius_voxels`/`noise_min_neighbors` could make
things worse at other settings, not just better at the shipped defaults. Swept each parameter
independently on the drill capture (`scripts/_ablation_cam2_fusion.py`, one warm model load, ~6s/
run) and, for the interesting settings, ran the full pipeline through `tip_targets()` to check the
actual forward-tip pre-tip clearance and press point, not just mesh triangle counts:

| Parameter | Value | Pre-tip clearance | Press point | Mesh effect |
|---|---:|---:|---|---|
| `noise_min_neighbors` | 0 (disabled) | **177.3mm** (known-bad) | unchanged | 17,374 combined tri |
| `noise_min_neighbors` | 4 (default) | 42.5mm | unchanged | 17,238 combined tri |
| `noise_min_neighbors` | 16 (aggressive) | 42.3mm (no further gain) | unchanged | 16,941 combined tri (-297 vs default) |
| `noise_radius_voxels` | 1 (3mm, too tight) | 42.3mm | unchanged | 16,840 combined tri (-398 vs default) |
| `noise_radius_voxels` | 3 (9mm, default) | 42.5mm | unchanged | 17,238 combined tri |
| `noise_radius_voxels` | 9 (27mm, too loose) | **100.3mm** (2.4x worse) | unchanged | 17,363 combined tri (barely fewer than disabled) |
| `gap_guard_voxels` | 0/1 (disabled/weak) | 42.5mm (unaffected) | unchanged | **21,372 / 21,106 combined tri** (+24%/+22%: duplicate/seam geometry readmitted, see the gap-guard section above) |
| `gap_guard_voxels` | 6 (default) / 12 / 24 | 42.5mm (unaffected) | unchanged | 17,238 / 17,196 / 17,196 combined tri |

Findings:
- **The press contact never moved, under any setting tested.** Confirms (again, more broadly this
  time) that press's `+X` drift the user originally asked about isn't a cam2-fusion artifact at
  all -- it's downstream of the real `_press_band` double-scoring on genuine top-surface geometry.
- **`noise_radius_voxels` is the parameter that can genuinely make the forward-tip fix worse, not
  just less effective.** Too loose (9 voxels/27mm) lets the exact kind of far outlier this filter
  exists to catch blend back in with real, dense surface (near-zero points filtered: 2 cam2 pixels,
  11 cam1, 33 cam3 -- vs. 53/48/101 at the default), undoing most of the 177mm->42.5mm fix (back up
  to 100.3mm). Too tight (1 voxel/3mm, barely bigger than the TSDF's own voxel size) over-filters
  real sparse surface for essentially no benefit over the default (triangle count drops the most of
  any setting tested, clearance doesn't improve). **3 voxels (9mm) sits in the gap between these
  two failure modes** -- this isn't just "the number we tried first," the sweep confirms it's
  functionally load-bearing.
- **`noise_min_neighbors` has a wide safe range above the default** -- 4 vs. 16 barely changes the
  outcome (42.5mm vs 42.3mm) while costing some real mesh detail (-297 triangles) for no measurable
  gain; 4 is already past the point of diminishing returns, not under it.
- **`gap_guard_voxels` and the noise filter are fully decoupled** -- gap_guard only ever affects
  `combined.ply`'s triangle count (mesh visual quality / seam duplication, already covered in depth
  above), never the object cloud `tip_targets()`/`select_contact_points()` actually score over.
  Nothing tested here changes that prior finding; 6 remains the shipped default pending the
  still-open "ship 1 instead?" question from that section.
- **A clean-looking mesh does not imply a clean clearance computation.** `noise_radius_voxels=9`'s
  mesh (`meshes.png` for that run) is visually indistinguishable from the default's -- the
  few surviving outlier points that wreck `tip_targets()`'s bounding box are a vanishingly small
  fraction of ~76k points and don't show up at mesh-render scale. The forward-tip pre-tip distance
  specifically (not a mesh screenshot) is the right thing to check when tuning this parameter.
