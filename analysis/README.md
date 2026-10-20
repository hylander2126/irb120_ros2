# analysis/ — offline estimation and paper results

Not a ROS package (COLCON_IGNORE). Plain Python; it imports `irb120_control`,
`irb120_models` and `irb120_perception` straight from this checkout (`_paths.py`), so
no build or sourcing is needed.

## Regenerate results

    python3 make_paper_results.py --data-root ~/Documents/irb120_data/official_v1/raw
    # -> ~/Documents/irb120_data/official_v1/results/{summary.md, stages.csv, stages.json,
    #    tier1_errors.png, tier2_estimates.png, provenance.json}
    python3 make_paper_results.py --data-root ... --vision     # + camera tilt tracks (slow, cached)

It never writes into the data root. `provenance.json` records the analysis commit and
flags episodes recorded from a dirty checkout or without `--official`.

Dry runs on the development archive (2026-10-20):
- `--data-root ~/Documents/irb120_data/dev_2026-10/cylinder_suite`
- `--data-root ~/Documents/irb120_data/dev_2026-10/tipT_suite`

Both complete. The fixed pivot with the record's support distance (fixed_oracle on cylinders, fixed_supplied on tipT) reproduces the earlier suite numbers.

## Methods (per tip stage)

Each method is a support model × a CoM source; the full definitions are in the
`make_paper_results.py` docstring.

| Method | Support | CoM | Where |
|---|---|---|---|
| fixed_supplied | perceived/commanded pivot (production estimator) | supplied outer-edge distance (irb120_models; table-edge measurement) | all |
| fixed_hull | same | filled convex hull of the cloud | all |
| motion_supplied | recovered from ball motion | supplied, from the perceived outer wall | press-pull |
| motion_hull | recovered from ball motion | hull centroid | press-pull |
| joint_supplied | camera prior (support-side outer envelope per height; any shape) + motion posterior, with band and gates | supplied, from the perceived outer wall | press-pull + scene cloud (recorded or rebuilt) |
| joint_hull | same | hull centroid | press-pull + scene cloud |
| fixed_oracle | pivot with the fillet-aware support distance | diagnostic only (uses support GT) | filleted bases |
| fixed_supplied_vision | as fixed_supplied, camera tilt | diagnostic | `--vision` |

`compare_methods.py` runs all of them on the tipT and cylinder suites.
`backfill_scene_clouds.py` rebuilds support scene clouds for snapshots that predate them
(from their saved depth), into `~/Documents/irb120_data/derived/`.
Its output is in `~/Documents/irb120_data/method_comparison_dev/`.

## Files

- `make_paper_results.py`: entry point. `compare_methods.py`: method comparison on development data.
- `references.py`: supplied CoM distances (inputs) and mass/height references (scoring) by object label.
- `joint_fit.py`, `scene_profile.py`, `motion_contact.py`, `rolling.py`, `orange.py`, `cloud_geometry.py`, `hull_com.py`: methods.
- `JOINT_FIT.md`: method note and development results for the joint fit.
- `tests/`: `python3 -m pytest -q tests`.

The camera-agreement check used by the pipeline and here is
`irb120_control/util/registration_check.py`.
