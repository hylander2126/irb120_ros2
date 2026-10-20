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

Both complete. reference_d reproduces the earlier suite numbers.

## Methods (per tip stage)

| Name | Needs | Gives |
|---|---|---|
| reference_d | measured CoM distance (tier 1) | production estimator m, z_c; errors |
| hull_d | pre-action cloud | first moments h_x = m d, h_z = m z (identifiable); m, z_c under a filled-hull uniform-density assumption |
| orange | press-pull, cloud | shape-free instantaneous support m, z_c (CoM on the hull centroid) |
| joint | press-pull, support_scene_cloud | support geometry prior→posterior, m with a 95% band, accepted/borderline/rejected (JOINT_FIT.md) |

On the development data:
- **The hull assumption fails** on concave shapes (tipT E, +103% mass) and on short arms (cylinder D, −38%).
- **First moments are the robust GT-free output.** Which mass method is the headline is a decision to freeze before scoring official data.

## Files

- `make_paper_results.py`: entry point.
- `references.py`: scoring references by object label (tier 1) or None (tier 2).
- `joint_fit.py`, `scene_profile.py`, `motion_contact.py`, `rolling.py`, `orange.py`, `cloud_geometry.py`, `hull_com.py`: methods.
- `JOINT_FIT.md`: method note and development results for the joint fit.
- `tests/`: `python3 -m pytest -q tests`.

The camera-agreement check used by the pipeline and here is
`irb120_control/util/registration_check.py`.
