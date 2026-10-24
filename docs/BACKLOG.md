# Maintained development backlog

This is the current task index, not a second operating manual. Completed work
belongs in Git history and dated evidence; commands belong in package guides.

| Remaining work | Evidence / boundary |
|---|---|
| Diagnose cam2 stereo-depth/table offset independently of handeye extrinsics | [FFS diagnostic history](../irb120_perception/docs/FFS_DIAGNOSTICS.md); new extrinsic did not fix drill_test3. Compare board PnP/depth at multiple distances. |
| Validate contact/pivot accuracy on held-out scenes | [Contact policy](../irb120_perception/CONTACT_SELECTION.md); observed candidates and repeatability are not independent pivot GT. |
| Characterize FFS episode latency and camera/arm-clear acquisition on the real rig | Automatic fresh capture replaces legacy seeded DBSCAN updates; software tests do not establish hardware timing or manipulation success. |
| Port analytical shove solver and fixed-time trajectory validation | [Shove check](../irb120_control/SHOVE_CHECK.md); geometric checks are not timed execution approval. |
| Reconcile forward-arc controller and estimator wrench-reference origins | Controller still uses historical +82.25 mm tool0-X; estimator uses ball-to-mount geometry. Verify mechanical conventions before changing controller torque thresholds. |
| Resolve empirical angle/contact biases before applying corrections | Workspace `notes/PENDING_CORRECTIONS.md` and `notes/validating_GT_experiments_analysis.md`; preserve unvalidated corrections as pending. |
| Complete object configuration provenance and missing support/uncertainty measurements | [Models](../irb120_models/README.md); unknowns must remain unknown. |
| Review mesh storage/generated-asset policy | Cylinder variants are active experiment inputs. Generated STL files are referenced by generated URDFs; do not delete one side alone. |
| Package workspace FFS scripts/checkpoints reproducibly | [Deployment contract](../irb120_perception/docs/STEREO_OFFLINE.md); src alone is not yet a complete FFS install. |

Recent consolidation completed: retired live detectors/legacy TSDF entry points,
FFS episode acquisition, current contact policy, archived rollout documents,
headless handeye guide, calibration-run output isolation, and stale packaging/YAML
cleanup. Active camera transforms, F/T calibration and production/calibration limit
policies are preserved.

Workspace assistant daily notes remain chronological history. Prior workspace
planning/audit documents are indexed in `notes/README.md`; their hypotheses are
not current controller settings. The research paper scope and validation evidence
remain separate from runtime instructions.
