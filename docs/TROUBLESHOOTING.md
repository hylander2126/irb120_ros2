# Cell troubleshooting

## RViz splash hang

Historical Jazzy RViz/MoveIt configs containing a top-level `Transformation:`
display configuration could hang during startup. Remove that stale block rather
than adding another `_NEW` RViz copy. If the symptom persists, inspect RViz logs
and isolate the responsible display; not every startup hang has this cause.
The dated investigation remains in [engineering history](archive/NOTES.md).

## EGM and duplicate bringup

Run one T1 RWS connection and one T2 hardware/controller stack. Run either T3
production bringup or T3 handeye bringup; do not run both or launch another T2.
Duplicate `/controller_manager`, robot_state_publisher or EGM-handler nodes can
cause misleading spawner errors and missing-description loops.

RWS connectivity does not establish EGM UDP delivery. Follow the networking
checks in the repository README. The custom hardware interface withholds writes
until valid measured joints seed commands; EGM reconnect is not permission to
replay a contact motion. Stop/restart through the existing EGM lifecycle and
confirm measured state before another command.

## Failed planning versus transport loss

A rejected MoveIt plan is not evidence that EGM dropped. Inspect MoveIt error,
start-state bounds, current model, collision geometry and the fixed cable window
before attributing the failure to transport. Calibration needs its own limits
and ChArUco model; production retains the +/-60 degree joint-6 cable window.

## Perception failure

FFS snapshots leave a failed perceive stage and per-step logs under `ffs_runs/`.
Check camera stream/profile, interpreter/checkpoint availability, native stereo
pair timestamps, and reconstruction/contact errors. There is no DBSCAN fallback.
Do not reuse a pre-motion capture to work around a failed fresh snapshot.

Camera reprojection RMS and repeated calibration results do not certify depth
accuracy. Cam2's recent new extrinsic did not remove the FFS table-plane offset;
use calibrated board PnP versus depth checks to separate geometry and depth
errors. Keep per-camera disagreements visible; do not tune them away by changing
assumed object dimensions. See the workflow's dated diagnostics for evidence.
