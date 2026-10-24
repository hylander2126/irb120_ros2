# irb120_control

Cell bringup, motion primitives, episode recording and estimation for the real
ABB IRB120. Follow [repository bringup](../README.md) before issuing motion.

## Supported perception and episodes

Perceive stages now use fresh offline FFS stereo capture, reconstruction and
observed-surface contact selection. No live detector or detector activation
service runs in the stack. The arm must be clear of the stationary object during
capture; after an interaction, the primitive returns before taking fresh frames.
The stack defaults to the `offline_stereo` camera profile.

```bash
# Perception only; no robot motion. Creates an episode for this object label.
ros2 run irb120_control episode snapshot flashlight
ros2 run irb120_control episode show
ros2 run irb120_control episode preview --episode /path/to/episode

# Full interaction workflow, with the existing operator gates.
ros2 run irb120_control run_pipeline flashlight
```

The pipeline's `--tip auto|press|forward`, `--order tip-push|push-tip`,
`--force-ref`, `--quality`, `--trials`, and `--auto` options remain available.
`--auto` skips interaction gates, so use it only under the existing reviewed
experiment protocol. FFS adds capture/reconstruction latency; each perceive is
fresh rather than a fast DBSCAN registration update.

A reviewed standalone FFS run can be imported without recapture:

```bash
ros2 run irb120_control episode snapshot flashlight --episode /path/to/episode \
  --ffs-run /path/to/run --reviewed-unchanged
```

The run must contain `capture/`, `recon/` and `contacts/` from the same capture.
The flag explicitly confirms review and that the object/scene has not moved.
Capture hashes, calibration and contact evidence are retained. A capture already
used before an episode motion cannot be imported again after that motion.
Import is not MoveIt or hardware-execution approval.

```
<log root>/<object>/episodes/<timestamp>/
  episode.json               source revision, stages, events, outcomes
  ffs_runs/<timestamp>/      raw capture, reconstruction, contact report, step logs
  00_perceive/               contacts.json, object_cloud.npz, snapshot.json,
                             calibrated images/depth, review graphics, source identity
  01_press_pull_tip/         wrench/pose NPZ, metadata, videos and frame stamps
  02_perceive/               fresh FFS measurement after motion
```

Contact geometry is invalid after motion, including failed motion that may have
moved the object. Failed snapshots remain failed and cannot authorize a new
interaction. No fallback detector starts when FFS is unavailable.

## Motion primitives

| Command | Purpose |
|---|---|
| `ros2 run irb120_control arc_static OBJECT --episode last` | Controlled press-pull tipping |
| `ros2 run irb120_control fwd_arc OBJECT --episode last` | Forward arc tipping; also the pipeline's forward-tip primitive |
| `ros2 run irb120_control push OBJECT --episode last` | Slow identification push |
| `ros2 run irb120_control keyboard_jog` | Manual Cartesian jog via Servo |
| `ros2 run irb120_control shove check request.json --output check.json` | Planning-only shove feasibility check |

See [forward arc](FORWARD_ARC.md), [shove check](SHOVE_CHECK.md),
[tipping model/frames](docs/tipping_model.md), and
[contact selection](../irb120_perception/CONTACT_SELECTION.md).
Use each command's `--help` for its actual supported options. `fwd_tip.py`
remains shared implementation, but `fwd_tip` is not an installed console command.

Motion primitives retain approach/contact validation, operator gates, F/T tare,
force/freshness/tracking guards, video and wrench logging, EGM cleanup and verified
return checks. Completing the finger's return does not prove the object has
returned upright. Finger arc angle is not independent visual object tilt.
The fixed production joint-6 cable window is +/-60 degrees; calibration uses its
separate configuration and ChArUco model.

The shove command checks geometry only and never dispatches motion. Its
`execution_ready` and `timing_verified` remain false. Read its guide before
interpreting a full Cartesian path as a valid launch at the requested timing.

## Official data and analysis

`run_pipeline --official --data-root PATH` retains the existing clean-checkout,
protocol/manifest and reviewed-calibration checks in `util/official_run.py`.
The official root defaults to `$IRB120_OFFICIAL_ROOT` or
`~/Documents/irb120_data/official_v1`. Do not rewrite raw official data when
condensing documentation or replaying reconstruction.

`episode note`, `episode outcome`, and `episode show` maintain episode annotations.
Estimation is available through `estimate_params`; paper/result workflows live in
[analysis](../analysis/README.md). Ground-truth/reference definitions belong to
[irb120_models](../irb120_models/README.md), controller settings to
`config/object_control.yaml`, and active F/T calibration to `config/ft_calibration.json`.

## Hardware ownership

The local `irb120_abb_hardware_interface` is intentional: its distinct plugin
identity, measured-state command seeding and EGM loss/rearm guards are required.
Do not replace it with the upstream vendor plugin during cleanup. Vendor
checkouts remain pinned by `irb120_jazzy.repos`.

[Perception workflow](../irb120_perception/docs/STEREO_OFFLINE.md) describes the
workspace scripts, venv and model dependencies. [Troubleshooting](../docs/TROUBLESHOOTING.md)
contains durable cell diagnostics. Old experiment descriptions, commands and
rollout notes are retained only in [the archive](../docs/archive/README.md).
