#!/usr/bin/env python3
"""Single `ros2 run` entry point for irb120_handeye's offline CLI tools.

Usage:
  ros2 run irb120_handeye handeye <subcommand> [args...]

Subcommands:
  calibrate         run_handeye_calibration  (headless multi-camera solve)
  run-poses         run_calibration_poses    (MoveIt HandEyeCalibration panel poses)
  record-pose       record_calibration_pose  (interactive pose teacher)
  generate-target   generate_charuco_target  (print-ready ChArUco board)
  diagnose          diagnose_handeye_samples (offline re-diagnose/re-solve)
  check-depth-pnp   check_depth_vs_pnp       (depth vs. PnP comparison)

Everything after the subcommand passes straight through to that tool's own argparse, unchanged — e.g. `handeye calibrate --cameras realsense3` is exactly `run_handeye_calibration --cameras realsense3` was before this was consolidated into one console script. This exists only to keep `ros2 run irb120_handeye <TAB>` from listing six near-identical-looking offline tools; none of these run as ROS nodes, so none of them need to be launched by a launch file.
"""
import importlib
import sys

_SUBCOMMANDS = {
    'calibrate': 'irb120_handeye.run_handeye_calibration',
    'run-poses': 'irb120_handeye.run_calibration_poses',
    'record-pose': 'irb120_handeye.record_calibration_pose',
    'generate-target': 'irb120_handeye.generate_charuco_target',
    'diagnose': 'irb120_handeye.diagnose_handeye_samples',
    'check-depth-pnp': 'irb120_handeye.check_depth_vs_pnp',
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        return 0 if len(sys.argv) >= 2 else 1

    name = sys.argv[1]
    module_name = _SUBCOMMANDS.get(name)
    if module_name is None:
        print(f"handeye: unknown subcommand {name!r} "
              f"(expected one of: {', '.join(_SUBCOMMANDS)})", file=sys.stderr)
        return 1

    # Re-point argv so the submodule's own argparse (which parses sys.argv[1:])
    # sees exactly what it would if it were still its own console script.
    sys.argv = [f"{sys.argv[0]} {name}"] + sys.argv[2:]
    return importlib.import_module(module_name).main()


if __name__ == '__main__':
    sys.exit(main())
