#!/usr/bin/env python3
"""Offline re-diagnosis / re-solve of a saved hand-eye sample set.

run_handeye_calibration.py writes one handeye_samples_<camera>.yaml (raw
per-pose transforms and board-corner pixels) next to every cam_tf_*.launch.py
it produces, and prints a per-pose reprojection-error report: a pose whose
corners reproject well above the set's median disagrees with the others
(misdetection, board slipped, arm not settled, TF glitch, ...).

This script reloads that same raw data -- no robot, no cameras, nothing
live needed -- and lets you:
  - reprint that report (e.g. to revisit a run after the fact), and
  - actually drop specific poses (--exclude-poses) and re-solve, then write
    the corresponding cam_tf_<ns>_<err>px.launch.py (--write-launch) once
    you're happy with it.

Sample files written before the reprojection diagnostic don't contain
corner pixels and can't be loaded; re-run run_handeye_calibration.

The "pose#" in all of this is the "Pose i/total" index run_handeye_calibration.py
printed live for that pose (1-based, into the --pose-file used for that run)
-- NOT a position in the accepted-samples list, since some poses may have
been missed/rejected for a given camera.

Usage:
  ros2 run irb120_handeye diagnose_handeye_samples --in ~/handeye_samples_realsense3.yaml
  ros2 run irb120_handeye diagnose_handeye_samples --in ~/handeye_samples_realsense3.yaml \\
      --exclude-poses "4,9" --write-launch
"""
import argparse
import os

from irb120_handeye.run_handeye_calibration import (
    METHODS,
    _print_reprojection_report,
    _print_solution_summary,
    _samples_from_dicts,
    _solve_handeye,
    _write_launch_file,
)

import yaml


def _load(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    s = _samples_from_dicts(data['samples'])
    return data.get('camera'), data.get('link_frame'), data.get('board_type'), data.get('method'), s


def _exclude(s: dict, pose_nums: set) -> dict:
    keep = [k for k, p in enumerate(s['pose_idx']) if p not in pose_nums]
    dropped = len(s['pose_idx']) - len(keep)
    if dropped != len(pose_nums):
        found = {s['pose_idx'][k] for k in range(len(s['pose_idx']))}
        missing = pose_nums - found
        if missing:
            print(f'  note: pose(s) {sorted(missing)} not present in this sample set (nothing to exclude)')
    return {key: [vals[k] for k in keep] for key, vals in s.items()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--in', dest='in_path', required=True, help='handeye_samples_<camera>.yaml to load.')
    p.add_argument('--exclude-poses', default='', metavar='"P,P,..."',
                   help='Pose numbers to drop before solving (the "Pose i/total" index printed live) -- '
                        'comma-separated, in quotes so the shell passes it through as one argument, '
                        'e.g. --exclude-poses "4,9,17".')
    p.add_argument('--method', choices=sorted(METHODS), default=None,
                   help='Closed-form solver that seeds the reprojection refinement. '
                        'Defaults to the method recorded in the samples file.')
    p.add_argument('--write-launch', action='store_true',
                   help='Write a new cam_tf_<ns>_<err>mm.launch.py from this solve, same as '
                        'run_handeye_calibration.py.')
    p.add_argument('--out-dir', default=None, help='Defaults to the samples file\'s own directory.')
    args = p.parse_args()

    in_path = os.path.abspath(args.in_path)
    try:
        ns, link, board_type, saved_method, s = _load(in_path)
    except ValueError as exc:
        print(f'Cannot load {in_path}: {exc}')
        return 1
    method_key = args.method or saved_method or 'park'
    n_total = len(s['R_g2b'])
    print(f'Loaded {n_total} samples for [{ns}] from {in_path} (board={board_type}, method={method_key})')

    exclude = {int(x) for x in args.exclude_poses.split(',') if x.strip()}
    if exclude:
        s = _exclude(s, exclude)
        print(f'Excluded pose(s) {sorted(exclude)}: {len(s["R_g2b"])}/{n_total} samples remain')

    n = len(s['R_g2b'])
    if n < 3:
        print(f'Only {n} samples remain -- cv2.calibrateHandEye needs >= 3. Exclude fewer poses.')
        return 1

    sol = _solve_handeye(s, method_key)
    print(f'\n[{ns}] {n} poses')
    _print_solution_summary(link, sol, method_key)
    _print_reprojection_report(ns, sol, s)

    if args.write_launch:
        out_dir = args.out_dir or os.path.dirname(in_path)
        os.makedirs(out_dir, exist_ok=True)
        path = _write_launch_file(out_dir, ns, link, sol, n, method_key)
        print(f'\nwrote {path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
