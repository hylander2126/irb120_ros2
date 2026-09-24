#!/usr/bin/env python3
"""Offline re-diagnosis / re-solve of a saved hand-eye sample set.

run_handeye_calibration.py now writes one handeye_samples_<camera>.yaml
(raw per-pose AX/AB data) next to every cam_tf_*.launch.py it produces, and
already prints a leave-one-out (LOO) diagnostic live: for each accepted
pose, it re-solves with that one pose left out and reports how much the
AX=XB residual improves -- a pose whose removal meaningfully helps is a
likely "polluter" (misdetection, board slipped, arm not settled, TF
glitch, ...).

This script reloads that same raw data -- no robot, no cameras, nothing
live needed -- and lets you:
  - reprint that diagnostic (e.g. to revisit a run after the fact), and
  - actually drop specific poses (--exclude-poses) and see the resulting
    residual, then write the corresponding cam_tf_<ns>_<err>mm.launch.py
    (--write-launch) once you're happy with it.

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
    _leave_one_out_diagnosis,
    _matrix_to_quat,
    _print_loo_report,
    _samples_from_dicts,
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
                   help='Defaults to the method recorded in the samples file.')
    p.add_argument('--write-launch', action='store_true',
                   help='Write a new cam_tf_<ns>_<err>mm.launch.py from this solve, same as '
                        'run_handeye_calibration.py.')
    p.add_argument('--out-dir', default=None, help='Defaults to the samples file\'s own directory.')
    args = p.parse_args()

    in_path = os.path.abspath(args.in_path)
    ns, link, board_type, saved_method, s = _load(in_path)
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

    R_sol, t_sol, rot_res, trans_res, loo_ranked = _leave_one_out_diagnosis(s, method_key)
    quat = _matrix_to_quat(R_sol)

    print(f'\n[{ns}] {n} poses -> base_link -> {link}:')
    print(f'  xyz = [{t_sol[0]:.6f}, {t_sol[1]:.6f}, {t_sol[2]:.6f}]')
    print(f'  quat(xyzw) = [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]')
    print(f'  AX=XB residual over {len(rot_res)} pose pairs: '
          f'rotation mean={rot_res.mean():.3f} deg max={rot_res.max():.3f} deg, '
          f'translation mean={trans_res.mean():.2f} mm max={trans_res.max():.2f} mm')
    if loo_ranked:
        _print_loo_report(ns, float(rot_res.mean()), float(trans_res.mean()), loo_ranked, s)
    else:
        print(f'  (skipping leave-one-out diagnostic: need >= 6 poses, have {n})')

    if args.write_launch:
        out_dir = args.out_dir or os.path.dirname(in_path)
        os.makedirs(out_dir, exist_ok=True)
        path = _write_launch_file(out_dir, ns, link, t_sol, quat, float(trans_res.mean()), n, method_key)
        print(f'\nwrote {path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
