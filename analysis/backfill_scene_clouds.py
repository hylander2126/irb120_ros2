"""Rebuild support scene clouds for perceive snapshots that predate their recording.

    python3 backfill_scene_clouds.py <data folder> [...]

Uses each snapshot's saved aligned depth and camera calibration (the same function the live
pipeline calls); writes to the mirror under _paths.DERIVED, never into the snapshot.
"""
import sys
from pathlib import Path

import _paths
from irb120_control.util.support_cloud_recording import record_support_scene


def main(roots):
    for root in roots:
        for snap in sorted(Path(root).expanduser().resolve().glob('**/*_perceive')):
            if not (snap / 'object_cloud.npz').exists() or _paths.scene_dir(snap) is not None:
                continue
            out = _paths.DERIVED / snap.relative_to(_paths.DATA)
            try:
                r = record_support_scene(snap, out_dir=out)
                print(f'{snap.relative_to(_paths.DATA)}: {r["points"]} points, {r["views_recorded"]} views')
            except Exception as exc:
                print(f'{snap.relative_to(_paths.DATA)}: skipped ({exc})')


if __name__ == '__main__':
    main(sys.argv[1:])
