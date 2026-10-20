"""Camera agreement check from one perceive snapshot (no ROS imports).

Each camera's own view of the table around the object is fitted with a robust plane, and
the plane heights at the object are compared across cameras. All cameras see the same
table, so their heights should agree; a spread means camera-to-robot registration (or a
per-camera depth offset) is off by about that much, and every perceived support location
inherits it. In the October 2026 development data the spread was 2-3 mm between cam1 and
cam3 and ~8 mm for cam2; in the cylinder joint fit that was the largest single mass
uncertainty.

Input: support_scene_cloud.npz (support_cloud_recording) + object_cloud.npz in the folder.

CLI:  python3 -m irb120_control.util.registration_check <perceive folder> [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

TABLE_Z_BASE_LINK = -0.021      # configured table height in base_link
TOLERANCE_M = 0.0015            # max table-height spread across cameras to pass
ANNULUS_M = (0.015, 0.12)       # table ring: this far beyond the object's footprint
MIN_TABLE_POINTS = 200


def camera_tables(scene_xyz, camera_id, object_xyz) -> dict:
    center = np.median(object_xyz[:, :2], axis=0)
    footprint = float(np.quantile(np.linalg.norm(object_xyz[:, :2] - center, axis=1), 0.95))
    out = {}
    for cam in np.unique(camera_id):
        p = scene_xyz[camera_id == cam]
        rho = np.linalg.norm(p[:, :2] - center, axis=1)
        near = ((rho > footprint + ANNULUS_M[0]) & (rho < footprint + ANNULUS_M[1])
                & (np.abs(p[:, 2] - TABLE_Z_BASE_LINK) < 0.015))
        q = p[near]
        if len(q) < MIN_TABLE_POINTS:
            out[int(cam)] = dict(status="insufficient table points", points=int(len(q)))
            continue
        X = np.c_[q[:, :2] - center, np.ones(len(q))]
        sol = least_squares(lambda c: X @ c - q[:, 2], [0, 0, TABLE_Z_BASE_LINK], loss="soft_l1", f_scale=0.001)
        r = sol.fun
        out[int(cam)] = dict(status="ok", points=int(len(q)),
                             height_at_object_m=float(sol.x[2] - TABLE_Z_BASE_LINK),
                             tilt_deg=float(np.degrees(np.arctan(np.hypot(*sol.x[:2])))),
                             residual_sigma_m=float(1.4826 * np.median(np.abs(r - np.median(r)))))
    return out


def check_folder(folder, tolerance_m: float = TOLERANCE_M, cloud_dir=None) -> dict:
    """cloud_dir: where support_scene_cloud.npz lives if not in the snapshot (offline backfill)."""
    folder = Path(folder)
    scene_file = Path(cloud_dir or folder) / "support_scene_cloud.npz"
    object_file = folder / "object_cloud.npz"
    if not (scene_file.exists() and object_file.exists()):
        return dict(status="unavailable", reason="needs support_scene_cloud.npz and object_cloud.npz", folder=str(folder))
    scene = np.load(scene_file)
    tables = camera_tables(scene["xyz"].astype(float), scene["camera_id"], np.load(object_file)["xyz"].astype(float))
    heights = {c: t["height_at_object_m"] for c, t in tables.items() if t["status"] == "ok"}
    if len(heights) < 2:
        return dict(status="unavailable", reason="fewer than two cameras see the table", cameras=tables, folder=str(folder))
    spread = float(np.ptp(list(heights.values())))
    return dict(status="pass" if spread <= tolerance_m else "fail", spread_m=spread, tolerance_m=tolerance_m,
                cameras={f"cam{c}": t for c, t in tables.items()}, folder=str(folder))


def describe(result: dict) -> str:
    if result["status"] == "unavailable":
        return f"camera agreement: unavailable ({result['reason']})"
    heights = ", ".join(f"{c} {t['height_at_object_m'] * 1000:+.1f}" for c, t in result["cameras"].items()
                        if t["status"] == "ok")
    return (f"camera agreement: {result['status'].upper()} - table height spread "
            f"{result['spread_m'] * 1000:.1f} mm (tolerance {result['tolerance_m'] * 1000:.1f}); "
            f"per camera vs configured table [mm]: {heights}")


def main(argv=None) -> int:
    folders = sys.argv[1:] if argv is None else argv
    if not folders:
        print(__doc__)
        return 2
    worst = 0
    for f in folders:
        r = check_folder(f)
        print(f"{f}\n    {describe(r)}")
        worst = max(worst, {"pass": 0, "unavailable": 1, "fail": 2}[r["status"]])
    return worst


if __name__ == "__main__":
    sys.exit(main())
