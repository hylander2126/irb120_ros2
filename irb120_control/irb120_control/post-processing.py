#!/usr/bin/env python3
"""Plotting script for IRB120 F/T NPZ logs (Push and Squash-Pull)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_forces(npz_path: Path) -> None:
    print(f"Loading {npz_path} ...")
    data = np.load(str(npz_path))

    # Handle both the new (ft_time_s) and old (time_s) formats
    if 'ft_time_s' in data:
        t = data['ft_time_s']
    elif 'time_s' in data:
        t = data['time_s']
    else:
        print(f"Error: No timeline found in {npz_path.name}")
        return

    fx = data['fx']
    fy = data['fy']
    fz = data['fz']

    plt.figure(figsize=(10, 6))
    plt.plot(t, fx, label='Force X')
    plt.plot(t, fy, label='Force Y')
    plt.plot(t, fz, label='Force Z')
    
    plt.title(f"Force Components - {npz_path.name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)


def find_latest_npz(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    files = list(directory.glob("*.npz"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot F/T data from push/squash_pull .npz logs.")
    parser.add_argument(
        "file", 
        nargs="?", 
        help="Optional: Path to a specific .npz file to plot. If omitted, finds the latest push and squash logs automatically."
    )
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.is_file():
            print(f"File not found: {args.file}")
            return 1
        plot_forces(path)
        return 0

    # Auto-find the workspace root by looking up the tree for 'runtime_logs'
    current_dir = Path(__file__).resolve().parent
    runtime_logs_dir = None
    for p in [current_dir] + list(current_dir.parents):
        if (p / "runtime_logs").is_dir():
            runtime_logs_dir = p / "runtime_logs"
            break
            
    if not runtime_logs_dir:
        print("Could not find 'runtime_logs' directory in workspace.")
        return 1

    push_dir = runtime_logs_dir / "push"
    squash_dir = runtime_logs_dir / "squash_pull"

    latest_push = find_latest_npz(push_dir)
    latest_squash = find_latest_npz(squash_dir)

    if latest_push:
        print("Found latest push log.")
        plot_forces(latest_push)
    else:
        print("No push logs found.")

    if latest_squash:
        print("Found latest squash_pull log.")
        plot_forces(latest_squash)
    else:
        print("No squash_pull logs found.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())