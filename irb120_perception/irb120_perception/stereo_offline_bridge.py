#!/usr/bin/env python3
"""`ros2 run irb120_perception stereo_offline <subcommand> [args...]`.

Subcommands: capture, reconstruct, report, contacts, worker.

These scripts live in <workspace>/scripts and need the irb_venv interpreter
(torch/open3d/pyrealsense2 aren't installed for the system interpreter this
package is built under). Each subcommand here just re-execs the matching
real script under that venv, the same prefix trick perception.launch.py uses
for the live SAM nodes, so `ros2 run` works without the caller remembering
the venv path or script location. All arguments after the subcommand pass
straight through to the real script's own argparse. See scripts/STEREO_OFFLINE.md.
"""
import os
import sys
from pathlib import Path

WORKSPACE = Path.home() / 'Documents' / 'irb120_ws'
VENV_PYTHON = Path.home() / '.virtualenvs' / 'irb_venv' / 'bin' / 'python3'


def _run(script_name):
    script = WORKSPACE / 'scripts' / script_name
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(script)] + sys.argv[1:])


def main_capture():
    _run('capture_stereo_snapshot.py')


def main_reconstruct():
    _run('reconstruct_stereo_offline.py')


def main_report():
    _run('report_stereo_offline.py')


def main_contacts():
    _run('select_stereo_contacts.py')


def main_worker():
    _run('stereo_offline_worker.py')


_SUBCOMMANDS = {
    'capture': main_capture,
    'reconstruct': main_reconstruct,
    'report': main_report,
    'contacts': main_contacts,
    'worker': main_worker,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(f"usage: ros2 run irb120_perception stereo_offline "
              f"{{{','.join(_SUBCOMMANDS)}}} [args...]")
        return 0 if len(sys.argv) >= 2 else 1

    name = sys.argv[1]
    func = _SUBCOMMANDS.get(name)
    if func is None:
        print(f"stereo_offline: unknown subcommand {name!r} "
              f"(expected one of: {', '.join(_SUBCOMMANDS)})", file=sys.stderr)
        return 1

    # Each main_* calls _run(), which os.execv's with sys.argv[1:] — shift
    # argv so that slice is exactly the subcommand's own remaining args.
    sys.argv = [f"{sys.argv[0]} {name}"] + sys.argv[2:]
    func()
