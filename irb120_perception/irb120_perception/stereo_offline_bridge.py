#!/usr/bin/env python3
"""ros2 run entry points for the offline FFS/RealSense stereo pipeline.

These scripts live in <workspace>/scripts and need the irb_venv interpreter
(torch/open3d/pyrealsense2 aren't installed for the system interpreter this
package is built under). Each entry point here just re-execs the real script
under that venv, the same prefix trick perception.launch.py uses for the live
SAM nodes, so `ros2 run` works without the caller remembering the venv path.
See scripts/STEREO_OFFLINE.md.
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
