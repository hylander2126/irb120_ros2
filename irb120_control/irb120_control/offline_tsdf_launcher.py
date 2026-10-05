"""ROS console-script wrapper for the Open3D CPU virtual environment."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    python = os.environ.get('IRB120_TSDF_PYTHON', '/home/hylander/irb_venv/bin/python3')
    package_root = str(Path(__file__).resolve().parent.parent)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = package_root + os.pathsep + environment.get('PYTHONPATH', '')
    try:
        return subprocess.run([python, '-m', 'irb120_control.offline_tsdf', *sys.argv[1:]], env=environment).returncode
    except FileNotFoundError:
        print(f'Open3D Python was not found at {python}. Set IRB120_TSDF_PYTHON to its interpreter.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
