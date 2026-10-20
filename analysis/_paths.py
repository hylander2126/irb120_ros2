"""Import paths and data locations shared by the analysis scripts (no ROS needed).

Data roots (override with environment variables):
  IRB120_DEV_DATA       development archive  (default ~/Documents/irb120_data/dev_2026-10)
  IRB120_OFFICIAL_ROOT  official data        (default ~/Documents/irb120_data/official_v1)
"""
import os
import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
REPO = ANALYSIS.parent
for package in ('irb120_control', 'irb120_models', 'irb120_perception'):
    path = str(REPO / package)
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/irb120-analysis-mpl')

DATA = Path.home() / 'Documents' / 'irb120_data'
DEV_DATA = Path(os.environ.get('IRB120_DEV_DATA', DATA / 'dev_2026-10'))
OFFICIAL_ROOT = Path(os.environ.get('IRB120_OFFICIAL_ROOT', DATA / 'official_v1'))
