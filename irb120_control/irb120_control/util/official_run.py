"""Official-data mode for run_pipeline (--official).

An official episode is only recorded when the code and the inputs it depends on are pinned:
  - the irb120_ros2 checkout is clean (git_dirty false), so episode.json's commit is exact;
  - the data folder has PROTOCOL.md and MANIFEST.json (written before the campaign);
  - the hashes of the protocol, manifest, F/T calibration and camera extrinsic launch files
    go into every episode.json, so a later change to any of them is visible.
Episodes then go to <data root>/raw/ (IRB120_LOG_ROOT, inherited by the motion steps).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from irb120_control.util.runtime_log_dir import LOG_ROOT_ENV, git_commit_info

OFFICIAL_ROOT_ENV = "IRB120_OFFICIAL_ROOT"
DEFAULT_OFFICIAL_ROOT = Path.home() / "Documents" / "irb120_data" / "official_v1"
_PACKAGE_DIR = Path(__file__).resolve().parents[2]          # .../irb120_control (package)
_REPO_DIR = _PACKAGE_DIR.parent                             # .../irb120_ros2


class OfficialRunError(RuntimeError):
    pass


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _git_tag(repo_dir: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "describe", "--tags", "--exact-match"], cwd=repo_dir,
                                       stderr=subprocess.DEVNULL, text=True).strip() or None
    except Exception:
        return None


def official_root(override: str | None = None) -> Path:
    return Path(override or os.environ.get(OFFICIAL_ROOT_ENV) or DEFAULT_OFFICIAL_ROOT).expanduser()


def prepare(root_override: str | None = None, git_info: dict | None = None) -> dict:
    """Check the preconditions, point IRB120_LOG_ROOT at <root>/raw, and return the
    provenance block for episode.json. Raises OfficialRunError naming what is missing."""
    root = official_root(root_override)
    git = git_info if git_info is not None else git_commit_info()
    problems = []
    if git.get("git_commit") is None:
        problems.append("git commit unknown (run from the irb120_ros2 checkout)")
    elif git.get("git_dirty"):
        problems.append("irb120_ros2 has uncommitted changes; commit (and tag) before official runs")
    for name in ("PROTOCOL.md", "MANIFEST.json"):
        if not (root / name).exists():
            problems.append(f"{root / name} is missing")
    ft = _PACKAGE_DIR / "config" / "ft_calibration.json"
    if not ft.exists():
        problems.append(f"{ft} is missing (the preprocessor would fall back to legacy constants)")
    if problems:
        raise OfficialRunError("Not recording official data:\n  - " + "\n  - ".join(problems))
    extrinsics = sorted((_REPO_DIR / "irb120_handeye" / "launch").glob("cam_tf_*.launch.py"))
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    os.environ[LOG_ROOT_ENV] = str(raw)
    return {
        "data_root": str(root),
        "git_tag": _git_tag(_REPO_DIR),
        "protocol_sha256": _sha256(root / "PROTOCOL.md"),
        "manifest_sha256": _sha256(root / "MANIFEST.json"),
        "ft_calibration_sha256": _sha256(ft),
        "camera_extrinsics_sha256": {p.name: _sha256(p) for p in extrinsics},
    }
