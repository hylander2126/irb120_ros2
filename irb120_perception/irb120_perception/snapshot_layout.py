"""Where files live in a perceive stage, and where episodes live under a log root.

Perceive stage (since 2026-10-24), each file written once:
  NN_perceive/
    capture/     raw stereo + RGB (camN.npz), capture.json
    recon/       FFS reconstruction: camN_aligned_depth.npz, camN_camera_info.json,
                 camN_color.png, masks, pivot reviews, shared object_cloud.npz/json
    logs/        one log per FFS step
    contacts.json, handoff.json, contacts.png
    object_cloud.npz/json     links to the shared recon/ files (no duplicate data)
    snapshot.json, support_scene_cloud.*

Older stages kept copies of the per-camera files at the stage root; snapshot_file()
finds a file in either layout. Episodes are <root>/<object>/<stamp>/ (older:
<root>/<object>/episodes/<stamp>/); episode_jsons() finds both.
"""
from pathlib import Path

SUBDIRS = ('', 'recon', 'capture')


def snapshot_file(folder, name: str) -> Path:
    """`name` in a perceive stage: the stage root, then recon/, then capture/.
    Returns the stage-root path when it is nowhere, so callers' exists() checks still work."""
    folder = Path(folder)
    for sub in SUBDIRS:
        path = folder/sub/name
        if path.exists():
            return path
    return folder/name


def episode_jsons(root, obj: str = '*', recursive: bool = False) -> list:
    """episode.json paths for `obj` under a log root, both layouts, sorted by episode name.
    recursive: search every depth below root (suites of copied object folders)."""
    root = Path(root)
    if recursive:
        found = set(root.glob('**/episode.json'))
    else:
        found = set(root.glob(f'{obj}/*/episode.json')) | set(root.glob(f'{obj}/episodes/*/episode.json'))
    return sorted(found, key=lambda p: (p.parent.name, str(p)))
