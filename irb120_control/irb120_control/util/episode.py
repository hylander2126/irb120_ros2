"""Episodes: one folder + one summary file per pipeline run on one object.

  runtime_logs/<object>/episodes/<YYYYmmdd_HHMMSS>/
    episode.json         # summary: git commit, camera extrinsics, stages, events, notes, outcome
    00_perceive/         # perception_snapshot.take_snapshot()
    01_push/             # push.py: F/T + pose .npz, metadata .json, videos
    02_perceive/
    03_press_pull_tip/   # arc_static.py
    04_perceive/

Every stage entry in episode.json has its folder, start/end time, status
('running' until it ends, so a crash is visible), results and file list.

CLI for the bits done by hand:
  ros2 run irb120_control episode snapshot flashlight   # new episode with one snapshot; no motion
  ros2 run irb120_control episode show
  ros2 run irb120_control episode note "object slid 5 mm during push"
  ros2 run irb120_control episode outcome success --reason "tipped and settled back"
"""

import argparse
import json
import os
import socket
from datetime import datetime
from pathlib import Path

from irb120_control.util.runtime_log_dir import git_commit_info, load_object_params, resolve_workspace_root

OUTCOMES = ("success", "partial", "failure", "aborted")
CAMERA_LINKS = ("realsense_link", "realsense2_link", "realsense3_link")


def write_json_atomic(path: Path, data) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _episodes_dir(object_name: str = "*") -> Path:
    return resolve_workspace_root() / "runtime_logs" / object_name / "episodes"


class Stage:
    def __init__(self, episode, index: int, path: Path):
        self.episode, self.index, self.path = episode, index, path

    def end(self, status: str, **results) -> None:
        """status: 'ok', 'failed' or 'aborted'."""
        self.episode.end_stage(self, status, **results)


class Episode:
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def create(cls, object_name: str, node=None, note: str = "") -> "Episode":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, n = _episodes_dir(object_name) / stamp, 1
        while path.exists():  # a second episode within the same second
            n += 1
            path = path.with_name(f"{stamp}_{n}")
        path.mkdir(parents=True)
        calibration = {}
        if node is not None:  # camera extrinsics as TF has them now (whichever cam_tf files are launched)
            from irb120_control.util.perception_snapshot import lookup
            for link in CAMERA_LINKS:
                tf = lookup(node, "base_link", link)
                calibration[link] = None if tf is None else {"R": tf[0].tolist(), "t": tf[1].tolist()}
        write_json_atomic(path / "episode.json", {
            "object": object_name,
            "created": _now(),
            "host": socket.gethostname(),
            **git_commit_info(),
            "object_params": load_object_params(object_name),
            "camera_extrinsics_in_base_link": calibration,
            "notes": [{"time": _now(), "text": note}] if note else [],
            "events": [],
            "stages": [],
            "outcome": None,
        })
        print(f"[episode] created {path}")
        return cls(path)

    @classmethod
    def resolve(cls, spec: str, object_name: str, node=None) -> "Episode":
        """'new', 'last' (most recent episode for this object), or a path."""
        if spec == "new":
            return cls.create(object_name, node)
        if spec == "last":
            episodes = sorted(_episodes_dir(object_name).glob("*/episode.json"))
            if not episodes:
                raise FileNotFoundError(f"no episodes for {object_name}")
            return cls(episodes[-1].parent)
        return cls(Path(spec).expanduser())

    # --- episode.json

    def load(self) -> dict:
        with open(self.path / "episode.json") as f:
            return json.load(f)

    def _update(self, change) -> None:
        data = self.load()
        change(data)
        write_json_atomic(self.path / "episode.json", data)

    def last_stage_kind(self):
        stages = self.load()["stages"]
        return stages[-1]["kind"] if stages else None

    def begin_stage(self, kind: str, **info) -> Stage:
        index = len(self.load()["stages"])
        folder = f"{index:02d}_{kind}"
        self._update(lambda d: d["stages"].append({
            "dir": folder, "kind": kind, **info, "started": _now(), "ended": None,
            "status": "running", "results": {}, "files": []}))
        (self.path / folder).mkdir()
        print(f"[episode] {folder} started")
        return Stage(self, index, self.path / folder)

    def end_stage(self, stage: Stage, status: str, **results) -> None:
        files = [{"name": str(p.relative_to(stage.path)), "bytes": p.stat().st_size}
                 for p in sorted(stage.path.rglob("*")) if p.is_file()]

        def change(d):
            d["stages"][stage.index].update(ended=_now(), status=status, results=results, files=files)
        self._update(change)
        print(f"[episode] {stage.path.name} -> {status} "
              f"({len(files)} files, {sum(f['bytes'] for f in files) / 1e6:.1f} MB)")

    def add_event(self, text: str) -> None:
        self._update(lambda d: d["events"].append({"time": _now(), "text": text}))

    def add_note(self, text: str) -> None:
        self._update(lambda d: d["notes"].append({"time": _now(), "text": text}))

    def set_outcome(self, result: str, reason: str = "") -> None:
        self._update(lambda d: d.__setitem__("outcome", {"result": result, "reason": reason, "time": _now()}))

    def summary(self) -> str:
        d = self.load()
        lines = [f"Episode {self.path}  ({d['object']}, git {str(d.get('git_commit'))[:10]})"]
        for s in d["stages"]:
            mb = sum(f["bytes"] for f in s["files"]) / 1e6
            short = {k: v for k, v in s["results"].items() if not isinstance(v, (dict, list))}
            lines.append(f"  {s['dir']:<20} {s['status']:<8} {mb:6.1f} MB  {short}")
        lines += [f"  note: {n['text']}" for n in d["notes"]]
        lines.append(f"  outcome: {d['outcome']}")
        return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["snapshot", "show", "note", "outcome"])
    p.add_argument("text", nargs="?", default="",
                   help="snapshot: object name; note: the text; outcome: " + "/".join(OUTCOMES))
    p.add_argument("--reason", default="", help="outcome reason")
    p.add_argument("--episode", default=None, help="episode folder (default: the most recent one)")
    args = p.parse_args()

    if args.command == "snapshot":
        import rclpy
        from irb120_control.util.perception_snapshot import take_snapshot
        rclpy.init()
        node = rclpy.create_node("episode_snapshot")
        episode = Episode.create(args.text, node, "snapshot only")
        take_snapshot(node, episode, label="manual")
        node.destroy_node()
        rclpy.shutdown()
        print(episode.summary())
        return

    if args.episode:
        episode = Episode(args.episode)
    else:
        episodes = sorted(_episodes_dir().glob("*/episode.json"), key=lambda f: f.stat().st_mtime)
        if not episodes:
            raise SystemExit("no episodes yet")
        episode = Episode(episodes[-1].parent)

    if args.command == "note":
        episode.add_note(args.text)
    elif args.command == "outcome":
        if args.text not in OUTCOMES:
            raise SystemExit(f"outcome must be one of {OUTCOMES}")
        episode.set_outcome(args.text, args.reason)
    print(episode.summary())


if __name__ == "__main__":
    main()
