#!/usr/bin/env python3
"""Run the whole pipeline on one object, one episode per trial:

    perceive -> push -> perceive -> press-pull tip -> perceive -> outcome

The motion steps are the normal `push` and `arc_static` scripts, run as child
processes into the same episode. Each takes its targets from the snapshot
before it and takes the snapshot after it (see their docstrings), so this
script only sequences them: it stops the episode at a failed push (outcome
'aborted', with the reason) and otherwise asks you for the outcome at the end.
Each step's result is also written to episode.json's `events`.

TODO: decide forward tip vs press-pull here once the forward-tip controller exists.

Usage:
    ros2 run irb120_control run_pipeline flashlight
    ros2 run irb120_control run_pipeline flashlight --trials 10
"""

import argparse
import subprocess
import sys

import rclpy

from irb120_control.util.episode import OUTCOMES, Episode
from irb120_control.util.runtime_log_dir import VALID_OBJECTS


def run_step(episode: Episode, kind: str, command: list) -> tuple:
    """Run one motion script into `episode`. Returns (motion stage status, snapshot after it ok?)."""
    n_before = len(episode.load()["stages"])
    command = ["ros2", "run", "irb120_control", *command, "--episode", str(episode.path)]
    print(f"\n[pipeline] $ {' '.join(command)}")
    returncode = subprocess.run(command).returncode
    new = episode.load()["stages"][n_before:]
    status = next((s["status"] for s in reversed(new) if s["kind"] == kind), "missing")
    snapshot_ok = bool(new) and new[-1]["kind"] == "perceive" and new[-1]["status"] == "ok"
    episode.add_event(f"{kind}: exit {returncode}, stages " + ", ".join(f"{s['dir']} {s['status']}" for s in new))
    return status, snapshot_ok


def run_episode(node, obj: str, quality: str, note: str) -> Episode:
    episode = Episode.create(obj, node, note)

    status, snapshot_ok = run_step(episode, "push", ["push", obj])
    if status != "ok" or not snapshot_ok:
        episode.set_outcome("aborted", f"push {status}" + ("" if snapshot_ok else ", no usable snapshot after it"))
        return episode

    status, snapshot_ok = run_step(episode, "press_pull_tip", ["arc_static", obj, "--quality", quality])
    if not snapshot_ok:
        episode.add_note("no usable snapshot after the tip")

    print(f"\n[pipeline] press-pull tip: {status}")
    result = ""
    while result not in OUTCOMES:
        result = input(f"Outcome? {'/'.join(OUTCOMES)}: ").strip().lower()
    episode.set_outcome(result, input("Reason / note (Enter to skip): ").strip())
    return episode


def main(args=None) -> int:
    raw_args = list(sys.argv[1:] if args is None else args)
    ros_args_index = raw_args.index("--ros-args") if "--ros-args" in raw_args else len(raw_args)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("object", choices=sorted(VALID_OBJECTS))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--quality", choices=["h264", "lossless"], default="h264", help="tip video quality")
    parsed = parser.parse_args(raw_args[:ros_args_index])

    rclpy.init(args=raw_args[ros_args_index:])
    node = rclpy.create_node("run_pipeline")  # only to record camera extrinsics into each episode
    episodes = []
    try:
        for trial in range(1, parsed.trials + 1):
            if trial > 1 and input(f"\nReset the {parsed.object} for trial {trial}, then Enter (q to stop): ") == "q":
                break
            episodes.append(run_episode(node, parsed.object, parsed.quality, f"trial {trial}/{parsed.trials}"))
            print("\n" + episodes[-1].summary())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print(f"\n[pipeline] {len(episodes)} episode(s):")
    for episode in episodes:
        print(f"    {episode.path.name}  {(episode.load()['outcome'] or {}).get('result')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
