#!/usr/bin/env python3
"""Run the whole pipeline on one object, one episode per trial:

    perceive -> tip -> perceive -> push -> perceive -> outcome     (default, --order tip-push)
    perceive -> push -> perceive -> tip -> perceive -> outcome     (--order push-tip)

Why tip first: see "Step order" in the package README.

The object name is only the episode label; nothing object-specific is looked
up (the press force is --force-ref). This script takes the first snapshot
itself; the motion steps are the normal `push` and `arc_static` scripts, run as
child processes into the same episode. Each takes its targets from the latest
snapshot and takes the next one after returning home.

Before every motion step it prints the snapshot that step will use (contacts,
pivot, approach targets) and opens its final_selection image, then waits:
Enter runs the step, 'r' re-perceives (e.g. after nudging the object), 'q'
stops the episode. --auto skips these waits and stops at the first failed step.
Each step's result is also written to episode.json's `events`.

Ctrl-C during a motion step is passed to that step, and this script waits for
it to finish its own cleanup (stop, home, save logs) before stopping.

TODO: forward (top-push) tip once its controller exists -- see choose_tip().

Usage:
    ros2 run irb120_control run_pipeline heart_a
    ros2 run irb120_control run_pipeline heart_a --trials 10 --force-ref 7
    ros2 run irb120_control run_pipeline heart_a --order push-tip --auto
"""

import argparse
import json
import math
import subprocess
import sys

import numpy as np
import rclpy

from irb120_control.push import PUSH_DISTANCE, PUSH_STANDOFF
from irb120_control.util.episode import OUTCOMES, Episode
from irb120_control.util.perception_snapshot import take_snapshot
from irb120_control.util.push_geometry import push_approach
from irb120_perception.press_targets import PRESS_STANDOFF

DEFAULT_FORCE_REF_N = 5.0
PUSH_MAX_NORMAL_ANGLE_DEG = 20.0  # push.py refuses contacts whose normal is further than this from -X

# Motion step kind -> (contact it needs, description for the prompt)
STEPS = {
    "press_pull_tip": ("press", "press-pull tip"),
    "push": ("planar_push", "push"),
}


def choose_tip(contacts: dict) -> str:
    """Which tip to run on this snapshot.

    TODO: decide forward (top-push) tip vs press-pull here once the forward-tip
    controller exists. Always press-pull until then.
    """
    return "press_pull_tip"


def step_command(kind: str, args) -> list:
    if kind == "press_pull_tip":
        return ["arc_static", args.object, "--force-ref", str(args.force_ref), "--quality", args.quality]
    if kind == "push":
        return ["push", args.object]
    raise ValueError(f"no command for step {kind}")


# ------------------------------------------------------------------ snapshot summary

def _mm(p) -> str:
    return "(" + ", ".join(f"{1000 * v:6.0f}" for v in p) + ") mm"


def _latest_perceive(episode: Episode):
    return next((s for s in reversed(episode.load()["stages"]) if s["kind"] == "perceive"), None)


def show_snapshot(episode: Episode, open_images: bool) -> dict | None:
    """Print the latest snapshot's contacts and the targets the motion steps would derive
    from them. Returns its contacts, or None if the snapshot failed."""
    stage = _latest_perceive(episode)
    if stage is None:
        print("\n[pipeline] no snapshot in this episode")
        return None
    folder = episode.path / stage["dir"]
    snapshot_file = folder / "snapshot.json"
    record = json.loads(snapshot_file.read_text()) if snapshot_file.exists() else {}
    print(f"\n=== {stage['dir']} ({record.get('label') or '-'}): {stage['status']}  {folder}")
    if not record.get("ok"):
        print(f"    FAILED: {record.get('error', 'no snapshot.json')}")
        return None

    view = (record.get("experimental_view_check") or {}).get("decision")
    rgb = record.get("rgb_pivot") or {}
    print(f"    {record.get('points')} points (DBSCAN {record.get('dbscan_points')})"
          f"   view check: {view or '-'}"
          f"   RGB pivot: {'applied' if rgb.get('applied') else 'not applied'}"
          + (f" ({rgb['reason']})" if rgb.get("reason") and not rgb.get("applied") else ""))
    print("    all positions in base_link")
    contacts = json.loads((folder / "contacts.json").read_text())

    press = contacts.get("press") or {}
    if press.get("available"):
        g = press["geometry"]
        ball, pivot = np.asarray(press["ball_center"]), np.asarray(g["pivot"])
        radius = math.hypot(ball[0] - pivot[0], ball[2] - pivot[2])
        print(f"  press        contact {_mm(press['point'])}   pivot {_mm(pivot)} [{g.get('pivot_source', '?')}]")
        print(f"               pre-squash ball {_mm(ball + [0, 0, PRESS_STANDOFF])}"
              f"   arc radius {1000 * radius:.0f} mm   pull dir {np.round(g['direction'], 2).tolist()}")
    else:
        print(f"  press        NONE: {press.get('reason')}")

    forward = contacts.get("forward_tip") or {}
    if forward.get("available"):
        pivot = forward["geometry"]["pivot"]
        print(f"  forward_tip  contact {_mm(forward['point'])}   pivot {_mm(pivot)}"
              f"   height above pivot {1000 * (forward['point'][2] - pivot[2]):.0f} mm   (not implemented)")
    else:
        print(f"  forward_tip  NONE: {forward.get('reason')}")

    push = contacts.get("planar_push") or {}
    if push.get("available"):
        angle = push.get("normal_angle_deg", float("nan"))
        verdict = "ok" if angle <= PUSH_MAX_NORMAL_ANGLE_DEG else f"push will refuse (> {PUSH_MAX_NORMAL_ANGLE_DEG:.0f} deg)"
        print(f"  planar_push  contact {_mm(push['point'])}   normal {angle:.1f} deg from -X ({verdict})")
        try:
            cloud = np.load(folder / "object_cloud.npz")["xyz"]
            pre, travel = push_approach(push["ball_center"], cloud, PUSH_STANDOFF, PUSH_DISTANCE)
            print(f"               pre-push ball {_mm(pre)}   travel {1000 * travel:.0f} mm in +X")
        except (ValueError, OSError, KeyError) as exc:
            print(f"               pre-push: {exc}")
    else:
        print(f"  planar_push  NONE: {push.get('reason')}")

    image = next((folder / n for n in ("final_selection.jpg", "cam1_overlay.png") if (folder / n).exists()), None)
    if image is not None:
        print(f"    image: {image}")
        if open_images:
            subprocess.Popen(["xdg-open", str(image)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return contacts


def gate(label: str, ready: bool, auto: bool) -> str:
    """'go', 'r' (re-perceive) or 'q' (stop the episode)."""
    if auto:
        return "go" if ready else "q"
    prompt = (f"\n[pipeline] Next: {label}. Enter = go, r = re-perceive, q = stop episode: " if ready else
              f"\n[pipeline] Can't run {label} on this snapshot. r = re-perceive, q = stop episode: ")
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return "q"
        if answer == "" and ready:
            return "go"
        if answer in ("r", "q"):
            return answer


# ------------------------------------------------------------------ steps

def run_step(episode: Episode, kind: str, command: list) -> tuple:
    """Run one motion script into `episode`. Returns (its motion stage entry or None,
    snapshot after it ok?). Ctrl-C is left to the child to clean up after; it is
    re-raised here once the child has exited."""
    n_before = len(episode.load()["stages"])
    command = ["ros2", "run", "irb120_control", *command, "--episode", str(episode.path)]
    print(f"\n[pipeline] $ {' '.join(command)}")
    proc = subprocess.Popen(command)
    interrupted = False
    while True:
        try:
            returncode = proc.wait()
            break
        except KeyboardInterrupt:
            interrupted = True
            print("\n[pipeline] Ctrl-C: waiting for the step to stop and clean up...")
    new = episode.load()["stages"][n_before:]
    stage = next((s for s in reversed(new) if s["kind"] == kind), None)
    snapshot_ok = bool(new) and new[-1]["kind"] == "perceive" and new[-1]["status"] == "ok"
    episode.add_event(f"{kind}: exit {returncode}, stages " + ", ".join(f"{s['dir']} {s['status']}" for s in new))
    if interrupted:
        raise KeyboardInterrupt
    return stage, snapshot_ok


def run_episode(node, args, note: str) -> Episode:
    episode = Episode.create(args.object, node, note, allow_unconfigured_object=True)
    episode.add_event(f"pipeline: order {args.order}, force_ref {args.force_ref} N, "
                      f"{'auto' if args.auto else 'step'} mode")
    take_snapshot(node, episode, label="initial")

    for phase in args.order.split("-"):
        while True:
            contacts = show_snapshot(episode, args.open_images)
            kind = choose_tip(contacts or {}) if phase == "tip" else "push"
            needed, label = STEPS[kind]
            ready = contacts is not None and (contacts.get(needed) or {}).get("available", False)
            answer = gate(label, ready, args.auto)
            if answer == "r":
                take_snapshot(node, episode, label="retake")
                continue
            break
        if answer == "q":
            episode.set_outcome("aborted", f"stopped before {label}"
                                + ("" if ready else f": no usable {needed} contact"))
            return episode

        stage, snapshot_ok = run_step(episode, kind, step_command(kind, args))
        status = stage["status"] if stage else "missing"
        results = {k: v for k, v in (stage or {}).get("results", {}).items() if not isinstance(v, (dict, list))}
        print(f"\n[pipeline] {label}: {status}  {results}")
        if not snapshot_ok:
            episode.set_outcome("aborted", f"{label} {status}, no usable snapshot after it")
            return episode
        if status != "ok":
            episode.add_note(f"{label} {status}")
            if args.auto:
                episode.set_outcome("aborted", f"{label} {status}")
                return episode

    show_snapshot(episode, args.open_images)
    result = ""
    while result not in OUTCOMES:
        result = input(f"\nOutcome? {'/'.join(OUTCOMES)}: ").strip().lower()
    episode.set_outcome(result, input("Reason / note (Enter to skip): ").strip())
    return episode


def main(args=None) -> int:
    raw_args = list(sys.argv[1:] if args is None else args)
    ros_args_index = raw_args.index("--ros-args") if "--ros-args" in raw_args else len(raw_args)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("object", help="episode label only (e.g. heart_a)")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--order", choices=["tip-push", "push-tip"], default="tip-push")
    parser.add_argument("--force-ref", type=float, default=DEFAULT_FORCE_REF_N, metavar="N",
                        help=f"press-pull initial press force (default {DEFAULT_FORCE_REF_N} N)")
    parser.add_argument("--quality", choices=["h264", "lossless"], default="h264", help="tip video quality")
    parser.add_argument("--auto", action="store_true",
                        help="don't wait between steps; stop the episode at the first failed step")
    parser.add_argument("--no-images", dest="open_images", action="store_false",
                        help="don't open each snapshot's final_selection image")
    parsed = parser.parse_args(raw_args[:ros_args_index])

    rclpy.init(args=raw_args[ros_args_index:])
    node = rclpy.create_node("run_pipeline")  # camera extrinsics into each episode, and the snapshots
    episodes = []
    try:
        for trial in range(1, parsed.trials + 1):
            if trial > 1 and input(f"\nReset the {parsed.object} for trial {trial}, then Enter (q to stop): ") == "q":
                break
            episodes.append(run_episode(node, parsed, f"trial {trial}/{parsed.trials}"))
            print("\n" + episodes[-1].summary())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(f"\n[pipeline] {len(episodes)} episode(s):")
    for episode in episodes:
        print(f"    {episode.path.name}  {(episode.load()['outcome'] or {}).get('result')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
