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

A step stopped by a sensor fault (stale F/T or pose; SENSOR_FAULTS) is not
treated as a tip failure. A single interactive trial asks: retry the step,
fall back to the other tip, or stop. With --trials > 1 or
--auto the trial is ended as aborted ("sensor fault: ...") and the next one starts.
Each step's result is also written to episode.json's `events`.

Ctrl-C during a motion step is passed to that step, and this script waits for
it to finish its own cleanup (stop, home, save logs) before stopping.

Which tip: forward tip first when its pivot (the far edge, fitted in cam3) is a
straight edge, press-pull first when it is curved (a point contact) or unknown.
A failed forward tip is followed by a press-pull; a press-pull that fails, or
whose ball leaves its circle about the pivot (the finger slipped on the object,
as on the monitor), by a forward tip -- each on the snapshot taken after it,
and at most one of each. --tip press|forward forces one. See choose_tip().

Usage:
    ros2 run irb120_control run_pipeline heart_a
    ros2 run irb120_control run_pipeline heart_a --trials 10 --force-ref 7
    ros2 run irb120_control run_pipeline heart_a --order push-tip --auto
    ros2 run irb120_control run_pipeline monitor --tip forward
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import rclpy

from irb120_control.push import PUSH_DISTANCE, PUSH_STANDOFF
from irb120_control.util.episode import OUTCOMES, Episode
from irb120_control.util.perception_snapshot import take_snapshot
from irb120_control.util.push_geometry import push_approach
from irb120_control.util.tip_choice import NOSLIP_TOL_MM, TIP_MODES, choose_tip, forward_pivot_shape  # noqa: F401
from irb120_perception.press_targets import PRESS_STANDOFF

DEFAULT_FORCE_REF_N = 5.0
PUSH_MAX_NORMAL_ANGLE_DEG = 20.0  # push.py refuses contacts whose normal is further than this from -X
ARC_STATE_IDS = (3, 4)  # arc_static STATE_IDS ARC, UNARC
SQUASH_STATE_ID = 1

# Motion step kind -> (contact it needs, description for the prompt)
STEPS = {
    "press_pull_tip": ("press", "press-pull tip"),
    "forward_tip": ("forward_tip", "forward tip"),
    "push": ("planar_push", "push"),
}
# Step stops caused by the rig, not the object (arc_static abort_reason, fwd_tip failure_reason).
# They say nothing about the object, so they never trigger the forward-tip fallback.
SENSOR_FAULTS = {"ft_stale", "stale_force", "missing_or_stale_pose", "nonfinite_wrench"}


def press_pull_slip_mm(stage_dir: Path) -> float | None:
    """Peak-to-peak |ball - commanded pivot| over the last attempt's ARC/UNARC, in mm.

    Without slip the ball stays on a circle about the pivot (it rolls, which barely
    changes that distance); the monitor's slipping finger gave 8.4 mm. None when the
    stage has no log with an arc and a commanded pivot.
    """
    logs = sorted(Path(stage_dir).glob("arc_static_*.npz"))
    if not logs:
        return None
    data = np.load(logs[-1])
    if "commanded_pivot" not in data or not len(data["controller_state_id"]):
        return None
    state = data["controller_state_id"].astype(int)
    squash = np.flatnonzero(state == SQUASH_STATE_ID)
    arc = np.isin(state, ARC_STATE_IDS)
    arc[:squash[-1] if len(squash) else 0] = False  # adaptive retries share one log
    if not arc.any():
        return None
    ball = np.column_stack([data["x"], data["y"], data["z"]])[arc]
    radius = np.linalg.norm(ball - np.asarray(data["commanded_pivot"], dtype=float), axis=1)
    return float(np.ptp(radius) * 1000)


def step_command(kind: str, args) -> list:
    if kind == "press_pull_tip":
        return ["arc_static", args.object, "--force-ref", str(args.force_ref), "--quality", args.quality]
    if kind == "forward_tip":
        return ["fwd_tip", args.object, "--quality", args.quality]
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
    if record.get("tip_choice"):
        print(f"    perception's tip choice: {record['tip_choice']['tip']} ({record['tip_choice']['reason']})")
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
              f" [{forward['geometry'].get('pivot_source', '?')}]"
              f"   height above pivot {1000 * (forward['point'][2] - pivot[2]):.0f} mm")
        shape, why = forward_pivot_shape(contacts)
        print(f"               pivot shape {shape.upper()}" + (f" ({why})" if why else ""))
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


def sensor_fault(stage: dict | None) -> str | None:
    results = (stage or {}).get("results", {})
    reason = results.get("abort_reason") or results.get("failure_reason")
    return reason if reason in SENSOR_FAULTS else None


def fault_gate(label: str, fault: str, fallback_label: str | None) -> str:
    """After a sensor fault: 'retry', 'fallback' (to `fallback_label`, if given) or 'q'."""
    can_fall_back = fallback_label is not None
    fallback = f", f = {fallback_label} instead" if can_fall_back else ""
    while True:
        try:
            answer = input(f"[pipeline] Enter = retry {label}{fallback}, q = stop episode: ").strip().lower()
        except EOFError:
            return "q"
        if answer == "":
            return "retry"
        if answer == "f" and can_fall_back:
            return "fallback"
        if answer == "q":
            return answer


def gate(label: str, ready: bool, auto: bool, skippable: bool = False) -> str:
    """'go', 'r' (re-perceive), 's' (skip this step; only if skippable) or 'q' (stop the episode)."""
    if auto:
        return "go" if ready else "q"
    skip = ", s = skip it" if skippable else ""
    prompt = (f"\n[pipeline] Next: {label}. Enter = go, r = re-perceive{skip}, q = stop episode: " if ready else
              f"\n[pipeline] Can't run {label} on this snapshot. r = re-perceive{skip}, q = stop episode: ")
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return "q"
        if answer == "" and ready:
            return "go"
        if answer in ("r", "q") or (answer == "s" and skippable):
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
        tried = {}  # tip kind -> dict(status, slip_mm)
        while True:  # one step; a tip phase may add the other tip after the first
            outcome = run_phase_step(node, episode, args, phase, tried)
            if outcome == "retry":
                continue  # same choice again, on the snapshot taken after the faulted step
            if outcome is None or isinstance(outcome, Episode):
                break
            tried[outcome.pop("kind")] = outcome
        if isinstance(outcome, Episode):
            return episode

    show_snapshot(episode, args.open_images)
    result = ""
    while result not in OUTCOMES:
        result = input(f"\nOutcome? {'/'.join(OUTCOMES)}: ").strip().lower()
    episode.set_outcome(result, input("Reason / note (Enter to skip): ").strip())
    return episode


def run_phase_step(node, episode: Episode, args, phase: str, tried: dict):
    """Run the phase's next step. `tried`: the tips already run in this phase, kind ->
    dict(status, slip_mm). Returns the episode if it ended (outcome set), a tip's
    result dict(kind, status, slip_mm) if another tip may follow it, "retry" to run
    the same step again after a sensor fault, else None (phase done)."""
    while True:
        contacts = show_snapshot(episode, args.open_images)
        if phase == "tip":
            kind, reason = choose_tip(contacts or {}, args.tip, tried.get("press_pull_tip"),
                                      tried.get("forward_tip"))
            if kind is None:
                if tried:
                    episode.add_event(f"tip choice: {reason}")
                    if args.auto and all(r["status"] != "ok" for r in tried.values()):
                        episode.set_outcome("aborted", reason)
                        return episode
                return None
            print(f"\n[pipeline] tip choice: {STEPS[kind][1]} ({reason})")
        else:
            kind, reason = "push", None
        needed, label = STEPS[kind]
        ready = contacts is not None and (contacts.get(needed) or {}).get("available", False)
        answer = gate(label, ready, args.auto, skippable=bool(tried))
        if answer == "s":
            episode.add_event(f"tip choice: {reason}; skipped by operator")
            return None
        if answer == "r":
            take_snapshot(node, episode, label="retake")
            continue
        break
    if answer == "q":
        episode.set_outcome("aborted", f"stopped before {label}"
                            + ("" if ready else f": no usable {needed} contact"))
        return episode
    if reason is not None:
        episode.add_event(f"tip choice: {label} ({reason})")

    stage, snapshot_ok = run_step(episode, kind, step_command(kind, args))
    status = stage["status"] if stage else "missing"
    results = {k: v for k, v in (stage or {}).get("results", {}).items() if not isinstance(v, (dict, list))}
    slip = press_pull_slip_mm(episode.path / stage["dir"]) if kind == "press_pull_tip" and stage else None
    print(f"\n[pipeline] {label}: {status}  {results}"
          + ("" if slip is None else f"  slip {slip:.1f} mm (tolerance {NOSLIP_TOL_MM:.0f} mm)"))
    if slip is not None:
        episode.add_event(f"{label}: ball-to-pivot spread {slip:.1f} mm over the arc")
    fault = sensor_fault(stage)
    if fault:
        print(f"\n[pipeline] !!! {label} stopped by a SENSOR FAULT ({fault}), not by the object."
              f"\n[pipeline] !!! Says nothing about this object; see {stage['dir']}.")
        episode.add_event(f"{label}: sensor fault {fault}")
    if not snapshot_ok:
        episode.set_outcome("aborted", (f"sensor fault: {fault} during {label}" if fault else f"{label} {status}")
                            + ", no usable snapshot after it")
        return episode
    if fault:
        if args.auto or args.trials > 1:
            print("[pipeline] !!! Ending this trial as aborted and moving on.")
            episode.set_outcome("aborted", f"sensor fault: {fault} during {label}")
            return episode
        other = "forward_tip" if kind == "press_pull_tip" else "press_pull_tip"
        answer = fault_gate(label, fault, STEPS[other][1]
                            if kind in TIP_MODES.values() and other not in tried else None)
        if answer == "q":
            episode.set_outcome("aborted", f"sensor fault: {fault} during {label}; stopped by operator")
            return episode
        if answer == "retry":
            episode.add_event(f"{label}: retrying after sensor fault")
            return "retry"
        episode.add_event(f"{label}: falling back to {STEPS[other][1]} after sensor fault")
    if status != "ok":
        episode.add_note(f"{label} {status}")
    if kind in TIP_MODES.values():
        return dict(kind=kind, status=status, slip_mm=slip)  # choose_tip decides on the other tip
    if status != "ok" and args.auto:
        episode.set_outcome("aborted", f"{label} {status}")
        return episode
    return None


def main(args=None) -> int:
    raw_args = list(sys.argv[1:] if args is None else args)
    ros_args_index = raw_args.index("--ros-args") if "--ros-args" in raw_args else len(raw_args)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("object", help="episode label only (e.g. heart_a)")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--order", choices=["tip-push", "push-tip"], default="tip-push")
    parser.add_argument("--force-ref", type=float, default=DEFAULT_FORCE_REF_N, metavar="N",
                        help=f"press-pull initial press force (default {DEFAULT_FORCE_REF_N} N)")
    parser.add_argument("--tip", choices=["auto", *TIP_MODES], default="auto",
                        help="auto: forward tip first if its pivot is a straight edge, else press-pull; "
                             "then the other tip if the first failed (or press-pull slipped) (default)")
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
        outcome = episode.load()["outcome"] or {}
        print(f"    {episode.path.name}  {outcome.get('result')}"
              + (f"  ({outcome['reason']})" if outcome.get("result") == "aborted" and outcome.get("reason") else ""))
    faulted = sum(str((e.load()["outcome"] or {}).get("reason", "")).startswith("sensor fault") for e in episodes)
    if faulted:
        print(f"[pipeline] {faulted} of {len(episodes)} episode(s) lost to sensor faults")
    return 0


if __name__ == "__main__":
    sys.exit(main())
