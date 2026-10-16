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
treated as an object failure: the tip cascade runs it again, and a push ends the
episode as aborted if no snapshot follows it. Each step's result is also written
to episode.json's `events`.

Ctrl-C during a motion step is passed to that step, and this script waits for
it to finish its own cleanup (stop, home, save logs) before stopping.

Which tip: forward tip first when its pivot (the far edge, fitted in cam3) is a
straight edge, press-pull first when it is curved (a point contact) or unknown;
press-pull starts on its conservative contact (directly above the pivot). If that
fails, the tip cascade (util/tip_cascade.py) picks the next attempt from why it
failed -- riskier press contacts, more press force, the other tip -- each on the
snapshot taken after the one before, up to MAX_ATTEMPTS. Only the first attempt
waits at the gate in step mode. --tip press|forward keeps the cascade to that tip.
A forward tip that slid the object instead of tipping it (its slide probe, or a
full-travel push that shifted the upright object) is the episode's slide
interaction: press-pull follows, and the push phase is skipped (--auto) or
offered as skippable.

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
from irb120_control.util.perception_snapshot import latest_object_cloud, take_snapshot
from irb120_control.util.push_geometry import push_approach, horizontal_push_direction
from irb120_control.util.tip_cascade import (
    MAX_ATTEMPTS, PRESS, SENSOR_FAULTS, classify, describe, first_attempt, next_attempt,
    object_change, press_contact,
)
from irb120_control.util.tip_choice import TIP_MODES, forward_pivot_shape
from irb120_perception.press_targets import PRESS_STANDOFF

DEFAULT_FORCE_REF_N = 5.0
PUSH_MAX_NORMAL_ANGLE_DEG = 20.0  # push.py refuses contacts whose normal is further than this from -X
ARC_STATE_IDS = (3, 4)  # arc_static STATE_IDS ARC, UNARC
SQUASH_STATE_ID = 1

# Stage kind the motion script records, where it differs from the step kind (fwd_arc).
STAGE_KINDS = {"forward_tip": "forward_arc"}

# Motion step kind -> (contact it needs, description for the prompt)
STEPS = {
    "press_pull_tip": ("press", "press-pull tip"),
    "forward_tip": ("forward_tip", "forward tip"),
    "push": ("planar_push", "push"),
}


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


def step_command(kind: str, args, plan: dict | None = None) -> list:
    """The motion script for a step; `plan` (a tip_cascade attempt) for the tips."""
    if kind == "press_pull_tip":
        return ["arc_static", args.object, "--force-ref", f"{plan['force_ref']:.3f}",
                "--press-candidate", str(plan["press_candidate"]), "--quality", args.quality]
    if kind == "forward_tip":   # the arc version: it rubs the face far less than fwd_tip's straight push
        return ["fwd_arc", args.object, "--quality", args.quality]
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
        for alternative in contacts.get("press_alternatives") or []:
            print(f"               riskier #{alternative['rank']}: contact {_mm(alternative['point'])}"
                  f"   +{1000 * alternative['torque_gain']:.1f} mm torque/pull"
                  f"   {1000 * alternative['lateral_offset_m']:+.0f} mm sideways")
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
            direction = horizontal_push_direction(push["normal"])
            pre, travel = push_approach(push["ball_center"], cloud, PUSH_STANDOFF, PUSH_DISTANCE, direction)
            print(f"               pre-push ball {_mm(pre)}   travel {1000 * travel:.0f} mm"
                  f" along {np.round(direction, 3).tolist()} (force check before full push)")
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
    stage = next((s for s in reversed(new) if s["kind"] == STAGE_KINDS.get(kind, kind)), None)
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

    slides = []  # forward tips that slid the object: each is a slide interaction already
    for phase in args.order.split("-"):
        ended = (run_tip_cascade(node, episode, args, slides) if phase == "tip"
                 else run_push(node, episode, args, slides[-1] if slides else None))
        if ended is not None:
            return episode

    show_snapshot(episode, args.open_images)
    result = ""
    while result not in OUTCOMES:
        result = input(f"\nOutcome? {'/'.join(OUTCOMES)}: ").strip().lower()
    episode.set_outcome(result, input("Reason / note (Enter to skip): ").strip())
    return episode


def run_tip_cascade(node, episode: Episode, args, slides: list | None = None):
    """The tip phase: attempts chosen by util/tip_cascade.py, each on the snapshot taken
    after the one before, until one tips the object or none is left. In step mode only
    the first attempt waits at the gate. A forward tip that slid the object is recorded
    as the episode's slide interaction (its stage dir appended to `slides`). Returns the
    episode if it ended, else None."""
    history = []
    while True:
        contacts = show_snapshot(episode, args.open_images) or {}
        plan, reason = (next_attempt(history, contacts, args.tip, args.force_ref) if history
                        else first_attempt(contacts, args.tip, args.force_ref))
        if plan is None:
            print(f"\n[pipeline] tip cascade done after {len(history)} attempt(s): {reason}")
            episode.add_event(f"tip cascade done: {reason}")
            if history[-1]["outcome"] == "moved":
                print("[pipeline] !!! The object moved or fell: check it before going on.")
            if history[-1]["outcome"] != "success":
                episode.add_note(f"tip failed: {reason}")
                if args.auto:
                    episode.set_outcome("aborted", reason)
                    return episode
            return None
        kind, label = plan["kind"], describe(plan)
        contact = press_contact(contacts, plan) if kind == PRESS else contacts.get("forward_tip") or {}
        ready = bool(contact.get("available"))
        print(f"\n[pipeline] tip attempt {len(history) + 1}/{MAX_ATTEMPTS}: {label} ({reason})")
        if not history:
            answer = gate(label, ready, args.auto)
            if answer == "r":
                take_snapshot(node, episode, label="retake")
                continue
            if answer == "q":
                episode.set_outcome("aborted", f"stopped before {label}"
                                    + ("" if ready else f": no usable {STEPS[kind][0]} contact"))
                return episode
        elif not ready:  # a retry whose contact is gone from the new snapshot
            history.append(dict(attempt=plan, outcome="failed", stage=None, press_point=None,
                                final_force=None, why=f"no usable {STEPS[kind][0]} contact on this snapshot"))
            continue
        episode.add_event(f"tip attempt {len(history) + 1}: {label} ({reason})")

        before = latest_object_cloud(episode)
        stage, snapshot_ok = run_step(episode, kind, step_command(kind, args, plan))
        results = None if stage is None else dict(stage.get("results", {}), status=stage["status"])
        slip = press_pull_slip_mm(episode.path / stage["dir"]) if kind == PRESS and stage else None
        change = object_change(before, latest_object_cloud(episode)) if snapshot_ok else None
        outcome, why = classify(kind, results, slip, change)
        history.append(dict(attempt=plan, outcome=outcome, why=why, stage=stage and stage["dir"],
                            press_point=contact.get("point") if kind == PRESS else None,
                            final_force=(results or {}).get("final_force_ref_n")))
        short = {k: v for k, v in (results or {}).items() if not isinstance(v, (dict, list))}
        print(f"\n[pipeline] {label}: {outcome.upper()} -- {why}\n           {short}"
              + ("" if change is None else
                 f"\n           object shift {change['shift_mm']:.1f} mm, "
                 f"top height change {change['height_change_mm']:+.1f} mm"))
        episode.add_event(f"tip attempt {len(history)}: {outcome} ({why})")
        if outcome == "slid" and stage:
            if slides is not None:
                slides.append(stage["dir"])
            episode.add_event(f"slide interaction: {stage['dir']} (the forward tip slid the object; "
                              f"slide_force_n {(results or {}).get('slide_force_n')})")
        if not snapshot_ok:
            episode.set_outcome("aborted", f"{label}: {outcome} ({why}), no usable snapshot after it")
            return episode


def run_push(node, episode: Episode, args, slide: str | None = None):
    """The push phase. `slide`: the stage of a forward tip that already slid the object;
    then the push is skipped in --auto mode and offered as skippable in step mode.
    Returns the episode if it ended, else None."""
    if slide and args.auto:
        print(f"\n[pipeline] push skipped: the forward tip in {slide} already slid the object")
        episode.add_event(f"push: skipped, {slide} is the slide interaction")
        return None
    while True:
        contacts = show_snapshot(episode, args.open_images)
        ready = contacts is not None and (contacts.get("planar_push") or {}).get("available", False)
        if slide:
            print(f"\n[pipeline] the forward tip in {slide} already slid the object: s skips the push")
        answer = gate("push", ready, args.auto, skippable=bool(slide))
        if answer != "r":
            break
        take_snapshot(node, episode, label="retake")
    if answer == "s":
        episode.add_event(f"push: skipped by the operator, {slide} is the slide interaction")
        return None
    if answer == "q":
        episode.set_outcome("aborted", "stopped before push" + ("" if ready else ": no usable planar_push contact"))
        return episode
    stage, snapshot_ok = run_step(episode, "push", step_command("push", args))
    status = stage["status"] if stage else "missing"
    results = {k: v for k, v in (stage or {}).get("results", {}).items() if not isinstance(v, (dict, list))}
    print(f"\n[pipeline] push: {status}  {results}")
    fault = sensor_fault(stage)
    if fault:
        print(f"\n[pipeline] !!! push stopped by a SENSOR FAULT ({fault}), not by the object.")
        episode.add_event(f"push: sensor fault {fault}")
    if not snapshot_ok:
        episode.set_outcome("aborted", (f"sensor fault: {fault} during push" if fault else f"push {status}")
                            + ", no usable snapshot after it")
        return episode
    if status != "ok":
        episode.add_note(f"push {status}")
        if args.auto:
            episode.set_outcome("aborted", f"push {status}")
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
                             "after a failure, riskier press contacts and the other tip "
                             "(util/tip_cascade.py) (default). press|forward: that tip only")
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
