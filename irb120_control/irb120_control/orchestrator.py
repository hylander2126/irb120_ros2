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
stops the episode; before a tip, 'n' cycles through the tip options (tip_options:
the cascade's pick, the other tip, the press backups), opening each one's image;
running one that is not the cascade's pick asks why, logged as an event.
--auto skips these waits and stops at the first failed step.

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
snapshot taken after the one before, up to MAX_ATTEMPTS. Every attempt waits at
the gate in step mode. --tip press|forward keeps the cascade to that tip.
A forward tip that slid the object instead of tipping it (its slide probe, or a
full-travel push that shifted the upright object) is the episode's slide
interaction: press-pull follows, and the push phase is skipped (--auto) or
offered as skippable.

Usage:
    ros2 run irb120_control run_pipeline heart_a
    ros2 run irb120_control run_pipeline heart_a --trials 10 --force-ref 7
    ros2 run irb120_control run_pipeline heart_a --order push-tip --auto
    ros2 run irb120_control run_pipeline monitor --tip forward
    ros2 run irb120_control run_pipeline heart_a --trials 5 --official
"""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import rclpy

from irb120_control.push import PUSH_CONTACT_MARGIN, PUSH_DISTANCE, PUSH_STANDOFF
from irb120_control.util.episode import OUTCOMES, Episode
from irb120_control.util.official_run import OfficialRunError, prepare as prepare_official
from irb120_control.util.registration_check import check_folder as check_registration, describe as describe_registration
from irb120_control.util.perception_snapshot import latest_object_cloud, take_snapshot
from irb120_control.util.push_geometry import push_approach, horizontal_push_direction
from irb120_control.util.tip_cascade import (
    FORWARD, MAX_ATTEMPTS, PRESS, SENSOR_FAULTS, attempt, classify, describe, first_attempt,
    next_attempt, object_change, press_contact,
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
                "--press-candidate", str(plan["press_candidate"]), "--quality", args.quality,
                *(["--press-backup", str(plan["press_backup"])] if plan.get("press_backup") else [])]
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


_viewer = {"path": None, "proc": None}


def open_image(image: Path) -> None:
    """Show `image`, closing the viewer this run opened before, so only the latest
    contacts are on screen. The viewer stays in xdg-open's new session (process group)."""
    if _viewer["path"] == str(image) and _viewer["proc"] is not None:
        return
    if _viewer["proc"] is not None:
        try:
            os.killpg(_viewer["proc"].pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    _viewer["path"] = str(image)
    _viewer["proc"] = subprocess.Popen(["xdg-open", str(image)], stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, start_new_session=True)


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
        print(f"  press        contact {_mm(press['point'])}   pivot {_mm(pivot)} [{g.get('pivot_source', '?')}]"
              + ("   APPROACH BLOCKED (pre-press inside the object box): cascade starts riskier"
                 if press.get("approachable") is False else ""))
        print(f"               pre-squash ball {_mm(ball + [0, 0, PRESS_STANDOFF])}"
              f"   arc radius {1000 * radius:.0f} mm   pull dir {np.round(g['direction'], 2).tolist()}")
        for alternative in contacts.get("press_alternatives") or []:
            print(f"               riskier #{alternative['rank']}: contact {_mm(alternative['point'])}"
                  f"   +{1000 * alternative['torque_gain']:.1f} mm torque/pull"
                  f"   {1000 * alternative['lateral_offset_m']:+.0f} mm sideways")
        for backup in contacts.get("press_backups") or []:
            print(f"               backup B{backup['rank']}: contact {_mm(backup['point'])}"
                  f"   {1000 * backup['along_edge_offset']:+.0f} mm from pivot center"
                  f"   vs chosen {1000 * press.get('along_edge_offset', float('nan')):+.0f} mm")
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
            pre, travel = push_approach(push["ball_center"], cloud, PUSH_STANDOFF,
                                         PUSH_CONTACT_MARGIN + PUSH_DISTANCE, direction)
            print(f"               pre-push ball {_mm(pre)}   travel {1000 * travel:.0f} mm"
                  f" along {np.round(direction, 3).tolist()} (force check before full push)")
        except (ValueError, OSError, KeyError) as exc:
            print(f"               pre-push: {exc}")
    else:
        print(f"  planar_push  NONE: {push.get('reason')}")

    image = next((folder / n for n in ("contacts.png", "final_selection.png", "final_selection.jpg", "cam1_overlay.png") if (folder / n).exists()), None)
    if image is not None:
        print(f"    image: {image}")
        if open_images:
            open_image(image)
    return contacts


MAX_TIP_OPTIONS = 4  # the cascade's pick plus up to 3 more, cycled with 'n' at the gate


def tip_options(plan: dict, contacts: dict, args) -> list:
    """The plans 'n' cycles through at the gate: the cascade's pick first, then the other
    tip (forward tip, or press-pull on the conservative contact), then each press backup
    (snapshot's press_backups); only tips --tip allows, at most MAX_TIP_OPTIONS."""
    force = plan["force_ref"] if plan["kind"] == PRESS else args.force_ref
    others = []
    if plan["kind"] == PRESS and args.tip != "press" and (contacts.get("forward_tip") or {}).get("available"):
        others.append(attempt(FORWARD))
    if args.tip != "forward" and (contacts.get("press") or {}).get("available"):
        if plan["kind"] != PRESS:
            others.append(attempt(PRESS, 0, force))
        others += [attempt(PRESS, 0, force, press_backup=b["rank"]) for b in contacts.get("press_backups") or []]
    key = lambda p: (p["kind"], p["press_candidate"], p.get("press_backup", 0))  # noqa: E731
    return ([plan] + [o for o in others if key(o) != key(plan)])[:MAX_TIP_OPTIONS]


def option_image(episode: Episode, plan: dict) -> Path | None:
    """The latest snapshot's image of `plan`'s contact: contacts_B<k>.png for a backup,
    contacts_A<k>.png for a riskier alternative, else contacts.png."""
    stage = _latest_perceive(episode)
    if stage is None:
        return None
    folder = episode.path / stage["dir"]
    if plan.get("press_backup"):
        names = (f"contacts_B{plan['press_backup']}.png", f"final_selection_B{plan['press_backup']}.jpg")
    elif plan["kind"] == PRESS and plan.get("press_candidate"):
        names = (f"contacts_A{plan['press_candidate']}.png",)
    else:
        names = ("contacts.png", "final_selection.jpg")
    return next((folder / n for n in names if (folder / n).exists()), None)


def sensor_fault(stage: dict | None) -> str | None:
    results = (stage or {}).get("results", {})
    reason = results.get("abort_reason") or results.get("failure_reason")
    return reason if reason in SENSOR_FAULTS else None


def gate(label: str, ready: bool, auto: bool, skippable: bool = False, can_next: bool = False) -> str:
    """'go', 'r' (re-perceive), 's' (skip this step; only if skippable), 'n' (next-best
    contact; only if can_next) or 'q' (stop the episode)."""
    if auto:
        return "go" if ready else "q"
    skip = (", s = skip it" if skippable else "") + (", n = next option" if can_next else "")
    prompt = (f"\n[pipeline] Next: {label}. Enter = go, r = re-perceive{skip}, q = stop episode: " if ready else
              f"\n[pipeline] Can't run {label} on this snapshot. r = re-perceive{skip}, q = stop episode: ")
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return "q"
        if answer == "" and ready:
            return "go"
        if answer in ("r", "q") or (answer == "s" and skippable) or (answer == "n" and can_next):
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
    extra = {"official": args.official_info} if args.official_info else {}
    episode = Episode.create(args.object, node, note, allow_unconfigured_object=True, extra=extra)
    episode.add_event(f"pipeline: order {args.order}, force_ref {args.force_ref} N, "
                      f"{'auto' if args.auto else 'step'} mode"
                      + (", OFFICIAL" if args.official_info else ""))
    snapshot = take_snapshot(node, episode, label="initial")
    if not snapshot["ok"]:
        episode.set_outcome("aborted", snapshot["reason"])
        return episode
    if not registration_ok(episode, args):
        episode.set_outcome("aborted", "camera agreement check failed before any motion")
        return episode

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


def registration_ok(episode: Episode, args) -> bool:
    """Camera agreement on the initial snapshot (util/registration_check.py). Recorded in
    episode.json. A failure only stops an official episode, and only if the operator says so."""
    stage = _latest_perceive(episode)
    if stage is None:
        return True
    result = check_registration(episode.path / stage["dir"])
    episode.add_event(describe_registration(result))
    print(f"\n[pipeline] {describe_registration(result)}")
    if result["status"] != "fail" or not args.official_info:
        return True
    if args.auto:
        return False
    return input("[pipeline] Cameras disagree. Continue this official episode anyway? [y/N] ").strip().lower() == "y"


def run_tip_cascade(node, episode: Episode, args, slides: list | None = None):
    """The tip phase: attempts chosen by util/tip_cascade.py, each on the snapshot taken
    after the one before, until one tips the object or none is left. In step mode
    every attempt waits at the gate. A forward tip that slid the object is recorded
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
            if history[-1]["outcome"] == "stopped":
                episode.set_outcome("aborted", reason)
                return episode
            if history[-1]["outcome"] != "success":
                episode.add_note(f"tip failed: {reason}")
                if args.auto:
                    episode.set_outcome("aborted", reason)
                    return episode
            return None
        options, chosen, index = tip_options(plan, contacts, args), describe(plan), 0
        if args.open_images and plan["kind"] == PRESS and (plan.get("press_candidate") or plan.get("press_backup")):
            image = option_image(episode, plan)  # the cascade's pick is not the contacts.png press
            if image is not None:
                open_image(image)
        print(f"\n[pipeline] tip attempt {len(history) + 1}/{MAX_ATTEMPTS}: {chosen} ({reason})")
        while True:
            plan = options[index]
            kind, label = plan["kind"], describe(plan)
            contact = press_contact(contacts, plan) if kind == PRESS else contacts.get("forward_tip") or {}
            ready = bool(contact.get("available"))
            if history and not ready and args.auto:  # a retry whose contact is gone from the new snapshot
                break
            answer = gate(label, ready, args.auto, can_next=len(options) > 1)
            if answer != "n":
                break
            index = (index + 1) % len(options)
            plan = options[index]
            contact = press_contact(contacts, plan) if plan["kind"] == PRESS else contacts.get("forward_tip") or {}
            print(f"\n[pipeline] option {index + 1}/{len(options)}: {describe(plan)}"
                  + (f", contact {_mm(contact['point'])}" if contact.get("point") else "")
                  + (f", {1000 * contact['along_edge_offset']:+.0f} mm from pivot center"
                     if contact.get("along_edge_offset") is not None else ""))
            image = option_image(episode, plan) if args.open_images else None
            if image is not None:
                open_image(image)
        if history and not ready and args.auto:
            history.append(dict(attempt=plan, outcome="failed", stage=None, press_point=None,
                                final_force=None, why=f"no usable {STEPS[kind][0]} contact on this snapshot"))
            continue
        if answer == "r":
            take_snapshot(node, episode, label="retake")
            continue
        if answer == "q":
            episode.set_outcome("aborted", f"stopped before {label}"
                                + ("" if ready else f": no usable {STEPS[kind][0]} contact"))
            return episode
        if index:
            try:
                why = input(f"Why {label} over the {chosen}? (Enter to skip): ").strip()
            except EOFError:
                why = ""
            episode.add_event(f"operator override: {label} instead of {chosen}; "
                              f"contact {np.round(contact['point'], 4).tolist()}; reason: {why or '-'}")
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
    parser.add_argument("--official", action="store_true",
                        help="record official data: requires a clean commit, PROTOCOL.md and MANIFEST.json "
                             "in the data root; episodes go to <data root>/raw (util/official_run.py)")
    parser.add_argument("--data-root", default=None,
                        help="official data root (default $IRB120_OFFICIAL_ROOT or ~/Documents/irb120_data/official_v1)")
    parsed = parser.parse_args(raw_args[:ros_args_index])
    parsed.official_info = None
    if parsed.official:
        try:
            parsed.official_info = prepare_official(parsed.data_root)
        except OfficialRunError as exc:
            print(f"[pipeline] {exc}")
            return 1
        print(f"[pipeline] OFFICIAL run -> {parsed.official_info['data_root']}/raw "
              f"(tag {parsed.official_info['git_tag'] or 'none'})")

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
