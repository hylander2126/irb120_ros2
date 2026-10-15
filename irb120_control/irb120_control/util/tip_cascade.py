"""Adaptive tipping: after each tip attempt, why it failed decides what to try next.

Design notes: runtime_logs/ADAPTIVE_PRESS_IDEAS.md. The first attempt is choose_tip's.
After each attempt classify() names the outcome, and next_attempt() picks the next
attempt on the snapshot taken after it:

  press-pull slipped (lost contact again at the force cap, or ball-to-pivot spread
  > NOSLIP_TOL_MM), or failed otherwise
      -> a riskier press contact (the snapshot's press_alternatives: more torque per
         pull), starting at the force the last press-pull ended on
      -> forward tip -> stop
  forward tip hit its force limit, drifted, pushed its full travel without tipping,
  slid the object instead of tipping it, was too easy (the object nearly toppled
  under a contact force too small to control), or failed otherwise
      -> press-pull (the conservative contact, then riskier ones) -> stop
  contact missed -> the same attempt once more, then as a failure
  sensor fault   -> the same attempt again
  object moved or fell over -> stop (a forward-tip slide that left it upright is
      not "moved": the slide was the push, and press-pull runs on the new snapshot)
  success        -> stop

Each tip runs at most once per cascade except for those retries, and there are at
most MAX_ATTEMPTS attempts in all. Inside one press-pull, arc_static itself steps the
press force by 1.25x on each lost contact, up to PRESS_FORCE_MAX_N.
"""

import numpy as np

from irb120_control.util.tip_choice import NOSLIP_TOL_MM, choose_tip

MAX_ATTEMPTS = 4
MOVED_MM = 10.0           # object centroid shift (in the table plane) that stops the cascade
HEIGHT_CHANGE_MM = 10.0   # change in the object's top height: left tilted, or fell over
SAME_CONTACT_M = .01      # press contacts closer than this are the same contact
PRESS_FORCE_MAX_N = 13.0  # = arc_static.ADAPTIVE_FORCE_MAX_N
# Step stops caused by the rig, not the object (arc_static abort_reason, fwd_tip failure_reason).
SENSOR_FAULTS = {"ft_stale", "stale_force", "missing_or_stale_pose", "nonfinite_wrench"}

PRESS, FORWARD = "press_pull_tip", "forward_tip"
# Outcomes that end the cascade, and those after which the same attempt runs again.
FINAL = ("success", "moved")


def attempt(kind: str, press_candidate: int = 0, force_ref: float | None = None,
            retry: bool = False) -> dict:
    """One planned attempt. press_candidate: 0 = the snapshot's press contact, K >= 1 its
    K-th press_alternatives entry. force_ref: press-pull's starting press force."""
    return dict(kind=kind, press_candidate=press_candidate, force_ref=force_ref, retry=retry)


def describe(plan: dict) -> str:
    if plan["kind"] == FORWARD:
        return "forward tip" + (" (retry)" if plan["retry"] else "")
    contact = "conservative contact" if not plan["press_candidate"] else \
        f"riskier contact #{plan['press_candidate']}"
    return (f"press-pull, {contact}, from {plan['force_ref']:.2f} N"
            + (" (retry)" if plan["retry"] else ""))


def press_contact(contacts: dict, plan: dict) -> dict:
    """The press contact `plan` would use on this snapshot ({} if there is none)."""
    if plan["press_candidate"] == 0:
        return contacts.get("press") or {}
    alternatives = contacts.get("press_alternatives") or []
    return alternatives[plan["press_candidate"] - 1] if plan["press_candidate"] <= len(alternatives) else {}


def object_change(before, after) -> dict | None:
    """How far the object moved across one attempt, from the object clouds (base_link) of
    the snapshots before and after it: centroid shift in the table plane and change in top
    height (95th percentile z), both in mm. None if either cloud is too small."""
    before, after = (np.asarray(c, dtype=float).reshape(-1, 3) for c in (before, after))
    before, after = (c[np.isfinite(c).all(axis=1)] for c in (before, after))
    if len(before) < 3 or len(after) < 3:
        return None
    shift = np.linalg.norm(np.median(after[:, :2], axis=0) - np.median(before[:, :2], axis=0))
    height = np.percentile(after[:, 2], 95) - np.percentile(before[:, 2], 95)
    return dict(shift_mm=float(shift * 1000), height_change_mm=float(height * 1000))


def classify(kind: str, results: dict | None, slip_mm: float | None = None,
             change: dict | None = None) -> tuple:
    """(outcome, why) of one attempt from its stage results. Outcomes: success, slip,
    no_contact, sensor_fault, force_limit, drift, no_tip, slid, too_easy, moved, failed.

    slid: the forward tip translated the object instead of tipping it, seen by its slide
    probe (failure 'slide') or, without one, as a full-travel push after which the object
    shifted but kept its height. Its push is the slide interaction (friction sample)."""
    upright = change is None or abs(change["height_change_mm"]) <= HEIGHT_CHANGE_MM
    if kind == FORWARD and results is not None and upright:
        if results.get("failure_reason") == "slide":
            force, travel = results.get("slide_force_n"), results.get("slide_travel_m")
            return "slid", ("the object slid instead of tipping"
                            + (f" ({force:.2f} N over {1000 * travel:.0f} mm)" if force and travel else ""))
        if results.get("reversal_reason") == "travel_limit" and change and change["shift_mm"] > MOVED_MM:
            return "slid", f"pushed the full travel and the object slid {change['shift_mm']:.0f} mm instead of tipping"
    if change is not None and (change["shift_mm"] > MOVED_MM or not upright):
        return "moved", (f"object moved {change['shift_mm']:.0f} mm, top height changed "
                         f"{change['height_change_mm']:+.0f} mm")
    if results is None:
        return "failed", "the step left no result"
    fault = results.get("abort_reason") or results.get("failure_reason")
    if fault in SENSOR_FAULTS:
        return "sensor_fault", fault
    if kind == PRESS:
        if results.get("completed"):
            if slip_mm is not None and slip_mm > NOSLIP_TOL_MM:
                return "slip", f"tipped, but the finger slipped ({slip_mm:.1f} mm > {NOSLIP_TOL_MM:.0f} mm)"
            return "success", "tipped" + ("" if results.get("balance_reached", True)
                                          else " (arc limit reached before the balance point)")
        if results.get("contact_missed"):
            return "no_contact", "no press force where the contact was expected"
        if results.get("force_cap_reached"):
            return "slip", f"lost contact at {results.get('final_force_ref_n', PRESS_FORCE_MAX_N):.2f} N, the force cap"
        return "failed", fault or f"press-pull {results.get('status', 'failed')}"
    if results.get("completed"):
        return "success", "tipped"
    reversal, failure = results.get("reversal_reason"), results.get("failure_reason")
    if reversal == "contact_not_found":
        return "no_contact", "no contact within the expected travel"
    if reversal == "too_easy":
        force = results.get("contact_force_n")
        return "too_easy", ("contact" + (f" force {force:.2f} N" if force else "") + " too small to tip on:"
                            " the object is close to toppling forward (or too light)")
    if failure == "force_limit":
        return "force_limit", "push force reached its limit"
    if failure == "height_lateral_or_orientation_drift":
        return "drift", "the finger drifted off its line"
    if reversal == "travel_limit":
        return "no_tip", "pushed the full travel without the force dropping"
    return "failed", failure or reversal or "forward tip failed"


def first_attempt(contacts: dict, mode: str = "auto", force_ref: float = 5.0) -> tuple:
    """(attempt, reason): choose_tip's tip, with press-pull on the conservative contact."""
    kind, reason = choose_tip(contacts, mode)
    return attempt(kind, 0, force_ref if kind == PRESS else None), reason


def _next_press(history: list, contacts: dict, force_ref: float):
    """The best press contact on this snapshot not already tried, or None."""
    tried = [np.asarray(h["press_point"]) for h in history
             if h["attempt"]["kind"] == PRESS and h.get("press_point") is not None]
    ended = [h["final_force"] for h in history
             if h["attempt"]["kind"] == PRESS and h["outcome"] == "slip" and h.get("final_force")]
    start = min(max([force_ref, *ended]), PRESS_FORCE_MAX_N)
    alternatives = contacts.get("press_alternatives") or []
    for k, press in [(0, contacts.get("press"))] + list(enumerate(alternatives, 1)):
        if not (press or {}).get("available"):
            continue
        point = np.asarray(press["point"], dtype=float)
        if any(np.linalg.norm(point - other) < SAME_CONTACT_M for other in tried):
            continue
        return attempt(PRESS, k, start)
    return None


def next_attempt(history: list, contacts: dict, mode: str = "auto", force_ref: float = 5.0) -> tuple:
    """(attempt or None, reason) after the attempts in `history`, on this snapshot's contacts.

    history: one dict per attempt run, oldest first: attempt (as planned), outcome and why
    (classify), press_point (the press contact used, base_link) and final_force (press-pull's
    last press force)."""
    last = history[-1]
    outcome, plan = last["outcome"], last["attempt"]
    if outcome in FINAL:
        return None, last["why"]
    if len(history) >= MAX_ATTEMPTS:
        return None, f"{MAX_ATTEMPTS} attempts made; last: {last['why']}"
    if outcome == "sensor_fault":
        return dict(plan, retry=True), f"sensor fault ({last['why']}): same attempt again"
    if outcome == "no_contact" and not plan["retry"]:
        return dict(plan, retry=True), f"{last['why']}: same attempt once more on the new snapshot"

    press_allowed, forward_allowed = mode in ("auto", "press"), mode in ("auto", "forward")
    why = f"{plan['kind'].replace('_', ' ')} {outcome} ({last['why']})"
    press = _next_press(history, contacts, force_ref) if press_allowed else None
    forward_tried = any(h["attempt"]["kind"] == FORWARD for h in history)
    forward_ready = (forward_allowed and not forward_tried
                     and (contacts.get("forward_tip") or {}).get("available"))
    if plan["kind"] == PRESS:
        if press is not None:
            return press, f"{why}: {describe(press)}"
        if forward_ready:
            return attempt(FORWARD), f"{why}, no untried press contact left: forward tip"
        return None, f"{why}; no untried press contact" + ("" if forward_allowed and not forward_tried
                                                        else ", forward tip not allowed or already tried")
    if press is not None:
        return press, f"{why}: {describe(press)}"
    return None, f"{why}; no untried press contact"
