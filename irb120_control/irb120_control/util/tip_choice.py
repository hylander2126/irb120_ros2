"""Which tip to run on a snapshot (press-pull or forward tip), and why.

Shared by the perception snapshot, which records its first-tip choice in
snapshot.json, and the orchestrator, which also decides on a second tip after
the first one ran.
"""

NOSLIP_TOL_MM = 4.0  # = estimate_params.NOSLIP_TOL_MM: spread of |ball - pivot| over the arc
TIP_MODES = {"press": "press_pull_tip", "forward": "forward_tip"}
TIP_CONTACTS = {"press_pull_tip": "press", "forward_tip": "forward_tip"}  # tip -> contact it needs


def _available(contacts: dict, kind: str) -> bool:
    return bool((contacts.get(TIP_CONTACTS[kind]) or {}).get("available"))


def forward_pivot_shape(contacts: dict) -> tuple:
    """('straight' | 'curved' | 'unknown', reason) of the forward-tip pivot.

    From the snapshot's cam3 SAM boundary fit (forward_pivot_review), written into
    contacts.json as forward_tip.pivot_shape. 'unknown' when that fit was not
    possible (no cam3 mask, noisy table, no support run) or the snapshot predates it.
    """
    shape = (contacts.get("forward_tip") or {}).get("pivot_shape") or {}
    if shape.get("shape") in ("straight", "curved"):
        return shape["shape"], shape.get("reason") or ""
    return "unknown", shape.get("reason") or "no cam3 pivot fit in this snapshot"


def choose_tip(contacts: dict, mode: str = "auto", press_result: dict | None = None,
               forward_result: dict | None = None) -> tuple:
    """(step kind or None, reason) for the next tip on this snapshot.

    auto, first tip: decided by the forward pivot's geometry alone
    (forward_pivot_shape). A straight edge suits forward tipping: the fixed-pivot
    push about an extended edge. A curved pivot is a point contact, about which
    the object can roll or yaw under a forward push; press-pull is preferred
    there, and when the shape is unknown (press-pull has the estimator and its
    pivot is the near edge cam1/cam2 see). Either way, the other tip runs when the
    preferred one has no contact.

    After a tip (`press_result` / `forward_result`, dict(status, slip_mm)): a
    press-pull that failed or slipped, or a forward tip that failed, is followed
    by the other tip, unless that has already run; None means no further tip.
    Whether press-pull will slip depends on the mass, which is unknown until it
    has run, so slip is detected afterwards rather than predicted.
    """
    if mode != "auto":
        kind = TIP_MODES[mode]
        return (kind, f"--tip {mode}") if press_result is None and forward_result is None else \
            (None, f"--tip {mode}: one tip only")
    if press_result is None and forward_result is None:
        shape, why = forward_pivot_shape(contacts)
        why = f"forward pivot {shape}" + (f" ({why})" if why else "")
        if shape == "straight":
            if _available(contacts, "forward_tip"):
                return "forward_tip", why
            return "press_pull_tip", f"{why}, but no forward-tip contact: {(contacts.get('forward_tip') or {}).get('reason')}"
        if _available(contacts, "press_pull_tip"):
            return "press_pull_tip", f"{why}: press-pull preferred"
        if _available(contacts, "forward_tip"):
            return "forward_tip", f"{why}, but no press contact: {(contacts.get('press') or {}).get('reason')}"
        return "press_pull_tip", "neither tip contact available"
    if press_result is not None and forward_result is not None:
        return None, "both tips tried"
    if press_result is not None:
        slip = press_result.get("slip_mm")
        if press_result.get("status") == "ok" and (slip is None or slip <= NOSLIP_TOL_MM):
            return None, "press-pull ok" + ("" if slip is None else f", slip {slip:.1f} mm")
        why = (f"press-pull {press_result.get('status')}" if press_result.get("status") != "ok"
               else f"press-pull slipped {slip:.1f} mm > {NOSLIP_TOL_MM:.0f} mm")
        if not _available(contacts, "forward_tip"):
            return None, f"{why}; no forward-tip contact to fall back to"
        return "forward_tip", f"{why}; falling back to forward tip"
    if forward_result.get("status") == "ok":
        return None, "forward tip ok"
    why = f"forward tip {forward_result.get('status')}"
    if not _available(contacts, "press_pull_tip"):
        return None, f"{why}; no press contact to fall back to"
    return "press_pull_tip", f"{why}; falling back to press-pull"
