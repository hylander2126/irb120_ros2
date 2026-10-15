"""Which tip to run on a snapshot (press-pull or forward tip), and why.

Shared by the perception snapshot, which records its choice in snapshot.json,
and run_pipeline, whose tip cascade (util/tip_cascade.py) starts with it.
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


def choose_tip(contacts: dict, mode: str = "auto") -> tuple:
    """(step kind, reason) for the first tip on this snapshot; what follows a failed
    tip is util/tip_cascade.py's.

    auto: decided by the forward pivot's geometry alone (forward_pivot_shape). A
    straight edge suits forward tipping: the fixed-pivot push about an extended edge.
    A curved pivot is a point contact, about which the object can roll or yaw under a
    forward push; press-pull is preferred there, and when the shape is unknown
    (press-pull has the estimator and its pivot is the near edge cam1/cam2 see).
    Either way, the other tip runs when the preferred one has no contact.
    """
    if mode != "auto":
        return TIP_MODES[mode], f"--tip {mode}"
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
