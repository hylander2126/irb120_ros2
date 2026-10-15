"""Tell a frozen control loop from a dead sensor stream.

A sensor-freshness watchdog measures the age of the last *received* sample. If
this process itself freezes (swap, GC, scheduling), no callbacks run: when it
resumes, the control timer can fire before the queued samples are handled and
the watchdog sees an old sample although the sensor never stopped. Seen on
2026-10-09: a 100 ms freeze of arc_static aborted UNARC with ft_stale while
the sensor stamps ran on every ~2 ms; the backlog was delivered 1 ms later.

So a tick that itself ran late holds still for one tick instead of judging
staleness; the next, on-time tick (one control period later) judges it after
the backlog has been taken. A real dropout is still caught one tick later. The
freeze is taken off the sample's age first: on 2026-10-15 fwd_tip froze 514 ms
and its sample, fresh when the freeze began, came out 507 ms old -- past a
fixed age cap, so the run stopped with stale_force.

The freezes during control were Python's full (generation-2) GC pass: one every
~7.7k logged rows, each longer than the last as the in-memory logs grew (86 ms
-> 268 ms over one 83 s fwd_tip run). gc_paused() holds it off for the loop.
"""

from contextlib import contextmanager
import gc
import time

STALL_SEC = 0.05      # a tick this much after the previous one means the process froze
MAX_DEFER_SEC = 0.5   # never defer the verdict on a sample this old before the freeze


class StallGuard:
    def __init__(self):
        self._last = None

    def reset(self) -> None:
        """Call when the control timer (re)starts, so the gap before it isn't a stall."""
        self._last = None

    def tick(self) -> float:
        """Call once at the top of every control tick. The gap since the previous
        tick if this one ran late (the freeze length), else 0.0."""
        now = time.monotonic()
        gap = 0. if self._last is None else now - self._last
        self._last = now
        return gap if gap > STALL_SEC else 0.

    @staticmethod
    def defer(stall_s: float, sample_age_s: float) -> bool:
        """True if a stale verdict on a sample this old should wait one tick."""
        return stall_s > 0. and sample_age_s - stall_s < MAX_DEFER_SEC


@contextmanager
def gc_paused():
    """Hold off the cyclic GC (reference counting still frees memory) for a control loop."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
