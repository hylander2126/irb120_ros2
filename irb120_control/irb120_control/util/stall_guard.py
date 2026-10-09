"""Tell a frozen control loop from a dead sensor stream.

A sensor-freshness watchdog measures the age of the last *received* sample. If
this process itself freezes (swap, GC, scheduling), no callbacks run: when it
resumes, the control timer can fire before the queued samples are handled and
the watchdog sees an old sample although the sensor never stopped. Seen on
2026-10-09: a 100 ms freeze of arc_static aborted UNARC with ft_stale while
the sensor stamps ran on every ~2 ms; the backlog was delivered 1 ms later.

So a tick that itself ran late holds still for one tick instead of judging
staleness; the next, on-time tick (one control period later) judges it after
the backlog has been taken. A real dropout is still caught one tick later. Past
MAX_DEFER_SEC the sample is too old to wait for, late tick or not.
"""

import time

STALL_SEC = 0.05      # a tick this much after the previous one means the process froze
MAX_DEFER_SEC = 0.5   # never defer the verdict on a sample older than this


class StallGuard:
    def __init__(self):
        self._last = None

    def reset(self) -> None:
        """Call when the control timer (re)starts, so the gap before it isn't a stall."""
        self._last = None

    def tick(self) -> bool:
        """Call once at the top of every control tick. True if this tick ran late."""
        now = time.monotonic()
        late = self._last is not None and now - self._last > STALL_SEC
        self._last = now
        return late

    @staticmethod
    def defer(late: bool, sample_age_s: float) -> bool:
        """True if a stale verdict on a sample this old should wait one tick."""
        return late and sample_age_s < MAX_DEFER_SEC
