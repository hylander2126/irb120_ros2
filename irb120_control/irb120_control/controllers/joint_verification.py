"""Measured-joint goal verification, independent of action-result success."""
import math


class JointGoalVerifier:
    def __init__(self, names, target, started, tolerance=.01, dwell=.3, max_age=.25):
        self.names = list(names)
        self.target = list(target)
        self.started = started
        self.tolerance = tolerance
        self.dwell = dwell
        self.max_age = max_age
        self.last_stamp = None
        self.since = None
        self.samples = 0
        self.report = dict(verified=False, reason='no fresh joint feedback',
                           target=self.target, joint_names=self.names,
                           tolerance_rad=tolerance, dwell_seconds=dwell)

    def observe(self, stamp, now, names, positions):
        if (not math.isfinite(stamp) or not math.isfinite(now) or
                stamp <= self.started or not 0 <= now-stamp <= self.max_age or
                (self.last_stamp is not None and stamp <= self.last_stamp)):
            self.since = None
            self.report.update(verified=False, reason='stale or nonmonotonic feedback')
            return False
        if self.last_stamp is not None and stamp-self.last_stamp > self.max_age:
            self.since = None
        self.last_stamp = stamp
        values = dict(zip(names, positions))
        if (len(names) != len(positions) or len(set(names)) != len(names) or
                any(n not in values or not math.isfinite(values[n]) for n in self.names)):
            self.since = None
            self.report.update(verified=False, reason='missing or invalid joint positions')
            return False
        actual = [float(values[n]) for n in self.names]
        error = max(abs(a-b) for a,b in zip(actual,self.target))
        self.samples += 1
        self.report.update(actual=actual, max_error_rad=error, stamp=stamp, samples=self.samples)
        if error > self.tolerance:
            self.since = None
            self.report.update(verified=False, reason='measured joints outside tolerance')
            return False
        if self.since is None:
            self.since = stamp
        verified = stamp-self.since >= self.dwell and self.samples >= 3
        self.report.update(verified=verified, reason='verified' if verified else 'waiting for stable dwell')
        return verified
