"""Pivot-axis circle geometry, independent of ROS. Positions are in metres."""
import math
import numpy as np


class PivotArc:
    def __init__(self, pivot, axis, ball, direction):
        self.pivot, self.ball = np.asarray(pivot, float), np.asarray(ball, float)
        axis, direction = np.asarray(axis, float), np.asarray(direction, float)
        if any(v.shape != (3,) or not np.isfinite(v).all()
               for v in (self.pivot, self.ball, axis, direction)) or np.linalg.norm(axis) < 1e-9:
            raise ValueError('Arc needs finite pivot, axis, ball and direction')
        self.axis = axis / np.linalg.norm(axis)
        self.center = self.pivot + self.axis * ((self.ball - self.pivot) @ self.axis)
        self.radial = self.ball - self.center
        self.radius = float(np.linalg.norm(self.radial))
        if self.radius < .001:
            raise ValueError('Contact is too close to the pivot axis')
        if np.cross(self.axis, self.radial) @ direction < 0:
            self.axis = -self.axis
        self.u = self.radial / self.radius
        self.v = np.cross(self.axis, self.u)
        if self.v @ direction < .2:
            raise ValueError('Arc tangent does not push into the selected face')

    def point(self, angle):
        return self.center + self.radius * (math.cos(angle) * self.u + math.sin(angle) * self.v)

    def tangent(self, angle):
        return -math.sin(angle) * self.u + math.cos(angle) * self.v

    def angle(self, point):
        r = np.asarray(point) - self.center
        return math.atan2(r @ self.v, r @ self.u)


def pivot_torque(force, moment, sensor_origin, pivot, axis):
    """Full wrench shifted from its physical origin, projected onto the edge."""
    return float((np.asarray(moment) + np.cross(np.asarray(sensor_origin) - pivot, force)) @ axis)
