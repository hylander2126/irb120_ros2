# Tipping model and frame contract

The task estimates mass and CoM height from an object's measured tipping response
and the finger wrench, with supplied horizontal pivot-to-CoM distance. The
press-pull primitive applies a controlled downward press and pulls toward the
robot; the forward primitive follows an arc around the estimated far pivot.
Contact geometry, no-slip assumptions and object support observability must be
assessed separately from fit residuals.

The shared estimator is `estimation/tip_core.py`. Its quasi-static pivot model is

```
tau(phi) = m g (com_x cos(phi) - com_z sin(phi))
```

Here CoM distances are relative to the selected pivot, not interchangeable with
body-frame coordinates. Object registries define their frames and provenance in
[irb120_models](../../irb120_models/README.md). The estimator accepts proprioceptive
and visual tilt tracks; finger motion alone cannot distinguish all slip/tilt
cases. Its onset-anchored and free-fit methods have different assumptions; see
the module docstring and [analysis methods](../../analysis/README.md).

Wrench axes are `tool0` despite the historical `ft_link` message header. Pose
logs retain the tool0 reference; `util/ft_frames.py` supplies the current
ball-to-mount wrench reference conversion. Do not assume the sensor axes align
with world, copy the obsolete 82.25 mm adapter offset, or read an old fixed 4 N
example as a current controller setting. `world` and `base_link` are distinct.

F/T gravity/tare compensation does not compensate accelerating tool inertia.
The quasi-static tipping estimator and dynamic shove release model therefore
have separate validity limits. Current motion thresholds live in the primitive
code and reviewed object-control configuration.

The original simplified problem sketch is retained in
[the archive](../../docs/archive/irb120_control__CLAUDE_README_FOR_PROBLEM_STATEMENT.md).
