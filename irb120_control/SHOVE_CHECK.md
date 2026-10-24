# Shove geometric feasibility check

`shove check` queries MoveIt without executing motion or activating EGM. It checks
contact IK, then a 1 mm resolution Cartesian path through every supplied sample,
including release and the braking endpoint. Full coverage, MoveIt success, and
the fixed joint-6 +/-60 degree cable window are required. Absolute revolute jumps
over 0.1 rad truncate the path. Contact IK is seeded from explicit input joints;
the Cartesian path starts at the solved contact, not the current arm pose.

With MoveIt available and the workspace sourced, after building irb120_control:

```bash
ros2 run irb120_control shove check request.json --output check.json
```

Or use the source directly with the package directory on PYTHONPATH:

```bash
python3 -m irb120_control.shove check request.json --output check.json
```

See [examples/shove_check.json](examples/shove_check.json) for the schema. Its
numbers are illustrative, not an approved hardware launch. Supply all seed joints
and ball-center poses in base_link, unit xyzw orientations, strictly increasing
seconds starting at initial contact, and a release_index strictly before the
braking endpoint. Use dense samples of the actual calculated curve.

Candidates are tried in preference order; first complete path wins. Alternative
candidates must come from a fresh physics solve for the same task, such as a
shorter contact interval with higher acceleration. Merely speeding up an identical
geometric stroke does not fix workspace exclusion. This increment provides the
candidate selection hook; it does not yet port the analytical solver or generate
these alternatives automatically.

The JSON report preserves the request, attempted candidates, coverage/error codes,
contact robot state, and returned MoveIt trajectories. Exit 0 means a geometric
path was found; exit 2 means none was found. Failure does not prove workspace
exclusion: collisions, cable constraints, seed choice, and service failures also
matter.

The existing planning scene is used unchanged, with collision avoidance enabled.
An object at contact can reject IK unless its intended ball contact is already
represented by an appropriate allowed-collision policy. The checker never removes
objects or changes that policy. Scene completeness and a moving-object sweep are
not certified. No approach path is checked.

`timing_verified` and `execution_ready` remain false even on geometric success.
MoveIt retiming does not preserve the requested launch timing. Next work is the
analytical solver adapter, fixed-time joint compilation and interpolated dynamic
limit/FK checks, plus scoped contact and moving-object scene validation. The saved
trajectory is diagnostic and must not be dispatched as a hardware shove.

Validation: pure candidate tests and real ROS message adapter tests use fake
service replies; no live robot or MoveIt service was contacted.
