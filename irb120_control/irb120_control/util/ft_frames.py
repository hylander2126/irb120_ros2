"""NetFT frame convention retained from the original tool-transform setup.

The old tool0_to_ft joint was identity (commit 9296f57). NetFT's internal
axis transform was configured for those axes. root_finger is a CAD frame
with different axes. Published ft_link labels are retained for consumers.
The estimator expects the tool0 reference pose and adds its own 0.08225 m
wrench-origin offset; do not log the distal origin in its place.
"""

WRENCH_AXES_FRAME = 'tool0'
ESTIMATOR_POSE_FRAME = 'tool0'
WRENCH_HEADER_FRAME = 'ft_link'
