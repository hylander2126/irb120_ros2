"""
SAM backend launch file — for side-by-side comparison against DBSCAN.

Launches `object_detector_sam` directly against all configured cameras.
Offline perception assumes the arm is clear of the workspace, so this launch
intentionally bypasses `robot_mask_filter`.

Interpreter: this node needs torch/mobile_sam, which live in `~/irb_venv`,
not in irb120_perception's normal (system-Python) build — `irb120_perception`
itself is built with the workspace's regular system colcon and must stay
that way (see "CPU budget" below for why). Rather than rebuilding the whole
package under the venv's Python — which was tried once and reverted after it
took down EGM on live hardware, see the note below — this launch file routes
only this one node's process through the venv interpreter via `prefix`,
bypassing whatever Python the package happens to be built with.

CPU budget — do not remove `additional_env` below without understanding why
it's there: pip-installed numpy (pulled in by `pip install torch`) bundles
OpenBLAS built with MAX_THREADS=64 and no thread-count limit set anywhere in
this environment, unlike the system numpy `robot_mask_filter`/`object_detector`
normally run under. The first time this package was built with the venv's
Python, `robot_mask_filter` alone hit >1100% CPU (a 20-thread BLAS pool
spun up on every point-cloud callback, ~90 Hz across three cameras, forever
while active) and starved the 250 Hz EGM control loop into repeated session
resets on a live robot.

`sam_num_threads` (default 8, was hardcoded to 1) caps
OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS for this node's process.
8 is not "less safe than 1" here — profiled directly (see object_detector_sam
.py's docstring, "Speed tuning"): 8 threads was the *fastest* setting tested
(20, i.e. all cores, was slightly slower — oversubscription overhead), and
this node runs in bounded, throttled bursts (at most one every
`min_reseg_interval_sec`), not continuously, so 8 threads still leaves 12+
of this machine's 20 cores free throughout each burst — unlike the incident
above, which was unbounded thread count running
forever. Pass `sam_num_threads:=1` to go back to the original conservative
value if you'd rather not take that reasoning on faith.

Pipeline topology: each camera supplies synchronized colour + aligned depth;
SAM runs per camera and merges only matching 3-D instances in base_link.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# The venv holding torch/mobile_sam — see this file's docstring, "Interpreter".
VENV_PYTHON = os.path.expanduser('~/irb_venv/bin/python3')


def generate_launch_description() -> LaunchDescription:
    sam_device_arg = DeclareLaunchArgument(
        'sam_device', default_value='cpu',
        description="Torch device (use 'cuda' on a machine with a supported NVIDIA GPU).")
    cam2_color_arg = DeclareLaunchArgument('cam2_color_topic', default_value='/realsense2/color/image_raw')
    cam2_depth_arg = DeclareLaunchArgument('cam2_depth_topic', default_value='/realsense2/aligned_depth_to_color/image_raw')
    cam2_info_arg = DeclareLaunchArgument('cam2_camera_info_topic', default_value='/realsense2/color/camera_info')
    cam3_color_arg = DeclareLaunchArgument('cam3_color_topic', default_value='/realsense3/color/image_raw')
    cam3_depth_arg = DeclareLaunchArgument('cam3_depth_topic', default_value='/realsense3/aligned_depth_to_color/image_raw')
    cam3_info_arg = DeclareLaunchArgument('cam3_camera_info_topic', default_value='/realsense3/color/camera_info')
    sam_checkpoint_arg = DeclareLaunchArgument(
        'sam_checkpoint',
        default_value='',
        description=(
            "Path to the MobileSAM (or full SAM) checkpoint .pt/.pth file. "
            "Required — the node logs an error and never produces detections "
            "without it. See object_detector_sam.py's docstring for the download step."
        ),
    )

    active_at_start_arg = DeclareLaunchArgument(
        'active_at_start',
        default_value='true',
        description=(
            "Whether object_detector_sam starts out processing immediately, "
            "or idle until something calls its /object_detector_sam/set_active service."
        ),
    )
    active_param = {'active': ParameterValue(LaunchConfiguration('active_at_start'), value_type=bool)}

    # ---- Speed/accuracy knobs — see object_detector_sam.py's docstring,
    # "Speed tuning", for the profiled numbers behind these defaults and
    # tradeoffs before changing them. ----------------------------------------
    points_per_side_arg = DeclareLaunchArgument(
        'points_per_side',
        default_value='8',
        description=(
            "Side length of the point-prompt grid (total prompts = this^2). "
            "Dominant speed cost — profiled ~linear in points_per_side^2: "
            "8->~12s, 12->~23s, 16->~39s (thread count aside). Recall drops "
            "off a cliff below 8 (objects go missing, not just slower "
            "convergence) — do not go below 8 without re-checking recall on "
            "your actual scene. Default kept at 16 (favors accuracy); try 8-10 "
            "for comparison runs where you've confirmed recall holds."
        ),
    )
    points_per_batch_arg = DeclareLaunchArgument(
        'points_per_batch',
        default_value='64',
        description=(
            "Point-prompt batch size for the mask decoder. Does not change "
            "total prompt count (points_per_side^2 does) — only how they're "
            "grouped per decoder call. Not a load-bearing speed knob; left "
            "tunable in case CPU cache/batching behavior differs from what "
            "was profiled here."
        ),
    )
    min_reseg_interval_arg = DeclareLaunchArgument(
        'min_reseg_interval_sec',
        default_value='5.0',
        description=(
            "Minimum seconds between re-segmentation runs while active — a "
            "throttle, not a one-shot (an earlier version ran exactly once "
            "per activation and then never again; see object_detector_sam.py's "
            "docstring). Lower this for faster comparison-viewing turnaround "
            "if a run finishes well under this window; raise it if you want "
            "fewer, more deliberate CPU bursts while watching RViz."
        ),
    )
    sam_num_threads_arg = DeclareLaunchArgument(
        'sam_num_threads',
        default_value='8',
        description=(
            "Caps OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS for "
            "this node's process. Profiled as the *fastest* setting (20 = "
            "all cores was slightly slower, pure oversubscription overhead) "
            "— not a speed/safety tradeoff at this value. See this file's "
            "docstring, 'CPU budget', for why 8 is still safe next to the "
            "EGM loop. Pass 1 to go back to the original, more conservative "
            "value if preferred."
        ),
    )

    sam_node = Node(
        package='irb120_perception',
        executable='object_detector_sam',
        name='object_detector_sam',
        output='screen',
        # Runs this node's process under the venv's Python regardless of
        # which interpreter irb120_perception itself was built with — see
        # this file's docstring, "Interpreter".
        prefix=[VENV_PYTHON],
        additional_env={
            'OMP_NUM_THREADS':      LaunchConfiguration('sam_num_threads'),
            'OPENBLAS_NUM_THREADS': LaunchConfiguration('sam_num_threads'),
            'MKL_NUM_THREADS':      LaunchConfiguration('sam_num_threads'),
        },
        parameters=[{
            'base_frame':          'base_link',
            'color_topic':         '/realsense/color/image_raw',
            'depth_topic':         '/realsense/aligned_depth_to_color/image_raw',
            'camera_info_topic':   '/realsense/color/camera_info',
            'color_topic2':        LaunchConfiguration('cam2_color_topic'),
            'depth_topic2':        LaunchConfiguration('cam2_depth_topic'),
            'camera_info_topic2':  LaunchConfiguration('cam2_camera_info_topic'),
            'color_topic3':        LaunchConfiguration('cam3_color_topic'),
            'depth_topic3':        LaunchConfiguration('cam3_depth_topic'),
            'camera_info_topic3':  LaunchConfiguration('cam3_camera_info_topic'),
            # Same workspace ROI as perception.launch.py's DBSCAN params —
            # keep these two in sync by hand if you retune the workspace box.
            'roi_x_min':  0.15,
            'roi_x_max':  0.80,
            'roi_y_min': -0.25,
            'roi_y_max':  0.25,
            'roi_z_min': -0.01,
            'roi_z_max':  0.50,
            'sam_model_type':  'vit_t',   # MobileSAM — CPU-friendly. 'vit_h' etc. also load, but are impractically slow on CPU.
            'sam_checkpoint':  LaunchConfiguration('sam_checkpoint'),
            'sam_device':      LaunchConfiguration('sam_device'),
            'points_per_side':  ParameterValue(LaunchConfiguration('points_per_side'), value_type=int),
            'points_per_batch': ParameterValue(LaunchConfiguration('points_per_batch'), value_type=int),
            'min_reseg_interval_sec': ParameterValue(LaunchConfiguration('min_reseg_interval_sec'), value_type=float),
            'pred_iou_thresh': 0.86,
            'stability_score_thresh': 0.92,
            'min_mask_region_px': 400,
            'min_cluster_pts': 30,
            'max_cluster_pts': 50000,
            'max_depth_gap_ratio': 0.3,
            'table_plane_distance': 0.008,
            'smooth_alpha': 0.3,
        }, active_param],
    )

    return LaunchDescription([
        sam_device_arg,
        cam2_color_arg,
        cam2_depth_arg,
        cam2_info_arg,
        cam3_color_arg,
        cam3_depth_arg,
        cam3_info_arg,
        sam_checkpoint_arg,
        active_at_start_arg,
        points_per_side_arg,
        points_per_batch_arg,
        min_reseg_interval_arg,
        sam_num_threads_arg,
        sam_node,
    ])
