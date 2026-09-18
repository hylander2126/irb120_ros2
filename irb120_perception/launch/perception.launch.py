"""
Perception launch file.

Switch between segmentation backends via the 'method' argument:

  ros2 launch irb120_perception perception.launch.py method:=dbscan
  ros2 launch irb120_perception perception.launch.py method:=sam

Enable the debug pipeline (perception_debugger node) with:

  ros2 launch irb120_perception perception.launch.py method:=sam debug_perception:=true

robot_mask_filter and object_detector are compute-heavy but only actually
needed briefly (see their own docstrings' "On/off gate" section) —
`press_point_check.check_press_point()` already toggles them on/off around
each check regardless of this launch arg. This one just controls what state
they're in before the first check ever runs (or if you want to watch live
detections in RViz without running a control script):

  ros2 launch irb120_perception perception.launch.py active_at_start:=false

DBSCAN: runs under system python, no GPU needed.
SAM:    runs under the venv python (~/.venvs/.venv_torch_SAM/bin/python3),
        requires CUDA GPU and SAM 2 weights.

Pipeline topology:

  RealSense (cam1) ──┬──▶ robot_mask_filter ──▶ object_detector_dbscan
  RealSense (cam2) ──┘        │ ~/points_masked_dbscan  (both cameras fused, DBSCAN input)
                               └ ~/depth_masked_sam      (cam1 only, SAM input)

  DBSCAN gets both cameras' point clouds fused (see robot_mask_filter's
  docstring) to reduce occlusion and increase point density. SAM stays
  single-camera — set cam2_cloud_topic:='' to disable fusion entirely.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

PKG_SHARE      = get_package_share_directory('irb120_perception')
SAM_WEIGHTS    = os.path.join(PKG_SHARE, 'weights', 'sam2.1_hiera_tiny.pt')
VENV_SITE_PKGS = os.path.expanduser(
    '~/.venvs/.venv_torch_SAM/lib/python3.12/site-packages')

# Masked topic names published by robot_mask_filter
MASKED_CLOUD = '/robot_mask_filter/points_masked_dbscan'
MASKED_DEPTH = '/robot_mask_filter/depth_masked_sam'


def generate_launch_description() -> LaunchDescription:
    method_arg = DeclareLaunchArgument(
        'method',
        default_value='dbscan',
        description="Segmentation backend: 'dbscan' or 'sam'",
    )

    debug_arg = DeclareLaunchArgument(
        'debug_perception',
        default_value='false',
        description='Launch the perception_debugger node for on-demand SAM pipeline inspection.',
    )

    cam2_cloud_arg = DeclareLaunchArgument(
        'cam2_cloud_topic',
        default_value='/realsense2/depth/color/points',
        description=(
            "Second camera's point cloud, fused into the DBSCAN input by "
            "robot_mask_filter. Set to '' to disable fusion and run DBSCAN "
            "on camera 1 alone."
        ),
    )

    active_at_start_arg = DeclareLaunchArgument(
        'active_at_start',
        default_value='true',
        description=(
            "Whether robot_mask_filter + object_detector start out processing "
            "immediately (default, matches historical always-on behaviour) or "
            "idle until the first check_press_point() call activates them."
        ),
    )
    active_param = {'active': ParameterValue(LaunchConfiguration('active_at_start'), value_type=bool)}

    # ---- Robot mask filter (always running, both backends benefit) ----------
    mask_filter_node = Node(
        package='irb120_perception',
        executable='robot_mask_filter',
        name='robot_mask_filter',
        output='screen',
        parameters=[{
            'base_frame':  'base_link',
            'input_cloud': '/realsense/depth/color/points',
            'input_cloud2': LaunchConfiguration('cam2_cloud_topic'),
            'input_depth': '/realsense/aligned_depth_to_color/image_raw',
            'camera_info': '/realsense/color/camera_info',
            'robot_mask_padding': 0.08,  # arm mesh links — increase if arm still leaks through
            # Capsule segments (wrist sensor stack + finger) are fixed in
            # RobotMaskFilter.CAPSULE_SEGMENTS, not a launch param — see that
            # node's docstring. (A 'robot_mask_capsules' key used to be passed
            # here, but the node never declared/read it — dead config left
            # over from before the finger/sensor assembly was rebuilt, and it
            # still named the now-nonexistent ft_link/finger_link frames.)
        }, active_param],
    )

    # ---- DBSCAN parameters — reads from masked, camera-fused pointcloud -----
    dbscan_params = {
        'input_cloud_pc':      MASKED_CLOUD,
        'base_frame':          'base_link',
        'roi_x_min':  0.15,
        'roi_x_max':  0.80,
        'roi_y_min': -0.25,
        'roi_y_max':  0.25,
        'roi_z_min': -0.01,  # Table at Z≈-0.02 (tilts up to ≈-0.015 at far x with the 2026-09-18 cam calibration)
        'roi_z_max':  0.50,
        'voxel_size':      0.005,
        'dbscan_eps':      0.02,
        'dbscan_min_pts':  20,
        'min_cluster_pts': 30,
        'max_cluster_pts': 50000,
        'smooth_alpha':    0.3,
        # Union all surviving clusters into one object — only safe if the
        # workspace is scoped to a single physical item per detection cycle.
        # See object_detector_dbscan's docstring.
        'single_object_mode': True,
    }

    # ---- SAM parameters — reads from masked depth image --------------------
    sam_params = {
        'input_cloud':    MASKED_DEPTH,
        'input_image':    '/realsense/color/image_raw',
        'camera_info':    '/realsense/color/camera_info',
        'base_frame':     'base_link',
        'roi_x_min':  0.15,
        'roi_x_max':  0.80,
        'roi_y_min': -0.25,
        'roi_y_max':  0.25,
        'roi_z_min': -0.01,  # Table at Z≈-0.02; objects start above -0.01
        'roi_z_max':  0.50,
        'voxel_size':          0.005,
        'sam_weights':         SAM_WEIGHTS,
        'sam_config':          'configs/sam2.1/sam2.1_hiera_t.yaml',
        'sam_points_per_side':  8,
        'sam_iou_thresh':       0.85,
        'sam_min_mask_area':    1000,
        'sam_min_cluster_pts':  30,
        'sam_prominent_only':   True,
        'depth_median_ksize':   5,
        'outlier_std_ratio':    2.0,
        'smooth_alpha':         0.3,
    }

    # ---- DBSCAN node (system python) ----------------------------------------
    dbscan_node = Node(
        package='irb120_perception',
        executable='object_detector_dbscan',
        name='object_detector',
        output='screen',
        parameters=[dbscan_params, active_param],
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('method'), 'dbscan')),
    )

    # ---- SAM node -----------------------------------------------------------
    sam_node = Node(
        package='irb120_perception',
        executable='object_detector_sam',
        name='object_detector',
        output='screen',
        parameters=[sam_params, active_param],
        additional_env={'PYTHONPATH': VENV_SITE_PKGS + ':' + os.environ.get('PYTHONPATH', '')},
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('method'), 'sam')),
    )

    # ---- Debug node (optional) ----------------------------------------------
    debugger_node = Node(
        package='irb120_perception',
        executable='perception_debugger',
        name='perception_debugger',
        output='screen',
        condition=IfCondition(LaunchConfiguration('debug_perception')),
    )

    # ---- Press point selector -----------------------------------------------
    # NOT launched here: it's a one-shot node (compute once, print, publish,
    # exit — see its module docstring), so including it in this persistent
    # bringup would just fire it once at launch startup, likely before any
    # object is even detected. Invoke it deliberately instead, once detections
    # are live, either:
    #
    #   ros2 run irb120_perception press_point_selector
    #
    # or programmatically from another node's code — import
    # `select_press_point` (pure function) or `PressPointSelector` (full
    # node, `.run_once()` method) from irb120_perception.press_point_selector.
    # See that module's docstring for both.

    return LaunchDescription([
        method_arg,
        debug_arg,
        cam2_cloud_arg,
        active_at_start_arg,
        mask_filter_node,
        dbscan_node,
        sam_node,
        debugger_node,
    ])
