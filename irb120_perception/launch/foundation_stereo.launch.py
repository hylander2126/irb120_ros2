"""
Fast-FoundationStereo depth node — launch wrapper.

Reusable per camera: run once per `camera_name` in its own terminal, next to
that camera's `bringup_camN.launch.py` (which must publish infra1/infra2 —
on by default now via `raw_depth`/`depth_diagnostics`). Does not touch any
existing topic or node.

  ros2 launch irb120_perception foundation_stereo.launch.py camera_name:=realsense
  ros2 launch irb120_perception foundation_stereo.launch.py camera_name:=realsense2
  ros2 launch irb120_perception foundation_stereo.launch.py camera_name:=realsense3

Interpreter: like `object_detector_sam`, this node needs torch, which lives
in the `irb_venv` virtualenv, not in `irb120_perception`'s normal
system-Python build. Routed through `prefix` the same way — see
`perception_sam.launch.py`'s docstring ("Interpreter", "CPU budget") for why
`irb120_perception` itself must stay built under the system interpreter, and
why `num_threads` below caps this process's BLAS thread pools next to the
250 Hz EGM loop.

Output: /<camera_name>/depth_ffs/image_raw (32FC1, metres, left-IR optical
frame — not aligned_depth_to_color, see foundation_stereo_depth.py's
docstring) and /<camera_name>/depth_ffs/camera_info.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# The venv holding torch/Fast-FoundationStereo — see this file's docstring, "Interpreter".
VENV_PYTHON = os.path.expanduser('~/.virtualenvs/irb_venv/bin/python3')
_DEFAULT_REPO = os.path.expanduser('~/Documents/irb120_ws/third_party/Fast-FoundationStereo')
_DEFAULT_CKPT = os.path.join(_DEFAULT_REPO, 'weights', '23-36-37', 'model_best_bp2_serialize.pth')


def _launch(context, *args, **kwargs):
    camera_name = LaunchConfiguration('camera_name').perform(context)

    def _topic(arg_name, default_built):
        value = LaunchConfiguration(arg_name).perform(context)
        return value if value else default_built

    node = Node(
        package='irb120_perception',
        executable='foundation_stereo_depth',
        name=f'foundation_stereo_depth_{camera_name}',
        output='screen',
        prefix=[VENV_PYTHON],
        additional_env={
            'OMP_NUM_THREADS': LaunchConfiguration('num_threads').perform(context),
            'OPENBLAS_NUM_THREADS': LaunchConfiguration('num_threads').perform(context),
            'MKL_NUM_THREADS': LaunchConfiguration('num_threads').perform(context),
        },
        parameters=[{
            'infra1_topic': _topic('infra1_topic', f'/{camera_name}/infra1/image_rect_raw'),
            'infra2_topic': _topic('infra2_topic', f'/{camera_name}/infra2/image_rect_raw'),
            'camera_info_topic': _topic('camera_info_topic', f'/{camera_name}/infra1/camera_info'),
            'right_camera_info_topic': f'/{camera_name}/infra2/camera_info',
            'output_image_topic': _topic('output_image_topic', f'/{camera_name}/depth_ffs/image_raw'),
            'output_camera_info_topic': _topic('output_camera_info_topic', f'/{camera_name}/depth_ffs/camera_info'),
            'checkpoint_path': LaunchConfiguration('checkpoint_path').perform(context),
            'ffs_repo_path': LaunchConfiguration('ffs_repo_path').perform(context),
            'baseline_m': float(LaunchConfiguration('baseline_m').perform(context)),
            'valid_iters': int(LaunchConfiguration('valid_iters').perform(context)),
            'max_disp': int(LaunchConfiguration('max_disp').perform(context)),
            'device': LaunchConfiguration('device').perform(context),
            'min_depth_m': float(LaunchConfiguration('min_depth_m').perform(context)),
            'max_depth_m': float(LaunchConfiguration('max_depth_m').perform(context)),
            'sync_slop_sec': float(LaunchConfiguration('sync_slop_sec').perform(context)),
        }],
    )
    return [node]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('camera_name', default_value='realsense',
                              description='Which RealSense to attach to: realsense | realsense2 | realsense3.'),
        DeclareLaunchArgument('infra1_topic', default_value='', description='Override; default derived from camera_name.'),
        DeclareLaunchArgument('infra2_topic', default_value='', description='Override; default derived from camera_name.'),
        DeclareLaunchArgument('camera_info_topic', default_value='', description='Override; default derived from camera_name.'),
        DeclareLaunchArgument('output_image_topic', default_value='', description='Override; default derived from camera_name.'),
        DeclareLaunchArgument('output_camera_info_topic', default_value='', description='Override; default derived from camera_name.'),
        DeclareLaunchArgument('checkpoint_path', default_value=_DEFAULT_CKPT,
                              description='Fast-FoundationStereo .pth checkpoint.'),
        DeclareLaunchArgument('ffs_repo_path', default_value=_DEFAULT_REPO,
                              description='Cloned NVlabs/Fast-FoundationStereo repo (core/ and Utils.py).'),
        DeclareLaunchArgument('baseline_m', default_value='0.0',
                              description='Zero derives the per-camera baseline from rectified CameraInfo.P; positive metres explicitly overrides.'),
        DeclareLaunchArgument('valid_iters', default_value='8'),
        DeclareLaunchArgument('max_disp', default_value='192'),
        DeclareLaunchArgument('device', default_value='cuda'),
        DeclareLaunchArgument('min_depth_m', default_value='0.2'),
        DeclareLaunchArgument('max_depth_m', default_value='1.4'),
        DeclareLaunchArgument('sync_slop_sec', default_value='0.01'),
        DeclareLaunchArgument('num_threads', default_value='4',
                              description='Caps OMP/OPENBLAS/MKL thread pools for this process — see this file\'s docstring and perception_sam.launch.py "CPU budget".'),
        OpaqueFunction(function=_launch),
    ])
