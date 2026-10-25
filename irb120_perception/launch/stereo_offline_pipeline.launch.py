"""Run the full offline FFS stereo pipeline for one named object:
capture -> reconstruct -> observed-only contacts (review:=true adds mesh/report). FFS only; no RealSense comparison.

Everything lands under runtime_logs/<object_name>/<timestamp>/ in the workspace:
  capture/   raw matched stereo IR + RGB + TF (stereo_offline capture)
  recon/     FFS reconstruction + report (stereo_offline reconstruct, stereo_offline report)
  contacts/  observed-only contact review bundle (stereo_offline contacts)

Each step only runs if the previous one exits 0; a failure stops the chain and is logged.

Assumes the raw-depth camera profile (infra1/infra2/depth/color, emitters on, unfiltered) is
already streaming from another terminal, with one stationary object placed and the arm clear.
Either of these satisfies that:
  ros2 launch irb120_handeye bringup_cams.launch.py profile:=offline_stereo   # cameras only, no robot
  ros2 launch irb120_control bringup_stack.launch.py            # full T1/T2/T3 robot+camera stack
This launch file does not start or stop cameras, so the same camera session (standalone or as
part of bringup_stack) can be reused across multiple objects/runs — e.g. the T1 rws / T2 ctrl /
T3 bringup_stack / T4 <this file> workflow: T4 just needs object_name once T1-T3 are up.

Usage:
  ros2 launch irb120_perception stereo_offline_pipeline.launch.py object_name:=flashlight
"""
from datetime import datetime
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration

WORKSPACE = Path.home() / 'Documents' / 'irb120_ws'


def _launch_setup(context, *args, **kwargs):
    object_name = LaunchConfiguration('object_name').perform(context)
    if not object_name:
        raise RuntimeError('object_name launch argument is required, e.g. object_name:=flashlight')
    timeout = LaunchConfiguration('timeout').perform(context)
    gap_guard_voxels = LaunchConfiguration('gap_guard_voxels').perform(context)
    noise_radius_voxels = LaunchConfiguration('noise_radius_voxels').perform(context)
    noise_min_neighbors = LaunchConfiguration('noise_min_neighbors').perform(context)
    cam2_cross_checks = LaunchConfiguration('cam2_cross_checks').perform(context).lower() == 'true'
    cam2_z_offset_mm = LaunchConfiguration('cam2_z_offset_mm').perform(context)
    review = LaunchConfiguration('review').perform(context).lower() == 'true'

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = WORKSPACE / 'runtime_logs' / object_name / stamp
    capture_dir = run_dir / 'capture'
    recon_dir = run_dir / 'recon'
    contacts_dir = run_dir / 'contacts'

    capture = ExecuteProcess(
        name='stereo_offline_capture', output='screen',
        cmd=['ros2', 'run', 'irb120_perception', 'stereo_offline', 'capture',
             str(capture_dir), '--stationary-arm-clear', '--timeout', timeout],
    )
    reconstruct = ExecuteProcess(
        name='stereo_offline_reconstruct', output='screen',
        cmd=['ros2', 'run', 'irb120_perception', 'stereo_offline', 'reconstruct',
             str(capture_dir), '--output', str(recon_dir),
             '--gap-guard-voxels', gap_guard_voxels,
             '--noise-radius-voxels', noise_radius_voxels,
             '--noise-min-neighbors', noise_min_neighbors,
             '--cam2-z-offset-mm', cam2_z_offset_mm] + (['--cam2-cross-checks'] if cam2_cross_checks else [])
             + (['--mesh'] if review else []),
    )
    report = ExecuteProcess(
        name='stereo_offline_report', output='screen',
        cmd=['ros2', 'run', 'irb120_perception', 'stereo_offline', 'report', str(recon_dir)],
    )
    contacts = ExecuteProcess(
        name='stereo_offline_contacts', output='screen',
        cmd=['ros2', 'run', 'irb120_perception', 'stereo_offline', 'contacts',
             str(recon_dir), '--output', str(contacts_dir)] + (['--review-mesh'] if review else []),
    )

    def _then(step_name, next_action):
        def _on_exit(event, context):
            if event.returncode != 0:
                return [LogInfo(msg=(
                    f'stereo_offline_pipeline: {step_name} failed (exit {event.returncode}); '
                    f'stopping. object={object_name} run_dir={run_dir}'))]
            return [next_action]
        return _on_exit

    return [
        LogInfo(msg=f'stereo_offline_pipeline: object={object_name} run_dir={run_dir}'),
        capture,
        RegisterEventHandler(OnProcessExit(target_action=capture, on_exit=_then('capture', reconstruct))),
        RegisterEventHandler(OnProcessExit(target_action=reconstruct, on_exit=_then('reconstruct', report if review else contacts))),
        RegisterEventHandler(OnProcessExit(target_action=report, on_exit=_then('report', contacts))),
        RegisterEventHandler(OnProcessExit(target_action=contacts, on_exit=lambda event, context: [LogInfo(msg=(
            f'stereo_offline_pipeline: done for {object_name} (contacts exit {event.returncode}). '
            f'See {run_dir}/contacts/README.md'))])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('review', default_value='false',
                              description='Build optional meshes and reconstruction reports; MVP default is off.'),
        DeclareLaunchArgument(
            'object_name',
            description='Object label; outputs go under runtime_logs/<object_name>/<timestamp>/.',
        ),
        DeclareLaunchArgument(
            'timeout', default_value='45',
            description='Seconds to wait for a stationary, matched capture (passed to stereo_offline capture).',
        ),
        DeclareLaunchArgument(
            'gap_guard_voxels', default_value='0',
            description=(
                'Voxels (3mm each) of clearance a cam2-only mesh triangle needs from cam1/cam3 '
                'geometry to survive meshing; 0 disables the guard entirely. See '
                'scripts/STEREO_OFFLINE.md "Cam2/top-mesh gap guard".'
            ),
        ),
        DeclareLaunchArgument(
            'noise_radius_voxels', default_value='3',
            description=(
                'Voxels (3mm each) defining the neighborhood radius for the per-camera '
                'isolated-point noise filter -- cam1/cam2/cam3 are each filtered independently '
                '(drops pixels, e.g. at a top cap only one camera sees, that have too few nearby '
                'same-camera neighbors to look like real surface).'
            ),
        ),
        DeclareLaunchArgument(
            'cam2_z_offset_mm', default_value='auto',
            description='auto ensures total +4 mm relative to audited calibration, accounting for the captured TF; '
                        'a number explicitly adds that many mm during reconstruction.'),
        DeclareLaunchArgument(
            'cam2_cross_checks', default_value='false',
            description='Re-enable dropping cam2 points that cam1/cam3 corroborate or conflict with.'),
        DeclareLaunchArgument(
            'noise_min_neighbors', default_value='4',
            description=(
                'Minimum same-camera neighbors (including itself) within noise_radius_voxels to '
                'keep a pixel; 0 disables the filter.'
            ),
        ),
        OpaqueFunction(function=_launch_setup),
    ])
