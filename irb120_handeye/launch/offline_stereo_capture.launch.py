"""Opt-in stationary high-resolution capture; cameras + calibrated TF only.
Stop existing camera drivers before launching; no robot/controller nodes.
"""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    handeye=Path(get_package_share_directory('irb120_handeye'))
    driver=Path(get_package_share_directory('realsense2_camera'))/'launch/rs_launch.py'
    nodes=[]
    for name,serial,tf in [
        ('realsense','243522072478','cam_tf_realsense_1.7px.launch.py'),
        ('realsense2','750612071219','cam_tf_realsense2_2.1px.launch.py'),
        ('realsense3','213622073793','cam_tf_realsense3_1.6px.launch.py')]:
        nodes.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(str(driver)),launch_arguments={
            'camera_name':name,'camera_namespace':'','serial_no':f"'{serial}'",
            'config_file':str(handeye/'config/realsense_offline_stereo.yaml'),
            'clip_distance':'1.4',
        }.items()))
        nodes.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(str(handeye/'launch'/tf))))
    return LaunchDescription(nodes)
