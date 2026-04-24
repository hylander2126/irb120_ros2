from glob import glob

from setuptools import find_packages, setup

package_name = 'irb120_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/handeye_calibrations', glob('handeye_calibrations/*')),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
        ('share/' + package_name + '/scripts', glob('scripts/*')),
        ('share/' + package_name + '/meshes/irb120_3_58/collision', glob('meshes/irb120_3_58/collision/*')),
        ('share/' + package_name + '/meshes/irb120_3_58/visual', glob('meshes/irb120_3_58/visual/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hylander2126',
    maintainer_email='stevenhyland1@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'egm_handler = irb120_control.egm_handler:main',
            'egm_ready_waiter = irb120_control.egm_ready_waiter:main',
            'safe_stop = irb120_control.safe_stop:main',
            'run_handeye_calibration_poses = irb120_control.run_handeye_calibration_poses:main',
            'keyboard_jog = irb120_control.keyboard_jog:main',
            'netft_preprocessor = irb120_control.netft_preprocessor:main',
            'squash_pull = irb120_control.squash_pull:main',
            'camera_hull_recorder = irb120_control.camera_hull_recorder:main',
        ],
    },
)
