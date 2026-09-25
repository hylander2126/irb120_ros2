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
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.json')),
        ('share/' + package_name + '/handeye_calibrations', glob('handeye_calibrations/*')),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro') + glob('urdf/*.stl') + glob('urdf/*.urdf')),
        ('share/' + package_name + '/meshes/irb120_3_58/collision', glob('meshes/irb120_3_58/collision/*')),
        ('share/' + package_name + '/meshes/irb120_3_58/visual', glob('meshes/irb120_3_58/visual/*')),
        ('share/' + package_name + '/meshes/sensor_and_adapter_assembly', glob('meshes/sensor_and_adapter_assembly/*')),
        ('share/' + package_name + '/meshes/finger_assembly', glob('meshes/finger_assembly/*')),
        ('share/' + package_name + '/meshes', glob('meshes/*.stl')),
        ('share/' + package_name + '/meshes/ft_sensor', glob('meshes/ft_sensor/*')),
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
            'estimate_params        = irb120_control.estimation.estimate_params:main',
            'fbd_plot               = irb120_control.estimation.fbd_plot:main',
            'tst                    = irb120_control.estimation.tst:main',
            'netft_preprocessor     = irb120_control.monitoring.netft_preprocessor:main',
            'camera_hull_recorder   = irb120_control.monitoring.camera_hull_recorder:main',
            'egm_handler            = irb120_control.util.egm_handler:main',
            'calibrate_ft_sensor    = irb120_control.util.calibrate_ft_sensor:main',
            'arc_static             = irb120_control.arc_static:main',
            'arc_static_batch       = irb120_control.arc_static_batch:main',
            'push                   = irb120_control.push:main',
            'keyboard_jog           = irb120_control.keyboard_jog:main',
            'episode                = irb120_control.util.episode:main',
            'run_pipeline           = irb120_control.orchestrator:main',
        ],
    },
)
