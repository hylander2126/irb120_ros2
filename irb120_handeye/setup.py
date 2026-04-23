from glob import glob

from setuptools import find_packages, setup

package_name = 'irb120_handeye'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/calibrations', glob('calibrations/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hylander2126',
    maintainer_email='stevenhyland1@gmail.com',
    description='Hand-eye calibration tools for the IRB120.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'run_calibration_poses = irb120_handeye.run_calibration_poses:main',
        ],
    },
)
