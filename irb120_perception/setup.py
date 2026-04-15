from glob import glob
from setuptools import find_packages, setup

package_name = 'irb120_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/weights', glob('weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hylander2126',
    maintainer_email='stevenhyland1@gmail.com',
    description='Pointcloud object detection and convex hull extraction for the IRB120 workspace.',
    license='TODO',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'object_detector = irb120_perception.object_detector:main',
        ],
    },
)
