import os

from setuptools import find_packages, setup

package_name = 'irb120_models'


def tree_data_files(*tops):
    """Install each tree under share/, keeping the folder layout package:// URIs use."""
    entries = []
    for top in tops:
        for root, _, files in os.walk(top):
            if files:
                entries.append((os.path.join('share', package_name, root),
                                [os.path.join(root, f) for f in files]))
    return entries


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        *tree_data_files('urdf', 'meshes', 'objects'),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hylander2126',
    maintainer_email='stevenhyland1@gmail.com',
    description='IRB120 robot description and ground-truth object models.',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'object_models = irb120_models.cli:main',
        ],
    },
)
