from setuptools import setup

package_name = 'pernav_world_transform'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='RearAxle-to-world PointCloud2 transformer node.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rearaxle_to_world_node = pernav_world_transform.rearaxle_to_world_node:main',
            'fov_filter_node = pernav_world_transform.fov_filter_node:main',
        ],
    },
)
