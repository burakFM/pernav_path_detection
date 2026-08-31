from setuptools import setup

package_name = 'pernav_path_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/path_pipeline.params.yaml']),
        ('share/' + package_name + '/launch', ['launch/path_pipeline.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Simple ROS2 Python pub/sub package for bring-up checks',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_publisher = pernav_path_node.simple_publisher:main',
            'path_pipeline = pernav_path_node.path_pipeline_node:main',
        ],
    },
)
