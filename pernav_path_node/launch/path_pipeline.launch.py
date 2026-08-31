from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description() -> LaunchDescription:
    package_name = 'pernav_path_node'
    params_file = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'path_pipeline.params.yaml',
    )

    path_pipeline_node = Node(
        package=package_name,
        executable='path_pipeline',
        name='path_pipeline_node',
        output='screen',
        parameters=[params_file],
    )

    return LaunchDescription([
        path_pipeline_node,
    ])
