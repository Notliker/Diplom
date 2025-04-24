#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_dir = get_package_share_directory('gesture_control')
    webcam_params = os.path.join(package_dir, 'params', 'webcam_params.yaml')
    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            namespace='usb_cam',
            parameters=[webcam_params],
            output='screen'
        ),
        Node(
            package='gesture_control',
            namespace='gesture_control',
            executable='gesture_node',
            name='gesture_node',
            prefix='gnome-terminal --',
            output='screen'
        ),
        Node(
            package='gesture_control',
            namespace='gesture_control',
            executable='visualizer',
            name='visualizer',
            output='screen'
        ),
        Node(
            package='gesture_control',
            namespace='gesture_control',
            executable='processes',
            name='processes',
            prefix='gnome-terminal --',
            output='screen'
        ),
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(package_dir, 'visualize.rviz')],
            output='screen'
        ),
                # MAVROS (упрощённо, без namespace/name и whitelist)
        Node(
            package='mavros',
            executable='mavros_node',
            namespace='mavros',
            parameters=[os.path.join(package_dir, 'params', 'mavros_params.yaml')],
            output='screen',
            arguments=['--ros-args', '--log-level', 'DEBUG'],
        ),

    ])
