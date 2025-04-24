from setuptools import setup
from glob import glob
import os

package_name = 'gesture_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
    ('share/ament_index/resource_index/packages', ['resource/gesture_control']),
    ('share/gesture_control', ['package.xml']),
    ('share/gesture_control/params', ['params/params.yaml']),
    ('share/gesture_control/params', ['params/webcam_params.yaml']),
    ('share/gesture_control/params', ['params/mavros_params.yaml']),
    ('share/gesture_control/launch', ['launch/gesture_control_launch.py']),
    ('share/gesture_control', ['visualize.rviz']),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kira',
    maintainer_email='kira@todo.todo',
    description='Gesture-based quadcopter control with MAVLink',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_node = gesture_control.gesture_node:main',
            'visualizer = gesture_control.visualizer:main',
            'processes = gesture_control.processes:main',
        ],
    },
)