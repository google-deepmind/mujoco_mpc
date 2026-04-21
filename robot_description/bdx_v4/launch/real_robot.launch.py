# Copyright 2025 tonly_robot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Real Robot Launch File

Launches complete system for real robot deployment:
- Robot description (URDF/Xacro)
- Controller manager with hardware_interface_real
- joint_state_broadcaster (for /joint_states)
- joint_pd_controller_real (main controller)
- Policy inference (C++ or Python)
- Optional: RViz2 for visualization

Usage:
    ros2 launch robot_description real_robot.launch.py robot:=bdx_v4 policy_node:=cpp
    ros2 launch robot_description real_robot.launch.py robot:=bdx_v4 policy_node:=py use_rviz:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, OpaqueFunction
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import LaunchConfigurationEquals
import os
import xacro


def process_xacro(context):
    """Process Xacro file to generate robot description."""
    robot_description_path = FindPackageShare('robot_description')
    robot_name = LaunchConfiguration('robot').perform(context)

    xacro_file = PathJoinSubstitution([
        robot_description_path,
        'robots',
        robot_name,
        'urdf',
        f'{robot_name}.real.xacro'
    ]).perform(context)

    if not os.path.exists(xacro_file):
        raise FileNotFoundError(f"URDF file not found: {xacro_file}")

    robot_description_config = xacro.process_file(xacro_file).toxml()
    return {'robot_description': robot_description_config}


def generate_launch_description():
    # Declare launch arguments
    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='bdx_v4',
        description='Robot name'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Start RViz2 for visualization'
    )

    policy_node_arg = DeclareLaunchArgument(
        'policy_node',
        default_value='cpp',
        description='Policy inference node type: cpp (production) or py (development)',
        choices=['cpp', 'py']
    )

    policy_name_arg = DeclareLaunchArgument(
        'policy_name',
        default_value='standing',
        description='Policy name to load (standing, laugh_big, etc.)'
    )

    control_freq_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='100.0',
        description='Control frequency in Hz'
    )

    # Path variables
    robot_description_path = FindPackageShare('robot_description')
    robot_name = LaunchConfiguration('robot')

    # Controller configuration
    controller_config_file = PathJoinSubstitution([
        robot_description_path,
        'robots',
        robot_name,
        'config',
        'controller_real.yaml'
    ])

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            # Will be set by OpaqueFunction
            {'use_sim_time': False}  # CRITICAL: Real hardware uses system time
        ],
        output='screen'
    )

    # Controller Manager (spawns hardware_interface_real)
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='controller_manager',
        parameters=[
            controller_config_file,
            {'use_sim_time': False}  # CRITICAL for real hardware
        ],
        output='screen'
    )

    # Joint State Broadcaster (publishes /joint_states)
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster',
                   '--controller-manager-timeout', '50'],
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    # Joint PD Controller (real robot)
    joint_pd_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_pd_controller',
                   '--controller-manager-timeout', '50'],
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    # Policy Inference Node (C++ - production)
    policy_inference_cpp_node = Node(
        package='policy_inference',
        executable='policy_inference_node',
        name='policy_inference',
        parameters=[
            {'use_sim_time': False},
            {'robot_name': robot_name},
            {'policy_name': LaunchConfiguration('policy_name')},
            {'control_frequency': LaunchConfiguration('control_frequency')},
            {'observation_frequency': 1000.0},  # IMU rate
        ],
        output='screen',
        condition=LaunchConfigurationEquals('policy_node', 'cpp')
    )

    # Policy Inference Node (Python - development)
    policy_inference_py_node = Node(
        package='policy_inference_py',
        executable='policy_inference.py',
        name='policy_inference',
        parameters=[{
            'use_sim_time': False,
            'robot_name': robot_name,
            'policy_name': LaunchConfiguration('policy_name'),
            'control_frequency': LaunchConfiguration('control_frequency'),
        }],
        output='screen',
        condition=LaunchConfigurationEquals('policy_node', 'py')
    )

    # RViz2 (optional)
    rviz_config_file = PathJoinSubstitution([
        robot_description_path,
        'robots',
        robot_name,
        'rviz',
        'real_robot.rviz'
    ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': False}],
        condition=LaunchConfigurationEquals('use_rviz', 'true'),
        remappings=[
            ('/joint_states', '/joint_states'),
        ]
    )

    # Event handler: unload controllers on exit (safety)
    unload_controllers_on_exit = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_pd_controller_spawner,
            on_exit=[
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['-u', 'joint_pd_controller'],  # -u for unload
                    parameters=[{'use_sim_time': False}],
                    output='screen'
                )
            ]
        )
    )

    return LaunchDescription([
        # Launch arguments
        robot_arg,
        use_rviz_arg,
        policy_node_arg,
        policy_name_arg,
        control_freq_arg,

        # Core nodes
        robot_state_publisher_node,
        controller_manager_node,

        # Controllers (configure and activate automatically)
        joint_state_broadcaster_spawner,
        joint_pd_controller_spawner,

        # Policy inference (C++ or Python, mutually exclusive)
        policy_inference_cpp_node,
        policy_inference_py_node,

        # Visualization (optional)
        rviz_node,

        # Safety: unload controllers on exit
        unload_controllers_on_exit,
    ])
