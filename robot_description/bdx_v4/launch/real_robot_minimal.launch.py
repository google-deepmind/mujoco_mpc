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
Minimal Real Robot Launch File

Launches ONLY the core hardware interface without auto-loading any controllers.
Use this for manual controller development, debugging, and testing.

Use cases:
- Manual controller loading via CLI
- Debugging hardware interface issues
- Testing custom controllers
- Development without full system startup

Usage:
    # Launch minimal system
    ros2 launch robot_description real_robot_minimal.launch.py robot:=bdx_v4

    # In another terminal, manually load controllers:
    ros2 control list_hardware_interfaces
    ros2 control list_controllers

    # Load joint_state_broadcaster manually
    ros2 run controller_manager spawner joint_state_broadcaster

    # Load your custom controller
    ros2 run controller_manager spawner my_custom_controller

    # Configure and activate controller
    ros2 control configure_controller joint_pd_controller
    ros2 control activate_controller joint_pd_controller
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
import xacro


def generate_launch_description():
    # Declare launch arguments
    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='bdx_v4',
        description='Robot name'
    )

    verbose_arg = DeclareLaunchArgument(
        'verbose',
        default_value='false',
        description='Enable verbose output from hardware interface'
    )

    # Path variables
    robot_description_path = FindPackageShare('robot_description')
    robot_name = LaunchConfiguration('robot')

    # Process URDF/Xacro
    xacro_file = PathJoinSubstitution([
        robot_description_path,
        'robots',
        robot_name,
        'urdf',
        f'{robot_name}.real.xacro'
    ])

    # Process xacro file
    # Note: In a real launch, this would be processed at runtime
    # For simplicity, we're using a placeholder that should be replaced
    # with actual xacro processing

    # Controller configuration (minimal - no auto-loaded controllers)
    controller_config_file = PathJoinSubstitution([
        robot_description_path,
        'robots',
        robot_name,
        'config',
        'controller_real.yaml'
    ])

    # Robot State Publisher (minimal, for TF tree)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': False},  # CRITICAL: Real hardware uses system time
            {'publish_frequency': 50.0},  # Hz
        ],
        output='screen'
    )

    # Controller Manager with hardware_interface_real
    # IMPORTANT: No controllers are auto-loaded!
    # Controllers must be loaded manually via CLI or spawner nodes
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='controller_manager',
        parameters=[
            controller_config_file,
            {'use_sim_time': False},
            {'controller_manager_update_rate': 500},  # Hz (hardware interface read/write rate)
        ],
        output='screen',
        # Add verbose output if requested
        arguments=['--ros-args', '--log-level', 'info'] if LaunchConfiguration('verbose').perform(None) == 'true' else []
    )

    return LaunchDescription([
        # Launch arguments
        robot_arg,
        verbose_arg,

        # Core nodes ONLY
        robot_state_publisher_node,
        controller_manager_node,

        # NOTE: No controller spawners!
        # Controllers must be loaded manually:
        # ros2 run controller_manager spawner joint_state_broadcaster
        # ros2 run controller_manager spawner joint_pd_controller
    ])


# Convenience function for manual controller loading examples
def print_manual_instructions():
    """
    Print instructions for manual controller loading.
    Call this after launching the minimal system.
    """
    print("\n" + "="*70)
    print("MINIMAL REAL ROBOT LAUNCH - MANUAL CONTROLLER LOADING")
    print("="*70)
    print("\n1. Check hardware interface status:")
    print("   ros2 control list_hardware_interfaces")
    print("\n2. List available controllers:")
    print("   ros2 control list_controllers")
    print("\n3. Load and configure joint_state_broadcaster:")
    print("   ros2 run controller_manager spawner joint_state_broadcaster")
    print("\n4. Load joint_pd_controller:")
    print("   ros2 run controller_manager spawner joint_pd_controller")
    print("\n5. Check controller status:")
    print("   ros2 control list_controllers")
    print("\n6. View joint states:")
    print("   ros2 topic echo /joint_states")
    print("\n7. View IMU data:")
    print("   ros2 topic echo /imu")
    print("\n8. Configure controller (if inactive):")
    print("   ros2 control configure_controller joint_pd_controller")
    print("\n9. Activate controller (if configured but inactive):")
    print("   ros2 control activate_controller joint_pd_controller")
    print("\n10. Deactivate controller:")
    print("    ros2 control deactivate_controller joint_pd_controller")
    print("\n" + "="*70)
    print("\nFor safety, motors will NOT receive commands until both")
    print("configure_controller AND activate_controller are called.")
    print("="*70 + "\n")
