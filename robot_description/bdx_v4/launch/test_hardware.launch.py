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
Hardware Test Launch File

Launches hardware interface in read-only mode for testing and diagnostics.
No control loops are activated, making it safe for hardware verification.

Use cases:
- Verify motor mappings and configuration
- Check IMU data quality
- Test UDP communication
- Monitor motor temperatures and errors
- Debug hardware interface issues

Usage:
    ros2 launch robot_description test_hardware.launch.py robot:=bdx_v4
    ros2 launch robot_description test_hardware.launch.py robot:=bdx_v4 use_rviz:=true

During testing:
    ros2 topic echo /imu              # Check IMU data
    ros2 topic echo /joint_states      # Check joint positions
    ros2 control list_hardware_interfaces  # Check hardware status
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import LaunchConfigurationEquals
import os


def generate_launch_description():
    # Declare launch arguments
    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='bdx_v4',
        description='Robot name'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Start RViz2 for visualization'
    )

    diagnostic_freq_arg = DeclareLaunchArgument(
        'diagnostic_frequency',
        default_value='1.0',
        description='Hardware diagnostic publishing frequency (Hz)'
    )

    # Path variables
    robot_description_path = FindPackageShare('robot_description')
    robot_name = LaunchConfiguration('robot')

    # Controller configuration (real hardware)
    controller_config_file = PathJoinSubstitution([
        robot_description_path,
        'robots',
        robot_name,
        'config',
        'controller_real.yaml'
    ])

    # RViz config
    rviz_config_file = PathJoinSubstitution([
        robot_description_path,
        'robots',
        robot_name,
        'rviz',
        'real_robot.rviz'
    ])

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': LaunchConfiguration('robot_description_xml')},
            {'use_sim_time': False}  # CRITICAL: Real hardware uses system time
        ],
        output='screen'
    )

    # Controller Manager (hardware interface only, no active controllers)
    # Note: hardware_interface_real will configure but NOT activate control loops
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='controller_manager',
        parameters=[
            controller_config_file,
            {'use_sim_time': False}
        ],
        output='screen'
    )

    # Joint State Broadcaster (READ-ONLY - publishes joint states from hardware)
    # This allows monitoring joint positions without enabling control
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager-timeout', '50'
        ],
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    # Hardware Diagnostics Node (monitors hardware health)
    # Note: This would be a custom node that publishes diagnostic information
    # For now, we'll use rqt_console to monitor logs
    diagnostics_monitor_node = Node(
        package='rqt_console',
        executable='rqt_console',
        name='diagnostics_monitor',
        condition=LaunchConfigurationEquals('use_rviz', 'false')
    )

    # RViz2 (optional - for visualization)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[
            {'use_sim_time': False}
        ],
        condition=LaunchConfigurationEquals('use_rviz', 'true'),
        remappings=[
            ('/joint_states', '/joint_states'),
        ]
    )

    return LaunchDescription([
        # Launch arguments
        robot_arg,
        use_rviz_arg,
        diagnostic_freq_arg,

        # NOTE: robot_description_xml must be passed as argument
        # Usage: ros2 launch ... robot_description_xml:=$(xacro ...)

        # Core nodes
        robot_state_publisher_node,
        controller_manager_node,

        # Read-only joint state broadcaster
        joint_state_broadcaster_spawner,

        # Diagnostics/visualization
        diagnostics_monitor_node,
        rviz_node,
    ])


# Helper function to generate robot description from command line
def generate_launch_description_with_xacro():
    """
    Alternative version that processes Xacro internally.
    Use this if you want to avoid passing robot_description_xml as argument.
    """
    import xacro

    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
    from launch_ros.actions import Node
    from launch_ros.substitutions import FindPackageShare

    # Declare launch arguments
    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value='bdx_v4',
        description='Robot name'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Start RViz2 for visualization'
    )

    # Process Xacro file
    robot_description_path = FindPackageShare('robot_description')
    robot_name = LaunchConfiguration('robot')

    xacro_file = PathJoinSubstitution([
        robot_description_path,
        'robots',
        robot_name,
        'urdf',
        f'{robot_name}.real.xacro'
    ])

    # NOTE: This requires the launch file to be run with a context that
    # can perform substitutions. For simplicity, we recommend passing
    # robot_description_xml as a command-line argument.

    # For actual use, consider using the xacro processing in the main launch file
    # or creating a separate setup script.

    return LaunchDescription([
        robot_arg,
        use_rviz_arg,
        # Add nodes here (same as above but with processed xacro)
    ])
