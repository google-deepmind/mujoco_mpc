#!/bin/bash
#
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

###############################################################################
# Real Robot Launch Helper Script
#
# Convenience wrapper for launching the real robot with common configurations.
#
# Usage:
#   ./launch_real_robot.sh [OPTIONS]
#
# Options:
#   -r, --robot NAME          Robot name (default: bdx_v4)
#   -p, --policy NAME         Policy name (default: standing)
#   -n, --policy-type TYPE    Policy node type: cpp|py (default: cpp)
#   -f, --freq HZ             Control frequency in Hz (default: 100)
#   -t, --test                Hardware test mode (no controllers)
#   -m, --minimal             Minimal launch (manual controller loading)
#   -v, --rviz                Enable RViz visualization
#   --no-rviz                 Disable RViz visualization
#   -h, --help                Show this help message
#
# Examples:
#   # Full system with C++ policy (production)
#   ./launch_real_robot.sh -p standing
#
#   # Full system with Python policy (development)
#   ./launch_real_robot.sh -p laugh_big -n py
#
#   # Hardware test only (no control)
#   ./launch_real_robot.sh --test
#
#   # Minimal launch for debugging
#   ./launch_real_robot.sh --minimal
#
#   # With RViz visualization
#   ./launch_real_robot.sh -p standing -v
###############################################################################

set -e  # Exit on error

# Default values
ROBOT_NAME="bdx_v4"
POLICY_NAME="standing"
POLICY_TYPE="cpp"
CONTROL_FREQ="100"
USE_RVIZ="false"
LAUNCH_TYPE="full"  # full, test, minimal

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print usage
print_usage() {
    cat << EOF
${BLUE}Real Robot Launch Helper${NC}

${GREEN}Usage:${NC}
    $0 [OPTIONS]

${GREEN}Options:${NC}
    -r, --robot NAME          Robot name (default: bdx_v4)
    -p, --policy NAME         Policy name (default: standing)
    -n, --policy-type TYPE    Policy node type: cpp|py (default: cpp)
    -f, --freq HZ             Control frequency in Hz (default: 100)
    -t, --test                Hardware test mode (no controllers)
    -m, --minimal             Minimal launch (manual controller loading)
    -v, --rviz                Enable RViz visualization
    --no-rviz                 Disable RViz visualization
    -h, --help                Show this help message

${GREEN}Examples:${NC}
    # Full system with C++ policy (production)
    $0 -p standing

    # Full system with Python policy (development)
    $0 -p laugh_big -n py

    # Hardware test only (no control)
    $0 --test

    # Minimal launch for debugging
    $0 --minimal

    # With RViz visualization
    $0 -p standing -v

${GREEN}Quick Tests:${NC}
    # Check IMU data
    ros2 topic echo /imu

    # Check joint states
    ros2 topic echo /joint_states

    # Check controller status
    ros2 control list_controllers

    # Check hardware interfaces
    ros2 control list_hardware_interfaces
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--robot)
            ROBOT_NAME="$2"
            shift 2
            ;;
        -p|--policy)
            POLICY_NAME="$2"
            shift 2
            ;;
        -n|--policy-type)
            POLICY_TYPE="$2"
            if [[ ! "$POLICY_TYPE" =~ ^(cpp|py)$ ]]; then
                print_error "Invalid policy type: $POLICY_TYPE. Must be 'cpp' or 'py'"
                exit 1
            fi
            shift 2
            ;;
        -f|--freq)
            CONTROL_FREQ="$2"
            shift 2
            ;;
        -t|--test)
            LAUNCH_TYPE="test"
            shift
            ;;
        -m|--minimal)
            LAUNCH_TYPE="minimal"
            shift
            ;;
        -v|--rviz)
            USE_RVIZ="true"
            shift
            ;;
        --no-rviz)
            USE_RVIZ="false"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if ROS workspace is sourced
if [ -z "$ROS_DISTRO" ]; then
    print_error "ROS environment not sourced!"
    print_info "Please source your ROS workspace first:"
    print_info "  source /opt/ros/humble/setup.bash"
    print_info "  source install/setup.bash"
    exit 1
fi

# Print launch configuration
print_info "Launch Configuration:"
print_info "  Robot:        ${ROBOT_NAME}"
print_info "  Launch Type:  ${LAUNCH_TYPE}"
print_info "  Policy:       ${POLICY_NAME}"
print_info "  Policy Type:  ${POLICY_TYPE}"
print_info "  Frequency:    ${CONTROL_FREQ} Hz"
print_info "  RViz:         ${USE_RVIZ}"
echo ""

# Safety warnings
if [ "$LAUNCH_TYPE" == "full" ]; then
    print_warning "Starting FULL system with active control!"
    print_warning "Ensure robot is safely mounted and clear of obstacles."
    print_warning "Keep emergency stop accessible at all times."
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Launch cancelled."
        exit 0
    fi
elif [ "$LAUNCH_TYPE" == "test" ]; then
    print_info "Starting HARDWARE TEST mode (no active control)"
    print_info "Use this to verify motor mappings, IMU, and UDP communication"
elif [ "$LAUNCH_TYPE" == "minimal" ]; then
    print_info "Starting MINIMAL launch (manual controller loading)"
    print_info "Controllers must be loaded manually via CLI"
fi

echo ""

# Launch based on type
case $LAUNCH_TYPE in
    full)
        print_info "Launching full system..."
        ros2 launch robot_description real_robot.launch.py \
            robot:="${ROBOT_NAME}" \
            policy_name:="${POLICY_NAME}" \
            policy_node:="${POLICY_TYPE}" \
            control_frequency:="${CONTROL_FREQ}" \
            use_rviz:="${USE_RVIZ}"
        ;;
    test)
        print_info "Launching hardware test..."
        ros2 launch robot_description test_hardware.launch.py \
            robot:="${ROBOT_NAME}" \
            use_rviz:="${USE_RVIZ}"
        ;;
    minimal)
        print_info "Launching minimal system..."
        ros2 launch robot_description real_robot_minimal.launch.py \
            robot:="${ROBOT_NAME}"
        ;;
esac
