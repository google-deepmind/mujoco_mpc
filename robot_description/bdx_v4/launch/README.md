# Real Robot Launch Files

Complete launch system for real robot deployment with safety, testing, and development modes.

## Overview

The real robot launch system provides three launch modes for different use cases:

1. **`real_robot.launch.py`** - Full system deployment with active control
2. **`test_hardware.launch.py`** - Hardware testing without control loops
3. **`real_robot_minimal.launch.py`** - Core hardware interface only

All launch files use `hardware_interface_real` for UDP communication with the STM32H723 comms-board (192.168.2.131:55151).

## Quick Start

### Using Helper Script (Recommended)

```bash
cd src/robot_description/robots/bdx_v4/launch

# Full system with standing policy (production)
./launch_real_robot.sh -p standing

# Full system with Python policy (development)
./launch_real_robot.sh -p laugh_big -n py

# Hardware test only (verify motors, IMU, UDP)
./launch_real_robot.sh --test

# Minimal launch (manual controller loading)
./launch_real_robot.sh --minimal
```

### Direct ROS2 Launch

```bash
# Source workspace
source install/setup.bash

# Full system
ros2 launch robot_description real_robot.launch.py robot:=bdx_v4 policy_name:=standing policy_node:=cpp

# Hardware test
ros2 launch robot_description test_hardware.launch.py robot:=bdx_v4

# Minimal
ros2 launch robot_description real_robot_minimal.launch.py robot:=bdx_v4
```

## Launch Files

### 1. real_robot.launch.py (Full System)

**Purpose:** Complete real robot deployment with active control loops.

**Components:**
- Robot description (URDF/Xacro)
- Controller manager with `hardware_interface_real`
- `joint_state_broadcaster` (publishes `/joint_states`)
- `joint_pd_controller` (main control loop)
- Policy inference node (C++ or Python)
- Optional: RViz2 visualization

**Launch Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `robot` | `bdx_v4` | Robot name |
| `policy_name` | `standing` | Policy to load (standing, laugh_big, etc.) |
| `policy_node` | `cpp` | Policy node: `cpp` (production) or `py` (development) |
| `control_frequency` | `100.0` | Control frequency in Hz |
| `use_rviz` | `true` | Enable RViz2 visualization |

**Examples:**

```bash
# Production (C++ policy, standing)
ros2 launch robot_description real_robot.launch.py

# Development (Python policy, custom policy)
ros2 launch robot_description real_robot.launch.py \
    policy_name:=laugh_big \
    policy_node:=py

# Custom control frequency
ros2 launch robot_description real_robot.launch.py \
    control_frequency:=200.0

# Without RViz (headless)
ros2 launch robot_description real_robot.launch.py \
    use_rviz:=false
```

**Safety Features:**
- Controllers unload automatically on exit
- `use_sim_time:=false` enforced for real hardware
- Hardware interface validates motor mappings on startup

---

### 2. test_hardware.launch.py (Hardware Test)

**Purpose:** Test hardware interface without enabling control loops. Safe for verification and diagnostics.

**Components:**
- Robot description
- Controller manager (hardware interface only)
- `joint_state_broadcaster` (read-only)
- Optional: RViz2 visualization
- Optional: rqt_console for log monitoring

**Use Cases:**
- Verify motor mappings and configuration
- Check IMU data quality and rate
- Test UDP communication to STM32
- Monitor motor temperatures and errors
- Debug hardware interface issues

**Launch Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `robot` | `bdx_v4` | Robot name |
| `use_rviz` | `false` | Enable RViz2 visualization |
| `diagnostic_frequency` | `1.0` | Diagnostic frequency (Hz) |

**Examples:**

```bash
# Basic hardware test
ros2 launch robot_description test_hardware.launch.py

# With RViz visualization
ros2 launch robot_description test_hardware.launch.py use_rviz:=true

# During testing, monitor topics:
ros2 topic echo /imu              # IMU data
ros2 topic echo /joint_states      # Joint positions
ros2 control list_hardware_interfaces  # Hardware status
```

**What to Check:**

1. **UDP Connection:** Look for "Successfully configured! Robot IP: 192.168.2.131"
2. **IMU Data:** Verify `/imu` topic publishes at ~400 Hz (quaternion rate)
3. **Joint States:** Check all 14 joints report positions
4. **Hardware Interfaces:** Confirm all joints are in "active" state

---

### 3. real_robot_minimal.launch.py (Minimal)

**Purpose:** Core hardware interface only. No auto-loaded controllers. For manual controller development and debugging.

**Components:**
- Robot description
- Controller manager with `hardware_interface_real`
- NO auto-loaded controllers

**Use Cases:**
- Manual controller loading via CLI
- Debugging custom controllers
- Testing controller configurations
- Development without full system

**Launch Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `robot` | `bdx_v4` | Robot name |
| `verbose` | `false` | Enable verbose output |

**Examples:**

```bash
# Launch minimal system
ros2 launch robot_description real_robot_minimal.launch.py

# In another terminal, manually manage controllers:
ros2 control list_hardware_interfaces
ros2 control list_controllers

# Load joint_state_broadcaster
ros2 run controller_manager spawner joint_state_broadcaster

# Load joint_pd_controller
ros2 run controller_manager spawner joint_pd_controller

# Check status
ros2 control list_controllers
```

**Manual Controller Management:**

```bash
# Configure controller (prepare for activation)
ros2 control configure_controller joint_pd_controller

# Activate controller (start control loop)
ros2 control activate_controller joint_pd_controller

# Deactivate controller (stop control loop)
ros2 control deactivate_controller joint_pd_controller
```

---

## Helper Script: launch_real_robot.sh

Convenience wrapper with common configurations and safety checks.

**Usage:**

```bash
./launch_real_robot.sh [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-r, --robot NAME` | Robot name (default: bdx_v4) |
| `-p, --policy NAME` | Policy name (default: standing) |
| `-n, --policy-type TYPE` | Policy node: `cpp` or `py` (default: cpp) |
| `-f, --freq HZ` | Control frequency in Hz (default: 100) |
| `-t, --test` | Hardware test mode (no controllers) |
| `-m, --minimal` | Minimal launch (manual loading) |
| `-v, --rviz` | Enable RViz |
| `--no-rviz` | Disable RViz |
| `-h, --help` | Show help message |

**Examples:**

```bash
# Full system with standing policy
./launch_real_robot.sh -p standing

# Full system with Python policy
./launch_real_robot.sh -p laugh_big -n py

# Hardware test
./launch_real_robot.sh --test

# Minimal launch
./launch_real_robot.sh --minimal

# With RViz
./launch_real_robot.sh -p standing -v
```

**Safety Features:**
- Interactive confirmation before starting full system
- ROS environment checking
- Colored output for readability
- Quick test command reference

---

## Configuration Files

### controller_real.yaml

Located at: `src/robot_description/robots/bdx_v4/config/controller_real.yaml`

**Key Differences from Simulation:**
- `use_sim_time: false` (real hardware time)
- Lower `max_effort` limits (safety)
- No `initial_position` parameter (reads actual position)

**Example:**

```yaml
controller_manager:
  ros__parameters:
    use_sim_time: false  # CRITICAL for real hardware
    update_rate: 500     # Hz

joint_state_broadcaster:
  ros__parameters:
    publish_rate: 500.0

joint_pd_controller:
  ros__parameters:
    joints:
      - left_hip_yaw_joint
      - left_hip_roll_joint
      # ... all 14 joints
    command_interfaces:
      - effort
    state_interfaces:
      - position
      - velocity
      - effort
    max_effort: [50.0, 50.0, ...]  # Safety limits
```

---

## Troubleshooting

### 1. UDP Connection Fails

**Error:** `Failed to initialize UDP communication`

**Solutions:**
```bash
# Check network connectivity
ping 192.168.2.131

# Verify RTC IP configuration
ifconfig
# Should show RTC IP: 192.168.2.1

# Check firewall
sudo ufw status
sudo ufw allow 55151/udp

# Verify STM32 is powered on and running
```

### 2. Controllers Fail to Load

**Error:** `Controller manager not ready`

**Solutions:**
```bash
# Check controller manager status
ros2 control list_controllers

# Check hardware interfaces
ros2 control list_hardware_interfaces

# Verify controller config file
cat src/robot_description/robots/bdx_v4/config/controller_real.yaml
```

### 3. IMU Data Not Publishing

**Error:** `/imu` topic has no messages

**Solutions:**
```bash
# Check if IMU publisher is created
ros2 topic list | grep imu

# Verify IMU frame ID in URDF
ros2 launch robot_description test_hardware.launch.py use_rviz:=false

# Check hardware logs
ros2 topic echo /rosout | grep -i imu
```

### 4. Motor Sequence Errors

**Error:** `Too many sequence errors, entering safe state`

**Solutions:**
```bash
# Check for packet loss
# Reduce command rate if needed

# Verify motor ID mappings in URDF
# Check: src/robot_description/robots/bdx_v4/urdf/ros2_control_real.xacro

# Test with minimal launch first
ros2 launch robot_description real_robot_minimal.launch.py
```

### 5. Wrong Time Source

**Error:** `use_sim_time is true, should be false for real hardware`

**Solution:**
All real robot launch files enforce `use_sim_time: false`. If you see this error, ensure you're using the real robot launch files (not Gazebo/MuJoCo).

---

## Common Workflows

### Workflow 1: First-Time Hardware Verification

```bash
# 1. Test hardware (no control)
ros2 launch robot_description test_hardware.launch.py

# 2. In another terminal, check IMU data
ros2 topic echo /imu

# 3. Check joint states
ros2 topic echo /joint_states

# 4. Verify hardware interfaces
ros2 control list_hardware_interfaces
```

### Workflow 2: Development with Python Policy

```bash
# 1. Launch with Python policy node
./launch_real_robot.sh -p standing -n py -v

# 2. Monitor policy commands
ros2 topic echo /joint_pd_controller/commands

# 3. Check controller performance
ros2 topic hz /joint_states

# 4. View in RViz
# (RViz launches automatically with -v flag)
```

### Workflow 3: Production Deployment

```bash
# 1. Ensure robot is safely mounted
# 2. Clear area around robot
# 3. Have emergency stop ready

# 4. Launch full system
./launch_real_robot.sh -p standing

# 5. Monitor system health
ros2 topic hz /imu
ros2 topic hz /joint_states
ros2 control list_controllers
```

### Workflow 4: Debugging Controller Issues

```bash
# 1. Launch minimal system
ros2 launch robot_description real_robot_minimal.launch.py

# 2. Load joint_state_broadcaster only
ros2 run controller_manager spawner joint_state_broadcaster

# 3. Check joint states are publishing
ros2 topic echo /joint_states

# 4. Load controller manually
ros2 run controller_manager spawner joint_pd_controller

# 5. Configure and activate
ros2 control configure_controller joint_pd_controller
ros2 control activate_controller joint_pd_controller
```

---

## Hardware Safety Checklist

Before launching full system with active control:

- [ ] Robot is securely mounted to test stand
- [ ] Area around robot is clear of obstacles and people
- [ ] Emergency stop button is accessible
- [ ] UDP connection to STM32 is verified (`ping 192.168.2.131`)
- [ ] IMU data is valid (check with test_hardware.launch.py)
- [ ] All motors respond correctly (check joint states)
- [ ] Controller configuration is correct (check YAML)
- [ ] Policy file exists in `src/policy_assets/`

---

## RViz Configuration

RViz config file: `src/robot_description/robots/bdx_v4/rviz/real_robot.rviz`

**Displays:**
- Robot model (visual only)
- Grid (XY plane, reference: base_link)
- TF tree (all transforms)

**View:**
- Orbit camera at 2.5m distance
- Focal point: (0, 0, 0.5)
- Pitch: 0.5 rad, Yaw: 0.785 rad

**Customization:**
```bash
# Launch with custom RViz config
ros2 launch robot_description real_robot.launch.py \
    use_rviz:=true

# In RViz: File → Save Config As...
```

---

## Reference Documentation

- **Main README:** `../../../../../README.md`
- **Comms Board Protocol:** `../../../../../docs/comms-board-protocol.md`
- **Hardware Interface:** `../../../../../src/hardware_interfaces/README.md`
- **Motor Protocols:** `../../../../../src/comms_board_protocol/README.md`
- **Controller Configuration:** `../config/controller_real.yaml`
- **Motor Mappings:** `../urdf/ros2_control_real.xacro`
