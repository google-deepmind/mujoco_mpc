// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tasks/quadruped/quadruped.h"

#include <mujoco/mujoco.h>
#include "task.h"
#include "utilities.h"

namespace mjpc {

// --------------------- Residuals for quadruped task --------------------
//   Number of residuals: 4
//     Residual (0): position_z - average(foot position)_z - height_goal
//     Residual (1): position - goal_position
//     Residual (2): orientation - goal_orientation
//     Residual (3): control
//   Number of parameters: 1
//     Parameter (1): height_goal
// -----------------------------------------------------------------------
void Quadruped::Residual(const double* parameters, const mjModel* model,
                         const mjData* data, double* residual) {
  // ---------- Residual (0) ----------
  // standing height goal
  double height_goal = parameters[0];

  // system's standing height
  double standing_height = mjpc::SensorByName(model, data, "position")[2];

  // average foot height
  double FRz = mjpc::SensorByName(model, data, "FR")[2];
  double FLz = mjpc::SensorByName(model, data, "FL")[2];
  double RRz = mjpc::SensorByName(model, data, "RR")[2];
  double RLz = mjpc::SensorByName(model, data, "RL")[2];
  double avg_foot_height = 0.25 * (FRz + FLz + RRz + RLz);

  residual[0] = (standing_height - avg_foot_height) - height_goal;

  // ---------- Residual (1) ----------
  // goal position
  const double* goal_position = data->mocap_pos;

  // system's position
  double* position = mjpc::SensorByName(model, data, "position");

  // position error
  mju_sub3(residual + 1, position, goal_position);

  // ---------- Residual (2) ----------
  // goal orientation
  double goal_rotmat[9];
  const double* goal_orientation = data->mocap_quat;
  mju_quat2Mat(goal_rotmat, goal_orientation);

  // system's orientation
  double body_rotmat[9];
  double* orientation = mjpc::SensorByName(model, data, "orientation");
  mju_quat2Mat(body_rotmat, orientation);

  mju_sub(residual + 4, body_rotmat, goal_rotmat, 9);

  // ---------- Residual (3) ----------
  mju_copy(residual + 13, data->ctrl, model->nu);
}

// -------- Transition for quadruped task --------
//   If quadruped is within tolerance of goal ->
//   set goal to next from keyframes.
// -----------------------------------------------
int Quadruped::Transition(int state, const mjModel* model, mjData* data,
                          Task* task) {
  int new_state = state;

  // ---------- Compute tolerance ----------
  // goal position
  const double* goal_position = data->mocap_pos;

  // goal orientation
  const double* goal_orientation = data->mocap_quat;

  // system's position
  double* position = mjpc::SensorByName(model, data, "position");

  // system's orientation
  double* orientation = mjpc::SensorByName(model, data, "orientation");

  // position error
  double position_error[3];
  mju_sub3(position_error, position, goal_position);
  double position_error_norm = mju_norm3(position_error);

  // orientation error
  double geodesic_distance =
      1.0 - mju_abs(mju_dot(goal_orientation, orientation, 4));

  // ---------- Check tolerance ----------
  double tolerance = 1.5e-1;
  if (position_error_norm <= tolerance && geodesic_distance <= tolerance) {
    // update task state
    new_state += 1;
    if (new_state == model->nkey) {
      new_state = 0;
    }
  }

  // ---------- Set goal ----------
  mju_copy3(data->mocap_pos, model->key_mpos + 3 * new_state);
  mju_copy4(data->mocap_quat, model->key_mquat + 4 * new_state);

  return new_state;
}

void Quadruped::ResidualFloor(const double* parameters, const mjModel* model,
                              const mjData* data, double* residual) {
  int counter = 0;
  // ---------- Upright ----------

  // torso z vector shoulf be [0 0 1]
  double* upright = mjpc::SensorByName(model, data, "torso_up");

  residual[counter] = upright[0];
  counter += 1;
  residual[counter] = upright[1];
  counter += 1;
  residual[counter] = upright[2] - 1;
  counter += 1;

  // ---------- Velocity ----------

  // CoM linear velocity, in the forward direction, should equal velocity_goal
  double* linvel = mjpc::SensorByName(model, data, "torso_subtreelinvel");
  double linvel_ego[3];
  int torso_id = mj_name2id(model, mjOBJ_XBODY, "trunk");
  mju_rotVecMatT(linvel_ego, linvel, data->xmat+9*torso_id);

  double velocity_goal = parameters[0];
  residual[counter] = linvel_ego[0] - velocity_goal;
  counter += 1;

  // foot average velocity, in the forward direction, should equal CoM velocity
  double* FRvel = mjpc::SensorByName(model, data, "FRvel");
  double* FLvel = mjpc::SensorByName(model, data, "FLvel");
  double* RRvel = mjpc::SensorByName(model, data, "RRvel");
  double* RLvel = mjpc::SensorByName(model, data, "RLvel");

  // average foot velocity
  double foot_vel[3] = {0};
  mju_add3(foot_vel, FRvel, FLvel);
  mju_addTo3(foot_vel, RRvel);
  mju_addTo3(foot_vel, RLvel);
  mju_scl3(foot_vel, foot_vel, 0.25);
  // in torso frame
  mju_rotVecMatT(foot_vel, foot_vel, data->xmat+9*torso_id);

  residual[counter] = foot_vel[0] - velocity_goal;
  counter += 1;


  // ---------- Yaw ----------

  // CoM linear velocity, in the torso frame
  double* torso_forward = mjpc::SensorByName(model, data, "torso_forward");
  double torso_heading[2] = {torso_forward[0], torso_forward[1]};
  mju_normalize(torso_heading, 2);

  double heading_goal = parameters[1];
  residual[counter] = torso_heading[0] - mju_cos(heading_goal);
  counter += 1;
  residual[counter] = torso_heading[1] - mju_sin(heading_goal);
  counter += 1;

  // ---------- Control ----------
  mju_copy(residual + counter, data->ctrl, model->nu);
  counter += model->nu;

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i=0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i("mismatch between total user-sensor dimension "
                "and actual length of residual %d", counter);
  }
}

}  // namespace mjpc
