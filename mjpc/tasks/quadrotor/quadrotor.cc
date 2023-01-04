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

#include "tasks/quadrotor/quadrotor.h"

#include <mujoco/mujoco.h>
#include "task.h"
#include "utilities.h"

namespace mjpc {

// --------------- Residuals for quadrotor task ---------------
//   Number of residuals: 5
//     Residual (0): position - goal position
//     Residual (1): orientation - goal orientation
//     Residual (2): linear velocity - goal linear velocity
//     Residual (3): angular velocity - goal angular velocity
//     Residual (4): control
//   Number of parameters: 6
// ------------------------------------------------------------
void Quadrotor::Residual(const double* parameters, const mjModel* model,
                         const mjData* data, double* residuals) {
  // ---------- Residual (0) ----------
  double* position = mjpc::SensorByName(model, data, "position");
  mju_sub(residuals, position, data->mocap_pos, 3);

  // ---------- Residual (1) ----------
  double quadrotor_mat[9];
  double* orientation = mjpc::SensorByName(model, data, "orientation");
  mju_quat2Mat(quadrotor_mat, orientation);

  double goal_mat[9];
  mju_quat2Mat(goal_mat, data->mocap_quat);

  mju_sub(residuals + 3, quadrotor_mat, goal_mat, 9);

  // ---------- Residual (2) ----------
  double* linear_velocity = mjpc::SensorByName(model, data, "linear_velocity");
  mju_sub(residuals + 12, linear_velocity, parameters, 3);

  // ---------- Residual (3) ----------
  double* angular_velocity =
      mjpc::SensorByName(model, data, "angular_velocity");
  mju_sub(residuals + 15, angular_velocity, parameters + 3, 3);

  // ---------- Residual (4) ----------
  mju_copy(residuals + 18, data->ctrl, model->nu);
}

// ----- Transition for quadrotor task -----
int Quadrotor::Transition(int state, const mjModel* model, mjData* data,
                          Task* task) {
  int new_state = state;

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

  double tolerance = 5.0e-1;
  if (position_error_norm <= tolerance && geodesic_distance <= tolerance) {
    // update task state
    new_state += 1;
    if (new_state == model->nkey) {
      new_state = 0;
    }
  }

  // set goal
  mju_copy3(data->mocap_pos, model->key_mpos + 3 * new_state);
  mju_copy4(data->mocap_quat, model->key_mquat + 4 * new_state);

  return new_state;
}

}  // namespace mjpc
