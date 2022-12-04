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

#include "tasks/humanoid/humanoid.h"

#include <iostream>

#include <mujoco/mujoco.h>
#include "utilities.h"

namespace mjpc {

// ------------------ Residuals for humanoid stand task ------------
//   Number of residuals: 6
//     Residual (0): Desired height
//     Residual (1): Balance: COM_xy - average(feet position)_xy
//     Residual (2): Com Vel: should be 0 and equal feet average vel
//     Residual (3): Control: minimise control
//     Residual (4): Joint vel: minimise joint velocity
//   Number of parameters: 1
//     Parameter (0): height_goal
// ----------------------------------------------------------------
void Humanoid::ResidualStand(const double* parameters, const mjModel* model,
                             const mjData* data, double* residual) {
    // ----- action ----- //
  mju_copy(residual, data->ctrl, model->nu);

  // ----- COM feet xy error ----- //
  // center of mass position
  double* com_position = mjpc::SensorByName(model, data, "torso_subtreecom");

  // feet sensor positions
  double* f1_position = mjpc::SensorByName(model, data, "sp0");
  double* f2_position = mjpc::SensorByName(model, data, "sp1");
  double* f3_position = mjpc::SensorByName(model, data, "sp2");
  double* f4_position = mjpc::SensorByName(model, data, "sp3");

  // average feet xy position
  double fxy_avg[2] = {0.0};
  mju_addTo(fxy_avg, f1_position, 2);
  mju_addTo(fxy_avg, f2_position, 2);
  mju_addTo(fxy_avg, f3_position, 2);
  mju_addTo(fxy_avg, f4_position, 2);
  mju_scl(fxy_avg, fxy_avg, 0.25, 2);

  // center of mass and average feet position xy error
  mju_subFrom(fxy_avg, com_position, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[model->nu] = com_feet_distance;

  // ----- torso COM xy error ----- //
  double* torso_position = mjpc::SensorByName(model, data, "torso_position");
  double torso_com_error[2];
  mju_sub(torso_com_error, torso_position, com_position, 2);
  double torso_com_distance = mju_norm(torso_com_error, 2);
  residual[model->nu + 1] = torso_com_distance;

  // ----- head feet vertical error ----- //
  double* head_position = mjpc::SensorByName(model, data, "head_position");
  double head_feet_error =
      head_position[2] - 0.25 * (f1_position[2] + f2_position[2] +
                                 f3_position[2] + f4_position[2]);
  residual[model->nu + 2] = head_feet_error - parameters[0];

  // ----- COM xy velocity ----- //
  double* com_velocity =
      mjpc::SensorByName(model, data, "torso_subtreelinvel");
  mju_copy(residual + model->nu + 3, com_velocity, 2);
  
  // ----- joint velocity ----- //
  mju_copy(residual + model->nu + 5, data->qvel, model->nv);

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  // int user_sensor_dim = 0;
  // for (int i=0; i < model->nsensor; i++) {
  //   if (model->sensor_type[i] == mjSENS_USER) {
  //     user_sensor_dim += model->sensor_dim[i];
  //   }
  // }
  // if (user_sensor_dim != counter) {
  //   mju_error_i("mismatch between total user-sensor dimension "
  //               "and actual length of residual %d", counter);
  // }
}

}  // namespace mjpc
