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

#include "tasks/walker/walker.h"

#include <mujoco/mujoco.h>
#include "utilities.h"

namespace mjpc {
// --------- Residuals for walker task --------
//   Number of residuals: 4
//     Residual (0): control
//     Residual (1): position_z - height_goal
//     Residual (2): body_z_axis - 1.0
//     Residual (3): velocity_x - speed_goal
//   Parameters: 2
//     Parameter (0): height_goal
//     Parameter (1): speed_goal
// --------------------------------------------
void Walker::Residual(const double* parameters, const mjModel* model,
                      const mjData* data, double* residual) {
  int counter = 0;
  // ---------- Residual (0) ----------
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // ---------- Residual (1) -----------
  double height = mjpc::SensorByName(model, data, "torso_position")[2];
  residual[counter++] = height - parameters[0];

  // ---------- Residual (2) ----------
  double torso_up = mjpc::SensorByName(model, data, "torso_zaxis")[2];
  residual[counter++] = torso_up - 1.0;

  // ---------- Residual (3) ----------
  double com_vel = mjpc::SensorByName(model, data, "torso_subtreelinvel")[0];
  residual[counter++] = com_vel - parameters[1];

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
