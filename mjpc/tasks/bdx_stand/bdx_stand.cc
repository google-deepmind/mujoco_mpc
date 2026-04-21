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

#include "mjpc/tasks/bdx_stand/bdx_stand.h"

#include <algorithm>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {

std::string BDXStand::XmlPath() const {
  return GetModelPath("bdx_stand/task.xml");
}

std::string BDXStand::Name() const { return "BDX Stand"; }

// ------- Residuals for BDX Stand task ------------
//     Residual(0): height - feet height
//     Residual(1): feet height penalty
//     Residual(2): balance
//     Residual(3-4): center of mass xy velocity
//     Residual(5): ctrl
//     Residual(6): upright
//     Residual(7): joint velocity
// -------------------------------------------
void BDXStand::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  // start counter
  int counter = 0;

  // ----- sensors ------ //
  double* left_foot = SensorByName(model, data, "stand_left_foot_pos");
  double* right_foot = SensorByName(model, data, "stand_right_foot_pos");
  double* base_pos = SensorByName(model, data, "base_pos");
  double* com = SensorByName(model, data, "stand_com");
  double* com_vel = SensorByName(model, data, "stand_com_vel");
  double* torso_up = SensorByName(model, data, "stand_torso_up");

  // ----- Height ----- //
  double foot_avg_z = 0.5 * (left_foot[2] + right_foot[2]);
  double height_err = base_pos[2] - foot_avg_z - parameters_[0];
  residual[counter++] = height_err;

  // ----- Feet Height Penalty ----- //
  double feet_height_err = std::max(0.0, foot_avg_z);
  residual[counter++] = 10.0 * feet_height_err;

  // ----- Balance: capture point (XY plane) ----- //
  double capture_point[2] = {
    com[0] + com_vel[0] * 0.2,
    com[1] + com_vel[1] * 0.2
  };

  double foot_center[2] = {
    0.5 * (left_foot[0] + right_foot[0]),
    0.5 * (left_foot[1] + right_foot[1])
  };

  double balance_err[2] = {
    foot_center[0] - capture_point[0],
    foot_center[1] - capture_point[1]
  };
  residual[counter++] = mju_norm(balance_err, 2);

  // ----- COM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], com_vel, 2);
  counter += 2;

  // ----- Control ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // ----- Upright ----- //
  residual[counter++] = torso_up[2] - 1.0;

  // ----- Joint velocity ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
  counter += model->nv - 6;

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

void BDXStand::TransitionLocked(mjModel* model, mjData* d) {
  // set height goal from custom parameter if available
  if (d->time < 1e-6) {
    // parameters[0] is set from XML <custom> residual_Height Goal
    // default is already loaded by agent, no-op here
    parameters[0] = 0.287;
  }
}

}  // namespace mjpc
