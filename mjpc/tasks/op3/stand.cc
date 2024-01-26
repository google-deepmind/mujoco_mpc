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

#include "mjpc/tasks/op3/stand.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {
std::string OP3::XmlPath() const { return GetModelPath("op3/task.xml"); }
std::string OP3::Name() const { return "OP3"; }

// ------- Residuals for OP3 task ------------
//     Residual(0): height - feet height
//     Residual(1): balance
//     Residual(2): center of mass xy velocity
//     Residual(3): ctrl - ctrl_nominal
//     Residual(4): upright
//     Residual(5): joint velocity
// -------------------------------------------
void OP3::ResidualFn::Residual(const mjModel* model, const mjData* data,
                               double* residual) const {
  // start counter
  int counter = 0;

  // get mode
  int mode = current_mode_;

  // ----- sensors ------ //
  double* head_position = SensorByName(model, data, "head_position");
  double* left_foot_position = SensorByName(model, data, "left_foot_position");
  double* right_foot_position =
      SensorByName(model, data, "right_foot_position");
  double* left_hand_position = SensorByName(model, data, "left_hand_position");
  double* right_hand_position =
      SensorByName(model, data, "right_hand_position");
  double* torso_up = SensorByName(model, data, "torso_up");
  double* hand_right_up = SensorByName(model, data, "hand_right_up");
  double* hand_left_up = SensorByName(model, data, "hand_left_up");
  double* foot_right_up = SensorByName(model, data, "foot_right_up");
  double* foot_left_up = SensorByName(model, data, "foot_left_up");
  double* com_position = SensorByName(model, data, "body_subtreecom");
  double* com_velocity = SensorByName(model, data, "body_subtreelinvel");

  // ----- Height ----- //
  if (mode == kModeStand) {
    double head_feet_error = head_position[2] - 0.5 * (left_foot_position[2] +
                                                       right_foot_position[2]);
    residual[counter++] = head_feet_error - parameters_[0];
  } else if (mode == kModeHandstand) {
    double hand_feet_error =
        0.5 * (left_foot_position[2] + right_foot_position[2]) -
        0.5 * (left_hand_position[2] - right_hand_position[2]);
    residual[counter++] = hand_feet_error - parameters_[0];
  }

  // ----- Balance: CoM-feet xy error ----- //

  // capture point
  double kFallTime = 0.05;
  double capture_point[3] = {com_position[0], com_position[1], com_position[2]};
  mju_addToScl3(capture_point, com_velocity, kFallTime);

  // average feet xy position
  double fxy_avg[2] = {0.0};
  if (mode == kModeStand) {
    mju_addTo(fxy_avg, left_foot_position, 2);
    mju_addTo(fxy_avg, right_foot_position, 2);
  } else if (mode == kModeHandstand) {
    mju_addTo(fxy_avg, left_hand_position, 2);
    mju_addTo(fxy_avg, right_hand_position, 2);
  }

  mju_scl(fxy_avg, fxy_avg, 0.5, 2);
  mju_subFrom(fxy_avg, capture_point, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[counter++] = com_feet_distance;

  // ----- COM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], com_velocity, 2);
  counter += 2;

  // ----- Ctrl difference ----- //
  mju_sub(residual + counter, data->ctrl,
          model->key_qpos + model->nq * mode + 7, model->nu);
  counter += model->nu;

  // ----- Upright ----- //
  double standing = 1.0;
  double z_ref[3] = {0.0, 0.0, 1.0};

  if (mode == kModeStand) {
    // right foot
    mju_sub3(&residual[counter], foot_right_up, z_ref);
    mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
    counter += 3;

    mju_sub3(&residual[counter], foot_left_up, z_ref);
    mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
    counter += 3;

    // torso
    residual[counter++] = torso_up[2] - 1.0;

    // zero remaining residual
    mju_zero(residual + counter, 6);
    counter += 6;
  } else if (mode == kModeHandstand) {
    // right hand
    mju_sub3(&residual[counter], hand_right_up, z_ref);
    mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
    counter += 3;

    // left hand
    mju_add3(&residual[counter], hand_left_up, z_ref);
    mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
    counter += 3;

    // right foot
    mju_add3(&residual[counter], foot_right_up, z_ref);
    mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
    counter += 3;

    // left foot
    mju_add3(&residual[counter], foot_left_up, z_ref);
    mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
    counter += 3;

    // torso
    residual[counter++] = 1.0 * (torso_up[2] + 1.0);
  }

  // ----- Joint velocity ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
  counter += model->nv - 6;

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

void OP3::TransitionLocked(mjModel* model, mjData* d) {
  // check for mode change
  if (residual_.current_mode_ != mode) {
    // update mode for residual
    residual_.current_mode_ = mode;

    // set height goal based on mode (stand, handstand)
    parameters[0] = kModeHeight[mode];
  }
}

}  // namespace mjpc
