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

#include "mjpc/tasks/humanoid_bench/walk_reward.h"

#include <cmath>
#include <limits>

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
// this is a one to one re-implementation of the reward as it is done in
// the humanoid_bench paper //
// it is used for the Stand, Walk, and Run task, with respective values
// for walk_speed //
double walk_reward(const mjModel *model, const mjData *data,
                   const double walk_speed, const double stand_height) {
  // initialize reward
  double reward = 1.0;

  // ----- standing ----- //
  double head_height = SensorByName(model, data, "head_height")[2];
  double standing =
      tolerance(head_height, {stand_height, INFINITY}, stand_height / 4);

  reward *= standing;

  // ----- torso upright ----- //
  double torso_upright = SensorByName(model, data, "torso_upright")[2];
  double upright =
      tolerance(torso_upright, {0.9, INFINITY}, 1.9, "linear", 0.0);

  reward *= upright;

  // ----- small control ----- //
  double small_control = 0.0;
  for (int i = 0; i < model->nu; i++) {
    small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic");
  }
  small_control /= model->nu;  // average over all controls
  small_control = (4 + small_control) / 5;

  reward *= small_control;

  // ----- move speed ----- //
  if (walk_speed == 0.0) {
    double horizontal_velocity_x =
        SensorByName(model, data, "center_of_mass_velocity")[0];
    double horizontal_velocity_y =
        SensorByName(model, data, "center_of_mass_velocity")[1];
    double dont_move = (tolerance(horizontal_velocity_x, {0.0, 0.0}, 2) +
                        tolerance(horizontal_velocity_y, {0.0, 0.0}, 2)) /
                       2;
    reward *= dont_move;
  } else {
    double com_velocity =
        SensorByName(model, data, "center_of_mass_velocity")[0];
    double move = tolerance(com_velocity, {walk_speed, INFINITY},
                            std::abs(walk_speed), "linear", 0.0);
    move = (5 * move + 1) / 6;
    reward *= move;
  }
  return reward;
}
}  // namespace mjpc
