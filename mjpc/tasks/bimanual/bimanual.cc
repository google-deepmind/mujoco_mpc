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

#include "mjpc/tasks/bimanual/bimanual.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string Bimanual::XmlPath() const {
  return GetModelPath("bimanual/task.xml");
}
std::string Bimanual::Name() const { return "Bimanual"; }

// ------- Residuals for bimanual task ------
//     Residual (0): Control effort
//     Residual (1): Block position
//     Residual (2): Left reach
//     Residual (3): Right reach
//     Residual (4): Joint velocity
//     Residual (5): Orientation
//     Residual (6): Nominal pose
// ------------------------------------------
void Bimanual::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  // counter
  int counter = 0;

  // end effector positions
  double* ee_left = SensorByName(model, data, "ee_left");
  double* ee_right = SensorByName(model, data, "ee_right");

  // base positions
  double base_left[3] = {-0.5, 0.0, 0.0};
  double base_right[3] = {0.5, 0.0, 0.0};

  // block position
  double* block_pos = SensorByName(model, data, "block_pos");
  
  // goal position
  double* goal_pos = SensorByName(model, data, "goal_pos");

  // -- distances -- //

  // left ee to block
  double left_block_err[3];
  mju_sub3(left_block_err, ee_left, block_pos);
  double left_block_dist = mju_norm3(left_block_err);

  // right ee to block
  double right_block_err[3];
  mju_sub3(right_block_err, ee_right, block_pos);
  double right_block_dist = mju_norm3(right_block_err);

  // left base to goal
  double left_goal_err[3];
  mju_sub3(left_goal_err, base_left, goal_pos);
  double left_goal_dist = mju_norm3(left_goal_err);

  // right base to goal
  double right_goal_err[3];
  mju_sub3(right_goal_err, base_right, goal_pos);
  double right_goal_dist = mju_norm3(right_goal_err);
  
  double left_reach_scale = 1.0;
  double right_reach_scale = 1.0;

  if (left_goal_dist < right_goal_dist && left_block_dist < right_block_dist) {
    right_reach_scale *= 0.1;
  } else if (left_goal_dist < right_goal_dist &&
             left_block_dist >= right_block_dist) {
    left_reach_scale *= 0.1;
  } else if (left_goal_dist >= right_goal_dist &&
             left_block_dist >= right_block_dist) {
    left_reach_scale *= 0.1;
  } else if (left_goal_dist >= right_goal_dist &&
             left_block_dist < right_block_dist) {
    right_reach_scale *= 0.1;
  }

  // ----- Effort ----- //
  mju_sub(residual + counter, data->ctrl, model->key_qpos, 6);
  residual[counter + 6] = 0.1 * (data->ctrl[6] - 0.057);
  counter += 7;

  mju_sub(residual + counter, data->ctrl + 7, model->key_qpos + 8, 6);
  residual[counter + 6] = 0.1 * (data->ctrl[13] - 0.057);
  counter += 7;

  // ----- Left reach ----- //
  mju_sub3(residual + counter, block_pos, ee_left);
  mju_scl3(residual + counter, residual + counter, left_reach_scale);
  counter += 3;

  // ----- Right reach ----- //
  mju_sub3(residual + counter, block_pos, ee_right);
  mju_scl3(residual + counter, residual + counter, right_reach_scale);
  counter += 3;

  // ----- Block position ----- //
  mju_sub3(residual + counter, block_pos, goal_pos);
  counter += 3;

  

  // ----- Joint velocity ----- //
  // mju_copy(residual + counter, data->qvel, 16);
  // counter += 16;

  // ----- Orientation ----- //
  double* block0 = SensorByName(model, data, "block0");
  double* block1 = SensorByName(model, data, "block1");
  double* block2 = SensorByName(model, data, "block2");
  double* block3 = SensorByName(model, data, "block3");
  double* block4 = SensorByName(model, data, "block4");
  double* block5 = SensorByName(model, data, "block5");
  double* block6 = SensorByName(model, data, "block6");
  double* block7 = SensorByName(model, data, "block7");

  double* goal0 = SensorByName(model, data, "goal0");
  double* goal1 = SensorByName(model, data, "goal1");
  double* goal2 = SensorByName(model, data, "goal2");
  double* goal3 = SensorByName(model, data, "goal3");
  double* goal4 = SensorByName(model, data, "goal4");
  double* goal5 = SensorByName(model, data, "goal5");
  double* goal6 = SensorByName(model, data, "goal6");
  double* goal7 = SensorByName(model, data, "goal7");

  mju_sub3(residual + counter, block0, goal0);
  counter += 3;
  mju_sub3(residual + counter, block1, goal1);
  counter += 3;
  mju_sub3(residual + counter, block2, goal2);
  counter += 3;
  mju_sub3(residual + counter, block3, goal3);
  counter += 3;
  mju_sub3(residual + counter, block4, goal4);
  counter += 3;
  mju_sub3(residual + counter, block5, goal5);
  counter += 3;
  mju_sub3(residual + counter, block6, goal6);
  counter += 3;
  mju_sub3(residual + counter, block7, goal7);
  counter += 3;

  // ----- Nominal pose ----- //
  // mju_sub(residual + counter, data->qpos, model->key_qpos, 8);
  // counter += 8;

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

}  // namespace mjpc
