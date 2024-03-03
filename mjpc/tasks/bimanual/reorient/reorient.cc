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

#include "mjpc/tasks/bimanual/reorient/reorient.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::aloha {
std::string Reorient::XmlPath() const {
  return GetModelPath("bimanual/reorient/task.xml");
}
std::string Reorient::Name() const { return "Bimanual Reorient"; }

void Reorient::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  int counter = 0;

  // reach
  double* left_gripper = SensorByName(model, data, "left/gripper");
  double* box = SensorByName(model, data, "box");
  mju_sub3(residual + counter, left_gripper, box);
  counter += 3;

  double* right_gripper = SensorByName(model, data, "right/gripper");
  mju_sub3(residual + counter, right_gripper, box);
  counter += 3;

  // bring
  double* target = SensorByName(model, data, "target");
  mju_sub3(residual + counter, box, target);
  counter += 3;

  // ---------- Cube orientation ----------
  double *cube_orientation = SensorByName(model, data, "cube_orientation");
  double *goal_cube_orientation =
      SensorByName(model, data, "cube_goal_orientation");
  mju_normalize4(goal_cube_orientation);

  mju_subQuat(residual + counter, goal_cube_orientation, cube_orientation);
  counter += 3;

  // ---------- Cube linear velocity ----------
  double *cube_linear_velocity =
      SensorByName(model, data, "cube_linear_velocity");

  mju_copy(residual + counter, cube_linear_velocity, 3);
  counter += 3;

  CheckSensorDim(model, counter);
}

}  // namespace mjpc::aloha
