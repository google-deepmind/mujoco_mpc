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

#include "mjpc/tasks/planar_manipulator/planar_manipulator.h"

#include <mujoco/mujoco.h>

#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string PlanarManipulator::XmlPath() const {
  return GetModelPath("planar_manipulator/task.xml");
}
std::string PlanarManipulator::Name() const { return "Planar Manipulator"; }

// ----- Residuals for ball task -----
//   Number of residuals: 9
//     Residual (0-1): grasp
//     Residual (2):   close
//     Residual (3-4): bring
//     Residual (5-8): control
// -----------------------------------
void PlanarManipulator::Residual(const mjModel* model, const mjData* data,
                                 double* residual) const {
  // Grasp
  double* ball_pos = mjpc::SensorByName(model, data, "ball_pos");
  double* grasp_pos = mjpc::SensorByName(model, data, "grasp_pos");
  residual[0] = ball_pos[0] - grasp_pos[0];
  residual[1] = ball_pos[2] - grasp_pos[2];

  // Close
  double* closure = mjpc::SensorByName(model, data, "grasp");
  const double kDistThreshhold = .04;
  double dist = mju_sqrt(residual[0] * residual[0] + residual[1] * residual[1]);
  double out = 1 / (1 + mju_exp(-(dist - kDistThreshhold) / kDistThreshhold));
  double in = 1 - out;
  residual[2] = (closure[0] - .6) * in + out * (closure[0] + .2);

  // Bring
  residual[3] = ball_pos[0] - data->mocap_pos[0];
  residual[4] = ball_pos[2] - data->mocap_pos[2];

  // Control
  residual[5] = data->qfrc_actuator[0];
  residual[6] = data->qfrc_actuator[1];
  residual[7] = data->qfrc_actuator[2];
  residual[8] = data->qfrc_actuator[3];
}

}  // namespace mjpc