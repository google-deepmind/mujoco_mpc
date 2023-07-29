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

#include "mjpc/tasks/cartpole/cartpole.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string Cartpole::XmlPath() const {
  return GetModelPath("cartpole/task.xml");
}
std::string Cartpole::Name() const { return "Cartpole"; }

// ------- Residuals for cartpole task ------
//     Vertical: Pole angle cosine should be -1
//     Centered: Cart should be at goal position
//     Velocity: Pole angular velocity should be small
//     Control:  Control should be small
// ------------------------------------------
void Cartpole::Residual(const mjModel* model, const mjData* data,
                        double* residual) const {
  // ---------- Vertical ----------
  residual[0] = std::cos(data->qpos[1]) - 1;

  // ---------- Centered ----------
  residual[1] = data->qpos[0] - parameters[0];

  // ---------- Velocity ----------
  residual[2] = data->qvel[1];

  // ---------- Control ----------
  residual[3] = data->ctrl[0];
}

}  // namespace mjpc
