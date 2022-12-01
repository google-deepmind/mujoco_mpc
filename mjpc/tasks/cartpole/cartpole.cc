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

#include "tasks/cartpole/cartpole.h"

#include <cmath>

#include <mujoco/mujoco.h>

namespace mjpc {

// ------- Residuals for cartpole task ------
//   Number of residuals: 4
//     Residual (0): distance from vertical
//     Residual (1): distance from goal
//     Residual (2): angular velocity
//     Residual (3): control
// ------------------------------------------
void Cartpole::Residual(const double* parameters, const mjModel* model,
                        const mjData* data, double* residual) {
  // ---------- Residual (0) ----------
  residual[0] = std::cos(data->qpos[1]) + 1;

  // ---------- Residual (1) ----------
  residual[1] = data->qpos[0] - parameters[0];

  // ---------- Residual (2) ----------
  residual[2] = data->qvel[1];

  // ---------- Residual (3) ----------
  residual[3] = data->ctrl[0];
}

}  // namespace mjpc
