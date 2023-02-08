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

#include "mjpc/tasks/acrobot/acrobot.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string Acrobot::XmlPath() const {
  return GetModelPath("acrobot/task.xml");
}
std::string Acrobot::Name() const { return "Acrobot"; }

// ---------- Residuals for acrobot task ---------
//   Number of residuals: 5
//     Residual (0-1): Distance from tip to goal
//     Residual (2-3): Joint velocity
//     Residual (4):   Control
// -----------------------------------------------
void Acrobot::Residual(const mjModel* model, const mjData* data,
                       double* residual) const {
  // ---------- Residual (0-1) ----------
  mjtNum* goal_xpos = &data->site_xpos[3 * 0];
  mjtNum* tip_xpos = &data->site_xpos[3 * 1];
  residual[0] = goal_xpos[2] - tip_xpos[2];
  residual[1] = goal_xpos[0] - tip_xpos[0];

  // ---------- Residual (2-3) ----------
  residual[2] = data->qvel[0];
  residual[3] = data->qvel[1];

  // ---------- Residual (4) ----------
  residual[4] = data->ctrl[0];
}

}  // namespace mjpc
