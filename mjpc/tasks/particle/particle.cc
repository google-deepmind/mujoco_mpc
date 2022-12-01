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

#include "tasks/particle/particle.h"

#include <mujoco/mujoco.h>
#include "utilities.h"

namespace mjpc {

// -------- Residuals for particle task -------
//   Number of residuals: 3
//     Residual (0): position - goal_position
//     Residual (1): velocity
//     Residual (2): control
// --------------------------------------------
void Particle::Residual(const double* parameters, const mjModel* model,
                        const mjData* data, double* residual) {
  // ----- residual (0) ----- //
  mju_copy(residual, data->qpos, model->nq);
  residual[0] -= data->mocap_pos[0];
  residual[1] -= data->mocap_pos[1];

  // ----- residual (1) ----- //
  mju_copy(residual + 2, data->qvel, model->nv);

  // ----- residual (2) ----- //
  mju_copy(residual + 4, data->ctrl, model->nu);
}

}  // namespace mjpc
