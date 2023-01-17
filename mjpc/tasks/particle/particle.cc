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
  double* position = SensorByName(model, data, "position");
  double* goal = SensorByName(model, data, "goal");
  mju_sub(residual, position, goal, model->nq);

  // ----- residual (1) ----- //
  double* velocity = SensorByName(model, data, "velocity");
  mju_copy(residual + 2, velocity, model->nv);

  // ----- residual (2) ----- //
  mju_copy(residual + 4, data->ctrl, model->nu);
}

// -------- Residuals for particle task -------
//   Number of residuals: 3
//     Residual (0): position - goal_position
//     Residual (1): velocity
//     Residual (2): control
// --------------------------------------------
void Particle::ResidualTimeVarying(const double* parameters,
                                   const mjModel* model, const mjData* data,
                                   double* residual) {
  // ----- residual (0) ----- //
  // some Lissajous curve
  double goal[2] {0.25 * mju_sin(data->time), 0.25 * mju_cos(data->time/mjPI)};
  double* position = SensorByName(model, data, "position");
  mju_sub(residual, position, goal, model->nq);

  // ----- residual (1) ----- //
  double* velocity = SensorByName(model, data, "velocity");
  mju_copy(residual + 2, velocity, model->nv);

  // ----- residual (2) ----- //
  mju_copy(residual + 4, data->ctrl, model->nu);
}

}  // namespace mjpc
