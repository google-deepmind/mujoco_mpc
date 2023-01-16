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

#ifndef MJPC_TASKS_PARTICLE_PARTICLE_H_
#define MJPC_TASKS_PARTICLE_PARTICLE_H_

#include <mujoco/mujoco.h>

namespace mjpc {
struct Particle {
// -------- Residuals for particle move-to-goal task -------
//   Number of residuals: 3
//     Residual (0): position - goal_position
//     Residual (1): velocity
//     Residual (2): control
// --------------------------------------------
static void ResidualGoal(const double* parameters, const mjModel* model,
                     const mjData* data, double* residual);

// -------- Residuals for particle track task -------
//   Number of residuals: 3
//     Residual (0): position - goal_position
//     Residual (1): velocity
//     Residual (2): control
// --------------------------------------------
static void ResidualTrack(const double* parameters, const mjModel* model,
                     const mjData* data, double* residual);

// -------- Transition for particle track task --------
// Set goal to next.
// -----------------------------------------------
static int Transition(int state, const mjModel* model, mjData* data);
};
}  // namespace mjpc

#endif  // MJPC_TASKS_PARTICLE_PARTICLE_H_
