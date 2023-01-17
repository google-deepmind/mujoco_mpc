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

#ifndef MJPC_TASKS_QUADROTOR_QUADROTOR_H_
#define MJPC_TASKS_QUADROTOR_QUADROTOR_H_

#include <mujoco/mujoco.h>
#include "task.h"

namespace mjpc {
struct Quadrotor {
// --------------- Residuals for quadrotor task ---------------
//   Number of residuals: 5
//     Residual (0): position - goal position
//     Residual (1): orientation - goal orientation
//     Residual (2): linear velocity - goal linear velocity
//     Residual (3): angular velocity - goal angular velocity
//     Residual (4): control
//   Number of parameters: 6
// ------------------------------------------------------------
static void Residual(const double* parameters, const mjModel* model,
                     const mjData* data, double* residuals);

// ----- Transition for quadrotor task -----
static int Transition(int state, const mjModel* model, mjData* data,
                      Task* task);
};
}  // namespace mjpc

#endif  // MJPC_TASKS_QUADROTOR_QUADROTOR_H_
