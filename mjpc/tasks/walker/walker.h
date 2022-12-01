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

#ifndef MJPC_TASKS_WALKER_WALKER_H_
#define MJPC_TASKS_WALKER_WALKER_H_

#include <mujoco/mujoco.h>

namespace mjpc {
struct Walker {
// --------- Residuals for walker task --------
//   Number of residuals: 4
//     Residual (0): control
//     Residual (1): position_z - height_goal
//     Residual (2): body_z_axis - 1.0
//     Residual (3): velocity_x - speed_goal
//   Parameters: 2
//     Parameter (0): height_goal
//     Parameter (1): speed_goal
// --------------------------------------------
static void Residual(const double* parameters, const mjModel* model,
                     const mjData* data, double* residual);
};
}  // namespace mjpc

#endif  // MJPC_TASKS_WALKER_WALKER_H_
