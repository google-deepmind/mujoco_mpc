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

#ifndef MJPC_TASKS_QUADRUPED_QUADRUPED_H_
#define MJPC_TASKS_QUADRUPED_QUADRUPED_H_

#include <mujoco/mujoco.h>
#include "task.h"

namespace mjpc {
struct Quadruped {
// --------------------- Residuals for quadruped task --------------------
//   Number of residuals: 4
//     Residual (0): position_z - average(foot position)_z - height_goal
//     Residual (1): position - goal_position
//     Residual (2): orientation - goal_orientation
//     Residual (3): control
//   Number of parameters: 1
//     Parameter (1): height_goal
// -----------------------------------------------------------------------
static void Residual(const double* parameters, const mjModel* model,
                     const mjData* data, double* residual);

static void ResidualFloor(const double* parameters, const mjModel* model,
                          const mjData* data, double* residual);

// -------- Transition for quadruped task --------
//   If quadruped is within tolerance of goal ->
//   set goal to next from keyframes.
// -----------------------------------------------
static int Transition(int state, const mjModel* model, mjData* data,
                      Task* task);
};
}  // namespace mjpc

#endif  // MJPC_TASKS_QUADRUPED_QUADRUPED_H_
