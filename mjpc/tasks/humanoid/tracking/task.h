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

#ifndef MJPC_TASKS_HUMANOID_TRACKING_TASK_H_
#define MJPC_TASKS_HUMANOID_TRACKING_TASK_H_

#include <mujoco/mujoco.h>
#include "../../../task.h"

namespace mjpc {
namespace Humanoid {

struct Tracking {

  // -------------- Residuals for humanoid tracking task ------------
  //   Number of residuals: TODO(hartikainen)
  //     Residual (0): TODO(hartikainen)
  //   Number of parameters: TODO(hartikainen)
  //     Parameter (0): TODO(hartikainen)
  // ----------------------------------------------------------------
  static void Residual(const double* parameters, const mjModel* model,
                       const mjData* data, double* residual);

  // ------------ Transition for humanoid tracking task -------------
  //   TODO(hartikainen)
  // ----------------------------------------------------------------
  static int Transition(int state, const mjModel* model, mjData* data,
                        Task* task);

};

}  // namespace Humanoid
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_TRACKING_TASK_H_
