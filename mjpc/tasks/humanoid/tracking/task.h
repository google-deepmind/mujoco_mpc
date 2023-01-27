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
namespace humanoid {

struct Tracking {

  // ------------- Residuals for humanoid tracking task -------------
  //   Number of residuals:
  //     Residual (0): Joint vel: minimise joint velocity
  //     Residual (1): Control: minimise control
  //     Residual (2-11): Tracking position: minimise tracking position error
  //         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
  //     Residual (11-20): Tracking velocity: minimise tracking velocity error
  //         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
  //   Number of parameters: 0
  // ----------------------------------------------------------------
  static void Residual(const double* parameters, const mjModel* model,
                       const mjData* data, double* residual);

  // --------------------- Transition for humanoid task -------------------------
  //   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
  //   Linearly interpolate between two consecutive key frames in order to
  //   smooth the transitions between keyframes.
  // ----------------------------------------------------------------------------
  static void Transition(const mjModel* model, mjData* data, Task* task);

};

}  // namespace humanoid
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_TRACKING_TASK_H_
