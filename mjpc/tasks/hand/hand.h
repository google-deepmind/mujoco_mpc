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

#ifndef MJPC_TASKS_HAND_HAND_H_
#define MJPC_TASKS_HAND_HAND_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class Hand : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  // ---------- Residuals for in-hand manipulation task ---------
  //   Number of residuals: 5
  //     Residual (0): cube_position - palm_position
  //     Residual (1): cube_orientation - cube_goal_orientation
  //     Residual (2): cube linear velocity
  //     Residual (3): cube angular velocity
  //     Residual (4): control
  // ------------------------------------------------------------
  void Residual(const mjModel* model, const mjData* data,
                double* residual) const override;
// ----- Transition for in-hand manipulation task -----
//   If cube is within tolerance or floor ->
//   reset cube into hand.
// -----------------------------------------------
  void Transition(const mjModel* model, mjData* data) override;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_HAND_HAND_H_
