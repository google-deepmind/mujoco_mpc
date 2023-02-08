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

#ifndef MJPC_TASKS_HUMANOID_STAND_TASK_H_
#define MJPC_TASKS_HUMANOID_STAND_TASK_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace humanoid {

class Stand : public Task {
 public:
  // ------------------ Residuals for humanoid stand task ------------
  //   Number of residuals: 6
  //     Residual (0): control
  //     Residual (1): COM_xy - average(feet position)_xy
  //     Residual (2): torso_xy - COM_xy
  //     Residual (3): head_z - feet^{(i)}_position_z - height_goal
  //     Residual (4): velocity COM_xy
  //     Residual (5): joint velocity
  //   Number of parameters: 1
  //     Parameter (0): height_goal
  // ----------------------------------------------------------------
  void Residual(const mjModel* model, const mjData* data,
                double* residual) const override;
  std::string Name() const override;
  std::string XmlPath() const override;
};

}  // namespace humanoid
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_STAND_TASK_H_
