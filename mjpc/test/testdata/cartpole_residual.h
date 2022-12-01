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

#ifndef MJPC_TASKS_CARTPOLE_RESIDUAL_H_
#define MJPC_TASKS_CARTPOLE_RESIDUAL_H_

#include <mujoco/mujoco.h>

void cartpole_residual(double* residual, const double* residual_parameters,
                       const mjModel* model, const mjData* data) {
  // goal position
  mju_copy(residual, data->qpos, model->nq);
  residual[1] -= 3.141592;

  // goal velocity
  mju_copy(residual + 2, data->qvel, model->nv);

  // action
  mju_copy(residual + 4, data->ctrl, model->nu);
}

#endif  // MJPC_TASKS_CARTPOLE_RESIDUAL_H_
