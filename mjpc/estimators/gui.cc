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

#include "mjpc/estimators/gui.h"

#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/utilities.h"

namespace mjpc {

// Initialize
void EstimatorGUIData::Initialize(const mjModel* model, int nprocess,
                                  int nsensor) {
  // timestep
  timestep = model->opt.timestep;

  // integrator
  integrator = model->opt.integrator;

  // process noise
  process_noise.resize(nprocess);

  // sensor noise
  sensor_noise.resize(nsensor);

  // scale prior
  scale_prior = GetNumberOrDefault(1.0, model, "batch_scale_prior");

  // estimation horizon
  horizon = GetNumberOrDefault(3, model, "batch_configuration_length");
}

}  // namespace mjpc
