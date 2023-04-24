// Copyright 2023 DeepMind Technologies Limited
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

#include "mjpc/estimators/batch/estimator.h"
#include "mjpc/utilities.h"

namespace mjpc {

// convert sequence of configurations to velocities
void ConfigurationToVelocity(double* velocity, const double* configuration,
                             int configuration_length,
                             const mjModel* model) {
  // loop over configuration trajectory
  for (int t = 0; t < configuration_length - 1; t++) {
    // previous and current configurations
    const double* q0 = configuration + t * model->nq;
    const double* q1 = configuration + (t + 1) * model->nq;

    // compute velocity
    double* v1 = velocity + t * model->nv;
    StateDiff(model, v1, q0, q1, model->opt.timestep);
  }
}

// convert sequence of configurations to velocities
void VelocityToAcceleration(double* acceleration, const double* velocity,
                             int velocity_length,
                             const mjModel* model) {
  // loop over velocity trajectory
  for (int t = 0; t < velocity_length - 1; t++) {
    // previous and current configurations
    const double* v0 = velocity + t * model->nv;
    const double* v1 = velocity + (t + 1) * model->nv;

    // compute acceleration
    double* a1 = acceleration + t * model->nv;
    mju_sub(a1, v1, v0, model->nv);
    mju_scl(a1, a1, 1.0 / model->opt.timestep, model->nv);
  }
}


}  // namespace mjpc
