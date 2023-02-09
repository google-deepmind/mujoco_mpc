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

#ifndef MJPC_PLANNERS_MODEL_DERIVATIVES_H_
#define MJPC_PLANNERS_MODEL_DERIVATIVES_H_

#include <cstdlib>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

// data and methods for model derivatives
class ModelDerivatives {
 public:
  // constructor
  ModelDerivatives() = default;

  // destructor
  ~ModelDerivatives() = default;

  // allocate memory
  void Allocate(int dim_state_derivative, int dim_action, int dim_sensor,
                int T);

  // reset memory to zeros
  void Reset(int dim_state_derivative, int dim_action, int dim_sensor, int T);

  // compute derivatives at all time steps
  void Compute(const mjModel* m, const std::vector<UniqueMjData>& data,
               const double* x, const double* u, const double* h, int dim_state,
               int dim_state_derivative, int dim_action, int dim_sensor, int T,
               double tol, int mode, ThreadPool& pool);

  // Jacobians
  std::vector<double> A;  // model Jacobians wrt state
                          //   (T * dim_state_derivative * dim_state_derivative)
  std::vector<double> B;  // model Jacobians wrt action
                          //   (T * dim_state_derivative * dim_action)
  std::vector<double> C;  // output Jacobians wrt state
                          //   (T * dim_sensor * dim_state_derivative)
  std::vector<double> D;  // output Jacobians wrt action
                          //   (T * dim_sensor * dim_action)
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_MODEL_DERIVATIVES_H_
