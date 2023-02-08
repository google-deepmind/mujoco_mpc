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

#ifndef MJPC_PLANNERS_GRADIENT_GRADIENT_H_
#define MJPC_PLANNERS_GRADIENT_GRADIENT_H_

#include <vector>

#include "mjpc/planners/cost_derivatives.h"
#include "mjpc/planners/gradient/policy.h"
#include "mjpc/planners/model_derivatives.h"

namespace mjpc {

// data and methods to perform gradient descent
class Gradient {
 public:
  // constructor
  Gradient() = default;

  // destructor
  ~Gradient() = default;

  // ----- methods ----- //

  // allocate memory
  void Allocate(int dim_state_derivative, int dim_action, int T);

  // reset memory to zeros
  void Reset(int dim_state_derivative, int dim_action, int T);

  // compute gradient at one time step
  int GradientStep(int n, int m, const double *Wx, const double *A,
                   const double *B, const double *cx, const double *cu,
                   double *Vx, double *du, double *Qx, double *Qu);

  // compute gradient for entire trajectory
  int Compute(GradientPolicy *p, const ModelDerivatives *md,
              const CostDerivatives *cd, int dim_state_derivative,
              int dim_action, int T);

  // ----- members ----- //
  std::vector<double> Vx;  // cost-to-go gradient
  double dV[2];            // cost-to-go error
  std::vector<double> Qx;  // objetive state gradient
  std::vector<double> Qu;  // cost action gradient
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_GRADIENT_GRADIENT_H_
