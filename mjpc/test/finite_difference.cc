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

#include "mjpc/test/finite_difference.h"

#include <functional>

#include <mujoco/mujoco.h>

namespace mjpc {

// allocate memory
void FiniteDifferenceGradient::Allocate(
    std::function<double(const double*, int)> f, int n, double eps) {
  // dimension
  dimension = n;

  // epsilon
  epsilon = eps;

  // evaluation function
  eval = f;

  // gradient
  gradient.resize(n);

  // workspace
  workspace.resize(n);
}

// compute gradient
void FiniteDifferenceGradient::Gradient(const double* x) {
  // set workspace
  mju_copy(workspace.data(), x, dimension);

  // centered finite difference
  for (int i = 0; i < dimension; i++) {
    // positive
    workspace[i] += 0.5 * epsilon;
    double fp = eval(workspace.data(), dimension);

    // negative
    workspace[i] -= 1.0 * epsilon;
    double fn = eval(workspace.data(), dimension);
    gradient[i] = (fp - fn) / epsilon;

    // reset
    workspace[i] = x[i];
  }
}

// allocate memory
void FiniteDifferenceHessian::Allocate(
    std::function<double(const double*, int)> f, int n, double eps) {
  // dimension
  dimension = n;

  // epsilon
  epsilon = eps;

  // evaluation function
  eval = f;

  // Hessian
  hessian.resize(n * n);

  // workspaces
  workspace1.resize(n);
  workspace2.resize(n);
  workspace3.resize(n);
}

// compute gradient
void FiniteDifferenceHessian::Hessian(const double* x) {
  // set workspace
  mju_copy(workspace1.data(), x, dimension);
  mju_copy(workspace2.data(), x, dimension);
  mju_copy(workspace3.data(), x, dimension);

  // evaluate at candidate
  double fx = eval(x, dimension);

  // centered finite difference
  for (int i = 0; i < dimension; i++) {
    for (int j = 0; j < dimension; j++) {
      if (i > j) continue;  // skip bottom triangle
      // workspace 1
      workspace1[i] += epsilon;
      workspace1[j] += epsilon;

      double fij = eval(workspace1.data(), dimension);

      // workspace 2
      workspace2[i] += epsilon;
      double fi = eval(workspace2.data(), dimension);

      // workspace 3
      workspace3[j] += epsilon;
      double fj = eval(workspace3.data(), dimension);

      // Hessian value
      hessian[i * dimension + j] = (fij - fi - fj + fx) / (epsilon * epsilon);

      // reset workspace 1
      workspace1[i] = x[i];
      workspace1[j] = x[j];

      // reset workspace 2
      workspace2[i] = x[i];

      // reset workspace 3
      workspace3[j] = x[j];
    }
  }
  // set bottom triangle
  for (int i = 0; i < dimension; i++) {
    for (int j = 0; j < dimension; j++) {
      if (i > j) {
        hessian[i * dimension + j] = hessian[j * dimension + i];
      }
    }
  }
}

}  // namespace mjpc
