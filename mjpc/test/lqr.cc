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

#include "mjpc/test/lqr.h"

#include <mujoco/mujoco.h>

namespace mjpc {

// ----- LQR problem ----- //

// linear dynamics
void f(double* y, const double* x, const double* u) {
  y[0] = x[0] + x[1];
  y[1] = x[1] + u[0];
}

// dynamics state Jacobian
void fx(double* A, const double* x, const double* u) {
  A[0] = 1.0;
  A[1] = 1.0;
  A[2] = 0.0;
  A[3] = 1.0;
}

// dynamics action Jacobian
void fu(double* B, const double* x, const double* u) {
  B[0] = 0.0;
  B[1] = 1.0;
}

// cost
double c(const double* x, const double* u) {
  double J = 0.0;
  J += 0.5 * mju_dot(x, x, 2);
  if (u) {
    J += 0.5 * mju_dot(u, u, 1);
  }
  return J;
}

// cost gradient state
void cx(double* Jx, const double* x, const double* u) {
  Jx[0] = x[0];
  Jx[1] = x[1];
}

// cost gradient action
void cu(double* Ju, const double* x, const double* u) {
  if (u) {
    Ju[0] = u[0];
  }
}

// cost Hessian state
void cxx(double* Jxx, const double* x, const double* u) {
  Jxx[0] = 1.0;
  Jxx[1] = 0.0;
  Jxx[2] = 0.0;
  Jxx[3] = 1.0;
}

// cost Hessian action
void cuu(double* Juu, const double* x, const double* u) {
  if (u) {
    Juu[0] = 1.0;
  }
}

// cost Hessian action
void cxu(double* Jxu, const double* x, const double* u) {
  if (u) {
    Jxu[0] = 0.0;
    Jxu[1] = 0.0;
  }
}

// rollout return
double RolloutReturn(double* x, const double* u, const double* x0, int T) {
  // initial state
  mju_copy(x, x0, 2);

  // cost
  double J = 0.0;

  // dynamics rollout
  for (int t = 0; t < T - 1; t++) {
    J += c(x + t * 2, u + t * 1);
    f(x + (t + 1) * 2, x + t * 2, u + t * 1);
  }
  J += c(x + (T - 1) * 2, nullptr);

  return J;
}

}  // namespace mjpc
