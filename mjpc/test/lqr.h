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

#ifndef MJPC_TEST_LQR_H_
#define MJPC_TEST_LQR_H_

#include <mujoco/mujoco.h>

namespace mjpc {

// ----- LQR problem ----- //

// linear dynamics
void f(double* y, const double* x, const double* u);

// dynamics state Jacobian
void fx(double* A, const double* x, const double* u);

// dynamics action Jacobian
void fu(double* B, const double* x, const double* u);

// cost
double c(const double* x, const double* u);

// cost gradient state
void cx(double* Jx, const double* x, const double* u);

// cost gradient action
void cu(double* Ju, const double* x, const double* u);

// cost Hessian state
void cxx(double* Jxx, const double* x, const double* u);

// cost Hessian action
void cuu(double* Juu, const double* x, const double* u);

// cost Hessian action
void cxu(double* Jxu, const double* x, const double* u);

// rollout return
double RolloutReturn(double* x, const double* u, const double* x0, int T);

}  // namespace mjpc

#endif  // MJPC_TEST_LQR_H_
