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

#include "gtest/gtest.h"

namespace mjpc {
namespace {

// void R(double* r, const double* x, const double* u) {
//   r[0] = 0.1 * x[0];
//   r[1] = 0.2 * x[1];
//   r[2] = 0.3 * u[0];
// }

// void Rx(double* rx, const double* x, const double* u) {
//   mju_fill(rx, 0.0, 3 * 2);
// }

// void Ru(double* ru, const double* x, const double* u) {
//   mju_fill(ru, 0.0, 3 * 1);
// }

// TEST(CostDerivativesTest, RiskHessian) {
//   // task
//   Task task;

//   // set cost
//   task.num_residual = 1;
//   task.num_cost = 1;
//   task.dim_norm_residual.resize(1);
//   task.dim_norm_residual[0] = 3;
//   task.num_norm_parameter.resize(1);
//   task.num_norm_parameter[0] = 0;
//   task.norm.resize(1);
//   task.norm[0] = NormType::kNull;
//   task.weight.resize(1);
//   task.weight[0] = 1.23;
//   task.norm_parameter.resize(1);
//   task.norm_parameter[0] = 0.0;
//   task.risk = 0.35;
//   task.num_residual_parameters = 0;
//   task.residual_parameters.resize(1);
//   task.residual_parameters[0] = 0.0;
// }

}  // namespace
}  // namespace mjpc
