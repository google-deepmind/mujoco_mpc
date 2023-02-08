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

#include "mjpc/planners/linear_solve.h"

#include <mujoco/mujoco.h>

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

// least-squares problems (A in R^{r x c}, r >= c)
TEST(LinearSolve, LeastSquares) {
  // A1 = [1.0 0.0; 0.0 3.0], b1 = [-0.2, 0.7]
  // x1* = [-0.2, 0.233333]
  const int r1 = 2;
  const int c1 = 2;
  double A1[r1 * c1] = {1.0, 0.0, 0.0, 3.0};
  double b1[c1] = {-0.2, 0.7};
  double x1[r1];

  mjpc::LinearSolve solver;
  solver.Initialize(r1, c1);
  solver.Solve(x1, A1, b1);

  EXPECT_NEAR(x1[0], -0.2, 1.0e-3);
  EXPECT_NEAR(x1[1], 0.233333, 1.0e-3);

  // A2 = [1.0 0.1; 0.25, 0.2; 2.4 1.3], b2 = [0.35; -0.75]
  // x2* = [0.54640356, -0.9435285]
  const int r2 = 3;
  const int c2 = 2;
  double A2[r2 * c2] = {1.0, 0.1, 0.25, 0.2, 2.4, 1.3};
  double b2[r2] = {0.35, -0.75, 0.2};
  double x2[c2];

  solver.Initialize(r2, c2);
  solver.Solve(x2, A2, b2);

  EXPECT_NEAR(x2[0], 0.54640356, 1.0e-3);
  EXPECT_NEAR(x2[1], -0.9435285, 1.0e-3);
}

// least-norm problems (A in R^{r x c}, r < c)
TEST(LinearSolve, LeastNorm) {
  // A1 = [1.0 0.0; 0.0 3.0], b1 = [-0.2, 0.7]
  // x1* = [-0.2, 0.233333]
  const int r1 = 2;
  const int c1 = 2;
  double A1[r1 * c1] = {1.0, 0.0, 0.0, 3.0};
  double b1[c1] = {-0.2, 0.7};
  double x1[r1];

  mjpc::LinearSolve solver;
  solver.Initialize(r1, c1);
  solver.Solve(x1, A1, b1);

  EXPECT_NEAR(x1[0], -0.2, 1.0e-3);
  EXPECT_NEAR(x1[1], 0.233333, 1.0e-3);
}

}  // namespace
