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

#include "mjpc/planners/gradient/gradient.h"

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>
#include "mjpc/planners/cost_derivatives.h"
#include "mjpc/planners/model_derivatives.h"
#include "mjpc/test/finite_difference.h"
#include "mjpc/test/lqr.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// test gradient descent on LQR problem
TEST(GradientTest, Gradient) {
  // ----- LQR problem ----- //

  // dimensions
  const int n = 2;
  const int m = 1;

  // horizon
  const int T = 3;

  // initial state
  double x0[n] = {0.0, 0.0};

  // actions
  double u[(T - 1) * m];
  mju_fill(u, 0.5, (T - 1) * m);

  // state trajectory
  double x[T * n];

  // rollout
  RolloutReturn(x, u, x0, T);

  // ----- gradient descent ----- //

  // // gradient descent
  Gradient gd;
  gd.Allocate(n, m, T);
  gd.Reset(n, m, T);

  // // policy
  GradientPolicy p;
  p.k.resize(m * T);

  // model derivatives
  ModelDerivatives md;
  md.Allocate(n, m, 0, T);

  // cost derivatives
  CostDerivatives cd;
  cd.Allocate(n, m, 0, T, n + m);

  // set derivatives
  for (int t = 0; t < T; t++) {
    cx(DataAt(cd.cx, t * n), x + t * n, (t == T - 1 ? nullptr : u + t * m));
    if (t == T - 1) continue;
    cu(DataAt(cd.cu, t * m), x + t * n, u + t * m);
    fx(DataAt(md.A, t * n * n), x + t * n, u + t * m);
    fu(DataAt(md.B, t * n * m), x + t * n, u + t * m);
  }

  // gradient descent
  gd.Compute(&p, &md, &cd, n, m, T);

  // ----- finite difference ----- //

  // evaluation function
  auto eval_action = [&x, &x0](const double* a, int b) {
    return RolloutReturn(x, a, x0, T);
  };

  // set up finite difference
  const int dim_action = (T - 1) * m;
  FiniteDifferenceGradient fd_action;
  fd_action.Allocate(eval_action, dim_action, 1.0e-6);

  // gradient
  fd_action.Gradient(u);

  // test gradient
  for (int i = 0; i < dim_action; i++) {
    EXPECT_NEAR(fd_action.gradient[i], gd.Qu[i], 1.0e-3);
  }

  // ----- finite difference ----- //

  // evaluation function
  auto eval_initial_state = [&x, &u](const double* a, int b) {
    return RolloutReturn(x, u, a, T);
  };

  // set up finite difference
  const int dim_initial_state = n;
  FiniteDifferenceGradient fd_initial_state;
  fd_initial_state.Allocate(eval_initial_state, dim_initial_state, 1.0e-6);

  // gradient
  fd_initial_state.Gradient(x0);

  // test gradient
  for (int i = 0; i < dim_initial_state; i++) {
    EXPECT_NEAR(fd_initial_state.gradient[i], gd.Vx[i], 1.0e-3);
  }
}

}  // namespace
}  // namespace mjpc
