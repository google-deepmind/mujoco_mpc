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

#include "mjpc/planners/ilqg/backward_pass.h"

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>
#include "mjpc/planners/cost_derivatives.h"
#include "mjpc/planners/model_derivatives.h"
#include "mjpc/test/lqr.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// test iLQG backward pass on LQR problem
TEST(iLQGTest, BackwardPass) {
  // ----- LQR problem ----- //

  // dimensions
  const int n = 2;
  const int m = 1;

  // action limits
  double action_limits[2 * m] = {-1.0, 1.0};

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

  // ----- backward pass ----- //

  // backward pass
  iLQGBackwardPass bp;
  bp.Allocate(n, m, T);

  // policy
  iLQGPolicy p;
  p.trajectory.Initialize(n, m, 0, 0, T);
  p.trajectory.Allocate(T);
  p.feedback_gain.resize(m * n * T);
  p.action_improvement.resize(m * T);

  // settings
  iLQGSettings settings;

  // model derivatives
  ModelDerivatives md;
  md.Allocate(n, m, 0, T);

  // cost derivatives
  CostDerivatives cd;
  cd.Allocate(n, m, 0, T, n + m);

  // set derivatives
  for (int t = 0; t < T; t++) {
    cx(DataAt(cd.cx, t * n), x + t * n, (t == T - 1 ? nullptr : u + t * m));
    cxx(DataAt(cd.cxx, t * n * n), x + t * n,
        (t == T - 1 ? nullptr : u + t * m));

    if (t == T - 1) continue;
    cu(DataAt(cd.cu, t * m), x + t * n, u + t * m);
    cuu(DataAt(cd.cuu, t * m * m), x + t * n, u + t * m);
    cxu(DataAt(cd.cxu, t * n * m), x + t * n, u + t * m);
    fx(DataAt(md.A, t * n * n), x + t * n, u + t * m);
    fu(DataAt(md.B, t * n * m), x + t * n, u + t * m);
  }

  // boxqp
  BoxQP boxqp;
  boxqp.Allocate(m);

  // backward pass
  bp.Riccati(&p, &md, &cd, n, m, T, 0.0, boxqp, u, action_limits, settings);

  // ----- oracle ----- //
  double Vx[T * n] = {0.0, 0.0, 0.5, 1.25, 0.5, 1.0};

  double Vxx[T * n * n] = {2.71428571, 2.0, 2.0, 4.0, 2.0, 1.0,
                           1.0,        2.5, 1.0, 0.0, 0.0, 1.0};

  double feedback_gain[(T - 1) * m * n] = {-0.285714285, -1.0, 0.0, -0.5};

  double action_improvement[(T - 1) * m] = {-0.5, -0.75};

  for (int t = 0; t < T; t++) {
    // cost-to-go gradient
    for (int i = 0; i < n; i++) {
      EXPECT_NEAR(bp.Vx[t * n + i], Vx[t * n + i], 1.0e-5);
    }

    // cost-to-go Hessian
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        EXPECT_NEAR(bp.Vxx[t * n * n + n * i + j], Vxx[t * n * n + n * i + j],
                    1.0e-5);
      }
    }

    if (t == T - 1) continue;

    // action improvement
    for (int i = 0; i < m; i++) {
      EXPECT_NEAR(p.action_improvement[t * m + i],
                  action_improvement[t * m + i], 1.0e-5);
    }

    // feedback
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        EXPECT_NEAR(p.feedback_gain[t * m * n + n * i + j],
                    feedback_gain[t * m * n + n * i + j], 1.0e-5);
      }
    }
  }
}

}  // namespace
}  // namespace mjpc
