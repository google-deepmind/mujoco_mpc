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

#include <vector>

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>
#include "mjpc/planners/gradient/spline_mapping.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// test cubic spline mapping
TEST(GradientTest, CubicTest) {
  // spline points
  const int S = 6;

  // domain
  std::vector<double> x = {0.1, 0.3, 0.7, 1.2, 1.21, 1.6};

  // values
  const int n = 2;
  double y[n * S] = {-1.0, 0.2, 0.5, 0.7, 0.1,   0.34,
                     -0.7, 0.9, 0.2, 0.1, -0.05, 1.0};

  // FiniteDifferenceSlope matrix
  double SM[(2 * n * S) * (n * S)];
  mju_zero(SM, (2 * n * S) * (n * S));

  // point-to-point mapping
  for (int i = 0; i < S; i++) {
    for (int j = 0; j < n; j++) {
      int row = n * S * (n * i + j);
      int col = n * i + j;
      SM[row + col] = 1.0;
    }
  }

  // point-to-FiniteDifferenceSlope mapping
  int shift = (n * S) * (n * S);
  for (int i = 0; i < S; i++) {
    double dt1 = (i > 0 ? 1.0 / (x[i] - x[i - 1]) : 0.0);
    double dt2 = (i < S - 1 ? 1.0 / (x[i + 1] - x[i]) : 0.0);
    if (i > 0 && i < S - 1) {
      dt1 *= 0.5;
      dt2 *= 0.5;
    }
    for (int j = 0; j < n; j++) {
      // i - 1
      if (i - 1 >= 0) {
        int row = n * S * (n * i + j);
        int col = n * (i - 1) + j;
        SM[shift + row + col] = -dt1;
      }

      // i
      int row = n * S * (n * i + j);
      int col = n * i + j;
      SM[shift + row + col] = dt1 - dt2;

      // i + 1
      if (i + 1 <= S - 1) {
        int row = n * S * (n * i + j);
        int col = n * (i + 1) + j;
        SM[shift + row + col] = dt2;
      }
    }
  }

  double ps[2 * n * S];
  mju_mulMatVec(ps, SM, y, 2 * n * S, n * S);

  // slopes
  double s[n * S];
  for (int i = 0; i < S; i++) {
    for (int j = 0; j < n; j++) {
      s[n * i + j] = FiniteDifferenceSlope(x[i], x, y, n, S, j);
    }
  }

  // times
  const int T = 10;
  double t[T];
  double dt = (x[S - 1] - x[0]) / (T - 1);
  t[0] = x[0];
  for (int i = 1; i < T; i++) {
    t[i] = t[i - 1] + dt;
  }

  // point-FiniteDifferenceSlope-to-cubic mapping
  double CM[(n * T) * (2 * n * S)];
  mju_zero(CM, (n * T) * (2 * n * S));

  int row, col;
  int bounds[2];
  double coefficients[4];
  for (int i = 0; i < T; i++) {
    FindInterval(bounds, x, t[i], S);
    CubicCoefficients(coefficients, t[i], x, S);
    for (int j = 0; j < n; j++) {
      // p0
      row = 2 * n * S * (n * i + j);
      col = n * bounds[0] + j;
      CM[row + col] = coefficients[0];

      // m0
      row = 2 * n * S * (n * i + j);
      col = n * S + n * bounds[0] + j;
      CM[row + col] = coefficients[1];

      if (bounds[0] != bounds[1]) {
        // p1
        row = 2 * n * S * (n * i + j);
        col = n * bounds[1] + j;
        CM[row + col] = coefficients[2];

        // m1
        row = 2 * n * S * (n * i + j);
        col = n * S + n * bounds[1] + j;
        CM[row + col] = coefficients[3];
      }
    }
  }

  // mapping
  double M[(n * T) * (n * S)];
  mju_mulMatMat(M, CM, SM, n * T, 2 * n * S, n * S);

  // ci
  double ci[n * T];
  for (int i = 0; i < T; i++) {
    CubicInterpolation(ci + n * i, t[i], x, y, n, S);
  }

  // ci mat
  double ci_mat[n * T];
  mju_mulMatVec(ci_mat, M, y, n * T, n * S);

  // error
  double error[n * T];
  mju_sub(error, ci, ci_mat, n * T);
  EXPECT_NEAR(mju_L1(error, n * T), 0.0, 1.0e-5);

  CubicSplineMapping csm;
  csm.Allocate(n);
  csm.Compute(x, S, t, T);

  // mapping error
  double map_error[n * T * n * S];
  mju_sub(map_error, M, csm.mapping.data(), n * T * n * S);
  EXPECT_NEAR(mju_L1(map_error, n * T * n * S), 0.0, 1.0e-5);
}

}  // namespace
}  // namespace mjpc
