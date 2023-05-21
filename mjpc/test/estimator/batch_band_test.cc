// Copyright 2023 DeepMind Technologies Limited
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

#include <mujoco/mujoco.h>

#include <vector>

#include "gtest/gtest.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(BandSolverTest, Solve) {
  // A * x = b
  double A[9] = {3, 1, 0, 1, 3, 1, 0, 1, 3};
  double b[3] = {1, 2, 3};

  // ----- dense solve ----- //

  // factor
  double Fd[9];
  mju_copy(Fd, A, 9);
  mju_cholFactor(Fd, 3, 0.0);

  // solve
  double xd[3];
  mju_cholSolve(xd, Fd, b, 3);

  // ----- band solve ----- //

  int ntotal = 3;
  int nband = 2;
  int ndense = 0;

  // convert dense to band matrix
  const int nnz = BandMatrixNonZeros(ntotal, nband);

  std::vector<double> Fb(nnz);
  mju_dense2Band(Fb.data(), A, ntotal, nband, ndense);

  // factor
  mju_cholFactorBand(Fb.data(), ntotal, nband, ndense, 0.0, 0.0);

  // solve
  double xb[3];
  mju_cholSolveBand(xb, Fb.data(), b, ntotal, nband, ndense);

  // ----- solution error ----- //
  double error[3];
  mju_sub(error, xb, xd, 3);

  // test
  EXPECT_NEAR(mju_norm(error, 3) / 3, 0.0, 1.0e-3);
}

}  // namespace
}  // namespace mjpc
