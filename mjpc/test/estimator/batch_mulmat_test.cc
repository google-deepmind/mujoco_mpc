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

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include <vector>

#include "gtest/gtest.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(MulMatTest, BandBlockDiagonal) {
  printf("Band Matrix x Block Diagonal Matrix:\n");

  // dimensions 
  int n = 2;
  int T = 5;
  int nband = 3 * n;
  int ntotal = n * T;
  int nnz = BandMatrixNonZeros(ntotal, nband);

  // ----- create random band matrix ----- //
  std::vector<double> F(ntotal * ntotal);
  std::vector<double> A(ntotal * ntotal);
  std::vector<double> Aband(nnz);
  std::vector<double> Abanddense(ntotal * ntotal);

  // sample matrix square root
  absl::BitGen gen_;
  for (int i = 0; i < ntotal * ntotal; i++) {
    F[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // A = F' F
  mju_mulMatTMat(A.data(), F.data(), F.data(), ntotal, ntotal, ntotal);

  // band(A) 
  mju_dense2Band(Aband.data(), A.data(), ntotal, nband, 0);

  // dense band(A) 
  mju_band2Dense(Abanddense.data(), Aband.data(), ntotal, nband, 0, true);

  printf("A band:\n");
  mju_printMat(Abanddense.data(), ntotal, ntotal);

  // ----- create random block diagonal matrix ----- // 
  std::vector<double> Dblocks(n * n * T);
  std::vector<double> Ddense(ntotal * ntotal);

  // sample random blocks 
  for (int i = 0; i < n * n * T; i++) {
    Dblocks[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // dense block diagonal matrix 
  mju_zero(Ddense.data(), ntotal * ntotal);

  for (int t = 0; t < T; t++) {
    double* block = Dblocks.data() + n * n * t;
    SetMatrixInMatrix(Ddense.data(), block, 1.0, ntotal, ntotal, n, n, n * t, n * t);
  }

  printf("D dense:\n");
  mju_printMat(Ddense.data(), ntotal, ntotal);

  // ----- compute: B = D' * A * D ----- //
  std::vector<double> B(ntotal * ntotal);
  std::vector<double> tmp(ntotal * ntotal);

  // tmp = A * D 
  mju_mulMatMat(tmp.data(), Abanddense.data(), Ddense.data(), ntotal, ntotal, ntotal);

  // B = D' * tmp
  mju_mulMatTMat(B.data(), Ddense.data(), tmp.data(), ntotal, ntotal, ntotal);

  printf("B = D' * A * D");
  mju_printMat(B.data(), ntotal, ntotal);
}

}  // namespace
}  // namespace mjpc
