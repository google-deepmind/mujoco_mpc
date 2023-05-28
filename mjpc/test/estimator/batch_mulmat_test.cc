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

// zero block (size: rb x cb) in mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void ZeroBlockInMatrix(double* mat, int rm, int cm, int rb, int cb, int ri,
                       int ci) {
  // loop over block rows
  for (int i = 0; i < rb; i++) {
    // loop over block columns
    for (int j = 0; j < cb; j++) {
      mat[(ri + i) * cm + ci + j] = 0.0;
    }
  }
}

// square dense to block band matrix
void DenseToBlockBand(double* res, const double* mat, int dim, int dblock,
                      int nblock) {
  // number of block rows / columns
  int num_blocks = dim / dblock;

  // copy
  mju_copy(res, mat, dim * dim);

  // zero off-band blocks
  for (int i = 0; i < num_blocks; i++) {
    for (int j = i + nblock; j < num_blocks; j++) {
      ZeroBlockInMatrix(res, dim, dim, dblock, dblock, i * dblock, j * dblock);
      ZeroBlockInMatrix(res, dim, dim, dblock, dblock, j * dblock, i * dblock);
    }
  }
}

TEST(MulMatTest, BlockDiagonalTBandBlockDiagonal) {
  printf("BlockDiagonalTBandBlockDiagonal:\n");

  // dimensions
  // int dblock = 2;
  // int nblock = 2;
  // int T = 5;
  int dblock = 20;
  int nblock = 3;
  int T = 32;
  int ntotal = dblock * T;

  // ----- create random band matrix ----- //
  std::vector<double> F(ntotal * ntotal);
  std::vector<double> A(ntotal * ntotal);
  std::vector<double> Aband(ntotal * ntotal);

  // sample matrix square root
  absl::BitGen gen_;
  for (int i = 0; i < ntotal * ntotal; i++) {
    F[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // A = F' F
  mju_mulMatTMat(A.data(), F.data(), F.data(), ntotal, ntotal, ntotal);

  // band(A)
  DenseToBlockBand(Aband.data(), A.data(), ntotal, dblock, nblock);

  // ----- create random block diagonal matrix ----- //
  std::vector<double> Dblocks(dblock * dblock * T);
  std::vector<double> Ddense(ntotal * ntotal);
  std::vector<double> DT(ntotal * ntotal);

  // sample random blocks
  for (int i = 0; i < dblock * dblock * T; i++) {
    Dblocks[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // dense block diagonal matrix
  mju_zero(Ddense.data(), ntotal * ntotal);

  for (int t = 0; t < T; t++) {
    double* block = Dblocks.data() + dblock * dblock * t;
    SetBlockInMatrix(Ddense.data(), block, 1.0, ntotal, ntotal, dblock, dblock,
                     dblock * t, dblock * t);
  }

  mju_transpose(DT.data(), Ddense.data(), ntotal, ntotal);

  // ----- compute: B = D' * A * D ----- //
  std::vector<double> B(ntotal * ntotal);
  std::vector<double> tmp(ntotal * ntotal);

  // start timer
  auto dense_start = std::chrono::steady_clock::now();

  // tmp = A * D
  mju_mulMatMat(tmp.data(), Aband.data(), Ddense.data(), ntotal, ntotal,
                ntotal);

  // B = D' * tmp
  mju_mulMatTMat(B.data(), Ddense.data(), tmp.data(), ntotal, ntotal, ntotal);

  // end timer
  double timer_dense = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - dense_start)
                           .count();

  printf("dense time: %.5f\n", 1.0e-3 * timer_dense);

  // ----- custom method D' A D ----- //
  int num_upperband = 0;
  for (int i = 0; i < nblock; i++) {
    num_upperband += T - i;
  }

  int nscratch = 4 * dblock * dblock * num_upperband;
  std::vector<double> scratch(nscratch);
  std::vector<double> Bcustom(ntotal * ntotal);

  // start timer
  auto sparse_start = std::chrono::steady_clock::now();

  // compute
  BlockDiagonalTBlockBandBlockDiagonal(Bcustom.data(), Aband.data(),
                                       Dblocks.data(), dblock, nblock, T,
                                       scratch.data());

  // end timer
  double timer_sparse = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - sparse_start)
                            .count();

  printf("sparse time: %.5f\n", 1.0e-3 * timer_sparse);

  // ----- error ----- //
  std::vector<double> error(ntotal * ntotal);
  mju_sub(error.data(), Bcustom.data(), B.data(), ntotal * ntotal);

  printf("error: %.5f\n", mju_norm(error.data(), ntotal * ntotal));

  // ----- test ----- // 
  EXPECT_NEAR(mju_norm(error.data(), ntotal * ntotal), 0.0, 1.0e-4);
}

TEST(MulMatTest, RectBandTBlockDiagonalRectBand) {
  printf("RectBandTBlockDiagonalRectBand: \n");

  // dimensions
  int nr = 39;
  int nv = 20;
  int nc = nv * 3;
  int T = 32;

  // ----- random block diagonal blocks ----- //
  std::vector<double> Dblocks(nr * nr * (T - 2));
  std::vector<double> D((nr * (T - 2)) * (nr * (T - 2)));

  absl::BitGen gen_;
  for (int t = 0; t < T - 2; t++) {
    // sample random matrix square root
    std::vector<double> F(nr * nr);
    for (int i = 0; i < nr * nr; i++) {
      F[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    double* block = Dblocks.data() + nr * nr * t;
    mju_mulMatTMat(block, F.data(), F.data(), nr, nr, nr);

    // set block in matrix
    SetBlockInMatrix(D.data(), block, 1.0, nr * (T - 2), nr * (T - 2), nr, nr,
                     t * nr, t * nr);
  }

  // ----- random rectangular blocks ----- //
  std::vector<double> Jblocks(nr * nc * (T - 2));
  std::vector<double> J((nr * (T - 2)) * (nv * T));

  for (int t = 0; t < T - 2; t++) {
    double* block = Jblocks.data() + nr * nc * t;
    for (int i = 0; i < nr * nc; i++) {
      block[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }

    // set block
    SetBlockInMatrix(J.data(), block, 1.0, nr * (T - 2), nv * T, nr, nc, t * nr,
                     t * nv);
  }

  // ----- J' D J ----- //
  std::vector<double> tmp0((nr * (T - 2)) * (nv * T));
  std::vector<double> tmp1((nv * T) * (nv * T));

  // start timer
  auto dense_start = std::chrono::steady_clock::now();

  // tmp0 = D * J
  mju_mulMatMat(tmp0.data(), D.data(), J.data(), nr * (T - 2), nr * (T - 2),
                nv * T);

  // tmp1
  mju_mulMatTMat(tmp1.data(), J.data(), tmp0.data(), nr * (T - 2), nv * T,
                 nv * T);

  // end timer
  double timer_dense = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - dense_start)
                           .count();

  printf("dense time: %.5f\n", 1.0e-3 * timer_dense);

  // ----- custom J' D * J ----- //
  std::vector<double> JDJ((nv * T) * (nv * T));
  
  int nscratch = nr * nc * (T - 2) + nc * nc * (T - 2);
  std::vector<double> scratch(nscratch);

  // start timer
  auto sparse_start = std::chrono::steady_clock::now();

  // compute
  RectBandTBlockDiagonalRectBand(JDJ.data(), Dblocks.data(), Jblocks.data(), nr,
                                 nc, nv, T - 2, scratch.data());

  // end timer
  double timer_sparse = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - sparse_start)
                           .count();

  printf("sparse time: %.5f\n", 1.0e-3 * timer_sparse);

  // ----- error ----- //
  std::vector<double> error((nv * T) * (nv * T));
  mju_sub(error.data(), JDJ.data(), tmp1.data(), (nv * T) * (nv * T));
  printf("error: %.5f\n", mju_norm(error.data(), (nv * T) * (nv * T)));

  EXPECT_NEAR(mju_norm(error.data(), (nv * T) * (nv * T)), 0.0, 1.0e-5);
}

}  // namespace
}  // namespace mjpc
