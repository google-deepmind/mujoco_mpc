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

#include <mutex>
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
  double timer_dense = GetDuration(dense_start);

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
  double timer_sparse = GetDuration(sparse_start);

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
  double timer_dense = GetDuration(dense_start);

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
  double timer_sparse = GetDuration(sparse_start);

  printf("sparse time: %.5f\n", 1.0e-3 * timer_sparse);

  // ----- error ----- //
  std::vector<double> error((nv * T) * (nv * T));
  mju_sub(error.data(), JDJ.data(), tmp1.data(), (nv * T) * (nv * T));
  printf("error: %.5f\n", mju_norm(error.data(), (nv * T) * (nv * T)));

  EXPECT_NEAR(mju_norm(error.data(), (nv * T) * (nv * T)), 0.0, 1.0e-5);
}

// multiply band-diagonal matrix with vector sparse
// void mju_bandMulMatVecSparse(mjtNum* res, const mjtNum* mat, const mjtNum*
// vec,
//                              int ntotal, int nband, mjtByte flg_sym) {
//   for (int i = 0; i < ntotal; i++) {
//     int width = mjMIN(i + 1, nband);
//     int adr = i * nband + nband - width;
//     int offset = mjMAX(0, i - nband + 1);

//     printf("row (%i):\n", i);
//     printf("  width = %i\n", width);
//     printf("  adr = %i\n", adr);
//     printf("  offset = %i\n", offset);

//     printf("  mat = ");
//     mju_printMat(mat + adr, 1, width);
//     printf("  vec = ");
//     mju_printMat(vec + offset, 1, width);

//     res[i] = mju_dot(mat + adr, vec + offset, width);  // lower triangle
//     if (flg_sym) {
//       // strict upper triangle
//       mju_addToScl(res + offset, mat + adr, vec[i], width - 1);

//       printf("  (ut) mat = ");
//       mju_printMat(mat + adr, 1, width - 1);
//       printf("  (ut) vec[%i] = %.4f\n", i, vec[i]);
//     }
//   }
// }

void mju_bandMulMatVecSparse_(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                              int ntotal, int nband, mjtByte flg_sym,
                              int vec_shift, int vec_length) {
  // vector interval
  int v1 = vec_shift;
  int v2 = vec_shift + vec_length - 1;

  for (int i = 0; i < ntotal; i++) {
    int width = mjMIN(i + 1, nband);
    int adr = i * nband + nband - width;
    int offset = mjMAX(0, i - nband + 1);

    // band interval
    int b1 = offset;
    int b2 = offset + width - 1;

    // overlap
    int overlap = mju_max(0, mju_min(b2, v2) - mju_max(b1, v1) + 1);

    if (overlap) {
      int start = mju_max(0, v1 - b1);
      res[i] = mju_dot(mat + adr + start, vec + offset + start, overlap);
    } else {
      res[i] = 0.0;
    }

    // strict upper triangle
    if (flg_sym && mju_max(0, mju_min(b2, i) - mju_max(b1, i) + 1)) {
      // strict upper triangle
      mju_addToScl(res + offset, mat + adr, vec[i], width - 1);
    }
  }
}

void BandMatrixColumnSparsity(int& shift, int& length, int ntotal, int dblock,
                              int nblock, int num_blocks, int col) {
  // get block column
  int blk_col = col / dblock;
  // set length
  length = dblock * (nblock + 2);

  // first nblock - 1 columns
  if (blk_col < nblock - 1) {
    length -= (nblock - blk_col - 1) * dblock;
  }

  // last nblock - 1 columns
  if (blk_col > num_blocks - nblock) {
    length -= (blk_col - (num_blocks - nblock)) * dblock;
  }

  // set shift
  shift = mju_max(0, blk_col - nblock + 1) * dblock;
}

TEST(MulMatTest, BandSparse) {
  printf("A * B * A':\n");

  // dimensions
  int dblock = 20;
  int nblock = 3;
  int T = 32;
  int ntotal = dblock * T;

  printf("dim = %i\n", ntotal);

  // ----- create random band matrix (0) ----- //
  std::vector<double> F(ntotal * ntotal);
  std::vector<double> A_(ntotal * ntotal);
  std::vector<double> A(ntotal * ntotal);

  // sample matrix square root
  absl::BitGen gen_;
  for (int i = 0; i < ntotal * ntotal; i++) {
    F[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // A = F' F
  mju_mulMatTMat(A_.data(), F.data(), F.data(), ntotal, ntotal, ntotal);

  // band(A)
  DenseToBlockBand(A.data(), A_.data(), ntotal, dblock, nblock);

  // ----- create random band matrix (1) ----- //
  std::vector<double> G(ntotal * ntotal);
  std::vector<double> B_(ntotal * ntotal);
  std::vector<double> B(ntotal * ntotal);

  // sample matrix square root
  for (int i = 0; i < ntotal * ntotal; i++) {
    G[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // B = G' G
  mju_mulMatTMat(B_.data(), G.data(), G.data(), ntotal, ntotal, ntotal);

  // band(B)
  DenseToBlockBand(B.data(), B_.data(), ntotal, dblock, nblock);

  // ----- compute: A * B * A' ----- //
  std::vector<double> ABAT(ntotal * ntotal);
  std::vector<double> tmp(ntotal * ntotal);

  // start timer
  auto dense_start = std::chrono::steady_clock::now();

  // tmp = B * A'
  mju_mulMatMatT(tmp.data(), B.data(), A.data(), ntotal, ntotal, ntotal);

  // ABA' = A * tmp
  mju_mulMatMat(ABAT.data(), A.data(), tmp.data(), ntotal, ntotal, ntotal);

  // stop timer
  double timer_dense = GetDuration(dense_start);

  printf("dense time: %.5f\n", 1.0e-3 * timer_dense);

  // ----- band ----- //
  std::vector<double> Aband(ntotal * ntotal);
  std::vector<double> Bband(ntotal * ntotal);
  std::vector<double> ABATband(ntotal * ntotal);
  std::vector<double> mBA(ntotal * ntotal);
  int num_thread = 9;

  ThreadPool pool(num_thread);

  // start timer
  auto band_start = std::chrono::steady_clock::now();

  // get band representations
  mju_dense2Band(Aband.data(), A.data(), ntotal, dblock * nblock, 0);
  mju_dense2Band(Bband.data(), B.data(), ntotal, dblock * nblock, 0);

  // mul(B, A)
  int count_before = pool.GetCount();

  for (int i = 0; i < ntotal; i++) {
    pool.Schedule([&tmp, &A, &Bband, ntotal, dblock, nblock, i]() {
      mju_bandMulMatVec(tmp.data() + ntotal * i, Bband.data(),
                        A.data() + ntotal * i, ntotal, dblock * nblock, 0, 1,
                        true);
    });
  }
  pool.WaitCount(count_before + ntotal);
  pool.ResetCount();

  // mul(A, mul(B, A))
  count_before = pool.GetCount();
  for (int i = 0; i < ntotal; i++) {
    pool.Schedule([&ABATband, &Aband, &tmp, ntotal, dblock, nblock, i]() {
      mju_bandMulMatVec(ABATband.data() + ntotal * i, Aband.data(),
                        tmp.data() + ntotal * i, ntotal, dblock * nblock, 0, 1,
                        true);
    });
  }
  pool.WaitCount(count_before + ntotal);
  pool.ResetCount();

  // end timer
  double timer_band = GetDuration(band_start);
  printf("band time (nthread = %i): %.5f\n", num_thread, 1.0e-3 * timer_band);

  mju_copy(mBA.data(), tmp.data(), ntotal * ntotal);

  // ----- sparse ----- //
  std::vector<int> scratch(ntotal * ntotal);
  std::vector<double> ABATsparse(ntotal * ntotal);

  // start
  auto sparse_start = std::chrono::steady_clock::now();

  // get band representations
  mju_dense2Band(Aband.data(), A.data(), ntotal, dblock * nblock, 0);
  mju_dense2Band(Bband.data(), B.data(), ntotal, dblock * nblock, 0);

  // mul(B, A)
  count_before = pool.GetCount();

  for (int i = 0; i < ntotal; i++) {
    pool.Schedule([&tmp, &A, &Bband, ntotal, dblock, nblock, T, i]() {
      int vec_shift, vec_length;
      BandMatrixColumnSparsity(vec_shift, vec_length, ntotal, dblock, nblock, T,
                               i);
      mju_bandMulMatVecSparse_(tmp.data() + ntotal * i, Bband.data(),
                               A.data() + ntotal * i, ntotal, dblock * nblock,
                               true, vec_shift, vec_length);
    });
  }
  pool.WaitCount(count_before + ntotal);
  pool.ResetCount();

  // mul(A, mul(B, A))
  count_before = pool.GetCount();
  for (int i = 0; i < ntotal; i++) {
    pool.Schedule([&ABATsparse, &Aband, &tmp, ntotal, dblock, nblock, T, i]() {
      int vec_shift, vec_length;
      BandMatrixColumnSparsity(vec_shift, vec_length, ntotal, dblock, nblock, T,
                               i);
      vec_shift = 0;
      vec_length = ntotal;
      mju_bandMulMatVecSparse_(ABATsparse.data() + ntotal * i, Aband.data(),
                               tmp.data() + ntotal * i, ntotal, dblock * nblock,
                               true, vec_shift, vec_length);
    });
  }
  pool.WaitCount(count_before + ntotal);
  pool.ResetCount();

  // end timer
  double timer_sparse = GetDuration(sparse_start);

  printf("sparse time (nthread = %i): %.5f\n", num_thread,
         1.0e-3 * timer_sparse);

  // -- error -- //
  std::vector<double> error(ntotal * ntotal);

  mju_sub(error.data(), ABATband.data(), ABAT.data(), ntotal * ntotal);
  printf("error (band) = %.5f\n", mju_norm(error.data(), ntotal * ntotal));

  mju_sub(error.data(), ABATsparse.data(), ABAT.data(), ntotal * ntotal);
  printf("error (sparse) = %.5f\n", mju_norm(error.data(), ntotal * ntotal));

  mju_sub(error.data(), tmp.data(), mBA.data(), ntotal * ntotal);
  printf("BA error (sparse) = %.5f\n", mju_norm(error.data(), ntotal * ntotal));

  // test sparse
  double X[16] = {
      1, 2, 3, 0, 2, 4, 5, 6, 3, 5, 7, 8, 0, 6, 8, 9,
  };

  double L[16] = {
      1, 0, 0, 0, 2, 4, 0, 0, 3, 5, 7, 0, 0, 6, 8, 9,
  };

  double b[4] = {
      10,
      11,
      12,
      13,
  };

  double bsparse[4] = {
      0,
      11,
      12,
      0,
  };

  int vec_shift = 1;
  int vec_len = 2;

  double Xband[16];

  ntotal = 4;
  int nband = 3;
  int ndense = 0;

  mju_dense2Band(Xband, X, ntotal, nband, ndense);

  printf("X:\n");
  mju_printMat(X, ntotal, ntotal);

  // printf("Xband:\n");
  // mju_printMat(Xband, ntotal, nband);

  printf("L:\n");
  mju_printMat(L, ntotal, ntotal);

  printf("b:\n");
  mju_printMat(b, ntotal, 1);

  double y[4];
  mju_mulMatVec(y, X, b, ntotal, ntotal);
  printf("y:\n");
  mju_printMat(y, ntotal, 1);

  double yband[4];
  mju_bandMulMatVec(yband, Xband, b, ntotal, nband, 0, 1, true);

  printf("yband:\n");
  mju_printMat(yband, ntotal, 1);

  std::vector<double> e(ntotal);
  mju_sub(e.data(), yband, y, ntotal);
  printf("error: %.4f\n", mju_norm(e.data(), ntotal));

  // ----- lower triangle ----- //
  double ylow0[4];
  mju_bandMulMatVecSparse_(ylow0, Xband, bsparse, ntotal, nband, true,
                           vec_shift, vec_len);

  double ylow1[4];
  mju_mulMatVec(ylow1, X, bsparse, ntotal, ntotal);

  mju_sub(e.data(), ylow0, ylow1, ntotal);
  printf("lower error: %.4f\n", mju_norm(e.data(), ntotal));
}

TEST(BandMatrix, ColumnSparsity) {
  // dimension
  int dblock = 2;
  int nblock = 3;
  int num_blocks = 5;
  int ntotal = num_blocks * dblock;

  // test values
  int shift, length;

  int col = 0;
  BandMatrixColumnSparsity(shift, length, ntotal, dblock, nblock, num_blocks,
                           col);
  EXPECT_EQ(shift, 0);
  EXPECT_EQ(length, 6);

  col = 1;
  BandMatrixColumnSparsity(shift, length, ntotal, dblock, nblock, num_blocks,
                           col);
  EXPECT_EQ(shift, 0);
  EXPECT_EQ(length, 6);

  col = 2;
  BandMatrixColumnSparsity(shift, length, ntotal, dblock, nblock, num_blocks,
                           col);
  EXPECT_EQ(shift, 0);
  EXPECT_EQ(length, 8);

  col = 3;
  BandMatrixColumnSparsity(shift, length, ntotal, dblock, nblock, num_blocks,
                           col);
  EXPECT_EQ(shift, 0);
  EXPECT_EQ(length, 8);

  col = 4;
  BandMatrixColumnSparsity(shift, length, ntotal, dblock, nblock, num_blocks,
                           col);
  EXPECT_EQ(shift, 0);
  EXPECT_EQ(length, 10);

  col = 5;
  BandMatrixColumnSparsity(shift, length, ntotal, dblock, nblock, num_blocks,
                           col);
  EXPECT_EQ(shift, 0);
  EXPECT_EQ(length, 10);

  col = 6;
  BandMatrixColumnSparsity(shift, length, ntotal, dblock, nblock, num_blocks,
                           col);
  EXPECT_EQ(shift, 2);
  EXPECT_EQ(length, 8);

  col = 7;
  BandMatrixColumnSparsity(shift, length, ntotal, dblock, nblock, num_blocks,
                           col);
  EXPECT_EQ(shift, 2);
  EXPECT_EQ(length, 8);

  col = 8;
  BandMatrixColumnSparsity(shift, length, ntotal, dblock, nblock, num_blocks,
                           col);
  EXPECT_EQ(shift, 4);
  EXPECT_EQ(length, 6);

  col = 9;
  BandMatrixColumnSparsity(shift, length, ntotal, dblock, nblock, num_blocks,
                           col);
  EXPECT_EQ(shift, 4);
  EXPECT_EQ(length, 6);

  // dimensions
  dblock = 21;
  nblock = 3;
  num_blocks = 7;
  ntotal = num_blocks * dblock;

  col = 0;
  BandMatrixColumnSparsity(shift, length, ntotal, dblock, nblock, num_blocks,
                           col);
  EXPECT_EQ(shift, 0);
  EXPECT_EQ(length, nblock * dblock);

  col = ntotal - 1;
  BandMatrixColumnSparsity(shift, length, ntotal, dblock, nblock, num_blocks,
                           col);
  EXPECT_EQ(shift, (num_blocks - nblock) * dblock);
  EXPECT_EQ(length, nblock * dblock);
}

}  // namespace
}  // namespace mjpc
