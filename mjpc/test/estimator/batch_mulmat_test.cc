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
void DenseToBlockBand(double* res, const double* mat, int dim, int dblock, int nblock) {
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

// block-diagonal' * block band * block-diagonal
void BlockDiagonalTBlockBandBlockDiagonal(double* res, const double* blkband, const double* blkdiag, int dim, int dblock, int nblock) {
  // number of block rows / columns
  int num_blocks = dim / dblock;

  // allocate 
  std::vector<double> bbij_(dblock * dblock);
  double* bbij = bbij_.data();

  std::vector<double> tmp0_(dblock * dblock);
  double* tmp0 = tmp0_.data();

  std::vector<double> tmp1_(dblock * dblock);
  double* tmp1 = tmp1_.data();

  std::vector<double> tmp2_(dblock * dblock);
  double* tmp2 = tmp2_.data();

  int count = 0;
  for (int i = 0; i < num_blocks; i++) {
    int num_cols = mju_min(nblock, num_blocks - i);
    for (int j = i; j < i + num_cols; j++) {
      // get matrices
      BlockFromMatrix(bbij, blkband, dblock, dblock, dim, dim, i * dblock, j * dblock);
      const double* bdi = blkdiag + dblock * dblock * i;
      const double* bdj = blkdiag + dblock * dblock * j;

      // -- bdi' * bbij * bdj -- // 

      // tmp0 = bbij * bdj 
      mju_mulMatMat(tmp0, bbij, bdj, dblock, dblock, dblock);

      // tmp1 = bdi' * tmp0 
      mju_mulMatTMat(tmp1, bdi, tmp0, dblock, dblock, dblock);

      // set block in matrix 
      SetBlockInMatrix(res, tmp1, 1.0, dim, dim, dblock, dblock, i * dblock, j * dblock);
      if (j > i) {
        mju_transpose(tmp2, tmp1, dblock, dblock);
        SetBlockInMatrix(res, tmp2, 1.0, dim, dim, dblock, dblock, j * dblock, i * dblock);
      } 
      count++;
    }
  }
  printf("block opts: %i\n", count);
}

TEST(MulMatTest, BandBlockDiagonal) {
  printf("Band Matrix x Block Diagonal Matrix:\n");

  // dimensions 
  int dblock = 2;
  int nblock = 2;
  int T = 5;
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

  printf("A:\n");
  mju_printMat(A.data(), ntotal, ntotal);


  // band(A) 
  DenseToBlockBand(Aband.data(), A.data(), ntotal, dblock, nblock);

  printf("A band:\n");
  mju_printMat(Aband.data(), ntotal, ntotal);

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
    SetBlockInMatrix(Ddense.data(), block, 1.0, ntotal, ntotal, dblock, dblock, dblock * t, dblock * t);
  }

  printf("D dense:\n");
  mju_printMat(Ddense.data(), ntotal, ntotal);

  mju_transpose(DT.data(), Ddense.data(), ntotal, ntotal);

  printf("D transpose: \n");
  mju_printMat(DT.data(), ntotal, ntotal);

  // ----- compute: B = D' * A * D ----- //
  std::vector<double> B(ntotal * ntotal);
  std::vector<double> tmp(ntotal * ntotal);

  // tmp = A * D 
  mju_mulMatMat(tmp.data(), Aband.data(), Ddense.data(), ntotal, ntotal, ntotal);

  printf("A * D = \n");
  mju_printMat(tmp.data(), ntotal, ntotal);

  // B = D' * tmp
  mju_mulMatTMat(B.data(), Ddense.data(), tmp.data(), ntotal, ntotal, ntotal);

  printf("B = D' * A * D = \n");
  mju_printMat(B.data(), ntotal, ntotal);

  // ----- custom method D' A D ----- //
  std::vector<double> Bcustom(ntotal * ntotal);
  BlockDiagonalTBlockBandBlockDiagonal(Bcustom.data(), Aband.data(), Dblocks.data(), ntotal, dblock, nblock);

  int num_blocks = ntotal / dblock;
  printf("num block ops: %i\n", (num_blocks - 1) * nblock + (nblock - 1));

  printf("B custom: \n");
  mju_printMat(Bcustom.data(), ntotal, ntotal);

  // ----- error ----- //
  std::vector<double> error(ntotal * ntotal);
  mju_sub(error.data(), Bcustom.data(), B.data(), ntotal * ntotal);

  printf("error: %.5f\n", mju_norm(error.data(), ntotal * ntotal));
}

}  // namespace
}  // namespace mjpc
