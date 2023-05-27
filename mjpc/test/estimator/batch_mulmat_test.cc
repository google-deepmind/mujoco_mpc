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

// block-diagonal' * block band * block-diagonal
void BlockDiagonalTBlockBandBlockDiagonal(double* res, const double* blkband,
                                          const double* blkdiag, int dim,
                                          int dblock, int nblock) {
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
      BlockFromMatrix(bbij, blkband, dblock, dblock, dim, dim, i * dblock,
                      j * dblock);
      const double* bdi = blkdiag + dblock * dblock * i;
      const double* bdj = blkdiag + dblock * dblock * j;

      // -- bdi' * bbij * bdj -- //

      // tmp0 = bbij * bdj
      mju_mulMatMat(tmp0, bbij, bdj, dblock, dblock, dblock);

      // tmp1 = bdi' * tmp0
      mju_mulMatTMat(tmp1, bdi, tmp0, dblock, dblock, dblock);

      // set block in matrix
      SetBlockInMatrix(res, tmp1, 1.0, dim, dim, dblock, dblock, i * dblock,
                       j * dblock);
      if (j > i) {
        mju_transpose(tmp2, tmp1, dblock, dblock);
        SetBlockInMatrix(res, tmp2, 1.0, dim, dim, dblock, dblock, j * dblock,
                         i * dblock);
      }
      count++;
    }
  }
  printf("block opts: %i\n", count);
}

TEST(MulMatTest, BlockDiagonalTBandBlockDiagonal) {
  printf("BlockDiagonalTBandBlockDiagonal:\n");

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
    SetBlockInMatrix(Ddense.data(), block, 1.0, ntotal, ntotal, dblock, dblock,
                     dblock * t, dblock * t);
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
  mju_mulMatMat(tmp.data(), Aband.data(), Ddense.data(), ntotal, ntotal,
                ntotal);

  printf("A * D = \n");
  mju_printMat(tmp.data(), ntotal, ntotal);

  // B = D' * tmp
  mju_mulMatTMat(B.data(), Ddense.data(), tmp.data(), ntotal, ntotal, ntotal);

  printf("B = D' * A * D = \n");
  mju_printMat(B.data(), ntotal, ntotal);

  // ----- custom method D' A D ----- //
  std::vector<double> Bcustom(ntotal * ntotal);
  BlockDiagonalTBlockBandBlockDiagonal(Bcustom.data(), Aband.data(),
                                       Dblocks.data(), ntotal, dblock, nblock);

  int num_blocks = ntotal / dblock;
  printf("num block ops: %i\n", (num_blocks - 1) * nblock + (nblock - 1));

  printf("B custom: \n");
  mju_printMat(Bcustom.data(), ntotal, ntotal);

  // ----- error ----- //
  std::vector<double> error(ntotal * ntotal);
  mju_sub(error.data(), Bcustom.data(), B.data(), ntotal * ntotal);

  printf("error: %.5f\n", mju_norm(error.data(), ntotal * ntotal));

  // ----- test ----- // 
  EXPECT_NEAR(mju_norm(error.data(), ntotal * ntotal), 0.0, 1.0e-4);

  // // ----- band * band ----- //
  // std::vector<double> BT(ntotal * ntotal);
  // mju_transpose(BT.data(), B.data(), ntotal, ntotal);

  // std::vector<double> BB(ntotal * ntotal);
  // std::vector<double> BBT(ntotal * ntotal);
  // mju_mulMatMat(BB.data(), B.data(), B.data(), ntotal, ntotal, ntotal);

  // printf("BB = \n");
  // mju_printMat(BB.data(), ntotal, ntotal);

  // printf("BBT = \n");
  // mju_transpose(BBT.data(), BB.data(), ntotal, ntotal);
  // mju_printMat(BBT.data(), ntotal, ntotal);

  // std::vector<double> BTBB(ntotal * ntotal);
  // mju_mulMatTMat(BTBB.data(), B.data(), BB.data(), ntotal, ntotal, ntotal);

  // printf("BTBB: \n");
  // mju_printMat(BTBB.data(), ntotal, ntotal);

  // // ----- factor ----- //
  // std::vector<double> AF(ntotal * ntotal);
  // mju_copy(AF.data(), A.data(), ntotal * ntotal);

  // mju_cholFactor(AF.data(), ntotal, 0.0);

  // printf("factor:\n");
  // mju_printMat(AF.data(), ntotal, ntotal);
}

// rectangular block' * block diagonal * rectangular block
void RectBandTBlockDiagonalRectBand(double* res, const double* blkdiag,
                                    const double* blkrect, int nr, int nc,
                                    int nci, int length) {
  // allocate blocks: diag * rect
  std::vector<double> dr_blk(nr * nc * length);

  // allocate blocks: rect' * diag * rect
  std::vector<double> rdr_blk(nc * nc * length);

  mju_zero(res, (nc + (length - 1) * nci) * (nc + (length - 1) * nci));

  // create blocks
  for (int i = 0; i < length; i++) {
    // unpack
    const double* diagi = blkdiag + nr * nr * i;
    const double* recti = blkrect + nr * nc * i;

    // d * r
    double* dr = dr_blk.data() + nr * nc * i;
    mju_mulMatMat(dr, diagi, recti, nr, nr, nc);

    // r' * d * r
    double* rdr = rdr_blk.data() + nc * nc * i;
    mju_mulMatTMat(rdr, recti, dr, nr, nc, nc);

    // set
    AddBlockInMatrix(res, rdr, 1.0, nc + (length - 1) * nci,
                     nc + (length - 1) * nci, nc, nc, nci * i, nci * i);
  }
}

TEST(MulMatTest, RectBandTBlockDiagonalRectBand) {
  // dimensions
  int nr = 2;
  int nv = 2;
  int nc = nv * 3;
  int T = 5;

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

    printf("diagonal block (%i) = \n", t);
    mju_printMat(block, nr, nr);

    // set block in matrix
    SetBlockInMatrix(D.data(), block, 1.0, nr * (T - 2), nr * (T - 2), nr, nr,
                     t * nr, t * nr);
  }

  printf("D = \n");
  mju_printMat(D.data(), nr * (T - 2), nr * (T - 2));

  // ----- random rectangular blocks ----- //
  std::vector<double> Jblocks(nr * nc * (T - 2));
  std::vector<double> J((nr * (T - 2)) * (nv * T));

  for (int t = 0; t < T - 2; t++) {
    double* block = Jblocks.data() + nr * nc * t;
    for (int i = 0; i < nr * nc; i++) {
      block[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }

    printf("rectangular block (%i) = \n", t);
    mju_printMat(block, nr, nc);

    // set block
    SetBlockInMatrix(J.data(), block, 1.0, nr * (T - 2), nv * T, nr, nc, t * nr,
                     t * nv);
  }

  printf("J = \n");
  mju_printMat(J.data(), nr * (T - 2), nv * T);

  // ----- J' D J ----- //

  // tmp0
  std::vector<double> tmp0((nr * (T - 2)) * (nv * T));
  mju_mulMatMat(tmp0.data(), D.data(), J.data(), nr * (T - 2), nr * (T - 2),
                nv * T);

  // tmp1
  std::vector<double> tmp1((nv * T) * (nv * T));
  mju_mulMatTMat(tmp1.data(), J.data(), tmp0.data(), nr * (T - 2), nv * T,
                 nv * T);

  printf("J' D J = \n");
  mju_printMat(tmp1.data(), nv * T, nv * T);

  // ----- custom J' D * J ----- //
  std::vector<double> JDJ((nv * T) * (nv * T));
  RectBandTBlockDiagonalRectBand(JDJ.data(), Dblocks.data(), Jblocks.data(), nr,
                                 nc, nv, T - 2);

  printf("custom: J' D J = \n");
  mju_printMat(JDJ.data(), nv * T, nv * T);

  // ----- error ----- //
  std::vector<double> error((nv * T) * (nv * T));
  mju_sub(error.data(), JDJ.data(), tmp1.data(), (nv * T) * (nv * T));
  printf("error: %.5f", mju_norm(error.data(), (nv * T) * (nv * T)));

  EXPECT_NEAR(mju_norm(error.data(), (nv * T) * (nv * T)), 0.0, 1.0e-5);
}

}  // namespace
}  // namespace mjpc
