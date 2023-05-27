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

// multiply band-diagonal matrix with sparse vector
void mju_bandMulMatVecSparse(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                             int ntotal, int nband, int ndense,
                             mjtByte flg_sym) {
  for (int i = 0; i < ntotal; i++) {
    int width = mjMIN(i + 1, nband);
    int adr = i * nband + nband - width;
    int offset = mjMAX(0, i - nband + 1);
    res[i] = mju_dot(mat + adr, vec + offset, width);  // lower triangle
    if (flg_sym) {
      // strict upper triangle
      mju_addToScl(res + offset, mat + adr, vec[i], width - 1);
    }
  }
}

void mju_bandMulMatVecPrint(mjtNum* res, const mjtNum* mat, const mjtNum* vec,
                             int ntotal, int nband, int ndense,
                             mjtByte flg_sym, int vind, int vlen) {
  mju_zero(res, ntotal);
  for (int i = 0; i < ntotal; i++) {
    int width = mjMIN(i + 1, nband);
    int adr = i * nband + nband - width;
    int offset = mjMAX(0, i - nband + 1);

    printf("(%i):\n", i);
    printf("  mat = ");
    mju_printMat(mat + adr, 1, width);

    printf("  vec = ");
    mju_printMat(vec+offset, 1, width);
    printf("  width = %i\n", width);
    
    // res[i] = mju_dot(mat + adr, vec + offset, width);  // lower triangle
    for (int j = 0; j < width; j++) {
      if (offset + j >= vind && offset + j < vind + vlen) {
        res[i] += mat[adr + j] * vec[offset + j];
      }
    }
    if (flg_sym) {
      // strict upper triangle
      printf("  upper triangle v[%i] = %f\n", i, vec[i]);
      printf("  res offset = %i\n", offset);
      printf("  width - 1 = %i\n", width - 1);
      if (i >= vind && i < vind + vlen) {
        mju_addToScl(res + offset, mat + adr, vec[i], width - 1);
      }
    }
  }
}

// get block (size: rb x cb) from mat (size: rm x cm) given mat upper row and left column indices (ri, ci)
void GetBlockFromMatrix(double* block, const double* A1, double s, int r1, int c1,
                       int r2, int c2, int ri, int ci) {
  // loop over A2 rows
  for (int i = 0; i < r2; i++) {
    // loop over A2 columns
    for (int j = 0; j < c2; j++) {
      A1[(ri + i) * c1 + ci + j] = s * A2[i * c2 + j];
    }
  }
}

TEST(MulMatTest, BandBlockDiagonal) {
  printf("Band Matrix x Block Diagonal Matrix:\n");

  // dimensions 
  int n = 2;
  int T = 5;
  int nband = 2 * n;
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
  std::vector<double> DT(ntotal * ntotal);

  // sample random blocks 
  for (int i = 0; i < n * n * T; i++) {
    Dblocks[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // dense block diagonal matrix 
  mju_zero(Ddense.data(), ntotal * ntotal);

  for (int t = 0; t < T; t++) {
    double* block = Dblocks.data() + n * n * t;
    SetBlockInMatrix(Ddense.data(), block, 1.0, ntotal, ntotal, n, n, n * t, n * t);
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
  mju_mulMatMat(tmp.data(), Abanddense.data(), Ddense.data(), ntotal, ntotal, ntotal);

  printf("A * D = \n");
  mju_printMat(tmp.data(), ntotal, ntotal);

  // B = D' * tmp
  mju_mulMatTMat(B.data(), Ddense.data(), tmp.data(), ntotal, ntotal, ntotal);

  printf("B = D' * A * D = \n");
  mju_printMat(B.data(), ntotal, ntotal);

  // ----- test sparse A * D ----- //
  // int col = 1;
  // int vind = col % n;
  // std::vector<double> x(ntotal);
  // mju_bandMulMatVecSparse(x.data(), Aband.data(), DT.data() + col * ntotal,
  //                         ntotal, nband, 0, true);

  // printf("column (%i) [vind (%i)]: \n", col, vind);
  // mju_printMat(x.data(), ntotal, 1);


  // ----- print bandMulMatVecSparse ----- //
  ntotal = 4;
  nband = 3;
  nnz = BandMatrixNonZeros(ntotal, nband);
  double X[16] = {1, 2, 3, 4, 2, 1, 5, 6, 3, 5, 1, 7, 4, 6, 7, 1};

  double Xbanddense[16] = {1, 2, 3, 0, 2, 1, 5, 6, 3, 5, 1, 7, 0, 6, 7, 1};
  double Xband[14];
  // int vind = 0;
  // int vlen = 4;
  // double b[4] = {21, 22, 23, 24};

  int vind = 1;
  int vlen = 2;
  double b[4] = {0, 22, 23, 0};

  double s0[4];
  double s1[4];

  mju_dense2Band(Xband, X, ntotal, nband, 0);

  printf("X = \n");
  mju_printMat(Xbanddense, ntotal, ntotal);
  printf("b = \n");
  mju_printMat(b, ntotal, 1);

  mju_bandMulMatVecPrint(s1, Xband, b, ntotal, nband, 0, true, vind, vlen);
  printf("s1 = \n");
  mju_printMat(s1, 1, ntotal);

  mju_bandMulMatVecSparse(s0, Xband, b, ntotal, nband, 0, true);
  printf("s0 = \n");
  mju_printMat(s0, 1, ntotal);
}

}  // namespace
}  // namespace mjpc
