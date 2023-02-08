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

#include "mjpc/planners/linear_solve.h"

#include <mujoco/mujoco.h>

#include "mjpc/utilities.h"

namespace mjpc {

// initialize solver
void LinearSolve::Initialize(int dim_row, int dim_col) {
  // set problem dimensions
  this->dim_row = dim_row;
  this->dim_col = dim_col;

  // allocate memory
  int max_dim = mju_max(dim_row, dim_col);
  matrix_cache.resize(max_dim * max_dim);
  vector_cache.resize(max_dim);
}

// solve Ax = b via least-squares or least-norm
void LinearSolve::Solve(double* x, const double* A, const double* b) {
  // ----- least-squares ----- //
  if (dim_row >= dim_col) {
    // M = A' A
    mju_mulMatTMat(matrix_cache.data(), A, A, dim_row, dim_col, dim_col);

    // cholesky(M)
    mju_cholFactor(matrix_cache.data(), dim_col, 0.0);

    // z = A' b
    mju_mulMatTVec(vector_cache.data(), A, b, dim_row, dim_col);

    // x = M \ z
    mju_cholSolve(x, matrix_cache.data(), vector_cache.data(), dim_col);
    // ----- least-norm ----- //
  } else {
    // M = A A'
    mju_mulMatMatT(matrix_cache.data(), A, A, dim_row, dim_col, dim_row);

    // cholesky(M)
    mju_cholFactor(matrix_cache.data(), dim_row, 0.0);

    // z = M \ b
    mju_cholSolve(vector_cache.data(), matrix_cache.data(), b, dim_row);

    // x = A' z
    mju_mulMatTVec(x, A, vector_cache.data(), dim_row, dim_col);
  }
}

}  // namespace mjpc
