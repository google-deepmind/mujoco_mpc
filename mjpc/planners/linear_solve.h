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

#ifndef MJPC_PLANNERS_LINEAR_SOLVE_H_
#define MJPC_PLANNERS_LINEAR_SOLVE_H_

#include <vector>

namespace mjpc {

// data and methods for linear solve (i.e., Ax = b) via least-squares or
// least-norm https://ee263.stanford.edu/lectures/ls.pdf
// https://ee263.stanford.edu/archive/min-norm.pdf
class LinearSolve {
 public:
  // constructor
  LinearSolve() = default;

  // destructor
  ~LinearSolve() = default;

  // ----- methods ----- //

  // allocate memory
  void Initialize(int dim_row, int dim_col);

  // reset memory to zeros
  void Solve(double* x, const double* A, const double* b);

  // members
  int dim_row;
  int dim_col;
  std::vector<double> matrix_cache;
  std::vector<double> vector_cache;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_LINEAR_SOLVE_H_
