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

TEST(CovarianceUpdate, Dense) {
  int n = 3;
  double A[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  double B[9] = {10, 11, 12, 13, 14, 15, 16, 17, 18};
  double BT[9];
  mju_transpose(BT, B, n, n);
  double sol[9];

  // A * B 
  mju_mulMatMat(sol, A, B, n, n, n);
  printf("A * B = \n"); 
  mju_printMat(sol, n, n);

  // A * B' 
  mju_mulMatMatT(sol, A, B, n, n, n);
  printf("A * B' = \n");
  mju_printMat(sol, n, n);

  // B * A 
  mju_mulMatMat(sol, B, A, n, n, n);
  printf("B * A = \n");
  mju_printMat(sol, n, n);

  // B' * A 
  mju_mulMatTMat(sol, B, A, n, n, n);
  printf("B' * A = \n");
  mju_printMat(sol, n, n);

  // test 
  for (int i = 0; i < n; i++) {
    mju_mulMatVec(sol + i * n, A, B + i * n, n, n);
  }
  printf("A * vec(B) = \n");
  mju_printMat(sol, n, n);

  for (int i = 0; i < n; i++) {
    mju_mulMatVec(sol + i * n, A, BT + i * n, n, n);
  }
  printf("A * vec(BT) = \n");
  mju_printMat(sol, n, n);
}

TEST(CovarianceUpdate, Band) {}

}  // namespace
}  // namespace mjpc
