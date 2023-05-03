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

#include "gtest/gtest.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(DifferentiateQuaternionTest, SubQuat) {
    // random quaternions
  double qa[4];
  double qb[4];

  for (int i = 0; i < 4; i++) {
    absl::BitGen gen_;
    qa[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    qb[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }
  mju_normalize4(qa);
  mju_normalize4(qb);

  // subQuat 
  double y[3];
  mju_subQuat(y, qa, qb);

  double eps = 1.0e-6;
  double Ja[9];      // quaternion difference Jacobian wrt to qa
  double Jb[9];      // quaternion difference Jacobian wrt to qb
  double JaT[9];     // quaternion difference Jacobian wrt to qa transposed
  double JbT[9];     // quaternion difference Jacobian wrt to qb transposed
  double dy[3];      // quaternion difference perturbation
  double dq[3];      // quaternion perturbation
  double qa_copy[4]; // qa copy 
  double qb_copy[4]; // qb copy

  for (int i = 0; i < 3; i++) {
    // perturbation 
    mju_zero3(dq);
    dq[i] = 1.0;

    // Jacobian qa
    mju_copy4(qa_copy, qa);
    mju_quatIntegrate(qa_copy, dq, eps);
    mju_subQuat(dy, qa_copy, qb);

    mju_sub3(JaT + i * 3, dy, y);
    mju_scl3(JaT + i * 3, JaT + i * 3, 1.0 / eps);

    // Jacobian qb 
    mju_copy4(qb_copy, qb);
    mju_quatIntegrate(qb_copy, dq, eps);
    mju_subQuat(dy, qa, qb_copy);

    mju_sub3(JbT + i * 3, dy, y);
    mju_scl3(JbT + i * 3, JbT + i * 3, 1.0 / eps);
  }

  // transpose result
  mju_transpose(Ja, JaT, 3, 3);
  mju_transpose(Jb, JbT, 3, 3);

  // ----- utilities ----- //
  double Ga[9];      // quaternion to 3D velocity Jacobian wrt to qa
  double Gb[9];      // quaternion to 3D velocity Jacobian wrt to qa

  // compute Jacobians
  DifferentiateSubQuat(Ga, Gb, qa, qb, 1.0);

  // ----- error ----- //
  double error_a[9];
  double error_b[9];
  mju_sub(error_a, Ja, Ga, 9);
  mju_sub(error_b, Jb, Gb, 9);

  // ----- test ----- //
  EXPECT_NEAR(mju_norm(error_a, 9) / 9, 0.0, 1.0e-3);
  EXPECT_NEAR(mju_norm(error_b, 9) / 9, 0.0, 1.0e-3);
}

}  // namespace
}  // namespace mjpc
