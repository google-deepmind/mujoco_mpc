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

// # void mju_subQuat(mjtNum res[3], const mjtNum qa[4], const mjtNum qb[4]) {
// #   // qdif = neg(qb)*qa
// #   mjtNum qneg[4], qdif[4];
// #   mju_negQuat(qneg, qb);
// #   mju_mulQuat(qdif, qneg, qa);

// #   // convert to 3D velocity
// #   mju_quat2Vel(res, qdif, 1);
// # }

TEST(DifferentiateQuaternionTest, QuaternionDifference) {
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

  printf("qa:\n");
  mju_printMat(qa, 1, 4);

  printf("qb:\n");
  mju_printMat(qb, 1, 4);

  // ----- finite difference ----- //
  // quaternion difference 
  double qdif[4];
  QuatDiff(qdif, qa, qb);

  printf("qdif:\n");
  mju_printMat(qdif, 1, 4);

  double eps = 1.0e-6;

  double Ja[12];     // quaternion difference Jacobian wrt to qa
  double Jb[12];     // quaternion difference Jacobian wrt to qb
  double JaT[12];    // quaternion difference Jacobian wrt to qa transposed
  double JbT[12];    // quaternion difference Jacobian wrt to qb transposed
  double dqdif[4];   // quaternion difference perturbation
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
    QuatDiff(dqdif, qa_copy, qb);

    mju_sub(JaT + i * 4, dqdif, qdif, 4);
    mju_scl(JaT + i * 4, JaT + i * 4, 1.0 / eps, 4);

    // Jacobian qb 
    mju_copy4(qb_copy, qb);
    mju_quatIntegrate(qb_copy, dq, eps);
    QuatDiff(dqdif, qa, qb_copy);

    mju_sub(JbT + i * 4, dqdif, qdif, 4);
    mju_scl(JbT + i * 4, JbT + i * 4, 1.0 / eps, 4);
  }

  // transpose result
  mju_transpose(Ja, JaT, 4, 3);
  mju_transpose(Jb, JbT, 4, 3);

  printf("Ja: \n");
  mju_printMat(Ja, 3, 4);

  printf("Jb: \n");
  mju_printMat(Jb, 3, 4);

  // ----- utilities ----- //
  // double Ga[9];      // quaternion difference Jacobian wrt to qa
  // double Gb[9];      // quaternion difference Jacobian wrt to qb
  // DifferentiateSubQuat(Ga, Gb, qa, qb, 1.0);
  
  // printf("Ga: \n");
  // mju_printMat(Ga, 3, 3);

  // printf("Gb: \n");
  // mju_printMat(Gb, 3, 3);
}

}  // namespace
}  // namespace mjpc
