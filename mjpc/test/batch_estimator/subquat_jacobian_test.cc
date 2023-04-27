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

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>

namespace mjpc {
namespace {

// TEST(SubQuatJacobianTest, Random) {
//   printf("subQuat Jacobian Test\n");

//   // random quaternions
//   double qa[4];
//   double qb[4];

//   for (int i = 0; i < 4; i++) {
//     absl::BitGen gen_;
//     qa[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
//     qb[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
//   }
//   mju_normalize4(qa);
//   mju_normalize4(qb);

//   printf("qa:\n");
//   mju_printMat(qa, 1, 4);

//   printf("qb:\n");
//   mju_printMat(qb, 1, 4);

//   // subQuat 
//   double y[3];
//   mju_subQuat(y, qa, qb);

//   printf("y:\n");
//   mju_printMat(y, 1, 3);

//   double eps = 1.0e-6;

//   double Ja[9];      // quaternion difference Jacobian wrt to qa
//   double Jb[9];      // quaternion difference Jacobian wrt to qb
//   double JaT[9];     // quaternion difference Jacobian wrt to qa transposed
//   double JbT[9];     // quaternion difference Jacobian wrt to qb transposed
//   double dy[3];      // quaternion difference perturbation
//   double dq[3];      // quaternion perturbation
//   double qa_copy[4]; // qa copy 
//   double qb_copy[4]; // qb copy

//   for (int i = 0; i < 3; i++) {
//     // perturbation 
//     mju_zero3(dq);
//     dq[i] = 1.0;

//     // Jacobian qa
//     mju_copy4(qa_copy, qa);
//     mju_quatIntegrate(qa_copy, dq, eps);
//     mju_subQuat(dy, qa_copy, qb);

//     mju_sub3(JaT + i * 3, dy, y);
//     mju_scl3(JaT + i * 3, JaT + i * 3, 1.0 / eps);

//     // Jacobian qb 
//     mju_copy4(qb_copy, qb);
//     mju_quatIntegrate(qb_copy, dq, eps);
//     mju_subQuat(dy, qa, qb_copy);

//     mju_sub3(JbT + i * 3, dy, y);
//     mju_scl3(JbT + i * 3, JbT + i * 3, 1.0 / eps);
//   }

//   // transpose result
//   mju_transpose(Ja, JaT, 3, 3);
//   mju_transpose(Jb, JbT, 3, 3);

//   printf("Ja: \n");
//   mju_printMat(Ja, 3, 3);

//   printf("Jb: \n");
//   mju_printMat(Jb, 3, 3);
// }

void mulQuatLeft(double* L, const double* quat) {
  // unpack
  double qs = quat[0];
  double q0 = quat[1];
  double q1 = quat[2];
  double q2 = quat[3];

  // row 1
  L[0] = qs;
  L[1] = -q0;
  L[2] = -q1;
  L[3] = -q2;

  // row 2
  L[4] = q0;
  L[5] = qs;
  L[6] = -q2;
  L[7] = q1;

  // row 3
  L[8] = q1;
  L[9] = q2;
  L[10] = qs;
  L[11] = -q0;

  // row 4
  L[12] = q2;
  L[13] = -q1;
  L[14] = q0;
  L[15] = qs;
}

// void mulQuatRight(double* R, const double* quat) {
//   // unpack
//   double qs = quat[0];
//   double q0 = quat[1];
//   double q1 = quat[2];
//   double q2 = quat[3];

//   // row 1
//   R[0] = qs;
//   R[1] = -q0;
//   R[2] = -q1;
//   R[3] = -q2;

//   // row 2
//   R[4] = q0;
//   R[5] = qs;
//   R[6] = q2;
//   R[7] = -q1;

//   // row 3
//   R[8] = q1;
//   R[9] = -q2;
//   R[10] = qs;
//   R[11] = q0;

//   // row 4
//   R[12] = q2;
//   R[13] = q1;
//   R[14] = -q0;
//   R[15] = qs;
// }

void mySubQuat(double* d, const double* qa, const double* qb) {
  double L[16];
  mulQuatLeft(L, qb);
  mju_mulMatTVec(d, L, qa, 4, 4);
  d[0] = 0.0;
}

// Subtract quaternions, express as 3D velocity: qb*quat(res) = qa.
void mju_subQuat2(mjtNum res[3], const mjtNum qa[4], const mjtNum qb[4]) {
  // qdif = neg(qb)*qa
  mjtNum qneg[4], qdif[4];
  mju_negQuat(qneg, qb);
  mju_mulQuat(qdif, qneg, qa);

  // convert to 3D velocity
  // mju_quat2Vel(res, qdif, 1);
  mju_copy(res, qdif + 1, 3);
}

TEST(SubQuatJacobianTest, Deterministic) {
  printf("subQuat Jacobian Test\n");

  // random quaternions
  double qa[4] = {1.0, 2.0, 3.0, 4.0};
  double qb[4] = {0.25, 0.3, 0.4, 0.5};

  mju_normalize4(qa);
  mju_normalize4(qb);

  printf("qa:\n");
  mju_printMat(qa, 1, 4);

  printf("qb:\n");
  mju_printMat(qb, 1, 4);

  // subQuat 
  double y[3];
  mju_subQuat2(y, qa, qb);

  printf("y:\n");
  mju_printMat(y, 1, 3);

  // double eps = 1.0e-6;

  // double Ja[9];      // quaternion difference Jacobian wrt to qa
  // double Jb[9];      // quaternion difference Jacobian wrt to qb
  // double JaT[9];     // quaternion difference Jacobian wrt to qa transposed
  // double JbT[9];     // quaternion difference Jacobian wrt to qb transposed
  // double dy[3];      // quaternion difference perturbation
  // double dq[3];      // quaternion perturbation
  // double qa_copy[4]; // qa copy 
  // double qb_copy[4]; // qb copy

  // for (int i = 0; i < 3; i++) {
  //   // perturbation 
  //   mju_zero3(dq);
  //   dq[i] = 1.0;

  //   // Jacobian qa
  //   mju_copy4(qa_copy, qa);
  //   mju_quatIntegrate(qa_copy, dq, eps);
  //   mju_subQuat(dy, qa_copy, qb);

  //   mju_sub3(JaT + i * 3, dy, y);
  //   mju_scl3(JaT + i * 3, JaT + i * 3, 1.0 / eps);

  //   // Jacobian qb 
  //   mju_copy4(qb_copy, qb);
  //   mju_quatIntegrate(qb_copy, dq, eps);
  //   mju_subQuat(dy, qa, qb_copy);

  //   mju_sub3(JbT + i * 3, dy, y);
  //   mju_scl3(JbT + i * 3, JbT + i * 3, 1.0 / eps);
  // }

  // // transpose result
  // mju_transpose(Ja, JaT, 3, 3);
  // mju_transpose(Jb, JbT, 3, 3);

  // printf("Ja: \n");
  // mju_printMat(Ja, 3, 3);

  // printf("Jb: \n");
  // mju_printMat(Jb, 3, 3);

  // analytical 
  // double q[4] = {1.0, 2.0, 3.0, 4.0};
  // double L[16];
  // mulQuatLeft(L, q);
  // printf("L:\n");
  // mju_printMat(L, 4, 4);

  // double R[16];
  // mulQuatRight(R, q);
  // printf("R:\n");
  // mju_printMat(R, 4, 4);

  // my subquat 
  double d[4];
  mySubQuat(d, qa, qb);

  printf("my y:\n");
  mju_printMat(d + 1, 1, 3);
}

}  // namespace
}  // namespace mjpc
