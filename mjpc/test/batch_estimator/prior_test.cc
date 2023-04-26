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
#include "mjpc/test/load.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

void subQuatJacobian(double* Ja, double* Jb, const double* qa,
                     const double* qb) {
  // subQuat
  double y[3];
  mju_subQuat(y, qa, qb);

  double eps = 1.0e-6;

  double JaT[9];      // quaternion difference Jacobian wrt to qa transposed
  double JbT[9];      // quaternion difference Jacobian wrt to qb transposed
  double dy[3];       // quaternion difference perturbation
  double dq[3];       // quaternion perturbation
  double qa_copy[4];  // qa copy
  double qb_copy[4];  // qb copy

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
}

TEST(PriorResidual, QuaternionOnly) {
  printf("Diff Test\n");

  // random quaternions
  double q0[4];
  double q1[4];
  double q2[4];

  for (int i = 0; i < 4; i++) {
    absl::BitGen gen_;
    q0[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    q1[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    q2[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }
  mju_normalize4(q0);
  mju_normalize4(q1);
  mju_normalize4(q2);

  printf("q0:\n");
  mju_printMat(q0, 1, 4);

  printf("q1:\n");
  mju_printMat(q1, 1, 4);

  printf("q2:\n");
  mju_printMat(q2, 1, 4);

  // Q prior
  double Qp[12] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};

  // Q random
  double Qr[12];
  mju_copy(Qr + 0 * 4, q0, 4);
  mju_copy(Qr + 1 * 4, q1, 4);
  mju_copy(Qr + 2 * 4, q2, 4);

  // prior residual
  auto prior_residual = [&Qp, &Qr](double* residual, const double* dQ) {
    for (int i = 0; i < 3; i++) {
      // perturb quaternion
      double q[4];
      mju_copy4(q, Qr + i * 4);
      const double* dq = dQ + i * 3;
      mju_quatIntegrate(q, dq, 1.0);
      mju_subQuat(residual + i * 3, q, Qp + i * 4);
    }
  };

  printf("residual\n");
  double r[3];
  mju_subQuat(r, q0, Qp + 0 * 4);
  mju_printMat(r, 1, 3);
  mju_subQuat(r, q1, Qp + 1 * 4);
  mju_printMat(r, 1, 3);
  mju_subQuat(r, q2, Qp + 2 * 4);
  mju_printMat(r, 1, 3);

  double residual[9];
  double dQ[9] = {0.0};
  prior_residual(residual, dQ);

  printf("residual (perturb): \n");
  mju_printMat(residual, 1, 9);

  printf("Jacobian:\n");
  FiniteDifferenceJacobian fdj(9, 9);
  fdj.Compute(prior_residual, dQ, 9, 9);
  mju_printMat(fdj.jacobian_.data(), 9, 9);

  // individual Jacobians
  double Jq0[9];
  double Jp0[9];
  double Jq1[9];
  double Jp1[9];
  double Jq2[9];
  double Jp2[9];

  subQuatJacobian(Jq0, Jp0, q0, Qp + 0 * 4);
  subQuatJacobian(Jq1, Jp1, q1, Qp + 1 * 4);
  subQuatJacobian(Jq2, Jp2, q2, Qp + 2 * 4);

  printf("Jq0: \n");
  mju_printMat(Jq0, 3, 3);

  printf("Jq1: \n");
  mju_printMat(Jq1, 3, 3);

  printf("Jq2: \n");
  mju_printMat(Jq2, 3, 3);
}

TEST(PriorResidual, Qpos) {
  // load model
  mjModel* model = LoadTestModel("box.xml");
  // mjData* data = mj_makeData(model);

  // random configurations 
  double z0[7];
  double z1[7];
  double z2[7];

  for (int i = 0; i < model->nq; i++) {
    absl::BitGen gen_;
    z0[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    z1[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    z2[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }
  mju_normalize4(z0 + 3);
  mju_normalize4(z1 + 3);
  mju_normalize4(z2 + 3);

  printf("z0: \n");
  mju_printMat(z0, 1, model->nq);

  printf("z1: \n");
  mju_printMat(z1, 1, model->nq);
  
  printf("z2: \n");
  mju_printMat(z2, 1, model->nq);

  // Q prior
  double Qp[21] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};

  // Q random
  double Qr[21];
  mju_copy(Qr + 0 * model->nq, z0, model->nq);
  mju_copy(Qr + 1 * model->nq, z1, model->nq);
  mju_copy(Qr + 2 * model->nq, z2, model->nq);

  // prior residual
  auto prior_residual = [&Qp, &Qr, &model](double* residual, const double* dQ) {
    for (int i = 0; i < 3; i++) {
      // perturb position
      double q[model->nq];
      mju_copy(q, Qr + i * model->nq, model->nq);
      const double* dq = dQ + i * model->nv;
      mj_integratePos(model, q, dq, 1.0);
      mj_differentiatePos(model, residual + i * model->nv, 1.0, Qp + i * model->nq, q);
    }
  };

  printf("residual\n");
  double r[model->nv];
  mj_differentiatePos(model, r, 1.0, Qp + 0 * model->nq, z0);
  mju_printMat(r, 1, model->nv);
  mj_differentiatePos(model, r, 1.0, Qp + 1 * model->nq, z1);
  mju_printMat(r, 1, model->nv);
  mj_differentiatePos(model, r, 1.0, Qp + 2 * model->nq, z2);
  mju_printMat(r, 1, model->nv);

  double residual[18];
  double dQ[18] = {0.0};
  prior_residual(residual, dQ);

  printf("residual (perturb): \n");
  mju_printMat(residual, 1, 18);

  printf("Jacobian:\n");
  FiniteDifferenceJacobian fdj(18, 18);
  fdj.Compute(prior_residual, dQ, 18, 18);
  mju_printMat(fdj.jacobian_.data(), 18, 18);  
}

}  // namespace
}  // namespace mjpc
