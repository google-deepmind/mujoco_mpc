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

TEST(MeasurementResidual, Qpos) {
  // load model
  mjModel* model = LoadTestModel("box.xml");
  mjData* data = mj_makeData(model);

  printf("sensor measurements: \n");
  mju_printMat(data->sensordata, 1, model->nsensordata);

  // random configurations 
  double z0[model->nq];
  double z1[model->nq];
  double z2[model->nq];

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

  // Q random
  double Qr[21];
  mju_copy(Qr + 0 * model->nq, z0, model->nq);
  mju_copy(Qr + 1 * model->nq, z1, model->nq);
  mju_copy(Qr + 2 * model->nq, z2, model->nq);

  // measurement residual
  auto measurement_residual = [&Qr, &model, &data](double* residual, const double* dQ) {
    double Q[21];
    double V1[6];
    double V2[6];
    double A1[6];

    // perturb configurations
    for (int i = 0; i < 3; i++) {
      double* q = Q + i * model->nq;
      double* qr = Qr + i * model->nq;
      const double* dq = dQ + i * model->nv;
      mju_copy(q, qr, model->nq);
      mj_integratePos(model, q, dq, 1.0);
    }

    // configurations
    double* q0 = Q + 0 * model->nq;
    double* q1 = Q + 1 * model->nq;
    double* q2 = Q + 2 * model->nq;

    // compute velocity 
    mj_differentiatePos(model, V1, model->opt.timestep, q0, q1);
    mj_differentiatePos(model, V2, model->opt.timestep, q1, q2);

    // compute acceleration 
    mju_sub(A1, V2, V1, model->nv);
    mju_scl(A1, A1, 1.0 / model->opt.timestep, model->nv);

    // set configuration, velocity, acceleration 
    mju_copy(data->qpos, q1, model->nq);
    mju_copy(data->qvel, V1, model->nv);
    mju_copy(data->qacc, A1, model->nv);

    // inverse dynamics
    mj_inverse(model, data);

    // printf("sensor:\n");
    // mju_printMat(data->sensordata, 1, model->nsensordata);

    // printf("q1: \n");
    // mju_printMat(q1, 1, model->nq);

    // printf("v1: \n");
    // mju_printMat(V1, 1, model->nv);

    // printf("a1: \n");
    // mju_printMat(A1, 1, model->nv);

    mju_copy(residual, data->sensordata, model->nsensordata);
  };

  const int num_sensor = 3;
  double residual[num_sensor];
  double dQ[18] = {0.0};
  measurement_residual(residual, dQ);

  printf("residual:\n");
  mju_printMat(residual, 1, num_sensor);

  printf("Jacobian:\n");
  FiniteDifferenceJacobian fdj(num_sensor, 18);
  fdj.Compute(measurement_residual, dQ, num_sensor, 18);
  mju_printMat(fdj.jacobian_.data(), num_sensor, 18);  
}

}  // namespace
}  // namespace mjpc
