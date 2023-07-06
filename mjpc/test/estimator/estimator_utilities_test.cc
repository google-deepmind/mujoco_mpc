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
#include "mjpc/estimators/estimator.h"
#include "mjpc/test/load.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(RecursivePrior, ConditionMatrixDense) {
  // dimensions
  const int n = 3;
  const int n0 = 1;
  const int n1 = n - n0;

  // symmetric matrix
  double mat[n * n] = {1.0, 0.1, 0.01, 0.1, 1.0, 0.1, 0.01, 0.1, 1.0};

  // scratch
  double mat00[n0 * n0];
  double mat10[n1 * n0];
  double mat11[n1 * n1];
  double tmp0[n1 * n0];
  double tmp1[n1 * n1];
  double res[n1 * n1];

  // condition matrix
  ConditionMatrix(res, mat, mat00, mat10, mat11, tmp0, tmp1, n, n0, n1);

  // solution
  double solution[4] = {0.99, 0.099, 0.099, 0.9999};

  // test
  double error[n1 * n1];
  mju_sub(error, res, solution, n1 * n1);

  EXPECT_NEAR(mju_norm(error, n1 * n1), 0.0, 1.0e-4);
}

TEST(RecursivePrior, ConditionMatrixBand) {
  // dimensions
  const int n = 4;
  const int n0 = 3;
  const int n1 = n - n0;
  const int nband = 2;

  // symmetric matrix
  double mat[n * n] = {1.0, 0.1, 0.0, 0.0, 0.1, 1.0, 0.1, 0.0,
                       0.0, 0.1, 1.0, 0.1, 0.0, 0.0, 0.1, 1.0};

  // scratch
  double mat00[n0 * n0];
  double mat10[n1 * n0];
  double mat11[n1 * n1];
  double tmp0[n1 * n0];
  double tmp1[n1 * n1];
  double bandfactor[n0 * n0];
  double res[n1 * n1];

  // condition matrix
  ConditionMatrix(res, mat, mat00, mat10, mat11, tmp0, tmp1, n, n0, n1,
                  bandfactor, nband);

  // solution
  double solution[n1 * n1] = {0.98989796};

  // test
  double error[n1 * n1];
  mju_sub(error, res, solution, n1 * n1);

  EXPECT_NEAR(mju_norm(error, n1 * n1), 0.0, 1.0e-4);
}

TEST(QuaternionInterpolation, Slerp) {
  // quaternions 
  double quat0[4] = {1.0, 0.0, 0.0, 0.0};
  double quat1[4] = {0.7071, 0.0, 0.7071, 0.0};
  mju_normalize4(quat1);
  // printf("quat0 = \n");
  // mju_printMat(quat0, 1, 4);
  // printf("quat1 = \n");
  // mju_printMat(quat1, 1, 4);

  // -- slerp: t = 0 -- //
  double t = 0.0;
  double slerp0[4];
  double jac00[9];
  double jac01[9];
  Slerp(slerp0, quat0, quat1, t, jac00, jac01);

  // printf("slerp0 = \n");
  // mju_printMat(slerp0, 1, 4);

  // test
  double error[4];
  mju_sub(error, slerp0, quat0, 4);
  EXPECT_NEAR(mju_norm(error, 4), 0.0, 1.0e-4);

  // -- slerp: t = 1.0 -- //
  t = 1.0;
  double slerp1[4];
  double jac10[9];
  double jac11[9];
  Slerp(slerp1, quat0, quat1, t, jac10, jac11);

  // printf("slerp1 = \n");
  // mju_printMat(slerp1, 1, 4);

  // test
  mju_sub(error, slerp1, quat1, 4);
  EXPECT_NEAR(mju_norm(error, 4), 0.0, 1.0e-4);

  // -- slerp: t = 0.5 -- //
  t = 0.5;
  double slerp05[4];
  double jac050[9];
  double jac051[9];
  Slerp(slerp05, quat0, quat1, t, jac050, jac051);

  // printf("slerp05 = \n");
  // mju_printMat(slerp05, 1, 4);

  // test
  double slerp05_solution[4] = {0.92387953, 0.0, 0.38268343, 0.0};
  mju_sub(error, slerp05, slerp05_solution, 4);
  EXPECT_NEAR(mju_norm(error, 4), 0.0, 1.0e-4);

  // ----- jacobians ----- // 
  double jac0fdT[9];
  double jac1fdT[9];
  mju_zero(jac0fdT, 9);
  mju_zero(jac1fdT, 9);
  double jac0fd[9];
  double jac1fd[9];

  // -- t = 0.5 -- //
  t = 0.5;

  // finite difference 
  double eps = 1.0e-6;
  double nudge[3];
  mju_zero(nudge, 3);

  for (int i = 0; i < 3; i++) {
    // perturb
    mju_zero(nudge, 3);
    nudge[i] += eps;

    // quat0 perturb
    double q0i[4];
    double slerp0i[4];
    mju_copy(q0i, quat0, 4);
    mju_quatIntegrate(q0i, nudge, 1.0);
    Slerp(slerp0i, q0i, quat1, t, NULL, NULL);
    double* dif0 = jac0fdT + 3 * i;
    mju_subQuat(dif0, slerp0i, slerp05);
    mju_scl(dif0, dif0, 1.0 / eps, 3);

    // quat1 perturb
    double q1i[4];
    double slerp1i[4];
    mju_copy(q1i, quat1, 4);
    mju_quatIntegrate(q1i, nudge, 1.0);
    Slerp(slerp1i, quat0, q1i, t, NULL, NULL);
    double* dif1 = jac1fdT + 3 * i;
    mju_subQuat(dif1, slerp1i, slerp05);
    mju_scl(dif1, dif1, 1.0 / eps, 3);
  }

  // transpose results 
  mju_transpose(jac0fd, jac0fdT, 3, 3);
  mju_transpose(jac1fd, jac1fdT, 3, 3);

  // error 
  double error_jac[9];
  
  mju_sub(error_jac, jac050, jac0fd, 9);
  EXPECT_NEAR(mju_norm(error_jac, 9) / 9, 0.0, 1.0e-3);

  mju_sub(error_jac, jac051, jac1fd, 9);
  EXPECT_NEAR(mju_norm(error_jac, 9) / 9, 0.0, 1.0e-3);

  // printf("jac0fd = \n");
  // mju_printMat(jac0fd, 3, 3);

  // printf("jac050 = \n");
  // mju_printMat(jac050, 3, 3);

  // printf("jac1fd = \n");
  // mju_printMat(jac1fd, 3, 3);

  // printf("jac051 = \n");
  // mju_printMat(jac051, 3, 3);

  // -- t = 0.0 -- //
  t = 0.0;

  // finite difference 

  for (int i = 0; i < 3; i++) {
    // perturb
    mju_zero(nudge, 3);
    nudge[i] += eps;

    // quat0 perturb
    double q0i[4];
    double slerp0i[4];
    mju_copy(q0i, quat0, 4);
    mju_quatIntegrate(q0i, nudge, 1.0);
    Slerp(slerp0i, q0i, quat1, t, NULL, NULL);
    double* dif0 = jac0fdT + 3 * i;
    mju_subQuat(dif0, slerp0i, slerp0);
    mju_scl(dif0, dif0, 1.0 / eps, 3);

    // quat1 perturb
    double q1i[4];
    double slerp1i[4];
    mju_copy(q1i, quat1, 4);
    mju_quatIntegrate(q1i, nudge, 1.0);
    Slerp(slerp1i, quat0, q1i, t, NULL, NULL);
    double* dif1 = jac1fdT + 3 * i;
    mju_subQuat(dif1, slerp1i, slerp0);
    mju_scl(dif1, dif1, 1.0 / eps, 3);
  }

  // transpose results 
  mju_transpose(jac0fd, jac0fdT, 3, 3);
  mju_transpose(jac1fd, jac1fdT, 3, 3);

  // error   
  mju_sub(error_jac, jac00, jac0fd, 9);
  EXPECT_NEAR(mju_norm(error_jac, 9) / 9, 0.0, 1.0e-3);

  mju_sub(error_jac, jac01, jac1fd, 9);
  EXPECT_NEAR(mju_norm(error_jac, 9) / 9, 0.0, 1.0e-3);

  // printf("jac0fd = \n");
  // mju_printMat(jac0fd, 3, 3);

  // printf("jac00 = \n");
  // mju_printMat(jac00, 3, 3);

  // printf("jac1fd = \n");
  // mju_printMat(jac1fd, 3, 3);

  // printf("jac01 = \n");
  // mju_printMat(jac01, 3, 3);

  // -- t = 1.0 -- //
  t = 1.0;

  // finite difference 

  for (int i = 0; i < 3; i++) {
    // perturb
    mju_zero(nudge, 3);
    nudge[i] += eps;

    // quat0 perturb
    double q0i[4];
    double slerp0i[4];
    mju_copy(q0i, quat0, 4);
    mju_quatIntegrate(q0i, nudge, 1.0);
    Slerp(slerp0i, q0i, quat1, t, NULL, NULL);
    double* dif0 = jac0fdT + 3 * i;
    mju_subQuat(dif0, slerp0i, slerp1);
    mju_scl(dif0, dif0, 1.0 / eps, 3);

    // quat1 perturb
    double q1i[4];
    double slerp1i[4];
    mju_copy(q1i, quat1, 4);
    mju_quatIntegrate(q1i, nudge, 1.0);
    Slerp(slerp1i, quat0, q1i, t, NULL, NULL);
    double* dif1 = jac1fdT + 3 * i;
    mju_subQuat(dif1, slerp1i, slerp1);
    mju_scl(dif1, dif1, 1.0 / eps, 3);
  }

  // transpose results 
  mju_transpose(jac0fd, jac0fdT, 3, 3);
  mju_transpose(jac1fd, jac1fdT, 3, 3);

  // error   
  mju_sub(error_jac, jac10, jac0fd, 9);
  EXPECT_NEAR(mju_norm(error_jac, 9) / 9, 0.0, 1.0e-3);

  mju_sub(error_jac, jac11, jac1fd, 9);
  EXPECT_NEAR(mju_norm(error_jac, 9) / 9, 0.0, 1.0e-3);

  // printf("jac0fd = \n");
  // mju_printMat(jac0fd, 3, 3);

  // printf("jac10 = \n");
  // mju_printMat(jac10, 3, 3);

  // printf("jac1fd = \n");
  // mju_printMat(jac1fd, 3, 3);

  // printf("jac11 = \n");
  // mju_printMat(jac11, 3, 3);
}

TEST(FiniteDifferenceVelocityAcceleration, Particle2D) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task1D.xml");
  mjData* data = mj_makeData(model);

  // dimensions
  int nq = model->nq, nv = model->nv;

  // pool
  ThreadPool pool(1);

  // ----- simulate ----- //

  // controller
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(10 * time);
    ctrl[1] = mju_cos(10 * time);
  };

  // trajectories
  int T = 200;
  std::vector<double> qpos(nq * T);
  std::vector<double> qvel(nv * T);
  std::vector<double> qacc(nv * T);

  // reset
  mj_resetData(model, data);

  // rollout
  for (int t = 0; t < T; t++) {
    // set control
    controller(data->ctrl, data->time);

    // forward computes instantaneous qacc
    mj_forward(model, data);

    // cache
    mju_copy(qpos.data() + t * nq, data->qpos, nq);
    mju_copy(qvel.data() + t * nv, data->qvel, nv);
    mju_copy(qacc.data() + t * nv, data->qacc, nv);

    // step using mj_Euler since mj_forward has been called
    // see mj_ step implementation here
    // https://
    // github.com/deepmind/mujoco/blob/main/src/engine/engine_forward.c#L831
    mj_Euler(model, data);
  }

  // ----- estimator ----- //

  // initialize
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);
  mju_copy(estimator.configuration.Data(), qpos.data(), nq * T);

  // compute velocity, acceleration
  estimator.ConfigurationEvaluation(pool);

  // velocity error
  std::vector<double> velocity_error(nv * (T - 1));
  mju_sub(velocity_error.data(), estimator.velocity.Data() + nv,
          qvel.data() + nv, nv * (T - 1));

  // velocity test
  EXPECT_NEAR(mju_norm(velocity_error.data(), nv * (T - 1)), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(estimator.velocity.Data(), nv), 0.0, 1.0e-5);

  // acceleration error
  std::vector<double> acceleration_error(nv * (T - 2));
  mju_sub(acceleration_error.data(), estimator.acceleration.Data() + nv,
          qacc.data() + nv, nv * (T - 2));

  // acceleration test
  EXPECT_NEAR(mju_norm(acceleration_error.data(), nv * (T - 2)), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(estimator.acceleration.Data(), nv), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(estimator.acceleration.Data() + nv * (T - 1), nv), 0.0,
              1.0e-5);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(FiniteDifferenceVelocityAcceleration, Box3D) {
  // load model
  mjModel* model = LoadTestModel("estimator/box/task0.xml");
  mjData* data = mj_makeData(model);

  // dimensions
  int nq = model->nq, nv = model->nv, nu = model->nu, ns = model->nsensordata;

  // pool
  ThreadPool pool(1);

  // ----- simulate ----- //
  // trajectories
  int T = 5;
  std::vector<double> qpos(nq * (T + 1));
  std::vector<double> qvel(nv * (T + 1));
  std::vector<double> qacc(nv * T);
  std::vector<double> ctrl(nu * T);
  std::vector<double> qfrc_actuator(nv * T);
  std::vector<double> sensordata(ns * (T + 1));

  // reset
  mj_resetData(model, data);

  // initialize TODO(taylor): improve initialization
  double qpos0[7] = {0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0};
  double qvel0[6] = {0.4, 0.05, -0.22, 0.01, -0.03, 0.24};
  mju_copy(data->qpos, qpos0, nq);
  mju_copy(data->qvel, qvel0, nv);

  // rollout
  for (int t = 0; t < T; t++) {
    // control
    mju_zero(data->ctrl, model->nu);

    // forward computes instantaneous qacc
    mj_forward(model, data);

    // cache
    mju_copy(qpos.data() + t * nq, data->qpos, nq);
    mju_copy(qvel.data() + t * nv, data->qvel, nv);
    mju_copy(qacc.data() + t * nv, data->qacc, nv);
    mju_copy(ctrl.data() + t * nu, data->ctrl, nu);
    mju_copy(qfrc_actuator.data() + t * nv, data->qfrc_actuator, nv);
    mju_copy(sensordata.data() + t * ns, data->sensordata, ns);

    // step using mj_Euler since mj_forward has been called
    // see mj_ step implementation here
    // https://github.com/deepmind/mujoco/blob/main/src/engine/engine_forward.c#L831
    mj_Euler(model, data);
  }

  // final cache
  mju_copy(qpos.data() + T * nq, data->qpos, nq);
  mju_copy(qvel.data() + T * nv, data->qvel, nv);

  mj_forward(model, data);
  mju_copy(sensordata.data() + T * ns, data->sensordata, ns);

  // ----- estimator ----- //

  // initialize
  Estimator estimator;
  estimator.Initialize(model);
  mju_copy(estimator.configuration.Data(), qpos.data(), nq * (T + 1));

  // compute velocity, acceleration
  estimator.ConfigurationEvaluation(pool);

  // velocity error
  std::vector<double> velocity_error(nv * T);
  mju_sub(velocity_error.data(), estimator.velocity.Data() + nv,
          qvel.data() + nv, nv * (T - 1));

  // velocity test
  EXPECT_NEAR(mju_norm(velocity_error.data(), nv * (T - 1)) / (nv * (T - 1)),
              0.0, 1.0e-3);

  // acceleration error
  std::vector<double> acceleration_error(nv * T);
  mju_sub(acceleration_error.data(), estimator.acceleration.Data() + nv,
          qacc.data() + nv, nv * (T - 2));

  // acceleration test
  EXPECT_NEAR(
      mju_norm(acceleration_error.data(), nv * (T - 1)) / (nv * (T - 1)), 0.0,
      1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
