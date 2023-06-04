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
#include "mjpc/estimators/batch.h"
#include "mjpc/estimators/trajectory.h"
#include "mjpc/test/load.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(BatchShift, Particle2D) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // dimensions
  int nq = model->nq, nv = model->nv, nu = model->nu, ns = model->nsensordata;

  // threadpool 
  ThreadPool pool(2);

  // ----- simulate ----- //

  // controller
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(100 * time);
    ctrl[1] = mju_cos(100 * time);
  };

  // trajectories
  int horizon_buffer = 10;
  Trajectory qpos_buffer(nq, horizon_buffer + 1);
  Trajectory qvel_buffer(nv, horizon_buffer + 1);
  Trajectory qacc_buffer(nv, horizon_buffer);
  Trajectory ctrl_buffer(nu, horizon_buffer);
  Trajectory qfrc_actuator_buffer(nv, horizon_buffer);
  Trajectory sensor_buffer(ns, horizon_buffer + 1);

  // reset
  mj_resetData(model, data);

  // rollout
  for (int t = 0; t < horizon_buffer; t++) {
    // set control
    controller(data->ctrl, data->time);

    // forward computes instantaneous qacc
    mj_forward(model, data);

    // cache
    qpos_buffer.Set(data->qpos, t);
    qvel_buffer.Set(data->qvel, t);
    qacc_buffer.Set(data->qacc, t);
    ctrl_buffer.Set(data->ctrl, t);
    qfrc_actuator_buffer.Set(data->qfrc_actuator, t);
    sensor_buffer.Set(data->sensordata, t);

    // step using mj_Euler since mj_forward has been called
    // see mj_ step implementation here
    // https://github.com/deepmind/mujoco/blob/main/src/engine/engine_forward.c#L831
    mj_Euler(model, data);
  }

  // final cache
  qpos_buffer.Set(data->qpos, horizon_buffer);
  qvel_buffer.Set(data->qvel, horizon_buffer);
  
  mj_forward(model, data);
  sensor_buffer.Set(data->sensordata, horizon_buffer);

  // print 
  printf("qpos: \n");
  for (int t = 0; t < qpos_buffer.length_; t++) {
    printf("  (%i): ", t);
    mju_printMat(qpos_buffer.Get(t), 1, qpos_buffer.dim_);
    printf("\n");
  }

  printf("ctrl: \n");
  for (int t = 0; t < ctrl_buffer.length_; t++) {
    printf("  (%i): ", t);
    mju_printMat(ctrl_buffer.Get(t), 1, ctrl_buffer.dim_);
    printf("\n");
  }

  printf("qfrc actuator: \n");
  for (int t = 0; t < qfrc_actuator_buffer.length_; t++) {
    printf("  (%i): ", t);
    mju_printMat(qfrc_actuator_buffer.Get(t), 1, qfrc_actuator_buffer.dim_);
    printf("\n");
  }

  printf("sensor: \n");
  for (int t = 0; t < sensor_buffer.length_; t++) {
    printf("  (%i): ", t);
    mju_printMat(sensor_buffer.Get(t), 1, sensor_buffer.dim_);
    printf("\n");
  }

  // ----- estimator ----- //
  // horizon
  int horizon_estimator = 5;

  // initialize
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(horizon_estimator);
  mju_copy(estimator.configuration_.Data(), qpos_buffer.Data(),
           nq * horizon_estimator);
  mju_copy(estimator.configuration_prior_.Data(),
           estimator.configuration_.Data(), nq * horizon_estimator);
  mju_copy(estimator.action_.Data(), ctrl_buffer.Data() + nu,
           nu * (horizon_estimator - 2));
  mju_copy(estimator.force_measurement_.Data(),
           qfrc_actuator_buffer.Data() + nv, nv * (horizon_estimator - 2));
  mju_copy(estimator.sensor_measurement_.Data(), sensor_buffer.Data() + ns,
           ns * (horizon_estimator - 2));

  // // ----- random perturbation ----- //

  // // set configuration to nominal
  // mju_copy(estimator.configuration_.Data(), qpos.data(), nq * T);

  // // randomly perturb
  // for (int t = 0; t < T; t++) {
  //   // unpack
  //   double* q = estimator.configuration_.Data() + t * nq;

  //   // add noise
  //   for (int i = 0; i < nq; i++) {
  //     absl::BitGen gen_;
  //     q[i] += 0.001 * absl::Gaussian<double>(gen_, 0.0, 1.0);
  //   }
  // }

  // // cost
  // double cost_random = estimator.Cost(pool);

  // // change to band covariance 
  // estimator.band_covariance_ = true;

  // // change verbosity 
  // estimator.verbose_optimize_ = true;

  // // optimize
  // estimator.Optimize(estimator.configuration_length_, pool);

  // // error 
  // std::vector<double> configuration_error(nq * T);
  // mju_sub(configuration_error.data(), estimator.configuration_.Data(), qpos.data(), nq * T);

  // // test cost decrease
  // EXPECT_LE(estimator.cost_, cost_random);

  // // test gradient tolerance
  // EXPECT_NEAR(mju_norm(estimator.cost_gradient_.data(), nv * T) / (nv * T), 0.0, estimator.gradient_tolerance_);

  // // test recovered configuration trajectory
  // EXPECT_NEAR(mju_norm(configuration_error.data(), nq * T) / (nq * T), 0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
