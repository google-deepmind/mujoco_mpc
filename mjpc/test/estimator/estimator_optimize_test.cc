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

TEST(BatchOptimize, Particle2D) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // dimensions
  int nq = model->nq, nv = model->nv, nu = model->nu, ns = model->nsensordata;

  // threadpool
  ThreadPool pool(1);

  // ----- simulate ----- //

  // controller
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(100 * time);
    ctrl[1] = mju_cos(100 * time);
  };

  // trajectories
  int T = 32;
  std::vector<double> qpos(nq * (T + 1));
  std::vector<double> qvel(nv * (T + 1));
  std::vector<double> qacc(nv * T);
  std::vector<double> ctrl(nu * T);
  std::vector<double> qfrc_actuator(nv * T);
  std::vector<double> sensordata(ns * (T + 1));

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
  estimator.SetConfigurationLength(T);
  mju_copy(estimator.configuration_.Data(), qpos.data(), nq * T);
  mju_copy(estimator.configuration_prior_.Data(), qpos.data(), nq * T);
  mju_copy(estimator.force_measurement_.Data(), qfrc_actuator.data(), nv * T);
  mju_copy(estimator.sensor_measurement_.Data(), sensordata.data(), ns * T);

  // set weights
  estimator.scale_prior_ = 1.0;
  estimator.scale_sensor_[0] = 1.0;
  estimator.scale_force_[0] = 1.0;

  // ----- random perturbation ----- //

  // set configuration to nominal
  mju_copy(estimator.configuration_.Data(), qpos.data(), nq * T);

  // randomly perturb
  for (int t = 0; t < T; t++) {
    // unpack
    double* q = estimator.configuration_.Data() + t * nq;

    // add noise
    for (int i = 0; i < nq; i++) {
      absl::BitGen gen_;
      q[i] += 0.001 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // cost
  double cost_random = estimator.Cost(pool);

  // optimize
  estimator.Optimize(pool);

  // error
  std::vector<double> configuration_error(nq * T);
  mju_sub(configuration_error.data(), estimator.configuration_.Data(),
          qpos.data(), nq * T);

  // test cost decrease
  EXPECT_LE(estimator.cost_, cost_random);

  // test gradient tolerance
  EXPECT_NEAR(mju_norm(estimator.cost_gradient_.data(), nv * T) / (nv * T), 0.0,
              estimator.gradient_tolerance_);

  // test recovered configuration trajectory
  EXPECT_NEAR(mju_norm(configuration_error.data(), nq * T) / (nq * T), 0.0,
              1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(BatchOptimize, Box3D) {
  // load model
  mjModel* model = LoadTestModel("estimator/box/task2.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv, nu = model->nu;
  int ns = model->nsensordata;

  // pool
  int num_thread = 1;
  ThreadPool pool(num_thread);

  // ----- simulate ----- //

  // trajectories
  int T = 32;
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
  double qvel0[6] = {0.01, -0.02, -0.03, 0.001, -0.002, 0.003};

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
  estimator.SetConfigurationLength(T);
  estimator.time_scaling_ = true;
  mju_copy(estimator.configuration_.Data(), qpos.data(), nq * T);
  mju_copy(estimator.configuration_prior_.Data(), qpos.data(), nq * T);
  mju_copy(estimator.force_measurement_.Data(), qfrc_actuator.data(), nv * T);
  mju_copy(estimator.sensor_measurement_.Data(), sensordata.data(), ns * T);

  // ----- random perturbation ----- //

  // randomly perturb
  std::vector<double> noise(nv);

  // loop over configurations
  for (int t = 0; t < T; t++) {
    // unpack
    double* q = estimator.configuration_.Get(t);

    // add noise
    for (int i = 0; i < nv; i++) {
      absl::BitGen gen_;
      noise[i] = 0.01 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }

    // integrate configuration
    mj_integratePos(model, q, noise.data(), 1.0);
  }

  // optimize
  estimator.Optimize(pool);

  // error
  std::vector<double> configuration_error(nq * T);
  mju_sub(configuration_error.data(), estimator.configuration_.Data(),
          qpos.data(), nq * T);

  // test cost decrease
  EXPECT_LE(estimator.cost_, estimator.cost_initial_);

  // test gradient tolerance
  EXPECT_NEAR(mju_norm(estimator.cost_gradient_.data(), nv * T) / (nv * T), 0.0,
              1.0e-3);

  // test configuration trajectory error
  EXPECT_NEAR(mju_norm(configuration_error.data(), nq * T) / (nq * T), 0.0,
              1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(BatchOptimize, Quadruped) {
  // verbose 
  bool verbose = false;

  // load model
  mjModel* model = LoadTestModel("estimator/quadruped/task.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv, nu = model->nu;
  int ns = model->nsensordata;

  if (verbose) {
    printf("Quadruped dimensions:\n");
    printf("nq: %i\n", nq);
    printf("nv: %i\n", nv);
    printf("nu: %i\n", nu);
    printf("ns: %i\n", ns);
  }
  
  // trajectories
  int T = 64;
  if (verbose) printf("T: %i\n", T);

  // pool
  int num_thread = 10;
  ThreadPool pool(num_thread);

  if (verbose) printf("num thread: %i\n", num_thread);

  // ----- simulate ----- //

  std::vector<double> qpos(nq * (T + 1));
  std::vector<double> qvel(nv * (T + 1));
  std::vector<double> qacc(nv * T);
  std::vector<double> ctrl(nu * T);
  std::vector<double> qfrc_actuator(nv * T);
  std::vector<double> sensordata(ns * (T + 1));

  // reset
  mj_resetData(model, data);

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
  estimator.SetConfigurationLength(T);
  estimator.time_scaling_ = true;
  estimator.verbose_optimize_ = verbose;
  estimator.verbose_prior_ = verbose;
  mju_copy(estimator.configuration_.Data(), qpos.data(), nq * T);
  mju_copy(estimator.configuration_prior_.Data(), qpos.data(), nq * T);
  mju_copy(estimator.force_measurement_.Data(),qfrc_actuator.data(), nv * T);
  mju_copy(estimator.sensor_measurement_.Data(), sensordata.data(), ns * T);

  // ----- random perturbation ----- //

  // randomly perturb
  std::vector<double> noise(nv);

  // loop over configurations
  for (int t = 0; t < T; t++) {
    // unpack
    double* q = estimator.configuration_.Get(t);

    // add noise
    for (int i = 0; i < nv; i++) {
      // absl::BitGen gen_;
      noise[i] = 0.05; // * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }

    // integrate configuration
    mj_integratePos(model, q, noise.data(), 1.0);
  }

  // settings
  estimator.max_smoother_iterations_ = 1;
  estimator.max_line_search_ = 10;

  // estimator.max_smoother_iterations_ = 10;
  // estimator.max_line_search_ = 100;

  // set weights
  mju_fill(estimator.scale_sensor_.data(), 1.0, estimator.model_->nsensor);
  mju_fill(estimator.scale_force_.data(), 1.0, 4);

  // cost (pre)
  double cost_random = estimator.Cost(pool);

  // optimize
  estimator.Optimize(pool);

  // error
  std::vector<double> configuration_error(nq * T);
  mju_sub(configuration_error.data(), estimator.configuration_.Data(),
  qpos.data(), nq * T);

  // test cost decrease
  EXPECT_LE(estimator.cost_, cost_random);

  // test gradient tolerance
  // EXPECT_NEAR(mju_norm(estimator.cost_gradient_.data(), nv * T) / (nv * T), 0.0,
  //             1.0e-2);

  // // // test configuration trajectory error
  // EXPECT_NEAR(mju_norm(configuration_error.data(), nq * T) / (nq * T), 0.0,
  //             1.0e-2);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
