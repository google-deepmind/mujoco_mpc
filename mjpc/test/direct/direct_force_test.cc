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

#include <cstddef>
#include <vector>
#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include "gtest/gtest.h"
#include "mjpc/direct/direct.h"
#include "mjpc/test/load.h"
#include "mjpc/test/simulation.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(ForceCost, Particle) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // discrete inverse dynamics
  model->opt.enableflags |= mjENBL_INVDISCRETE;

  // dimension
  int nq = model->nq, nv = model->nv;

  // ----- rollout ----- //
  int T = 10;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(10 * time);
    ctrl[1] = 10 * mju_cos(10 * time);
  };
  sim.Rollout(controller);

  // ----- optimizer ----- //
  Direct optimizer(model, T);
  optimizer.settings.sensor_flag = false;
  optimizer.settings.force_flag = true;

  // weights
  optimizer.noise_process[0] = 1.0;
  optimizer.noise_process[1] = 2.0;

  // copy configuration, qfrc_actuator
  mju_copy(optimizer.configuration.Data(), sim.qpos.Data(), nq * T);
  mju_copy(optimizer.force_measurement.Data(), sim.qfrc_actuator.Data(),
           nv * T);

  // corrupt configurations
  absl::BitGen gen_;
  for (int t = 0; t < T; t++) {
    double* q = optimizer.configuration.Get(t);
    for (int i = 0; i < nq; i++) {
      q[i] += 1.0e-1 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- cost ----- //
  auto cost_inverse_dynamics = [&optimizer = optimizer, &model = model,
                                &data = data](const double* configuration) {
    // dimensions
    int nq = model->nq, nv = model->nv;
    int nres = nv * (optimizer.ConfigurationLength() - 2);

    // velocity
    std::vector<double> v1(nv);
    std::vector<double> v2(nv);

    // acceleration
    std::vector<double> a1(nv);

    // residual
    std::vector<double> residual(nres);

    // initialize
    double cost = 0.0;

    // time scaling
    double time_scale = 1.0;
    double time_scale2 = 1.0;
    if (optimizer.settings.time_scaling_force) {
      time_scale =
          optimizer.model->opt.timestep * optimizer.model->opt.timestep;
      time_scale2 = time_scale * time_scale;
    }

    // loop over predictions
    for (int k = 0; k < optimizer.ConfigurationLength() - 2; k++) {
      // time index
      int t = k + 1;

      // unpack
      double* rk = residual.data() + k * nv;
      const double* q0 = configuration + (t - 1) * nq;
      const double* q1 = configuration + (t + 0) * nq;
      const double* q2 = configuration + (t + 1) * nq;
      double* f1 = optimizer.force_measurement.Get(t);

      // velocity
      mj_differentiatePos(model, v1.data(), model->opt.timestep, q0, q1);
      mj_differentiatePos(model, v2.data(), model->opt.timestep, q1, q2);

      // acceleration
      mju_sub(a1.data(), v2.data(), v1.data(), nv);
      mju_scl(a1.data(), a1.data(), 1.0 / model->opt.timestep, nv);

      // set state
      mju_copy(data->qpos, q1, nq);
      mju_copy(data->qvel, v1.data(), nv);
      mju_copy(data->qacc, a1.data(), nv);

      // inverse dynamics
      mj_inverse(model, data);

      // inverse dynamics error
      mju_sub(rk, data->qfrc_inverse, f1, nv);

      // weighted residual
      std::vector<double> wr(nv);

      // loop over nv
      for (int i = 0; i < nv; i++) {
        // weight
        double weight = time_scale2 / optimizer.noise_process[i] / nv /
                        (optimizer.ConfigurationLength() - 2);
        wr[i] = weight * rk[i];
      }

      // add weighted norm
      cost += 0.5 * mju_dot(wr.data(), rk, nv);
    }

    // weighted cost
    return cost;
  };

  // ----- lambda ----- //

  // problem dimension
  int nvar = nv * T;

  // cost
  double cost_lambda = cost_inverse_dynamics(optimizer.configuration.Data());

  // gradient
  FiniteDifferenceGradient fdg(nvar);
  fdg.Compute(cost_inverse_dynamics, optimizer.configuration.Data(), nvar);

  // Hessian
  FiniteDifferenceHessian fdh(nvar);
  fdh.Compute(cost_inverse_dynamics, optimizer.configuration.Data(), nvar);

  // ----- optimizer ----- //
  std::vector<double> cost_gradient(nvar);
  std::vector<double> cost_hessian(nvar * nvar);
  std::vector<double> cost_hessian_band(nvar * (3 * nv));
  double cost_optimizer =
      optimizer.Cost(cost_gradient.data(), cost_hessian_band.data());

  // band to dense Hessian
  mju_band2Dense(cost_hessian.data(), cost_hessian_band.data(), nvar, 3 * nv, 0,
                 1);
  // ----- error ----- //

  // cost
  double cost_error = cost_optimizer - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(nvar);
  mju_sub(gradient_error.data(), cost_gradient.data(), fdg.gradient.data(),
          nvar);
  EXPECT_NEAR(mju_norm(gradient_error.data(), nvar) / nvar, 0.0, 1.0e-3);

  // Hessian
  std::vector<double> hessian_error(nvar * nvar);
  mju_sub(hessian_error.data(), cost_hessian.data(), fdh.hessian.data(),
          nvar * nvar);
  EXPECT_NEAR(mju_norm(hessian_error.data(), nvar * nvar) / (nvar * nvar), 0.0,
              1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(ForceCost, Box) {
  // load model
  mjModel* model = LoadTestModel("estimator/box/task0.xml");
  mjData* data = mj_makeData(model);

  // discrete inverse dynamics
  model->opt.enableflags |= mjENBL_INVDISCRETE;

  // dimension
  int nq = model->nq, nv = model->nv;

  // ----- rollout ----- //
  int T = 10;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {};
  double qvel[6] = {0.1, 0.2, 0.3, -0.1, -0.2, -0.3};
  sim.SetState(data->qpos, qvel);
  sim.Rollout(controller);

  // ----- optimizer ----- //
  Direct optimizer(model, T);
  optimizer.settings.sensor_flag = false;
  optimizer.settings.force_flag = true;

  // weights
  optimizer.noise_process[0] = 1.0;
  optimizer.noise_process[1] = 2.0;
  optimizer.noise_process[2] = 3.0;

  // copy configuration, qfrc_actuator
  mju_copy(optimizer.configuration.Data(), sim.qpos.Data(), nq * T);
  mju_copy(optimizer.force_measurement.Data(), sim.qfrc_actuator.Data(),
           nv * T);

  // corrupt configurations
  absl::BitGen gen_;
  for (int t = 0; t < T; t++) {
    double* q = optimizer.configuration.Get(t);
    double dq[6];
    for (int i = 0; i < nv; i++) {
      dq[i] = 1.0e-2 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    mj_integratePos(model, q, dq, model->opt.timestep);
  }

  // ----- cost ----- //
  auto cost_inverse_dynamics = [&optimizer = optimizer, &model = model,
                                &data = data](const double* update) {
    // dimensions
    int nq = model->nq, nv = model->nv;
    int nres = nv * (optimizer.ConfigurationLength() - 2);
    int T = optimizer.ConfigurationLength();

    // configuration
    std::vector<double> configuration(nq * T);
    mju_copy(configuration.data(), optimizer.configuration.Data(), nq * T);
    for (int t = 0; t < T; t++) {
      double* q = configuration.data() + t * nq;
      const double* dq = update + t * nv;
      mj_integratePos(model, q, dq, 1.0);
    }

    // velocity
    std::vector<double> v1(nv);
    std::vector<double> v2(nv);

    // acceleration
    std::vector<double> a1(nv);

    // residual
    std::vector<double> residual(nres);

    // initialize
    double cost = 0.0;

    // time scaling
    double time_scale2 = 1.0;
    if (optimizer.settings.time_scaling_force) {
      time_scale2 =
          optimizer.model->opt.timestep * optimizer.model->opt.timestep *
          optimizer.model->opt.timestep * optimizer.model->opt.timestep;
    }

    // loop over predictions
    for (int k = 0; k < optimizer.ConfigurationLength() - 2; k++) {
      // time index
      int t = k + 1;

      // unpack
      double* rk = residual.data() + k * nv;
      const double* q0 = configuration.data() + (t - 1) * nq;
      const double* q1 = configuration.data() + (t + 0) * nq;
      const double* q2 = configuration.data() + (t + 1) * nq;
      double* f1 = optimizer.force_measurement.Get(t);

      // velocity
      mj_differentiatePos(model, v1.data(), model->opt.timestep, q0, q1);
      mj_differentiatePos(model, v2.data(), model->opt.timestep, q1, q2);

      // acceleration
      mju_sub(a1.data(), v2.data(), v1.data(), nv);
      mju_scl(a1.data(), a1.data(), 1.0 / model->opt.timestep, nv);

      // set state
      mju_copy(data->qpos, q1, nq);
      mju_copy(data->qvel, v1.data(), nv);
      mju_copy(data->qacc, a1.data(), nv);

      // inverse dynamics
      mj_inverse(model, data);

      // inverse dynamics error
      mju_sub(rk, data->qfrc_inverse, f1, nv);

      // weighted residual
      std::vector<double> wr(nv);

      // loop over nv
      for (int i = 0; i < nv; i++) {
        // weight
        double weight = time_scale2 / optimizer.noise_process[i] / nv /
                        (optimizer.ConfigurationLength() - 2);

        // weighted residual
        wr[i] = weight * rk[i];
      }

      // cost
      cost += 0.5 * mju_dot(wr.data(), rk, nv);
    }

    // weighted cost
    return cost;
  };

  // ----- lambda ----- //

  // problem dimension
  int nvar = nv * T;

  // update
  std::vector<double> update(nvar);
  mju_zero(update.data(), nvar);

  // cost
  double cost_lambda = cost_inverse_dynamics(update.data());

  // gradient
  FiniteDifferenceGradient fdg(nvar);
  fdg.Compute(cost_inverse_dynamics, update.data(), nvar);

  // Hessian
  FiniteDifferenceHessian fdh(nvar);
  fdh.Compute(cost_inverse_dynamics, update.data(), nvar);

  // ----- optimizer ----- //
  std::vector<double> cost_gradient(nvar);
  double cost_optimizer = optimizer.Cost(cost_gradient.data(), NULL);

  // ----- error ----- //

  // cost
  EXPECT_NEAR(cost_optimizer - cost_lambda, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(nvar);
  mju_sub(gradient_error.data(), cost_gradient.data(), fdg.gradient.data(),
          nvar);
  EXPECT_NEAR(mju_norm(gradient_error.data(), nvar) / nvar, 0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(ForceCost, ParticleDamped) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task1D_damped.xml");
  mjData* data = mj_makeData(model);

  // discrete inverse dynamics
  model->opt.enableflags |= mjENBL_INVDISCRETE;

  // dimension
  int nq = model->nq, nv = model->nv;

  // ----- rollout ----- //
  int T = 3;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {};
  double q0[1] = {0.1};
  sim.SetState(q0, NULL);
  sim.Rollout(controller);

  // ----- optimizer ----- //
  Direct optimizer(model, T);
  optimizer.settings.sensor_flag = false;
  optimizer.settings.force_flag = true;

  // weights
  optimizer.noise_process[0] = 1.0;
  optimizer.noise_process[1] = 2.0;

  // copy configuration, qfrc_actuator
  mju_copy(optimizer.configuration.Data(), sim.qpos.Data(), nq * T);
  mju_copy(optimizer.force_measurement.Data(), sim.qfrc_actuator.Data(),
           nv * T);

  // corrupt configurations
  absl::BitGen gen_;
  for (int t = 0; t < T; t++) {
    double* q = optimizer.configuration.Get(t);
    for (int i = 0; i < nq; i++) {
      q[i] += 1.0e-5 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- cost ----- //
  auto cost_inverse_dynamics = [&optimizer = optimizer, &model = model,
                                &data = data](const double* configuration) {
    // dimensions
    int nq = model->nq, nv = model->nv;
    int nres = nv * (optimizer.ConfigurationLength() - 2);

    // velocity
    std::vector<double> v1(nv);
    std::vector<double> v2(nv);

    // acceleration
    std::vector<double> a1(nv);

    // residual
    std::vector<double> residual(nres);

    // initialize
    double cost = 0.0;

    // time scaling
    double time_scale = 1.0;
    double time_scale2 = 1.0;
    if (optimizer.settings.time_scaling_force) {
      time_scale =
          optimizer.model->opt.timestep * optimizer.model->opt.timestep;
      time_scale2 = time_scale * time_scale;
    }

    // loop over predictions
    for (int k = 0; k < optimizer.ConfigurationLength() - 2; k++) {
      // time index
      int t = k + 1;

      // unpack
      double* rk = residual.data() + k * nv;
      const double* q0 = configuration + (t - 1) * nq;
      const double* q1 = configuration + (t + 0) * nq;
      const double* q2 = configuration + (t + 1) * nq;
      double* f1 = optimizer.force_measurement.Get(t);

      // velocity
      mj_differentiatePos(model, v1.data(), model->opt.timestep, q0, q1);
      mj_differentiatePos(model, v2.data(), model->opt.timestep, q1, q2);

      // acceleration
      mju_sub(a1.data(), v2.data(), v1.data(), nv);
      mju_scl(a1.data(), a1.data(), 1.0 / model->opt.timestep, nv);

      // set state
      mju_copy(data->qpos, q1, nq);
      mju_copy(data->qvel, v1.data(), nv);
      mju_copy(data->qacc, a1.data(), nv);

      // inverse dynamics
      mj_inverse(model, data);

      // inverse dynamics error
      mju_sub(rk, data->qfrc_inverse, f1, nv);

      // weighted residual
      std::vector<double> wr(nv);

      // loop over nv
      for (int i = 0; i < nv; i++) {
        // weight
        double weight = time_scale2 / optimizer.noise_process[i] / nv /
                        (optimizer.ConfigurationLength() - 2);
        wr[i] = weight * rk[i];
      }

      // add weighted norm
      cost += 0.5 * mju_dot(wr.data(), rk, nv);
    }

    // weighted cost
    return cost;
  };

  // ----- lambda ----- //

  // problem dimension
  int nvar = nv * T;

  // cost
  double cost_lambda = cost_inverse_dynamics(optimizer.configuration.Data());

  // gradient
  FiniteDifferenceGradient fdg(nvar);
  fdg.Compute(cost_inverse_dynamics, optimizer.configuration.Data(), nvar);

  // Hessian
  FiniteDifferenceHessian fdh(nvar);
  fdh.Compute(cost_inverse_dynamics, optimizer.configuration.Data(), nvar);

  // ----- optimizer ----- //
  std::vector<double> cost_gradient(nvar);
  std::vector<double> cost_hessian(nvar * nvar);
  std::vector<double> cost_hessian_band(nvar * (3 * nv));
  double cost_optimizer =
      optimizer.Cost(cost_gradient.data(), cost_hessian_band.data());

  // band to dense Hessian
  mju_band2Dense(cost_hessian.data(), cost_hessian_band.data(), nvar, 3 * nv, 0,
                 1);
  // ----- error ----- //

  // cost
  double cost_error = cost_optimizer - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(nvar);
  mju_sub(gradient_error.data(), cost_gradient.data(), fdg.gradient.data(),
          nvar);
  EXPECT_NEAR(mju_norm(gradient_error.data(), nvar) / nvar, 0.0, 1.0e-3);

  // Hessian
  std::vector<double> hessian_error(nvar * nvar);
  mju_sub(hessian_error.data(), cost_hessian.data(), fdh.hessian.data(),
          nvar * nvar);
  EXPECT_NEAR(mju_norm(hessian_error.data(), nvar * nvar) / (nvar * nvar), 0.0,
              1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
