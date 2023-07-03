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
#include "mjpc/estimators/estimator.h"
#include "mjpc/test/load.h"
#include "mjpc/test/simulation.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(MeasurementCost, Particle) {
  // load model
  // note: needs to be a linear system to satisfy Gauss-Newton Hessian
  // approximation
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv, ns = model->nsensordata;

  // threadpool
  ThreadPool pool(1);

  // ----- rollout ----- //
  int T = 10;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(10 * time);
    ctrl[1] = 10 * mju_cos(10 * time);
  };
  sim.Rollout(controller);

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);

  // copy configuration, measurement
  mju_copy(estimator.configuration.Data(), sim.qpos.Data(), nq * T);
  mju_copy(estimator.sensor_measurement.Data(), sim.sensor.Data(), ns * T);

  // corrupt configurations
  absl::BitGen gen_;
  for (int t = 0; t < T; t++) {
    double* q = estimator.configuration.Get(t);
    for (int i = 0; i < nq; i++) {
      q[i] += 1.0e-1 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // weights
  estimator.scale_sensor[0] = 1.1e-1;
  estimator.scale_sensor[1] = 2.2e-1;
  estimator.scale_sensor[2] = 3.3e-1;
  estimator.scale_sensor[3] = 4.4e-1;

  // TODO(taylor): test difference norms

  // ----- cost ----- //
  auto cost_measurement = [&estimator = estimator, &model = model,
                           &data = data](const double* configuration) {
    // dimensions
    int nq = model->nq, nv = model->nv, ns = model->nsensordata;
    int nres = ns * estimator.PredictionLength();

    // velocity
    std::vector<double> v1(nv);
    std::vector<double> v2(nv);

    // acceleration
    std::vector<double> a1(nv);

    // residual
    std::vector<double> residual(nres);

    // initialize
    double cost = 0.0;

    // loop over predictions
    for (int k = 0; k < estimator.PredictionLength(); k++) {
      // time index
      int t = k + 1;

      // unpack
      double* rk = residual.data() + k * ns;
      const double* q0 = configuration + (t - 1) * nq;
      const double* q1 = configuration + (t + 0) * nq;
      const double* q2 = configuration + (t + 1) * nq;
      double* y1 = estimator.sensor_measurement.Get(t);

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

      // measurement error
      mju_sub(rk, data->sensordata, y1, ns);

      // loop over sensors
      int shift = 0;

      for (int i = 0; i < model->nsensor; i++) {
        // sensor dimension
        int nsi = model->sensor_dim[i];

        // sensor residual
        double* rki = rk + shift;

        // weight
        double weight =
            estimator.scale_sensor[i] / nsi / estimator.PredictionLength();

        // parameters
        double* pi =
            estimator.norm_parameters_sensor.data() + MAX_NORM_PARAMETERS * i;

        // norm
        NormType normi = estimator.norm_sensor[i];

        // add weighted norm
        cost += weight * Norm(NULL, NULL, rki, pi, nsi, normi);

        // shift
        shift += nsi;
      }
    }

    return cost;
  };

  // problem dimension
  int nvar = nv * T;

  // ----- lambda ----- //

  // cost
  double cost_lambda = cost_measurement(estimator.configuration.Data());

  // gradient
  FiniteDifferenceGradient fdg(nvar);
  fdg.Compute(cost_measurement, estimator.configuration.Data(), nvar);

  // Hessian
  FiniteDifferenceHessian fdh(nvar);
  fdh.Compute(cost_measurement, estimator.configuration.Data(), nvar);

  // ----- estimator ----- //
  estimator.settings.prior_flag = false;
  estimator.settings.sensor_flag = true;
  estimator.settings.force_flag = false;
  // cost
  double cost_estimator = estimator.Cost(estimator.cost_gradient.data(),
                                         estimator.cost_hessian.data(), pool);

  // ----- error ----- //

  // cost
  double cost_error = cost_estimator - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(nvar);
  mju_sub(gradient_error.data(), estimator.cost_gradient.data(),
          fdg.gradient.data(), nvar);
  EXPECT_NEAR(mju_norm(gradient_error.data(), nvar) / nvar, 0.0, 1.0e-4);

  // Hessian
  std::vector<double> hessian_error(nvar * nvar);
  mju_sub(hessian_error.data(), estimator.cost_hessian.data(),
          fdh.hessian.data(), nvar * nvar);
  EXPECT_NEAR(mju_norm(hessian_error.data(), nvar) / (nvar * nvar), 0.0,
              1.0e-4);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(MeasurementCost, Box) {
  // load model
  // note: needs to be a linear system to satisfy Gauss-Newton Hessian
  // approximation
  mjModel* model = LoadTestModel("estimator/box/task0.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv, ns = model->nsensordata;

  // threadpool
  ThreadPool pool(1);

  // ----- rollout ----- //
  int T = 10;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {};
  double qvel[6] = {0.1, 0.2, 0.3, -0.1, -0.2, -0.3};
  sim.SetState(data->qpos, qvel);
  sim.Rollout(controller);

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);

  // copy configuration, measurement
  mju_copy(estimator.configuration.Data(), sim.qpos.Data(), nq * T);
  mju_copy(estimator.sensor_measurement.Data(), sim.sensor.Data(), ns * T);

  // corrupt configurations
  absl::BitGen gen_;
  for (int t = 0; t < T; t++) {
    double* q = estimator.configuration.Get(t);
    double dq[6];
    for (int i = 0; i < nv; i++) {
      dq[i] = 1.0e-1 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    mj_integratePos(model, q, dq, model->opt.timestep);
  }

  // weights
  estimator.scale_sensor[0] = 1.1e-2;
  estimator.scale_sensor[1] = 2.2e-2;
  estimator.scale_sensor[2] = 3.3e-2;
  estimator.scale_sensor[3] = 1.0e-2;
  estimator.scale_sensor[4] = 2.0e-2;
  estimator.scale_sensor[5] = 3.0e-2;
  estimator.scale_sensor[6] = 4.0e-2;
  estimator.scale_sensor[7] = 5.0e-2;
  estimator.scale_sensor[8] = 6.0e-2;
  estimator.scale_sensor[9] = 7.0e-2;
  estimator.scale_sensor[10] = 8.0e-2;
  estimator.scale_sensor[11] = 9.0e-2;
  estimator.scale_sensor[12] = 10.0e-2;

  // TODO(taylor): test difference norms

  // ----- cost ----- //
  auto cost_measurement = [&estimator = estimator, &model = model,
                           &data = data](const double* update) {
    // dimensions
    int nq = model->nq, nv = model->nv, ns = model->nsensordata;
    int nres = ns * estimator.PredictionLength();
    int T = estimator.ConfigurationLength();

    // configuration
    std::vector<double> configuration(nq * T);
    mju_copy(configuration.data(), estimator.configuration.Data(), nq * T);
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

    // loop over predictions
    for (int k = 0; k < estimator.PredictionLength(); k++) {
      // time index
      int t = k + 1;

      // unpack
      double* rk = residual.data() + k * ns;
      const double* q0 = configuration.data() + (t - 1) * nq;
      const double* q1 = configuration.data() + (t + 0) * nq;
      const double* q2 = configuration.data() + (t + 1) * nq;
      double* y1 = estimator.sensor_measurement.Get(t);

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

      // measurement error
      mju_sub(rk, data->sensordata, y1, ns);

      // loop over sensors
      int shift = 0;

      for (int i = 0; i < model->nsensor; i++) {
        // sensor dimension
        int nsi = model->sensor_dim[i];

        // sensor residual
        double* rki = rk + shift;

        // weight
        double weight =
            estimator.scale_sensor[i] / nsi / estimator.PredictionLength();

        // parameters
        double* pi =
            estimator.norm_parameters_sensor.data() + MAX_NORM_PARAMETERS * i;

        // norm
        NormType normi = estimator.norm_sensor[i];

        // add weighted norm
        cost += weight * Norm(NULL, NULL, rki, pi, nsi, normi);

        // shift
        shift += nsi;
      }
    }

    return cost;
  };

  // problem dimension
  int nvar = nv * T;

  // ----- lambda ----- //

  // update
  std::vector<double> update(nv * T);
  mju_zero(update.data(), nv * T);

  // cost
  double cost_lambda = cost_measurement(update.data());

  // gradient
  FiniteDifferenceGradient fdg(nvar);
  fdg.Compute(cost_measurement, update.data(), nvar);

  // Hessian
  FiniteDifferenceHessian fdh(nvar);
  fdh.Compute(cost_measurement, update.data(), nvar);

  // ----- estimator ----- //

  // cost
  estimator.settings.prior_flag = false;
  estimator.settings.sensor_flag = true;
  estimator.settings.force_flag = false;
  double cost_estimator = estimator.Cost(estimator.cost_gradient.data(),
                                         estimator.cost_hessian.data(), pool);

  // ----- error ----- //

  // cost
  double cost_error = cost_estimator - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(nvar);
  mju_sub(gradient_error.data(), estimator.cost_gradient.data(),
          fdg.gradient.data(), nvar);
  EXPECT_NEAR(mju_norm(gradient_error.data(), nvar) / nvar, 0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
