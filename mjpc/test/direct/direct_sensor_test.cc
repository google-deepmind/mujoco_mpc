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
#include "mjpc/norm.h"
#include "mjpc/direct/direct.h"
#include "mjpc/test/load.h"
#include "mjpc/test/simulation.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(SensorCost, Particle) {
  // load model
  // note: needs to be a linear system to satisfy Gauss-Newton Hessian
  // approximation
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // discrete inverse dynamics
  model->opt.enableflags |= mjENBL_INVDISCRETE;

  // dimension
  int nq = model->nq, nv = model->nv, ns = model->nsensordata;

  // ----- rollout ----- //
  int T = 3;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(10 * time);
    ctrl[1] = 10 * mju_cos(10 * time);
  };
  sim.Rollout(controller);

  // ----- optimizer ----- //
  Direct optimizer(model, T);
  optimizer.settings.sensor_flag = true;
  optimizer.settings.force_flag = false;
  optimizer.settings.first_step_position_sensors = true;
  optimizer.settings.last_step_position_sensors = true;
  optimizer.settings.last_step_velocity_sensors = true;

  // copy configuration, measurement
  mju_copy(optimizer.configuration.Data(), sim.qpos.Data(), nq * T);
  mju_copy(optimizer.sensor_measurement.Data(), sim.sensor.Data(), ns * T);

  // corrupt configurations
  absl::BitGen gen_;
  for (int t = 0; t < T; t++) {
    double* q = optimizer.configuration.Get(t);
    for (int i = 0; i < nq; i++) {
      q[i] += 1.0e-1;
    }
  }

  // weights
  optimizer.noise_sensor[0] = 1.1e-1;
  optimizer.noise_sensor[1] = 2.2e-1;
  optimizer.noise_sensor[2] = 3.3e-1;
  optimizer.noise_sensor[3] = 4.4e-1;

  // TODO(taylor): test difference norms

  // ----- cost ----- //
  auto cost_measurement = [&optimizer = optimizer, &model = model,
                           &data = data](const double* configuration) {
    // dimensions
    int nq = model->nq, nv = model->nv, ns = model->nsensordata;
    int nres = ns * optimizer.ConfigurationLength();

    // velocity
    std::vector<double> v1(nv);
    std::vector<double> v2(nv);

    // acceleration
    std::vector<double> a1(nv);

    // residual
    std::vector<double> residual(nres);
    std::fill(residual.begin(), residual.end(), 0.0);

    // initialize
    double cost = 0.0;

    // time scaling
    double time_scale = 1.0;
    double time_scale2 = 1.0;
    if (optimizer.settings.time_scaling_sensor) {
      time_scale =
          optimizer.model->opt.timestep * optimizer.model->opt.timestep;
      time_scale2 = time_scale * time_scale;
    }

    // loop over predictions
    for (int t = 0; t < optimizer.ConfigurationLength(); t++) {
      if (t == 0) {
        // first configuration
        mju_copy(data->qpos, configuration, nq);
        mju_zero(data->qvel, nv);
        mju_zero(data->qacc, nv);
        mju_zero(data->ctrl, model->nu);

        // first sensor
        double* y0 = optimizer.sensor_measurement.Get(t);

        // residual
        double* rk = residual.data();

        // position sensors
        mj_fwdPosition(model, data);
        mj_sensorPos(model, data);
        if (model->opt.enableflags & (mjENBL_ENERGY)) {
          mj_energyPos(model, data);
        }

        // measurement error
        mju_sub(rk, data->sensordata, y0, ns);

        // loop over sensors
        int shift = 0;

        for (int i = 0; i < model->nsensor; i++) {
          // sensor dimension
          int nsi = model->sensor_dim[i];

          // skip velocity, acceleration sensors
          if (model->sensor_needstage[i] != mjSTAGE_POS) {
            shift += nsi;
            continue;
          }

          // sensor residual
          double* rki = rk + shift;

          // weight
          double weight = 1.0 / optimizer.noise_sensor[i] / nsi /
                          optimizer.ConfigurationLength();

          // first time step
          weight *= optimizer.settings.first_step_position_sensors;

          // parameters
          double* pi =
              optimizer.norm_parameters_sensor.data() + kMaxNormParameters * i;

          // norm
          NormType normi = optimizer.norm_type_sensor[i];

          // add weighted norm
          cost += weight * Norm(NULL, NULL, rki, pi, nsi, normi);

          // shift
          shift += nsi;
        }
        continue;
      } else if (t == optimizer.ConfigurationLength() - 1) {
        // unpack
        double* rk = residual.data() + t * ns;
        const double* q0 = configuration + (t - 1) * nq;
        const double* q1 = configuration + (t + 0) * nq;
        double* y1 = optimizer.sensor_measurement.Get(t);

        // velocity
        mj_differentiatePos(model, v1.data(), model->opt.timestep, q0, q1);

        // first configuration
        mju_copy(data->qpos, q1, nq);
        mju_copy(data->qvel, v1.data(), nv);
        mju_zero(data->qacc, nv);
        mju_zero(data->ctrl, model->nu);

        // position sensors
        mj_fwdPosition(model, data);
        mj_sensorPos(model, data);
        if (model->opt.enableflags & (mjENBL_ENERGY)) {
          mj_energyPos(model, data);
        }

        // velocity sensors
        mj_fwdVelocity(model, data);
        mj_sensorVel(model, data);
        if (model->opt.enableflags & (mjENBL_ENERGY)) {
          mj_energyVel(model, data);
        }

        // measurement error
        mju_sub(rk, data->sensordata, y1, ns);

        // loop over sensors
        int shift = 0;

        for (int i = 0; i < model->nsensor; i++) {
          // sensor dimension
          int nsi = model->sensor_dim[i];

          // skip acceleration sensors
          if (model->sensor_needstage[i] == mjSTAGE_ACC) {
            shift += nsi;
            continue;
          }

          // sensor residual
          double* rki = rk + shift;

          // weight
          double weight = 1.0 / optimizer.noise_sensor[i] / nsi /
                          optimizer.ConfigurationLength();

          // first time step
          weight *= (optimizer.settings.last_step_position_sensors ||
                     optimizer.settings.last_step_velocity_sensors);

          // parameters
          double* pi =
              optimizer.norm_parameters_sensor.data() + kMaxNormParameters * i;

          // norm
          NormType normi = optimizer.norm_type_sensor[i];

          // add weighted norm
          cost += weight * Norm(NULL, NULL, rki, pi, nsi, normi);

          // shift
          shift += nsi;
        }
        continue;
      }

      // unpack
      double* rk = residual.data() + t * ns;
      const double* q0 = configuration + (t - 1) * nq;
      const double* q1 = configuration + (t + 0) * nq;
      const double* q2 = configuration + (t + 1) * nq;
      double* y1 = optimizer.sensor_measurement.Get(t);

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
        // sensor stage
        int sensor_stage = model->sensor_needstage[i];

        // sensor dimension
        int nsi = model->sensor_dim[i];

        // sensor residual
        double* rki = rk + shift;

        // time weight
        double time_weight = 1.0;
        if (sensor_stage == mjSTAGE_VEL) {
          time_weight = time_scale;
        } else if (sensor_stage == mjSTAGE_ACC) {
          time_weight = time_scale2;
        }

        // weight
        double weight = time_weight / optimizer.noise_sensor[i] / nsi /
                        optimizer.ConfigurationLength();

        // parameters
        double* pi =
            optimizer.norm_parameters_sensor.data() + kMaxNormParameters * i;

        // norm
        NormType normi = optimizer.norm_type_sensor[i];

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
  double cost_lambda = cost_measurement(optimizer.configuration.Data());

  // gradient
  FiniteDifferenceGradient fdg(nvar);
  fdg.Compute(cost_measurement, optimizer.configuration.Data(), nvar);

  // Hessian
  FiniteDifferenceHessian fdh(nvar);
  fdh.Compute(cost_measurement, optimizer.configuration.Data(), nvar);

  // ----- optimizer ----- //
  // cost
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
  EXPECT_NEAR(mju_norm(gradient_error.data(), nvar) / nvar, 0.0, 1.0e-4);

  // Hessian
  std::vector<double> hessian_error(nvar * nvar);
  mju_sub(hessian_error.data(), cost_hessian.data(), fdh.hessian.data(),
          nvar * nvar);
  EXPECT_NEAR(mju_norm(hessian_error.data(), nvar) / (nvar * nvar), 0.0,
              1.0e-4);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(SensorCost, Box) {
  // load model
  // note: needs to be a linear system to satisfy Gauss-Newton Hessian
  // approximation
  mjModel* model = LoadTestModel("estimator/box/task0.xml");
  mjData* data = mj_makeData(model);

  // discrete inverse dynamics
  model->opt.enableflags |= mjENBL_INVDISCRETE;

  // dimension
  int nq = model->nq, nv = model->nv, ns = model->nsensordata;

  // ----- rollout ----- //
  int T = 3;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {};
  double qvel[6] = {0.1, 0.2, 0.3, -0.1, -0.2, -0.3};
  sim.SetState(data->qpos, qvel);
  sim.Rollout(controller);

  // ----- optimizer ----- //
  Direct optimizer(model, T);
  optimizer.settings.first_step_position_sensors = true;
  optimizer.settings.last_step_position_sensors = true;
  optimizer.settings.last_step_velocity_sensors = true;

  // copy configuration, measurement
  mju_copy(optimizer.configuration.Data(), sim.qpos.Data(), nq * T);
  mju_copy(optimizer.sensor_measurement.Data(), sim.sensor.Data(), ns * T);

  // corrupt configurations
  absl::BitGen gen_;
  for (int t = 0; t < T; t++) {
    double* q = optimizer.configuration.Get(t);
    double dq[6];
    for (int i = 0; i < nv; i++) {
      dq[i] = 1.0e-1;
    }
    mj_integratePos(model, q, dq, model->opt.timestep);
  }

  // weights
  optimizer.noise_sensor[0] = 1.1e-2;
  optimizer.noise_sensor[1] = 2.2e-2;
  optimizer.noise_sensor[2] = 3.3e-2;
  optimizer.noise_sensor[3] = 1.0e-2;
  optimizer.noise_sensor[4] = 2.0e-2;
  optimizer.noise_sensor[5] = 3.0e-2;
  optimizer.noise_sensor[6] = 4.0e-2;
  optimizer.noise_sensor[7] = 5.0e-2;
  optimizer.noise_sensor[8] = 6.0e-2;
  optimizer.noise_sensor[9] = 7.0e-2;
  optimizer.noise_sensor[10] = 8.0e-2;
  optimizer.noise_sensor[11] = 9.0e-2;
  optimizer.noise_sensor[12] = 10.0e-2;

  // TODO(taylor): test difference norms

  // ----- cost ----- //
  auto cost_measurement = [&optimizer = optimizer, &model = model,
                           &data = data](const double* update) {
    // dimensions
    int nq = model->nq, nv = model->nv, ns = model->nsensordata;
    int nres = ns * optimizer.ConfigurationLength();
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

    // time scale
    double time_scale = 1.0;
    double time_scale2 = 1.0;
    if (optimizer.settings.time_scaling_sensor) {
      time_scale =
          optimizer.model->opt.timestep * optimizer.model->opt.timestep;
      time_scale2 = time_scale * time_scale;
    }

    // loop over predictions
    for (int t = 0; t < optimizer.ConfigurationLength(); t++) {
      if (t == 0) {
        // first configuration
        mju_copy(data->qpos, configuration.data(), nq);
        mju_zero(data->qvel, nv);
        mju_zero(data->qacc, nv);
        mju_zero(data->ctrl, model->nu);

        // first sensor
        double* y0 = optimizer.sensor_measurement.Get(t);

        // residual
        double* rk = residual.data();

        // position sensors
        mj_fwdPosition(model, data);
        mj_sensorPos(model, data);
        if (model->opt.enableflags & (mjENBL_ENERGY)) {
          mj_energyPos(model, data);
        }

        // measurement error
        mju_sub(rk, data->sensordata, y0, ns);

        // loop over sensors
        int shift = 0;

        for (int i = 0; i < model->nsensor; i++) {
          // sensor dimension
          int nsi = model->sensor_dim[i];

          // skip velocity, acceleration sensors
          if (model->sensor_needstage[i] != mjSTAGE_POS) {
            shift += nsi;
            continue;
          }

          // sensor residual
          double* rki = rk + shift;

          // weight
          double weight = 1.0 / optimizer.noise_sensor[i] / nsi /
                          optimizer.ConfigurationLength();

          // first time step
          weight *= optimizer.settings.first_step_position_sensors;

          // parameters
          double* pi =
              optimizer.norm_parameters_sensor.data() + kMaxNormParameters * i;

          // norm
          NormType normi = optimizer.norm_type_sensor[i];

          // add weighted norm
          cost += weight * Norm(NULL, NULL, rki, pi, nsi, normi);

          // shift
          shift += nsi;
        }
        continue;
      } else if (t == optimizer.ConfigurationLength() - 1) {
        // unpack
        double* rk = residual.data() + t * ns;
        const double* q0 = configuration.data() + (t - 1) * nq;
        const double* q1 = configuration.data() + (t + 0) * nq;
        double* y1 = optimizer.sensor_measurement.Get(t);

        // velocity
        mj_differentiatePos(model, v1.data(), model->opt.timestep, q0, q1);

        // first configuration
        mju_copy(data->qpos, q1, nq);
        mju_copy(data->qvel, v1.data(), nv);
        mju_zero(data->qacc, nv);
        mju_zero(data->ctrl, model->nu);

        // position sensors
        mj_fwdPosition(model, data);
        mj_sensorPos(model, data);
        if (model->opt.enableflags & (mjENBL_ENERGY)) {
          mj_energyPos(model, data);
        }

        // velocity sensors
        mj_fwdVelocity(model, data);
        mj_sensorVel(model, data);
        if (model->opt.enableflags & (mjENBL_ENERGY)) {
          mj_energyVel(model, data);
        }

        // measurement error
        mju_sub(rk, data->sensordata, y1, ns);

        // loop over sensors
        int shift = 0;

        for (int i = 0; i < model->nsensor; i++) {
          // sensor dimension
          int nsi = model->sensor_dim[i];

          // skip acceleration sensors
          if (model->sensor_needstage[i] == mjSTAGE_ACC) {
            shift += nsi;
            continue;
          }

          // sensor residual
          double* rki = rk + shift;

          // weight
          double weight = 1.0 / optimizer.noise_sensor[i] / nsi /
                          optimizer.ConfigurationLength();

          // first time step
          weight *= (optimizer.settings.last_step_position_sensors ||
                     optimizer.settings.last_step_velocity_sensors);

          // parameters
          double* pi =
              optimizer.norm_parameters_sensor.data() + kMaxNormParameters * i;

          // norm
          NormType normi = optimizer.norm_type_sensor[i];

          // add weighted norm
          cost += weight * Norm(NULL, NULL, rki, pi, nsi, normi);

          // shift
          shift += nsi;
        }
        continue;
      }

      // unpack
      double* rk = residual.data() + t * ns;
      const double* q0 = configuration.data() + (t - 1) * nq;
      const double* q1 = configuration.data() + (t + 0) * nq;
      const double* q2 = configuration.data() + (t + 1) * nq;
      double* y1 = optimizer.sensor_measurement.Get(t);

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
        // sensor stage
        int sensor_stage = model->sensor_needstage[i];

        // sensor dimension
        int nsi = model->sensor_dim[i];

        // sensor residual
        double* rki = rk + shift;

        // time weight
        double time_weight = 1.0;
        if (sensor_stage == mjSTAGE_VEL) {
          time_weight = time_scale;
        } else if (sensor_stage == mjSTAGE_ACC) {
          time_weight = time_scale2;
        }

        // weight
        double weight = time_weight / optimizer.noise_sensor[i] / nsi /
                        optimizer.ConfigurationLength();

        // parameters
        double* pi =
            optimizer.norm_parameters_sensor.data() + kMaxNormParameters * i;

        // norm
        NormType normi = optimizer.norm_type_sensor[i];

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

  // ----- optimizer ----- //

  // cost
  optimizer.settings.sensor_flag = true;
  optimizer.settings.force_flag = false;

  std::vector<double> cost_gradient(nvar);
  double cost_optimizer = optimizer.Cost(cost_gradient.data(), NULL);

  // ----- error ----- //

  // cost
  double cost_error = cost_optimizer - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(nvar);
  mju_sub(gradient_error.data(), cost_gradient.data(), fdg.gradient.data(),
          nvar);
  EXPECT_NEAR(mju_norm(gradient_error.data(), nvar) / nvar, 0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
