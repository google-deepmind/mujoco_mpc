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
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(MeasurementResidual, Particle) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv;

  // threadpool
  ThreadPool pool(1);

  // ----- configurations ----- //
  int T = 5;
  int dim_pos = nq * T;
  int dim_vel = nv * T;
  int dim_mea = model->nsensordata * T;
  int dim_res = model->nsensordata * (T - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> measurement(dim_mea);

  // random initialization
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < nq; i++) {
      absl::BitGen gen_;
      configuration[nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }

    for (int i = 0; i < model->nsensordata; i++) {
      absl::BitGen gen_;
      measurement[model->nsensordata * t + i] =
          absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);

  // copy configuration, measurement
  mju_copy(estimator.configuration_.Data(), configuration.data(), dim_pos);
  mju_copy(estimator.sensor_measurement_.Data(), measurement.data(), dim_mea);

  // ----- residual ----- //
  auto residual_measurement = [&measurement, &configuration_length = T, &model,
                               &data, nq,
                               nv](double* residual, const double* update) {
    // velocity
    std::vector<double> v1(nv);
    std::vector<double> v2(nv);

    // acceleration
    std::vector<double> a1(nv);

    // loop over predictions
    for (int k = 0; k < configuration_length - 2; k++) {
      // time index 
      int t = k + 1;

      // unpack
      double* rk = residual + k * model->nsensordata;
      const double* q0 = update + (t - 1) * nq;
      const double* q1 = update + (t + 0) * nq;
      const double* q2 = update + (t + 1) * nq;
      double* y1 = measurement.data() + t * model->nsensordata;

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
      mju_sub(rk, data->sensordata, y1, model->nsensordata);
    }
  };

  // initialize memory
  std::vector<double> residual(dim_res);
  std::vector<double> update(dim_vel);
  mju_copy(update.data(), configuration.data(), dim_pos);

  // ----- evaluate ----- //
  // (lambda)
  residual_measurement(residual.data(), update.data());

  // (estimator)
  estimator.ConfigurationToVelocityAcceleration();
  estimator.InverseDynamicsPrediction(pool);
  estimator.ResidualSensor();

  // error
  std::vector<double> residual_error(dim_mea);
  mju_sub(residual_error.data(), estimator.residual_sensor_.data(),
          residual.data(), dim_res);

  // test
  EXPECT_NEAR(mju_norm(residual_error.data(), dim_res) / (dim_res), 0.0,
              1.0e-5);

  // ----- Jacobian ----- //

  // finite-difference
  FiniteDifferenceJacobian fd(dim_res, dim_vel);
  fd.Compute(residual_measurement, update.data(), dim_res, dim_vel);

  // estimator
  estimator.InverseDynamicsDerivatives(pool);
  estimator.VelocityAccelerationDerivatives();
  for (int k = 0; k < estimator.prediction_length_; k++) {
    estimator.BlockSensor(k);
    estimator.SetBlockSensor(k);
  }

  // error
  std::vector<double> jacobian_error(dim_res * dim_vel);
  mju_sub(jacobian_error.data(), estimator.jacobian_sensor_.data(),
          fd.jacobian_.data(), dim_res * dim_vel);

  // test
  EXPECT_NEAR(
      mju_norm(jacobian_error.data(), dim_vel * dim_vel) / (dim_vel * dim_vel),
      0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(MeasurementResidual, Box) {
  // load model
  mjModel* model = LoadTestModel("estimator/box/task0.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv;

  // threadpool
  ThreadPool pool(1);

  // ----- configurations ----- //
  int T = 5;
  int dim_pos = nq * T;
  int dim_vel = nv * T;
  int dim_mea = model->nsensordata * T;
  int dim_res = model->nsensordata * (T - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> measurement(dim_mea);

  // random initialization
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < nq; i++) {
      absl::BitGen gen_;
      configuration[nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    // normalize quaternion
    mju_normalize4(configuration.data() + nq * t + 3);

    for (int i = 0; i < model->nsensordata; i++) {
      absl::BitGen gen_;
      measurement[model->nsensordata * t + i] =
          absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);

  // copy configuration, measurement
  mju_copy(estimator.configuration_.Data(), configuration.data(), dim_pos);
  mju_copy(estimator.sensor_measurement_.Data(), measurement.data(), dim_mea);

  // ----- residual ----- //
  auto residual_measurement = [&configuration, &measurement,
                               &configuration_length = T, &model, &data, nq,
                               nv](double* residual, const double* update) {
    // ----- integrate quaternion ----- //
    std::vector<double> qint(nq * configuration_length);
    mju_copy(qint.data(), configuration.data(), nq * configuration_length);

    // loop over configurations
    for (int t = 0; t < configuration_length; t++) {
      double* q = qint.data() + t * nq;
      const double* dq = update + t * nv;
      mj_integratePos(model, q, dq, 1.0);
    }

    // velocity
    std::vector<double> v1(nv);
    std::vector<double> v2(nv);

    // acceleration
    std::vector<double> a1(nv);

    // loop over predictions
    for (int k = 0; k < configuration_length - 2; k++) {
      // time index 
      int t = k + 1;

      // unpack
      double* rk = residual + k * model->nsensordata;
      double* q0 = qint.data() + (t - 1) * nq;
      double* q1 = qint.data() + (t + 0) * nq;
      double* q2 = qint.data() + (t + 1) * nq;
      double* y1 = measurement.data() + t * model->nsensordata;

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
      mju_sub(rk, data->sensordata, y1, model->nsensordata);
    }
  };

  // initialize memory
  std::vector<double> residual(dim_res);
  std::vector<double> update(dim_vel);
  mju_zero(update.data(), dim_vel);

  // ----- evaluate ----- //
  // (lambda)
  residual_measurement(residual.data(), update.data());

  // (estimator)
  estimator.ConfigurationToVelocityAcceleration();
  estimator.InverseDynamicsPrediction(pool);
  estimator.ResidualSensor();

  // // error
  std::vector<double> residual_error(dim_mea);
  mju_sub(residual_error.data(), estimator.residual_sensor_.data(),
          residual.data(), dim_res);

  // test
  EXPECT_NEAR(mju_norm(residual_error.data(), dim_res) / (dim_res), 0.0,
              1.0e-5);

  // ----- Jacobian ----- //

  // finite-difference
  FiniteDifferenceJacobian fd(dim_res, dim_vel);
  fd.Compute(residual_measurement, update.data(), dim_res, dim_vel);

  // estimator
  estimator.InverseDynamicsDerivatives(pool);
  estimator.VelocityAccelerationDerivatives();
  for (int k = 0; k < estimator.prediction_length_; k++) {
    estimator.BlockSensor(k);
    estimator.SetBlockSensor(k);
  }

  // error
  std::vector<double> jacobian_error(dim_res * dim_vel);
  mju_sub(jacobian_error.data(), estimator.jacobian_sensor_.data(),
          fd.jacobian_.data(), dim_res * dim_vel);

  // test
  EXPECT_NEAR(
      mju_norm(jacobian_error.data(), dim_res * dim_vel) / (dim_res * dim_vel),
      0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

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

  // ----- configurations ----- //
  int T = 5;
  int dim_pos = nq * T;
  int dim_vel = nv * T;
  int dim_mea = model->nsensordata * T;
  int dim_res = model->nsensordata * (T - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> measurement(dim_mea);

  // controller
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(10 * time);
    ctrl[1] = mju_cos(10 * time);
  };

  // reset
  mj_resetData(model, data);

  // rollout
  for (int t = 0; t < T - 1; t++) {
    // set control
    controller(data->ctrl, data->time);

    // forward computes instantaneous qacc
    mj_forward(model, data);

    // cache
    mju_copy(configuration.data() + t * nq, data->qpos, nq);
    mju_copy(measurement.data() + t * ns, data->sensordata, ns);

    // step using mj_Euler since mj_forward has been called
    // see mj_ step implementation here
    // https://github.com/deepmind/mujoco/blob/main/src/engine/engine_forward.c#L831
    mj_Euler(model, data);
  }

  // final cache
  mju_copy(configuration.data() + (T - 1) * nq, data->qpos, nq);
  mju_copy(measurement.data() + (T - 1) * model->nsensor, data->sensordata, ns);

  // random initialization
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < model->nsensordata; i++) {
      absl::BitGen gen_;
      measurement[model->nsensordata * t + i] +=
          0.025 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);

  // weights
  estimator.scale_sensor_[0] = 0.00125;
  estimator.scale_sensor_[1] = 0.0125;
  estimator.scale_sensor_[2] = 0.000125;
  estimator.scale_sensor_[3] = 0.0125;

  // norms
  estimator.norm_sensor_[0] = kL22;
  estimator.norm_sensor_[1] = kSmoothAbsLoss;
  estimator.norm_sensor_[2] = kQuadratic;
  estimator.norm_sensor_[3] = kL2;

  // parameters
  estimator.norm_parameters_sensor_[0 * 3 + 0] = 1.0;
  estimator.norm_parameters_sensor_[0 * 3 + 1] = 2.0;

  estimator.norm_parameters_sensor_[1 * 3 + 0] = 0.1;
  estimator.norm_parameters_sensor_[1 * 3 + 0] = 2.2;

  // estimator.norm_parameters_sensor_[2 * 3 + 0] = 0.01;
  // estimator.norm_parameters_sensor_[2 * 3 + 1] = 0.1;

  estimator.norm_parameters_sensor_[3 * 3 + 0] = 1.0;
  estimator.norm_parameters_sensor_[3 * 3 + 1] = 2.0;

  // copy configuration, measurement
  mju_copy(estimator.configuration_.Data(), configuration.data(), dim_pos);
  mju_copy(estimator.sensor_measurement_.Data(), measurement.data() + ns,
           dim_mea - ns);

  // ----- cost ----- //
  auto cost_measurement =
      [&measurement = estimator.sensor_measurement_,
       &configuration_length = estimator.configuration_length_, &model, &data,
       &dim_res, &weight = estimator.scale_sensor_,
       &params = estimator.norm_parameters_sensor_,
       &norms = estimator.norm_sensor_, nq, nv](const double* configuration) {
        // velocity
        std::vector<double> v1(nv);
        std::vector<double> v2(nv);

        // acceleration
        std::vector<double> a1(nv);

        // residual
        std::vector<double> residual(dim_res);

        // initialize
        double cost = 0.0;

        // loop over predictions
        for (int k = 0; k < configuration_length - 2; k++) {
          // time index 
          int t = k + 1;

          // unpack
          double* rk = residual.data() + k * model->nsensordata;
          const double* q0 = configuration + (t - 1) * nq;
          const double* q1 = configuration + (t + 0) * nq;
          const double* q2 = configuration + (t + 1) * nq;
          double* y1 = measurement.Get(t);

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
          mju_sub(rk, data->sensordata, y1, model->nsensordata);

          // loop over sensors
          int shift = 0;

          for (int i = 0; i < model->nsensor; i++) {
            // sensor dimension
            int dim_sensori = model->sensor_dim[i];

            // sensor residual
            double* rki = rk + shift;

            // time scaling
            double time_scale = 1.0;

            // stage 
            int stage = model->sensor_needstage[i];

            // time step
            double timestep = model->opt.timestep;

            // scale by sensor type
            if (stage == mjSTAGE_VEL) {
              time_scale = timestep * timestep;
            } else if (stage == mjSTAGE_ACC) {
              time_scale = timestep * timestep * timestep * timestep;
            }

            // add weighted norm
            cost += weight[i] / dim_sensori * time_scale *
                    Norm(NULL, NULL, rki, params.data() + 3 * i, dim_sensori,
                         norms[i]);

            // shift
            shift += dim_sensori;
          }
        }

        return cost;
      };

  // ----- lambda ----- //

  // cost
  double cost_lambda = cost_measurement(configuration.data());

  // gradient
  FiniteDifferenceGradient fdg(dim_vel);
  fdg.Compute(cost_measurement, configuration.data(), dim_vel);

  // Hessian
  FiniteDifferenceHessian fdh(dim_vel);
  fdh.Compute(cost_measurement, configuration.data(), dim_vel);

  // ----- estimator ----- //
  estimator.ConfigurationToVelocityAcceleration();
  estimator.InverseDynamicsPrediction(pool);
  estimator.InverseDynamicsDerivatives(pool);
  estimator.VelocityAccelerationDerivatives();
  estimator.ResidualSensor();
  for (int k = 0; k < estimator.prediction_length_; k++) {
    estimator.BlockSensor(k);
    estimator.SetBlockSensor(k);
  }

  // cost
  double cost_estimator =
      estimator.CostSensor(estimator.cost_gradient_sensor_.data(),
                           estimator.cost_hessian_sensor_.data());

  // ----- error ----- //

  // cost
  double cost_error = cost_estimator - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(dim_vel);
  mju_sub(gradient_error.data(), estimator.cost_gradient_sensor_.data(),
          fdg.gradient_.data(), dim_vel);
  EXPECT_NEAR(mju_norm(gradient_error.data(), dim_vel) / dim_vel, 0.0, 1.0e-3);

  // Hessian
  std::vector<double> hessian_error(dim_vel * dim_vel);
  mju_sub(hessian_error.data(), estimator.cost_hessian_sensor_.data(),
          fdh.hessian_.data(), dim_vel * dim_vel);
  EXPECT_NEAR(
      mju_norm(hessian_error.data(), dim_vel * dim_vel) / (dim_vel * dim_vel),
      0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(MeasurementCost, Box) {
  // load model
  mjModel* model = LoadTestModel("estimator/box/task0.xml");
  model->opt.timestep = 0.05;
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv;

  // threadpool
  ThreadPool pool(1);

  // configuration
  double qpos0[7] = {0.1, -0.2, 0.5, 0.0, 1.0, 0.0, 0.0};

  // ----- configurations ----- //
  int T = 5;
  int dim_pos = nq * T;
  int dim_vel = nv * T;
  int dim_mea = model->nsensordata * T;
  int dim_res = model->nsensordata * (T - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> measurement(dim_mea);

  // random initialization
  for (int t = 0; t < T; t++) {
    mju_copy(configuration.data() + t * nq, qpos0, nq);

    for (int i = 0; i < nq; i++) {
      absl::BitGen gen_;
      configuration[nq * t + i] = 0.01 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    // normalize quaternion
    mju_normalize4(configuration.data() + nq * t + 3);

    for (int i = 0; i < model->nsensordata; i++) {
      absl::BitGen gen_;
      measurement[model->nsensordata * t + i] =
          absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);

  // weights
  estimator.scale_sensor_[0] = 0.00125;
  estimator.scale_sensor_[1] = 0.0125;
  estimator.scale_sensor_[2] = 0.000125;
  estimator.scale_sensor_[3] = 0.125;
  estimator.scale_sensor_[4] = 0.0825;
  estimator.scale_sensor_[5] = 0.00355;

  // norms
  estimator.norm_sensor_[0] = kQuadratic;
  estimator.norm_sensor_[1] = kL2;
  estimator.norm_sensor_[2] = kSmoothAbsLoss;
  estimator.norm_sensor_[3] = kQuadratic;
  estimator.norm_sensor_[4] = kL22;
  estimator.norm_sensor_[5] = kSmoothAbsLoss;

  // parameters
  estimator.norm_parameters_sensor_[0 * 3 + 1] = 1.0;
  estimator.norm_parameters_sensor_[0 * 3 + 1] = 2.0;

  estimator.norm_parameters_sensor_[1 * 3 + 0] = 1.0;
  estimator.norm_parameters_sensor_[1 * 3 + 1] = 2.0;

  estimator.norm_parameters_sensor_[2 * 3 + 0] = 1.0;
  estimator.norm_parameters_sensor_[2 * 3 + 1] = 2.0;

  estimator.norm_parameters_sensor_[3 * 3 + 0] = 1.0;
  estimator.norm_parameters_sensor_[3 * 3 + 1] = 2.0;

  estimator.norm_parameters_sensor_[4 * 3 + 0] = 1.0;
  estimator.norm_parameters_sensor_[4 * 3 + 1] = 2.0;

  estimator.norm_parameters_sensor_[5 * 3 + 0] = 1.0;
  estimator.norm_parameters_sensor_[5 * 3 + 1] = 2.0;

  // copy configuration, measurement
  mju_copy(estimator.configuration_.Data(), configuration.data(), dim_pos);
  mju_copy(estimator.sensor_measurement_.Data(), measurement.data(), dim_mea);

  // ----- cost ----- //
  auto cost_measurement = [&configuration, &measurement, &dim_res,
                           &weight = estimator.scale_sensor_,
                           &params = estimator.norm_parameters_sensor_,
                           &norms = estimator.norm_sensor_,
                           &configuration_length = T, &model, &data, nq,
                           nv](const double* update) {
    // ----- integrate quaternion ----- //
    std::vector<double> qint(nq * configuration_length);
    mju_copy(qint.data(), configuration.data(), nq * configuration_length);

    // loop over configurations
    for (int t = 0; t < configuration_length; t++) {
      double* q = qint.data() + t * nq;
      const double* dq = update + t * nv;
      mj_integratePos(model, q, dq, 1.0);
    }

    // velocity
    std::vector<double> v1(nv);
    std::vector<double> v2(nv);

    // acceleration
    std::vector<double> a1(nv);

    // residual
    std::vector<double> residual(dim_res);

    // initialize
    double cost = 0.0;

    // loop over predictions
    for (int k = 0; k < configuration_length - 2; k++) {
      // time index 
      int t = k + 1;

      // unpack
      double* rk = residual.data() + k * model->nsensordata;
      double* q0 = qint.data() + (t - 1) * nq;
      double* q1 = qint.data() + (t + 0) * nq;
      double* q2 = qint.data() + (t + 1) * nq;
      double* y1 = measurement.data() + t * model->nsensordata;

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
      mju_sub(rk, data->sensordata, y1, model->nsensordata);

      // loop over sensors
      int shift = 0;

      for (int i = 0; i < model->nsensor; i++) {
        // sensor dimension
        int dim_sensori = model->sensor_dim[i];

        // sensor residual
        double* rki = rk + shift;

        // time scaling
        double time_scale = 1.0;

        // stage 
        int stage = model->sensor_needstage[i];

        // time step
        double timestep = model->opt.timestep;

        // scale by sensor type
        if (stage == mjSTAGE_VEL) {
          time_scale = timestep * timestep;
        } else if (stage == mjSTAGE_ACC) {
          time_scale = timestep * timestep * timestep * timestep;
        }

        // add weighted norm
        cost +=
            weight[i] / dim_sensori * time_scale *
            Norm(NULL, NULL, rki, params.data() + 3 * i, dim_sensori, norms[i]);

        // shift
        shift += dim_sensori;
      }
    }

    return cost;
  };

  // ----- lambda ----- //

  // cost
  std::vector<double> update(dim_vel);
  mju_zero(update.data(), dim_vel);
  double cost_lambda = cost_measurement(update.data());

  // gradient
  FiniteDifferenceGradient fdg(dim_vel);
  fdg.Compute(cost_measurement, update.data(), dim_vel);

  // ----- estimator ----- //
  estimator.ConfigurationToVelocityAcceleration();
  estimator.InverseDynamicsPrediction(pool);
  estimator.InverseDynamicsDerivatives(pool);
  estimator.VelocityAccelerationDerivatives();
  estimator.ResidualSensor();
  for (int k = 0; k < estimator.prediction_length_; k++) {
    estimator.BlockSensor(k);
    estimator.SetBlockSensor(k);
  }

  // cost
  double cost_estimator =
      estimator.CostSensor(estimator.cost_gradient_sensor_.data(), NULL);

  // ----- error ----- //

  // cost
  double cost_error = cost_estimator - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(dim_vel);
  mju_sub(gradient_error.data(), estimator.cost_gradient_sensor_.data(),
          fdg.gradient_.data(), dim_vel);
  EXPECT_NEAR(mju_norm(gradient_error.data(), dim_vel) / dim_vel, 0.0, 1.0e-2);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
