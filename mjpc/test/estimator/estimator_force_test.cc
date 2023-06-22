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
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(ForceResidual, Particle) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv;

  // threadpool
  ThreadPool pool(4);

  // ----- configurations ----- //
  int T = 5;
  int dim_pos = nq * T;
  int dim_vel = nv * T;
  int dim_id = nv * T;
  int dim_res = nv * (T - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> qfrc_actuator(dim_id);

  // random initialization
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < nq; i++) {
      absl::BitGen gen_;
      configuration[nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }

    for (int i = 0; i < nv; i++) {
      absl::BitGen gen_;
      qfrc_actuator[nv * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);

  // copy configuration, qfrc_actuator
  mju_copy(estimator.configuration_.Data(), configuration.data(), dim_pos);
  mju_copy(estimator.force_measurement_.Data(), qfrc_actuator.data(), dim_id);

  // ----- residual ----- //
  auto residual_inverse_dynamics = [&qfrc_actuator, &configuration_length = T,
                                    &model, &data, nq, nv](
                                       double* residual, const double* update) {
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
      double* rk = residual + k * nv;
      const double* q0 = update + (t - 1) * nq;
      const double* q1 = update + (t + 0) * nq;
      const double* q2 = update + (t + 1) * nq;
      double* f1 = qfrc_actuator.data() + t * nv;

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
    }
  };

  // initialize memory
  std::vector<double> residual(dim_res);
  std::vector<double> update(dim_vel);
  mju_copy(update.data(), configuration.data(), dim_pos);

  // ----- evaluate ----- //
  // (lambda)
  residual_inverse_dynamics(residual.data(), update.data());

  // (estimator)
  estimator.ConfigurationToVelocityAcceleration();
  estimator.InverseDynamicsPrediction(pool);
  estimator.ResidualForce();

  // error
  std::vector<double> residual_error(dim_id);
  mju_sub(residual_error.data(), estimator.residual_force_.data(),
          residual.data(), dim_res);

  // test
  EXPECT_NEAR(mju_norm(residual_error.data(), dim_res) / (dim_res), 0.0,
              1.0e-5);

  // ----- Jacobian ----- //

  // finite-difference
  FiniteDifferenceJacobian fd(dim_res, dim_vel);
  fd.Compute(residual_inverse_dynamics, update.data(), dim_res, dim_vel);

  // estimator
  estimator.InverseDynamicsDerivatives(pool);
  estimator.VelocityAccelerationDerivatives();

  for (int k = 0; k < estimator.prediction_length_; k++) {
    estimator.BlockForce(k);
    estimator.SetBlockForce(k);
  }

  // error
  std::vector<double> jacobian_error(dim_res * dim_vel);
  mju_sub(jacobian_error.data(), estimator.jacobian_force_.data(),
          fd.jacobian.data(), dim_res * dim_vel);

  // test
  EXPECT_NEAR(
      mju_norm(jacobian_error.data(), dim_res * dim_vel) / (dim_res * dim_vel),
      0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(ForceResidual, Box) {
  // load model
  mjModel* model = LoadTestModel("estimator/box/task0.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv;

  // threadpool
  ThreadPool pool(4);

  // initial configuration
  double qpos0[7] = {0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0};

  // ----- configurations ----- //
  int T = 5;
  int dim_pos = nq * T;
  int dim_vel = nv * T;
  int dim_id = nv * T;
  int dim_res = nv * (T - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> qfrc_actuator(dim_id);

  // random initialization
  for (int t = 0; t < T; t++) {
    mju_copy(configuration.data() + t * nq, qpos0, nq);

    for (int i = 0; i < nq; i++) {
      absl::BitGen gen_;
      configuration[nq * t + i] +=
          1.0e-2 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    // normalize quaternion
    mju_normalize4(configuration.data() + nq * t + 3);

    for (int i = 0; i < nv; i++) {
      absl::BitGen gen_;
      qfrc_actuator[nv * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);

  // copy configuration, qfrc_actuator
  mju_copy(estimator.configuration_.Data(), configuration.data(), dim_pos);
  mju_copy(estimator.force_measurement_.Data(), qfrc_actuator.data(), dim_id);

  // ----- residual ----- //
  auto residual_inverse_dynamics =
      [&configuration = estimator.configuration_,
       &qfrc_actuator = estimator.force_measurement_, &configuration_length = T,
       &model, &data, nq, nv](double* residual, const double* update) {
        // ----- integrate quaternion ----- //
        std::vector<double> qint(nq * configuration_length);
        mju_copy(qint.data(), configuration.Data(), nq * configuration_length);

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
          double* rk = residual + k * nv;
          double* q0 = qint.data() + (t - 1) * nq;
          double* q1 = qint.data() + (t + 0) * nq;
          double* q2 = qint.data() + (t + 1) * nq;
          double* f1 = qfrc_actuator.Get(t);

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
        }
      };

  // initialize memory
  std::vector<double> residual(dim_res);
  std::vector<double> update(dim_vel);
  mju_zero(update.data(), dim_vel);

  // ----- evaluate ----- //
  // (lambda)
  residual_inverse_dynamics(residual.data(), update.data());

  // (estimator)
  estimator.ConfigurationToVelocityAcceleration();
  estimator.InverseDynamicsPrediction(pool);
  estimator.ResidualForce();

  // error
  std::vector<double> residual_error(dim_id);
  mju_sub(residual_error.data(), estimator.residual_force_.data(),
          residual.data(), dim_res);

  // test
  EXPECT_NEAR(mju_norm(residual_error.data(), dim_res) / (dim_res), 0.0,
              1.0e-5);

  // ----- Jacobian ----- //

  // finite-difference
  FiniteDifferenceJacobian fd(dim_res, dim_vel);
  fd.Compute(residual_inverse_dynamics, update.data(), dim_res, dim_vel);

  // estimator
  estimator.InverseDynamicsDerivatives(pool);
  estimator.VelocityAccelerationDerivatives();
  for (int k = 0; k < estimator.prediction_length_; k++) {
    estimator.BlockForce(k);
    estimator.SetBlockForce(k);
  }

  // error
  std::vector<double> jacobian_error(dim_res * dim_vel);
  mju_sub(jacobian_error.data(), estimator.jacobian_force_.data(),
          fd.jacobian.data(), dim_res * dim_vel);

  // test
  EXPECT_NEAR(
      mju_norm(jacobian_error.data(), dim_res * dim_vel) / (dim_res * dim_vel),
      0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(ForceCost, Particle) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv;

  // threadpool
  ThreadPool pool(4);

  // initial configuration
  double qpos0[4] = {0.1, 0.3, -0.01, 0.25};

  // ----- configurations ----- //
  int T = 5;
  int dim_pos = nq * T;
  int dim_vel = nv * T;
  int dim_id = nv * T;
  int dim_res = nv * (T - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> qfrc_actuator(dim_id);

  // random initialization
  for (int t = 0; t < T; t++) {
    mju_copy(configuration.data() + t * nq, qpos0, nq);

    for (int i = 0; i < nq; i++) {
      absl::BitGen gen_;
      configuration[nq * t + i] +=
          0.01 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }

    for (int i = 0; i < nv; i++) {
      absl::BitGen gen_;
      qfrc_actuator[nv * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);
  estimator.time_scaling_ = false;

  // weights
  estimator.scale_force_[0] = 0.0055;
  estimator.scale_force_[1] = 0.0325;
  estimator.scale_force_[2] = 0.00025;

  // norms
  estimator.norm_force_[0] = kQuadratic;
  estimator.norm_force_[1] = kQuadratic;
  estimator.norm_force_[2] = kQuadratic;

  // copy configuration, qfrc_actuator
  mju_copy(estimator.configuration_.Data(), configuration.data(), dim_pos);
  mju_copy(estimator.force_measurement_.Data(), qfrc_actuator.data(), dim_id);

  // ----- cost ----- //
  auto cost_inverse_dynamics = [&qfrc_actuator = estimator.force_measurement_,
                                &configuration_length =
                                    estimator.configuration_length_,
                                &dim_res, &weight = estimator.scale_force_,
                                &params = estimator.norm_parameters_force_,
                                &norms = estimator.norm_force_, &model, &data,
                                nq, nv](const double* configuration) {
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
      double* rk = residual.data() + k * nv;
      const double* q0 = configuration + (t - 1) * nq;
      const double* q1 = configuration + (t + 0) * nq;
      const double* q2 = configuration + (t + 1) * nq;
      double* f1 = qfrc_actuator.Get(t);

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

      // add weighted norm
      cost += weight[2] / nv / (configuration_length - 2) *
              Norm(NULL, NULL, rk, params.data() + MAX_NORM_PARAMETERS * 2, nv,
                   norms[2]);
    }

    // weighted cost
    return cost;
  };

  // ----- lambda ----- //

  // cost
  double cost_lambda = cost_inverse_dynamics(configuration.data());

  // gradient
  FiniteDifferenceGradient fdg(dim_vel);
  fdg.Compute(cost_inverse_dynamics, configuration.data(), dim_vel);

  // Hessian
  FiniteDifferenceHessian fdh(dim_vel);
  fdh.Compute(cost_inverse_dynamics, configuration.data(), dim_vel);

  // ----- estimator ----- //
  estimator.ConfigurationToVelocityAcceleration();
  estimator.InverseDynamicsPrediction(pool);
  estimator.InverseDynamicsDerivatives(pool);
  estimator.VelocityAccelerationDerivatives();
  estimator.ResidualForce();
  for (int k = 0; k < estimator.prediction_length_; k++) {
    estimator.BlockForce(k);
    estimator.SetBlockForce(k);
  }

  // cost
  double cost_estimator =
      estimator.CostForce(estimator.cost_gradient_force_.data(),
                          estimator.cost_hessian_force_.data());

  // ----- error ----- //

  // cost
  double cost_error = cost_estimator - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(dim_vel);
  mju_sub(gradient_error.data(), estimator.cost_gradient_force_.data(),
          fdg.gradient.data(), dim_vel);
  EXPECT_NEAR(mju_norm(gradient_error.data(), dim_vel) / dim_vel, 0.0, 1.0e-3);

  // Hessian
  std::vector<double> hessian_error(dim_vel * dim_vel);
  mju_sub(hessian_error.data(), estimator.cost_hessian_force_.data(),
          fdh.hessian.data(), dim_vel * dim_vel);
  EXPECT_NEAR(
      mju_norm(hessian_error.data(), dim_vel * dim_vel) / (dim_vel * dim_vel),
      0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(ForceCost, Box) {
  // load model
  mjModel* model = LoadTestModel("estimator/box/task0.xml");
  model->opt.timestep = 0.035;
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv;

  // threadpool
  ThreadPool pool(4);

  // initial configuration
  double qpos0[7] = {0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0};

  // ----- configurations ----- //
  int T = 3;
  int dim_pos = nq * T;
  int dim_vel = nv * T;
  int dim_id = nv * T;
  int dim_res = nv * (T - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> qfrc_actuator(dim_id);

  // random initialization
  absl::BitGen gen_;
  for (int t = 0; t < T; t++) {
    mju_copy(configuration.data() + t * nq, qpos0, nq);

    for (int i = 0; i < nq; i++) {
      configuration[nq * t + i] +=
          1.0e-2 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    // normalize quaternion
    mju_normalize4(configuration.data() + nq * t + 3);

    for (int i = 0; i < nv; i++) {
      qfrc_actuator[nv * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);
  estimator.time_scaling_ = true;

  // weights
  estimator.scale_force_[0] = 0.00125;
  estimator.scale_force_[1] = 0.0125;

  // norms
  estimator.norm_force_[0] = kQuadratic;
  estimator.norm_force_[1] = kL2;

  // parameters
  // estimator.norm_parameters_force_[MAX_NORM_PARAMETERS * 0]
  estimator.norm_parameters_force_[MAX_NORM_PARAMETERS * 1 + 0] = 0.1;
  estimator.norm_parameters_force_[MAX_NORM_PARAMETERS * 1 + 1] = 0.075;

  // copy configuration, qfrc_actuator
  mju_copy(estimator.configuration_.Data(), configuration.data(), dim_pos);
  mju_copy(estimator.force_measurement_.Data(), qfrc_actuator.data(), dim_id);

  // ----- cost ----- //
  auto cost_inverse_dynamics = [&configuration = estimator.configuration_,
                                &qfrc_actuator = estimator.force_measurement_,
                                &configuration_length = T, &model, &dim_res,
                                &weight = estimator.scale_force_,
                                &params = estimator.norm_parameters_force_,
                                &norms = estimator.norm_force_, &data, nq,
                                nv](const double* update) {
    // ----- integrate quaternion ----- //
    std::vector<double> qint(nq * configuration_length);
    mju_copy(qint.data(), configuration.Data(), nq * configuration_length);

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
      double* rk = residual.data() + k * nv;
      double* q0 = qint.data() + (t - 1) * nq;
      double* q1 = qint.data() + (t + 0) * nq;
      double* q2 = qint.data() + (t + 1) * nq;
      double* f1 = qfrc_actuator.Get(t);

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

      // force residual
      double* rt_pos = rk;
      double* rt_rot = rk + 3;

      // time scaling
      double timestep = model->opt.timestep;
      double time_scale = timestep * timestep * timestep * timestep;

      // add weighted norm for free-joint position
      cost += weight[0] / 3 * time_scale / (configuration_length - 2) *
              Norm(NULL, NULL, rt_pos, params.data() + MAX_NORM_PARAMETERS * 0,
                    3, norms[0]);

      // add weighted norm for free-joint rotation
      cost += weight[1] / 3 * time_scale / (configuration_length - 2) *
              Norm(NULL, NULL, rt_rot, params.data() + MAX_NORM_PARAMETERS * 1,
                    3, norms[1]);
    }

    return cost;
  };

  // ----- lambda ----- //
  std::vector<double> update(dim_vel);
  mju_zero(update.data(), dim_vel);

  // cost
  double cost_lambda = cost_inverse_dynamics(update.data());

  // gradient
  FiniteDifferenceGradient fdg(dim_vel);
  fdg.Compute(cost_inverse_dynamics, update.data(), dim_vel);

  // ----- estimator ----- //

  // compute intermediate terms
  estimator.ConfigurationToVelocityAcceleration();
  estimator.InverseDynamicsPrediction(pool);
  estimator.InverseDynamicsDerivatives(pool);
  estimator.VelocityAccelerationDerivatives();
  estimator.ResidualForce();
  for (int k = 0; k < estimator.prediction_length_; k++) {
    estimator.BlockForce(k);
    estimator.SetBlockForce(k);
  }

  // cost
  double cost_estimator =
      estimator.CostForce(estimator.cost_gradient_force_.data(), NULL);

  // ----- error ----- //

  // cost
  double cost_error = cost_estimator - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(dim_vel);
  mju_sub(gradient_error.data(), estimator.cost_gradient_force_.data(),
          fdg.gradient.data(), dim_vel);
  EXPECT_NEAR(mju_norm(gradient_error.data(), dim_vel) / dim_vel, 0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(ForceBug, Particle) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // set time step 
  model->opt.timestep = 0.1;

  // reset 
  mj_resetData(model, data);

  // -- rollout -- // 

  // data
  int T = 3;
  std::vector<double> qpos(T * model->nq);
  std::vector<double> qvel(T * model->nv);
  std::vector<double> qacc(T * model->nv);
  std::vector<double> ctrl(T * model->nu);
  std::vector<double> qfrc(T * model->nv);
  std::vector<double> sensor(T * model->nsensordata);
  std::vector<double> time(T);

  // simulate 
  for (int t = 0; t < T; t++) {
    // ctrl 
    data->ctrl[0] = 10.0;
    data->ctrl[1] = 0.0;

    // forward 
    mj_forward(model, data);

    // cache 
    mju_copy(qpos.data() + t * model->nq, data->qpos, model->nq);
    mju_copy(qvel.data() + t * model->nv, data->qvel, model->nv);
    mju_copy(qacc.data() + t * model->nv, data->qacc, model->nv);
    mju_copy(ctrl.data() + t * model->nu, data->ctrl, model->nu);
    mju_copy(qfrc.data() + t * model->nv, data->qfrc_actuator, model->nv);
    mju_copy(sensor.data() + t * model->nsensordata, data->sensordata, model->nsensordata);
    time[t] = data->time;

    // Euler
    mj_Euler(model, data);
  }

  // show data 
  for (int t = 0; t < T; t++) {
    printf("t = %i\n", t);
    printf("configuration = ");
    mju_printMat(qpos.data() + t * model->nq, 1, model->nq);

    printf("velocity = ");
    mju_printMat(qvel.data() + t * model->nv, 1, model->nv);

    printf("acceleration = ");
    mju_printMat(qacc.data() + t * model->nv, 1, model->nv);

    printf("sensor = ");
    mju_printMat(sensor.data() + t * model->nsensordata, 1, model->nsensordata);

    printf("force = ");
    mju_printMat(qfrc.data() + t * model->nv, 1, model->nv);

    printf("ctrl = ");
    mju_printMat(ctrl.data() + t * model->nu, 1, model->nu);

    printf("time = %.4f\n", time[t]);
    printf("\n");
  }

  // estimator 
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);

  estimator.scale_prior_ = 0.0;

  estimator.scale_sensor_[0] = 1.0;
  estimator.scale_sensor_[1] = 1.0;
  estimator.scale_sensor_[2] = 1.0;
  estimator.scale_sensor_[3] = 1.0;

  estimator.scale_force_[0] = 1.0;
  estimator.scale_force_[1] = 1.0;
  estimator.scale_force_[2] = 1.0;

  estimator.time_scaling_ = false;
  estimator.prior_flag_ = false;
  estimator.sensor_flag_ = true;
  estimator.force_flag_ = true;
  estimator.max_smoother_iterations_ = 1;
  estimator.update_prior_weight_ = false;

  // set data 
  for (int t = 0; t < T; t++) {
    // double q[2] = {0.001, 0.002};
    // estimator.configuration_.Set(q, t);
    // estimator.configuration_.Set(qpos.data() + t * model->nq, t);
    estimator.sensor_measurement_.Set(sensor.data() + t * model->nsensordata, t);
    estimator.force_measurement_.Set(qfrc.data() + t * model->nv, t);
    estimator.ctrl_.Set(ctrl.data() + t * model->nu, t);
    estimator.time_.Set(time.data() + t, t);
  }

  // optimize 
  ThreadPool pool(1);
  estimator.Optimize(pool);


  // cost 
  estimator.verbose_cost_ = true;
  estimator.PrintCost();

  printf("\n");
  printf("search iterations = %i\n", estimator.iterations_line_search_);
  printf("smoother iterations = %i\n", estimator.iterations_smoother_);
  printf("step size = %.5f\n", estimator.step_size_);
  printf("regularization = %.5f\n", estimator.regularization_);
  printf("gradient norm = %.5f\n", estimator.gradient_norm_);
  printf("\n");

  // show results 
  for (int t = 0; t < T; t++) {
    printf("t = %i\n", t);
    printf("configuration = ");
    mju_printMat(estimator.configuration_.Get(t), 1, model->nq);

    printf("velocity = ");
    mju_printMat(estimator.velocity_.Get(t), 1, model->nv);

    printf("acceleration = ");
    mju_printMat(estimator.acceleration_.Get(t), 1, model->nv);

    printf("sensor measurement = ");
    mju_printMat(estimator.sensor_measurement_.Get(t), 1, model->nsensordata);

    printf("sensor prediction = ");
    mju_printMat(estimator.sensor_prediction_.Get(t), 1, model->nsensordata);

    printf("force measurement = ");
    mju_printMat(estimator.force_measurement_.Get(t), 1, model->nv);

    printf("force prediction = ");
    mju_printMat(estimator.force_prediction_.Get(t), 1, model->nv);

    printf("ctrl = ");
    mju_printMat(ctrl.data() + t * model->nu, 1, model->nu);

    printf("time = %.4f\n", time[t]);
    printf("\n");
  }

  // residuals 
  printf("prior residual: ");
  mju_printMat(estimator.residual_prior_.data(), 1, T * model->nv);

  printf("sensor residual: ");
  mju_printMat(estimator.residual_sensor_.data(), 1, (T - 2) * model->nsensordata);

  printf("force residual: ");
  mju_printMat(estimator.residual_force_.data(), 1, (T - 2) * model->nv);

  // delete
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
