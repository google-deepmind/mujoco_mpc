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
#include "mjpc/estimators/batch/estimator.h"
#include "mjpc/test/load.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(InverseDynamicsResidual, Particle) {
  // load model
  mjModel* model = LoadTestModel("particle2D.xml");
  mjData* data = mj_makeData(model);

  // ----- configurations ----- //
  int history = 5;
  int dim_pos = model->nq * history;
  int dim_vel = model->nv * history;
  int dim_id = model->nv * history;
  int dim_res = model->nv * (history - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> qfrc_actuator(dim_id);

  // random initialization
  for (int t = 0; t < history; t++) {
    for (int i = 0; i < model->nq; i++) {
      absl::BitGen gen_;
      configuration[model->nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }

    for (int i = 0; i < model->nv; i++) {
      absl::BitGen gen_;
      qfrc_actuator[model->nv * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.configuration_length_ = history;

  // copy configuration, qfrc_actuator
  mju_copy(estimator.configuration_.data(), configuration.data(), dim_pos);
  mju_copy(estimator.qfrc_actuator_.data(), qfrc_actuator.data(), dim_id);

  // ----- residual ----- //
  auto residual_inverse_dynamics =
      [&qfrc_actuator, &configuration_length = history, &model, &data](
          double* residual, const double* update) {
        // velocity
        std::vector<double> v1(model->nv);
        std::vector<double> v2(model->nv);

        // acceleration
        std::vector<double> a1(model->nv);

        // loop over time
        for (int t = 0; t < configuration_length - 2; t++) {
          // unpack
          double* rt = residual + t * model->nv;
          const double* q0 = update + t * model->nq;
          const double* q1 = update + (t + 1) * model->nq;
          const double* q2 = update + (t + 2) * model->nq;
          double* f1 = qfrc_actuator.data() + t * model->nv;

          // velocity
          mj_differentiatePos(model, v1.data(), model->opt.timestep, q0, q1);
          mj_differentiatePos(model, v2.data(), model->opt.timestep, q1, q2);

          // acceleration
          mju_sub(a1.data(), v2.data(), v1.data(), model->nv);
          mju_scl(a1.data(), a1.data(), 1.0 / model->opt.timestep, model->nv);

          // set state
          mju_copy(data->qpos, q1, model->nq);
          mju_copy(data->qvel, v1.data(), model->nv);
          mju_copy(data->qacc, a1.data(), model->nv);

          // inverse dynamics
          mj_inverse(model, data);

          // inverse dynamics error
          mju_sub(rt, data->qfrc_inverse, f1, model->nv);
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
  // finite-difference velocities
  ConfigurationToVelocity(estimator.velocity_.data(),
                          estimator.configuration_.data(),
                          estimator.configuration_length_, estimator.model_);

  // finite-difference accelerations
  VelocityToAcceleration(estimator.acceleration_.data(),
                         estimator.velocity_.data(),
                         estimator.configuration_length_ - 1, estimator.model_);

  // compute inverse dynamics
  estimator.ComputeInverseDynamics();
  estimator.ResidualInverseDynamics();

  // error
  std::vector<double> residual_error(dim_id);
  mju_sub(residual_error.data(), estimator.residual_inverse_dynamics_.data(),
          residual.data(), dim_res);

  // test
  EXPECT_NEAR(mju_norm(residual_error.data(), dim_res) / (dim_res), 0.0,
              1.0e-5);

  // ----- Jacobian ----- //

  // finite-difference
  FiniteDifferenceJacobian fd(dim_res, dim_vel);
  fd.Compute(residual_inverse_dynamics, update.data(), dim_res, dim_vel);

  // estimator
  estimator.ModelDerivatives();
  estimator.VelocityJacobianBlocks();
  estimator.AccelerationJacobianBlocks();
  estimator.JacobianInverseDynamics();

  // error
  std::vector<double> jacobian_error(dim_res * dim_vel);
  mju_sub(jacobian_error.data(), estimator.jacobian_inverse_dynamics_.data(),
          fd.jacobian_.data(), dim_res * dim_vel);

  // test
  EXPECT_NEAR(
      mju_norm(jacobian_error.data(), dim_vel * dim_vel) / (dim_vel * dim_vel),
      0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(InverseDynamicsResidual, Box) {
  // load model
  mjModel* model = LoadTestModel("box3D.xml");
  mjData* data = mj_makeData(model);

  // initial configuration
  double qpos0[7] = {0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0};

  // ----- configurations ----- //
  int history = 5;
  int dim_pos = model->nq * history;
  int dim_vel = model->nv * history;
  int dim_id = model->nv * history;
  int dim_res = model->nv * (history - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> qfrc_actuator(dim_id);

  // random initialization
  for (int t = 0; t < history; t++) {
    mju_copy(configuration.data() + t * model->nq, qpos0, model->nq);

    for (int i = 0; i < model->nq; i++) {
      absl::BitGen gen_;
      configuration[model->nq * t + i] +=
          1.0e-2 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    // normalize quaternion
    mju_normalize4(configuration.data() + model->nq * t + 3);

    for (int i = 0; i < model->nv; i++) {
      absl::BitGen gen_;
      qfrc_actuator[model->nv * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.configuration_length_ = history;

  // copy configuration, qfrc_actuator
  mju_copy(estimator.configuration_.data(), configuration.data(), dim_pos);
  mju_copy(estimator.qfrc_actuator_.data(), qfrc_actuator.data(), dim_id);

  // ----- residual ----- //
  auto residual_inverse_dynamics = [&configuration = estimator.configuration_,
                                    &qfrc_actuator = estimator.qfrc_actuator_,
                                    &configuration_length = history, &model,
                                    &data](double* residual,
                                           const double* update) {
    // ----- integrate quaternion ----- //
    std::vector<double> qint(model->nq * configuration_length);
    mju_copy(qint.data(), configuration.data(),
             model->nq * configuration_length);

    // loop over configurations
    for (int t = 0; t < configuration_length; t++) {
      double* q = qint.data() + t * model->nq;
      const double* dq = update + t * model->nv;
      mj_integratePos(model, q, dq, 1.0);
    }

    // velocity
    std::vector<double> v1(model->nv);
    std::vector<double> v2(model->nv);

    // acceleration
    std::vector<double> a1(model->nv);

    // loop over time
    for (int t = 0; t < configuration_length - 2; t++) {
      // unpack
      double* rt = residual + t * model->nv;
      double* q0 = qint.data() + t * model->nq;
      double* q1 = qint.data() + (t + 1) * model->nq;
      double* q2 = qint.data() + (t + 2) * model->nq;
      double* f1 = qfrc_actuator.data() + t * model->nv;

      // velocity
      mj_differentiatePos(model, v1.data(), model->opt.timestep, q0, q1);
      mj_differentiatePos(model, v2.data(), model->opt.timestep, q1, q2);

      // acceleration
      mju_sub(a1.data(), v2.data(), v1.data(), model->nv);
      mju_scl(a1.data(), a1.data(), 1.0 / model->opt.timestep, model->nv);

      // set state
      mju_copy(data->qpos, q1, model->nq);
      mju_copy(data->qvel, v1.data(), model->nv);
      mju_copy(data->qacc, a1.data(), model->nv);

      // inverse dynamics
      mj_inverse(model, data);

      // inverse dynamics error
      mju_sub(rt, data->qfrc_inverse, f1, model->nv);
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
  // finite-difference velocities
  ConfigurationToVelocity(estimator.velocity_.data(),
                          estimator.configuration_.data(),
                          estimator.configuration_length_, estimator.model_);

  // finite-difference accelerations
  VelocityToAcceleration(estimator.acceleration_.data(),
                         estimator.velocity_.data(),
                         estimator.configuration_length_ - 1, estimator.model_);

  // compute inverse dynamics
  estimator.ComputeInverseDynamics();
  estimator.ResidualInverseDynamics();

  // error
  std::vector<double> residual_error(dim_id);
  mju_sub(residual_error.data(), estimator.residual_inverse_dynamics_.data(),
          residual.data(), dim_res);

  // test
  EXPECT_NEAR(mju_norm(residual_error.data(), dim_res) / (dim_res), 0.0,
              1.0e-5);

  // ----- Jacobian ----- //

  // finite-difference
  FiniteDifferenceJacobian fd(dim_res, dim_vel);
  fd.Compute(residual_inverse_dynamics, update.data(), dim_res, dim_vel);

  // estimator
  estimator.ModelDerivatives();
  estimator.VelocityJacobianBlocks();
  estimator.AccelerationJacobianBlocks();
  estimator.JacobianInverseDynamics();

  // error
  std::vector<double> jacobian_error(dim_res * dim_vel);
  mju_sub(jacobian_error.data(), estimator.jacobian_inverse_dynamics_.data(),
          fd.jacobian_.data(), dim_res * dim_vel);

  // test
  EXPECT_NEAR(
      mju_norm(jacobian_error.data(), dim_res * dim_vel) / (dim_res * dim_vel),
      0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(InverseDynamicsCost, Particle) {
  // load model
  mjModel* model = LoadTestModel("particle2D.xml");
  mjData* data = mj_makeData(model);

  // initial configuration
  double qpos0[4] = {0.1, 0.3, -0.01, 0.25};

  // ----- configurations ----- //
  int history = 3;
  int dim_pos = model->nq * history;
  int dim_vel = model->nv * history;
  int dim_id = model->nv * history;
  int dim_res = model->nv * (history - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> qfrc_actuator(dim_id);

  // random initialization
  for (int t = 0; t < history; t++) {
    mju_copy(configuration.data() + t * model->nq, qpos0, model->nq);

    for (int i = 0; i < model->nq; i++) {
      absl::BitGen gen_;
      configuration[model->nq * t + i] +=
          0.01 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }

    for (int i = 0; i < model->nv; i++) {
      absl::BitGen gen_;
      qfrc_actuator[model->nv * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.configuration_length_ = history;
  estimator.weight_inverse_dynamics_ = 0.0035;

  // copy configuration, qfrc_actuator
  mju_copy(estimator.configuration_.data(), configuration.data(), dim_pos);
  mju_copy(estimator.qfrc_actuator_.data(), qfrc_actuator.data(), dim_id);

  // ----- cost ----- //
  auto cost_inverse_dynamics = [&qfrc_actuator, &configuration_length = history,
                                &dim_res,
                                &weight = estimator.weight_inverse_dynamics_,
                                &model, &data](const double* configuration) {
    // velocity
    std::vector<double> v1(model->nv);
    std::vector<double> v2(model->nv);

    // acceleration
    std::vector<double> a1(model->nv);

    // residual
    std::vector<double> residual(dim_res);

    // loop over time
    for (int t = 0; t < configuration_length - 2; t++) {
      // unpack
      double* rt = residual.data() + t * model->nv;
      const double* q0 = configuration + t * model->nq;
      const double* q1 = configuration + (t + 1) * model->nq;
      const double* q2 = configuration + (t + 2) * model->nq;
      double* f1 = qfrc_actuator.data() + t * model->nv;

      // velocity
      mj_differentiatePos(model, v1.data(), model->opt.timestep, q0, q1);
      mj_differentiatePos(model, v2.data(), model->opt.timestep, q1, q2);

      // acceleration
      mju_sub(a1.data(), v2.data(), v1.data(), model->nv);
      mju_scl(a1.data(), a1.data(), 1.0 / model->opt.timestep, model->nv);

      // set state
      mju_copy(data->qpos, q1, model->nq);
      mju_copy(data->qvel, v1.data(), model->nv);
      mju_copy(data->qacc, a1.data(), model->nv);

      // inverse dynamics
      mj_inverse(model, data);

      // inverse dynamics error
      mju_sub(rt, data->qfrc_inverse, f1, model->nv);
    }

    // weighted cost
    return weight * 0.5 * mju_dot(residual.data(), residual.data(), dim_res);
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
  // finite-difference velocities
  ConfigurationToVelocity(estimator.velocity_.data(),
                          estimator.configuration_.data(),
                          estimator.configuration_length_, estimator.model_);

  // finite-difference accelerations
  VelocityToAcceleration(estimator.acceleration_.data(),
                         estimator.velocity_.data(),
                         estimator.configuration_length_ - 1, estimator.model_);

  // compute intermediate terms
  estimator.ComputeInverseDynamics();
  estimator.ResidualInverseDynamics();
  estimator.ModelDerivatives();
  estimator.VelocityJacobianBlocks();
  estimator.AccelerationJacobianBlocks();
  estimator.JacobianInverseDynamics();

  // cost
  double cost_estimator = estimator.CostInverseDynamics(
      estimator.cost_gradient_inverse_dynamics_.data(),
      estimator.cost_hessian_inverse_dynamics_.data());

  // ----- error ----- //

  // cost
  double cost_error = cost_estimator - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(dim_vel);
  mju_sub(gradient_error.data(),
          estimator.cost_gradient_inverse_dynamics_.data(),
          fdg.gradient_.data(), dim_vel);
  EXPECT_NEAR(mju_norm(gradient_error.data(), dim_vel) / dim_vel, 0.0, 1.0e-3);

  // Hessian
  std::vector<double> hessian_error(dim_vel * dim_vel);
  mju_sub(hessian_error.data(), estimator.cost_hessian_inverse_dynamics_.data(),
          fdh.hessian_.data(), dim_vel * dim_vel);
  EXPECT_NEAR(
      mju_norm(hessian_error.data(), dim_vel * dim_vel) / (dim_vel * dim_vel),
      0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(InverseDynamicsCost, Box) {
  // load model
  mjModel* model = LoadTestModel("box3D.xml");
  model->opt.timestep = 0.035;
  mjData* data = mj_makeData(model);

  // initial configuration
  double qpos0[7] = {0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0};

  // ----- configurations ----- //
  int history = 3;
  int dim_pos = model->nq * history;
  int dim_vel = model->nv * history;
  int dim_id = model->nv * history;
  int dim_res = model->nv * (history - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> qfrc_actuator(dim_id);

  // random initialization
  for (int t = 0; t < history; t++) {
    mju_copy(configuration.data() + t * model->nq, qpos0, model->nq);

    for (int i = 0; i < model->nq; i++) {
      absl::BitGen gen_;
      configuration[model->nq * t + i] +=
          1.0e-2 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    // normalize quaternion
    mju_normalize4(configuration.data() + model->nq * t + 3);

    for (int i = 0; i < model->nv; i++) {
      absl::BitGen gen_;
      qfrc_actuator[model->nv * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.configuration_length_ = history;
  estimator.weight_inverse_dynamics_ = 0.00125;

  // copy configuration, qfrc_actuator
  mju_copy(estimator.configuration_.data(), configuration.data(), dim_pos);
  mju_copy(estimator.qfrc_actuator_.data(), qfrc_actuator.data(), dim_id);

  // ----- cost ----- //
  auto cost_inverse_dynamics = [&configuration = estimator.configuration_,
                                &qfrc_actuator = estimator.qfrc_actuator_,
                                &configuration_length = history, &model,
                                &dim_res,
                                &weight = estimator.weight_inverse_dynamics_,
                                &data](const double* update) {
    // ----- integrate quaternion ----- //
    std::vector<double> qint(model->nq * configuration_length);
    mju_copy(qint.data(), configuration.data(),
             model->nq * configuration_length);

    // loop over configurations
    for (int t = 0; t < configuration_length; t++) {
      double* q = qint.data() + t * model->nq;
      const double* dq = update + t * model->nv;
      mj_integratePos(model, q, dq, 1.0);
    }

    // velocity
    std::vector<double> v1(model->nv);
    std::vector<double> v2(model->nv);

    // acceleration
    std::vector<double> a1(model->nv);

    // residual
    std::vector<double> residual(dim_res);

    // loop over time
    for (int t = 0; t < configuration_length - 2; t++) {
      // unpack
      double* rt = residual.data() + t * model->nv;
      double* q0 = qint.data() + t * model->nq;
      double* q1 = qint.data() + (t + 1) * model->nq;
      double* q2 = qint.data() + (t + 2) * model->nq;
      double* f1 = qfrc_actuator.data() + t * model->nv;

      // velocity
      mj_differentiatePos(model, v1.data(), model->opt.timestep, q0, q1);
      mj_differentiatePos(model, v2.data(), model->opt.timestep, q1, q2);

      // acceleration
      mju_sub(a1.data(), v2.data(), v1.data(), model->nv);
      mju_scl(a1.data(), a1.data(), 1.0 / model->opt.timestep, model->nv);

      // set state
      mju_copy(data->qpos, q1, model->nq);
      mju_copy(data->qvel, v1.data(), model->nv);
      mju_copy(data->qacc, a1.data(), model->nv);

      // inverse dynamics
      mj_inverse(model, data);

      // inverse dynamics error
      mju_sub(rt, data->qfrc_inverse, f1, model->nv);
    }

    // weighted cost
    return weight * 0.5 * mju_dot(residual.data(), residual.data(), dim_res);
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
  // finite-difference velocities
  ConfigurationToVelocity(estimator.velocity_.data(),
                          estimator.configuration_.data(),
                          estimator.configuration_length_, estimator.model_);

  // finite-difference accelerations
  VelocityToAcceleration(estimator.acceleration_.data(),
                         estimator.velocity_.data(),
                         estimator.configuration_length_ - 1, estimator.model_);

  // compute intermediate terms
  estimator.ComputeInverseDynamics();
  estimator.ResidualInverseDynamics();
  estimator.ModelDerivatives();
  estimator.VelocityJacobianBlocks();
  estimator.AccelerationJacobianBlocks();
  estimator.JacobianInverseDynamics();

  // cost
  double cost_estimator = estimator.CostInverseDynamics(
      estimator.cost_gradient_inverse_dynamics_.data(), NULL);

  // ----- error ----- //

  // cost
  double cost_error = cost_estimator - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(dim_vel);
  mju_sub(gradient_error.data(),
          estimator.cost_gradient_inverse_dynamics_.data(),
          fdg.gradient_.data(), dim_vel);
  EXPECT_NEAR(mju_norm(gradient_error.data(), dim_vel) / dim_vel, 0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
