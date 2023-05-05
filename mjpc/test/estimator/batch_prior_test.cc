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
#include "mjpc/estimators/batch.h"
#include "mjpc/test/load.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(PriorResidual, Particle) {
  // load model
  mjModel* model = LoadTestModel("particle2D.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv;

  // ----- configurations ----- //
  int T = 5;
  int dim_pos = nq * T;
  int dim_vel = nv * T;
  std::vector<double> configuration(dim_pos);
  std::vector<double> prior(dim_pos);

  // random initialization
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < nq; i++) {
      absl::BitGen gen_;
      configuration[nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
      prior[nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.configuration_length_ = T;

  // copy configuration, prior
  mju_copy(estimator.configuration_.data(), configuration.data(), dim_pos);
  mju_copy(estimator.configuration_prior_.data(), prior.data(), dim_pos);

  // ----- residual ----- //
  auto residual_prior = [&prior, &configuration_length = T, &model, nq, nv](
                            double* residual, const double* update) {
    // integrated quaternion
    std::vector<double> qint(nq);

    // loop over time
    for (int t = 0; t < configuration_length; t++) {
      // terms
      double* rt = residual + t * nv;
      double* qt_prior = prior.data() + t * nq;

      // configuration difference
      mj_differentiatePos(model, rt, 1.0, qt_prior, update + t * nq);
    }
  };

  // initialize memory
  std::vector<double> residual(dim_vel);
  std::vector<double> update(dim_vel);
  mju_copy(update.data(), configuration.data(), dim_pos);

  // evaluate
  residual_prior(residual.data(), update.data());
  estimator.ResidualPrior();

  // error
  std::vector<double> residual_error(dim_vel);
  mju_sub(residual_error.data(), estimator.residual_prior_.data(),
          residual.data(), dim_vel);

  // test
  EXPECT_NEAR(mju_norm(residual_error.data(), dim_vel) / (dim_vel), 0.0,
              1.0e-5);

  // ----- Jacobian ----- //

  // finite-difference
  FiniteDifferenceJacobian fd(dim_vel, dim_vel);
  fd.Compute(residual_prior, update.data(), dim_vel, dim_vel);

  // estimator
  estimator.JacobianPriorBlocks();
  estimator.JacobianPrior();

  // error
  std::vector<double> jacobian_error(dim_vel * dim_vel);
  mju_sub(jacobian_error.data(), estimator.jacobian_prior_.data(),
          fd.jacobian_.data(), dim_vel * dim_vel);

  // test
  EXPECT_NEAR(
      mju_norm(jacobian_error.data(), dim_vel * dim_vel) / (dim_vel * dim_vel),
      0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(PriorResidual, Box) {
  // load model
  mjModel* model = LoadTestModel("box3D.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv;

  // ----- configurations ----- //
  int T = 5;
  int dim_pos = nq * T;
  int dim_vel = nv * T;
  std::vector<double> configuration(dim_pos);
  std::vector<double> prior(dim_pos);

  // random initialization
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < nq; i++) {
      absl::BitGen gen_;
      configuration[nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
      prior[nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    // normalize quaternion
    mju_normalize4(configuration.data() + nq * t + 3);
    mju_normalize4(prior.data() + nq * t + 3);
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.configuration_length_ = T;

  // copy configuration, prior
  mju_copy(estimator.configuration_.data(), configuration.data(), dim_pos);
  mju_copy(estimator.configuration_prior_.data(), prior.data(), dim_pos);

  // ----- residual ----- //
  auto residual_prior = [&configuration, &prior, &configuration_length = T,
                         &model, nq,
                         nv](double* residual, const double* update) {
    // integrated quaternion
    std::vector<double> qint(nq);

    // loop over time
    for (int t = 0; t < configuration_length; t++) {
      // terms
      double* rt = residual + t * nv;
      double* qt_prior = prior.data() + t * nq;
      double* qt = configuration.data() + t * nq;

      // ----- integrate ----- //
      mju_copy(qint.data(), qt, nq);
      const double* dq = update + t * nv;
      mj_integratePos(model, qint.data(), dq, 1.0);

      // configuration difference
      mj_differentiatePos(model, rt, 1.0, qt_prior, qint.data());
    }
  };

  // initialize memory
  std::vector<double> residual(dim_vel);
  std::vector<double> update(dim_vel);
  mju_zero(update.data(), dim_vel);

  // evaluate
  residual_prior(residual.data(), update.data());
  estimator.ResidualPrior();

  // error
  std::vector<double> residual_error(dim_vel);
  mju_sub(residual_error.data(), estimator.residual_prior_.data(),
          residual.data(), dim_vel);

  // test
  EXPECT_NEAR(mju_norm(residual_error.data(), dim_vel) / (dim_vel), 0.0,
              1.0e-5);

  // ----- Jacobian ----- //

  // finite-difference
  FiniteDifferenceJacobian fd(dim_vel, dim_vel);
  fd.Compute(residual_prior, update.data(), dim_vel, dim_vel);

  // estimator
  estimator.JacobianPriorBlocks();
  estimator.JacobianPrior();

  // error
  std::vector<double> jacobian_error(dim_vel * dim_vel);
  mju_sub(jacobian_error.data(), estimator.jacobian_prior_.data(),
          fd.jacobian_.data(), dim_vel * dim_vel);

  // test
  EXPECT_NEAR(
      mju_norm(jacobian_error.data(), dim_vel * dim_vel) / (dim_vel * dim_vel),
      0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(PriorCost, Particle) {
  // load model
  // note: needs to be a linear system to satisfy Gauss-Newton Hessian
  // approximation
  mjModel* model = LoadTestModel("particle2D.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv;

  // ----- configurations ----- //
  int T = 5;
  int dim_pos = nq * T;
  int dim_vel = nv * T;
  std::vector<double> configuration(dim_pos);
  std::vector<double> prior(dim_pos);

  // random initialization
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < nq; i++) {
      absl::BitGen gen_;
      configuration[nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
      prior[nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.configuration_length_ = T;
  estimator.weight_prior_ = 7.3;

  // copy configuration, prior
  mju_copy(estimator.configuration_.data(), configuration.data(), dim_pos);
  mju_copy(estimator.configuration_prior_.data(), prior.data(), dim_pos);

  // ----- cost ----- //
  auto cost_prior = [&prior, &configuration_length = T,
                     &weight = estimator.weight_prior_, nq,
                     nv](const double* configuration) {
    // dimension
    int dim_res = nv * configuration_length;

    // residual
    std::vector<double> residual(dim_res);

    // loop over time
    for (int t = 0; t < configuration_length; t++) {
      // terms
      double* rt = residual.data() + t * nv;
      double* qt_prior = prior.data() + t * nq;
      const double* qt = configuration + t * nq;

      // configuration difference
      mju_sub(rt, qt, qt_prior, nv);
    }

    return 0.5 * weight * mju_dot(residual.data(), residual.data(), dim_res);
  };

  // ----- lambda ----- //

  // cost
  double cost_lambda = cost_prior(configuration.data());

  // gradient
  FiniteDifferenceGradient fdg(dim_vel);
  fdg.Compute(cost_prior, configuration.data(), dim_vel);

  // Hessian
  FiniteDifferenceHessian fdh(dim_vel);
  fdh.Compute(cost_prior, configuration.data(), dim_vel);

  // ----- estimator ----- //
  estimator.ResidualPrior();
  estimator.JacobianPriorBlocks();
  estimator.JacobianPrior();
  estimator.CostPrior(&estimator.cost_prior_,
                      estimator.cost_gradient_prior_.data(),
                      estimator.cost_hessian_prior_.data());

  // ----- error ----- //

  // cost
  double cost_error = estimator.cost_prior_ - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(dim_vel);
  mju_sub(gradient_error.data(), estimator.cost_gradient_prior_.data(),
          fdg.gradient_.data(), dim_vel);
  EXPECT_NEAR(mju_norm(gradient_error.data(), dim_vel) / dim_vel, 0.0, 1.0e-3);

  // Hessian
  std::vector<double> hessian_error(dim_vel * dim_vel);
  mju_sub(hessian_error.data(), estimator.cost_hessian_prior_.data(),
          fdh.hessian_.data(), dim_vel * dim_vel);
  EXPECT_NEAR(
      mju_norm(hessian_error.data(), dim_vel * dim_vel) / (dim_vel * dim_vel),
      0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(PriorCost, Box) {
  // load model
  mjModel* model = LoadTestModel("box3D.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv;

  // ----- configurations ----- //
  int T = 5;
  int dim_pos = nq * T;
  int dim_vel = nv * T;
  std::vector<double> configuration(dim_pos);
  std::vector<double> prior(dim_pos);

  // random initialization
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < nq; i++) {
      absl::BitGen gen_;
      configuration[nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
      prior[nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    // normalize quaternion
    mju_normalize4(configuration.data() + nq * t + 3);
    mju_normalize4(prior.data() + nq * t + 3);
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.configuration_length_ = T;
  estimator.weight_prior_ = 7.3;

  // copy configuration, prior
  mju_copy(estimator.configuration_.data(), configuration.data(), dim_pos);
  mju_copy(estimator.configuration_prior_.data(), prior.data(), dim_pos);

  // ----- cost ----- //
  auto cost_prior = [&configuration, &prior, &configuration_length = T, &model,
                     &weight = estimator.weight_prior_, nq,
                     nv](const double* update) {
    // residual
    int dim_res = nv * configuration_length;
    std::vector<double> residual(dim_res);

    // integrated quaternion
    std::vector<double> qint(nq);

    // loop over time
    for (int t = 0; t < configuration_length; t++) {
      // terms
      double* rt = residual.data() + t * nv;
      double* qt_prior = prior.data() + t * nq;
      double* qt = configuration.data() + t * nq;

      // ----- integrate ----- //
      mju_copy(qint.data(), qt, nq);
      const double* dq = update + t * nv;
      mj_integratePos(model, qint.data(), dq, 1.0);

      // configuration difference
      mj_differentiatePos(model, rt, 1.0, qt_prior, qint.data());
    }

    return 0.5 * weight * mju_dot(residual.data(), residual.data(), dim_res);
  };

  // ----- lambda ----- //

  // cost
  std::vector<double> update(dim_vel);
  mju_zero(update.data(), dim_vel);
  double cost_lambda = cost_prior(update.data());

  // gradient
  FiniteDifferenceGradient fdg(dim_vel);
  fdg.Compute(cost_prior, update.data(), dim_vel);

  // ----- estimator ----- //
  estimator.ResidualPrior();
  estimator.JacobianPriorBlocks();
  estimator.JacobianPrior();
  estimator.CostPrior(&estimator.cost_prior_, estimator.cost_gradient_prior_.data(), NULL);

  // ----- error ----- //

  // cost
  double cost_error = estimator.cost_prior_ - cost_lambda;
  EXPECT_NEAR(cost_error, 0.0, 1.0e-5);

  // gradient
  std::vector<double> gradient_error(dim_vel);
  mju_sub(gradient_error.data(), estimator.cost_gradient_prior_.data(),
          fdg.gradient_.data(), dim_vel);
  EXPECT_NEAR(mju_norm(gradient_error.data(), dim_vel) / dim_vel, 0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
