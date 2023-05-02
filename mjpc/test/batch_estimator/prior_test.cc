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

TEST(PriorResidual, Box) {
  // load model
  mjModel* model = LoadTestModel("box3D.xml");
  mjData* data = mj_makeData(model);

  // ----- configurations ----- //
  int history = 5;
  int dim_pos = model->nq * history;
  int dim_vel = model->nv * history;
  std::vector<double> configuration(dim_pos);
  std::vector<double> prior(dim_pos);

  // random initialization 
  for (int t = 0; t < history; t++) {
    for (int i = 0; i < model->nq; i++) {
      absl::BitGen gen_;
      configuration[model->nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
      prior[model->nq * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    // normalize quaternion 
    mju_normalize4(configuration.data() + model->nq * t + 3);
    mju_normalize4(prior.data() + model->nq * t + 3);
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.configuration_length_ = history;

  // copy configuration, prior 
  mju_copy(estimator.configuration_.data(), configuration.data(), dim_pos);
  mju_copy(estimator.configuration_prior_.data(), prior.data(), dim_pos);

  // ----- residual ----- //
  auto residual_prior = [&configuration, &prior,
                         &configuration_length = history,
                         &model](double* residual, const double* update) {
    // integrated quaternion
    std::vector<double> qint(model->nq);

    // loop over time
    for (int t = 0; t < configuration_length; t++) {
      // terms
      double* rt = residual + t * model->nv;
      double* qt_prior = prior.data() + t * model->nq;
      double* qt = configuration.data() + t * model->nq;

      // ----- integrate ----- //
      mju_copy(qint.data(), qt, model->nq);
      const double* dq = update + t * model->nv;
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
  mju_sub(residual_error.data(), estimator.residual_prior_.data(), residual.data(), dim_vel);

  // test
  EXPECT_NEAR(mju_norm(residual_error.data(), dim_vel) / (dim_vel), 0.0, 1.0e-5);

  // ----- Jacobian ----- //

  // finite-difference
  FiniteDifferenceJacobian fd(dim_vel, dim_vel);
  fd.Compute(residual_prior, update.data(), dim_vel, dim_vel);

  // estimator
  estimator.JacobianPriorBlocks();
  estimator.JacobianPrior();

  // error 
  std::vector<double> jacobian_error(dim_vel * dim_vel);
  mju_sub(jacobian_error.data(), estimator.jacobian_prior_.data(), fd.jacobian_.data(), dim_vel * dim_vel);

  // test 
  EXPECT_NEAR(mju_norm(jacobian_error.data(), dim_vel * dim_vel) / (dim_vel * dim_vel), 0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
