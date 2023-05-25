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

TEST(Covariance, Update) {
  // load model
  mjModel* model = LoadTestModel("particle2D.xml");

  // dimensions
  int configuration_length = 5;
  int dim = model->nv * configuration_length;

  // ----- estimator ----- //
  Estimator estimator;
  estimator.solver_ = kCholeskyDenseSolver;
  estimator.Initialize(model);
  estimator.configuration_length_ = configuration_length;

  printf("dim: %i\n", estimator.model_->nv * estimator.configuration_length_);

  // ----- random matrix square roots ----- //
  std::vector<double> A(dim * dim);
  std::vector<double> B(dim * dim);
  std::vector<double> C(dim * dim);
  std::vector<double> D(dim * dim);

  // sample random matrix square roots
  absl::BitGen gen_;
  for (int i = 0; i < dim * dim; i++) {
    A[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    B[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // assemble PSD matrices
  mju_mulMatTMat(C.data(), A.data(), A.data(), dim, dim, dim);
  mju_mulMatTMat(D.data(), B.data(), B.data(), dim, dim, dim);

  // ----- covariance update: E <- E - E * H * E'
  std::vector<double> tmp0(dim * dim);
  std::vector<double> tmp1(dim * dim);
  std::vector<double> covariance_update(dim * dim);

  // tmp0 = H * E'
  mju_mulMatMatT(tmp0.data(), D.data(), C.data(), dim, dim, dim);

  // tmp1 = E * tmp0
  mju_mulMatMat(tmp1.data(), C.data(), tmp0.data(), dim, dim, dim);

  // E <- E - tmp1
  mju_sub(covariance_update.data(), C.data(), tmp1.data(), dim * dim);

  // ----- estimator covariance update ----- //

  // set values
  mju_copy(estimator.covariance_.data(), C.data(), dim * dim);
  mju_copy(estimator.cost_hessian_.data(), D.data(), dim * dim);

  // update
  estimator.CovarianceUpdate();

  // ----- error ----- //
  std::vector<double> error(dim * dim);

  mju_sub(error.data(), estimator.covariance_.data(), covariance_update.data(),
          dim * dim);

  // ------ test ----- //
  EXPECT_NEAR(mju_norm(error.data(), dim * dim) / (dim * dim), 0.0, 1.0e-3);

  // ----- band solver ----- //

  // set solver
  estimator.solver_ = kBandSolver;

  // set covariance
  mju_copy(estimator.covariance_.data(), C.data(), dim * dim);

  // set Hessian
  mju_dense2Band(estimator.cost_hessian_band_.data(), D.data(), dim,
                 3 * model->nv, 0);

  // update
  estimator.CovarianceUpdate();

  // ----- covariance update ----- //

  // convert Hessian to band Hessian
  std::vector<double> Dband(dim * dim);
  mju_band2Dense(Dband.data(), estimator.cost_hessian_band_.data(), dim,
                 3 * model->nv, 0, true);

  // tmp0 = H * E'
  mju_mulMatMatT(tmp0.data(), Dband.data(), C.data(), dim, dim, dim);

  // tmp1 = E * tmp0
  mju_mulMatMat(tmp1.data(), C.data(), tmp0.data(), dim, dim, dim);

  // E <- E - tmp1
  mju_sub(covariance_update.data(), C.data(), tmp1.data(), dim * dim);

  // ----- error ----- //
  mju_sub(error.data(), estimator.covariance_.data(), covariance_update.data(),
          dim * dim);

  // ------ test ----- //
  EXPECT_NEAR(mju_norm(error.data(), dim * dim) / (dim * dim), 0.0, 1.0e-3);
}

}  // namespace
}  // namespace mjpc
