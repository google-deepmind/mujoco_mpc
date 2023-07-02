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

TEST(PriorCost, Particle) {
  // load model
  // note: needs to be a linear system to satisfy Gauss-Newton Hessian
  // approximation
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

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

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);
  estimator.scale_prior_ = 5.0;
  estimator.band_prior_ = false;

  // copy configuration, prior
  mju_copy(estimator.configuration_.Data(), sim.qpos.Data(), nq * T);
  mju_copy(estimator.configuration_prior_.Data(), sim.qpos.Data(), nq * T);

  // corrupt configurations
  absl::BitGen gen_;
  for (int t = 0; t < T; t++) {
    double* q = estimator.configuration_.Get(t);
    for (int i = 0; i < nq; i++) {
      q[i] += absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- random prior ----- //
  int nvar = nv * T;
  std::vector<double> P(nvar * nvar);
  std::vector<double> F(nvar * nvar);

  // P = F' F
  for (int i = 0; i < nvar * nvar; i++) {
    F[i] = 0.1 * absl::Gaussian<double>(gen_, 0.0, 1.0);
  }
  mju_mulMatTMat(P.data(), F.data(), F.data(), nvar, nvar, nvar);

  // set prior
  mju_copy(estimator.weight_prior_dense_.data(), P.data(), nvar * nvar);

  // ----- cost ----- //
  auto cost_prior = [&estimator = estimator](const double* configuration) {
    // dimension
    int nq = estimator.model_->nq;
    int nv = estimator.model_->nv;
    int T = estimator.configuration_length_;
    int nvar = estimator.model_->nv * estimator.configuration_length_;

    // residual
    std::vector<double> residual(nvar);

    // loop over configurations
    for (int t = 0; t < T; t++) {
      // terms
      double* rt = residual.data() + t * nv;
      double* qt_prior = estimator.configuration_prior_.Get(t);
      const double* qt = configuration + t * nq;

      // configuration difference
      mju_sub(rt, qt, qt_prior, nv);
    }

    // ----- 0.5 * w * r' * P * r ----- //

    // scratch
    std::vector<double> scratch(nvar);
    mju_mulMatVec(scratch.data(), estimator.weight_prior_dense_.data(),
                  residual.data(), nvar, nvar);

    // weighted cost
    return 0.5 * estimator.scale_prior_ / nvar *
           mju_dot(residual.data(), scratch.data(), nvar);
  };

  // ----- lambda ----- //

  // cost
  double cost_lambda = cost_prior(estimator.configuration_.Data());

  // gradient
  FiniteDifferenceGradient fdg(nvar);
  fdg.Compute(cost_prior, estimator.configuration_.Data(), nvar);

  // Hessian
  FiniteDifferenceHessian fdh(nvar);
  fdh.Compute(cost_prior, estimator.configuration_.Data(), nvar);

  // ----- estimator ----- //
  estimator.ResidualPrior();
  for (int t = 0; t < estimator.configuration_length_; t++) {
    estimator.BlockPrior(t);
    estimator.SetBlockPrior(t);
  }

  // evaluate cost
  double cost_estimator =
      estimator.CostPrior(estimator.cost_gradient_prior_.data(),
                          estimator.cost_hessian_prior_.data());

  // ----- error ----- //

  // cost
  EXPECT_NEAR(cost_estimator - cost_lambda, 0.0, 1.0e-4);

  // gradient
  std::vector<double> gradient_error(nvar);
  mju_sub(gradient_error.data(), estimator.cost_gradient_prior_.data(),
          fdg.gradient.data(), nvar);
  EXPECT_NEAR(mju_norm(gradient_error.data(), nvar) / nvar, 0.0, 1.0e-4);

  // Hessian
  std::vector<double> hessian_error(nvar * nvar);
  mju_sub(hessian_error.data(), estimator.cost_hessian_prior_.data(),
          fdh.hessian.data(), nvar * nvar);
  EXPECT_NEAR(mju_norm(hessian_error.data(), nvar * nvar) / (nvar * nvar), 0.0,
              1.0e-3);

  // ----- band prior ----- //

  // remove off band elements
  DenseToBlockBand(P.data(), nvar, nv, 3);

  // set prior
  mju_copy(estimator.weight_prior_dense_.data(), P.data(), nvar * nvar);

  // change settings
  estimator.band_prior_ = true;

  // ----- cost ----- //
  auto cost_band_prior = [&estimator = estimator](const double* configuration) {
    // dimension
    int nq = estimator.model_->nq;
    int nv = estimator.model_->nv;
    int T = estimator.configuration_length_;
    int nvar = estimator.model_->nv * estimator.configuration_length_;

    // residual
    std::vector<double> residual(nvar);

    // loop over configurations
    for (int t = 0; t < T; t++) {
      // terms
      double* rt = residual.data() + t * nv;
      double* qt_prior = estimator.configuration_prior_.Get(t);
      const double* qt = configuration + t * nq;

      // configuration difference
      mju_sub(rt, qt, qt_prior, nv);
    }

    // ----- 0.5 * w * r' * P * r ----- //

    // scratch
    std::vector<double> scratch(nvar);
    mju_bandMulMatVec(scratch.data(), estimator.weight_prior_band_.data(),
                      residual.data(), nvar, 3 * nv, 0, 1, true);

    // weighted cost
    return 0.5 * estimator.scale_prior_ / nvar *
           mju_dot(residual.data(), scratch.data(), nvar);
  };

  // ----- lambda ----- //

  // cost
  double cost_band_lambda = cost_band_prior(estimator.configuration_.Data());

  // gradient
  fdg.Compute(cost_band_prior, estimator.configuration_.Data(), nvar);

  // Hessian
  fdh.Compute(cost_band_prior, estimator.configuration_.Data(), nvar);

  // ----- estimator ----- //

  // evaluate cost
  int cost_band_estimator =
      estimator.CostPrior(estimator.cost_gradient_prior_.data(),
                          estimator.cost_hessian_prior_.data());

  // ----- error ----- //

  // cost
  EXPECT_NEAR(cost_band_estimator - cost_band_lambda, 0.0, 1.0e-4);

  // gradient
  mju_sub(gradient_error.data(), estimator.cost_gradient_prior_.data(),
          fdg.gradient.data(), nvar);
  EXPECT_NEAR(mju_norm(gradient_error.data(), nvar) / nvar, 0.0, 1.0e-4);

  // Hessian
  mju_sub(hessian_error.data(), estimator.cost_hessian_prior_.data(),
          fdh.hessian.data(), nvar * nvar);
  EXPECT_NEAR(mju_norm(hessian_error.data(), nvar * nvar) / (nvar * nvar), 0.0,
              1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(PriorCost, Box) {
  // load model
  mjModel* model = LoadTestModel("estimator/box/task0.xml");
  mjData* data = mj_makeData(model);

  // dimension
  int nq = model->nq, nv = model->nv;

  // ----- rollout ----- //
  int T = 10;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {};
  double qvel[6] = {0.1, 0.2, -0.3, 0.25, -0.35, 0.1};
  sim.SetState(data->qpos, qvel);
  sim.Rollout(controller);

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(T);
  estimator.scale_prior_ = 5.0;
  estimator.band_prior_ = false;

  // copy configuration, prior
  mju_copy(estimator.configuration_.Data(), sim.qpos.Data(), nq * T);
  mju_copy(estimator.configuration_prior_.Data(), sim.qpos.Data(), nq * T);

  // corrupt configurations
  absl::BitGen gen_;
  double dq[6];
  for (int t = 0; t < T; t++) {
    double* q = estimator.configuration_.Get(t);
    for (int i = 0; i < nq; i++) {
      dq[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    mj_integratePos(model, q, dq, model->opt.timestep);
  }

  // ----- random prior ----- //
  int nvar = nv * T;
  std::vector<double> P(nvar * nvar);
  std::vector<double> F(nvar * nvar);

  // P = F' F
  for (int i = 0; i < nvar * nvar; i++) {
    F[i] = 0.1 * absl::Gaussian<double>(gen_, 0.0, 1.0);
  }
  mju_mulMatTMat(P.data(), F.data(), F.data(), nvar, nvar, nvar);
  std::vector<double> P_band(nvar * nvar);
  mju_dense2Band(P_band.data(), P.data(), nvar, 3 * nv, 0);

  // set prior
  mju_copy(estimator.weight_prior_band_.data(), P_band.data(), nvar * nvar);

  // ----- cost ----- //
  auto cost_prior = [&estimator = estimator](const double* update) {
    // dimension
    int nq = estimator.model_->nq;
    int nv = estimator.model_->nv;
    int T = estimator.configuration_length_;
    int nvar = estimator.model_->nv * estimator.configuration_length_;

    // configuration + update
    std::vector<double> configuration(nq * T);
    mju_copy(configuration.data(), estimator.configuration_.Data(), nq * T);
    for (int t = 0; t < T; t++) {
      double* q = configuration.data() + nq * t;
      const double* dq = update + nv * t;
      mj_integratePos(estimator.model_, q, dq, 1.0);
    }


    // residual
    std::vector<double> residual(nvar);

    // loop over configurations
    for (int t = 0; t < T; t++) {
      // terms
      double* rt = residual.data() + t * nv;
      double* qt_prior = estimator.configuration_prior_.Get(t);
      const double* qt = configuration.data() + t * nq;

      // configuration difference
      mju_sub(rt, qt, qt_prior, nv);
    }

    // ----- 0.5 * w * r' * P * r ----- //

    // scratch
    std::vector<double> scratch(nvar);
    mju_mulMatVec(scratch.data(), estimator.weight_prior_dense_.data(),
                  residual.data(), nvar, nvar);

    // weighted cost
    return 0.5 * estimator.scale_prior_ / nvar *
           mju_dot(residual.data(), scratch.data(), nvar);
  };

  // ----- lambda ----- //
  
  // update 
  std::vector<double> update(nv * T);
  mju_zero(update.data(), nv * T);

  // cost
  double cost_lambda = cost_prior(update.data());

  // gradient
  FiniteDifferenceGradient fdg(nvar);
  fdg.Compute(cost_prior, update.data(), nvar);

  // Hessian
  FiniteDifferenceHessian fdh(nvar);
  fdh.Compute(cost_prior, update.data(), nvar);

  // ----- estimator ----- //
  estimator.ResidualPrior();
  for (int t = 0; t < estimator.configuration_length_; t++) {
    estimator.BlockPrior(t);
    estimator.SetBlockPrior(t);
  }

  // evaluate cost
  double cost_estimator =
      estimator.CostPrior(estimator.cost_gradient_prior_.data(),
                          estimator.cost_hessian_prior_.data());

  // ----- error ----- //

  // cost
  EXPECT_NEAR(cost_estimator - cost_lambda, 0.0, 1.0e-4);

  // gradient
  std::vector<double> gradient_error(nvar);
  mju_sub(gradient_error.data(), estimator.cost_gradient_prior_.data(),
          fdg.gradient.data(), nvar);
  EXPECT_NEAR(mju_norm(gradient_error.data(), nvar) / nvar, 0.0, 1.0e-4);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
