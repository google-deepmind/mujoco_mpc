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

TEST(MeasurementResidual, Box) {
  // load model
  mjModel* model = LoadTestModel("box3D.xml");
  mjData* data = mj_makeData(model);

  // ----- configurations ----- //
  int history = 4;
  int dim_pos = model->nq * history;
  int dim_vel = model->nv * history;
  int dim_mea = model->nsensordata * history;
  int dim_res = model->nsensordata * (history - 2);

  std::vector<double> configuration(dim_pos);
  std::vector<double> prior(dim_pos);
  std::vector<double> measurement(dim_mea);

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

    for (int i = 0; i < model->nsensordata; i++) {
      absl::BitGen gen_;
      measurement[model->nsensordata * t + i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // ----- estimator ----- //
  Estimator estimator;
  estimator.Initialize(model);
  estimator.configuration_length_ = history;

  // copy configuration, prior, measurement
  mju_copy(estimator.configuration_.data(), configuration.data(), dim_pos);
  mju_copy(estimator.configuration_prior_.data(), prior.data(), dim_pos);
  mju_copy(estimator.measurement_sensor_.data(), measurement.data(), dim_mea);

  // ----- residual ----- //
  auto residual_measurement = [&configuration, &measurement,
                         &configuration_length = history,
                         &model, &data](double* residual, const double* update) {
    // ----- integrate quaternion ----- //
    std::vector<double> qint(model->nq * configuration_length);
    mju_copy(qint.data(), configuration.data(), model->nq * configuration_length);
    
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
      double* rt = residual + t * model->nsensordata;
      double* q0 = qint.data() + t * model->nq;
      double* q1 = qint.data() + (t + 1) * model->nq;
      double* q2 = qint.data() + (t + 2) * model->nq;
      double* y1 = measurement.data() + t * model->nsensordata;

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

      // measurement error
      mju_sub(rt, data->sensordata, y1, model->nsensordata);
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
  // finite-difference velocities
  ConfigurationToVelocity(estimator.velocity_.data(),
                          estimator.configuration_.data(),
                          estimator.configuration_length_, estimator.model_);

  // finite-difference accelerations
  VelocityToAcceleration(estimator.acceleration_.data(),
                         estimator.velocity_.data(),
                         estimator.configuration_length_ - 1, estimator.model_);

  // compute model measurements
  estimator.ComputeMeasurements();
  estimator.ResidualMeasurement();

  // // error 
  std::vector<double> residual_error(dim_mea);
  mju_sub(residual_error.data(), estimator.residual_measurement_.data(), residual.data(), dim_res);

  printf("residual (lambda): ");
  mju_printMat(residual.data(), 1, dim_res);

  printf("residual (estimator): ");
  mju_printMat(estimator.residual_measurement_.data(), 1, dim_res);

  // test
  EXPECT_NEAR(mju_norm(residual_error.data(), dim_res) / (dim_res), 0.0, 1.0e-5);

  // ----- Jacobian ----- //

  // finite-difference
  FiniteDifferenceJacobian fd(dim_res, dim_vel);
  fd.Compute(residual_measurement, update.data(), dim_res, dim_vel);

  // estimator
  estimator.ModelDerivatives();
  estimator.JacobianMeasurement();

  // error 
  std::vector<double> jacobian_error(dim_res * dim_vel);
  mju_sub(jacobian_error.data(), estimator.jacobian_prior_.data(), fd.jacobian_.data(), dim_res * dim_vel);

  printf("norm: %f\n", mju_norm(jacobian_error.data(), dim_res * dim_vel) / (dim_res * dim_vel));

  printf("measurement Jacobian (finite-difference):\n");
  mju_printMat(fd.jacobian_.data(), dim_res, dim_vel);

  printf("measurement Jacobian (estimator):\n");
  mju_printMat(estimator.jacobian_measurement_.data(), dim_res, dim_vel);

  // printf("measurement Jacobian blocks configuration (estimator):\n");
  // mju_printMat(estimator.jacobian_block_measurement_configuration_.data(), estimator.dim_measurement_, model->nv);


  // // test 
  // EXPECT_NEAR(mju_norm(jacobian_error.data(), dim_vel * dim_vel) / (dim_vel * dim_vel), 0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
