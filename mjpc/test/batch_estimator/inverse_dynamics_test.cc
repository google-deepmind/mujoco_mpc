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
  int history = 3;
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
  auto residual_inverse_dynamics = [&qfrc_actuator,
                         &configuration_length = history,
                         &model, &data](double* residual, const double* update) {    
    
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
  mju_sub(residual_error.data(), estimator.residual_inverse_dynamics_.data(), residual.data(), dim_res);

  // test
  EXPECT_NEAR(mju_norm(residual_error.data(), dim_res) / (dim_res), 0.0, 1.0e-5);

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
  mju_sub(jacobian_error.data(), estimator.jacobian_inverse_dynamics_.data(), fd.jacobian_.data(), dim_res * dim_vel);

  // test 
  EXPECT_NEAR(mju_norm(jacobian_error.data(), dim_vel * dim_vel) / (dim_vel * dim_vel), 0.0, 1.0e-3);

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
      configuration[model->nq * t + i] += 1.0e-2 * absl::Gaussian<double>(gen_, 0.0, 1.0);
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
  auto residual_inverse_dynamics = [&configuration = estimator.configuration_, &qfrc_actuator = estimator.qfrc_actuator_,
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
  mju_sub(residual_error.data(), estimator.residual_inverse_dynamics_.data(), residual.data(), dim_res);

  // test
  EXPECT_NEAR(mju_norm(residual_error.data(), dim_res) / (dim_res), 0.0, 1.0e-5);

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
  mju_sub(jacobian_error.data(), estimator.jacobian_inverse_dynamics_.data(), fd.jacobian_.data(), dim_res * dim_vel);

  // test 
  EXPECT_NEAR(mju_norm(jacobian_error.data(), dim_res * dim_vel) / (dim_res * dim_vel), 0.0, 1.0e-3);

  printf("Jacobian (finite difference):\n");
  mju_printMat(fd.jacobian_.data(), dim_res, dim_vel);

  printf("Jacobian (estimator):\n");
  mju_printMat(estimator.jacobian_inverse_dynamics_.data(), dim_res, dim_vel);

  // printf("dqdf (estimator):\n");
  // mju_printMat(estimator.jacobian_block_inverse_dynamics_configuration_.data(), model->nv, model->nv);

  // // finite difference configuration
  // auto id_q = [&model, &data, &configuration = estimator.configuration_, &velocity = estimator.velocity_, &acceleration = estimator.acceleration_](double* r, const double* x) {
  //   // copy configuration
  //   std::vector<double> q_copy(model->nq);
  //   mju_copy(q_copy.data(), configuration.data() + model->nq, model->nq);
  //   double* q = q_copy.data();

  //   // integrate
  //   mj_integratePos(model, q, x, 1.0);

  //   // unpack
  //   double* v = velocity.data();
  //   double* a = acceleration.data();

  //   // set (state, acceleration)
  //   mju_copy(data->qpos, q, model->nq);
  //   mju_copy(data->qvel, v, model->nv);
  //   mju_copy(data->qacc, a, model->nv);

  //   // inverse dynamics
  //   mj_inverse(model, data);

  //   // copy qfrc
  //   mju_copy(r, data->qfrc_inverse, model->nv);
  // };


  // FiniteDifferenceJacobian fdq(model->nv, model->nv);
  // fdq.epsilon_ = 1.0e-6;
  // std::vector<double> dq(model->nv);
  // mju_zero(dq.data(), model->nv);

  // fdq.Compute(id_q, dq.data(), model->nv, model->nv);

  // printf("dqdf (fd):\n");
  // mju_printMat(fdq.jacobian_transpose_.data(), model->nv, model->nv);

  // double dqdf[36];

  // mju_copy(data->qpos, estimator.configuration_.data() + model->nq, model->nq);
  // mju_copy(data->qvel, estimator.velocity_.data(), model->nv);
  // mju_copy(data->qacc, estimator.acceleration_.data(), model->nv);

  // mjd_inverseFD(model, data, 1.0e-6, 0, dqdf, NULL, NULL, NULL, NULL, NULL, NULL);
  // printf("dqdf (mjd):\n");
  // mju_printMat(dqdf, model->nv, model->nv);

  // printf("dvdf:\n");
  // mju_printMat(estimator.jacobian_block_inverse_dynamics_velocity_.data(), model->nv, model->nv);

  // // finite difference velocity
  // auto id_v = [&qpos = estimator.configuration_, &qacc = estimator.acceleration_, &model, &data](double* out, const double* in) {
  //   // unpack 
  //   double* q = qpos.data() + model->nq;
  //   // double* v = qvel.data();
  //   double* a = qacc.data();

  //   // set (state, acceleration) 
  //   mju_copy(data->qpos, q, model->nq);
  //   mju_copy(data->qvel, in, model->nv);
  //   mju_copy(data->qacc, a, model->nv);

  //   // inverse dynamics 
  //   mj_inverse(model, data);

  //   // copy qfrc
  //   mju_copy(out, data->qfrc_inverse, model->nv);
  // };


  // FiniteDifferenceJacobian fdv(model->nv, model->nv);
  // std::vector<double> dv(model->nv);
  // mju_copy(dv.data(), estimator.velocity_.data(), model->nv);

  // fdv.Compute(id_v, dv.data(), model->nv, model->nv);

  // printf("dvdf (fd):\n");
  // mju_printMat(fdv.jacobian_transpose_.data(), model->nv, model->nv);

  // double dvdf[36];
  // mju_copy(data->qpos, estimator.configuration_.data() + model->nq, model->nq);
  // mju_copy(data->qvel, estimator.velocity_.data(), model->nv);
  // mju_copy(data->qacc, estimator.acceleration_.data(), model->nv);

  // mjd_inverseFD(model, data, 1.0e-6, 1, NULL, dvdf, NULL, NULL, NULL, NULL, NULL);
  // printf("dvdf (mjd):\n");
  // mju_printMat(dvdf, model->nv, model->nv);

  // printf("dadf:\n");
  // mju_printMat(estimator.jacobian_block_inverse_dynamics_acceleration_.data(), model->nv, model->nv);

  // auto id_a = [&qpos = estimator.configuration_, &qvel = estimator.velocity_, &model, &data](double* out, const double* in) {
  //   // unpack 
  //   double* q = qpos.data() + model->nq;
  //   double* v = qvel.data();
  //   // double* a = qacc.data();

  //   // set (state, acceleration) 
  //   mju_copy(data->qpos, q, model->nq);
  //   mju_copy(data->qvel, v, model->nv);
  //   mju_copy(data->qacc, in, model->nv);

  //   // inverse dynamics 
  //   mj_inverse(model, data);

  //   // copy qfrc
  //   mju_copy(out, data->qfrc_inverse, model->nv);
  // };


  // FiniteDifferenceJacobian fda(model->nv, model->nv);
  // std::vector<double> da(model->nv);
  // mju_copy(da.data(), estimator.acceleration_.data(), model->nv);

  // fda.Compute(id_a, da.data(), model->nv, model->nv);

  // printf("dadf (fd):\n");
  // mju_printMat(fda.jacobian_transpose_.data(), model->nv, model->nv);

  double dadf[36];
  mju_copy(data->qpos, estimator.configuration_.data() + model->nq, model->nq);
  mju_copy(data->qvel, estimator.velocity_.data(), model->nv);
  mju_copy(data->qacc, estimator.acceleration_.data(), model->nv);

  mjd_inverseFD(model, data, 1.0e-6, 1, NULL, NULL, dadf, NULL, NULL, NULL, NULL);
  printf("dadf (mjd):\n");
  mju_printMat(dadf, model->nv, model->nv);

  // 
  printf("dfdq0 (fd):\n");
  auto id_q0 = [&qpos = estimator.configuration_, &model, &data](double* out, const double* in) {
    // configuration copy 
    int t = 0;
    std::vector<double> q_copy(model->nq);
    mju_copy(q_copy.data(), qpos.data() + t * model->nq, model->nq);
    mj_integratePos(model, q_copy.data(), in, 1.0);

    // unpack 
    double* q0 = q_copy.data();
    double* q1 = qpos.data() + model->nq;
    double* q2 = qpos.data() + 2 * model->nq;

    // q
    double* q = q1;

    // v 
    std::vector<double> v1(model->nv);
    std::vector<double> v2(model->nv);
    mj_differentiatePos(model, v1.data(), model->opt.timestep, q0, q1);
    mj_differentiatePos(model, v2.data(), model->opt.timestep, q1, q2);
    double* v = v1.data();

    // a 
    std::vector<double> a1(model->nv);
    mju_sub(a1.data(), v2.data(), v1.data(), model->nv);
    mju_scl(a1.data(), a1.data(), 1.0 / model->opt.timestep, model->nv);
    double* a = a1.data();

    // set (state, acceleration) 
    mju_copy(data->qpos, q, model->nq);
    mju_copy(data->qvel, v, model->nv);
    mju_copy(data->qacc, a, model->nv);

    // inverse dynamics 
    mj_inverse(model, data);

    // copy qfrc
    mju_copy(out, data->qfrc_inverse, model->nv);
  };


  FiniteDifferenceJacobian fdq0(model->nv, model->nv);
  std::vector<double> dq0(model->nv);
  mju_zero(dq0.data(), model->nv);
  fdq0.Compute(id_q0, dq0.data(), model->nv, model->nv);
  mju_printMat(fdq0.jacobian_.data(), model->nv, model->nv);

  // 
  printf("dfdq1 (fd):\n");
  auto id_q1 = [&qpos = estimator.configuration_, &model, &data](double* out, const double* in) {
    // configuration copy 
    int t = 1;
    std::vector<double> q_copy(model->nq);
    mju_copy(q_copy.data(), qpos.data() + t * model->nq, model->nq);
    mj_integratePos(model, q_copy.data(), in, 1.0);

    // unpack 
    double* q0 = qpos.data();
    double* q1 = q_copy.data();
    double* q2 = qpos.data() + 2 * model->nq;

    // q
    double* q = q1;

    // v 
    std::vector<double> v1(model->nv);
    std::vector<double> v2(model->nv);
    mj_differentiatePos(model, v1.data(), model->opt.timestep, q0, q1);
    mj_differentiatePos(model, v2.data(), model->opt.timestep, q1, q2);
    double* v = v1.data();

    // a 
    std::vector<double> a1(model->nv);
    mju_sub(a1.data(), v2.data(), v1.data(), model->nv);
    mju_scl(a1.data(), a1.data(), 1.0 / model->opt.timestep, model->nv);
    double* a = a1.data();

    // set (state, acceleration) 
    mju_copy(data->qpos, q, model->nq);
    mju_copy(data->qvel, v, model->nv);
    mju_copy(data->qacc, a, model->nv);

    // inverse dynamics 
    mj_inverse(model, data);

    // copy qfrc
    mju_copy(out, data->qfrc_inverse, model->nv);
  };


  FiniteDifferenceJacobian fdq1(model->nv, model->nv);
  std::vector<double> dq1(model->nv);
  mju_zero(dq1.data(), model->nv);
  fdq1.Compute(id_q1, dq1.data(), model->nv, model->nv);
  mju_printMat(fdq1.jacobian_.data(), model->nv, model->nv);

  // 
  printf("dfdq2 (fd):\n");
  auto id_q2 = [&qpos = estimator.configuration_, &model, &data](double* out, const double* in) {
    // configuration copy 
    int t = 2;
    std::vector<double> q_copy(model->nq);
    mju_copy(q_copy.data(), qpos.data() + t * model->nq, model->nq);
    mj_integratePos(model, q_copy.data(), in, 1.0);

    // unpack 
    double* q0 = qpos.data();
    double* q1 = qpos.data() + 1 * model->nq;
    double* q2 = q_copy.data();

    // q
    double* q = q1;

    // v 
    std::vector<double> v1(model->nv);
    std::vector<double> v2(model->nv);
    mj_differentiatePos(model, v1.data(), model->opt.timestep, q0, q1);
    mj_differentiatePos(model, v2.data(), model->opt.timestep, q1, q2);
    double* v = v1.data();

    // a 
    std::vector<double> a1(model->nv);
    mju_sub(a1.data(), v2.data(), v1.data(), model->nv);
    mju_scl(a1.data(), a1.data(), 1.0 / model->opt.timestep, model->nv);
    double* a = a1.data();

    // set (state, acceleration) 
    mju_copy(data->qpos, q, model->nq);
    mju_copy(data->qvel, v, model->nv);
    mju_copy(data->qacc, a, model->nv);

    // inverse dynamics 
    mj_inverse(model, data);

    // copy qfrc
    mju_copy(out, data->qfrc_inverse, model->nv);
  };


  FiniteDifferenceJacobian fdq2(model->nv, model->nv);
  std::vector<double> dq2(model->nv);
  mju_zero(dq2.data(), model->nv);
  fdq2.Compute(id_q2, dq2.data(), model->nv, model->nv);
  mju_printMat(fdq2.jacobian_.data(), model->nv, model->nv);


  // dadq2 
  double dadq2[9];
  DifferentiateDifferentiatePos(NULL, dadq2, model, model->opt.timestep, configuration.data() + model->nq, configuration.data() + 2 * model->nq);
  mju_scl(dadq2, dadq2, 1.0 / model->opt.timestep, 9);

  printf("dadq2:\n");
  mju_printMat(dadq2, 3, 3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
