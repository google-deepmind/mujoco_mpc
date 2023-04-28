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

#include <mujoco/mujoco.h>

#include <vector>

#include "gtest/gtest.h"
#include "mjpc/test/load.h"

#include "mjpc/estimators/batch/estimator.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(Particle2DTest, BatchEstimator) {
  printf("Particle 2D Test\n");

  // load model
  mjModel* model = LoadTestModel("particle2.xml");
  mjData* data = mj_makeData(model);
  
  // info 
  printf("nq: %i\n", model->nq);
  printf("nv: %i\n", model->nv);
  printf("nu: %i\n", model->nu);
  printf("ny: %i\n", model->nsensordata);

  // ----- simulate ----- //
  
  // controller
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(time);
    ctrl[1] = mju_cos(time);
  };

  // trajectories 
  int history = 5;
  std::vector<double> qpos(model->nq * (history + 1));
  std::vector<double> qvel(model->nv * (history + 1));
  std::vector<double> qacc(model->nv * (history + 1));
  std::vector<double> ctrl(model->nu * history);
  std::vector<double> qfrc_actuator(model->nv * history);
  std::vector<double> sensordata(model->nsensordata * (history + 1));

  // reset 
  mj_resetData(model, data);

  // cache initial state
  mju_copy(qpos.data(), data->qpos, model->nq);
  mju_copy(qvel.data(), data->qvel, model->nv);
  mju_copy(qacc.data(), data->qacc, model->nv);

  // rollout
  for (int t = 0; t < history; t++) {
    // set control
    controller(data->ctrl, data->time);

    // step 
    mj_step(model, data);

    // cache 
    mju_copy(qpos.data() + (t + 1) * model->nq, data->qpos, model->nq);
    mju_copy(qvel.data() + (t + 1) * model->nv, data->qvel, model->nv);
    mju_copy(qacc.data() + (t + 1) * model->nv, data->qacc, model->nv);
    mju_copy(ctrl.data() + t * model->nu, data->ctrl, model->nu);
    mju_copy(qfrc_actuator.data() + t * model->nv, data->qfrc_actuator, model->nv);
    mju_copy(sensordata.data() + t * model->nsensordata, data->sensordata, model->nsensordata);
  }

  // forward to evaluate sensors
  mj_forward(model, data);
  mju_copy(sensordata.data() + history * model->nsensor, data->sensordata, model->nsensordata);

  printf("\n");
  printf("qpos: \n");
  mju_printMat(qpos.data(), history + 1, model->nq);

  printf("qvel: \n");
  mju_printMat(qvel.data(), history + 1, model->nv);

  printf("qacc: \n");
  mju_printMat(qacc.data(), history + 1, model->nv);

  printf("ctrl: \n");
  mju_printMat(ctrl.data(), history, model->nu);

  printf("qfrc_actuator: \n");
  mju_printMat(qfrc_actuator.data(), history, model->nv);

  printf("sensordata:\n");
  mju_printMat(sensordata.data(), history + 1, model->nsensordata);

  // finite-difference estimates
  std::vector<double> velocity(model->nv * history);
  std::vector<double> acceleration(model->nv * (history - 1));

  ConfigurationToVelocity(velocity.data(), qpos.data(), history + 1, model);
  VelocityToAcceleration(acceleration.data(), velocity.data(), history, model);

  printf("velocity:\n");
  mju_printMat(velocity.data(), history, model->nv);

  // velocity error 
  std::vector<double> velocity_error(model->nv * history);
  mju_sub(velocity_error.data(), velocity.data(), qvel.data() + model->nv, model->nv * history);
  // printf("velocity error: %f\n", mju_norm(velocity_error.data(), model->nv * history) / (model->nv * history));
  EXPECT_NEAR(mju_norm(velocity_error.data(), model->nv * history) / (model->nv * history), 0.0, 1.0e-3);

  printf("acceleration:\n");
  mju_printMat(acceleration.data(), history - 1, model->nv);

  // acceleration error 
  std::vector<double> acceleration_error(model->nv * history);
  mju_sub(acceleration_error.data(), acceleration.data(), qacc.data() + 2 * model->nv, model->nv * (history - 2));
  // printf("acceleration error: %f\n", mju_norm(acceleration_error.data(), model->nv * (history - 1)) / (model->nv * (history - 1)));
  EXPECT_NEAR(mju_norm(acceleration_error.data(), model->nv * (history - 1)) / (model->nv * (history - 1)), 0.0, 1.0e-3);

  // std::vector<double> a1(model->nv);
  // mju_sub(a1.data(), velocity.data() + model->nv, velocity.data(), model->nv);
  // mju_scl(a1.data(), a1.data(), 1.0 / model->opt.timestep, model->nv);

  // printf("a1: \n");
  // mju_printMat(a1.data(), 1, model->nv);

  // std::vector<double> v1(model->nv);
  // std::vector<double> v2(model->nv);

  // // v1
  // mj_differentiatePos(model, v1.data(), model->opt.timestep, qpos.data(), qpos.data() + model->nq);

  // // v2
  // mj_differentiatePos(model, v2.data(), model->opt.timestep, qpos.data() + model->nv, qpos.data() + 2 * model->nv);

  // // a1_ 
  // std::vector<double> a1_(model->nv);
  // mju_sub(a1_.data(), v2.data(), v1.data(), model->nv);
  // mju_scl(a1_.data(), a1_.data(), 1.0 / model->opt.timestep, model->nv);

  // printf("a1_: \n");
  // mju_printMat(a1_.data(), 1, model->nv);

  // qfrc_inverse 
  // std::vector<double> qfrc_inverse(model->nv * (history - 2));

  // for (int t = 0; t < history - 2; t++) {
  //   // set state
  //   mju_copy(data->qpos, qpos.data() + (t + 1) * model->nq, model->nq);
  //   mju_copy(data->qvel, velocity.data() + t * model->nv, model->nv);
  //   mju_copy(data->qacc, acceleration.data() + t * model->nv, model->nv);

  //   // inverse dynamics 
  //   mj_inverse(model, data);

  //   // copy qfrc_inverse 
  //   mju_copy(qfrc_inverse.data() + t * model->nv, data->qfrc_inverse, model->nv);
  // }

  // printf("qfrc_inverse:\n");
  // mju_printMat(qfrc_inverse.data(), history - 2, model->nv);

  // estimator 
  // Estimator estimator;
  // estimator.Initialize(model);         // initialize w/ model
  // estimator.configuration_length_ = 3; // set history

  // delete data + model
  // mj_deleteData(data);
  // mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
