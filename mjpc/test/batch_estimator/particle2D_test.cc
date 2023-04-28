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
    mju_copy(qvel.data() + (t + 1) * model->nv, data->qvel, model->nv);
    mju_copy(ctrl.data() + t * model->nu, data->ctrl, model->nu);
    mju_copy(qfrc_actuator.data() + t * model->nv, data->qfrc_actuator, model->nv);
    mju_copy(sensordata.data() + t * model->nsensordata, data->sensordata, model->nsensordata);
  }

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
  std::vector<double> velocity(model->nv * (history - 1));
  // std::vector<double> acceleration(model->nv * (history - 2));

  ConfigurationToVelocity(velocity.data(), qpos.data(), history, model);
  // VelocityToAcceleration(acceleration.data(), velocity.data(), history - 1, model);

  // printf("velocity:\n");
  // mju_printMat(velocity.data(), history - 1, model->nv);

  // printf("acceleration:\n");
  // mju_printMat(acceleration.data(), history - 2, model->nv);

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
