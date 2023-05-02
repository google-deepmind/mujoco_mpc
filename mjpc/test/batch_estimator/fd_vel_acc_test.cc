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

TEST(FiniteDifferenceVelocityAcceleration, Particle2D) {
  // printf("Particle 2D Test\n");

  // load model
  mjModel* model = LoadTestModel("particle2D.xml");
  mjData* data = mj_makeData(model);
  
  // info 
  // printf("nq: %i\n", model->nq);
  // printf("nv: %i\n", model->nv);
  // printf("nu: %i\n", model->nu);
  // printf("ny: %i\n", model->nsensordata);

  // ----- simulate ----- //
  
  // controller
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(10 * time);
    ctrl[1] = mju_cos(10 * time);
  };

  // trajectories 
  int history = 5;
  std::vector<double> qpos(model->nq * (history + 1));
  std::vector<double> qvel(model->nv * (history + 1));
  std::vector<double> qacc(model->nv * history);
  std::vector<double> ctrl(model->nu * history);
  std::vector<double> qfrc_actuator(model->nv * history);
  std::vector<double> sensordata(model->nsensordata * (history + 1));

  // reset 
  mj_resetData(model, data);

  // rollout
  for (int t = 0; t < history; t++) {
    // set control
    controller(data->ctrl, data->time);

    // forward computes instantaneous qacc
    mj_forward(model, data);

    // cache
    mju_copy(qpos.data() + t * model->nq, data->qpos, model->nq);
    mju_copy(qvel.data() + t * model->nv, data->qvel, model->nv);
    mju_copy(qacc.data() + t * model->nv, data->qacc, model->nv);
    mju_copy(ctrl.data() + t * model->nu, data->ctrl, model->nu);
    mju_copy(qfrc_actuator.data() + t * model->nv, data->qfrc_actuator,
             model->nv);
    mju_copy(sensordata.data() + t * model->nsensordata, data->sensordata,
             model->nsensordata);

    // step using mj_Euler since mj_forward has been called
    // see mj_ step implementation here
    // https://github.com/deepmind/mujoco/blob/main/src/engine/engine_forward.c#L831
    mj_Euler(model, data);
  }

  // final cache
  mju_copy(qpos.data() + history * model->nq, data->qpos, model->nq);
  mju_copy(qvel.data() + history * model->nv, data->qvel, model->nv);
  mju_copy(sensordata.data() + history * model->nsensor, data->sensordata,
           model->nsensordata);

  // printf("\n");
  // printf("qpos: \n");
  // mju_printMat(qpos.data(), history + 1, model->nq);

  // printf("qvel: \n");
  // mju_printMat(qvel.data(), history + 1, model->nv);

  // printf("qacc: \n");
  // mju_printMat(qacc.data(), history, model->nv);

  // printf("ctrl: \n");
  // mju_printMat(ctrl.data(), history, model->nu);

  // printf("qfrc_actuator: \n");
  // mju_printMat(qfrc_actuator.data(), history, model->nv);

  // printf("sensordata:\n");
  // mju_printMat(sensordata.data(), history + 1, model->nsensordata);

  // finite-difference estimates
  std::vector<double> velocity(model->nv * history);
  std::vector<double> acceleration(model->nv * (history - 1));

  ConfigurationToVelocity(velocity.data(), qpos.data(), history + 1, model);
  VelocityToAcceleration(acceleration.data(), velocity.data(), history, model);

  // printf("velocity:\n");
  // mju_printMat(velocity.data(), history, model->nv);

  // velocity error 
  std::vector<double> velocity_error(model->nv * history);
  mju_sub(velocity_error.data(), velocity.data(), qvel.data() + model->nv, model->nv * (history - 1));
  EXPECT_NEAR(mju_norm(velocity_error.data(), model->nv * (history - 1)) / (model->nv * (history - 1)), 0.0, 1.0e-3);

  // printf("acceleration:\n");
  // mju_printMat(acceleration.data(), history - 1, model->nv);

  // acceleration error 
  std::vector<double> acceleration_error(model->nv * history);
  mju_sub(acceleration_error.data(), acceleration.data(), qacc.data() + model->nv, model->nv * (history - 2));
  // // printf("acceleration error: %f\n", mju_norm(acceleration_error.data(), model->nv * (history - 1)) / (model->nv * (history - 1)));
  EXPECT_NEAR(mju_norm(acceleration_error.data(), model->nv * (history - 1)) / (model->nv * (history - 1)), 0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(FiniteDifferenceVelocityAcceleration, Box3D) {
  // printf("Box 3D Test\n");

  // load model
  mjModel* model = LoadTestModel("box3D.xml");
  mjData* data = mj_makeData(model);
  
  // info 
  // printf("nq: %i\n", model->nq);
  // printf("nv: %i\n", model->nv);
  // printf("nu: %i\n", model->nu);
  // printf("ny: %i\n", model->nsensordata);

  // ----- simulate ----- //
  // trajectories 
  int history = 5;
  std::vector<double> qpos(model->nq * (history + 1));
  std::vector<double> qvel(model->nv * (history + 1));
  std::vector<double> qacc(model->nv * history);
  std::vector<double> ctrl(model->nu * history);
  std::vector<double> qfrc_actuator(model->nv * history);
  std::vector<double> sensordata(model->nsensordata * (history + 1));

  // reset 
  mj_resetData(model, data);

  // initialize TODO(taylor): improve initialization
  double qpos0[7] = {0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0};
  double qvel0[6] = {0.4, 0.05, -0.22, 0.01, -0.03, 0.24};
  mju_copy(data->qpos, qpos0, model->nq);
  mju_copy(data->qvel, qvel0, model->nv);

  // rollout
  for (int t = 0; t < history; t++) {
    // set control
    // controller(data->ctrl, data->time);
    mju_zero(data->ctrl, model->nu);

    // forward computes instantaneous qacc
    mj_forward(model, data);

    // cache
    mju_copy(qpos.data() + t * model->nq, data->qpos, model->nq);
    mju_copy(qvel.data() + t * model->nv, data->qvel, model->nv);
    mju_copy(qacc.data() + t * model->nv, data->qacc, model->nv);
    mju_copy(ctrl.data() + t * model->nu, data->ctrl, model->nu);
    mju_copy(qfrc_actuator.data() + t * model->nv, data->qfrc_actuator,
             model->nv);
    mju_copy(sensordata.data() + t * model->nsensordata, data->sensordata,
             model->nsensordata);

    // step using mj_Euler since mj_forward has been called
    // see mj_ step implementation here
    // https://github.com/deepmind/mujoco/blob/main/src/engine/engine_forward.c#L831
    mj_Euler(model, data);
  }

  // final cache
  mju_copy(qpos.data() + history * model->nq, data->qpos, model->nq);
  mju_copy(qvel.data() + history * model->nv, data->qvel, model->nv);
  mju_copy(sensordata.data() + history * model->nsensor, data->sensordata,
           model->nsensordata);

  // printf("\n");
  // printf("qpos: \n");
  // mju_printMat(qpos.data(), history + 1, model->nq);

  // printf("qvel: \n");
  // mju_printMat(qvel.data(), history + 1, model->nv);

  // printf("qacc: \n");
  // mju_printMat(qacc.data(), history, model->nv);

  // printf("qfrc_actuator: \n");
  // mju_printMat(qfrc_actuator.data(), history, model->nv);

  // printf("sensordata:\n");
  // mju_printMat(sensordata.data(), history + 1, model->nsensordata);

  // finite-difference estimates
  std::vector<double> velocity(model->nv * history);
  std::vector<double> acceleration(model->nv * (history - 1));

  ConfigurationToVelocity(velocity.data(), qpos.data(), history + 1, model);
  VelocityToAcceleration(acceleration.data(), velocity.data(), history, model);

  // printf("velocity:\n");
  // mju_printMat(velocity.data(), history, model->nv);

  // velocity error 
  std::vector<double> velocity_error(model->nv * history);
  mju_sub(velocity_error.data(), velocity.data(), qvel.data() + model->nv, model->nv * (history - 1));
  EXPECT_NEAR(mju_norm(velocity_error.data(), model->nv * (history - 1)) / (model->nv * (history - 1)), 0.0, 1.0e-3);

  // printf("acceleration:\n");
  // mju_printMat(acceleration.data(), history - 1, model->nv);

  // acceleration error 
  std::vector<double> acceleration_error(model->nv * history);
  mju_sub(acceleration_error.data(), acceleration.data(), qacc.data() + model->nv, model->nv * (history - 2));
  // // // printf("acceleration error: %f\n", mju_norm(acceleration_error.data(), model->nv * (history - 1)) / (model->nv * (history - 1)));
  EXPECT_NEAR(mju_norm(acceleration_error.data(), model->nv * (history - 1)) / (model->nv * (history - 1)), 0.0, 1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
