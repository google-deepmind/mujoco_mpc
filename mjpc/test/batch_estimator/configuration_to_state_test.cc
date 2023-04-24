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

namespace mjpc {
namespace {

TEST(RepresentationTest, ConfigurationToStateTest) {
  printf("Representation Test\n");

  // load model
  mjModel* model = LoadTestModel("box.xml");
  mjData* data = mj_makeData(model);
  mj_forward(model, data);

  printf("model timestep: %f\n", model->opt.timestep);
  printf("initial state:\n");
  mju_printMat(data->qpos, 1, model->nq);
  mju_printMat(data->qvel, 1, model->nv);

  // rollout
  int T = 10;
  std::vector<double> qpos(T * model->nq);
  std::vector<double> qvel(T * model->nv);
  std::vector<double> times(T);

  for (int t = 0; t < T - 1; t++) {
    // copy
    mju_copy(qpos.data() + t * model->nq, data->qpos, model->nq);
    mju_copy(qvel.data() + t * model->nv, data->qvel, model->nv);
    times[t] = data->time;

    // step
    mj_step(model, data);
  }

  // copy
  mju_copy(qpos.data() + (T - 1) * model->nq, data->qpos, model->nq);
  mju_copy(qvel.data() + (T - 1) * model->nv, data->qvel, model->nv);
  times[T - 1] = data->time;

  // printf("final state:\n");
  // mju_printMat(data->qpos, 1, model->nq);
  // mju_printMat(data->qvel, 1, model->nv);

  printf("configuration:\n");
  mju_printMat(qpos.data(), T, model->nq);

  printf("velocity\n");
  mju_printMat(qvel.data(), T, model->nv);

  printf("finite-difference velocity\n");
  std::vector<double> fdvel((T - 1) * model->nv);

  ConfigurationToVelocity(fdvel.data(), qpos.data(), T, model);
  mju_printMat(fdvel.data(), T - 1, model->nv);

  printf("finite-difference acceleration\n");
  std::vector<double> fdacc((T - 2) * model->nv);
  VelocityToAcceleration(fdacc.data(), fdvel.data(), T - 1, model);
  mju_printMat(fdacc.data(), T - 2, model->nv);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
