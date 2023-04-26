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

#include "gtest/gtest.h"
#include "mjpc/test/load.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(DifferentiateTest, State) {
  printf("Differentiate Test\n");

  // load model
  mjModel* model = LoadTestModel("box.xml");
  mjData* data = mj_makeData(model);

  // copy qpos 
  double qpos[7];
  mju_copy(qpos, data->qpos, model->nq);

  printf("qpos: \n");
  mju_printMat(qpos, 1, model->nq);

  printf("integrate: \n");
  double dq[6] = {0.0};
  printf("dq: \n");
  mju_printMat(dq, 1, model->nv);
  mj_integratePos(model, qpos, dq, 1.0);
  printf("integrated qpos: \n");
  mju_printMat(qpos, 1, model->nq);

  printf("integrated qpos - dq = {0.1, ...}\n");
  dq[0] = 0.1;
  mj_integratePos(model, qpos, dq, 1.0);
  mju_printMat(qpos, 1, model->nq);

  printf("integrated qpos - dq = {0, 0, 0, 0.1, ...}\n");
  dq[0] = 0.0;
  dq[3] = 0.1;
  mj_integratePos(model, qpos, dq, 1.0);
  mju_printMat(qpos, 1, model->nq);

  // ----- state differentiate ----- //
}

}  // namespace
}  // namespace mjpc
