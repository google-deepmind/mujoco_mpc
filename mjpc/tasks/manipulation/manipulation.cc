// Copyright 2022 DeepMind Technologies Limited
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

#include "mjpc/tasks/manipulation/manipulation.h"

#include <string>

#include <absl/container/flat_hash_map.h>
#include <absl/random/random.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/manipulation/common.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string manipulation::Bring::XmlPath() const {
  return GetModelPath("manipulation/task_panda_bring.xml");
}
std::string manipulation::Bring::Name() const { return "Panda Robotiq Bring"; }

void manipulation::Bring::Residual(const mjModel* model, const mjData* data,
                          double* residual) const {
  int counter = 0;

  // reach
  double hand[3] = {0};
  ComputeRobotiqHandPos(model, data, model_vals_, hand);

  double* object = SensorByName(model, data, "object");
  mju_sub3(residual + counter, hand, object);
  counter += 3;

  // bring
  for (int i=0; i < 8; i++) {
    double* object = SensorByName(model, data, std::to_string(i).c_str());
    double* target = SensorByName(model, data,
                                        (std::to_string(i) + "t").c_str());
    residual[counter++] = mju_dist3(object, target);
  }

  // careful
  int object_id = mj_name2id(model, mjOBJ_BODY, "object");
  residual[counter++] = CarefulCost(model, data, model_vals_, object_id);

  // away
  residual[counter++] = mju_min(0, hand[2] - 0.6);


  // sensor dim sanity check
  CheckSensorDim(model, counter);
}



void manipulation::Bring::Transition(const mjModel* model, mjData* data) {
  double residuals[100];
  double terms[10];
  Residual(model, data, residuals);
  CostTerms(terms, residuals, /*weighted=*/false);

  // bring is solved:
  if (data->time > 0 && data->userdata[0] == 0 && terms[1] < 0.04) {
    weight[0] = 0;  // disable reach
    weight[3] = 1;  // enable away

    data->userdata[0] = 1;
  }

  // away is solved, reset:
  if (data->userdata[0] == 1 && terms[3] < 0.01) {
    weight[0] = 1;  // enable reach
    weight[3] = 0;  // disable away

    absl::BitGen gen_;

    // initialise target:
    data->qpos[7+0] = 0.45;
    data->qpos[7+1] = 0;
    data->qpos[7+2] = 0.15;
    data->qpos[7+3] = absl::Uniform<double>(gen_, -1, 1);
    data->qpos[7+4] = absl::Uniform<double>(gen_, -1, 1);
    data->qpos[7+5] = absl::Uniform<double>(gen_, -1, 1);
    data->qpos[7+6] = absl::Uniform<double>(gen_, -1, 1);
    mju_normalize4(data->qpos + 13);

    // return stage: bring
    data->userdata[0] = 0;
  }
}

void manipulation::Bring::Reset(const mjModel* model) {
  Task::Reset(model);
  model_vals_ = ModelValues::FromModel(model);
}
}  // namespace mjpc
