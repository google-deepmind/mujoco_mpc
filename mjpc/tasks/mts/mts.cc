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

#include "tasks/mts/mts.h"

#include <absl/random/random.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "task.h"
#include "utilities.h"

namespace mjpc {

// ---------- Residuals for in-panda manipulation task ---------
//   Number of residuals: 5
//     Residual (0): cube_position - palm_position
//     Residual (1): cube_orientation - cube_goal_orientation
//     Residual (2): cube linear velocity
//     Residual (3): cube angular velocity
//     Residual (4): control
// ------------------------------------------------------------
void MTS::Residual(const double* parameters, const mjModel* model,
                         const mjData* data, double* residual) {
  int counter = 0;

  // reach
  double* hand = mjpc::SensorByName(model, data, "hand");
  double* object = mjpc::SensorByName(model, data, "object");
  mju_sub3(residual + counter, hand, object);
  counter += 3;

  // bring
  for (int i=0; i < 8; i++) {
    double* object = mjpc::SensorByName(model, data, std::to_string(i).c_str());
    double* target = mjpc::SensorByName(model, data,
                                        (std::to_string(i) + "t").c_str());
    residual[counter++] = mju_dist3(object, target);
  }

  // careful
  residual[counter] = 0;
  int panda_id = mj_name2id(model, mjOBJ_BODY, "panda/");
  int object_id = mj_name2id(model, mjOBJ_BODY, "object");
  for (int i=0; i < data->ncon; i++) {
    int b1 = model->geom_bodyid[data->contact[i].geom1];
    int b2 = model->geom_bodyid[data->contact[i].geom2];
    int r1 = model->body_rootid[b1];
    int r2 = model->body_rootid[b2];
    if (r1 == panda_id || r2 == panda_id) {  // contact with the panda
      if (r2 == object_id || r1 == object_id) {
        continue;  // contact with the object is okay
      }
      mjtNum force[6];
      mj_contactForce(model, data, i, force);
      residual[counter] += mju_norm3(force);
    }
  }
  residual[counter] = mju_log10(residual[counter] + 1);
  counter++;

  // away
  residual[counter++] = mju_min(0, hand[2] - 0.7);


  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}

int MTS::Transition(int stage, const mjModel* model, mjData* data, Task* task) {
  double residuals[100];
  double terms[10];
  task->Residuals(model, data, residuals);
  task->CostTerms(terms, residuals);

  // bring is solved:
  if (data->time > 0 && stage == 0 && terms[1] < 0.01) {
    task->weight[0] = 0;  // disable reach
    task->weight[3] = 1;  // enable away

    // return stage: away
    return 1;
  }

  // away is solved, reset:
  if (stage == 1 && terms[3] < 0.01) {
    task->weight[0] = 1;  // enable reach
    task->weight[3] = 0;  // disable away

    // initialise object:
    absl::BitGen gen_;
    data->qpos[0] = absl::Uniform<double>(gen_, .4, .7);
    data->qpos[1] = absl::Uniform<double>(gen_, -.15, .15);
    data->qpos[2] = 0.2;
    data->qpos[3] = absl::Uniform<double>(gen_, -1, 1);
    data->qpos[4] = absl::Uniform<double>(gen_, -1, 1);
    data->qpos[5] = absl::Uniform<double>(gen_, -1, 1);
    data->qpos[6] = absl::Uniform<double>(gen_, -1, 1);
    mju_normalize4(data->qpos + 3);

    // initialise target:
    data->qpos[7+0] = 0.6;
    data->qpos[7+1] = 0;
    data->qpos[7+2] = 0.15;
    data->qpos[7+3] = absl::Uniform<double>(gen_, -1, 1);
    data->qpos[7+4] = absl::Uniform<double>(gen_, -1, 1);
    data->qpos[7+5] = absl::Uniform<double>(gen_, -1, 1);
    data->qpos[7+6] = absl::Uniform<double>(gen_, -1, 1);
    mju_normalize4(data->qpos + 13);

    // return stage: bring
    return 0;
  }

  return stage;
}
}  // namespace mjpc
