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

#include "tasks/humanoid/tracking/task.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>

#include "task.h"
#include "utilities.h"
#include <mujoco/mujoco.h>

namespace {
std::tuple<int, int, double, double> ComputeInterpolationValues(double index,
                                                                int max_index) {
  int index_0 = std::floor(std::clamp(index, 0.0, (double)max_index));
  int index_1 = std::min(index_0 + 1, max_index);

  double weight_1 = std::clamp(index, 0.0, (double)max_index) - index_0;
  double weight_0 = 1.0 - weight_1;

  return {index_0, index_1, weight_0, weight_1};
}
}  // namespace

namespace mjpc {

// ------------- Residuals for humanoid tracking task -------------
//   Number of residuals:
//     Residual (0): Joint vel: minimise joint velocity
//     Residual (1): Control: minimise control
//     Residual (2-11): Tracking position: minimise tracking position error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//     Residual (11-20): Tracking velocity: minimise tracking velocity error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//   Number of parameters: 0
// ----------------------------------------------------------------
void humanoid::Tracking::Residual(const double *parameters,
                                  const mjModel *model, const mjData *data,
                                  double *residual) {
  // ----- get mocap frames ----- //
  // Hardcoded constant matching keyframes from CMU mocap dataset.
  float fps = 30.0;
  double current_index = data->time * fps;
  int last_key_index = (model->nkey) - 1;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  // ----- residual ----- //
  int counter = 0;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
  counter += model->nv - 6;

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  std::array<std::string, 16> body_names = {
      "pelvis",    "head",      "ltoe",  "rtoe",  "lheel",  "rheel",
      "lknee",     "rknee",     "lhand", "rhand", "lelbow", "relbow",
      "lshoulder", "rshoulder", "lhip",  "rhip",
  };

  // Compute interpolated frame.
  auto get_body_mpos = [&](const std::string &body_name, double result[3]) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    // current frame
    mju_scl3(result,
             model->key_mpos + model->nmocap * 3 * key_index_0 +
                 3 * body_mocapid,
             weight_0);

    // next frame
    mju_addToScl3(result,
                  model->key_mpos + model->nmocap * 3 * key_index_1 +
                      3 * body_mocapid,
                  weight_1);
  };

  auto get_body_sensor_pos = [&](const std::string &body_name,
                                 double result[3]) {
    std::string pos_sensor_name = "tracking_pos[" + body_name + "]";
    double *sensor_pos = SensorByName(model, data, pos_sensor_name.c_str());
    mju_copy3(result, sensor_pos);
  };

  double pelvis_mpos[3];
  get_body_mpos("pelvis", pelvis_mpos);

  double pelvis_sensor_pos[3];
  get_body_sensor_pos("pelvis", pelvis_sensor_pos);

  for (const auto &body_name : body_names) {
    double body_mpos[3];
    get_body_mpos(body_name, body_mpos);

    // current position
    double body_sensor_pos[3];
    get_body_sensor_pos(body_name, body_sensor_pos);

    if (body_name != "pelvis") {
      mju_subFrom3(body_mpos, pelvis_mpos);
      mju_subFrom3(body_sensor_pos, pelvis_sensor_pos);
    }

    mju_sub3(&residual[counter], body_mpos, body_sensor_pos);

    if (body_name != "pelvis") {
      if (0.85 < pelvis_sensor_pos[2] && pelvis_sensor_pos[2] < 0.95) {
        residual[counter + 2] = residual[counter + 2] * 0.3;
      }
    }

    counter += 3;
  }

  for (const auto &body_name : body_names) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    std::string linvel_sensor_name = "tracking_linvel[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    // compute finite-difference velocity
    mju_copy3(&residual[counter], model->key_mpos +
                                      model->nmocap * 3 * key_index_1 +
                                      3 * body_mocapid);
    mju_subFrom3(&residual[counter], model->key_mpos +
                                         model->nmocap * 3 * key_index_0 +
                                         3 * body_mocapid);
    mju_scl3(&residual[counter], &residual[counter], fps);

    // subtract current velocity
    double *sensor_linvel =
        SensorByName(model, data, linvel_sensor_name.c_str());
    mju_subFrom3(&residual[counter], sensor_linvel);

    counter += 3;
  }

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    std::printf("user_sensor_dim=%d, counter=%d", user_sensor_dim, counter);
    mju_error_i("mismatch between total user-sensor dimension "
                "and actual length of residual %d",
                counter);
  }
}

// --------------------- Transition for humanoid task -------------------------
//   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
//   Linearly interpolate between two consecutive key frames in order to
//   smooth the transitions between keyframes.
// ----------------------------------------------------------------------------
void humanoid::Tracking::Transition(const mjModel *model, mjData *d,
                                    Task *task) {
  // Hardcoded constant matching keyframes from CMU mocap dataset.
  float fps = 30.0;
  double current_index = d->time * fps;
  int last_key_index = (model->nkey) - 1;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  mjMARKSTACK;

  mjtNum* mocap_pos_0 = mj_stackAlloc(d, 3 * model->nmocap);
  mjtNum* mocap_pos_1 = mj_stackAlloc(d, 3 * model->nmocap);

  // Compute interpolated frame.
  mju_scl(mocap_pos_0, model->key_mpos + model->nmocap * 3 * key_index_0,
          weight_0, model->nmocap * 3);

  mju_scl(mocap_pos_1, model->key_mpos + model->nmocap * 3 * key_index_1,
          weight_1, model->nmocap * 3);

  mju_copy(d->mocap_pos, mocap_pos_0, model->nmocap * 3);
  mju_addTo(d->mocap_pos, mocap_pos_1, model->nmocap * 3);
  mjFREESTACK;
}

}  // namespace mjpc
