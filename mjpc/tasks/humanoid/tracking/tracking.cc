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

#include "mjpc/tasks/humanoid/tracking/tracking.h"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace {
// compute interpolation between mocap frames
std::tuple<int, int, double, double> ComputeInterpolationValues(double index,
                                                                int max_index) {
  int index_0 = std::floor(std::clamp(index, 0.0, (double)max_index));
  int index_1 = std::min(index_0 + 1, max_index);

  double weight_1 = std::clamp(index, 0.0, (double)max_index) - index_0;
  double weight_0 = 1.0 - weight_1;

  return {index_0, index_1, weight_0, weight_1};
}

// return length of motion trajectory
int TrajectoryLength(int id) {
  // Jump - CMU-CMU-02-02_04
  if (id == 0) {
    return 121;
    // Kick Spin - CMU-CMU-87-87_01
  } else if (id == 1) {
    return 154;
    // Spin Kick - CMU-CMU-88-88_06
  } else if (id == 2) {
    return 115;
    // Cartwheel (1) - CMU-CMU-88-88_07
  } else if (id == 3) {
    return 78;
    // Crouch Flip - CMU-CMU-88-88_08
  } else if (id == 4) {
    return 145;
    // Cartwheel (2) - CMU-CMU-88-88_09
  } else if (id == 5) {
    return 188;
    // Monkey Flip - CMU-CMU-90-90_19
  } else if (id == 6) {
    return 260;
    // Dance - CMU-CMU-103-103_08
  } else if (id == 7) {
    return 279;
    // Run - CMU-CMU-108-108_13
  } else if (id == 8) {
    return 39;
    // Walk - CMU-CMU-137-137_40
  } else if (id == 9) {
    return 510;
  }
  // TODO(taylor): Loop
  return 121 + 154 + 115 + 78 + 145 + 188 + 260 + 279 + 39 + 510;
}

// return starting keyframe index for motion
int MotionStartIndex(int id) {
  int start = 0;
  for (int i = 0; i < id; i++) {
    start += TrajectoryLength(i);
  }
  return start;
}

// names for humanoid bodies
const std::array<std::string, 16> body_names = {
    "pelvis",    "head",      "ltoe",  "rtoe",  "lheel",  "rheel",
    "lknee",     "rknee",     "lhand", "rhand", "lelbow", "relbow",
    "lshoulder", "rshoulder", "lhip",  "rhip",
};

}  // namespace

namespace mjpc {

std::string humanoid::Tracking::XmlPath() const {
  return GetModelPath("humanoid/tracking/task.xml");
}
std::string humanoid::Tracking::Name() const { return "Humanoid Track"; }

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
void humanoid::Tracking::Residual(const mjModel *model, const mjData *data,
                                  double *residual) const {
  // ----- get mocap frames ----- //
  // Hardcoded constant matching keyframes from CMU mocap dataset.
  float fps = 30.0;
  int start = MotionStartIndex(current_stage_);
  int length = TrajectoryLength(current_stage_);
  double current_index = (data->time - reference_time_) * fps + start;
  int last_key_index = start + length - 1;

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

  // ----- position ----- //
  // Compute interpolated frame.
  auto get_body_mpos = [&](const std::string &body_name, double result[3]) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    // current frame
    mju_scl3(
        result,
        model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid,
        weight_0);

    // next frame
    mju_addToScl3(
        result,
        model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid,
        weight_1);
  };

  auto get_body_sensor_pos = [&](const std::string &body_name,
                                 double result[3]) {
    std::string pos_sensor_name = "tracking_pos[" + body_name + "]";
    double *sensor_pos = SensorByName(model, data, pos_sensor_name.c_str());
    mju_copy3(result, sensor_pos);
  };

  // compute marker and sensor averages
  double avg_mpos[3] = {0};
  double avg_sensor_pos[3] = {0};
  int num_body = 0;
  for (const auto &body_name : body_names) {
    double body_mpos[3];
    double body_sensor_pos[3];
    get_body_mpos(body_name, body_mpos);
    mju_addTo3(avg_mpos, body_mpos);
    get_body_sensor_pos(body_name, body_sensor_pos);
    mju_addTo3(avg_sensor_pos, body_sensor_pos);
    num_body++;
  }
  mju_scl3(avg_mpos, avg_mpos, 1.0/num_body);
  mju_scl3(avg_sensor_pos, avg_sensor_pos, 1.0/num_body);

  // residual for averages
  mju_sub3(&residual[counter], avg_mpos, avg_sensor_pos);
  counter += 3;

  for (const auto &body_name : body_names) {
    double body_mpos[3];
    get_body_mpos(body_name, body_mpos);

    // current position
    double body_sensor_pos[3];
    get_body_sensor_pos(body_name, body_sensor_pos);

    mju_subFrom3(body_mpos, avg_mpos);
    mju_subFrom3(body_sensor_pos, avg_sensor_pos);

    mju_sub3(&residual[counter], body_mpos, body_sensor_pos);

    counter += 3;
  }

  // ----- velocity ----- //
  for (const auto &body_name : body_names) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    std::string linvel_sensor_name = "tracking_linvel[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    // compute finite-difference velocity
    mju_copy3(
        &residual[counter],
        model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid);
    mju_subFrom3(
        &residual[counter],
        model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid);
    mju_scl3(&residual[counter], &residual[counter], fps);

    // subtract current velocity
    double *sensor_linvel =
        SensorByName(model, data, linvel_sensor_name.c_str());
    mju_subFrom3(&residual[counter], sensor_linvel);

    counter += 3;
  }


  CheckSensorDim(model, counter);
}

// --------------------- Transition for humanoid task -------------------------
//   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
//   Linearly interpolate between two consecutive key frames in order to
//   smooth the transitions between keyframes.
// ----------------------------------------------------------------------------
void humanoid::Tracking::Transition(const mjModel *model, mjData *d) {
  // Hardcoded constant matching keyframes from CMU mocap dataset.
  float fps = 30.0;

  // get motion trajectory length
  int length = TrajectoryLength(stage);

  // get motion start index
  int start = MotionStartIndex(stage);

  // check for motion switch
  if (current_stage_ != stage || d->time == 0.0) {
    current_stage_ = stage;        // set motion id
    reference_time_ = d->time;      // set reference time

    // set initial state
    mju_copy(d->qpos, model->key_qpos + model->nq * start, model->nq);
    mju_copy(d->qvel, model->key_qvel + model->nv * start, model->nv);
  }

  // indices
  double current_index = (d->time - reference_time_) * fps + start;
  int last_key_index = start + length - 1;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  mjMARKSTACK;

  mjtNum *mocap_pos_0 = mj_stackAlloc(d, 3 * model->nmocap);
  mjtNum *mocap_pos_1 = mj_stackAlloc(d, 3 * model->nmocap);

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
