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

#include "mjpc/test/simulation.h"

#include <functional>

#include <mujoco/mujoco.h>

#include "mjpc/direct/trajectory.h"

namespace mjpc {

// constructor
Simulation::Simulation(const mjModel* model, int length) {
  // model + data
  if (this->model) mj_deleteModel(this->model);
  this->model = mj_copyModel(nullptr, model);
  data_ = mj_makeData(model);

  // rollout length
  length_ = length;

  // trajectories
  qpos.Initialize(model->nq, length);
  qvel.Initialize(model->nv, length);
  qacc.Initialize(model->nv, length);
  ctrl.Initialize(model->nu, length);
  time.Initialize(1, length);
  sensor.Initialize(model->nsensordata, length);
  qfrc_actuator.Initialize(model->nv, length);
}

// set state
void Simulation::SetState(const double* qpos, const double* qvel) {
  if (qpos) mju_copy(data_->qpos, qpos, model->nq);
  if (qvel) mju_copy(data_->qvel, qvel, model->nv);
}

// rollout
void Simulation::Rollout(
    std::function<void(double* ctrl, double time)> controller) {
  for (int t = 0; t < length_; t++) {
    // set ctrl
    if (controller) controller(data_->ctrl, data_->time);

    // forward computes instantaneous qacc
    mj_forward(model, data_);

    // cache
    qpos.Set(data_->qpos, t);
    qvel.Set(data_->qvel, t);
    qacc.Set(data_->qacc, t);
    ctrl.Set(data_->ctrl, t);
    time.Set(&data_->time, t);
    sensor.Set(data_->sensordata, t);
    qfrc_actuator.Set(data_->qfrc_actuator, t);

    // step using mj_Euler since mj_forward has been called
    // see mj_ step implementation here
    // https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_forward.c#L831
    mj_Euler(model, data_);
  }
}

}  // namespace mjpc
