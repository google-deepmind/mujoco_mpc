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

#include "mjpc/states/state.h"

#include <algorithm>
#include <mutex>
#include <shared_mutex>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {

// allocate memory
void State::Allocate(const mjModel* model) {
  const std::unique_lock<std::shared_mutex> lock(mtx_);
  state_.resize(model->nq + model->nv + model->na);
  mocap_.resize(7 * model->nmocap);
  userdata_.resize(model->nuserdata);
}

// reset memory to zeros
void State::Reset() {
  const std::unique_lock<std::shared_mutex> lock(mtx_);
  std::fill(state_.begin(), state_.end(), (double)0.0);
  std::fill(mocap_.begin(), mocap_.end(), 0.0);
  std::fill(userdata_.begin(), userdata_.end(), 0.0);
  time_ = 0.0;
}

// set state from data
void State::Set(const mjModel* model, const mjData* data) {
  if (model && data) {
    const std::unique_lock<std::shared_mutex> lock(mtx_);

    state_.resize(model->nq + model->nv + model->na);
    mocap_.resize(7 * model->nmocap);

    // state
    mju_copy(state_.data(), data->qpos, model->nq);
    mju_copy(DataAt(state_, model->nq), data->qvel, model->nv);
    mju_copy(DataAt(state_, model->nq + model->nv), data->act, model->na);

    // mocap
    for (int i = 0; i < model->nmocap; i++) {
      mju_copy(DataAt(mocap_, 7 * i), data->mocap_pos + 3 * i, 3);
      mju_copy(DataAt(mocap_, 7 * i + 3), data->mocap_quat + 4 * i, 4);
    }

    // userdata
    mju_copy(userdata_.data(), data->userdata, model->nuserdata);

    // time
    time_ = data->time;
  }
}

// set qpos
void State::SetPos(const mjModel* model, const double* qpos) {
  mju_copy(state_.data(), qpos, model->nq);
}

// set qvel
void State::SetVel(const mjModel* model, const double* qvel) {
  mju_copy(DataAt(state_, model->nq), qvel, model->nv);
}

// set time
void State::SetTime(const mjModel* model, double time) {
  time_ = time;
}

void State::CopyTo(double* dst_state, double* dst_mocap,
                   double* dst_userdata, double* dst_time) const {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  mju_copy(dst_state, this->state_.data(), this->state_.size());
  *dst_time = this->time_;
  mju_copy(dst_mocap, this->mocap_.data(), this->mocap_.size());
  mju_copy(dst_userdata, this->userdata_.data(), this->userdata_.size());
}

void State::CopyTo(const mjModel* model, mjData* data) const {
  const std::shared_lock<std::shared_mutex> lock(mtx_);

    // state
    mju_copy(data->qpos, state_.data(), model->nq);
    mju_copy(data->qvel, DataAt(state_, model->nq), model->nv);
    mju_copy(data->act, DataAt(state_, model->nq + model->nv), model->na);

    // mocap
    for (int i = 0; i < model->nmocap; i++) {
      mju_copy(data->mocap_pos + 3 * i, DataAt(mocap_, 7 * i), 3);
      mju_copy(data->mocap_quat + 4 * i, DataAt(mocap_, 7 * i + 3), 4);
    }

    // userdata
    mju_copy(data->userdata, userdata_.data(), model->nuserdata);

    // time
    data->time = time_;
}

}  // namespace mjpc
