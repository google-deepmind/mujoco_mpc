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

#include "mjpc/estimators/ekf.h"

#include <mujoco/mujoco.h>

#include <chrono>
#include <vector>

#include "mjpc/utilities.h"

namespace mjpc {

// initialize
void EKF::Initialize(mjModel* model) {
  // model
  this->model = model;

  // data
  data_ = mj_makeData(model);

  // dimension
  nstate_ = model->nq + model->nv + model->na;
  ndstate_ = 2 * model->nv + model->na;
  int ns = model->nsensordata;

  // state
  state.resize(nstate_);

  // covariance
  covariance.resize(ndstate_ * ndstate_);

  // process noise
  noise_process.resize(ndstate_);

  // sensor noise
  noise_sensor.resize(ns);

  // dynamics Jacobian
  dynamics_jacobian_.resize(ndstate_ * ndstate_);

  // sensor Jacobian
  sensor_jacobian_.resize(ns * ndstate_);

  // Kalman gain
  kalman_gain_.resize(ndstate_ * ns);

  // sensor error
  sensor_error_.resize(ns);

  // correction
  correction_.resize(ndstate_);

  // scratch
  tmp0_.resize(ndstate_ * ns);
  tmp1_.resize(ns * ns);
  tmp2_.resize(ns * ndstate_);
  tmp3_.resize(ndstate_ * ndstate_);
  tmp4_.resize(ndstate_ * ndstate_);
}

// reset memory
void EKF::Reset() {
  // dimension
  int nq = model->nq, nv = model->nv, na = model->na, ns = model->nsensordata;

  // data
  // mj_resetData(model, data_);

  // state
  mju_copy(state.data(), model->qpos0, nq);
  mju_zero(state.data() + nq, nv);
  mju_zero(state.data() + nq + nv, na);
  time = 0.0;

  // covariance
  mju_eye(covariance.data(), ndstate_);

  // process noise
  mju_zero(noise_process.data(), ndstate_);

  // sensor noise
  mju_zero(noise_sensor.data(), ns);

  // dynamics Jacobian
  mju_zero(dynamics_jacobian_.data(), ndstate_ * ndstate_);

  // sensor Jacobian
  mju_zero(sensor_jacobian_.data(), ns * ndstate_);

  // Kalman gain
  mju_zero(kalman_gain_.data(), ndstate_ * ns);

  // sensor error
  mju_zero(sensor_error_.data(), ns);

  // correction
  mju_zero(correction_.data(), ndstate_);

  // timer
  timer_measurement_ = 0.0;
  timer_prediction_ = 0.0;

  // scratch
  std::fill(tmp0_.begin(), tmp0_.end(), 0.0);
  std::fill(tmp1_.begin(), tmp1_.end(), 0.0);
  std::fill(tmp2_.begin(), tmp2_.end(), 0.0);
  std::fill(tmp3_.begin(), tmp3_.end(), 0.0);
  std::fill(tmp4_.begin(), tmp4_.end(), 0.0);
}

// update measurement
void EKF::UpdateMeasurement(const double* ctrl, const double* sensor) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimensions
  int nq = model->nq, nv = model->nv, na = model->na, nu = model->nu,
      ns = model->nsensordata;

  // set state
  mju_copy(data_->qpos, state.data(), nq);
  mju_copy(data_->qvel, state.data() + nq, nv);
  mju_copy(data_->act, state.data() + nq + nv, na);

  // set ctrl
  mju_copy(data_->ctrl, ctrl, nu);

  // forward to get sensor
  mj_forward(model, data_);

  mju_sub(sensor_error_.data(), sensor, data_->sensordata, ns);

  // -- Kalman gain: P * C' (C * P * C' + R)^-1 -- //

  // sensor Jacobian
  mjd_transitionFD(model, data_, settings.epsilon, settings.flg_centered, NULL,
                   NULL, sensor_jacobian_.data(), NULL);

  // P * C' = tmp0
  mju_mulMatMatT(tmp0_.data(), covariance.data(), sensor_jacobian_.data(),
                 ndstate_, ndstate_, ns);

  // C * P * C' = C * tmp0 = tmp1
  mju_mulMatMat(tmp1_.data(), sensor_jacobian_.data(), tmp0_.data(), ns,
                ndstate_, ns);

  // C * P * C' + R
  for (int i = 0; i < ns; i++) {
    tmp1_[ns * i + i] += noise_sensor[i];
  }

  // factorize: C * P * C' + R
  double reg = 0.0;
  double scl_reg = 10.0;
  int rank = 0;
  while (rank < ns) {
    // regularize
    if (reg > 0.0) {
      for (int i = 0; i < ns; i++) {
        tmp1_[ns * i + i] += (1.0 - 1.0 / scl_reg) * reg;
      }
    }
    // factorize
    rank = mju_cholFactor(tmp1_.data(), ns, 0.0);

    // check failure
    if (rank < ns && reg >= 1.0e16) {
      mju_error("measurement update rank: (%i / %i)\n", rank, ns);
    }

    // increase regularization
    if (reg > 0.0) {
      reg *= scl_reg;
    } else {
      reg = 1.0e-6;
    }
  }


  // -- correction: (P * C') * (C * P * C' + R)^-1 * sensor_error -- //

  // tmp2 = (C * P * C' + R) \ sensor_error
  mju_cholSolve(tmp2_.data(), tmp1_.data(), sensor_error_.data(), ns);

  // correction = (P * C') * (C * P * C' + R) \ sensor_error = tmp0 * tmp2
  mju_mulMatVec(correction_.data(), tmp0_.data(), tmp2_.data(), ndstate_, ns);

  // -- state update -- //

  // configuration
  mj_integratePos(model, state.data(), correction_.data(), 1.0);

  // velocity + act
  mju_addTo(state.data() + nq, correction_.data() + nv, nv + na);

  // -- covariance update -- //

  // tmp2 = (C * P * C' + R)^-1 (C * P) = tmp1 \ tmp0'
  for (int i = 0; i < ndstate_; i++) {
    mju_cholSolve(tmp2_.data() + ns * i, tmp1_.data(), tmp0_.data() + ns * i,
                  ns);
  }

  // tmp3 = (P * C') * (C * P * C' + R)^-1 (C * P) = tmp0 * tmp2'
  mju_mulMatMatT(tmp3_.data(), tmp0_.data(), tmp2_.data(), ndstate_, ns,
                 ndstate_);

  // covariance -= tmp3
  mju_subFrom(covariance.data(), tmp3_.data(), ndstate_ * ndstate_);

  // symmetrize
  mju_symmetrize(covariance.data(), covariance.data(), ndstate_);

  // stop timer (ms)
  timer_measurement_ = 1.0e-3 * GetDuration(start);
}

// update time
void EKF::UpdatePrediction() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimensions
  int nq = model->nq, nv = model->nv, na = model->na;

  // set state
  mju_copy(data_->qpos, state.data(), nq);
  mju_copy(data_->qvel, state.data() + nq, nv);
  mju_copy(data_->act, state.data() + nq + nv, na);

  // dynamics Jacobian
  mjd_transitionFD(model, data_, settings.epsilon, settings.flg_centered,
                   dynamics_jacobian_.data(), NULL, NULL, NULL);

  // integrate state
  // TODO(taylor): integrator option
  // mj_Euler(model, data_);
  mj_step(model, data_);

  // update state
  mju_copy(state.data(), data_->qpos, nq);
  mju_copy(state.data() + nq, data_->qvel, nv);
  mju_copy(state.data() + nq + nv, data_->act, na);

  // -- update covariance: P = A * P * A' -- //

  //  tmp = P * A'
  mju_mulMatMatT(tmp3_.data(), covariance.data(), dynamics_jacobian_.data(),
                 ndstate_, ndstate_, ndstate_);

  // P = A * tmp
  mju_mulMatMat(covariance.data(), dynamics_jacobian_.data(), tmp3_.data(),
                ndstate_, ndstate_, ndstate_);

  // process noise
  for (int i = 0; i < ndstate_; i++) {
    covariance[ndstate_ * i + i] += noise_process[i];
  }

  // symmetrize
  mju_symmetrize(covariance.data(), covariance.data(), ndstate_);

  // stop timer
  timer_prediction_ = 1.0e-3 * GetDuration(start);
}

}  // namespace mjpc
