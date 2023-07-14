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
  nstate_ = model->nq + model->nv;
  nvelocity_ = 2 * model->nv;
  int ns = model->nsensordata;

  // state
  state.resize(nstate_);

  // covariance
  covariance.resize(nvelocity_ * nvelocity_);

  // process noise
  noise_process.resize(nvelocity_);

  // sensor noise
  noise_sensor.resize(ns);

  // dynamics Jacobian
  dynamics_jacobian_.resize(nvelocity_ * nvelocity_);

  // sensor Jacobian
  sensor_jacobian_.resize(ns * nvelocity_);

  // Kalman gain
  kalman_gain_.resize(nvelocity_ * ns);

  // sensor error
  sensor_error_.resize(ns);

  // correction
  correction_.resize(nvelocity_);

  // scratch
  tmp0_.resize(nvelocity_ * ns);
  tmp1_.resize(ns * ns);
  tmp2_.resize(ns * nvelocity_);
  tmp3_.resize(nvelocity_ * nvelocity_);
}

// reset memory
void EKF::Reset() {
  // dimension 
  int nq = model->nq, nv = model->nv, ns = model->nsensordata;

  // data
  // mj_resetData(model, data_);

  // state
  mju_copy(state.data(), model->qpos0, nq);
  mju_zero(state.data() + nq, nv);
  time = 0.0;

  // covariance
  mju_eye(covariance.data(), nvelocity_);

  // process noise 
  mju_zero(noise_process.data(), nvelocity_);

  // sensor noise 
  mju_zero(noise_sensor.data(), ns);

  // dynamics Jacobian
  mju_zero(dynamics_jacobian_.data(), nvelocity_ * nvelocity_);

  // sensor Jacobian
  mju_zero(sensor_jacobian_.data(), ns * nvelocity_);

  // Kalman gain
  mju_zero(kalman_gain_.data(), nvelocity_ * ns);

  // sensor error
  mju_zero(sensor_error_.data(), ns);

  // correction
  mju_zero(correction_.data(), nvelocity_);

  // timer
  timer_measurement_ = 0.0;
  timer_prediction_ = 0.0;

  // scratch
  std::fill(tmp0_.begin(), tmp0_.end(), 0.0);
  std::fill(tmp1_.begin(), tmp1_.end(), 0.0);
  std::fill(tmp2_.begin(), tmp2_.end(), 0.0);
  std::fill(tmp3_.begin(), tmp3_.end(), 0.0);
}

// update measurement
void EKF::UpdateMeasurement(const double* ctrl, const double* sensor) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimensions
  int nq = model->nq, nv = model->nv, nu = model->nu, ns = model->nsensordata;

  // set state
  mju_copy(data_->qpos, state.data(), nq);
  mju_copy(data_->qvel, state.data() + nq, nv);

  // set ctrl
  mju_copy(data_->ctrl, ctrl, nu);

  // sensor Jacobian
  mjd_transitionFD(model, data_, settings.epsilon, settings.flg_centered, NULL,
                   NULL, sensor_jacobian_.data(), NULL);

  // forward to get sensor
  mj_forward(model, data_);

  // sensor error
  mju_sub(sensor_error_.data(), sensor, data_->sensordata, ns);

  // -- Kalman gain: P * C' (C * P * C' + R)^-1 -- //
  
  // P * C' = tmp0
  mju_mulMatMatT(tmp0_.data(), covariance.data(), sensor_jacobian_.data(),
                 nvelocity_, nvelocity_, ns);

  // C * P * C' = C * tmp0 = tmp1
  mju_mulMatMat(tmp1_.data(), sensor_jacobian_.data(), tmp0_.data(), ns,
                nvelocity_, ns);

  // C * P * C' + R
  for (int i = 0; i < ns; i++) {
    tmp1_[ns * i + i] += noise_sensor[i];
  }

  // factorize: C * P * C' + R
  mju_cholFactor(tmp1_.data(), ns, 0.0);

  // -- correction: (P * C') * (C * P * C' + R)^-1 * sensor_error -- //

  // tmp2 = (C * P * C' + R) \ sensor_error
  mju_cholSolve(tmp2_.data(), tmp1_.data(), sensor_error_.data(), ns);

  // correction = (P * C') * (C * P * C' + R) \ sensor_error = tmp0 * tmp2
  mju_mulMatVec(correction_.data(), tmp0_.data(), tmp2_.data(), nvelocity_, ns);

  // -- state update -- //

  // configuration
  mj_integratePos(model, state.data(), correction_.data(), 1.0);

  // velocity
  mju_addTo(state.data() + nq, correction_.data() + nv, nv);

  // -- covariance update -- //

  // tmp2 = (C * P * C' + R)^-1 (C * P) = tmp1 \ tmp0'
  for (int i = 0; i < nvelocity_; i++) {
    mju_cholSolve(tmp2_.data() + ns * i, tmp1_.data(), tmp0_.data() + ns * i,
                  ns);
  }

  // tmp3 = (P * C') * (C * P * C' + R)^-1 (C * P) = tmp0 * tmp2'
  mju_mulMatMatT(tmp3_.data(), tmp0_.data(), tmp2_.data(), nvelocity_, ns,
                 nvelocity_);

  // covariance -= tmp3
  mju_subFrom(covariance.data(), tmp3_.data(), nvelocity_ * nvelocity_);

  // stop timer (ms)
  timer_measurement_ = 1.0e-3 * GetDuration(start);

  // set time step
  if (settings.auto_timestep) {
    model->opt.timestep = 1.0e-3 * (timer_measurement_ + timer_prediction_);
  }
}

// update time
void EKF::UpdatePrediction() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // integrate state
  // TODO(taylor): integrator option
  mj_Euler(model, data_);

  // update state
  mju_copy(state.data(), data_->qpos, model->nq);
  mju_copy(state.data() + model->nq, data_->qvel, model->nv);

  // -- update covariance: P = A * P * A' -- //

  // dynamics Jacobian
  mjd_transitionFD(model, data_, settings.epsilon, settings.flg_centered,
                   dynamics_jacobian_.data(), NULL, NULL, NULL);

  //  tmp = P * A'
  mju_mulMatMatT(tmp3_.data(), covariance.data(), dynamics_jacobian_.data(),
                 nvelocity_, nvelocity_, nvelocity_);

  // P = A * tmp
  mju_mulMatMat(covariance.data(), dynamics_jacobian_.data(), tmp3_.data(),
                nvelocity_, nvelocity_, nvelocity_);

  // stop timer
  timer_prediction_ = 1.0e-3 * GetDuration(start);
}

}  // namespace mjpc
