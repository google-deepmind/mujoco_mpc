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

#include "mjpc/estimators/unscented.h"

#include <chrono>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/array_safety.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;

// initialize
void Unscented::Initialize(const mjModel* model) {
  // model
  if (this->model) mj_deleteModel(this->model);
  this->model = mj_copyModel(nullptr, model);

  // data
  if (this->data_) mj_deleteData(this->data_);
  data_ = mj_makeData(model);

  // settings
  settings.alpha = GetNumberOrDefault(1.0, model, "unscented_alpha");
  settings.beta = GetNumberOrDefault(2.0, model, "unscented_beta");

  // timestep
  this->model->opt.timestep = GetNumberOrDefault(this->model->opt.timestep,
                                                 model, "estimator_timestep");

  // dimension
  int nq = model->nq, nv = model->nv, na = model->na;
  nstate_ = nq + nv + na;
  ndstate_ = 2 * nv + na;
  nsigma_ = 2 * ndstate_ + 1;

  // sensor start index
  sensor_start_ = GetNumberOrDefault(0, model, "estimator_sensor_start");

  // number of sensors
  nsensor_ =
      GetNumberOrDefault(model->nsensor, model, "estimator_number_sensor");

  // sensor dimension
  nsensordata_ = 0;
  for (int i = 0; i < nsensor_; i++) {
    nsensordata_ += model->sensor_dim[sensor_start_ + i];
  }

  // sensor start index
  sensor_start_index_ = 0;
  for (int i = 0; i < sensor_start_; i++) {
    sensor_start_index_ += model->sensor_dim[i];
  }

  // state
  state.resize(nstate_);

  // covariance
  covariance.resize(ndstate_ * ndstate_);

  // process noise
  noise_process.resize(ndstate_);

  // sensor noise
  noise_sensor.resize(nsensordata_);

  // sigma points (nstate x (2 * ndstate_ + 1))
  sigma_.resize(nstate_ * nsigma_);

  // states (nstate x (2 * ndstate_ + 1))
  states_.resize(nstate_ * nsigma_);

  // sensors (nsensordata x (2 * ndstate + 1))
  sensors_.resize(nsensordata_ * nsigma_);

  // state mean (nstate)
  state_mean_.resize(nstate_);

  // sensor mean (nsensordata)
  sensor_mean_.resize(nsensordata_);

  // covariance factor (ndstate x ndstate)
  covariance_factor_.resize(ndstate_ * ndstate_);

  // factor column (ndstate)
  factor_column_.resize(ndstate_);

  // state difference (ndstate x nsigma_)
  state_difference_.resize(ndstate_ * nsigma_);

  // sensor difference (nsensordata_ x nsigma_)
  sensor_difference_.resize(nsensordata_ * nsigma_);

  // covariance sensor (nsensordata_ x nsensordata_)
  covariance_sensor_.resize(nsensordata_ * nsensordata_);

  // covariance state sensor (ndstate_ x nsensordata_)
  covariance_state_sensor_.resize(ndstate_ * nsensordata_);

  // covariance state state (ndstate_ x ndstate_)
  covariance_state_state_.resize(ndstate_ * ndstate_);

  // sensor difference outer product
  sensor_difference_outer_product_.resize(nsensordata_ * nsensordata_);

  // state sensor difference outer product
  state_sensor_difference_outer_product_.resize(ndstate_ * nsensordata_);

  // state state difference outer product
  state_state_difference_outer_product_.resize(ndstate_ * ndstate_);

  // covariance sensor factor
  covariance_sensor_factor_.resize(nsensordata_ * nsensordata_);

  // lambda
  double lambda = ndstate_ * (settings.alpha * settings.alpha - 1.0);

  // sigma step
  sigma_step = mju_sqrt(ndstate_ + lambda);

  // weights
  weight_mean0 = lambda / (ndstate_ + lambda);
  weight_covariance0 =
      weight_mean0 + 1.0 - settings.alpha * settings.alpha + settings.beta;
  weight_sigma = 1.0 / (2.0 * (ndstate_ + lambda));

  // sensor error
  sensor_error_.resize(nsensordata_);

  // correction
  correction_.resize(ndstate_);

  // scratch
  tmp0_.resize(nsensordata_ * ndstate_);
  tmp1_.resize(ndstate_ * ndstate_);

  // -- GUI data -- //

  // time step
  gui_timestep_ = this->model->opt.timestep;

  // integrator
  gui_integrator_ = this->model->opt.integrator;

  // process noise
  gui_process_noise_.resize(ndstate_);

  // sensor noise
  gui_sensor_noise_.resize(nsensordata_);
}

// reset memory
void Unscented::Reset(const mjData* data) {
  // dimension
  int nq = model->nq, nv = model->nv, na = model->na;

  if (data) {
    // state
    mju_copy(state.data(), data->qpos, nq);
    mju_copy(state.data() + nq, data->qvel, nv);
    mju_copy(state.data() + nq + nv, data->act, na);
    time = data->time;
  } else {
    // set home keyframe
    int home_id = mj_name2id(model, mjOBJ_KEY, "home");
    if (home_id >= 0) mj_resetDataKeyframe(model, data_, home_id);

    // state
    mju_copy(state.data(), data_->qpos, nq);
    mju_copy(state.data() + nq, data_->qvel, nv);
    mju_copy(state.data() + nq + nv, data_->act, na);
    time = data_->time;
  }

  // covariance
  mju_eye(covariance.data(), ndstate_);
  double covariance_scl =
      GetNumberOrDefault(1.0e-4, model, "estimator_covariance_initial_scale");
  mju_scl(covariance.data(), covariance.data(), covariance_scl,
          ndstate_ * ndstate_);

  // process noise
  double noise_process_scl =
      GetNumberOrDefault(1.0e-4, model, "estimator_process_noise_scale");
  std::fill(noise_process.begin(), noise_process.end(), noise_process_scl);

  // sensor noise
  double noise_sensor_scl =
      GetNumberOrDefault(1.0e-4, model, "estimator_sensor_noise_scale");
  std::fill(noise_sensor.begin(), noise_sensor.end(), noise_sensor_scl);

  // sigma points
  std::fill(sigma_.begin(), sigma_.end(), 0.0);

  // states
  std::fill(states_.begin(), states_.end(), 0.0);

  // sensors
  std::fill(sensors_.begin(), sensors_.end(), 0.0);

  // state mean
  std::fill(state_mean_.begin(), state_mean_.end(), 0.0);

  // sensor mean
  std::fill(sensor_mean_.begin(), sensor_mean_.end(), 0.0);

  // covariance factor
  std::fill(covariance_factor_.begin(), covariance_factor_.end(), 0.0);

  // factor column
  std::fill(factor_column_.begin(), factor_column_.end(), 0.0);

  // state difference
  std::fill(state_difference_.begin(), state_difference_.end(), 0.0);

  // sensor difference
  std::fill(sensor_difference_.begin(), sensor_difference_.end(), 0.0);

  // covariance sensor
  std::fill(covariance_sensor_.begin(), covariance_sensor_.end(), 0.0);

  // covariance state sensor
  std::fill(covariance_state_sensor_.begin(), covariance_state_sensor_.end(),
            0.0);

  // covariance state state
  std::fill(covariance_state_state_.begin(), covariance_state_state_.end(),
            0.0);

  // sensor difference outer product
  std::fill(sensor_difference_outer_product_.begin(),
            sensor_difference_outer_product_.end(), 0.0);

  // state sensor difference outer product
  std::fill(state_sensor_difference_outer_product_.begin(),
            state_sensor_difference_outer_product_.end(), 0.0);

  // state state difference outer product
  std::fill(state_state_difference_outer_product_.begin(),
            state_state_difference_outer_product_.end(), 0.0);

  // covariance sensor factor
  std::fill(covariance_sensor_factor_.begin(), covariance_sensor_factor_.end(),
            0.0);

  // sensor error
  mju_zero(sensor_error_.data(), nsensordata_);

  // correction
  mju_zero(correction_.data(), ndstate_);

  // timer
  timer_update_ = 0.0;

  // scratch
  std::fill(tmp0_.begin(), tmp0_.end(), 0.0);
  std::fill(tmp1_.begin(), tmp1_.end(), 0.0);

  // -- GUI data -- //

  // time step
  gui_timestep_ = model->opt.timestep;

  // integrator
  gui_integrator_ = model->opt.integrator;

  // process noise
  std::fill(gui_process_noise_.begin(), gui_process_noise_.end(), noise_process_scl);

  // sensor noise
  std::fill(gui_sensor_noise_.begin(), gui_sensor_noise_.end(), noise_sensor_scl);
}

// compute sigma points
void Unscented::SigmaPoints() {
  // dimensions
  int nq = model->nq, nv = model->nv, na = model->na;

  // factorize covariance
  mju_copy(covariance_factor_.data(), covariance.data(), ndstate_ * ndstate_);
  int rank = mju_cholFactor(covariance_factor_.data(), ndstate_, 0.0);

  // check failure
  if (rank < ndstate_) {
    // TODO(taylor): remove and return status
    mju_error("covariance factorization failure: (%i / %i)\n", rank, ndstate_);
  }

  // -- loop over points -- //

  // nominal
  mju_copy(sigma_.data() + (nsigma_ - 1) * nstate_, state.data(), nstate_);

  // unpack
  double* column = factor_column_.data();

  // loop over sigma points
  // TODO(taylor): thread?
  for (int i = 0; i < ndstate_; i++) {
    // zero column memory
    mju_zero(column, ndstate_);

    // column elements
    for (int j = i; j < ndstate_; j++) {
      column[j] = covariance_factor_[j * ndstate_ + i];
    }

    // scale
    mju_scl(column, column, sigma_step, ndstate_);

    // -- (+) step -- //
    double* sigma_plus = sigma_.data() + i * nstate_;
    mju_copy(sigma_plus, state.data(), nstate_);

    // qpos
    mj_integratePos(model, sigma_plus, column, 1.0);

    // qvel & qact
    mju_addTo(sigma_plus + nq, column + nv, nv + na);

    // -- (-) step -- //
    double* sigma_minus = sigma_.data() + (i + ndstate_) * nstate_;
    mju_copy(sigma_minus, state.data(), nstate_);

    // qpos
    mj_integratePos(model, sigma_minus, column, -1.0);

    // qvel
    mju_subFrom(sigma_minus + nq, column + nv, nv + na);
  }
}

// evaluate sigma points
// TODO(taylor): thread?
void Unscented::EvaluateSigmaPoints() {
  // dimensions
  int nq = model->nq, nv = model->nv, na = model->na;

  // zero memory
  mju_zero(state_mean_.data(), nstate_);
  mju_zero(sensor_mean_.data(), nsensordata_);

  // time cache
  double time_cache = data_->time;

  // loop over sigma points
  for (int i = 0; i < nsigma_; i++) {
    // set state
    double* sigma = sigma_.data() + i * nstate_;
    mju_copy(data_->qpos, sigma, nq);
    mju_copy(data_->qvel, sigma + nq, nv);
    mju_copy(data_->act, sigma + nq + nv, na);
    data_->time = time_cache;

    // step
    mj_step(model, data_);

    // get state
    double* s = states_.data() + i * nstate_;
    mju_copy(s, data_->qpos, nq);
    mju_copy(s + nq, data_->qvel, nv);
    mju_copy(s + nq + nv, data_->act, na);

    // get sensor
    double* y = sensors_.data() + i * nsensordata_;
    mju_copy(y, data_->sensordata + sensor_start_index_, nsensordata_);

    // update means
    double weight = (i == nsigma_ - 1 ? weight_mean0 : weight_sigma);
    mju_addToScl(state_mean_.data(), s, weight, nstate_);
    mju_addToScl(sensor_mean_.data(), y, weight, nsensordata_);
  }

  // compute correct quaternion means
  QuaternionMeans();
}

// compute sigma point differences
void Unscented::SigmaPointDifferences() {
  // dimensions
  int nq = model->nq, nv = model->nv, na = model->na;

  // unpack means
  double* sm = state_mean_.data();
  double* ym = sensor_mean_.data();

  // loop over sigma points
  for (int i = 0; i < nsigma_; i++) {
    // -- state difference -- //
    double* ds = state_difference_.data() + i * ndstate_;
    double* si = states_.data() + i * nstate_;

    // qpos
    mj_differentiatePos(model, ds, 1.0, sm, si);

    // qvel + act
    mju_sub(ds + nv, si + nq, sm + nq, nv + na);

    // sensor difference
    double* dy = sensor_difference_.data() + i * nsensordata_;
    double* yi = sensors_.data() + i * nsensordata_;
    mju_sub(dy, yi, ym, nsensordata_);
  }
}

// compute sigma covariances
void Unscented::SigmaCovariances() {
  // unpack
  double* dydy = sensor_difference_outer_product_.data();
  double* dsdy = state_sensor_difference_outer_product_.data();
  double* dsds = state_state_difference_outer_product_.data();

  double* cov_yy = covariance_sensor_.data();
  double* cov_sy = covariance_state_sensor_.data();
  double* cov_ss = covariance_state_state_.data();

  // zero memory
  mju_zero(cov_yy, nsensordata_ * nsensordata_);
  mju_zero(cov_sy, ndstate_ * nsensordata_);
  mju_zero(cov_ss, ndstate_ * ndstate_);

  // -- set noise -- //

  // sensor
  for (int i = 0; i < nsensordata_; i++) {
    cov_yy[nsensordata_ * i + i] = noise_sensor[i];
  }

  // process
  for (int i = 0; i < ndstate_; i++) {
    cov_ss[ndstate_ * i + i] = noise_process[i];
  }

  // loop over sigma points
  for (int i = 0; i < nsigma_; i++) {
    // unpack
    double* dy = sensor_difference_.data() + i * nsensordata_;
    double* ds = state_difference_.data() + i * ndstate_;

    // sensor difference outer product
    mju_mulMatMatT(dydy, dy, dy, nsensordata_, 1, nsensordata_);

    // state sensor difference outer product
    mju_mulMatMatT(dsdy, ds, dy, ndstate_, 1, nsensordata_);

    // state state difference outer product
    mju_mulMatMatT(dsds, ds, ds, ndstate_, 1, ndstate_);

    // -- update -- //

    // weight
    double weight = (i == nsigma_ - 1 ? weight_covariance0 : weight_sigma);

    // covariance sensor
    mju_addScl(cov_yy, cov_yy, dydy, weight, nsensordata_ * nsensordata_);

    // covariance state sensor
    mju_addScl(cov_sy, cov_sy, dsdy, weight, ndstate_ * nsensordata_);

    // covariance state state
    mju_addScl(cov_ss, cov_ss, dsds, weight, ndstate_ * ndstate_);
  }
}

// unscented filter update
void Unscented::Update(const double* ctrl, const double* sensor, int mode) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // time cache
  double time_cache = data_->time;

  // dimensions
  int nq = model->nq, nv = model->nv, na = model->na;

  // set ctrl
  mju_copy(data_->ctrl, ctrl, model->nu);

  // compute sigma points
  SigmaPoints();

  // evaluate sigma points
  EvaluateSigmaPoints();

  // compute sigma point difference
  SigmaPointDifferences();

  // compute sigma covariances
  SigmaCovariances();

  // factorize covariance sensor
  double* factor = covariance_sensor_factor_.data();
  mju_copy(factor, covariance_sensor_.data(), nsensordata_ * nsensordata_);
  int rank = mju_cholFactor(factor, nsensordata_, 0.0);

  // check failure
  if (rank < nsensordata_) {
    // TODO(taylor): remove and return status
    mju_error("covariance sensor factorization failure (%i / %i)\n", rank,
              nsensordata_);
  }

  // -- correction -- //

  // sensor error
  mju_sub(sensor_error_.data(), sensor + sensor_start_index_,
          sensor_mean_.data(), nsensordata_);

  // tmp0 = covariance_sensor \ sensor_error
  mju_cholSolve(tmp0_.data(), factor, sensor_error_.data(), nsensordata_);

  // correction = covariance_state_sensor * covariance_sensor \ sensor_error =
  // covariance_state_sensor * tmp0
  mju_mulMatVec(correction_.data(), covariance_state_sensor_.data(),
                tmp0_.data(), ndstate_, nsensordata_);

  // -- state update -- //

  // copy state
  mju_copy(state.data(), state_mean_.data(), nstate_);

  // qpos
  mj_integratePos(model, state.data(), correction_.data(), 1.0);

  // qvel + act
  mju_addTo(state.data() + nq, correction_.data() + nv, nv + na);

  // -- covariance update -- //

  mju_copy(covariance.data(), covariance_state_state_.data(),
           ndstate_ * ndstate_);

  // tmp0 = covariance_sensor^-1 covariance_state_sensor'
  for (int i = 0; i < ndstate_; i++) {
    mju_cholSolve(tmp0_.data() + nsensordata_ * i, factor,
                  covariance_state_sensor_.data() + nsensordata_ * i,
                  nsensordata_);
  }

  // tmp1 = covariance_state_sensor * (covariance_sensor)^-1
  // covariance_state_sensor' = covariance_state_sensor * tmp0'
  mju_mulMatMatT(tmp1_.data(), covariance_state_sensor_.data(), tmp0_.data(),
                 ndstate_, nsensordata_, ndstate_);

  // covariance -= tmp1
  mju_subFrom(covariance.data(), tmp1_.data(), ndstate_ * ndstate_);

  // symmetrize
  mju_symmetrize(covariance.data(), covariance.data(), ndstate_);

  // update time
  time = time_cache + model->opt.timestep;

  // stop timer (ms)
  timer_update_ = 1.0e-3 * GetDuration(start);
}

// quaternion means
// "Averaging Quaternions"
void Unscented::QuaternionMeans() {
  // K matrix
  double K[16];

  // outer product
  double Q[16];

  // loop over joints
  for (int i = 0; i < model->njnt; i++) {
    // joint type
    int jnt_type = model->jnt_type[i];

    // free or ball joint
    if (jnt_type == mjJNT_FREE || jnt_type == mjJNT_BALL) {
      // qpos address
      int qpos_adr = model->jnt_qposadr[i];

      // shift to quaternion address for free joint
      if (jnt_type == mjJNT_FREE) qpos_adr += 3;

      // zero K memory
      mju_zero(K, 16);

      // loop over states
      for (int j = 0; j < nsigma_; j++) {
        // get quaternion
        double* quat = states_.data() + j * nstate_ + qpos_adr;

        // compute outer product
        mju_mulMatMatT(Q, quat, quat, 4, 1, 4);

        // add outerproduct to K
        mju_addToScl(
            K, Q, 4.0 * (j == nsigma_ - 1 ? weight_covariance0 : weight_sigma),
            16);
      }

      // K = K - total_weight * I
      double total_weight = weight_covariance0 + (nsigma_ - 1) * weight_sigma;
      K[0] -= total_weight;
      K[5] -= total_weight;
      K[10] -= total_weight;
      K[15] -= total_weight;

      // update state mean quaternion with principal eigenvector
      PrincipalEigenVector4(state_mean_.data() + qpos_adr, K, 12.0);
    }
  }
}

// estimator-specific GUI elements
void Unscented::GUI(mjUI& ui) {
  // ----- estimator ------ //
  mjuiDef defEstimator[] = {
      {mjITEM_SECTION, "Estimator", 1, nullptr,
       "AP"},  // needs new section to satisfy mjMAXUIITEM
      {mjITEM_BUTTON, "Reset", 2, nullptr, ""},
      {mjITEM_SLIDERNUM, "Timestep", 2, &gui_timestep_, "1.0e-3 0.1"},
      {mjITEM_SELECT, "Integrator", 2, &gui_integrator_,
       "Euler\nRK4\nImplicit\nFastImplicit"},
      {mjITEM_END}};

  // add estimator
  mjui_add(&ui, defEstimator);

  // -- process noise -- //
  int nv = model->nv;
  int process_noise_shift = 0;
  mjuiDef defProcessNoise[kMaxProcessNoise + 2];

  // separator
  defProcessNoise[0] = {mjITEM_SEPARATOR, "Process Noise Std.", 1};
  process_noise_shift++;

  // add UI elements
  for (int i = 0; i < DimensionProcess(); i++) {
    // element
    defProcessNoise[process_noise_shift] = {
        mjITEM_SLIDERNUM, "", 2, gui_process_noise_.data() + i, "1.0e-8 0.01"};

    // set name
    mju::strcpy_arr(defProcessNoise[process_noise_shift].name, "");

    // shift
    process_noise_shift++;
  }

  // name UI elements
  int jnt_shift = 1;
  std::string jnt_name_pos;
  std::string jnt_name_vel;

  // loop over joints
  for (int i = 0; i < model->njnt; i++) {
    int name_jntadr = model->name_jntadr[i];
    std::string jnt_name(model->names + name_jntadr);

    // get joint type
    int jnt_type = model->jnt_type[i];

    // free
    switch (jnt_type) {
      case mjJNT_FREE:
        // position
        jnt_name_pos = jnt_name + " (pos 0)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 0].name,
                        jnt_name_pos.c_str());

        jnt_name_pos = jnt_name + " (pos 1)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 1].name,
                        jnt_name_pos.c_str());

        jnt_name_pos = jnt_name + " (pos 2)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 2].name,
                        jnt_name_pos.c_str());

        jnt_name_pos = jnt_name + " (pos 3)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 3].name,
                        jnt_name_pos.c_str());

        jnt_name_pos = jnt_name + " (pos 4)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 4].name,
                        jnt_name_pos.c_str());

        jnt_name_pos = jnt_name + " (pos 5)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 5].name,
                        jnt_name_pos.c_str());

        // velocity
        jnt_name_vel = jnt_name + " (vel 0)";
        mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 0].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (vel 1)";
        mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 1].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (vel 2)";
        mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 2].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (vel 3)";
        mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 3].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (vel 4)";
        mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 4].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (vel 5)";
        mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 5].name,
                        jnt_name_vel.c_str());

        // shift
        jnt_shift += 6;
        break;
      case mjJNT_BALL:
        // position
        jnt_name_pos = jnt_name + " (pos 0)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 0].name,
                        jnt_name_pos.c_str());

        jnt_name_pos = jnt_name + " (pos 1)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 1].name,
                        jnt_name_pos.c_str());

        jnt_name_pos = jnt_name + " (pos 2)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 2].name,
                        jnt_name_pos.c_str());

        // velocity
        jnt_name_vel = jnt_name + " (vel 0)";
        mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 0].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (vel 1)";
        mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 1].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (vel 2)";
        mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 2].name,
                        jnt_name_vel.c_str());

        // shift
        jnt_shift += 3;
        break;
      case mjJNT_HINGE:
        // position
        jnt_name_pos = jnt_name + " (pos)";
        mju::strcpy_arr(defProcessNoise[jnt_shift].name, jnt_name_pos.c_str());

        // velocity
        jnt_name_vel = jnt_name + " (vel)";
        mju::strcpy_arr(defProcessNoise[nv + jnt_shift].name,
                        jnt_name_vel.c_str());

        // shift
        jnt_shift++;
        break;
      case mjJNT_SLIDE:
        // position
        jnt_name_pos = jnt_name + " (pos)";
        mju::strcpy_arr(defProcessNoise[jnt_shift].name, jnt_name_pos.c_str());

        // velocity
        jnt_name_vel = jnt_name + " (vel)";
        mju::strcpy_arr(defProcessNoise[nv + jnt_shift].name,
                        jnt_name_vel.c_str());

        // shift
        jnt_shift++;
        break;
    }
  }

  // loop over act
  std::string act_str;
  for (int i = 0; i < model->na; i++) {
    act_str = "act (" + std::to_string(i) + ")";
    mju::strcpy_arr(defProcessNoise[nv + jnt_shift + i].name, act_str.c_str());
  }

  // end
  defProcessNoise[process_noise_shift] = {mjITEM_END};

  // add process noise
  mjui_add(&ui, defProcessNoise);

  // -- sensor noise -- //
  int sensor_noise_shift = 0;
  mjuiDef defSensorNoise[kMaxSensorNoise + 2];

  // separator
  defSensorNoise[0] = {mjITEM_SEPARATOR, "Sensor Noise Std.", 1};
  sensor_noise_shift++;

  // loop over sensors
  std::string sensor_str;
  for (int i = 0; i < nsensor_; i++) {
    std::string name_sensor(model->names +
                            model->name_sensoradr[sensor_start_ + i]);
    int dim_sensor = model->sensor_dim[sensor_start_ + i];

    // loop over sensor elements
    for (int j = 0; j < dim_sensor; j++) {
      // element
      defSensorNoise[sensor_noise_shift] = {
          mjITEM_SLIDERNUM, "", 2,
          gui_sensor_noise_.data() + sensor_noise_shift - 1, "1.0e-8 0.01"};

      // sensor name
      sensor_str = name_sensor;

      // add element index
      if (dim_sensor > 1) {
        sensor_str += " (" + std::to_string(j) + ")";
      }

      // set sensor name
      mju::strcpy_arr(defSensorNoise[sensor_noise_shift].name,
                      sensor_str.c_str());

      // shift
      sensor_noise_shift++;
    }
  }

  // end
  defSensorNoise[sensor_noise_shift] = {mjITEM_END};

  // add sensor noise
  mjui_add(&ui, defSensorNoise);
}

// set GUI data
void Unscented::SetGUIData() {
  mju_copy(noise_process.data(), gui_process_noise_.data(), DimensionProcess());
  mju_copy(noise_sensor.data(), gui_sensor_noise_.data(), DimensionSensor());
  model->opt.timestep = gui_timestep_;
  model->opt.integrator = gui_integrator_;
}

// estimator-specific plots
void Unscented::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                      int planner_shift, int timer_shift, int planning,
                      int* shift) {
  // Unscented info
  double estimator_bounds[2] = {-6, 6};

  // covariance trace
  double trace = Trace(covariance.data(), DimensionProcess());
  mjpc::PlotUpdateData(fig_planner, estimator_bounds,
                       fig_planner->linedata[planner_shift + 0][0] + 1,
                       mju_log10(trace), 100, planner_shift + 0, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[planner_shift + 0], "Covariance Trace");

  // Unscented timers
  double timer_bounds[2] = {0.0, 1.0};

  // update
  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[timer_shift + 0][0] + 1, TimerUpdate(),
                 100, timer_shift + 0, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[timer_shift + 0], "Update");
}

}  // namespace mjpc
