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

#include "mjpc/estimators/batch.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <string>

#include <mujoco/mujoco.h>

#include "mjpc/array_safety.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/estimators/gui.h"
#include "mjpc/norm.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;

// initialize batch estimator
void Batch::Initialize(const mjModel* model) {
  // model
  if (this->model) mj_deleteModel(this->model);
  this->model = mj_copyModel(nullptr, model);

  // data
  data_.clear();
  for (int i = 0; i < max_history_; i++) {
    data_.push_back(MakeUniqueMjData(mj_makeData(model)));
  }

  // timestep
  this->model->opt.timestep = GetNumberOrDefault(this->model->opt.timestep,
                                                 model, "estimator_timestep");

  // length of configuration trajectory
  configuration_length_ =
      GetNumberOrDefault(3, model, "batch_configuration_length");

  // check configuration length
  if (configuration_length_ > max_history_) {
    mju_error("configuration_length > max_history: increase max history\n");
  }

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

  // dimension
  int nq = model->nq, nv = model->nv, na = model->na;
  int ntotal = nv * max_history_;
  int nsensortotal = nsensordata_ * max_history_;

  // state dimensions
  nstate_ = nq + nv + na;
  ndstate_ = 2 * nv + na;

  // state
  state.resize(nstate_);

  // covariance
  covariance.resize(ndstate_ * ndstate_);

  // process noise
  noise_process.resize(ndstate_);

  // sensor noise
  noise_sensor.resize(nsensordata_);  // overallocate

  // -- trajectories -- //
  configuration.Initialize(nq, configuration_length_);
  velocity.Initialize(nv, configuration_length_);
  acceleration.Initialize(nv, configuration_length_);
  act.Initialize(na, configuration_length_);
  times.Initialize(1, configuration_length_);

  // ctrl
  ctrl.Initialize(model->nu, configuration_length_);

  // prior
  configuration_previous.Initialize(nq, configuration_length_);

  // sensor
  sensor_measurement.Initialize(nsensordata_, configuration_length_);
  sensor_prediction.Initialize(nsensordata_, configuration_length_);
  sensor_mask.Initialize(nsensor_, configuration_length_);

  // force
  force_measurement.Initialize(nv, configuration_length_);
  force_prediction.Initialize(nv, configuration_length_);

  // residual
  residual_prior_.resize(ntotal);
  residual_sensor_.resize(nsensortotal);
  residual_force_.resize(ntotal);

  // Jacobian
  jacobian_prior_.resize(settings.assemble_prior_jacobian * ntotal * ntotal);
  jacobian_sensor_.resize(settings.assemble_sensor_jacobian * nsensortotal *
                          ntotal);
  jacobian_force_.resize(settings.assemble_force_jacobian * ntotal * ntotal);

  // prior Jacobian block
  block_prior_current_configuration_.Initialize(nv * nv, configuration_length_);

  // sensor Jacobian blocks
  block_sensor_configuration_.Initialize(model->nsensordata * nv,
                                         configuration_length_);
  block_sensor_velocity_.Initialize(model->nsensordata * nv,
                                    configuration_length_);
  block_sensor_acceleration_.Initialize(model->nsensordata * nv,
                                        configuration_length_);
  block_sensor_configurationT_.Initialize(model->nsensordata * nv,
                                          configuration_length_);
  block_sensor_velocityT_.Initialize(model->nsensordata * nv,
                                     configuration_length_);
  block_sensor_accelerationT_.Initialize(model->nsensordata * nv,
                                         configuration_length_);

  block_sensor_previous_configuration_.Initialize(nsensordata_ * nv,
                                                  configuration_length_);
  block_sensor_current_configuration_.Initialize(nsensordata_ * nv,
                                                 configuration_length_);
  block_sensor_next_configuration_.Initialize(nsensordata_ * nv,
                                              configuration_length_);
  block_sensor_configurations_.Initialize(nsensordata_ * 3 * nv,
                                          configuration_length_);

  block_sensor_scratch_.Initialize(
      std::max(nv, nsensordata_) * std::max(nv, nsensordata_),
      configuration_length_);

  // force Jacobian blocks
  block_force_configuration_.Initialize(nv * nv, configuration_length_);
  block_force_velocity_.Initialize(nv * nv, configuration_length_);
  block_force_acceleration_.Initialize(nv * nv, configuration_length_);

  block_force_previous_configuration_.Initialize(nv * nv,
                                                 configuration_length_);
  block_force_current_configuration_.Initialize(nv * nv, configuration_length_);
  block_force_next_configuration_.Initialize(nv * nv, configuration_length_);
  block_force_configurations_.Initialize(nv * 3 * nv, configuration_length_);

  block_force_scratch_.Initialize(nv * nv, configuration_length_);

  // velocity Jacobian blocks
  block_velocity_previous_configuration_.Initialize(nv * nv,
                                                    configuration_length_);
  block_velocity_current_configuration_.Initialize(nv * nv,
                                                   configuration_length_);

  // acceleration Jacobian blocks
  block_acceleration_previous_configuration_.Initialize(nv * nv,
                                                        configuration_length_);
  block_acceleration_current_configuration_.Initialize(nv * nv,
                                                       configuration_length_);
  block_acceleration_next_configuration_.Initialize(nv * nv,
                                                    configuration_length_);

  // cost gradient
  cost_gradient_prior_.resize(ntotal);
  cost_gradient_sensor_.resize(ntotal);
  cost_gradient_force_.resize(ntotal);
  cost_gradient_.resize(ntotal);

  // cost Hessian
  cost_hessian_prior_band_.resize(ntotal * (3 * nv));
  cost_hessian_sensor_band_.resize(ntotal * (3 * nv));
  cost_hessian_force_band_.resize(ntotal * (3 * nv));
  cost_hessian_.resize(settings.filter * ntotal * ntotal);
  cost_hessian_band_.resize(ntotal * (3 * nv));
  cost_hessian_band_factor_.resize(ntotal * (3 * nv));

  // prior weights
  scale_prior = GetNumberOrDefault(1.0, model, "batch_scale_prior");
  weight_prior_.resize(settings.prior_flag * ntotal * ntotal);
  weight_prior_band_.resize(settings.prior_flag * ntotal * (3 * nv));

  // cost norms
  norm_type_sensor.resize(nsensor_);

  // TODO(taylor): method for xml to initial norm
  for (int i = 0; i < nsensor_; i++) {
    norm_type_sensor[i] =
        (NormType)GetNumberOrDefault(0, model, "batch_norm_sensor");

    // add support by parsing norm parameters
    if (norm_type_sensor[i] != 0) {
      mju_error("norm type not supported\n");
    }
  }

  // cost norm parameters
  norm_parameters_sensor.resize(nsensor_ * kMaxNormParameters);

  // TODO(taylor): initialize norm parameters from xml
  std::fill(norm_parameters_sensor.begin(), norm_parameters_sensor.end(), 0.0);

  // norm
  norm_sensor_.resize(nsensor_ * max_history_);
  norm_force_.resize(ntotal);

  // norm gradient
  norm_gradient_sensor_.resize(nsensortotal);
  norm_gradient_force_.resize(ntotal);

  // norm Hessian
  norm_hessian_sensor_.resize(settings.assemble_sensor_norm_hessian *
                              nsensortotal * nsensortotal);
  norm_hessian_force_.resize(settings.assemble_force_norm_hessian * ntotal *
                             ntotal);

  norm_blocks_sensor_.resize(nsensordata_ * nsensortotal);
  norm_blocks_force_.resize(nv * ntotal);

  // scratch
  scratch_prior_.resize(ntotal + 12 * nv * nv);
  scratch_sensor_.resize(3 * nv + nsensordata_ * 3 * nv + 9 * nv * nv);
  scratch_force_.resize(12 * nv * nv);
  scratch_expected_.resize(ntotal);

  // copy
  configuration_copy_.Initialize(nq, configuration_length_);

  // search direction
  search_direction_.resize(ntotal);

  // conditioned matrix
  mat00_.resize(settings.filter * ntotal * ntotal);
  mat10_.resize(settings.filter * ntotal * ntotal);
  mat11_.resize(settings.filter * ntotal * ntotal);
  condmat_.resize(settings.filter * ntotal * ntotal);
  scratch0_condmat_.resize(settings.filter * ntotal * ntotal);
  scratch1_condmat_.resize(settings.filter * ntotal * ntotal);

  // regularization
  regularization_ = settings.regularization_initial;

  // search type
  settings.search_type = (SearchType)GetNumberOrDefault(
      static_cast<int>(settings.search_type), model, "batch_search_type");

  // timer
  timer_.prior_step.resize(max_history_);
  timer_.sensor_step.resize(max_history_);
  timer_.force_step.resize(max_history_);

  // status
  gradient_norm_ = 0.0;
  search_direction_norm_ = 0.0;
  step_size_ = 1.0;
  solve_status_ = kUnsolved;

  // -- trajectory cache -- //
  configuration_cache_.Initialize(nq, configuration_length_);
  velocity_cache_.Initialize(nv, configuration_length_);
  acceleration_cache_.Initialize(nv, configuration_length_);
  act_cache_.Initialize(na, configuration_length_);
  times_cache_.Initialize(1, configuration_length_);

  // ctrl
  ctrl_cache_.Initialize(model->nu, configuration_length_);

  // prior
  configuration_previous_cache_.Initialize(nq, configuration_length_);

  // sensor
  sensor_measurement_cache_.Initialize(nsensordata_, configuration_length_);
  sensor_prediction_cache_.Initialize(nsensordata_, configuration_length_);
  sensor_mask_cache_.Initialize(nsensor_, configuration_length_);

  // force
  force_measurement_cache_.Initialize(nv, configuration_length_);
  force_prediction_cache_.Initialize(nv, configuration_length_);
}

// reset memory
void Batch::Reset(const mjData* data) {
  // dimension
  int nq = model->nq, nv = model->nv, na = model->na;

  // data
  mjData* d = data_[0].get();

  if (data) {
    // copy input data
    mj_copyData(d, model, data);
  } else {
    // set home keyframe
    int home_id = mj_name2id(model, mjOBJ_KEY, "home");
    if (home_id >= 0) mj_resetDataKeyframe(model, d, home_id);
  }

  // forward evaluation
  mj_forward(model, d);

  // state
  mju_copy(state.data(), d->qpos, nq);
  mju_copy(state.data() + nq, d->qvel, nv);
  mju_copy(state.data() + nq + nv, d->act, na);
  time = d->time;

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

  // scale prior
  scale_prior = GetNumberOrDefault(1.0e-1, model, "batch_scale_prior");

  // trajectories
  configuration.Reset();
  velocity.Reset();
  acceleration.Reset();
  act.Reset();
  times.Reset();
  ctrl.Reset();

  // prior
  configuration_previous.Reset();

  // sensor
  sensor_measurement.Reset();
  sensor_prediction.Reset();

  // sensor mask
  sensor_mask.Reset();
  for (int i = 0; i < nsensor_ * max_history_; i++) {
    sensor_mask.Data()[i] = 1;  // sensor on
  }

  // force
  force_measurement.Reset();
  force_prediction.Reset();

  // residual
  std::fill(residual_prior_.begin(), residual_prior_.end(), 0.0);
  std::fill(residual_sensor_.begin(), residual_sensor_.end(), 0.0);
  std::fill(residual_force_.begin(), residual_force_.end(), 0.0);

  // Jacobian
  std::fill(jacobian_prior_.begin(), jacobian_prior_.end(), 0.0);
  std::fill(jacobian_sensor_.begin(), jacobian_sensor_.end(), 0.0);
  std::fill(jacobian_force_.begin(), jacobian_force_.end(), 0.0);

  // prior Jacobian block
  block_prior_current_configuration_.Reset();

  // sensor Jacobian blocks
  block_sensor_configuration_.Reset();
  block_sensor_velocity_.Reset();
  block_sensor_acceleration_.Reset();
  block_sensor_configurationT_.Reset();
  block_sensor_velocityT_.Reset();
  block_sensor_accelerationT_.Reset();

  block_sensor_previous_configuration_.Reset();
  block_sensor_current_configuration_.Reset();
  block_sensor_next_configuration_.Reset();
  block_sensor_configurations_.Reset();

  block_sensor_scratch_.Reset();

  // force Jacobian blocks
  block_force_configuration_.Reset();
  block_force_velocity_.Reset();
  block_force_acceleration_.Reset();

  block_force_previous_configuration_.Reset();
  block_force_current_configuration_.Reset();
  block_force_next_configuration_.Reset();
  block_force_configurations_.Reset();

  block_force_scratch_.Reset();

  // velocity Jacobian blocks
  block_velocity_previous_configuration_.Reset();
  block_velocity_current_configuration_.Reset();

  // acceleration Jacobian blocks
  block_acceleration_previous_configuration_.Reset();
  block_acceleration_current_configuration_.Reset();
  block_acceleration_next_configuration_.Reset();

  // cost
  cost_prior_ = 0.0;
  cost_sensor_ = 0.0;
  cost_force_ = 0.0;
  cost_ = 0.0;
  cost_initial_ = 0.0;
  cost_previous_ = 1.0e32;

  // cost gradient
  std::fill(cost_gradient_prior_.begin(), cost_gradient_prior_.end(), 0.0);
  std::fill(cost_gradient_sensor_.begin(), cost_gradient_sensor_.end(), 0.0);
  std::fill(cost_gradient_force_.begin(), cost_gradient_force_.end(), 0.0);
  std::fill(cost_gradient_.begin(), cost_gradient_.end(), 0.0);

  // cost Hessian
  std::fill(cost_hessian_prior_band_.begin(), cost_hessian_prior_band_.end(),
            0.0);
  std::fill(cost_hessian_sensor_band_.begin(), cost_hessian_sensor_band_.end(),
            0.0);
  std::fill(cost_hessian_force_band_.begin(), cost_hessian_force_band_.end(),
            0.0);
  std::fill(cost_hessian_.begin(), cost_hessian_.end(), 0.0);
  std::fill(cost_hessian_band_.begin(), cost_hessian_band_.end(), 0.0);
  std::fill(cost_hessian_band_factor_.begin(), cost_hessian_band_factor_.end(),
            0.0);

  // weight
  std::fill(weight_prior_.begin(), weight_prior_.end(), 0.0);
  std::fill(weight_prior_band_.begin(), weight_prior_band_.end(), 0.0);

  // norm
  std::fill(norm_sensor_.begin(), norm_sensor_.end(), 0.0);
  std::fill(norm_force_.begin(), norm_force_.end(), 0.0);

  // norm gradient
  std::fill(norm_gradient_sensor_.begin(), norm_gradient_sensor_.end(), 0.0);
  std::fill(norm_gradient_force_.begin(), norm_gradient_force_.end(), 0.0);

  // norm Hessian
  std::fill(norm_hessian_sensor_.begin(), norm_hessian_sensor_.end(), 0.0);
  std::fill(norm_hessian_force_.begin(), norm_hessian_force_.end(), 0.0);

  std::fill(norm_blocks_sensor_.begin(), norm_blocks_sensor_.end(), 0.0);
  std::fill(norm_blocks_force_.begin(), norm_blocks_force_.end(), 0.0);

  // scratch
  std::fill(scratch_prior_.begin(), scratch_prior_.end(), 0.0);
  std::fill(scratch_sensor_.begin(), scratch_sensor_.end(), 0.0);
  std::fill(scratch_force_.begin(), scratch_force_.end(), 0.0);
  std::fill(scratch_expected_.begin(), scratch_expected_.end(), 0.0);

  // candidate
  configuration_copy_.Reset();

  // search direction
  std::fill(search_direction_.begin(), search_direction_.end(), 0.0);

  // conditioned matrix
  std::fill(mat00_.begin(), mat00_.end(), 0.0);
  std::fill(mat10_.begin(), mat10_.end(), 0.0);
  std::fill(mat11_.begin(), mat11_.end(), 0.0);
  std::fill(condmat_.begin(), condmat_.end(), 0.0);
  std::fill(scratch0_condmat_.begin(), scratch0_condmat_.end(), 0.0);
  std::fill(scratch1_condmat_.begin(), scratch1_condmat_.end(), 0.0);

  // timer
  std::fill(timer_.prior_step.begin(), timer_.prior_step.end(), 0.0);
  std::fill(timer_.sensor_step.begin(), timer_.sensor_step.end(), 0.0);
  std::fill(timer_.force_step.begin(), timer_.force_step.end(), 0.0);

  // timing
  ResetTimers();

  // status
  iterations_smoother_ = 0;
  iterations_search_ = 0;
  cost_count_ = 0;
  solve_status_ = kUnsolved;

  // initialize filter
  if (settings.filter) InitializeFilter();

  // trajectory cache
  configuration_cache_.Reset();
  velocity_cache_.Reset();
  acceleration_cache_.Reset();
  act_cache_.Reset();
  times_cache_.Reset();
  ctrl_cache_.Reset();

  // prior
  configuration_previous_cache_.Reset();

  // sensor
  sensor_measurement_cache_.Reset();
  sensor_prediction_cache_.Reset();

  // sensor mask
  sensor_mask_cache_.Reset();
  for (int i = 0; i < nsensor_ * configuration_length_; i++) {
    sensor_mask_cache_.Data()[i] = 1;  // sensor on
  }

  // force
  force_measurement_cache_.Reset();
  force_prediction_cache_.Reset();
}

// update
void Batch::Update(const double* ctrl, const double* sensor) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimensions
  int nq = model->nq, nv = model->nv, na = model->na, nu = model->nu;

  // data
  mjData* d = data_[0].get();

  // current time index
  int t = current_time_index_;

  // configurations
  double* q0 = configuration.Get(t - 1);
  double* q1 = configuration.Get(t);

  // -- next qpos -- //

  // set state
  mju_copy(d->qpos, q1, model->nq);
  mj_differentiatePos(model, d->qvel, model->opt.timestep, q0, q1);

  // set ctrl
  mju_copy(d->ctrl, ctrl, nu);

  // forward step
  mj_step(model, d);

  // -- set batch data -- //

  // set next qpos
  configuration.Set(d->qpos, t + 1);
  configuration_previous.Set(d->qpos, t + 1);

  // set next time
  times.Set(&d->time, t + 1);

  // set ctrl
  this->ctrl.Set(ctrl, t);

  // set sensor
  sensor_measurement.Set(sensor + sensor_start_index_, t);

  // set force measurement
  force_measurement.Set(d->qfrc_actuator, t);

  // -- measurement update -- //

  // cache configuration length
  int configuration_length_cache = configuration_length_;

  // set reduced configuration length for optimization
  configuration_length_ = current_time_index_ + 2;
  if (configuration_length_ != configuration_length_cache) {
    ShiftResizeTrajectory(0, configuration_length_);
  }

  // TODO(taylor): preallocate pool
  ThreadPool pool(1);
  Optimize(pool);

  // update state
  mju_copy(state.data(), configuration.Get(t + 1), nq);
  mju_copy(state.data() + nq, velocity.Get(t + 1), nv);
  mju_copy(state.data() + nq + nv, act.Get(t + 1), na);
  time = times.Get(t + 1)[0];

  // -- update prior weights -- //

  // dimension
  int nvar = nv * configuration_length_;
  int nband = 3 * nv;

  // prior weights
  double* weights = weight_prior_.data();

  // recursive update
  if (settings.recursive_prior_update &&
      configuration_length_ == configuration_length_cache) {
    // condition dimension
    int ncondition = nv * (configuration_length_ - 1);

    // band to dense cost Hessian
    mju_band2Dense(cost_hessian_.data(), cost_hessian_band_.data(), nvar, nband,
                   0, 1);

    // compute conditioned matrix
    double* condmat = condmat_.data();
    ConditionMatrix(condmat, cost_hessian_.data(), mat00_.data(), mat10_.data(),
                    mat11_.data(), scratch0_condmat_.data(),
                    scratch1_condmat_.data(), nvar, nv, ncondition);

    // zero memory
    mju_zero(weights, nvar * nvar);

    // set conditioned matrix in prior weights
    SetBlockInMatrix(weights, condmat, 1.0, nvar, nvar, ncondition, ncondition,
                     0, 0);

    // set bottom right to scale_prior * I
    for (int i = ncondition; i < nvar; i++) {
      weights[nvar * i + i] = scale_prior;
    }

    // make block band
    DenseToBlockBand(weights, nvar, nv, 3);
  } else {
    // dimension
    int nvar_new = nvar;
    if (current_time_index_ < configuration_length_ - 2) {
      nvar_new += nv;
    }

    // check same size
    if (nvar != nvar_new) {
      // get previous weights
      double* previous_weights = scratch0_condmat_.data();
      mju_copy(previous_weights, weights, nvar * nvar);

      // set previous in new weights (dimension may have increased)
      mju_zero(weights, nvar_new * nvar_new);
      SetBlockInMatrix(weights, previous_weights, 1.0, nvar_new, nvar_new, nvar,
                       nvar, 0, 0);

      // scale_prior * I
      for (int i = nvar; i < nvar_new; i++) {
        weights[nvar_new * i + i] = scale_prior;
      }

      // make block band
      DenseToBlockBand(weights, nvar_new, nv, 3);
    }
  }

  // restore configuration length
  if (configuration_length_ != configuration_length_cache) {
    ShiftResizeTrajectory(0, configuration_length_cache);
  }
  configuration_length_ = configuration_length_cache;

  // check estimation horizon
  if (current_time_index_ < configuration_length_ - 2) {
    current_time_index_++;
  } else {
    // shift trajectories once estimation horizon is filled
    Shift(1);
  }

  // stop timer
  timer_.update = 1.0e-3 * GetDuration(start);
}

// set configuration length
void Batch::SetConfigurationLength(int length) {
  // check length
  if (length > max_history_) {
    mju_error("length > max_history_\n");
  }

  // set configuration length
  configuration_length_ = std::max(length, kMinBatchHistory);

  // update trajectory lengths
  configuration.SetLength(configuration_length_);
  configuration_copy_.SetLength(configuration_length_);

  velocity.SetLength(configuration_length_);
  acceleration.SetLength(configuration_length_);
  act.SetLength(configuration_length_);
  times.SetLength(configuration_length_);

  ctrl.SetLength(configuration_length_);

  configuration_previous.SetLength(configuration_length_);

  sensor_measurement.SetLength(configuration_length_);
  sensor_prediction.SetLength(configuration_length_);
  sensor_mask.SetLength(configuration_length_);

  force_measurement.SetLength(configuration_length_);
  force_prediction.SetLength(configuration_length_);

  block_prior_current_configuration_.SetLength(configuration_length_);

  block_sensor_configuration_.SetLength(configuration_length_);
  block_sensor_velocity_.SetLength(configuration_length_);
  block_sensor_acceleration_.SetLength(configuration_length_);
  block_sensor_configurationT_.SetLength(configuration_length_);
  block_sensor_velocityT_.SetLength(configuration_length_);
  block_sensor_accelerationT_.SetLength(configuration_length_);

  block_sensor_previous_configuration_.SetLength(configuration_length_);
  block_sensor_current_configuration_.SetLength(configuration_length_);
  block_sensor_next_configuration_.SetLength(configuration_length_);
  block_sensor_configurations_.SetLength(configuration_length_);

  block_sensor_scratch_.SetLength(configuration_length_);

  block_force_configuration_.SetLength(configuration_length_);
  block_force_velocity_.SetLength(configuration_length_);
  block_force_acceleration_.SetLength(configuration_length_);

  block_force_previous_configuration_.SetLength(configuration_length_);
  block_force_current_configuration_.SetLength(configuration_length_);
  block_force_next_configuration_.SetLength(configuration_length_);
  block_force_configurations_.SetLength(configuration_length_);

  block_force_scratch_.SetLength(configuration_length_);

  block_velocity_previous_configuration_.SetLength(configuration_length_);
  block_velocity_current_configuration_.SetLength(configuration_length_);

  block_acceleration_previous_configuration_.SetLength(configuration_length_);
  block_acceleration_current_configuration_.SetLength(configuration_length_);
  block_acceleration_next_configuration_.SetLength(configuration_length_);
}

// shift trajectory heads
void Batch::Shift(int shift) {
  // update trajectory lengths
  configuration.Shift(shift);
  configuration_copy_.Shift(shift);

  velocity.Shift(shift);
  acceleration.Shift(shift);
  act.Shift(shift);
  times.Shift(shift);
  ctrl.Shift(shift);

  configuration_previous.Shift(shift);

  sensor_measurement.Shift(shift);
  sensor_prediction.Shift(shift);
  sensor_mask.Shift(shift);

  force_measurement.Shift(shift);
  force_prediction.Shift(shift);

  // not reusing Jacobian data--no need to shift

  // block_prior_current_configuration_.Shift(shift);

  // block_sensor_configuration_.Shift(shift);
  // block_sensor_velocity_.Shift(shift);
  // block_sensor_acceleration_.Shift(shift);
  // block_sensor_configurationT_.Shift(shift);
  // block_sensor_velocityT_.Shift(shift);
  // block_sensor_accelerationT_.Shift(shift);

  // block_sensor_previous_configuration_.Shift(shift);
  // block_sensor_current_configuration_.Shift(shift);
  // block_sensor_next_configuration_.Shift(shift);
  // block_sensor_configurations_.Shift(shift);

  // block_sensor_scratch_.Shift(shift);

  // block_force_configuration_.Shift(shift);
  // block_force_velocity_.Shift(shift);
  // block_force_acceleration_.Shift(shift);

  // block_force_previous_configuration_.Shift(shift);
  // block_force_current_configuration_.Shift(shift);
  // block_force_next_configuration_.Shift(shift);
  // block_force_configurations_.Shift(shift);

  // block_force_scratch_.Shift(shift);

  // block_velocity_previous_configuration_.Shift(shift);
  // block_velocity_current_configuration_.Shift(shift);

  // block_acceleration_previous_configuration_.Shift(shift);
  // block_acceleration_current_configuration_.Shift(shift);
  // block_acceleration_next_configuration_.Shift(shift);
}

// evaluate configurations
void Batch::ConfigurationEvaluation(ThreadPool& pool) {
  // finite-difference velocities, accelerations
  ConfigurationToVelocityAcceleration();

  // compute sensor and force predictions
  InverseDynamicsPrediction(pool);
}

// configurations derivatives
void Batch::ConfigurationDerivative(ThreadPool& pool) {
  // dimension
  int nvar = model->nv * configuration_length_;
  int nsen = nsensordata_ * (configuration_length_ - 1);
  int nforce = nsensordata_ * (configuration_length_ - 2);

  // operations
  int opprior = settings.prior_flag * configuration_length_;
  int opsensor = settings.sensor_flag * (configuration_length_ - 1);
  int opforce = settings.force_flag * (configuration_length_ - 2);

  // inverse dynamics derivatives
  InverseDynamicsDerivatives(pool);

  // velocity, acceleration derivatives
  VelocityAccelerationDerivatives();

  // -- Jacobians -- //
  auto timer_jacobian_start = std::chrono::steady_clock::now();

  // pool count
  int count_begin = pool.GetCount();

  // individual derivatives
  if (settings.prior_flag) {
    if (settings.assemble_prior_jacobian)
      mju_zero(jacobian_prior_.data(), nvar * nvar);
    JacobianPrior(pool);
  }
  if (settings.sensor_flag) {
    if (settings.assemble_sensor_jacobian)
      mju_zero(jacobian_sensor_.data(), nsen * nvar);
    JacobianSensor(pool);
  }
  if (settings.force_flag) {
    if (settings.assemble_force_jacobian)
      mju_zero(jacobian_force_.data(), nforce * nvar);
    JacobianForce(pool);
  }

  // wait
  pool.WaitCount(count_begin + opprior + opsensor + opforce);

  // reset count
  pool.ResetCount();

  // timers
  timer_.jacobian_prior += mju_sum(timer_.prior_step.data(), opprior);
  timer_.jacobian_sensor += mju_sum(timer_.sensor_step.data(), opsensor);
  timer_.jacobian_force += mju_sum(timer_.force_step.data(), opforce);
  timer_.jacobian_total += GetDuration(timer_jacobian_start);
}

// prior cost
double Batch::CostPrior(double* gradient, double* hessian) {
  // start timer
  auto start_cost = std::chrono::steady_clock::now();

  // residual dimension
  int nv = model->nv;
  int dim = model->nv * configuration_length_;

  // total scaling
  double scale = scale_prior / dim;

  // unpack
  double* r = residual_prior_.data();
  double* tmp = scratch_prior_.data();

  // zero memory
  if (gradient) mju_zero(gradient, dim);
  if (hessian) mju_zero(hessian, dim * (3 * nv));

  // initial cost
  double cost = 0.0;

  // dimensions
  int ntotal = dim;
  int nband = 3 * model->nv;
  int ndense = 0;

  // dense2band
  mju_dense2Band(weight_prior_band_.data(), weight_prior_.data(), ntotal, nband,
                 ndense);

  // compute cost
  if (!cost_skip_) {
    // residual
    ResidualPrior();

    // multiply: tmp = P * r
    mju_bandMulMatVec(tmp, weight_prior_band_.data(), r, ntotal, nband, ndense,
                      1, true);

    // weighted quadratic: 0.5 * w * r' * tmp
    cost = 0.5 * scale * mju_dot(r, tmp, dim);

    // stop cost timer
    timer_.cost_prior += GetDuration(start_cost);
  }

  // derivatives
  if (!gradient && !hessian) return cost;

  auto start_derivatives = std::chrono::steady_clock::now();

  // loop over configurations
  for (int t = 0; t < configuration_length_; t++) {
    // cost gradient wrt configuration
    if (gradient) {
      // unpack
      double* gt = gradient + t * nv;
      double* block = block_prior_current_configuration_.Get(t);

      // compute
      mju_mulMatTVec(gt, block, tmp + t * nv, nv, nv);

      // scale gradient: w * drdq' * scratch
      mju_scl(gt, gt, scale, nv);
    }

    // cost Hessian wrt configuration (sparse)
    if (hessian) {
      // TODO(taylor): skip terms for efficiency
      if (t < configuration_length_ - 2) {
        // mat
        double* mat = tmp + ntotal;
        mju_zero(mat, nband * nband);

        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            if (j <= i) {
              // unpack
              double* bbij = mat + nband * nband;
              double* tmp0 = bbij + nv * nv;
              double* tmp1 = tmp0 + nv * nv;

              // get matrices
              BlockFromMatrix(bbij, weight_prior_.data(), nv, nv, dim, dim,
                              (i + t) * nv, (j + t) * nv);
              const double* bdi = block_prior_current_configuration_.Get(i + t);
              const double* bdj = block_prior_current_configuration_.Get(j + t);

              // -- bdi' * bbij * bdj -- //

              // tmp0 = bbij * bdj
              mju_mulMatMat(tmp0, bbij, bdj, nv, nv, nv);

              // tmp1 = bdi' * tmp0
              mju_mulMatTMat(tmp1, bdi, tmp0, nv, nv, nv);

              // set scaled block in matrix
              SetBlockInMatrix(mat, tmp1, scale, nband, nband, nv, nv, i * nv,
                               j * nv);
            }
          }
        }
        // set mat in band Hessian
        SetBlockInBand(hessian, mat, 1.0, dim, nband, nband, t * nv, 0, false);
      }
    }
  }

  // stop derivatives timer
  timer_.cost_prior_derivatives += GetDuration(start_derivatives);

  return cost;
}

// prior residual
void Batch::ResidualPrior() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model->nv;

  // loop over configurations
  for (int t = 0; t < configuration_length_; t++) {
    // terms
    double* rt = residual_prior_.data() + t * nv;
    double* qt_prior = configuration_previous.Get(t);
    double* qt = configuration.Get(t);

    // configuration difference
    mj_differentiatePos(model, rt, 1.0, qt_prior, qt);
  }

  // stop timer
  timer_.residual_prior += GetDuration(start);
}

// prior Jacobian blocks
void Batch::BlockPrior(int index) {
  // unpack
  double* qt = configuration.Get(index);
  double* qt_prior = configuration_previous.Get(index);
  double* block = block_prior_current_configuration_.Get(index);

  // compute Jacobian
  DifferentiateDifferentiatePos(NULL, block, model, 1.0, qt_prior, qt);

  // assemble dense Jacobian
  if (settings.assemble_prior_jacobian) {
    // dimensions
    int nv = model->nv;
    int nvar = nv * configuration_length_;

    // set block
    SetBlockInMatrix(jacobian_prior_.data(), block, 1.0, nvar, nvar, nv, nv,
                     nv * index, nv * index);
  }
}

// prior Jacobian
// note: pool wait is called outside this function
void Batch::JacobianPrior(ThreadPool& pool) {
  // loop over predictions
  for (int t = 0; t < configuration_length_; t++) {
    // schedule by time step
    pool.Schedule([&batch = *this, t]() {
      // start Jacobian timer
      auto jacobian_prior_start = std::chrono::steady_clock::now();

      // block
      batch.BlockPrior(t);

      // stop Jacobian timer
      batch.timer_.prior_step[t] = GetDuration(jacobian_prior_start);
    });
  }
}

// sensor cost
double Batch::CostSensor(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // update dimension
  int nv = model->nv, ns = nsensordata_;
  int nvar = nv * configuration_length_;
  int nsen = ns * (configuration_length_ - 1);

  // residual
  if (!cost_skip_) ResidualSensor();

  // ----- cost ----- //

  // initialize
  double cost = 0.0;

  // zero memory
  if (gradient) mju_zero(gradient, nvar);
  if (hessian) mju_zero(hessian, nvar * (3 * nv));

  // time scaling
  double time_scale = 1.0;
  double time_scale2 = 1.0;
  if (settings.time_scaling_sensor) {
    time_scale = model->opt.timestep;
    time_scale2 = time_scale * time_scale;
  }

  // matrix shift
  int shift_matrix = 0;

  // loop over predictions
  for (int t = 0; t < configuration_length_ - 1; t++) {
    // residual
    double* rt = residual_sensor_.data() + ns * t;

    // mask
    // int* mask = sensor_mask.Get(t);

    // unpack block
    double* block;
    int block_columns;
    if (t == 0) {  // only position sensors
      block = block_sensor_configuration_.Get(t) + sensor_start_index_ * nv;
      block_columns = nv;
    } else {  // position, velocity, acceleration sensors
      block = block_sensor_configurations_.Get(t);
      block_columns = 3 * nv;
    }

    // shift
    int shift_sensor = 0;

    // loop over sensors
    for (int i = 0; i < nsensor_; i++) {
      // start cost timer
      auto start_cost = std::chrono::steady_clock::now();

      // sensor stage
      int sensor_stage = model->sensor_needstage[sensor_start_ + i];

      // time scaling weight
      double time_weight = 1.0;
      if (sensor_stage == mjSTAGE_VEL) {
        time_weight = time_scale;
      } else if (sensor_stage == mjSTAGE_ACC) {
        time_weight = time_scale2;
      }

      // check mask, skip if missing measurement
      // if (!mask[i]) continue;

      // dimension
      int nsi = model->sensor_dim[sensor_start_ + i];

      // sensor residual
      double* rti = rt + shift_sensor;

      // weight
      double weight =
          time_weight / noise_sensor[i] / nsi / (configuration_length_ - 1);

      // parameters
      double* pi = norm_parameters_sensor.data() + kMaxNormParameters * i;

      // norm
      NormType normi = norm_type_sensor[i];

      // norm gradient
      double* norm_gradient =
          norm_gradient_sensor_.data() + ns * t + shift_sensor;

      // norm Hessian
      double* norm_block = norm_blocks_sensor_.data() + shift_matrix;

      // ----- cost ----- //

      // norm
      norm_sensor_[nsensor_ * t + i] =
          Norm(gradient ? norm_gradient : NULL, hessian ? norm_block : NULL,
               rti, pi, nsi, normi);

      // weighted norm
      cost += weight * norm_sensor_[nsensor_ * t + i];

      // stop cost timer
      timer_.cost_sensor += GetDuration(start_cost);

      // assemble dense norm Hessian
      if (settings.assemble_sensor_norm_hessian) {
        // reset memory
        if (i == 0 && t == 0)
          mju_zero(norm_hessian_sensor_.data(), nsen * nsen);

        // set norm block
        SetBlockInMatrix(norm_hessian_sensor_.data(), norm_block, weight, nsen,
                         nsen, nsi, nsi, ns * t + shift_sensor,
                         ns * t + shift_sensor);
      }

      // gradient wrt configuration: dridq012' * dndri
      if (gradient) {
        // sensor block
        double* blocki = block + block_columns * shift_sensor;

        // scratch = dridq012' * dndri
        mju_mulMatTVec(scratch_sensor_.data(), blocki, norm_gradient, nsi,
                       block_columns);

        // add
        mju_addToScl(gradient + nv * std::max(0, t - 1),
                     scratch_sensor_.data(), weight, block_columns);
      }

      // Hessian (Gauss-Newton): drdq' * d2ndr2 * drdq
      if (hessian) {
        // sensor block
        double* blocki = block + block_columns * shift_sensor;

        // step 1: tmp0 = d2ndri2 * dridq
        double* tmp0 = scratch_sensor_.data();
        mju_mulMatMat(tmp0, norm_block, blocki, nsi, nsi, block_columns);

        // step 2: hessian = dridq' * tmp
        double* tmp1 = scratch_sensor_.data() + nsensordata_ * 3 * nv;
        mju_mulMatTMat(tmp1, blocki, tmp0, nsi, block_columns, block_columns);

        // set block in band Hessian
        SetBlockInBand(hessian, tmp1, weight, nvar, 3 * nv, block_columns,
                       nv * std::max(0, t - 1));
      }

      // shift by individual sensor dimension
      shift_sensor += nsi;
      shift_matrix += nsi * nsi;
    }
  }

  // stop timer
  timer_.cost_sensor_derivatives += GetDuration(start);

  return cost;
}

// force cost
double Batch::CostForce(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // update dimension
  int nv = model->nv;
  int nvar = nv * configuration_length_;
  int nforce = nv * (configuration_length_ - 2);

  // residual
  if (!cost_skip_) ResidualForce();

  // ----- cost ----- //

  // initialize
  double cost = 0.0;

  // zero memory
  if (gradient) mju_zero(gradient, nvar);
  if (hessian) mju_zero(hessian, nvar * (3 * nv));

  // time scaling
  double time_scale2 = 1.0;
  if (settings.time_scaling_force) {
    time_scale2 = model->opt.timestep * model->opt.timestep;
  }

  // loop over predictions
  for (int t = 1; t < configuration_length_ - 1; t++) {
    // unpack block
    double* block = block_force_configurations_.Get(t);

    // start cost timer
    auto start_cost = std::chrono::steady_clock::now();

    // residual
    double* rt = residual_force_.data() + t * nv;

    // norm gradient
    double* norm_gradient = norm_gradient_force_.data() + t * nv;

    // norm block
    double* norm_block = norm_blocks_force_.data() + t * nv * nv;
    mju_zero(norm_block, nv * nv);

    // ----- cost ----- //

    // quadratic cost
    for (int i = 0; i < nv; i++) {
      // weight
      double weight =
          time_scale2 / noise_process[i] / nv / (configuration_length_ - 2);

      // gradient
      norm_gradient[i] = weight * rt[i];

      // Hessian
      norm_block[nv * i + i] = weight;
    }

    // norm
    norm_sensor_[t] = 0.5 * mju_dot(rt, norm_gradient, nv);

    // weighted norm
    cost += norm_sensor_[t];

    // stop cost timer
    timer_.cost_force += GetDuration(start_cost);

    // assemble dense norm Hessian
    if (settings.assemble_force_norm_hessian) {
      // zero memory
      if (t == 1) mju_zero(norm_hessian_force_.data(), nforce * nforce);

      // set block
      SetBlockInMatrix(norm_hessian_force_.data(), norm_block, 1.0, nforce,
                       nforce, nv, nv, (t - 1) * nv, (t - 1) * nv);
    }

    // gradient wrt configuration: dridq012' * dndri
    if (gradient) {
      // scratch = dridq012' * dndri
      mju_mulMatTVec(scratch_force_.data(), block, norm_gradient, nv, 3 * nv);

      // add
      mju_addToScl(gradient + (t - 1) * nv, scratch_force_.data(), 1.0,
                   3 * nv);
    }

    // Hessian (Gauss-Newton): drdq' * d2ndr2 * drdq
    if (hessian) {
      // step 1: tmp0 = d2ndri2 * dridq
      double* tmp0 = scratch_force_.data();
      mju_mulMatMat(tmp0, norm_block, block, nv, nv, 3 * nv);

      // step 2: hessian = dridq' * tmp
      double* tmp1 = tmp0 + 3 * nv * nv;
      mju_mulMatTMat(tmp1, block, tmp0, nv, 3 * nv, 3 * nv);

      // set block in band Hessian
      SetBlockInBand(hessian, tmp1, 1.0, nvar, 3 * nv, 3 * nv, nv * (t - 1));
    }
  }

  // stop timer
  timer_.cost_force_derivatives += GetDuration(start);

  return cost;
}

// sensor residual
void Batch::ResidualSensor() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // loop over predictions
  for (int t = 0; t < configuration_length_ - 1; t++) {
    // terms
    double* rt = residual_sensor_.data() + t * nsensordata_;
    double* yt_sensor = sensor_measurement.Get(t);
    double* yt_model = sensor_prediction.Get(t);

    // sensor difference
    mju_sub(rt, yt_model, yt_sensor, nsensordata_);

    // zero out non-position sensors at first time step
    if (t == 0) {
      // loop over position sensors
      for (int i = 0; i < nsensor_; i++) {
        // sensor stage
        int sensor_stage = model->sensor_needstage[sensor_start_ + i];

        // check for position
        if (sensor_stage != mjSTAGE_POS) {
          // dimension
          int sensor_dim = model->sensor_dim[sensor_start_ + i];

          // address
          int sensor_adr = model->sensor_adr[sensor_start_ + i];

          // copy sensor data
          mju_zero(rt + sensor_adr - sensor_start_index_, sensor_dim);
        }
      }
    }
  }

  // stop timer
  timer_.residual_sensor += GetDuration(start);
}

// sensor Jacobian blocks (dsdq0, dsdq1, dsdq2)
void Batch::BlockSensor(int index) {
  // dimensions
  int nv = model->nv, ns = nsensordata_;
  int nvar = nv * configuration_length_;
  int nsen = nsensordata_ * (configuration_length_ - 1);

  // shift
  int shift = sensor_start_index_ * nv;

  // first time step
  if (index == 0) {
    double* block = block_sensor_configuration_.Get(0) + shift;

    // unpack
    double* dsdq012 = block_sensor_configurations_.Get(0);

    // set dsdq1
    SetBlockInMatrix(dsdq012, block, 1.0, ns, 3 * nv, ns, nv, 0, nv);

    // set block in dense Jacobian
    if (settings.assemble_sensor_jacobian) {
      SetBlockInMatrix(jacobian_sensor_.data(), block, 1.0, nsen, nvar,
                       nsensordata_, nv, 0, 0);
    }
    return;
  }

  // dqds
  double* dsdq = block_sensor_configuration_.Get(index) + shift;

  // dvds
  double* dsdv = block_sensor_velocity_.Get(index) + shift;

  // dads
  double* dsda = block_sensor_acceleration_.Get(index) + shift;

  // -- configuration previous: dsdq0 = dsdv * dvdq0 + dsda * dadq0 -- //

  // unpack
  double* dsdq0 = block_sensor_previous_configuration_.Get(index);
  double* tmp = block_sensor_scratch_.Get(index);

  // dsdq0 <- dvds' * dvdq0
  double* dvdq0 = block_velocity_previous_configuration_.Get(index);
  mju_mulMatMat(dsdq0, dsdv, dvdq0, ns, nv, nv);

  // dsdq0 += dads' * dadq0
  double* dadq0 = block_acceleration_previous_configuration_.Get(index);
  mju_mulMatMat(tmp, dsda, dadq0, ns, nv, nv);
  mju_addTo(dsdq0, tmp, ns * nv);

  // -- configuration current: dsdq1 = dsdq + dsdv * dvdq1 + dsda * dadq1 --

  // unpack
  double* dsdq1 = block_sensor_current_configuration_.Get(index);

  // dsdq1 <- dqds'
  mju_copy(dsdq1, dsdq, ns * nv);

  // dsdq1 += dvds' * dvdq1
  double* dvdq1 = block_velocity_current_configuration_.Get(index);
  mju_mulMatMat(tmp, dsdv, dvdq1, ns, nv, nv);
  mju_addTo(dsdq1, tmp, ns * nv);

  // dsdq1 += dads' * dadq1
  double* dadq1 = block_acceleration_current_configuration_.Get(index);
  mju_mulMatMat(tmp, dsda, dadq1, ns, nv, nv);
  mju_addTo(dsdq1, tmp, ns * nv);

  // -- configuration next: dsdq2 = dsda * dadq2 -- //

  // unpack
  double* dsdq2 = block_sensor_next_configuration_.Get(index);

  // dsdq2 = dads' * dadq2
  double* dadq2 = block_acceleration_next_configuration_.Get(index);
  mju_mulMatMat(dsdq2, dsda, dadq2, ns, nv, nv);

  // -- assemble dsdq012 block -- //

  // unpack
  double* dsdq012 = block_sensor_configurations_.Get(index);

  // set dfdq0
  SetBlockInMatrix(dsdq012, dsdq0, 1.0, ns, 3 * nv, ns, nv, 0, 0 * nv);

  // set dfdq1
  SetBlockInMatrix(dsdq012, dsdq1, 1.0, ns, 3 * nv, ns, nv, 0, 1 * nv);

  // set dfdq0
  SetBlockInMatrix(dsdq012, dsdq2, 1.0, ns, 3 * nv, ns, nv, 0, 2 * nv);

  // assemble dense Jacobian
  if (settings.assemble_sensor_jacobian) {
    // set block
    SetBlockInMatrix(jacobian_sensor_.data(), dsdq012, 1.0, nsen, nvar,
                     nsensordata_, 3 * nv, index * nsensordata_,
                     (index - 1) * nv);
  }
}

// sensor Jacobian
// note: pool wait is called outside this function
void Batch::JacobianSensor(ThreadPool& pool) {
  // loop over predictions
  for (int t = 0; t < configuration_length_ - 1; t++) {
    // schedule by time step
    pool.Schedule([&batch = *this, t]() {
      // start Jacobian timer
      auto jacobian_sensor_start = std::chrono::steady_clock::now();

      // block
      batch.BlockSensor(t);

      // stop Jacobian timer
      batch.timer_.sensor_step[t] = GetDuration(jacobian_sensor_start);
    });
  }
}

// force residual
void Batch::ResidualForce() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model->nv;

  // loop over predictions
  for (int t = 1; t < configuration_length_ - 1; t++) {
    // terms
    double* rt = residual_force_.data() + t * nv;
    double* ft_actuator = force_measurement.Get(t);
    double* ft_inverse = force_prediction.Get(t);

    // force difference
    mju_sub(rt, ft_inverse, ft_actuator, nv);
  }

  // stop timer
  timer_.residual_force += GetDuration(start);
}

// force Jacobian blocks (dfdq0, dfdq1, dfdq2)
void Batch::BlockForce(int index) {
  // dimensions
  int nv = model->nv;

  // dqdf
  double* dqdf = block_force_configuration_.Get(index);

  // dvdf
  double* dvdf = block_force_velocity_.Get(index);

  // dadf
  double* dadf = block_force_acceleration_.Get(index);

  // -- configuration previous: dfdq0 = dfdv * dvdq0 + dfda * dadq0 -- //

  // unpack
  double* dfdq0 = block_force_previous_configuration_.Get(index);
  double* tmp = block_force_scratch_.Get(index);

  // dfdq0 <- dvdf' * dvdq0
  double* dvdq0 = block_velocity_previous_configuration_.Get(index);
  mju_mulMatTMat(dfdq0, dvdf, dvdq0, nv, nv, nv);

  // dfdq0 += dadf' * dadq0
  double* dadq0 = block_acceleration_previous_configuration_.Get(index);
  mju_mulMatTMat(tmp, dadf, dadq0, nv, nv, nv);
  mju_addTo(dfdq0, tmp, nv * nv);

  // -- configuration current: dfdq1 = dfdq + dfdv * dvdq1 + dfda * dadq1 --

  // unpack
  double* dfdq1 = block_force_current_configuration_.Get(index);

  // dfdq1 <- dqdf'
  mju_transpose(dfdq1, dqdf, nv, nv);

  // dfdq1 += dvdf' * dvdq1
  double* dvdq1 = block_velocity_current_configuration_.Get(index);
  mju_mulMatTMat(tmp, dvdf, dvdq1, nv, nv, nv);
  mju_addTo(dfdq1, tmp, nv * nv);

  // dfdq1 += dadf' * dadq1
  double* dadq1 = block_acceleration_current_configuration_.Get(index);
  mju_mulMatTMat(tmp, dadf, dadq1, nv, nv, nv);
  mju_addTo(dfdq1, tmp, nv * nv);

  // -- configuration next: dfdq2 = dfda * dadq2 -- //

  // unpack
  double* dfdq2 = block_force_next_configuration_.Get(index);

  // dfdq2 = dadf' * dadq2
  double* dadq2 = block_acceleration_next_configuration_.Get(index);
  mju_mulMatTMat(dfdq2, dadf, dadq2, nv, nv, nv);

  // -- assemble dfdq012 block -- //

  // unpack
  double* dfdq012 = block_force_configurations_.Get(index);

  // set dfdq0
  SetBlockInMatrix(dfdq012, dfdq0, 1.0, nv, 3 * nv, nv, nv, 0, 0 * nv);

  // set dfdq1
  SetBlockInMatrix(dfdq012, dfdq1, 1.0, nv, 3 * nv, nv, nv, 0, 1 * nv);

  // set dfdq0
  SetBlockInMatrix(dfdq012, dfdq2, 1.0, nv, 3 * nv, nv, nv, 0, 2 * nv);

  // assemble dense Jacobian
  if (settings.assemble_force_jacobian) {
    // dimensions
    int nv = model->nv;
    int nvar = nv * configuration_length_;
    int nforce = nv * (configuration_length_ - 2);

    // set block
    SetBlockInMatrix(jacobian_force_.data(), dfdq012, 1.0, nforce, nvar, nv,
                     3 * nv, (index - 1) * nv, (index - 1) * nv);
  }
}

// force Jacobian
// note: pool wait is called outside this function
void Batch::JacobianForce(ThreadPool& pool) {
  // loop over predictions
  for (int t = 1; t < configuration_length_ - 1; t++) {
    // schedule by time step
    pool.Schedule([&batch = *this, t]() {
      // start Jacobian timer
      auto jacobian_force_start = std::chrono::steady_clock::now();

      // block
      batch.BlockForce(t);

      // stop Jacobian timer
      batch.timer_.force_step[t] = GetDuration(jacobian_force_start);
    });
  }
}

// compute force
// TODO(taylor): combine with Jacobian method
void Batch::InverseDynamicsPrediction(ThreadPool& pool) {
  // compute sensor and force predictions
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model->nq, nv = model->nv, na = model->na, nu = model->nu,
      ns = nsensordata_;

  // pool count
  int count_before = pool.GetCount();

  // first time step
  pool.Schedule([&batch = *this, nq, nv, nu]() {
    // time index
    int t = 0;

    // data
    mjData* d = batch.data_[t].get();

    // terms
    double* q0 = batch.configuration.Get(t);
    double* y0 = batch.sensor_prediction.Get(t);
    mju_zero(y0, batch.nsensordata_);

    // set data
    mju_copy(d->qpos, q0, nq);
    mju_zero(d->qvel, nv);
    mju_zero(d->ctrl, nu);
    d->time = batch.times.Get(t)[0];

    // position sensors
    mj_fwdPosition(batch.model, d);
    mj_sensorPos(batch.model, d);

    // loop over position sensors
    for (int i = 0; i < batch.nsensor_; i++) {
      // sensor stage
      int sensor_stage = batch.model->sensor_needstage[batch.sensor_start_ + i];

      // check for position
      if (sensor_stage == mjSTAGE_POS) {
        // dimension
        int sensor_dim = batch.model->sensor_dim[batch.sensor_start_ + i];

        // address
        int sensor_adr = batch.model->sensor_adr[batch.sensor_start_ + i];

        // copy sensor data
        mju_copy(y0 + sensor_adr - batch.sensor_start_index_,
                 d->sensordata + sensor_adr, sensor_dim);
      }
    }
  });

  // loop over predictions
  for (int t = 1; t < configuration_length_ - 1; t++) {
    // schedule
    pool.Schedule([&batch = *this, nq, nv, na, ns, nu, t]() {
      // terms
      double* qt = batch.configuration.Get(t);
      double* vt = batch.velocity.Get(t);
      double* at = batch.acceleration.Get(t);
      double* ct = batch.ctrl.Get(t);

      // data
      mjData* d = batch.data_[t].get();

      // set qt, vt, at
      mju_copy(d->qpos, qt, nq);
      mju_copy(d->qvel, vt, nv);
      mju_copy(d->qacc, at, nv);
      mju_copy(d->ctrl, ct, nu);

      // inverse dynamics
      mj_inverse(batch.model, d);

      // copy sensor
      double* st = batch.sensor_prediction.Get(t);
      mju_copy(st, d->sensordata + batch.sensor_start_index_, ns);

      // copy force
      double* ft = batch.force_prediction.Get(t);
      mju_copy(ft, d->qfrc_inverse, nv);

      // copy act
      double* act = batch.act.Get(t + 1);
      mju_copy(act, d->act, na);
    });
  }

  // wait
  pool.WaitCount(count_before + configuration_length_ - 1);
  pool.ResetCount();

  // stop timer
  timer_.cost_prediction += GetDuration(start);
}

// compute inverse dynamics derivatives (via finite difference)
void Batch::InverseDynamicsDerivatives(ThreadPool& pool) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model->nq, nv = model->nv, nu = model->nu;

  // pool count
  int count_before = pool.GetCount();

  // first time step
  pool.Schedule([&batch = *this, nq, nv, nu]() {
    // time index
    int t = 0;

    // data
    mjData* d = batch.data_[t].get();

    // terms
    double* q0 = batch.configuration.Get(t);
    double* dsdq = batch.block_sensor_configuration_.Get(t);

    // set data
    mju_copy(d->qpos, q0, nq);
    mju_zero(d->qvel, nv);
    mju_zero(d->ctrl, nu);
    d->time = batch.times.Get(t)[0];

    // finite-difference derivatives
    double* dqds = batch.block_sensor_configurationT_.Get(t);
    mjd_inverseFD(batch.model, d, batch.finite_difference.tolerance,
                  batch.finite_difference.flg_actuation, NULL, NULL, NULL, dqds,
                  NULL, NULL, NULL);
    // transpose
    mju_transpose(dsdq, dqds, nv, batch.model->nsensordata);

    // loop over position sensors
    for (int i = 0; i < batch.nsensor_; i++) {
      // sensor stage
      int sensor_stage = batch.model->sensor_needstage[batch.sensor_start_ + i];

      // dimension
      int sensor_dim = batch.model->sensor_dim[batch.sensor_start_ + i];

      // address
      int sensor_adr = batch.model->sensor_adr[batch.sensor_start_ + i];

      // check for position
      if (sensor_stage != mjSTAGE_POS) {
        // zero remaining rows
        mju_zero(dsdq + sensor_adr * nv, sensor_dim * nv);
      }
    }
  });

  // loop over predictions
  for (int t = 1; t < configuration_length_ - 1; t++) {
    // schedule
    pool.Schedule([&batch = *this, nq, nv, nu, t]() {
      // unpack
      double* q = batch.configuration.Get(t);
      double* v = batch.velocity.Get(t);
      double* a = batch.acceleration.Get(t);
      double* c = batch.ctrl.Get(t);

      double* dsdq = batch.block_sensor_configuration_.Get(t);
      double* dsdv = batch.block_sensor_velocity_.Get(t);
      double* dsda = batch.block_sensor_acceleration_.Get(t);
      double* dqds = batch.block_sensor_configurationT_.Get(t);
      double* dvds = batch.block_sensor_velocityT_.Get(t);
      double* dads = batch.block_sensor_accelerationT_.Get(t);
      double* dqdf = batch.block_force_configuration_.Get(t);
      double* dvdf = batch.block_force_velocity_.Get(t);
      double* dadf = batch.block_force_acceleration_.Get(t);
      mjData* data = batch.data_[t].get();  // TODO(taylor): WorkerID

      // set (state, acceleration) + ctrl
      mju_copy(data->qpos, q, nq);
      mju_copy(data->qvel, v, nv);
      mju_copy(data->qacc, a, nv);
      mju_copy(data->ctrl, c, nu);

      // finite-difference derivatives
      mjd_inverseFD(batch.model, data, batch.finite_difference.tolerance,
                    batch.finite_difference.flg_actuation, dqdf, dvdf, dadf,
                    dqds, dvds, dads, NULL);

      // transpose
      mju_transpose(dsdq, dqds, nv, batch.model->nsensordata);
      mju_transpose(dsdv, dvds, nv, batch.model->nsensordata);
      mju_transpose(dsda, dads, nv, batch.model->nsensordata);
    });
  }

  // wait
  pool.WaitCount(count_before + configuration_length_ - 1);

  // reset pool count
  pool.ResetCount();

  // stop timer
  timer_.inverse_dynamics_derivatives += GetDuration(start);
}

// update configuration trajectory
void Batch::UpdateConfiguration(
    EstimatorTrajectory<double>& candidate,
    const EstimatorTrajectory<double>& configuration,
    const double* search_direction, double step_size) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model->nq, nv = model->nv;

  // loop over configurations
  for (int t = 0; t < configuration_length_; t++) {
    // unpack
    const double* qt = configuration.Get(t);
    double* ct = candidate.Get(t);

    // copy
    mju_copy(ct, qt, nq);

    // search direction
    const double* dqt = search_direction + t * nv;

    // integrate
    mj_integratePos(model, ct, dqt, step_size);
  }

  // joint limits
  if (settings.joint_limits) JointLimits();

  // stop timer
  timer_.configuration_update += GetDuration(start);
}

// convert sequence of configurations to velocities and accelerations
void Batch::ConfigurationToVelocityAcceleration() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model->nv;

  // loop over configurations
  for (int t = 1; t < configuration_length_; t++) {
    // previous and current configurations
    const double* q0 = configuration.Get(t - 1);
    const double* q1 = configuration.Get(t);

    // compute velocity
    double* v1 = velocity.Get(t);
    mj_differentiatePos(model, v1, model->opt.timestep, q0, q1);

    // compute acceleration
    if (t > 1) {
      // previous velocity
      const double* v0 = velocity.Get(t - 1);

      // compute acceleration
      double* a1 = acceleration.Get(t - 1);
      mju_sub(a1, v1, v0, nv);
      mju_scl(a1, a1, 1.0 / model->opt.timestep, nv);
    }
  }

  // stop time
  timer_.cost_config_to_velacc += GetDuration(start);
}

// compute finite-difference velocity, acceleration derivatives
void Batch::VelocityAccelerationDerivatives() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model->nv;

  // loop over configurations
  for (int t = 1; t < configuration_length_; t++) {
    // unpack
    double* q1 = configuration.Get(t - 1);
    double* q2 = configuration.Get(t);
    double* dv2dq1 = block_velocity_previous_configuration_.Get(t);
    double* dv2dq2 = block_velocity_current_configuration_.Get(t);

    // compute velocity Jacobians
    DifferentiateDifferentiatePos(dv2dq1, dv2dq2, model, model->opt.timestep,
                                  q1, q2);

    // compute acceleration Jacobians
    if (t > 1) {
      // unpack
      double* dadq0 = block_acceleration_previous_configuration_.Get(t - 1);
      double* dadq1 = block_acceleration_current_configuration_.Get(t - 1);
      double* dadq2 = block_acceleration_next_configuration_.Get(t - 1);

      // previous velocity Jacobians
      double* dv1dq0 = block_velocity_previous_configuration_.Get(t - 1);
      double* dv1dq1 = block_velocity_current_configuration_.Get(t - 1);

      // dadq0 = -dv1dq0 / h
      mju_copy(dadq0, dv1dq0, nv * nv);
      mju_scl(dadq0, dadq0, -1.0 / model->opt.timestep, nv * nv);

      // dadq1 = dv2dq1 / h - dv1dq1 / h = (dv2dq1 - dv1dq1) / h
      mju_sub(dadq1, dv2dq1, dv1dq1, nv * nv);
      mju_scl(dadq1, dadq1, 1.0 / model->opt.timestep, nv * nv);

      // dadq2 = dv2dq2 / h
      mju_copy(dadq2, dv2dq2, nv * nv);
      mju_scl(dadq2, dadq2, 1.0 / model->opt.timestep, nv * nv);
    }
  }

  // stop timer
  timer_.velacc_derivatives += GetDuration(start);
}

// initialize filter mode
void Batch::InitializeFilter() {
  // dimensions
  int nq = model->nq, nv = model->nv;

  // filter mode status
  current_time_index_ = 1;

  // filter settings
  settings.gradient_tolerance = 1.0e-6;
  settings.max_smoother_iterations = 1;
  settings.max_search_iterations = 10;
  settings.recursive_prior_update = true;

  // timestep
  double timestep = model->opt.timestep;

  // set q1
  configuration.Set(state.data(), 1);
  configuration_previous.Set(configuration.Get(1), 1);

  // set q0
  double* q0 = configuration.Get(0);
  mju_copy(q0, state.data(), nq);
  mj_integratePos(model, q0, state.data() + nq, -1.0 * timestep);
  configuration_previous.Set(configuration.Get(0), 0);

  // set times
  double current_time = time - timestep;
  times.Set(&current_time, 0);
  for (int i = 1; i < configuration_length_; i++) {
    // increment time
    current_time += timestep;

    // set
    times.Set(&current_time, i);
  }

  // -- set initial position-based measurements -- //

  // data
  mjData* data = data_[0].get();

  // set q0
  mju_copy(data->qpos, q0, model->nq);

  // evaluate position
  mj_fwdPosition(model, data);
  mj_sensorPos(model, data);

  // y0
  double* y0 = sensor_measurement.Get(0);
  mju_zero(y0, nsensordata_);

  // loop over sensors
  for (int i = 0; i < nsensor_; i++) {
    // measurement sensor index
    int index = sensor_start_ + i;

    // need stage
    int sensor_stage = model->sensor_needstage[index];

    // position sensor
    if (sensor_stage == mjSTAGE_POS) {
      // address
      int sensor_adr = model->sensor_adr[index];

      // dimension
      int sensor_dim = model->sensor_dim[index];

      // set sensor
      mju_copy(y0 + sensor_adr - sensor_start_index_,
               data->sensordata + sensor_adr, sensor_dim);
    }
  }

  // prior weight
  int nvar = nv * configuration_length_;
  for (int i = 0; i < nvar; i++) {
    weight_prior_[nvar * i + i] = scale_prior;
  }
}

// shift head and resize trajectories
void Batch::ShiftResizeTrajectory(int new_head, int new_length) {
  // reset cache
  configuration_cache_.Reset();
  configuration_previous_cache_.Reset();
  velocity_cache_.Reset();
  acceleration_cache_.Reset();
  act_cache_.Reset();
  times_cache_.Reset();
  ctrl_cache_.Reset();
  sensor_measurement_cache_.Reset();
  sensor_prediction_cache_.Reset();
  sensor_mask_cache_.Reset();
  force_measurement_cache_.Reset();
  force_prediction_cache_.Reset();

  // -- set cache length -- //
  int length = configuration_length_;

  configuration_cache_.SetLength(length);
  configuration_previous_cache_.SetLength(length);
  velocity_cache_.SetLength(length);
  acceleration_cache_.SetLength(length);
  act_cache_.SetLength(length);
  times_cache_.SetLength(length);
  ctrl_cache_.SetLength(length);
  sensor_measurement_cache_.SetLength(length);
  sensor_prediction_cache_.SetLength(length);
  sensor_mask_cache_.SetLength(length);
  force_measurement_cache_.SetLength(length);
  force_prediction_cache_.SetLength(length);

  // copy data to cache
  for (int i = 0; i < length; i++) {
    configuration_cache_.Set(configuration.Get(i), i);
    configuration_previous_cache_.Set(configuration_previous.Get(i), i);
    velocity_cache_.Set(velocity.Get(i), i);
    acceleration_cache_.Set(acceleration.Get(i), i);
    act_cache_.Set(act.Get(i), i);
    times_cache_.Set(times.Get(i), i);
    ctrl_cache_.Set(ctrl.Get(i), i);
    sensor_measurement_cache_.Set(sensor_measurement.Get(i), i);
    sensor_prediction_cache_.Set(sensor_prediction.Get(i), i);
    sensor_mask_cache_.Set(sensor_mask.Get(i), i);
    force_measurement_cache_.Set(force_measurement.Get(i), i);
    force_prediction_cache_.Set(force_prediction.Get(i), i);
  }

  // set configuration length
  SetConfigurationLength(new_length);

  // set trajectory data
  int length_copy = std::min(length, new_length);
  for (int i = 0; i < length_copy; i++) {
    configuration.Set(configuration_cache_.Get(new_head + i), i);
    configuration_previous.Set(configuration_previous_cache_.Get(new_head + i),
                               i);
    velocity.Set(velocity_cache_.Get(new_head + i), i);
    acceleration.Set(acceleration_cache_.Get(new_head + i), i);
    act.Set(act_cache_.Get(new_head + i), i);
    times.Set(times_cache_.Get(new_head + i), i);
    ctrl.Set(ctrl_cache_.Get(new_head + i), i);
    sensor_measurement.Set(sensor_measurement_cache_.Get(new_head + i), i);
    sensor_prediction.Set(sensor_prediction_cache_.Get(new_head + i), i);
    sensor_mask.Set(sensor_mask_cache_.Get(new_head + i), i);
    force_measurement.Set(force_measurement_cache_.Get(new_head + i), i);
    force_prediction.Set(force_prediction_cache_.Get(new_head + i), i);
  }
}

// compute total cost
double Batch::Cost(double* gradient, double* hessian, ThreadPool& pool) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // evaluate configurations
  if (!cost_skip_) ConfigurationEvaluation(pool);

  // derivatives
  if (gradient || hessian) {
    ConfigurationDerivative(pool);
  }

  // start cost derivative timer
  auto start_cost_derivatives = std::chrono::steady_clock::now();

  // pool count
  int count_begin = pool.GetCount();

  bool gradient_flag = (gradient ? true : false);
  bool hessian_flag = (hessian ? true : false);

  // -- individual cost derivatives -- //

  // prior
  if (settings.prior_flag) {
    pool.Schedule([&batch = *this, gradient_flag, hessian_flag]() {
      batch.cost_prior_ = batch.CostPrior(
          gradient_flag ? batch.cost_gradient_prior_.data() : NULL,
          hessian_flag ? batch.cost_hessian_prior_band_.data() : NULL);
    });
  }

  // sensor
  if (settings.sensor_flag) {
    pool.Schedule([&batch = *this, gradient_flag, hessian_flag]() {
      batch.cost_sensor_ = batch.CostSensor(
          gradient_flag ? batch.cost_gradient_sensor_.data() : NULL,
          hessian_flag ? batch.cost_hessian_sensor_band_.data() : NULL);
    });
  }

  // force
  if (settings.force_flag) {
    pool.Schedule([&batch = *this, gradient_flag, hessian_flag]() {
      batch.cost_force_ = batch.CostForce(
          gradient_flag ? batch.cost_gradient_force_.data() : NULL,
          hessian_flag ? batch.cost_hessian_force_band_.data() : NULL);
    });
  }

  // wait
  pool.WaitCount(count_begin + settings.prior_flag + settings.sensor_flag +
                 settings.force_flag);
  pool.ResetCount();

  // total cost
  double cost = cost_prior_ + cost_sensor_ + cost_force_;

  // total gradient, hessian
  TotalGradient(gradient);
  TotalHessian(hessian);

  // counter
  if (!cost_skip_) cost_count_++;

  // -- stop timer -- //

  // cost time
  if (!cost_skip_) {
    timer_.cost += GetDuration(start);
  }

  // cost derivative time
  if (gradient || hessian) {
    timer_.cost_derivatives += GetDuration(start);
    timer_.cost_total_derivatives += GetDuration(start_cost_derivatives);
  }

  // reset skip flag
  cost_skip_ = false;

  // total cost
  return cost;
}

// compute total gradient
void Batch::TotalGradient(double* gradient) {
  if (!gradient) return;

  // start gradient timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int dim = configuration_length_ * model->nv;

  // individual gradients
  if (settings.prior_flag) {
    mju_copy(gradient, cost_gradient_prior_.data(), dim);
  } else {
    mju_zero(gradient, dim);
  }
  if (settings.sensor_flag)
    mju_addTo(gradient, cost_gradient_sensor_.data(), dim);
  if (settings.force_flag)
    mju_addTo(gradient, cost_gradient_force_.data(), dim);

  // stop gradient timer
  timer_.cost_gradient += GetDuration(start);
}

// compute total Hessian
void Batch::TotalHessian(double* hessian) {
  if (!hessian) return;

  // start Hessian timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model->nv;
  int ntotal = nv * configuration_length_;
  int nband = 3 * nv;

  // zero memory
  mju_zero(hessian, ntotal * nband);

  // individual Hessians
  if (settings.prior_flag) {
    mju_addTo(hessian, cost_hessian_prior_band_.data(), ntotal * nband);
  }

  if (settings.sensor_flag) {
    mju_addTo(hessian, cost_hessian_sensor_band_.data(), ntotal * nband);
  }

  if (settings.force_flag) {
    mju_addTo(hessian, cost_hessian_force_band_.data(), ntotal * nband);
  }

  // stop Hessian timer
  timer_.cost_hessian += GetDuration(start);
}

// optimize trajectory estimate
void Batch::Optimize(ThreadPool& pool) {
  // start timer
  auto start_optimize = std::chrono::steady_clock::now();

  // set status
  gradient_norm_ = 0.0;
  search_direction_norm_ = 0.0;
  solve_status_ = kUnsolved;

  // dimensions
  int nconfig = model->nq * configuration_length_;
  int nvar = model->nv * configuration_length_;

  // reset timers
  ResetTimers();

  // joint limits 
  if (settings.joint_limits) JointLimits();

  // initial cost
  cost_count_ = 0;
  cost_skip_ = false;
  cost_ = Cost(NULL, NULL, pool);
  cost_initial_ = cost_;

  // print initial cost
  PrintCost();

  // ----- smoother iterations ----- //

  // reset
  iterations_smoother_ = 0;
  iterations_search_ = 0;

  // iterations
  for (; iterations_smoother_ < settings.max_smoother_iterations;
       iterations_smoother_++) {
    // evalute cost derivatives
    cost_skip_ = true;
    Cost(cost_gradient_.data(), cost_hessian_band_.data(), pool);

    // start timer
    auto start_search = std::chrono::steady_clock::now();

    // -- gradient -- //
    double* gradient = cost_gradient_.data();

    // gradient tolerance check
    gradient_norm_ = mju_norm(gradient, nvar) / nvar;
    if (gradient_norm_ < settings.gradient_tolerance) {
      break;
    }

    // ----- line / curve search ----- //

    // copy configuration
    mju_copy(configuration_copy_.Data(), configuration.Data(), nconfig);

    // initialize
    double cost_candidate = cost_;
    int iteration_search = 0;
    step_size_ = 1.0;
    regularization_ = settings.regularization_initial;
    improvement_ = -1.0;

    // -- search direction -- //

    // check regularization
    if (regularization_ >= kMaxBatchRegularization - 1.0e-6) {
      // set solve status
      solve_status_ = kMaxRegularizationFailure;

      // failure
      return;
    }

    // compute initial search direction
    SearchDirection();

    // check small search direction
    if (search_direction_norm_ < settings.search_direction_tolerance) {
      // set solve status
      solve_status_ = kSmallDirectionFailure;

      // failure
      return;
    }

    // backtracking until cost decrease
    // TODO(taylor): Armijo, Wolfe conditions
    while (cost_candidate >= cost_) {
      // check for max iterations
      if (iteration_search > settings.max_search_iterations) {
        // set solve status
        solve_status_ = kMaxIterationsFailure;

        // failure
        return;
      }

      // search type
      if (iteration_search > 0) {
        switch (settings.search_type) {
          case SearchType::kLineSearch:
            // decrease step size
            step_size_ *= settings.step_scaling;
            break;
          case SearchType::kCurveSearch:
            // increase regularization
            IncreaseRegularization();

            // recompute search direction
            SearchDirection();

            // check small search direction
            if (search_direction_norm_ < settings.search_direction_tolerance) {
              // set solve status
              solve_status_ = kSmallDirectionFailure;

              // failure
              return;
            }
            break;
          default:
            mju_error("Invalid search type.\n");
            break;
        }
      }

      // candidate
      UpdateConfiguration(configuration, configuration_copy_,
                          search_direction_.data(), -1.0 * step_size_);

      // cost
      cost_skip_ = false;
      cost_candidate = Cost(NULL, NULL, pool);

      // improvement
      improvement_ = cost_ - cost_candidate;

      // update iteration
      iteration_search++;
    }

    // increment
    iterations_search_ += iteration_search;

    // update cost
    cost_previous_ = cost_;
    cost_ = cost_candidate;

    // check cost difference
    cost_difference_ = std::abs(cost_ - cost_previous_);
    if (cost_difference_ < settings.cost_tolerance) {
      // set status
      solve_status_ = kCostDifferenceFailure;

      // failure
      return;
    }

    // curve search
    if (settings.search_type == kCurveSearch) {
      // expected = g' d + 0.5 d' H d

      // g' * d
      expected_ =
          mju_dot(cost_gradient_.data(), search_direction_.data(), nvar);

      // tmp = H * d
      double* tmp = scratch_expected_.data();
      mju_bandMulMatVec(tmp, cost_hessian_band_.data(),
                        search_direction_.data(), nvar, 3 * model->nv, 0, 1,
                        true);

      // expected += 0.5 d' tmp
      expected_ += 0.5 * mju_dot(search_direction_.data(), tmp, nvar);

      // check for no expected decrease
      if (expected_ <= 0.0) {
        // set status
        solve_status_ = kExpectedDecreaseFailure;

        // failure
        return;
      }

      // reduction ratio
      reduction_ratio_ = improvement_ / expected_;

      // update regularization
      if (reduction_ratio_ > 0.75) {
        // decrease
        regularization_ =
            mju_max(kMinBatchRegularization,
                    regularization_ / settings.regularization_scaling);
      } else if (reduction_ratio_ < 0.25) {
        // increase
        regularization_ =
            mju_min(kMaxBatchRegularization,
                    regularization_ * settings.regularization_scaling);
      }
    }

    // end timer
    timer_.search += GetDuration(start_search);

    // print cost
    PrintCost();
  }

  // stop timer
  timer_.optimize = GetDuration(start_optimize);

  // set solve status
  if (iterations_smoother_ >= settings.max_smoother_iterations) {
    solve_status_ = kMaxIterationsFailure;
  } else {
    solve_status_ = kSolved;
  }

  // status
  PrintOptimize();
}

// search direction
void Batch::SearchDirection() {
  // start timer
  auto search_direction_start = std::chrono::steady_clock::now();

  // dimensions
  int ntotal = configuration_length_ * model->nv;
  int nband = 3 * model->nv;
  int ndense = 0;

  // -- band Hessian -- //

  // unpack
  double* direction = search_direction_.data();
  double* gradient = cost_gradient_.data();
  double* hessian_band = cost_hessian_band_.data();
  double* hessian_band_factor = cost_hessian_band_factor_.data();

  // -- linear system solver -- //

  // increase regularization until full rank
  double min_diag = 0.0;
  while (min_diag <= 0.0) {
    // failure
    if (regularization_ >= kMaxBatchRegularization) {
      printf("min diag = %f\n", min_diag);
      mju_error("cost Hessian factorization failure: MAX REGULARIZATION\n");
    }

    // copy
    mju_copy(hessian_band_factor, hessian_band,
              ntotal * (3 * model->nv));

    // factorize
    min_diag = mju_cholFactorBand(hessian_band_factor, ntotal, nband, ndense,
                                  regularization_, 0.0);

    // increase regularization
    if (min_diag <= 0.0) {
      IncreaseRegularization();
    }
  }

  // compute search direction
  mju_cholSolveBand(direction, hessian_band_factor, gradient, ntotal, nband,
                    ndense);

  // search direction norm
  search_direction_norm_ = InfinityNorm(direction, ntotal);

  // set regularization
  if (regularization_ > 0.0) {
    for (int i = 0; i < ntotal; i++) {
      hessian_band[i * nband + nband - 1] += regularization_;
    }
  }

  // end timer
  timer_.search_direction += GetDuration(search_direction_start);
}

// print Optimize iteration
void Batch::PrintIteration() {
  if (!settings.verbose_iteration) return;
}

// print Optimize status
void Batch::PrintOptimize() {
  if (!settings.verbose_optimize) return;

  // title
  printf("Batch::Optimize Status:\n\n");

  // timing
  printf("Timing:\n");

  printf("\n");
  printf("  cost : %.3f (ms) \n", 1.0e-3 * timer_.cost / cost_count_);
  printf("    - prior: %.3f (ms) \n", 1.0e-3 * timer_.cost_prior / cost_count_);
  printf("    - sensor: %.3f (ms) \n",
         1.0e-3 * timer_.cost_sensor / cost_count_);
  printf("    - force: %.3f (ms) \n", 1.0e-3 * timer_.cost_force / cost_count_);
  printf("    - qpos -> qvel, qacc: %.3f (ms) \n",
         1.0e-3 * timer_.cost_config_to_velacc / cost_count_);
  printf("    - prediction: %.3f (ms) \n",
         1.0e-3 * timer_.cost_prediction / cost_count_);
  printf("    - residual prior: %.3f (ms) \n",
         1.0e-3 * timer_.residual_prior / cost_count_);
  printf("    - residual sensor: %.3f (ms) \n",
         1.0e-3 * timer_.residual_sensor / cost_count_);
  printf("    - residual force: %.3f (ms) \n",
         1.0e-3 * timer_.residual_force / cost_count_);
  printf("    [cost_count = %i]\n", cost_count_);
  printf("\n");
  printf("  cost derivatives [total]: %.3f (ms) \n",
         1.0e-3 * timer_.cost_derivatives);
  printf("    - inverse dynamics derivatives: %.3f (ms) \n",
         1.0e-3 * timer_.inverse_dynamics_derivatives);
  printf("    - vel., acc. derivatives: %.3f (ms) \n",
         1.0e-3 * timer_.velacc_derivatives);
  printf("    - jacobian [total]: %.3f (ms) \n",
         1.0e-3 * timer_.jacobian_total);
  printf("      < prior: %.3f (ms) \n", 1.0e-3 * timer_.jacobian_prior);
  printf("      < sensor: %.3f (ms) \n", 1.0e-3 * timer_.jacobian_sensor);
  printf("      < force: %.3f (ms) \n", 1.0e-3 * timer_.jacobian_force);
  printf("    - gradient, hessian [total]: %.3f (ms) \n",
         1.0e-3 * timer_.cost_total_derivatives);
  printf("      < prior: %.3f (ms) \n", 1.0e-3 * timer_.cost_prior_derivatives);
  printf("      < sensor: %.3f (ms) \n",
         1.0e-3 * timer_.cost_sensor_derivatives);
  printf("      < force: %.3f (ms) \n", 1.0e-3 * timer_.cost_force_derivatives);
  printf("      < gradient assemble: %.3f (ms) \n ",
         1.0e-3 * timer_.cost_gradient);
  printf("      < hessian assemble: %.3f (ms) \n",
         1.0e-3 * timer_.cost_hessian);
  printf("\n");
  printf("  search [total]: %.3f (ms) \n", 1.0e-3 * timer_.search);
  printf("    - direction: %.3f (ms) \n", 1.0e-3 * timer_.search_direction);
  printf("    - cost: %.3f (ms) \n",
         1.0e-3 * (timer_.cost - timer_.cost / cost_count_));
  printf("      < prior: %.3f (ms) \n",
         1.0e-3 * (timer_.cost_prior - timer_.cost_prior / cost_count_));
  printf("      < sensor: %.3f (ms) \n",
         1.0e-3 * (timer_.cost_sensor - timer_.cost_sensor / cost_count_));
  printf("      < force: %.3f (ms) \n",
         1.0e-3 * (timer_.cost_force - timer_.cost_force / cost_count_));
  printf("      < qpos -> qvel, qacc: %.3f (ms) \n",
         1.0e-3 * (timer_.cost_config_to_velacc -
                   timer_.cost_config_to_velacc / cost_count_));
  printf(
      "      < prediction: %.3f (ms) \n",
      1.0e-3 * (timer_.cost_prediction - timer_.cost_prediction / cost_count_));
  printf(
      "      < residual prior: %.3f (ms) \n",
      1.0e-3 * (timer_.residual_prior - timer_.residual_prior / cost_count_));
  printf(
      "      < residual sensor: %.3f (ms) \n",
      1.0e-3 * (timer_.residual_sensor - timer_.residual_sensor / cost_count_));
  printf(
      "      < residual force: %.3f (ms) \n",
      1.0e-3 * (timer_.residual_force - timer_.residual_force / cost_count_));
  printf("\n");
  printf("  TOTAL: %.3f (ms) \n", 1.0e-3 * (timer_.optimize));
  printf("\n");

  // status
  printf("Status:\n");
  printf("  search iterations: %i\n", iterations_search_);
  printf("  smoother iterations: %i\n", iterations_smoother_);
  printf("  step size: %.6f\n", step_size_);
  printf("  regularization: %.6f\n", regularization_);
  printf("  gradient norm: %.6f\n", gradient_norm_);
  printf("  search direction norm: %.6f\n", search_direction_norm_);
  printf("  cost difference: %.6f\n", cost_difference_);
  printf("  solve status: %s\n", StatusString(solve_status_).c_str());
  printf("  cost count: %i\n", cost_count_);
  printf("\n");

  // cost
  printf("Cost:\n");
  printf("  final: %.3f\n", cost_);
  printf("    - prior: %.3f\n", cost_prior_);
  printf("    - sensor: %.3f\n", cost_sensor_);
  printf("    - force: %.3f\n", cost_force_);
  printf("  <initial: %.3f>\n", cost_initial_);
  printf("\n");

  fflush(stdout);
}

// print cost
void Batch::PrintCost() {
  if (settings.verbose_cost) {
    printf("cost (total): %.3f\n", cost_);
    printf("  prior: %.3f\n", cost_prior_);
    printf("  sensor: %.3f\n", cost_sensor_);
    printf("  force: %.3f\n", cost_force_);
    printf("  [initial: %.3f]\n", cost_initial_);
    fflush(stdout);
  }
}

// reset timers
void Batch::ResetTimers() {
  timer_.inverse_dynamics_derivatives = 0.0;
  timer_.velacc_derivatives = 0.0;
  timer_.jacobian_prior = 0.0;
  timer_.jacobian_sensor = 0.0;
  timer_.jacobian_force = 0.0;
  timer_.jacobian_total = 0.0;
  timer_.cost_prior_derivatives = 0.0;
  timer_.cost_sensor_derivatives = 0.0;
  timer_.cost_force_derivatives = 0.0;
  timer_.cost_total_derivatives = 0.0;
  timer_.cost_gradient = 0.0;
  timer_.cost_hessian = 0.0;
  timer_.cost_derivatives = 0.0;
  timer_.cost = 0.0;
  timer_.cost_prior = 0.0;
  timer_.cost_sensor = 0.0;
  timer_.cost_force = 0.0;
  timer_.cost_config_to_velacc = 0.0;
  timer_.cost_prediction = 0.0;
  timer_.residual_prior = 0.0;
  timer_.residual_sensor = 0.0;
  timer_.residual_force = 0.0;
  timer_.search_direction = 0.0;
  timer_.search = 0.0;
  timer_.configuration_update = 0.0;
  timer_.optimize = 0.0;
  timer_.prior_weight_update = 0.0;
  timer_.prior_set_weight = 0.0;
  timer_.update_trajectory = 0.0;
}

// batch status string
std::string StatusString(int code) {
  switch (code) {
    case kUnsolved:
      return "UNSOLVED";
    case kSearchFailure:
      return "SEACH_FAILURE";
    case kMaxIterationsFailure:
      return "MAX_ITERATIONS_FAILURE";
    case kSmallDirectionFailure:
      return "SMALL_DIRECTION_FAILURE";
    case kMaxRegularizationFailure:
      return "MAX_REGULARIZATION_FAILURE";
    case kCostDifferenceFailure:
      return "COST_DIFFERENCE_FAILURE";
    case kExpectedDecreaseFailure:
      return "EXPECTED_DECREASE_FAILURE";
    case kSolved:
      return "SOLVED";
    default:
      return "STATUS_CODE_ERROR";
  }
}

// estimator-specific GUI elements
void Batch::GUI(mjUI& ui, EstimatorGUIData& data) {
  // ----- estimator ------ //
  mjuiDef defEstimator[] = {
      {mjITEM_SECTION, "Estimator", 1, nullptr,
       "AP"},  // needs new section to satisfy mjMAXUIITEM
      {mjITEM_BUTTON, "Reset", 2, nullptr, ""},
      {mjITEM_SLIDERNUM, "Timestep", 2, &data.timestep, "1.0e-3 0.1"},
      {mjITEM_SELECT, "Integrator", 2, &data.integrator,
       "Euler\nRK4\nImplicit\nFastImplicit"},
      {mjITEM_SLIDERINT, "Horizon", 2, &data.horizon, "3 3"},
      {mjITEM_SLIDERNUM, "Prior Scale", 2, &data.scale_prior, "1.0e-8 0.1"},
      {mjITEM_END}};

  // set estimation horizon limits
  mju::sprintf_arr(defEstimator[4].other, "%i %i", kMinBatchHistory,
                   kMaxFilterHistory);

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
  for (int i = 0; i < nv; i++) {
    // element
    defProcessNoise[process_noise_shift] = {
        mjITEM_SLIDERNUM, "", 2, data.process_noise.data() + i, "1.0e-8 0.01"};

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
        // velocity
        jnt_name_vel = jnt_name + " (0)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 0].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (1)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 1].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (2)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 2].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (3)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 3].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (4)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 4].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (5)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 5].name,
                        jnt_name_vel.c_str());

        // shift
        jnt_shift += 6;
        break;
      case mjJNT_BALL:
        // velocity
        jnt_name_vel = jnt_name + " (0)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 0].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (1)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 1].name,
                        jnt_name_vel.c_str());

        jnt_name_vel = jnt_name + " (2)";
        mju::strcpy_arr(defProcessNoise[jnt_shift + 2].name,
                        jnt_name_vel.c_str());

        // shift
        jnt_shift += 3;
        break;
      case mjJNT_HINGE:
        // velocity
        jnt_name_vel = jnt_name;
        mju::strcpy_arr(defProcessNoise[jnt_shift].name, jnt_name_vel.c_str());

        // shift
        jnt_shift++;
        break;
      case mjJNT_SLIDE:
        // velocity
        jnt_name_vel = jnt_name;
        mju::strcpy_arr(defProcessNoise[jnt_shift].name, jnt_name_vel.c_str());

        // shift
        jnt_shift++;
        break;
    }
  }

  // loop over act
  // std::string act_str;
  // for (int i = 0; i < model->na; i++) {
  //   act_str = "act (" + std::to_string(i) + ")";
  //   mju::strcpy_arr(defProcessNoise[nv + jnt_shift + i].name,
  //   act_str.c_str());
  // }

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

    // element
    defSensorNoise[sensor_noise_shift] = {
        mjITEM_SLIDERNUM, "", 2,
        data.sensor_noise.data() + sensor_noise_shift - 1, "1.0e-8 0.01"};

    // sensor name
    sensor_str = name_sensor;

    // set sensor name
    mju::strcpy_arr(defSensorNoise[sensor_noise_shift].name,
                    sensor_str.c_str());

    // shift
    sensor_noise_shift++;
  }

  // end
  defSensorNoise[sensor_noise_shift] = {mjITEM_END};

  // add sensor noise
  mjui_add(&ui, defSensorNoise);
}

// set GUI data
void Batch::SetGUIData(EstimatorGUIData& data) {
  mju_copy(noise_process.data(), data.process_noise.data(), DimensionProcess());
  mju_copy(noise_sensor.data(), data.sensor_noise.data(), DimensionSensor());
  model->opt.timestep = data.timestep;
  model->opt.integrator = data.integrator;

  // store estimation horizon
  int horizon = data.horizon;

  // changing horizon cases
  if (horizon > configuration_length_) {  // increase horizon
    // -- prior weights resize -- //
    int nvar = model->nv * configuration_length_;
    int nvar_new = model->nv * horizon;

    // get previous weights
    double* weights = weight_prior_.data();
    double* previous_weights = scratch0_condmat_.data();
    mju_copy(previous_weights, weights, nvar * nvar);

    // set previous in new weights (dimension may have increased)
    mju_zero(weights, nvar_new * nvar_new);
    SetBlockInMatrix(weights, previous_weights, 1.0, nvar_new, nvar_new, nvar,
                     nvar, 0, 0);

    // scale_prior * I
    for (int i = nvar; i < nvar_new; i++) {
      weights[nvar_new * i + i] = scale_prior;
    }

    // modify trajectories
    ShiftResizeTrajectory(0, horizon);

    // update configuration length
    configuration_length_ = horizon;
  } else if (horizon < configuration_length_) {  // decrease horizon
    // -- prior weights resize -- //
    int nvar = model->nv * configuration_length_;
    int nvar_new = model->nv * horizon;

    // get previous weights
    double* weights = weight_prior_.data();
    double* previous_weights = scratch0_condmat_.data();
    BlockFromMatrix(previous_weights, weights, nvar_new, nvar_new, nvar, nvar,
                    0, 0);

    // set previous in new weights (dimension may have increased)
    mju_zero(weights, nvar * nvar);
    mju_copy(weights, previous_weights, nvar_new * nvar_new);

    // compute difference in estimation horizons
    int horizon_diff = configuration_length_ - horizon;

    // modify trajectories
    ShiftResizeTrajectory(horizon_diff, horizon);

    // update configuration length and current time index
    configuration_length_ = horizon;
    current_time_index_ -= horizon_diff;
  }
}

// estimator-specific plots
void Batch::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                  int planner_shift, int timer_shift, int planning,
                  int* shift) {
  // Batch info

  // TODO(taylor): covariance trace
  // double estimator_bounds[2] = {-6, 6};

  // // covariance trace
  // double trace = Trace(covariance.data(), DimensionProcess());
  // mjpc::PlotUpdateData(fig_planner, estimator_bounds,
  //                      fig_planner->linedata[planner_shift + 0][0] + 1,
  //                      mju_log10(trace), 100, planner_shift + 0, 0, 1, -100);

  // // legend
  // mju::strcpy_arr(fig_planner->linename[planner_shift + 0], "Covariance
  // Trace");

  // Batch timers
  double timer_bounds[2] = {0.0, 1.0};

  // update
  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[timer_shift + 0][0] + 1, timer_.update,
                 100, timer_shift + 0, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[timer_shift + 0], "Update");
}

// increase regularization
void Batch::IncreaseRegularization() {
  regularization_ = mju_min(kMaxBatchRegularization,
                            regularization_ * settings.regularization_scaling);
}

// project configurations onto joint limits
void Batch::JointLimits() {
  // loop over configurations
  for (int t = 0; t < configuration_length_; t++) {
    // configuration
    double* qt = configuration.Get(t);

    // loop over joints
    for (int i = 0; i < model->njnt; i++) {
      // check for limits
      if (model->jnt_limited[i]) {
        // get joint element
        double* qti = qt + model->jnt_qposadr[i];

        // clip
        qti[0] = mju_clip(qti[0], model->jnt_range[2 * i],
                          model->jnt_range[2 * i + 1]);
      }
    }
  }
}

}  // namespace mjpc
