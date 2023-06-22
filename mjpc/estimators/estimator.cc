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

#include "mjpc/estimators/estimator.h"

#include <chrono>

#include "mjpc/estimators/buffer.h"
#include "mjpc/norm.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

// initialize estimator
void Estimator::Initialize(mjModel* model) {
  // model
  model_ = model;

  // data
  for (int i = 0; i < MAX_HISTORY; i++) {
    data_.push_back(MakeUniqueMjData(mj_makeData(model)));
  }

  // dimension
  int nq = model->nq, nv = model->nv;

  // length of configuration trajectory
  configuration_length_ =
      GetNumberOrDefault(32, model, "estimator_configuration_length");

  // number of predictions
  prediction_length_ = configuration_length_ - 2;

  // -- trajectories -- //
  configuration_.Initialize(nq, configuration_length_);
  velocity_.Initialize(nv, configuration_length_);
  acceleration_.Initialize(nv, configuration_length_);
  time_.Initialize(1, configuration_length_);

  // ctrl 
  ctrl_.Initialize(model->nu, configuration_length_);

  // prior
  configuration_prior_.Initialize(nq, configuration_length_);

  // sensor
  dim_sensor_ = model->nsensordata;  // TODO(taylor): grab from xml
  num_sensor_ = model->nsensor;      // TODO(taylor): grab from xml
  sensor_measurement_.Initialize(dim_sensor_, configuration_length_);
  sensor_prediction_.Initialize(dim_sensor_, configuration_length_);
  sensor_mask_.Initialize(num_sensor_, configuration_length_);

  // free joint dof flag
  free_dof_.resize(model->nv);
  std::fill(free_dof_.begin(), free_dof_.end(), false);

  // number of free joints
  num_free_ = 0;
  for (int i = 0; i < model->njnt; i++) {
    if (model->jnt_type[i] == mjJNT_FREE) {
      num_free_++;
      int adr = model->jnt_dofadr[i];
      std::fill(free_dof_.begin() + adr, free_dof_.begin() + adr + 6, true);
    }
  }

  // force
  force_measurement_.Initialize(nv, configuration_length_);
  force_prediction_.Initialize(nv, configuration_length_);

  // residual
  residual_prior_.resize(nv * MAX_HISTORY);
  residual_sensor_.resize(dim_sensor_ * MAX_HISTORY);
  residual_force_.resize(nv * MAX_HISTORY);

  // Jacobian
  jacobian_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  jacobian_sensor_.resize((dim_sensor_ * MAX_HISTORY) * (nv * MAX_HISTORY));
  jacobian_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // prior Jacobian block
  block_prior_current_configuration_.Initialize(nv * nv, configuration_length_);

  // sensor Jacobian blocks
  block_sensor_configuration_.Initialize(dim_sensor_ * nv, prediction_length_);
  block_sensor_velocity_.Initialize(dim_sensor_ * nv, prediction_length_);
  block_sensor_acceleration_.Initialize(dim_sensor_ * nv, prediction_length_);

  block_sensor_previous_configuration_.Initialize(dim_sensor_ * nv,
                                                  prediction_length_);
  block_sensor_current_configuration_.Initialize(dim_sensor_ * nv,
                                                 prediction_length_);
  block_sensor_next_configuration_.Initialize(dim_sensor_ * nv,
                                              prediction_length_);
  block_sensor_configurations_.Initialize(dim_sensor_ * 3 * nv,
                                          prediction_length_);

  block_sensor_scratch_.Initialize(
      mju_max(nv, dim_sensor_) * mju_max(nv, dim_sensor_), prediction_length_);

  // force Jacobian blocks
  block_force_configuration_.Initialize(nv * nv, prediction_length_);
  block_force_velocity_.Initialize(nv * nv, prediction_length_);
  block_force_acceleration_.Initialize(nv * nv, prediction_length_);

  block_force_previous_configuration_.Initialize(nv * nv, prediction_length_);
  block_force_current_configuration_.Initialize(nv * nv, prediction_length_);
  block_force_next_configuration_.Initialize(nv * nv, prediction_length_);
  block_force_configurations_.Initialize(nv * 3 * nv, prediction_length_);

  block_force_scratch_.Initialize(nv * nv, prediction_length_);

  // velocity Jacobian blocks
  block_velocity_previous_configuration_.Initialize(nv * nv,
                                                    configuration_length_ - 1);
  block_velocity_current_configuration_.Initialize(nv * nv,
                                                   configuration_length_ - 1);

  // acceleration Jacobian blocks
  block_acceleration_previous_configuration_.Initialize(nv * nv,
                                                        prediction_length_);
  block_acceleration_current_configuration_.Initialize(nv * nv,
                                                       prediction_length_);
  block_acceleration_next_configuration_.Initialize(nv * nv,
                                                    prediction_length_);

  // cost gradient
  cost_gradient_prior_.resize(nv * MAX_HISTORY);
  cost_gradient_sensor_.resize(nv * MAX_HISTORY);
  cost_gradient_force_.resize(nv * MAX_HISTORY);
  cost_gradient_.resize(nv * MAX_HISTORY);

  // cost Hessian
  cost_hessian_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_sensor_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_band_.resize(BandMatrixNonZeros(nv * MAX_HISTORY, 3 * nv));
  cost_hessian_factor_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // prior weights
  scale_prior_ = GetNumberOrDefault(1.0, model, "estimator_scale_prior");
  weight_prior_dense_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  weight_prior_band_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch_prior_weight_.resize(2 * nv * nv);

  // sensor scale
  // TODO(taylor): only grab measurement sensors
  scale_sensor_.resize(num_sensor_);

  // TODO(taylor): method for xml to initial weight
  for (int i = 0; i < num_sensor_; i++) {
    scale_sensor_[i] = GetNumberOrDefault(1.0, model, "estimator_scale_sensor");
  }

  // force scale
  scale_force_.resize(NUM_FORCE_TERMS);

  scale_force_[0] =
      GetNumberOrDefault(1.0, model, "estimator_scale_force_free_position");
  scale_force_[1] =
      GetNumberOrDefault(1.0, model, "estimator_scale_force_free_rotation");
  scale_force_[2] =
      GetNumberOrDefault(1.0, model, "estimator_scale_force_nonfree");

  // cost norms
  // TODO(taylor): only grab measurement sensors
  norm_sensor_.resize(num_sensor_);

  // TODO(taylor): method for xml to initial norm
  for (int i = 0; i < num_sensor_; i++) {
    norm_sensor_[i] =
        (NormType)GetNumberOrDefault(0, model, "estimator_norm_sensor");
  }

  norm_force_[0] =
      (NormType)GetNumberOrDefault(0, model, "estimator_norm_force_free_position");
  norm_force_[1] =
      (NormType)GetNumberOrDefault(0, model, "estimator_norm_force_free_rotation");
  norm_force_[2] =
      (NormType)GetNumberOrDefault(0, model, "estimator_norm_force_nonfree");

  // cost norm parameters
  norm_parameters_sensor_.resize(num_sensor_ * MAX_NORM_PARAMETERS);
  norm_parameters_force_.resize(NUM_FORCE_TERMS * MAX_NORM_PARAMETERS);

  // TODO(taylor): initialize norm parameters from xml
  std::fill(norm_parameters_sensor_.begin(), norm_parameters_sensor_.end(),
            0.0);
  std::fill(norm_parameters_force_.begin(), norm_parameters_force_.end(), 0.0);

  // norm gradient
  norm_gradient_sensor_.resize(dim_sensor_ * MAX_HISTORY);
  norm_gradient_force_.resize(nv * MAX_HISTORY);

  // norm Hessian
  norm_hessian_sensor_.resize(dim_sensor_ * dim_sensor_ * MAX_HISTORY);
  norm_hessian_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  norm_blocks_sensor_.resize(dim_sensor_ * dim_sensor_ * MAX_HISTORY);
  norm_blocks_force_.resize(nv * nv * MAX_HISTORY);

  // scratch
  scratch0_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch1_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  scratch0_sensor_.resize(mju_max(nv, dim_sensor_) * mju_max(nv, dim_sensor_) *
                          MAX_HISTORY);
  scratch1_sensor_.resize(mju_max(nv, dim_sensor_) * mju_max(nv, dim_sensor_) *
                          MAX_HISTORY);

  scratch0_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch1_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch2_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // copy
  configuration_copy_.Initialize(nq, configuration_length_);

  // search direction
  search_direction_.resize(nv * MAX_HISTORY);

  // regularization
  regularization_ = regularization_initial_;

  // search type
  search_type_ =
      (SearchType)GetNumberOrDefault(0, model, "estimator_search_type");

  // initial state 
  qpos0_.resize(model->nq);
  mju_copy(qpos0_.data(), model->qpos0, model->nq);
  qvel0_.resize(model->nv);
  mju_zero(qvel0_.data(), model->nv);

  // timer
  timer_prior_step_.resize(MAX_HISTORY);
  timer_sensor_step_.resize(MAX_HISTORY);
  timer_force_step_.resize(MAX_HISTORY);

  // status
  hessian_factor_ = false;
  num_new_ = configuration_length_;
  gradient_norm_ = 0.0;

  // state index
  state_index_ = configuration_length_ - 1;

  // settings
  band_covariance_ =
      (bool)GetNumberOrDefault(1, model, "estimator_band_covariance");

  // reset
  Reset();
}

// set configuration length
void Estimator::SetConfigurationLength(int length) {
  // set configuration length
  configuration_length_ = mju_max(length, MIN_HISTORY);

  // set prediction length
  prediction_length_ = configuration_length_ - 2;

  // update trajectory lengths
  configuration_.length_ = length;
  configuration_copy_.length_ = length;

  velocity_.length_ = length;
  acceleration_.length_ = length;
  time_.length_ = length;

  ctrl_.length_ = length;

  configuration_prior_.length_ = length;

  sensor_measurement_.length_ = length;
  sensor_prediction_.length_ = length;
  sensor_mask_.length_ = length;

  force_measurement_.length_ = length;
  force_prediction_.length_ = length;

  block_prior_current_configuration_.length_ = length;

  block_sensor_configuration_.length_ = prediction_length_;
  block_sensor_velocity_.length_ = prediction_length_;
  block_sensor_acceleration_.length_ = prediction_length_;

  block_sensor_previous_configuration_.length_ = prediction_length_;
  block_sensor_current_configuration_.length_ = prediction_length_;
  block_sensor_next_configuration_.length_ = prediction_length_;
  block_sensor_configurations_.length_ = prediction_length_;

  block_sensor_scratch_.length_ = prediction_length_;

  block_force_configuration_.length_ = prediction_length_;
  block_force_velocity_.length_ = prediction_length_;
  block_force_acceleration_.length_ = prediction_length_;

  block_force_previous_configuration_.length_ = prediction_length_;
  block_force_current_configuration_.length_ = prediction_length_;
  block_force_next_configuration_.length_ = prediction_length_;
  block_force_configurations_.length_ = prediction_length_;

  block_force_scratch_.length_ = prediction_length_;

  block_velocity_previous_configuration_.length_ = length - 1;
  block_velocity_current_configuration_.length_ = length - 1;

  block_acceleration_previous_configuration_.length_ = prediction_length_;
  block_acceleration_current_configuration_.length_ = prediction_length_;
  block_acceleration_next_configuration_.length_ = prediction_length_;

  // state index
  state_index_ = mju_max(1, mju_min(state_index_, configuration_length_ - 1));

  // status
  num_new_ = configuration_length_;
  initialized_ = false;
  step_size_ = 1.0;
  gradient_norm_ = 0.0;
}

// shift trajectory heads
void Estimator::ShiftTrajectoryHead(int shift) {
  // update trajectory lengths
  configuration_.ShiftHeadIndex(shift);
  configuration_copy_.ShiftHeadIndex(shift);

  velocity_.ShiftHeadIndex(shift);
  acceleration_.ShiftHeadIndex(shift);
  time_.ShiftHeadIndex(shift);

  ctrl_.ShiftHeadIndex(shift);

  configuration_prior_.ShiftHeadIndex(shift);

  sensor_measurement_.ShiftHeadIndex(shift);
  sensor_prediction_.ShiftHeadIndex(shift);
  sensor_mask_.ShiftHeadIndex(shift);

  force_measurement_.ShiftHeadIndex(shift);
  force_prediction_.ShiftHeadIndex(shift);

  block_prior_current_configuration_.ShiftHeadIndex(shift);

  block_sensor_configuration_.ShiftHeadIndex(shift);
  block_sensor_velocity_.ShiftHeadIndex(shift);
  block_sensor_acceleration_.ShiftHeadIndex(shift);

  block_sensor_previous_configuration_.ShiftHeadIndex(shift);
  block_sensor_current_configuration_.ShiftHeadIndex(shift);
  block_sensor_next_configuration_.ShiftHeadIndex(shift);
  block_sensor_configurations_.ShiftHeadIndex(shift);

  block_sensor_scratch_.ShiftHeadIndex(shift);

  block_force_configuration_.ShiftHeadIndex(shift);
  block_force_velocity_.ShiftHeadIndex(shift);
  block_force_acceleration_.ShiftHeadIndex(shift);

  block_force_previous_configuration_.ShiftHeadIndex(shift);
  block_force_current_configuration_.ShiftHeadIndex(shift);
  block_force_next_configuration_.ShiftHeadIndex(shift);
  block_force_configurations_.ShiftHeadIndex(shift);

  block_force_scratch_.ShiftHeadIndex(shift);

  block_velocity_previous_configuration_.ShiftHeadIndex(shift);
  block_velocity_current_configuration_.ShiftHeadIndex(shift);

  block_acceleration_previous_configuration_.ShiftHeadIndex(shift);
  block_acceleration_current_configuration_.ShiftHeadIndex(shift);
  block_acceleration_next_configuration_.ShiftHeadIndex(shift);
}

// reset memory
void Estimator::Reset() {
  // trajectories
  configuration_.Reset();
  velocity_.Reset();
  acceleration_.Reset();
  time_.Reset();

  ctrl_.Reset();

  // prior
  configuration_prior_.Reset();

  // sensor
  sensor_measurement_.Reset();
  sensor_prediction_.Reset();

  // sensor mask
  sensor_mask_.Reset();
  std::fill(sensor_mask_.data_.begin(), sensor_mask_.data_.end(), 1);

  // force
  force_measurement_.Reset();
  force_prediction_.Reset();

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

  // cost gradient
  std::fill(cost_gradient_prior_.begin(), cost_gradient_prior_.end(), 0.0);
  std::fill(cost_gradient_sensor_.begin(), cost_gradient_sensor_.end(), 0.0);
  std::fill(cost_gradient_force_.begin(), cost_gradient_force_.end(), 0.0);
  std::fill(cost_gradient_.begin(), cost_gradient_.end(), 0.0);

  // cost Hessian
  std::fill(cost_hessian_prior_.begin(), cost_hessian_prior_.end(), 0.0);
  std::fill(cost_hessian_sensor_.begin(), cost_hessian_sensor_.end(), 0.0);
  std::fill(cost_hessian_force_.begin(), cost_hessian_force_.end(), 0.0);
  std::fill(cost_hessian_.begin(), cost_hessian_.end(), 0.0);
  std::fill(cost_hessian_band_.begin(), cost_hessian_band_.end(), 0.0);
  std::fill(cost_hessian_factor_.begin(), cost_hessian_factor_.end(), 0.0);

  // weight
  std::fill(weight_prior_dense_.begin(), weight_prior_dense_.end(), 0.0);
  std::fill(weight_prior_band_.begin(), weight_prior_band_.end(), 0.0);
  std::fill(scratch_prior_weight_.begin(), scratch_prior_weight_.end(), 0.0);

  // norm gradient
  std::fill(norm_gradient_sensor_.begin(), norm_gradient_sensor_.end(), 0.0);
  std::fill(norm_gradient_force_.begin(), norm_gradient_force_.end(), 0.0);

  // norm Hessian
  std::fill(norm_hessian_sensor_.begin(), norm_hessian_sensor_.end(), 0.0);
  std::fill(norm_hessian_force_.begin(), norm_hessian_force_.end(), 0.0);

  std::fill(norm_blocks_sensor_.begin(), norm_blocks_sensor_.end(), 0.0);
  std::fill(norm_blocks_force_.begin(), norm_blocks_force_.end(), 0.0);

  // scratch
  std::fill(scratch0_prior_.begin(), scratch0_prior_.end(), 0.0);
  std::fill(scratch1_prior_.begin(), scratch1_prior_.end(), 0.0);

  std::fill(scratch0_sensor_.begin(), scratch0_sensor_.end(), 0.0);
  std::fill(scratch1_sensor_.begin(), scratch1_sensor_.end(), 0.0);

  std::fill(scratch0_force_.begin(), scratch0_force_.end(), 0.0);
  std::fill(scratch1_force_.begin(), scratch1_force_.end(), 0.0);
  std::fill(scratch2_force_.begin(), scratch2_force_.end(), 0.0);

  // candidate
  configuration_copy_.Reset();

  // search direction
  std::fill(search_direction_.begin(), search_direction_.end(), 0.0);

  // initial state 
  mju_copy(qpos0_.data(), model_->qpos0, model_->nq);
  mju_zero(qvel0_.data(), model_->nv);

  // timer
  std::fill(timer_prior_step_.begin(), timer_prior_step_.end(), 0.0);
  std::fill(timer_sensor_step_.begin(), timer_sensor_step_.end(), 0.0);
  std::fill(timer_force_step_.begin(), timer_force_step_.end(), 0.0);

  // timing
  ResetTimers();

  // status
  iterations_smoother_ = 0;
  iterations_line_search_ = 0;
  cost_count_ = 0;
  initialized_ = false;
}

// prior cost
double Estimator::CostPrior(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // residual dimension
  int nv = model_->nv;
  int dim = model_->nv * configuration_length_;

  // total scaling
  double scale = scale_prior_ / dim;

  // unpack
  double* r = residual_prior_.data();
  double* P = (band_covariance_ ? weight_prior_band_.data()
                                : weight_prior_dense_.data());
  double* tmp = scratch0_prior_.data();

  // compute cost
  if (band_covariance_) {  // approximate covariance
    // dimensions
    int ntotal = dim;
    int nband = 3 * model_->nv;
    int ndense = 0;

    // multiply: tmp = P * r
    mju_bandMulMatVec(tmp, P, r, ntotal, nband, ndense, 1, true);
  } else {  // exact covariance
    // multiply: tmp = P * r
    mju_mulMatVec(tmp, P, r, dim, dim);
  }

  // weighted quadratic: 0.5 * w * r' * tmp
  double cost = 0.5 * scale * mju_dot(r, tmp, dim);

  // stop cost timer
  timer_cost_prior_ += GetDuration(start);

  // derivatives
  if (!gradient && !hessian) return cost;

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
    if (hessian && band_covariance_) {
      // number of columns to loop over for row
      int num_cols = mju_min(3, configuration_length_ - t);

      for (int j = t; j < t + num_cols; j++) {
        // shift index
        int shift = 0;  // shift_index(i, j);

        // unpack
        double* bbij =
            scratch1_prior_.data() + 4 * nv * nv * shift + 0 * nv * nv;
        double* tmp0 =
            scratch1_prior_.data() + 4 * nv * nv * shift + 1 * nv * nv;
        double* tmp1 =
            scratch1_prior_.data() + 4 * nv * nv * shift + 2 * nv * nv;
        double* tmp2 =
            scratch1_prior_.data() + 4 * nv * nv * shift + 3 * nv * nv;

        // get matrices
        BlockFromMatrix(bbij, weight_prior_dense_.data(), nv, nv, dim, dim,
                        t * nv, j * nv);
        const double* bdi = block_prior_current_configuration_.Get(t);
        const double* bdj = block_prior_current_configuration_.Get(j);

        // -- bdi' * bbij * bdj -- //

        // tmp0 = bbij * bdj
        mju_mulMatMat(tmp0, bbij, bdj, nv, nv, nv);

        // tmp1 = bdi' * tmp0
        mju_mulMatTMat(tmp1, bdi, tmp0, nv, nv, nv);

        // set scaled block in matrix
        SetBlockInMatrix(hessian, tmp1, scale, dim, dim, nv, nv, t * nv,
                         j * nv);
        if (j > t) {
          mju_transpose(tmp2, tmp1, nv, nv);
          SetBlockInMatrix(hessian, tmp2, scale, dim, dim, nv, nv, j * nv,
                           t * nv);
        }
      }
    }
  }

  // serial method for dense computation
  if (hessian && !band_covariance_) {
    // unpack
    double* J = jacobian_prior_.data();

    // multiply: scratch = P * drdq
    mju_mulMatMat(tmp, P, J, dim, dim, dim);

    // step 2: hessian = drdq' * scratch
    mju_mulMatTMat(hessian, J, tmp, dim, dim, dim);

    // step 3: scale
    mju_scl(hessian, hessian, scale, dim * dim);
  }

  // stop derivatives timer
  timer_cost_prior_derivatives_ += GetDuration(start);

  return cost;
}

// prior residual
void Estimator::ResidualPrior() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model_->nv;

  // loop over configurations
  for (int t = 0; t < configuration_length_; t++) {
    // terms
    double* rt = residual_prior_.data() + t * nv;
    double* qt_prior = configuration_prior_.Get(t);
    double* qt = configuration_.Get(t);

    // configuration difference
    mj_differentiatePos(model_, rt, 1.0, qt_prior, qt);
  }

  // stop timer
  timer_residual_prior_ += GetDuration(start);
}

// set block in prior Jacobian
void Estimator::SetBlockPrior(int index) {
  // dimension
  int nv = model_->nv, dim = model_->nv * configuration_length_;

  // reset Jacobian to zero
  mju_zero(jacobian_prior_.data() + index * nv * dim, nv * dim);

  // unpack
  double* block = block_prior_current_configuration_.Get(index);

  // set block in matrix
  SetBlockInMatrix(jacobian_prior_.data(), block, 1.0, dim, dim, nv, nv,
                   index * nv, index * nv);
}

// prior Jacobian blocks
void Estimator::BlockPrior(int index) {
  // unpack
  double* qt = configuration_.Get(index);
  double* qt_prior = configuration_prior_.Get(index);
  double* block = block_prior_current_configuration_.Get(index);

  // compute Jacobian
  DifferentiateDifferentiatePos(NULL, block, model_, 1.0, qt_prior, qt);
}

// prior Jacobian
// note: pool wait is called outside this function
void Estimator::JacobianPrior(ThreadPool& pool) {
  // start index
  int start_index = reuse_data_ * mju_max(0, configuration_length_ - num_new_);

  // loop over predictions
  for (int t = 0; t < configuration_length_; t++) {
    // schedule by time step
    pool.Schedule([&estimator = *this, start_index, t]() {
      // start Jacobian timer
      auto jacobian_prior_start = std::chrono::steady_clock::now();

      // block
      if (t >= start_index) estimator.BlockPrior(t);

      // assemble
      if (!estimator.band_covariance_) estimator.SetBlockPrior(t);

      // stop Jacobian timer
      estimator.timer_prior_step_[t] = GetDuration(jacobian_prior_start);
    });
  }
}

// sensor cost
double Estimator::CostSensor(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // update dimension
  int dim_update = model_->nv * configuration_length_;
  int nv = model_->nv;

  // ----- cost ----- //

  // initialize
  double cost = 0.0;
  int shift = 0;
  int shift_mat = 0;

  // zero memory
  if (gradient) mju_zero(gradient, dim_update);
  if (hessian) mju_zero(hessian, dim_update * dim_update);

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // time index
    int t = k + 1;

    // mask
    int* mask = sensor_mask_.Get(t);

    // unpack block
    double* block = block_sensor_configurations_.Get(k);

    // sensor shift
    int shift_sensor = 0;

    // loop over sensors
    for (int i = 0; i < num_sensor_; i++) {
      // start cost timer
      auto start_cost = std::chrono::steady_clock::now();

      // check mask, skip if missing measurement
      if (!mask[i]) continue;

      // dimension
      int nsi = model_->sensor_dim[i];

      // weight
      double weight = scale_sensor_[i];

      // time scaling, accounts for finite difference division by timestep
      double time_scale = 1.0;

      if (time_scaling_) {
        // stage
        int stage = model_->sensor_needstage[i];

        // time step
        double timestep = model_->opt.timestep;

        // scale by sensor type
        if (stage == mjSTAGE_VEL) {
          time_scale = timestep * timestep;
        } else if (stage == mjSTAGE_ACC) {
          time_scale = timestep * timestep * timestep * timestep;
        }
      }

      // total scaling
      double scale = weight / nsi * time_scale / (configuration_length_ - 2);

      // ----- cost ----- //
      cost +=
          scale * Norm(gradient ? norm_gradient_sensor_.data() + shift : NULL,
                       hessian ? norm_blocks_sensor_.data() + shift_mat : NULL,
                       residual_sensor_.data() + shift,
                       norm_parameters_sensor_.data() + MAX_NORM_PARAMETERS * i,
                       nsi, norm_sensor_[i]);

      // stop cost timer
      timer_cost_sensor_ += GetDuration(start_cost);

      // gradient wrt configuration: dridq012' * dndri
      if (gradient) {
        // sensor block
        double* blocki = block + (3 * nv) * shift_sensor;

        // scratch = dridq012' * dndri
        mju_mulMatTVec(scratch0_sensor_.data(), blocki,
                       norm_gradient_sensor_.data() + shift, nsi, 3 * nv);

        // add
        mju_addToScl(gradient + k * nv, scratch0_sensor_.data(), scale, 3 * nv);
      }

      // Hessian (Gauss-Newton): drdq' * d2ndr2 * drdq
      if (hessian) {
        // sensor block
        double* blocki = block + (3 * nv) * shift_sensor;

        // step 1: tmp0 = d2ndri2 * dridq
        double* tmp0 = scratch0_sensor_.data();
        mju_mulMatMat(tmp0, norm_blocks_sensor_.data() + shift_mat, blocki, nsi,
                      nsi, 3 * nv);

        // step 2: hessian = dridq' * tmp
        double* tmp1 = scratch1_sensor_.data();
        mju_mulMatTMat(tmp1, blocki, tmp0, nsi, 3 * nv, 3 * nv);

        // add
        AddBlockInMatrix(hessian, tmp1, scale, dim_update, dim_update, 3 * nv,
                         3 * nv, nv * k, nv * k);
      }

      // shift
      shift += nsi;
      shift_mat += nsi * nsi;
      shift_sensor += nsi;
    }
  }

  // stop timer
  timer_cost_sensor_derivatives_ += GetDuration(start);

  return cost;
}

// sensor residual
void Estimator::ResidualSensor() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // time index
    int t = k + 1;

    // terms
    double* rk = residual_sensor_.data() + k * dim_sensor_;
    double* yt_sensor = sensor_measurement_.Get(t);
    double* yt_model = sensor_prediction_.Get(t);

    // sensor difference
    mju_sub(rk, yt_model, yt_sensor, dim_sensor_);
  }

  // stop timer
  timer_residual_sensor_ += GetDuration(start);
}

// set block in sensor Jacobian
void Estimator::SetBlockSensor(int index) {
  // velocity dimension
  int nv = model_->nv, ns = dim_sensor_;

  // residual dimension
  int dim_residual = ns * prediction_length_;

  // update dimension
  int dim_update = nv * configuration_length_;

  // reset Jacobian to zero
  mju_zero(jacobian_sensor_.data() + index * ns * dim_update, ns * dim_update);

  // indices
  int row = index * ns;
  int col_previous = index * nv;
  int col_current = (index + 1) * nv;
  int col_next = (index + 2) * nv;

  // ----- configuration previous ----- //

  // unpack
  double* dsdq0 = block_sensor_previous_configuration_.Get(index);

  // set
  SetBlockInMatrix(jacobian_sensor_.data(), dsdq0, 1.0, dim_residual,
                   dim_update, dim_sensor_, nv, row, col_previous);

  // ----- configuration current ----- //

  // unpack
  double* dsdq1 = block_sensor_current_configuration_.Get(index);

  // set
  SetBlockInMatrix(jacobian_sensor_.data(), dsdq1, 1.0, dim_residual,
                   dim_update, dim_sensor_, nv, row, col_current);

  // ----- configuration next ----- //

  // unpack
  double* dsdq2 = block_sensor_next_configuration_.Get(index);

  // set
  SetBlockInMatrix(jacobian_sensor_.data(), dsdq2, 1.0, dim_residual,
                   dim_update, dim_sensor_, nv, row, col_next);
}

// sensor Jacobian blocks (dsdq0, dsdq1, dsdq2)
void Estimator::BlockSensor(int index) {
  // dimensions
  int nv = model_->nv, ns = dim_sensor_;

  // dqds
  double* dqds = block_sensor_configuration_.Get(index);

  // dvds
  double* dvds = block_sensor_velocity_.Get(index);

  // dads
  double* dads = block_sensor_acceleration_.Get(index);

  // -- configuration previous: dsdq0 = dsdv * dvdq0 + dsda * dadq0 -- //

  // unpack
  double* dsdq0 = block_sensor_previous_configuration_.Get(index);
  double* tmp = block_sensor_scratch_.Get(index);

  // dsdq0 <- dvds' * dvdq0
  double* dvdq0 = block_velocity_previous_configuration_.Get(index);
  mju_mulMatTMat(dsdq0, dvds, dvdq0, nv, ns, nv);

  // dsdq0 += dads' * dadq0
  double* dadq0 = block_acceleration_previous_configuration_.Get(index);
  mju_mulMatTMat(tmp, dads, dadq0, nv, ns, nv);
  mju_addTo(dsdq0, tmp, ns * nv);

  // -- configuration current: dsdq1 = dsdq + dsdv * dvdq1 + dsda * dadq1 --

  // unpack
  double* dsdq1 = block_sensor_current_configuration_.Get(index);

  // dsdq1 <- dqds'
  mju_transpose(dsdq1, dqds, nv, ns);

  // dsdq1 += dvds' * dvdq1
  double* dvdq1 = block_velocity_current_configuration_.Get(index);
  mju_mulMatTMat(tmp, dvds, dvdq1, nv, ns, nv);
  mju_addTo(dsdq1, tmp, ns * nv);

  // dsdq1 += dads' * dadq1
  double* dadq1 = block_acceleration_current_configuration_.Get(index);
  mju_mulMatTMat(tmp, dads, dadq1, nv, ns, nv);
  mju_addTo(dsdq1, tmp, ns * nv);

  // -- configuration next: dsdq2 = dsda * dadq2 -- //

  // unpack
  double* dsdq2 = block_sensor_next_configuration_.Get(index);

  // dsdq2 = dads' * dadq2
  double* dadq2 = block_acceleration_next_configuration_.Get(index);
  mju_mulMatTMat(dsdq2, dads, dadq2, nv, ns, nv);

  // -- assemble dsdq012 block -- //

  // unpack
  double* dsdq012 = block_sensor_configurations_.Get(index);

  // set dfdq0
  SetBlockInMatrix(dsdq012, dsdq0, 1.0, ns, 3 * nv, ns, nv, 0, 0 * nv);

  // set dfdq1
  SetBlockInMatrix(dsdq012, dsdq1, 1.0, ns, 3 * nv, ns, nv, 0, 1 * nv);

  // set dfdq0
  SetBlockInMatrix(dsdq012, dsdq2, 1.0, ns, 3 * nv, ns, nv, 0, 2 * nv);
}

// sensor Jacobian
// note: pool wait is called outside this function
void Estimator::JacobianSensor(ThreadPool& pool) {
  // start index
  int start_index = reuse_data_ * mju_max(0, prediction_length_ - num_new_);

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // schedule by time step
    pool.Schedule([&estimator = *this, start_index, k]() {
      // start Jacobian timer
      auto jacobian_sensor_start = std::chrono::steady_clock::now();

      // block
      if (k >= start_index) estimator.BlockSensor(k);

      // assemble
      if (!estimator.band_covariance_) estimator.SetBlockSensor(k);

      // stop Jacobian timer
      estimator.timer_sensor_step_[k] = GetDuration(jacobian_sensor_start);
    });
  }
}

// force cost
double Estimator::CostForce(double* gradient, double* hessian) {
  // start derivative timer
  auto start = std::chrono::steady_clock::now();

  // update dimension
  int dim_update = model_->nv * configuration_length_;
  int nv = model_->nv;

  // time scaling
  double timestep = model_->opt.timestep;
  double time_scale =
      (time_scaling_ ? timestep * timestep * timestep * timestep : 1.0);

  // initialize
  double cost = 0.0;

  // zero memory
  if (gradient) mju_zero(gradient, dim_update);
  if (hessian) mju_zero(hessian, dim_update * dim_update);

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // unpack residual
    double* residual = residual_force_.data() + nv * k;

    // unpack block
    double* block = block_force_configurations_.Get(k);

    // ----- free joints ----- //

    // scaling
    double scale_free_pos =
        scale_force_[0] / 3 * time_scale / (configuration_length_ - 2);
    double scale_free_rot =
        scale_force_[1] / 3 * time_scale / (configuration_length_ - 2);

    // loop over free joints
    for (int i = 0; i < model_->njnt; i++) {
      // check joint type
      if (model_->jnt_type[i] != mjJNT_FREE) continue;

      // start cost timer
      auto start_cost = std::chrono::steady_clock::now();

      // get dof address
      int adr = model_->jnt_dofadr[i];

      // unpack residual and blocks
      double* residual_pos = residual + adr;
      double* residual_rot = residual + adr + 3;

      // cost free position
      cost +=
          scale_free_pos *
          Norm(gradient ? norm_gradient_force_.data() + nv * k : NULL,
               hessian ? norm_blocks_force_.data() + nv * nv * k : NULL,
               residual_pos,
               norm_parameters_force_.data() + MAX_NORM_PARAMETERS * 0, 3,
               norm_force_[0]);

      // cost free rotation
      cost +=
          scale_free_rot *
          Norm(gradient ? norm_gradient_force_.data() + nv * k + 3 : NULL,
               hessian ? norm_blocks_force_.data() + nv * nv * k + 3 * 3 : NULL,
               residual_rot,
               norm_parameters_force_.data() + MAX_NORM_PARAMETERS * 1, 3,
               norm_force_[1]);

      // stop cost timer
      timer_cost_force_ += GetDuration(start_cost);

      // ----- derivatives ----- //
      if (gradient) {
        // unpacks block
        double* block_pos = block + (3 * nv) * adr;
        double* block_rot = block + (3 * nv) * (adr + 3);

        // -- position -- //

        // scratch = dridq012' * dndri
        mju_mulMatTVec(scratch0_force_.data(), block_pos,
                       norm_gradient_force_.data() + nv * k, 3, 3 * nv);

        // add
        mju_addToScl(gradient + k * nv, scratch0_force_.data(), scale_free_pos,
                     3 * nv);

        // -- rotation -- //

        // scratch = dridq012' * dndri
        mju_mulMatTVec(scratch0_force_.data(), block_rot,
                       norm_gradient_force_.data() + nv * k + 3, 3, 3 * nv);

        // add
        mju_addToScl(gradient + k * nv, scratch0_force_.data(), scale_free_rot,
                     3 * nv);
      }

      if (hessian) {
        // unpacks block
        double* block_pos = block + (3 * nv) * adr;
        double* block_rot = block + (3 * nv) * (adr + 3);

        // -- position -- //
        // step 1: tmp0 = d2ndri2 * dridq012
        double* tmp0 = scratch0_force_.data();
        mju_mulMatMat(tmp0, norm_blocks_force_.data() + nv * nv * k, block_pos,
                      3, 3, 3 * nv);

        // step 2: tmp1 = dridq' * tmp0
        double* tmp1 = scratch1_force_.data();
        mju_mulMatTMat(tmp1, block_pos, tmp0, 3, 3 * nv, 3 * nv);

        // add
        AddBlockInMatrix(hessian, tmp1, scale_free_pos, dim_update, dim_update,
                         3 * nv, 3 * nv, nv * k, nv * k);

        // -- rotation -- //
        // step 1: tmp0 = d2ndri2 * dridq012
        mju_mulMatMat(tmp0, norm_blocks_force_.data() + nv * nv * k + 3 * 3,
                      block_rot, 3, 3, 3 * nv);

        // step 2: tmp1 = dridq' * tmp0
        mju_mulMatTMat(tmp1, block_rot, tmp0, 3, 3 * nv, 3 * nv);

        // add
        AddBlockInMatrix(hessian, tmp1, scale_free_rot, dim_update, dim_update,
                         3 * nv, 3 * nv, nv * k, nv * k);
      }
    }

    // ----- ball, hinge, and slide joints ----- //

    // start cost timer
    auto start_cost = std::chrono::steady_clock::now();

    // non-free dimension
    int dim_nonfree = nv - num_free_ * 6;

    // skip
    if (dim_nonfree == 0) continue;

    // residual
    double* residual_nonfree = scratch2_force_.data();
    double* block_nonfree = scratch2_force_.data() + dim_nonfree;

    // assemble residual and Jacobian, skipping free-joint rows
    int shift = 0;
    for (int i = 0; i < nv; i++) {
      if (!free_dof_[i]) {
        // assemble residual
        residual_nonfree[shift] = residual[i];

        // assemble Jacobian
        mju_copy(block_nonfree + shift * (3 * nv), block + i * (3 * nv),
                 3 * nv);
        // shift
        shift += 1;
      }
    }

    // scaling
    double scale_nonfree = scale_force_[2] / dim_nonfree * time_scale /
                           (configuration_length_ - 2);

    // add weighted norm
    cost += scale_nonfree *
            Norm(gradient ? norm_gradient_force_.data() : NULL,
                 hessian ? norm_blocks_force_.data() : NULL, residual_nonfree,
                 norm_parameters_force_.data() + MAX_NORM_PARAMETERS * 2,
                 dim_nonfree, norm_force_[2]);

    // stop cost timer
    timer_cost_force_ += GetDuration(start_cost);

    // ----- derivatives ----- //

    if (gradient) {
      // scratch = dridq012' * dndri
      mju_mulMatTVec(scratch0_force_.data(), block_nonfree,
                      norm_gradient_force_.data(), dim_nonfree, 3 * nv);

      // add
      mju_addToScl(gradient + k * nv, scratch0_force_.data(), scale_nonfree,
                   3 * nv);
    }

    if (hessian) {
      // step 1: tmp0 = d2ndri2 * dridq012
      double* tmp0 = scratch0_force_.data();
      mju_mulMatMat(tmp0, norm_blocks_force_.data(), block_nonfree, dim_nonfree,
                    dim_nonfree, 3 * nv);

      // step 2: tmp1 = dridq' * tmp0
      double* tmp1 = scratch1_force_.data();
      mju_mulMatTMat(tmp1, block_nonfree, tmp0, dim_nonfree, 3 * nv, 3 * nv);

      // add
      AddBlockInMatrix(hessian, tmp1, scale_nonfree, dim_update, dim_update,
                       3 * nv, 3 * nv, nv * k, nv * k);
    }
  }

  // stop timer
  timer_cost_force_derivatives_ += GetDuration(start);

  return cost;
}

// force residual
void Estimator::ResidualForce() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model_->nv;

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // time index
    int t = k + 1;

    // terms
    double* rk = residual_force_.data() + k * nv;
    double* ft_actuator = force_measurement_.Get(t);
    double* ft_inverse_ = force_prediction_.Get(t);

    // force difference
    mju_sub(rk, ft_inverse_, ft_actuator, nv);
  }

  // stop timer
  timer_residual_force_ += GetDuration(start);
}

// set block in force Jacobian
void Estimator::SetBlockForce(int index) {
  // velocity dimension
  int nv = model_->nv;

  // residual dimension
  int dim_residual = nv * prediction_length_;

  // update dimension
  int dim_update = nv * configuration_length_;

  // reset Jacobian to zero
  mju_zero(jacobian_force_.data() + index * nv * dim_update, nv * dim_update);

  // indices
  int row = index * nv;
  int col_previous = index * nv;
  int col_current = (index + 1) * nv;
  int col_next = (index + 2) * nv;

  // ----- configuration previous ----- //
  // unpack
  double* dfdq0 = block_force_previous_configuration_.Get(index);

  // set
  SetBlockInMatrix(jacobian_force_.data(), dfdq0, 1.0, dim_residual, dim_update,
                   nv, nv, row, col_previous);

  // ----- configuration current ----- //

  // unpack
  double* dfdq1 = block_force_current_configuration_.Get(index);

  // set
  SetBlockInMatrix(jacobian_force_.data(), dfdq1, 1.0, dim_residual, dim_update,
                   nv, nv, row, col_current);

  // ----- configuration next ----- //

  // unpack
  double* dfdq2 = block_force_next_configuration_.Get(index);

  // set
  AddBlockInMatrix(jacobian_force_.data(), dfdq2, 1.0, dim_residual, dim_update,
                   nv, nv, row, col_next);
}

// force Jacobian (dfdq0, dfdq1, dfdq2)
void Estimator::BlockForce(int index) {
  // velocity dimension
  int nv = model_->nv;

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

  // -- configuration current: dfdq1 = dfdq + dfdv * dvdq1 + dfda * dadq1 -- //

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

  // dfdq2 <- dadf' * dadq2
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
}

// force Jacobian
void Estimator::JacobianForce(ThreadPool& pool) {
  // start index
  int start_index = reuse_data_ * mju_max(0, prediction_length_ - num_new_);

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // schedule by time step
    pool.Schedule([&estimator = *this, start_index, k]() {
      // start Jacobian timer
      auto jacobian_force_start = std::chrono::steady_clock::now();

      // block
      if (k >= start_index) estimator.BlockForce(k);

      // assemble
      if (!estimator.band_covariance_) estimator.SetBlockForce(k);

      // stop Jacobian timer
      estimator.timer_force_step_[k] = GetDuration(jacobian_force_start);
    });
  }
}

// compute force
void Estimator::InverseDynamicsPrediction(ThreadPool& pool) {
  // compute sensor and force predictions
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model_->nq, nv = model_->nv, nu = model_->nu, ns = dim_sensor_;

  // start index
  int start_index = reuse_data_ * mju_max(0, prediction_length_ - num_new_);

  // pool count
  int count_before = pool.GetCount();

  // loop over predictions
  for (int k = start_index; k < prediction_length_; k++) {
    // schedule
    pool.Schedule([&estimator = *this, nq, nv, ns, nu, k]() {
      // time index
      int t = k + 1;

      // terms
      double* qt = estimator.configuration_.Get(t);
      double* vt = estimator.velocity_.Get(t);
      double* at = estimator.acceleration_.Get(t);
      double* ct = estimator.ctrl_.Get(t);

      // data
      mjData* d = estimator.data_[k].get();

      // set qt, vt, at
      mju_copy(d->qpos, qt, nq);
      mju_copy(d->qvel, vt, nv);
      mju_copy(d->qacc, at, nv);
      mju_copy(d->ctrl, ct, nu);

      // inverse dynamics
      mj_inverse(estimator.model_, d);

      // copy sensor
      double* st = estimator.sensor_prediction_.Get(t);
      mju_copy(st, d->sensordata, ns);

      // copy force
      double* ft = estimator.force_prediction_.Get(t);
      mju_copy(ft, d->qfrc_inverse, nv);
    });
  }

  // wait
  pool.WaitCount(count_before + (prediction_length_ - start_index));
  pool.ResetCount();

  // stop timer
  timer_cost_prediction_ += GetDuration(start);
}

// compute inverse dynamics derivatives (via finite difference)
void Estimator::InverseDynamicsDerivatives(ThreadPool& pool) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model_->nq, nv = model_->nv, nu = model_->nu;

  // pool count
  int count_before = pool.GetCount();

  // start index
  int start_index = reuse_data_ * mju_max(0, prediction_length_ - num_new_);

  // loop over predictions
  for (int k = start_index; k < prediction_length_; k++) {
    // schedule
    pool.Schedule([&estimator = *this, nq, nv, nu, k]() {
      // time index
      int t = k + 1;

      // unpack
      double* q = estimator.configuration_.Get(t);
      double* v = estimator.velocity_.Get(t);
      double* a = estimator.acceleration_.Get(t);
      double* c = estimator.ctrl_.Get(t);

      double* dqds = estimator.block_sensor_configuration_.Get(k);
      double* dvds = estimator.block_sensor_velocity_.Get(k);
      double* dads = estimator.block_sensor_acceleration_.Get(k);
      double* dqdf = estimator.block_force_configuration_.Get(k);
      double* dvdf = estimator.block_force_velocity_.Get(k);
      double* dadf = estimator.block_force_acceleration_.Get(k);
      mjData* data = estimator.data_[k].get();  // TODO(taylor): WorkerID

      // set (state, acceleration) + ctrl
      mju_copy(data->qpos, q, nq);
      mju_copy(data->qvel, v, nv);
      mju_copy(data->qacc, a, nv);
      mju_copy(data->ctrl, c, nu);

      // finite-difference derivatives
      mjd_inverseFD(estimator.model_, data,
                    estimator.finite_difference_.tolerance,
                    estimator.finite_difference_.flg_actuation, dqdf, dvdf,
                    dadf, dqds, dvds, dads, NULL);
    });
  }

  // wait
  pool.WaitCount(count_before + (prediction_length_ - start_index));

  // reset pool count
  pool.ResetCount();

  // stop timer
  timer_inverse_dynamics_derivatives_ += GetDuration(start);
}

// update configuration trajectory
void Estimator::UpdateConfiguration(
    EstimatorTrajectory<double>& candidate,
    const EstimatorTrajectory<double>& configuration,
    const double* search_direction, double step_size) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model_->nq, nv = model_->nv;

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
    mj_integratePos(model_, ct, dqt, step_size);
  }

  // stop timer
  timer_configuration_update_ += GetDuration(start);
}

// convert sequence of configurations to velocities and accelerations
void Estimator::ConfigurationToVelocityAcceleration() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model_->nv;

  // start index
  int start_index =
      reuse_data_ * mju_max(0, (configuration_length_ - 1) - num_new_);

  // loop over configurations
  for (int k = start_index; k < configuration_length_ - 1; k++) {
    // time index
    int t = k + 1;

    // previous and current configurations
    const double* q0 = configuration_.Get(t - 1);
    const double* q1 = configuration_.Get(t);

    // compute velocity
    double* v1 = velocity_.Get(t);
    mj_differentiatePos(model_, v1, model_->opt.timestep, q0, q1);

    // compute acceleration
    if (t > 1) {
      // previous velocity
      const double* v0 = velocity_.Get(t - 1);

      // compute acceleration
      double* a1 = acceleration_.Get(t - 1);
      mju_sub(a1, v1, v0, nv);
      mju_scl(a1, a1, 1.0 / model_->opt.timestep, nv);
    }
  }

  // stop time
  timer_cost_config_to_velacc_ += GetDuration(start);
}

// compute finite-difference velocity, acceleration derivatives
void Estimator::VelocityAccelerationDerivatives() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model_->nv;

  // start index
  int start_index =
      reuse_data_ * mju_max(0, (configuration_length_ - 1) - num_new_);

  // loop over configurations
  for (int k = start_index; k < configuration_length_ - 1; k++) {
    // time index
    int t = k + 1;

    // unpack
    double* q1 = configuration_.Get(t - 1);
    double* q2 = configuration_.Get(t);
    double* dv2dq1 = block_velocity_previous_configuration_.Get(k);
    double* dv2dq2 = block_velocity_current_configuration_.Get(k);

    // compute velocity Jacobians
    DifferentiateDifferentiatePos(dv2dq1, dv2dq2, model_, model_->opt.timestep,
                                  q1, q2);

    // compute acceleration Jacobians
    if (t > 1) {
      // unpack
      double* dadq0 = block_acceleration_previous_configuration_.Get(k - 1);
      double* dadq1 = block_acceleration_current_configuration_.Get(k - 1);
      double* dadq2 = block_acceleration_next_configuration_.Get(k - 1);

      // previous velocity Jacobians
      double* dv1dq0 = block_velocity_previous_configuration_.Get(k - 1);
      double* dv1dq1 = block_velocity_current_configuration_.Get(k - 1);

      // dadq0 = -dv1dq0 / h
      mju_copy(dadq0, dv1dq0, nv * nv);
      mju_scl(dadq0, dadq0, -1.0 / model_->opt.timestep, nv * nv);

      // dadq1 = dv2dq1 / h - dv1dq1 / h = (dv2dq1 - dv1dq1) / h
      mju_sub(dadq1, dv2dq1, dv1dq1, nv * nv);
      mju_scl(dadq1, dadq1, 1.0 / model_->opt.timestep, nv * nv);

      // dadq2 = dv2dq2 / h
      mju_copy(dadq2, dv2dq2, nv * nv);
      mju_scl(dadq2, dadq2, 1.0 / model_->opt.timestep, nv * nv);
    }
  }

  // stop timer
  timer_velacc_derivatives_ += GetDuration(start);
}

// compute total cost
double Estimator::Cost(ThreadPool& pool) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // finite-difference velocities, accelerations
  ConfigurationToVelocityAcceleration();

  // compute sensor and force predictions
  InverseDynamicsPrediction(pool);

  // residuals
  if (prior_flag_) ResidualPrior();
  if (sensor_flag_) ResidualSensor();
  if (force_flag_) ResidualForce();

  // costs
  cost_prior_ = (prior_flag_ ? CostPrior(NULL, NULL) : 0.0);
  cost_sensor_ = (sensor_flag_ ? CostSensor(NULL, NULL) : 0.0);
  cost_force_ = (force_flag_ ? CostForce(NULL, NULL) : 0.0);

  // total cost
  double cost = cost_prior_ + cost_sensor_ + cost_force_;

  // counter
  cost_count_++;

  // stop timer
  timer_cost_ += GetDuration(start);

  // total cost
  return cost;
}

// compute total gradient
void Estimator::CostGradient() {
  // start gradient timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int dim = configuration_length_ * model_->nv;

  // unpack
  double* gradient = cost_gradient_.data();

  // individual gradients
  if (prior_flag_) {
    mju_copy(gradient, cost_gradient_prior_.data(), dim);
  } else {
    mju_zero(gradient, dim);
  }
  if (sensor_flag_) mju_addTo(gradient, cost_gradient_sensor_.data(), dim);
  if (force_flag_) mju_addTo(gradient, cost_gradient_force_.data(), dim);

  // stop gradient timer
  timer_cost_gradient_ += GetDuration(start);
}

// compute total Hessian
void Estimator::CostHessian() {
  // start Hessian timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int dim = configuration_length_ * model_->nv;

  // unpack
  double* hessian = cost_hessian_.data();

  if (band_copy_) {
    // zero memory
    mju_zero(hessian, dim * dim);

    // individual Hessians
    if (prior_flag_)
      SymmetricBandMatrixCopy(hessian, cost_hessian_prior_.data(), model_->nv,
                              3, dim, configuration_length_, 0, 0, 0, 0,
                              scratch0_prior_.data());
    if (sensor_flag_)
      SymmetricBandMatrixCopy(hessian, cost_hessian_sensor_.data(), model_->nv,
                              3, dim, configuration_length_, 0, 0, 0, 0,
                              scratch0_sensor_.data());
    if (force_flag_)
      SymmetricBandMatrixCopy(hessian, cost_hessian_force_.data(), model_->nv,
                              3, dim, configuration_length_, 0, 0, 0, 0,
                              scratch0_force_.data());
  } else {
    // individual Hessians
    if (prior_flag_) {
      mju_copy(hessian, cost_hessian_prior_.data(), dim * dim);
    } else {
      mju_zero(hessian, dim * dim);
    }
    if (sensor_flag_)
      mju_addTo(hessian, cost_hessian_sensor_.data(), dim * dim);
    if (force_flag_) mju_addTo(hessian, cost_hessian_force_.data(), dim * dim);
  }

  // stop Hessian timer
  timer_cost_hessian_ += GetDuration(start);
}

// covariance update
void Estimator::PriorWeightUpdate(ThreadPool& pool) {
  // skip
  if (skip_update_prior_weight) return;

  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model_->nv;
  int ntotal = nv * configuration_length_;

  // ----- update prior weights ----- //
  // start timer
  auto start_set_weight = std::chrono::steady_clock::now();

  // weight
  double* weight = weight_prior_dense_.data();

  // Hessian
  double* hessian = cost_hessian_.data();

  // zero memory
  mju_zero(weight, ntotal * ntotal);

  // copy Hessian block to upper left
  if (configuration_length_ - num_new_ > 0 && update_prior_weight_) {
    SymmetricBandMatrixCopy(weight, hessian, nv, nv, ntotal,
                            configuration_length_ - num_new_, 0, 0, num_new_,
                            num_new_, scratch_prior_weight_.data());
  }

  // set s * I to lower right
  for (int i = update_prior_weight_ * nv * (configuration_length_ - num_new_);
       i < ntotal; i++) {
    weight[ntotal * i + i] = scale_prior_;
  }

  // stop timer
  timer_prior_set_weight_ += GetDuration(start_set_weight);

  // stop timer
  timer_prior_weight_update_ += GetDuration(start);

  // status
  PrintPriorWeightUpdate();
}

// optimize trajectory estimate
void Estimator::Optimize(ThreadPool& pool) {
  // start timer
  auto start_optimize = std::chrono::steady_clock::now();

  // dimensions
  int nconfig = model_->nq * configuration_length_;
  int nvar = model_->nv * configuration_length_;

  // operations
  int nprior = prior_flag_ * configuration_length_;
  int nsensor = sensor_flag_ * prediction_length_;
  int nforce = force_flag_ * prediction_length_;

  // reset timers
  ResetTimers();

  // prior update
  PriorWeightUpdate(pool);

  // initial cost
  cost_count_ = 0;
  cost_ = Cost(pool);
  cost_initial_ = cost_;

  // print initial cost
  PrintCost();

  // ----- smoother iterations ----- //

  // reset
  iterations_smoother_ = 0;
  iterations_line_search_ = 0;

  // iterations
  for (; iterations_smoother_ < max_smoother_iterations_;
       iterations_smoother_++) {
    // ----- cost derivatives ----- //

    // start timer (total cost derivatives)
    auto cost_derivatives_start = std::chrono::steady_clock::now();

    // inverse dynamics derivatives
    InverseDynamicsDerivatives(pool);

    // velocity, acceleration derivatives
    VelocityAccelerationDerivatives();

    // -- Jacobians -- //
    auto timer_jacobian_start = std::chrono::steady_clock::now();

    // pool count
    int count_begin = pool.GetCount();

    // individual derivatives
    if (prior_flag_) JacobianPrior(pool);
    if (sensor_flag_) JacobianSensor(pool);
    if (force_flag_) JacobianForce(pool);

    // wait
    pool.WaitCount(count_begin + nprior + nsensor + nforce);

    // reset count
    pool.ResetCount();

    // timers
    timer_jacobian_prior_ += mju_sum(timer_prior_step_.data(), nprior);
    timer_jacobian_sensor_ += mju_sum(timer_sensor_step_.data(), nsensor);
    timer_jacobian_force_ += mju_sum(timer_force_step_.data(), nforce);
    timer_jacobian_total_ += GetDuration(timer_jacobian_start);

    // -- cost derivatives -- //

    // start timer
    auto start_cost_total_derivatives = std::chrono::steady_clock::now();

    // pool count
    count_begin = pool.GetCount();

    // individual derivatives
    if (prior_flag_) {
      pool.Schedule([&estimator = *this]() {
        estimator.CostPrior(estimator.cost_gradient_prior_.data(),
                            estimator.cost_hessian_prior_.data());
      });
    }
    if (sensor_flag_) {
      pool.Schedule([&estimator = *this]() {
        estimator.CostSensor(estimator.cost_gradient_sensor_.data(),
                             estimator.cost_hessian_sensor_.data());
      });
    }
    if (force_flag_) {
      pool.Schedule([&estimator = *this]() {
        estimator.CostForce(estimator.cost_gradient_force_.data(),
                            estimator.cost_hessian_force_.data());
      });
    }
    // wait
    pool.WaitCount(count_begin + prior_flag_ + sensor_flag_ + force_flag_);

    // pool reset
    pool.ResetCount();

    // reset num_new_
    num_new_ = configuration_length_;  // update all data now

    // stop timer
    timer_cost_total_derivatives_ += GetDuration(start_cost_total_derivatives);

    // gradient
    double* gradient = cost_gradient_.data();
    CostGradient();

    // gradient tolerance check
    gradient_norm_ = mju_norm(gradient, nvar) / nvar;
    if (gradient_norm_ < gradient_tolerance_) {
      break;
    }

    // Hessian
    CostHessian();

    // stop timer
    timer_cost_derivatives_ += GetDuration(cost_derivatives_start);

    // ----- line / curve search ----- //
    // start timer
    auto line_search_start = std::chrono::steady_clock::now();

    // copy configuration
    mju_copy(configuration_copy_.Data(), configuration_.Data(), nconfig);

    // initialize
    double cost_candidate = cost_;
    int iteration_search = 0;
    step_size_ = 1.0;
    regularization_ = mju_max(MIN_REGULARIZATION, regularization_);

    // initial search direction
    SearchDirection();

    // backtracking until cost decrease
    // TODO(taylor): Armijo, Wolfe conditions
    while (cost_candidate >= cost_) {
      // check for max iterations
      if (iteration_search > max_line_search_) {
        // reset configuration
        mju_copy(configuration_.Data(), configuration_copy_.Data(), nconfig);

        // restore velocity, acceleration
        ConfigurationToVelocityAcceleration();

        printf("line search failure\n");

        // failure
        return;
      }

      // search type
      if (iteration_search > 0) {
        switch (search_type_) {
          case SearchType::kLineSearch:
            // decrease step size
            step_size_ *= step_scaling_;
            break;
          case SearchType::kCurveSearch:
            // increase regularization
            regularization_ = mju_min(
                MAX_REGULARIZATION, regularization_ * regularization_scaling_);
            // recompute search direction
            SearchDirection();
            break;
          default:
            mju_error("Invalid search type.\n");
            break;
        }
      }

      // candidate
      UpdateConfiguration(configuration_, configuration_copy_,
                          search_direction_.data(), -1.0 * step_size_);

      // cost
      cost_candidate = Cost(pool);

      // update iteration
      iteration_search++;
    }

    // increment
    iterations_line_search_ += iteration_search;

    // end timer
    timer_search_ += GetDuration(line_search_start);

    // update cost
    cost_ = cost_candidate;

    // decrease regularization
    regularization_ = mju_max(
                MIN_REGULARIZATION, regularization_ / regularization_scaling_);

    // print cost
    PrintCost();
  }

  // ----- GUI update ----- //
  // update costs
  // set timers
  // set status

  // stop timer
  timer_optimize_ = GetDuration(start_optimize);

  // status
  PrintOptimize();
}

// regularize Hessian
void Estimator::Regularize() {
  // dimension
  int nvar = configuration_length_ * model_->nv;

  // regularize
  // TODO(taylor): LM reg.
  for (int j = 0; j < nvar; j++) {
    cost_hessian_[j * nvar + j] += regularization_;
  }
}

// search direction
void Estimator::SearchDirection() {
  // start timer
  auto search_direction_start = std::chrono::steady_clock::now();

  // dimensions
  int ntotal = configuration_length_ * model_->nv;
  int nband = 3 * model_->nv;
  int ndense = 0;

  // regularize
  Regularize();

  // -- band Hessian -- //

  // unpack
  double* direction = search_direction_.data();
  double* gradient = cost_gradient_.data();
  double* hessian = cost_hessian_.data();
  double* hessian_band = cost_hessian_band_.data();

  // -- linear system solver -- //

  // select solver
  if (band_covariance_) {  // band solver
    // dense to band
    mju_dense2Band(hessian_band, cost_hessian_.data(), ntotal, nband, ndense);

    // factorize
    mju_cholFactorBand(hessian_band, ntotal, nband, ndense, 0.0, 0.0);

    // compute search direction
    mju_cholSolveBand(direction, hessian_band, gradient, ntotal, nband, ndense);
  } else {  // dense solver
    // factorize
    double* factor = cost_hessian_factor_.data();
    mju_copy(factor, hessian, ntotal * ntotal);
    mju_cholFactor(factor, ntotal, 0.0);

    // compute search direction
    mju_cholSolve(direction, factor, gradient, ntotal);
  }

  // set prior reset flag
  if (!hessian_factor_) {
    hessian_factor_ = true;
  }

  // end timer
  timer_search_direction_ += GetDuration(search_direction_start);
}

// print Optimize status
void Estimator::PrintOptimize() {
  if (!verbose_optimize_) return;

  // title
  printf("Estimator::Optimize Status:\n\n");

  // timing
  printf("Timing:\n");

  PrintPriorWeightUpdate();

  printf("\n");
  printf("  cost (initial): %.3f (ms) \n", 1.0e-3 * timer_cost_ / cost_count_);
  printf("    - prior: %.3f (ms) \n", 1.0e-3 * timer_cost_prior_ / cost_count_);
  printf("    - sensor: %.3f (ms) \n",
         1.0e-3 * timer_cost_sensor_ / cost_count_);
  printf("    - force: %.3f (ms) \n", 1.0e-3 * timer_cost_force_ / cost_count_);
  printf("    - qpos -> qvel, qacc: %.3f (ms) \n",
         1.0e-3 * timer_cost_config_to_velacc_ / cost_count_);
  printf("    - prediction: %.3f (ms) \n",
         1.0e-3 * timer_cost_prediction_ / cost_count_);
  printf("    - residual prior: %.3f (ms) \n",
         1.0e-3 * timer_residual_prior_ / cost_count_);
  printf("    - residual sensor: %.3f (ms) \n",
         1.0e-3 * timer_residual_sensor_ / cost_count_);
  printf("    - residual force: %.3f (ms) \n",
         1.0e-3 * timer_residual_force_ / cost_count_);
  printf("\n");
  printf("  cost derivatives [total]: %.3f (ms) \n",
         1.0e-3 * timer_cost_derivatives_);
  printf("    - inverse dynamics derivatives: %.3f (ms) \n",
         1.0e-3 * timer_inverse_dynamics_derivatives_);
  printf("    - vel., acc. derivatives: %.3f (ms) \n",
         1.0e-3 * timer_velacc_derivatives_);
  printf("    - jacobian [total]: %.3f (ms) \n",
         1.0e-3 * timer_jacobian_total_);
  printf("      < prior: %.3f (ms) \n", 1.0e-3 * timer_jacobian_prior_);
  printf("      < sensor: %.3f (ms) \n", 1.0e-3 * timer_jacobian_sensor_);
  printf("      < force: %.3f (ms) \n", 1.0e-3 * timer_jacobian_force_);
  printf("    - gradient, hessian [total]: %.3f (ms) \n",
         1.0e-3 * timer_cost_total_derivatives_);
  printf("      < prior: %.3f (ms) \n", 1.0e-3 * timer_cost_prior_derivatives_);
  printf("      < sensor: %.3f (ms) \n",
         1.0e-3 * timer_cost_sensor_derivatives_);
  printf("      < force: %.3f (ms) \n", 1.0e-3 * timer_cost_force_derivatives_);
  printf("      < gradient assemble: %.3f (ms) \n",
         1.0e-3 * timer_cost_gradient_);
  printf("      < hessian assemble: %.3f (ms) \n",
         1.0e-3 * timer_cost_hessian_);
  printf("\n");
  printf("  search [total]: %.3f (ms) \n", 1.0e-3 * timer_search_);
  printf("    - direction: %.3f (ms) \n", 1.0e-3 * timer_search_direction_);
  printf("    - cost: %.3f (ms) \n",
         1.0e-3 * (timer_cost_ - timer_cost_ / cost_count_));
  printf("      < prior: %.3f (ms) \n",
         1.0e-3 * (timer_cost_prior_ - timer_cost_prior_ / cost_count_));
  printf("      < sensor: %.3f (ms) \n",
         1.0e-3 * (timer_cost_sensor_ - timer_cost_sensor_ / cost_count_));
  printf("      < force: %.3f (ms) \n",
         1.0e-3 * (timer_cost_force_ - timer_cost_force_ / cost_count_));
  printf("      < qpos -> qvel, qacc: %.3f (ms) \n",
         1.0e-3 * (timer_cost_config_to_velacc_ -
                   timer_cost_config_to_velacc_ / cost_count_));
  printf(
      "      < prediction: %.3f (ms) \n",
      1.0e-3 * (timer_cost_prediction_ - timer_cost_prediction_ / cost_count_));
  printf(
      "      < residual prior: %.3f (ms) \n",
      1.0e-3 * (timer_residual_prior_ - timer_residual_prior_ / cost_count_));
  printf(
      "      < residual sensor: %.3f (ms) \n",
      1.0e-3 * (timer_residual_sensor_ - timer_residual_sensor_ / cost_count_));
  printf(
      "      < residual force: %.3f (ms) \n",
      1.0e-3 * (timer_residual_force_ - timer_residual_force_ / cost_count_));
  printf("\n");
  printf("  TOTAL: %.3f (ms) \n", 1.0e-3 * (timer_optimize_));
  printf("\n");

  // status
  printf("Status:\n");
  printf("  iterations line search: %i\n", iterations_line_search_);
  printf("  iterations smoother: %i\n", iterations_smoother_);
  printf("\n");

  // cost
  printf("Cost:\n");
  printf("  initial: %.3f\n", cost_initial_);
  printf("  final: %.3f\n", cost_);
  printf("\n");
}

// print cost
void Estimator::PrintCost() {
  if (verbose_cost_) {
    printf("cost (total): %.3f\n", cost_);
    printf("  prior: %.3f\n", cost_prior_);
    printf("  sensor: %.3f\n", cost_sensor_);
    printf("  force: %.3f\n", cost_force_);
    printf("  [initial: %.3f]\n", cost_initial_);
  }
}

// print prior weight update status
// print Optimize status
void Estimator::PrintPriorWeightUpdate() {
  if (!verbose_prior_) return;

  // timing
  printf("  prior weight update [total]: %.3f (ms) \n",
         1.0e-3 * timer_prior_weight_update_);
  printf("    - set weight: %.3f (ms) \n", 1.0e-3 * timer_prior_set_weight_);
  printf("\n");
}

// reset timers
void Estimator::ResetTimers() {
  timer_inverse_dynamics_derivatives_ = 0.0;
  timer_velacc_derivatives_ = 0.0;
  timer_jacobian_prior_ = 0.0;
  timer_jacobian_sensor_ = 0.0;
  timer_jacobian_force_ = 0.0;
  timer_jacobian_total_ = 0.0;
  timer_cost_prior_derivatives_ = 0.0;
  timer_cost_sensor_derivatives_ = 0.0;
  timer_cost_force_derivatives_ = 0.0;
  timer_cost_total_derivatives_ = 0.0;
  timer_cost_gradient_ = 0.0;
  timer_cost_hessian_ = 0.0;
  timer_cost_derivatives_ = 0.0;
  timer_cost_ = 0.0;
  timer_cost_prior_ = 0.0;
  timer_cost_sensor_ = 0.0;
  timer_cost_force_ = 0.0;
  timer_cost_config_to_velacc_ = 0.0;
  timer_cost_prediction_ = 0.0;
  timer_residual_prior_ = 0.0;
  timer_residual_sensor_ = 0.0;
  timer_residual_force_ = 0.0;
  timer_search_direction_ = 0.0;
  timer_search_ = 0.0;
  timer_configuration_update_ = 0.0;
  timer_optimize_ = 0.0;
  timer_prior_weight_update_ = 0.0;
  timer_prior_set_weight_ = 0.0;
  timer_update_trajectory_ = 0.0;
}

// get qpos estimate
double* Estimator::GetPosition() { return configuration_.Get(state_index_); }

// get qvel estimate
double* Estimator::GetVelocity() { return velocity_.Get(state_index_); }

// initialize trajectories
void Estimator::InitializeTrajectories(
    const EstimatorTrajectory<double>& measurement,
    const EstimatorTrajectory<int>& measurement_mask,
    const EstimatorTrajectory<double>& ctrl,
    const EstimatorTrajectory<double>& time) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // -- set initial configurations -- //

  // set first configuration
  double* q0 = configuration_.Get(0);
  mju_copy(q0, qpos0_.data(), model_->nq);
  mj_integratePos(model_, q0, qvel0_.data(), -1.0 * model_->opt.timestep);

  // set second configuration
  configuration_.Set(qpos0_.data(), 1);

  // set initial time 
  time_.Set(time.Get(0), 0);

  // data
  mjData* data = data_[0].get();

  // set state 
  mju_copy(data->qpos, qpos0_.data(), model_->nq);
  mju_copy(data->qvel, qvel0_.data(), model_->nv);
  data->time = time.Get(1)[0];

  // set new measurements, ctrl -> qfrc_actuator, rollout new configurations,
  // new time
  for (int i = 1; i < configuration_length_ - 1; i++) {
    // buffer index
    int buffer_index = time.length_ - (configuration_length_ - 1) + i;

    // get time
    time_.Set(&data->time, i);

    // set/get ctrl
    const double* ui = ctrl.Get(buffer_index);
    ctrl_.Set(ui, i);
    mju_copy(data->ctrl, ui, model_->nu);

    // step dynamics
    mj_step(model_, data);

    // set measurement
    const double* yi = measurement.Get(buffer_index);
    sensor_measurement_.Set(yi, i);

    // set mask
    const int* mi = measurement_mask.Get(buffer_index);
    sensor_mask_.Set(mi, i);

    // copy qfrc_actuator
    force_measurement_.Set(data->qfrc_actuator, i);

    // copy configuration
    configuration_.Set(data->qpos, i + 1);
  }

  // set last time 
  time_.Set(&data->time, configuration_length_ - 1);

  // copy configuration to prior
  mju_copy(configuration_prior_.Data(), configuration_.Data(),
           model_->nq * configuration_length_);

  // stop timer
  timer_update_trajectory_ += GetDuration(start);
}

// update trajectories
int Estimator::UpdateTrajectories_(
    int num_new, const EstimatorTrajectory<double>& measurement,
    const EstimatorTrajectory<int>& measurement_mask,
    const EstimatorTrajectory<double>& ctrl,
    const EstimatorTrajectory<double>& time) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // shift trajectory heads
  ShiftTrajectoryHead(num_new);

  // get data
  mjData* data = data_[0].get();

  // set new measurements, ctrl -> qfrc_actuator, rollout new configurations,
  // new time
  for (int i = 0; i < num_new; i++) {
    // time index
    int t = i + configuration_length_ - num_new - 1;

    // buffer index
    int b = i + measurement.length_ - num_new;

    // set measurement
    const double* yi = measurement.Get(b);
    sensor_measurement_.Set(yi, t);

    // set measurement mask
    const int* mi = measurement_mask.Get(b);
    sensor_mask_.Set(mi, t);

    // set time
    const double* ti = time.Get(b);
    time_.Set(ti, t);

    // ----- forward dynamics ----- //

    // set ctrl
    const double* ui = ctrl.Get(b);
    ctrl_.Set(ui, t);
    mju_copy(data->ctrl, ui, model_->nu);

    // set qpos
    double* q0 = configuration_.Get(t - 1);
    double* q1 = configuration_.Get(t);
    mju_copy(data->qpos, q1, model_->nq);

    // set qvel
    mj_differentiatePos(model_, data->qvel, model_->opt.timestep, q0, q1);

    // set time 
    data->time = time.Get(b)[0];

    // step dynamics
    mj_step(model_, data);

    // copy qfrc_actuator
    force_measurement_.Set(data->qfrc_actuator, t);

    // copy configuration
    configuration_.Set(data->qpos, t + 1);
  }

  // set last time 
  time_.Set(&data->time, configuration_length_ - 1);

  // copy configuration to prior
  mju_copy(configuration_prior_.Data(), configuration_.Data(),
           model_->nq * configuration_length_);

  // stop timer
  timer_update_trajectory_ += GetDuration(start);

  return num_new;
}

// update trajectories
int Estimator::UpdateTrajectories(
    const EstimatorTrajectory<double>& measurement,
    const EstimatorTrajectory<int>& measurement_mask,
    const EstimatorTrajectory<double>& ctrl,
    const EstimatorTrajectory<double>& time) {
  // lastest buffer time
  double time_buffer_last = *time.Get(time.length_ - 1);

  // latest estimator time
  double time_estimator_last =
      *time_.Get(time_.length_ - 2);  // index to latest measurement time

  // compute number of new elements
  int num_new =
      std::round(mju_max(0.0, time_buffer_last - time_estimator_last) /
                 model_->opt.timestep);

  UpdateTrajectories_(num_new, measurement, measurement_mask, ctrl, time);

  return num_new;
}

// update
int Estimator::Update(const Buffer& buffer, ThreadPool& pool) {
  int num_new = 0;
  if (buffer.Length() >= configuration_length_ - 1) {
    num_new_ = configuration_length_;
    if (!initialized_) {
      InitializeTrajectories(buffer.sensor_, buffer.sensor_mask_, buffer.ctrl_,
                             buffer.time_);
      initialized_ = true;
    } else {
      num_new_ = UpdateTrajectories(buffer.sensor_, buffer.sensor_mask_,
                                    buffer.ctrl_, buffer.time_);
    }
    num_new = num_new_;

    // optimize
    Optimize(pool);
  }
  return num_new;
}

// get terms from GUI
void Estimator::GetGUI() {
  // lock
  const std::lock_guard<std::mutex> lock(mutex_);

  // settings
  configuration_length_ = gui_configuration_length_;
  max_smoother_iterations_ = gui_max_smoother_iterations_;

  // weights
  scale_prior_ = gui_scale_prior_;
  scale_sensor_ = gui_weight_sensor_;
  scale_force_ = gui_weight_force_;
}

// set terms to GUI
void Estimator::SetGUI() {
  // lock
  const std::lock_guard<std::mutex> lock(mutex_);

  // costs
  gui_cost_prior_ = cost_prior_;
  gui_cost_sensor_ = cost_sensor_;
  gui_cost_force_ = cost_force_;
  gui_cost_ = cost_;

  // status
  gui_regularization_ = regularization_;
  gui_step_size_ = step_size_;

  // timers
}

}  // namespace mjpc
