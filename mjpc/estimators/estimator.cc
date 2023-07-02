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
  configuration.Initialize(nq, configuration_length_);
  velocity.Initialize(nv, configuration_length_);
  acceleration.Initialize(nv, configuration_length_);
  time.Initialize(1, configuration_length_);

  // ctrl
  ctrl.Initialize(model->nu, configuration_length_);

  // prior
  configuration_previous.Initialize(nq, configuration_length_);

  // sensor
  dim_sensor_ = model->nsensordata;  // TODO(taylor): grab from xml
  num_sensor_ = model->nsensor;      // TODO(taylor): grab from xml
  sensor_measurement.Initialize(dim_sensor_, configuration_length_);
  sensor_prediction.Initialize(dim_sensor_, configuration_length_);
  sensor_mask.Initialize(num_sensor_, configuration_length_);

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
  force_measurement.Initialize(nv, configuration_length_);
  force_prediction.Initialize(nv, configuration_length_);

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
  cost_gradient.resize(nv * MAX_HISTORY);

  // cost Hessian
  cost_hessian_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_sensor_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_band_.resize(BandMatrixNonZeros(nv * MAX_HISTORY, 3 * nv));
  cost_hessian_factor_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // prior weights
  scale_prior = GetNumberOrDefault(1.0, model, "estimator_scale_prior");
  weight_prior.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  weight_prior_band.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch_prior_weight_.resize(2 * nv * nv);

  // sensor scale
  // TODO(taylor): only grab measurement sensors
  scale_sensor.resize(num_sensor_);

  // TODO(taylor): method for xml to initial weight
  for (int i = 0; i < num_sensor_; i++) {
    scale_sensor[i] = GetNumberOrDefault(1.0, model, "estimator_scale_sensor");
  }

  // force scale
  scale_force.resize(NUM_FORCE_TERMS);

  scale_force[0] =
      GetNumberOrDefault(1.0, model, "estimator_scale_force_free_position");
  scale_force[1] =
      GetNumberOrDefault(1.0, model, "estimator_scale_force_free_rotation");
  scale_force[2] =
      GetNumberOrDefault(1.0, model, "estimator_scale_force_nonfree");

  // cost norms
  // TODO(taylor): only grab measurement sensors
  norm_sensor.resize(num_sensor_);

  // TODO(taylor): method for xml to initial norm
  for (int i = 0; i < num_sensor_; i++) {
    norm_sensor[i] =
        (NormType)GetNumberOrDefault(0, model, "estimator_norm_sensor");
  }

  norm_force[0] = (NormType)GetNumberOrDefault(
      0, model, "estimator_norm_force_free_position");
  norm_force[1] = (NormType)GetNumberOrDefault(
      0, model, "estimator_norm_force_free_rotation");
  norm_force[2] =
      (NormType)GetNumberOrDefault(0, model, "estimator_norm_force_nonfree");

  // cost norm parameters
  norm_parameters_sensor.resize(num_sensor_ * MAX_NORM_PARAMETERS);
  norm_parameters_force.resize(NUM_FORCE_TERMS * MAX_NORM_PARAMETERS);

  // TODO(taylor): initialize norm parameters from xml
  std::fill(norm_parameters_sensor.begin(), norm_parameters_sensor.end(),
            0.0);
  std::fill(norm_parameters_force.begin(), norm_parameters_force.end(), 0.0);

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
  regularization_ = regularization_initial;

  // search type
  search_type =
      (SearchType)GetNumberOrDefault(0, model, "estimator_search_type");

  // initial state
  qpos0_.resize(model->nq);
  mju_copy(qpos0_.data(), model->qpos0, model->nq);
  qvel0_.resize(model->nv);
  mju_zero(qvel0_.data(), model->nv);

  // timer
  timer_.prior_step.resize(MAX_HISTORY);
  timer_.sensor_step.resize(MAX_HISTORY);
  timer_.force_step.resize(MAX_HISTORY);

  // status
  hessian_factor_ = false;
  num_new_ = configuration_length_;
  gradient_norm_ = 0.0;

  // settings
  band_prior = (bool)GetNumberOrDefault(1, model, "estimator_band_covariance");

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
  configuration.SetLength(length);
  configuration_copy_.SetLength(length);

  velocity.SetLength(length);
  acceleration.SetLength(length);
  time.SetLength(length);

  ctrl.SetLength(length);

  configuration_previous.SetLength(length);

  sensor_measurement.SetLength(length);
  sensor_prediction.SetLength(length);
  sensor_mask.SetLength(length);

  force_measurement.SetLength(length);
  force_prediction.SetLength(length);

  block_prior_current_configuration_.SetLength(length);

  block_sensor_configuration_.SetLength(prediction_length_);
  block_sensor_velocity_.SetLength(prediction_length_);
  block_sensor_acceleration_.SetLength(prediction_length_);

  block_sensor_previous_configuration_.SetLength(prediction_length_);
  block_sensor_current_configuration_.SetLength(prediction_length_);
  block_sensor_next_configuration_.SetLength(prediction_length_);
  block_sensor_configurations_.SetLength(prediction_length_);

  block_sensor_scratch_.SetLength(prediction_length_);

  block_force_configuration_.SetLength(prediction_length_);
  block_force_velocity_.SetLength(prediction_length_);
  block_force_acceleration_.SetLength(prediction_length_);

  block_force_previous_configuration_.SetLength(prediction_length_);
  block_force_current_configuration_.SetLength(prediction_length_);
  block_force_next_configuration_.SetLength(prediction_length_);
  block_force_configurations_.SetLength(prediction_length_);

  block_force_scratch_.SetLength(prediction_length_);

  block_velocity_previous_configuration_.SetLength(length - 1);
  block_velocity_current_configuration_.SetLength(length - 1);

  block_acceleration_previous_configuration_.SetLength(prediction_length_);
  block_acceleration_current_configuration_.SetLength(prediction_length_);
  block_acceleration_next_configuration_.SetLength(prediction_length_);

  // status
  num_new_ = configuration_length_;
  initialized_ = false;
  step_size_ = 1.0;
  gradient_norm_ = 0.0;
}

// shift trajectory heads
void Estimator::Shift(int shift) {
  // update trajectory lengths
  configuration.Shift(shift);
  configuration_copy_.Shift(shift);

  velocity.Shift(shift);
  acceleration.Shift(shift);
  time.Shift(shift);

  ctrl.Shift(shift);

  configuration_previous.Shift(shift);

  sensor_measurement.Shift(shift);
  sensor_prediction.Shift(shift);
  sensor_mask.Shift(shift);

  force_measurement.Shift(shift);
  force_prediction.Shift(shift);

  block_prior_current_configuration_.Shift(shift);

  block_sensor_configuration_.Shift(shift);
  block_sensor_velocity_.Shift(shift);
  block_sensor_acceleration_.Shift(shift);

  block_sensor_previous_configuration_.Shift(shift);
  block_sensor_current_configuration_.Shift(shift);
  block_sensor_next_configuration_.Shift(shift);
  block_sensor_configurations_.Shift(shift);

  block_sensor_scratch_.Shift(shift);

  block_force_configuration_.Shift(shift);
  block_force_velocity_.Shift(shift);
  block_force_acceleration_.Shift(shift);

  block_force_previous_configuration_.Shift(shift);
  block_force_current_configuration_.Shift(shift);
  block_force_next_configuration_.Shift(shift);
  block_force_configurations_.Shift(shift);

  block_force_scratch_.Shift(shift);

  block_velocity_previous_configuration_.Shift(shift);
  block_velocity_current_configuration_.Shift(shift);

  block_acceleration_previous_configuration_.Shift(shift);
  block_acceleration_current_configuration_.Shift(shift);
  block_acceleration_next_configuration_.Shift(shift);
}

// reset memory
void Estimator::Reset() {
  // trajectories
  configuration.Reset();
  velocity.Reset();
  acceleration.Reset();
  time.Reset();

  ctrl.Reset();

  // prior
  configuration_previous.Reset();

  // sensor
  sensor_measurement.Reset();
  sensor_prediction.Reset();

  // sensor mask
  sensor_mask.Reset();
  for (int i = 0; i < num_sensor_ * configuration_length_; i++) {
    sensor_mask.Data()[i] = 1;
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
  cost_prior = 0.0;
  cost_sensor = 0.0;
  cost_force = 0.0;
  cost = 0.0;
  cost_initial = 0.0;

  // cost gradient
  std::fill(cost_gradient_prior_.begin(), cost_gradient_prior_.end(), 0.0);
  std::fill(cost_gradient_sensor_.begin(), cost_gradient_sensor_.end(), 0.0);
  std::fill(cost_gradient_force_.begin(), cost_gradient_force_.end(), 0.0);
  std::fill(cost_gradient.begin(), cost_gradient.end(), 0.0);

  // cost Hessian
  std::fill(cost_hessian_prior_.begin(), cost_hessian_prior_.end(), 0.0);
  std::fill(cost_hessian_sensor_.begin(), cost_hessian_sensor_.end(), 0.0);
  std::fill(cost_hessian_force_.begin(), cost_hessian_force_.end(), 0.0);
  std::fill(cost_hessian.begin(), cost_hessian.end(), 0.0);
  std::fill(cost_hessian_band_.begin(), cost_hessian_band_.end(), 0.0);
  std::fill(cost_hessian_factor_.begin(), cost_hessian_factor_.end(), 0.0);

  // weight
  std::fill(weight_prior.begin(), weight_prior.end(), 0.0);
  std::fill(weight_prior_band.begin(), weight_prior_band.end(), 0.0);
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
  std::fill(timer_.prior_step.begin(), timer_.prior_step.end(), 0.0);
  std::fill(timer_.sensor_step.begin(), timer_.sensor_step.end(), 0.0);
  std::fill(timer_.force_step.begin(), timer_.force_step.end(), 0.0);

  // timing
  ResetTimers();

  // status
  iterations_smoother_ = 0;
  iterations_line_search_ = 0;
  cost_count_ = 0;
  initialized_ = false;
}

// evaluate configurations
void Estimator::ConfigurationEvaluation(ThreadPool& pool) {
  // finite-difference velocities, accelerations
  ConfigurationToVelocityAcceleration();

  // compute sensor and force predictions
  InverseDynamicsPrediction(pool);
}

// configurations derivatives
void Estimator::ConfigurationDerivative(ThreadPool& pool) {
  // operations
  int nprior = prior_flag * configuration_length_;
  int nsensor = sensor_flag * prediction_length_;
  int nforce = force_flag * prediction_length_;

  // inverse dynamics derivatives
  InverseDynamicsDerivatives(pool);

  // velocity, acceleration derivatives
  VelocityAccelerationDerivatives();

  // -- Jacobians -- //
  auto timer_jacobian_start = std::chrono::steady_clock::now();

  // pool count
  int count_begin = pool.GetCount();

  // individual derivatives
  if (prior_flag) JacobianPrior(pool);
  if (sensor_flag) JacobianSensor(pool);
  if (force_flag) JacobianForce(pool);

  // wait
  pool.WaitCount(count_begin + nprior + nsensor + nforce);

  // reset count
  pool.ResetCount();

  // timers
  timer_.jacobian_prior += mju_sum(timer_.prior_step.data(), nprior);
  timer_.jacobian_sensor += mju_sum(timer_.sensor_step.data(), nsensor);
  timer_.jacobian_force += mju_sum(timer_.force_step.data(), nforce);
  timer_.jacobian_total += GetDuration(timer_jacobian_start);
}

// prior cost
double Estimator::CostPrior(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // residual dimension
  int nv = model_->nv;
  int dim = model_->nv * configuration_length_;

  // residual
  ResidualPrior();

  // total scaling
  double scale = scale_prior / dim;

  // unpack
  double* r = residual_prior_.data();
  double* P =
      (band_prior ? weight_prior_band.data() : weight_prior.data());
  double* tmp = scratch0_prior_.data();

  // compute cost
  if (band_prior) {  // approximate covariance
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
  timer_.cost_prior += GetDuration(start);

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
    if (hessian && band_prior) {
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
        BlockFromMatrix(bbij, weight_prior.data(), nv, nv, dim, dim,
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
  if (hessian && !band_prior) {
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
  timer_.cost_prior_derivatives += GetDuration(start);

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
    double* qt_prior = configuration_previous.Get(t);
    double* qt = configuration.Get(t);

    // configuration difference
    mj_differentiatePos(model_, rt, 1.0, qt_prior, qt);
  }

  // stop timer
  timer_.residual_prior += GetDuration(start);
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
  double* qt = configuration.Get(index);
  double* qt_prior = configuration_previous.Get(index);
  double* block = block_prior_current_configuration_.Get(index);

  // compute Jacobian
  DifferentiateDifferentiatePos(NULL, block, model_, 1.0, qt_prior, qt);
}

// prior Jacobian
// note: pool wait is called outside this function
void Estimator::JacobianPrior(ThreadPool& pool) {
  // start index
  int start_index = reuse_data * mju_max(0, configuration_length_ - num_new_);

  // loop over predictions
  for (int t = 0; t < configuration_length_; t++) {
    // schedule by time step
    pool.Schedule([&estimator = *this, start_index, t]() {
      // start Jacobian timer
      auto jacobian_prior_start = std::chrono::steady_clock::now();

      // block
      if (t >= start_index) estimator.BlockPrior(t);

      // assemble
      if (!estimator.band_prior) estimator.SetBlockPrior(t);

      // stop Jacobian timer
      estimator.timer_.prior_step[t] = GetDuration(jacobian_prior_start);
    });
  }
}

// sensor cost
double Estimator::CostSensor(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // update dimension
  int dim_update = model_->nv * configuration_length_;
  int nv = model_->nv, ns = dim_sensor_;

  // residual
  ResidualSensor();

  // ----- cost ----- //

  // initialize
  double cost = 0.0;

  // zero memory
  if (gradient) mju_zero(gradient, dim_update);
  if (hessian) mju_zero(hessian, dim_update * dim_update);

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // time index
    int t = k + 1;

    // residual
    double* rk = residual_sensor_.data() + ns * k;

    // mask
    int* mask = sensor_mask.Get(t);

    // unpack block
    double* block = block_sensor_configurations_.Get(k);

    // shift
    int shift = 0;

    // loop over sensors
    for (int i = 0; i < num_sensor_; i++) {
      // start cost timer
      auto start_cost = std::chrono::steady_clock::now();

      // check mask, skip if missing measurement
      if (!mask[i]) continue;

      // dimension
      int nsi = model_->sensor_dim[i];

      // sensor residual
      double* rki = rk + shift;

      // weight
      double weight = scale_sensor[i] / nsi / prediction_length_;

      // parameters
      double* pi = norm_parameters_sensor.data() + MAX_NORM_PARAMETERS * i;

      // norm
      NormType normi = norm_sensor[i];

      // ----- cost ----- //
      cost += weight * Norm(gradient ? norm_gradient_sensor_.data() : NULL,
                            hessian ? norm_blocks_sensor_.data() : NULL, rki,
                            pi, nsi, normi);

      // stop cost timer
      timer_.cost_sensor += GetDuration(start_cost);

      // gradient wrt configuration: dridq012' * dndri
      if (gradient) {
        // sensor block
        double* blocki = block + (3 * nv) * shift;

        // scratch = dridq012' * dndri
        mju_mulMatTVec(scratch0_sensor_.data(), blocki,
                       norm_gradient_sensor_.data(), nsi, 3 * nv);

        // add
        mju_addToScl(gradient + k * nv, scratch0_sensor_.data(), weight,
                     3 * nv);
      }

      // Hessian (Gauss-Newton): drdq' * d2ndr2 * drdq
      if (hessian) {
        // sensor block
        double* blocki = block + (3 * nv) * shift;

        // step 1: tmp0 = d2ndri2 * dridq
        double* tmp0 = scratch0_sensor_.data();
        mju_mulMatMat(tmp0, norm_blocks_sensor_.data(), blocki, nsi, nsi,
                      3 * nv);

        // step 2: hessian = dridq' * tmp
        double* tmp1 = scratch1_sensor_.data();
        mju_mulMatTMat(tmp1, blocki, tmp0, nsi, 3 * nv, 3 * nv);

        // add
        AddBlockInMatrix(hessian, tmp1, weight, dim_update, dim_update, 3 * nv,
                         3 * nv, nv * k, nv * k);
      }

      // shift
      shift += nsi;
    }
  }

  // stop timer
  timer_.cost_sensor_derivatives += GetDuration(start);

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
    double* yt_sensor = sensor_measurement.Get(t);
    double* yt_model = sensor_prediction.Get(t);

    // sensor difference
    mju_sub(rk, yt_model, yt_sensor, dim_sensor_);
  }

  // stop timer
  timer_.residual_sensor += GetDuration(start);
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
  int start_index = reuse_data * mju_max(0, prediction_length_ - num_new_);

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // schedule by time step
    pool.Schedule([&estimator = *this, start_index, k]() {
      // start Jacobian timer
      auto jacobian_sensor_start = std::chrono::steady_clock::now();

      // block
      if (k >= start_index) estimator.BlockSensor(k);

      // assemble
      if (!estimator.band_prior) estimator.SetBlockSensor(k);

      // stop Jacobian timer
      estimator.timer_.sensor_step[k] = GetDuration(jacobian_sensor_start);
    });
  }
}

// compute force
// TODO(taylor): combine with Jacobian method
void Estimator::InverseDynamicsPrediction(ThreadPool& pool) {
  // compute sensor and force predictions
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model_->nq, nv = model_->nv, nu = model_->nu, ns = dim_sensor_;

  // start index
  int start_index = reuse_data * mju_max(0, prediction_length_ - num_new_);

  // pool count
  int count_before = pool.GetCount();

  // loop over predictions
  for (int k = start_index; k < prediction_length_; k++) {
    // schedule
    pool.Schedule([&estimator = *this, nq, nv, ns, nu, k]() {
      // time index
      int t = k + 1;

      // terms
      double* qt = estimator.configuration.Get(t);
      double* vt = estimator.velocity.Get(t);
      double* at = estimator.acceleration.Get(t);
      double* ct = estimator.ctrl.Get(t);

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
      double* st = estimator.sensor_prediction.Get(t);
      mju_copy(st, d->sensordata, ns);

      // copy force
      double* ft = estimator.force_prediction.Get(t);
      mju_copy(ft, d->qfrc_inverse, nv);
    });
  }

  // wait
  pool.WaitCount(count_before + (prediction_length_ - start_index));
  pool.ResetCount();

  // stop timer
  timer_.cost_prediction += GetDuration(start);
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
  int start_index = reuse_data * mju_max(0, prediction_length_ - num_new_);

  // loop over predictions
  for (int k = start_index; k < prediction_length_; k++) {
    // schedule
    pool.Schedule([&estimator = *this, nq, nv, nu, k]() {
      // time index
      int t = k + 1;

      // unpack
      double* q = estimator.configuration.Get(t);
      double* v = estimator.velocity.Get(t);
      double* a = estimator.acceleration.Get(t);
      double* c = estimator.ctrl.Get(t);

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
                    estimator.finite_difference.tolerance,
                    estimator.finite_difference.flg_actuation, dqdf, dvdf, dadf,
                    dqds, dvds, dads, NULL);
    });
  }

  // wait
  pool.WaitCount(count_before + (prediction_length_ - start_index));

  // reset pool count
  pool.ResetCount();

  // stop timer
  timer_.inverse_dynamics_derivatives += GetDuration(start);
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
  timer_.configuration_update += GetDuration(start);
}

// convert sequence of configurations to velocities and accelerations
void Estimator::ConfigurationToVelocityAcceleration() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model_->nv;

  // start index
  int start_index =
      reuse_data * mju_max(0, (configuration_length_ - 1) - num_new_);

  // loop over configurations
  for (int k = start_index; k < configuration_length_ - 1; k++) {
    // time index
    int t = k + 1;

    // previous and current configurations
    const double* q0 = configuration.Get(t - 1);
    const double* q1 = configuration.Get(t);

    // compute velocity
    double* v1 = velocity.Get(t);
    mj_differentiatePos(model_, v1, model_->opt.timestep, q0, q1);

    // compute acceleration
    if (t > 1) {
      // previous velocity
      const double* v0 = velocity.Get(t - 1);

      // compute acceleration
      double* a1 = acceleration.Get(t - 1);
      mju_sub(a1, v1, v0, nv);
      mju_scl(a1, a1, 1.0 / model_->opt.timestep, nv);
    }
  }

  // stop time
  timer_.cost_config_to_velacc += GetDuration(start);
}

// compute finite-difference velocity, acceleration derivatives
void Estimator::VelocityAccelerationDerivatives() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model_->nv;

  // start index
  int start_index =
      reuse_data * mju_max(0, (configuration_length_ - 1) - num_new_);

  // loop over configurations
  for (int k = start_index; k < configuration_length_ - 1; k++) {
    // time index
    int t = k + 1;

    // unpack
    double* q1 = configuration.Get(t - 1);
    double* q2 = configuration.Get(t);
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
  timer_.velacc_derivatives += GetDuration(start);
}

// compute total cost
// TODO(taylor): fix timers
double Estimator::Cost(double* gradient, double* hessian, ThreadPool& pool) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // evaluate configurations
  ConfigurationEvaluation(pool);

  // derivatives
  if (gradient || hessian) {
    ConfigurationDerivative(pool);
  }

  // pool count
  int count_begin = pool.GetCount();

  bool gradient_flag = (gradient ? true : false);
  bool hessian_flag = (hessian ? true : false);

  // individual derivatives
  if (prior_flag) {
    pool.Schedule([&estimator = *this, gradient_flag, hessian_flag]() {
      estimator.cost_prior = estimator.CostPrior(
          gradient_flag ? estimator.cost_gradient_prior_.data() : NULL,
          hessian_flag ? estimator.cost_hessian_prior_.data() : NULL);
    });
  }
  if (sensor_flag) {
    pool.Schedule([&estimator = *this, gradient_flag, hessian_flag]() {
      estimator.cost_sensor = estimator.CostSensor(
          gradient_flag ? estimator.cost_gradient_sensor_.data() : NULL,
          hessian_flag ? estimator.cost_hessian_sensor_.data() : NULL);
    });
  }
  if (force_flag) {
    pool.Schedule([&estimator = *this, gradient_flag, hessian_flag]() {
      estimator.cost_force = estimator.CostForce(
          gradient_flag ? estimator.cost_gradient_force_.data() : NULL,
          hessian_flag ? estimator.cost_hessian_force_.data() : NULL);
    });
  }

  // wait
  pool.WaitCount(count_begin + prior_flag + sensor_flag + force_flag);
  pool.ResetCount();

  // total cost
  double cost = cost_prior + cost_sensor + cost_force;

  // total gradient, hessian
  if (gradient) TotalGradient();
  if (hessian) TotalHessian();

  // counter
  cost_count_++;

  // stop timer
  timer_.cost += GetDuration(start);

  // total cost
  return cost;
}

// compute total gradient
void Estimator::TotalGradient() {
  // start gradient timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int dim = configuration_length_ * model_->nv;

  // unpack
  double* gradient = cost_gradient.data();

  // individual gradients
  if (prior_flag) {
    mju_copy(gradient, cost_gradient_prior_.data(), dim);
  } else {
    mju_zero(gradient, dim);
  }
  if (sensor_flag) mju_addTo(gradient, cost_gradient_sensor_.data(), dim);
  if (force_flag) mju_addTo(gradient, cost_gradient_force_.data(), dim);

  // stop gradient timer
  timer_.cost_gradient += GetDuration(start);
}

// compute total Hessian
void Estimator::TotalHessian() {
  // start Hessian timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int dim = configuration_length_ * model_->nv;

  // unpack
  double* hessian = cost_hessian.data();

  if (band_copy) {
    // zero memory
    mju_zero(hessian, dim * dim);

    // individual Hessians
    if (prior_flag)
      SymmetricBandMatrixCopy(hessian, cost_hessian_prior_.data(), model_->nv,
                              3, dim, configuration_length_, 0, 0, 0, 0,
                              scratch0_prior_.data());
    if (sensor_flag)
      SymmetricBandMatrixCopy(hessian, cost_hessian_sensor_.data(), model_->nv,
                              3, dim, configuration_length_, 0, 0, 0, 0,
                              scratch0_sensor_.data());
    if (force_flag)
      SymmetricBandMatrixCopy(hessian, cost_hessian_force_.data(), model_->nv,
                              3, dim, configuration_length_, 0, 0, 0, 0,
                              scratch0_force_.data());
  } else {
    // individual Hessians
    if (prior_flag) {
      mju_copy(hessian, cost_hessian_prior_.data(), dim * dim);
    } else {
      mju_zero(hessian, dim * dim);
    }
    if (sensor_flag)
      mju_addTo(hessian, cost_hessian_sensor_.data(), dim * dim);
    if (force_flag) mju_addTo(hessian, cost_hessian_force_.data(), dim * dim);
  }

  // stop Hessian timer
  timer_.cost_hessian += GetDuration(start);
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
  double* weight = weight_prior.data();

  // Hessian
  double* hessian = cost_hessian.data();

  // zero memory
  mju_zero(weight, ntotal * ntotal);

  // copy Hessian block to upper left
  if (configuration_length_ - num_new_ > 0 && update_prior_weight) {
    SymmetricBandMatrixCopy(weight, hessian, nv, nv, ntotal,
                            configuration_length_ - num_new_, 0, 0, num_new_,
                            num_new_, scratch_prior_weight_.data());
  }

  // set s * I to lower right
  for (int i = update_prior_weight * nv * (configuration_length_ - num_new_);
       i < ntotal; i++) {
    weight[ntotal * i + i] = scale_prior;
  }

  // stop timer
  timer_.prior_set_weight += GetDuration(start_set_weight);

  // stop timer
  timer_.prior_weight_update += GetDuration(start);

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

  // reset timers
  ResetTimers();

  // prior update
  PriorWeightUpdate(pool);

  // initial cost
  cost_count_ = 0;
  cost = Cost(NULL, NULL, pool);
  cost_initial = cost;

  // print initial cost
  PrintCost();

  // ----- smoother iterations ----- //

  // reset
  iterations_smoother_ = 0;
  iterations_line_search_ = 0;

  // iterations
  for (; iterations_smoother_ < max_smoother_iterations;
       iterations_smoother_++) {
    // ----- cost derivatives ----- //

    // // start timer (total cost derivatives)
    // auto cost_derivatives_start = std::chrono::steady_clock::now();

    // // configuration derivatives
    // ConfigurationDerivative(pool);

    // // -- cost derivatives -- //

    // // start timer
    // auto start_cost_total_derivatives = std::chrono::steady_clock::now();

    // // pool count
    // int count_begin = pool.GetCount();

    // // individual derivatives
    // if (prior_flag) {
    //   pool.Schedule([&estimator = *this]() {
    //     estimator.CostPrior(estimator.cost_gradient_prior_.data(),
    //                         estimator.cost_hessian_prior_.data());
    //   });
    // }
    // if (sensor_flag) {
    //   pool.Schedule([&estimator = *this]() {
    //     estimator.CostSensor(estimator.cost_gradient_sensor_.data(),
    //                          estimator.cost_hessian_sensor_.data());
    //   });
    // }
    // if (force_flag) {
    //   pool.Schedule([&estimator = *this]() {
    //     estimator.CostForce(estimator.cost_gradient_force_.data(),
    //                         estimator.cost_hessian_force_.data());
    //   });
    // }
    // // wait
    // pool.WaitCount(count_begin + prior_flag + sensor_flag + force_flag);

    // // pool reset
    // pool.ResetCount();

    Cost(cost_gradient.data(), cost_hessian.data(), pool);

    // reset num_new_
    num_new_ = configuration_length_;  // update all data now

    // stop timer
    // timer_.cost_total_derivatives +=
    // GetDuration(start_cost_total_derivatives);

    // gradient
    double* gradient = cost_gradient.data();
    // TotalGradient();

    // gradient tolerance check
    gradient_norm_ = mju_norm(gradient, nvar) / nvar;
    if (gradient_norm_ < gradient_tolerance) {
      break;
    }

    // Hessian
    // TotalHessian();

    // stop timer
    // timer_.cost_derivatives += GetDuration(cost_derivatives_start);

    // ----- line / curve search ----- //
    // start timer
    auto line_search_start = std::chrono::steady_clock::now();

    // copy configuration
    mju_copy(configuration_copy_.Data(), configuration.Data(), nconfig);

    // initialize
    double cost_candidate = cost;
    int iteration_search = 0;
    step_size_ = 1.0;
    regularization_ =
        regularization_initial;  // #(MIN_REGULARIZATION, regularization_);

    // initial search direction
    SearchDirection();

    // backtracking until cost decrease
    // TODO(taylor): Armijo, Wolfe conditions
    while (cost_candidate >= cost) {
      // check for max iterations
      if (iteration_search > max_line_search) {
        // reset configuration
        mju_copy(configuration.Data(), configuration_copy_.Data(), nconfig);

        // restore velocity, acceleration
        ConfigurationToVelocityAcceleration();

        // evaluate cost
        cost = Cost(NULL, NULL, pool);

        printf("line search failure\n");

        // failure
        return;
      }

      // search type
      if (iteration_search > 0) {
        switch (search_type) {
          case SearchType::kLineSearch:
            // decrease step size
            step_size_ *= step_scaling;
            break;
          case SearchType::kCurveSearch:
            // increase regularization
            regularization_ = mju_min(
                MAX_REGULARIZATION, regularization_ * regularization_scaling);
            // recompute search direction
            SearchDirection();
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
      cost_candidate = Cost(NULL, NULL, pool);

      // update iteration
      iteration_search++;
    }

    // increment
    iterations_line_search_ += iteration_search;

    // end timer
    timer_.search += GetDuration(line_search_start);

    // update cost
    cost = cost_candidate;

    // decrease regularization
    regularization_ =
        mju_max(MIN_REGULARIZATION, regularization_ / regularization_scaling);

    // print cost
    PrintCost();
  }

  // stop timer
  timer_.optimize = GetDuration(start_optimize);

  // status
  PrintOptimize();
}

// regularize Hessian
void Estimator::Regularize() {
  // dimension
  int nvar = configuration_length_ * model_->nv;

  // H + reg * I
  for (int j = 0; j < nvar; j++) {
    cost_hessian[j * nvar + j] += regularization_;
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
  double* gradient = cost_gradient.data();
  double* hessian = cost_hessian.data();
  double* hessian_band = cost_hessian_band_.data();

  // -- linear system solver -- //

  // select solver
  if (band_prior) {  // band solver
    // dense to band
    mju_dense2Band(hessian_band, cost_hessian.data(), ntotal, nband, ndense);

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
  timer_.search_direction += GetDuration(search_direction_start);
}

// print Optimize status
void Estimator::PrintOptimize() {
  if (!verbose_optimize) return;

  // title
  printf("Estimator::Optimize Status:\n\n");

  // timing
  printf("Timing:\n");

  PrintPriorWeightUpdate();

  printf("\n");
  printf("  cost (initial): %.3f (ms) \n", 1.0e-3 * timer_.cost / cost_count_);
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
  printf("      < gradient assemble: %.3f (ms) \n",
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
  printf("  iterations line search: %i\n", iterations_line_search_);
  printf("  iterations smoother: %i\n", iterations_smoother_);
  printf("\n");

  // cost
  printf("Cost:\n");
  printf("  initial: %.3f\n", cost_initial);
  printf("  final: %.3f\n", cost);
  printf("\n");
}

// print cost
void Estimator::PrintCost() {
  if (verbose_cost) {
    printf("cost (total): %.3f\n", cost);
    printf("  prior: %.3f\n", cost_prior);
    printf("  sensor: %.3f\n", cost_sensor);
    printf("  force: %.3f\n", cost_force);
    printf("  [initial: %.3f]\n", cost_initial);
  }
}

// print prior weight update status
// print Optimize status
void Estimator::PrintPriorWeightUpdate() {
  if (!verbose_prior) return;

  // timing
  printf("  prior weight update [total]: %.3f (ms) \n",
         1.0e-3 * timer_.prior_weight_update);
  printf("    - set weight: %.3f (ms) \n", 1.0e-3 * timer_.prior_set_weight);
  printf("\n");
}

// reset timers
void Estimator::ResetTimers() {
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

// initialize trajectories
void Estimator::InitializeTrajectories(
    const EstimatorTrajectory<double>& measurement,
    const EstimatorTrajectory<int>& measurement_mask,
    const EstimatorTrajectory<double>& ctrl,
    const EstimatorTrajectory<double>& time) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // set num new
  num_new_ = configuration_length_;

  // -- set initial configurations -- //

  // set first configuration
  double* q0 = configuration.Get(0);
  mju_copy(q0, qpos0_.data(), model_->nq);
  mj_integratePos(model_, q0, qvel0_.data(), -1.0 * model_->opt.timestep);

  // set second configuration
  configuration.Set(qpos0_.data(), 1);

  // set initial time
  this->time.Set(time.Get(0), 0);

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
    int buffer_index = time.Length() - (configuration_length_ - 1) + i;

    // get time
    this->time.Set(&data->time, i);

    // set/get ctrl
    const double* ui = ctrl.Get(buffer_index);
    this->ctrl.Set(ui, i);
    mju_copy(data->ctrl, ui, model_->nu);

    // step dynamics
    mj_step(model_, data);

    // set measurement
    const double* yi = measurement.Get(buffer_index);
    sensor_measurement.Set(yi, i);

    // set mask
    const int* mi = measurement_mask.Get(buffer_index);
    sensor_mask.Set(mi, i);

    // copy qfrc_actuator
    force_measurement.Set(data->qfrc_actuator, i);

    // copy configuration
    configuration.Set(data->qpos, i + 1);
  }

  // set last time
  this->time.Set(&data->time, configuration_length_ - 1);

  // copy configuration to prior
  mju_copy(configuration_previous.Data(), configuration.Data(),
           model_->nq * configuration_length_);

  // stop timer
  timer_.update_trajectory += GetDuration(start);
}

// update trajectories
int Estimator::UpdateTrajectories_(
    int num_new, const EstimatorTrajectory<double>& measurement,
    const EstimatorTrajectory<int>& measurement_mask,
    const EstimatorTrajectory<double>& ctrl,
    const EstimatorTrajectory<double>& time) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // set number of new elements
  num_new_ = num_new;

  // shift trajectory heads
  Shift(num_new);

  // get data
  mjData* data = data_[0].get();

  // set new measurements, ctrl -> qfrc_actuator, rollout new configurations,
  // new time
  for (int i = 0; i < num_new; i++) {
    // time index
    int t = i + configuration_length_ - num_new - 1;

    // buffer index
    int b = i + measurement.Length() - num_new;

    // set measurement
    const double* yi = measurement.Get(b);
    sensor_measurement.Set(yi, t);

    // set measurement mask
    const int* mi = measurement_mask.Get(b);
    sensor_mask.Set(mi, t);

    // set time
    const double* ti = time.Get(b);
    this->time.Set(ti, t);

    // ----- forward dynamics ----- //

    // set ctrl
    const double* ui = ctrl.Get(b);
    this->ctrl.Set(ui, t);
    mju_copy(data->ctrl, ui, model_->nu);

    // set qpos
    double* q0 = configuration.Get(t - 1);
    double* q1 = configuration.Get(t);
    mju_copy(data->qpos, q1, model_->nq);

    // set qvel
    mj_differentiatePos(model_, data->qvel, model_->opt.timestep, q0, q1);

    // set time
    data->time = time.Get(b)[0];

    // step dynamics
    mj_step(model_, data);

    // copy qfrc_actuator
    force_measurement.Set(data->qfrc_actuator, t);

    // copy configuration
    configuration.Set(data->qpos, t + 1);
  }

  // set last time
  this->time.Set(&data->time, configuration_length_ - 1);

  // copy configuration to prior
  mju_copy(configuration_previous.Data(), configuration.Data(),
           model_->nq * configuration_length_);

  // stop timer
  timer_.update_trajectory += GetDuration(start);

  return num_new;
}

// update trajectories
int Estimator::UpdateTrajectories(
    const EstimatorTrajectory<double>& measurement,
    const EstimatorTrajectory<int>& measurement_mask,
    const EstimatorTrajectory<double>& ctrl,
    const EstimatorTrajectory<double>& time) {
  // lastest buffer time
  double time_buffer_last = *time.Get(time.Length() - 1);

  // latest estimator time
  double time_estimator_last =
      *time.Get(time.Length() - 2);  // index to latest measurement time

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
      InitializeTrajectories(buffer.sensor, buffer.sensor_mask, buffer.ctrl,
                             buffer.time);
      initialized_ = true;
    } else {
      num_new_ = UpdateTrajectories(buffer.sensor, buffer.sensor_mask,
                                    buffer.ctrl, buffer.time);
    }
    num_new = num_new_;

    // optimize
    Optimize(pool);
  }
  return num_new;
}

// force cost
double Estimator::CostForce(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // update dimension
  int dim_update = model_->nv * configuration_length_;
  int nv = model_->nv;

  // residual
  ResidualForce();

  // ----- cost ----- //

  // initialize
  double cost = 0.0;

  // zero memory
  if (gradient) mju_zero(gradient, dim_update);
  if (hessian) mju_zero(hessian, dim_update * dim_update);

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // unpack block
    double* block = block_force_configurations_.Get(k);

    // start cost timer
    auto start_cost = std::chrono::steady_clock::now();

    // residual
    double* rk = residual_force_.data() + k * nv;

    // weight
    double weight = scale_force[0] / nv / prediction_length_;

    // parameters
    double* pk = norm_parameters_force.data() + MAX_NORM_PARAMETERS * 0;

    // norm
    NormType normk = norm_force[0];

    // ----- cost ----- //
    cost += weight * Norm(gradient ? norm_gradient_force_.data() : NULL,
                          hessian ? norm_blocks_force_.data() : NULL, rk, pk,
                          nv, normk);

    // stop cost timer
    timer_.cost_force += GetDuration(start_cost);

    // gradient wrt configuration: dridq012' * dndri
    if (gradient) {
      // scratch = dridq012' * dndri
      mju_mulMatTVec(scratch0_force_.data(), block, norm_gradient_force_.data(),
                     nv, 3 * nv);

      // add
      mju_addToScl(gradient + k * nv, scratch0_force_.data(), weight, 3 * nv);
    }

    // Hessian (Gauss-Newton): drdq' * d2ndr2 * drdq
    if (hessian) {
      // step 1: tmp0 = d2ndri2 * dridq
      double* tmp0 = scratch0_force_.data();
      mju_mulMatMat(tmp0, norm_blocks_force_.data(), block, nv, nv, 3 * nv);

      // step 2: hessian = dridq' * tmp
      double* tmp1 = scratch1_force_.data();
      mju_mulMatTMat(tmp1, block, tmp0, nv, 3 * nv, 3 * nv);

      // add
      AddBlockInMatrix(hessian, tmp1, weight, dim_update, dim_update, 3 * nv,
                       3 * nv, nv * k, nv * k);
    }
  }

  // stop timer
  timer_.cost_force_derivatives += GetDuration(start);

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
    double* ft_actuator = force_measurement.Get(t);
    double* ft_inverse = force_prediction.Get(t);

    // force difference
    mju_sub(rk, ft_inverse, ft_actuator, nv);
  }

  // stop timer
  timer_.residual_force += GetDuration(start);
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
  SetBlockInMatrix(jacobian_force_.data(), dfdq2, 1.0, dim_residual, dim_update,
                   nv, nv, row, col_next);
}

// force Jacobian blocks (dfdq0, dfdq1, dfdq2)
void Estimator::BlockForce(int index) {
  // dimensions
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
}

// force Jacobian
// note: pool wait is called outside this function
void Estimator::JacobianForce(ThreadPool& pool) {
  // start index
  int start_index = reuse_data * mju_max(0, prediction_length_ - num_new_);

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // schedule by time step
    pool.Schedule([&estimator = *this, start_index, k]() {
      // start Jacobian timer
      auto jacobian_force_start = std::chrono::steady_clock::now();

      // block
      if (k >= start_index) estimator.BlockForce(k);

      // assemble
      if (!estimator.band_prior) estimator.SetBlockForce(k);

      // stop Jacobian timer
      estimator.timer_.force_step[k] = GetDuration(jacobian_force_start);
    });
  }
}

}  // namespace mjpc
