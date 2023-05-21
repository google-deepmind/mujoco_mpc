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

#include <chrono>

#include "mjpc/norm.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

// initialize estimator
void Estimator::Initialize(mjModel* model) {
  // model
  model_ = model;

  // delete mjData instances since model might have changed.
  data_.clear();

  // allocate one mjData for nominal.
  ResizeMjData(model_, MAX_HISTORY); // TODO(taylor): set to 1, fix segfault...

  // dimension
  int nq = model->nq, nv = model->nv;

  // trajectories
  configuration_length_ = GetNumberOrDefault(10, model, "batch_length");
  configuration_.resize(nq * MAX_HISTORY);
  velocity_.resize(nv * MAX_HISTORY);
  acceleration_.resize(nv * MAX_HISTORY);
  time_.resize(MAX_HISTORY);

  // prior
  configuration_prior_.resize(nq * MAX_HISTORY);

  // sensor
  dim_sensor_ = model->nsensordata;  // TODO(taylor): grab from model
  sensor_measurement_.resize(dim_sensor_ * MAX_HISTORY);
  sensor_prediction_.resize(dim_sensor_ * MAX_HISTORY);

  // force
  force_measurement_.resize(nv * MAX_HISTORY);
  force_prediction_.resize(nv * MAX_HISTORY);

  // residual
  residual_prior_.resize(nv * MAX_HISTORY);
  residual_sensor_.resize(dim_sensor_ * MAX_HISTORY);
  residual_force_.resize(nv * MAX_HISTORY);

  // Jacobian
  jacobian_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  jacobian_sensor_.resize((dim_sensor_ * MAX_HISTORY) * (nv * MAX_HISTORY));
  jacobian_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // prior Jacobian block
  block_prior_configuration_.resize((nv * nv) * MAX_HISTORY);

  // sensor Jacobian blocks
  block_sensor_configuration_.resize((dim_sensor_ * nv) * MAX_HISTORY);
  block_sensor_velocity_.resize((dim_sensor_ * nv) * MAX_HISTORY);
  block_sensor_acceleration_.resize((dim_sensor_ * nv) * MAX_HISTORY);
  block_sensor_scratch_.resize((dim_sensor_ * nv));

  // force Jacobian blocks
  block_force_configuration_.resize((nv * nv) * MAX_HISTORY);
  block_force_velocity_.resize((nv * nv) * MAX_HISTORY);
  block_force_acceleration_.resize((nv * nv) * MAX_HISTORY);
  block_force_scratch_.resize((nv * nv));

  // velocity Jacobian blocks
  block_velocity_previous_configuration_.resize((nv * nv) * MAX_HISTORY);
  block_velocity_current_configuration_.resize((nv * nv) * MAX_HISTORY);

  // acceleration Jacobian blocks
  block_acceleration_previous_configuration_.resize((nv * nv) * MAX_HISTORY);
  block_acceleration_current_configuration_.resize((nv * nv) * MAX_HISTORY);
  block_acceleration_next_configuration_.resize((nv * nv) * MAX_HISTORY);

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

  // weight TODO(taylor): matrices
  weight_prior_ = GetNumberOrDefault(1.0, model, "batch_weight_prior");
  weight_sensor_ = GetNumberOrDefault(1.0, model, "batch_weight_sensor");
  weight_force_ = GetNumberOrDefault(1.0, model, "batch_weight_force");

  // cost norms
  norm_prior_ = (NormType)GetNumberOrDefault(0, model, "batch_norm_prior");
  norm_sensor_ = (NormType)GetNumberOrDefault(0, model, "batch_norm_sensor");
  norm_force_ = (NormType)GetNumberOrDefault(0, model, "batch_norm_force");

  // cost norm parameters
  norm_parameters_prior_.resize(3);
  norm_parameters_sensor_.resize(3);
  norm_parameters_force_.resize(3);

  // norm gradient
  norm_gradient_prior_.resize(nv * MAX_HISTORY);
  norm_gradient_sensor_.resize(dim_sensor_ * MAX_HISTORY);
  norm_gradient_force_.resize(nv * MAX_HISTORY);

  // norm Hessian
  norm_hessian_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  norm_hessian_sensor_.resize((dim_sensor_ * MAX_HISTORY) *
                              (dim_sensor_ * MAX_HISTORY));
  norm_hessian_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // cost scratch
  cost_scratch_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_scratch_sensor_.resize((dim_sensor_ * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_scratch_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // candidate
  configuration_copy_.resize(nq * MAX_HISTORY);

  // search direction
  search_direction_.resize(nv * MAX_HISTORY);

  // reset
  Reset();
}

// reset memory
void Estimator::Reset() {
  // trajectories
  std::fill(configuration_.begin(), configuration_.end(), 0.0);
  std::fill(velocity_.begin(), velocity_.end(), 0.0);
  std::fill(acceleration_.begin(), acceleration_.end(), 0.0);
  std::fill(time_.begin(), time_.end(), 0.0);

  // prior
  std::fill(configuration_prior_.begin(), configuration_prior_.end(), 0.0);

  // sensor
  std::fill(sensor_measurement_.begin(), sensor_measurement_.end(), 0.0);
  std::fill(sensor_prediction_.begin(), sensor_prediction_.end(), 0.0);

  // force
  std::fill(force_measurement_.begin(), force_measurement_.end(), 0.0);
  std::fill(force_prediction_.begin(), force_prediction_.end(), 0.0);

  // residual
  std::fill(residual_prior_.begin(), residual_prior_.end(), 0.0);
  std::fill(residual_sensor_.begin(), residual_sensor_.end(), 0.0);
  std::fill(residual_force_.begin(), residual_force_.end(), 0.0);

  // Jacobian
  std::fill(jacobian_prior_.begin(), jacobian_prior_.end(), 0.0);
  std::fill(jacobian_sensor_.begin(), jacobian_sensor_.end(), 0.0);
  std::fill(jacobian_force_.begin(), jacobian_force_.end(), 0.0);

  // prior Jacobian block
  std::fill(block_prior_configuration_.begin(),
            block_prior_configuration_.end(), 0.0);

  // sensor Jacobian blocks
  std::fill(block_sensor_configuration_.begin(),
            block_sensor_configuration_.end(), 0.0);
  std::fill(block_sensor_velocity_.begin(), block_sensor_velocity_.end(), 0.0);
  std::fill(block_sensor_acceleration_.begin(),
            block_sensor_acceleration_.end(), 0.0);
  std::fill(block_sensor_scratch_.begin(), block_sensor_scratch_.end(), 0.0);

  // force Jacobian blocks
  std::fill(block_force_configuration_.begin(),
            block_force_configuration_.end(), 0.0);
  std::fill(block_force_velocity_.begin(), block_force_velocity_.end(), 0.0);
  std::fill(block_force_acceleration_.begin(), block_force_acceleration_.end(),
            0.0);
  std::fill(block_force_scratch_.begin(), block_force_scratch_.end(), 0.0);

  // velocity Jacobian blocks
  std::fill(block_velocity_previous_configuration_.begin(),
            block_velocity_previous_configuration_.end(), 0.0);
  std::fill(block_velocity_current_configuration_.begin(),
            block_velocity_current_configuration_.end(), 0.0);

  // acceleration Jacobian blocks
  std::fill(block_acceleration_previous_configuration_.begin(),
            block_acceleration_previous_configuration_.end(), 0.0);
  std::fill(block_acceleration_current_configuration_.begin(),
            block_acceleration_current_configuration_.end(), 0.0);
  std::fill(block_acceleration_next_configuration_.begin(),
            block_acceleration_next_configuration_.end(), 0.0);

  // cost
  cost_prior_ = 0.0;
  cost_sensor_ = 0.0;
  cost_force_ = 0.0;
  cost_ = 0.0;

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

  // norm gradient
  std::fill(norm_gradient_prior_.begin(), norm_gradient_prior_.end(), 0.0);
  std::fill(norm_gradient_sensor_.begin(), norm_gradient_sensor_.end(), 0.0);
  std::fill(norm_gradient_force_.begin(), norm_gradient_force_.end(), 0.0);

  // norm Hessian
  std::fill(norm_hessian_prior_.begin(), norm_hessian_prior_.end(), 0.0);
  std::fill(norm_hessian_sensor_.begin(), norm_hessian_sensor_.end(), 0.0);
  std::fill(norm_hessian_force_.begin(), norm_hessian_force_.end(), 0.0);

  // cost scratch
  std::fill(cost_scratch_prior_.begin(), cost_scratch_prior_.end(), 0.0);
  std::fill(cost_scratch_sensor_.begin(), cost_scratch_sensor_.end(), 0.0);
  std::fill(cost_scratch_force_.begin(), cost_scratch_force_.end(), 0.0);

  // candidate
  std::fill(configuration_copy_.begin(), configuration_copy_.end(), 0.0);

  // search direction
  std::fill(search_direction_.begin(), search_direction_.end(), 0.0);

  // timing
  timer_total_ = 0.0;
  timer_inverse_dynamics_derivatives_ = 0.0;
  timer_velacc_derivatives_ = 0.0;
  timer_jacobian_prior_ = 0.0;
  timer_jacobian_sensor_ = 0.0;
  timer_jacobian_force_ = 0.0;
  timer_cost_prior_derivatives_ = 0.0;
  timer_cost_sensor_derivatives_ = 0.0;
  timer_cost_force_derivatives_ = 0.0;
  timer_cost_gradient_ = 0.0;
  timer_cost_hessian_ = 0.0;
  timer_cost_derivatives_ = 0.0;
  timer_search_direction_ = 0.0;
  timer_line_search_ = 0.0;

  // status
  iterations_smoother_ = 0;
  iterations_line_search_ = 0;
}

// prior cost TODO(taylor): normalize by dimension
double Estimator::CostPrior(double* gradient, double* hessian) {
  // residual dimension
  int dim = model_->nv * configuration_length_;

  // compute cost
  double cost =
      Norm(gradient ? norm_gradient_prior_.data() : NULL,
           hessian ? norm_hessian_prior_.data() : NULL, residual_prior_.data(),
           norm_parameters_prior_.data(), dim, norm_prior_);
  cost *= weight_prior_;  // TODO(taylor): weight -> matrix

  // compute cost gradient wrt configuration
  if (gradient) {
    // scale gradient by weight
    mju_scl(norm_gradient_prior_.data(), norm_gradient_prior_.data(),
            weight_prior_, dim);

    // compute total gradient wrt configuration: drdq' * dndr
    mju_mulMatTVec(gradient, jacobian_prior_.data(),
                   norm_gradient_prior_.data(), dim, dim);
  }

  // compute cost Hessian wrt configuration
  if (hessian) {
    // scale Hessian by weight
    mju_scl(norm_hessian_prior_.data(), norm_hessian_prior_.data(),
            weight_prior_, dim * dim);

    // compute Gauss-Newton Hessian: drdq' * d2ndr2 * drdq

    // step 1: scratch = d2ndr2 * drdq
    mju_mulMatMat(cost_scratch_prior_.data(), norm_hessian_prior_.data(),
                  jacobian_prior_.data(), dim, dim, dim);

    // step 2: hessian = drdq' * scratch
    mju_mulMatTMat(hessian, jacobian_prior_.data(), cost_scratch_prior_.data(),
                   dim, dim, dim);
  }

  return cost;
}

// prior residual
void Estimator::ResidualPrior() {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // loop over configurations
  for (int t = 0; t < configuration_length_; t++) {
    // terms
    double* rt = residual_prior_.data() + t * nv;
    double* qt_prior = configuration_prior_.data() + t * nq;
    double* qt = configuration_.data() + t * nq;

    // configuration difference
    mj_differentiatePos(model_, rt, 1.0, qt_prior, qt);
  }
}

// prior Jacobian
void Estimator::JacobianPrior() {
  // dimension
  int nv = model_->nv, dim = model_->nv * configuration_length_;

  // reset Jacobian to zero
  mju_zero(jacobian_prior_.data(), dim * dim);

  // loop over configurations
  for (int t = 0; t < configuration_length_; t++) {
    // unpack
    double* block = block_prior_configuration_.data() + t * nv * nv;

    // set block in matrix
    SetMatrixInMatrix(jacobian_prior_.data(), block, 1.0, dim, dim, nv, nv,
                      t * nv, t * nv);
  }
}

// prior Jacobian blocks
void Estimator::JacobianPriorBlocks() {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // loop over configurations
  for (int t = 0; t < configuration_length_; t++) {
    // unpack
    double* qt = configuration_.data() + t * nq;
    double* qt_prior = configuration_prior_.data() + t * nq;
    double* block = block_prior_configuration_.data() + t * nv * nv;

    // compute Jacobian
    DifferentiateDifferentiatePos(NULL, block, model_, 1.0, qt_prior, qt);
  }
}

// sensor cost TODO(taylor): normalize by dimension
double Estimator::CostSensor(double* gradient, double* hessian) {
  // residual dimension
  int dim_residual = dim_sensor_ * (configuration_length_ - 2);

  // update dimension
  int dim_update = model_->nv * configuration_length_;

  // compute cost
  double cost = Norm(gradient ? norm_gradient_sensor_.data() : NULL,
                     hessian ? norm_hessian_sensor_.data() : NULL,
                     residual_sensor_.data(), norm_parameters_sensor_.data(),
                     dim_residual, norm_sensor_);
  cost *= weight_sensor_;  // TODO(taylor): weight -> matrix

  // compute cost gradient wrt configuration
  if (gradient) {
    // scale gradient by weight
    mju_scl(norm_gradient_sensor_.data(), norm_gradient_sensor_.data(),
            weight_sensor_, dim_residual);

    // compute total gradient wrt configuration: drdq' * dndr
    mju_mulMatTVec(gradient, jacobian_sensor_.data(),
                   norm_gradient_sensor_.data(), dim_residual, dim_update);
  }

  // compute cost Hessian wrt configuration
  if (hessian) {
    // scale Hessian by weight
    mju_scl(norm_hessian_sensor_.data(), norm_hessian_sensor_.data(),
            weight_sensor_, dim_residual * dim_residual);

    // compute Gauss-Newton Hessian: drdq' * d2ndr2 * drdq

    // step 1: scratch = d2ndr2 * drdq
    mju_mulMatMat(cost_scratch_sensor_.data(), norm_hessian_sensor_.data(),
                  jacobian_sensor_.data(), dim_residual, dim_residual,
                  dim_update);

    // step 2: hessian = drdq' * scratch
    mju_mulMatTMat(hessian, jacobian_sensor_.data(),
                   cost_scratch_sensor_.data(), dim_residual, dim_update,
                   dim_update);
  }

  return cost;
}

// sensor residual
void Estimator::ResidualSensor() {
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // terms
    double* rt = residual_sensor_.data() + t * dim_sensor_;
    double* yt_sensor = sensor_measurement_.data() + t * dim_sensor_;
    double* yt_model = sensor_prediction_.data() + t * dim_sensor_;

    // sensor difference
    mju_sub(rt, yt_model, yt_sensor, dim_sensor_);
  }
}

// sensor Jacobian
void Estimator::JacobianSensor() {
  // velocity dimension
  int nv = model_->nv;

  // residual dimension
  int dim_residual = dim_sensor_ * (configuration_length_ - 2);

  // update dimension
  int dim_update = nv * configuration_length_;

  // reset Jacobian to zero
  mju_zero(jacobian_sensor_.data(), dim_residual * dim_update);

  // loop over sensors
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // dqds
    double* dqds = block_sensor_configuration_.data() + t * dim_sensor_ * nv;

    // dvds
    double* dvds = block_sensor_velocity_.data() + t * dim_sensor_ * nv;

    // dads
    double* dads = block_sensor_acceleration_.data() + t * dim_sensor_ * nv;

    // indices
    int row = t * dim_sensor_;
    int col_previous = t * nv;
    int col_current = (t + 1) * nv;
    int col_next = (t + 2) * nv;

    // ----- configuration previous ----- //
    // dvds' * dvdq0
    double* dvdq0 = block_velocity_previous_configuration_.data() + t * nv * nv;
    mju_mulMatTMat(block_sensor_scratch_.data(), dvds, dvdq0, nv, dim_sensor_,
                   nv);
    AddMatrixInMatrix(jacobian_sensor_.data(), block_sensor_scratch_.data(),
                      1.0, dim_residual, dim_update, dim_sensor_, nv, row,
                      col_previous);

    // dads' * dadq0
    double* dadq0 =
        block_acceleration_previous_configuration_.data() + t * nv * nv;
    mju_mulMatTMat(block_sensor_scratch_.data(), dads, dadq0, nv, dim_sensor_,
                   nv);
    AddMatrixInMatrix(jacobian_sensor_.data(), block_sensor_scratch_.data(),
                      1.0, dim_residual, dim_update, dim_sensor_, nv, row,
                      col_previous);

    // ----- configuration current ----- //
    // dqds
    mju_transpose(block_sensor_scratch_.data(), dqds, nv, dim_sensor_);
    AddMatrixInMatrix(jacobian_sensor_.data(), block_sensor_scratch_.data(),
                      1.0, dim_residual, dim_update, dim_sensor_, nv, row,
                      col_current);

    // dvds' * dvdq1
    double* dvdq1 = block_velocity_current_configuration_.data() + t * nv * nv;
    mju_mulMatTMat(block_sensor_scratch_.data(), dvds, dvdq1, nv, dim_sensor_,
                   nv);
    AddMatrixInMatrix(jacobian_sensor_.data(), block_sensor_scratch_.data(),
                      1.0, dim_residual, dim_update, dim_sensor_, nv, row,
                      col_current);

    // dads' * dadq1
    double* dadq1 =
        block_acceleration_current_configuration_.data() + t * nv * nv;
    mju_mulMatTMat(block_sensor_scratch_.data(), dads, dadq1, nv, dim_sensor_,
                   nv);
    AddMatrixInMatrix(jacobian_sensor_.data(), block_sensor_scratch_.data(),
                      1.0, dim_residual, dim_update, dim_sensor_, nv, row,
                      col_current);

    // ----- configuration next ----- //

    // dads' * dadq2
    double* dadq2 = block_acceleration_next_configuration_.data() + t * nv * nv;
    mju_mulMatTMat(block_sensor_scratch_.data(), dads, dadq2, nv, dim_sensor_,
                   nv);
    AddMatrixInMatrix(jacobian_sensor_.data(), block_sensor_scratch_.data(),
                      1.0, dim_residual, dim_update, dim_sensor_, nv, row,
                      col_next);
  }
}

// force cost TODO(taylor): normalize by dimension
double Estimator::CostForce(double* gradient, double* hessian) {
  // residual dimension
  int dim_residual = model_->nv * (configuration_length_ - 2);

  // update dimension
  int dim_update = model_->nv * configuration_length_;

  // compute cost
  double cost =
      Norm(gradient ? norm_gradient_force_.data() : NULL,
           hessian ? norm_hessian_force_.data() : NULL, residual_force_.data(),
           norm_parameters_force_.data(), dim_residual, norm_force_);
  cost *= weight_force_;  // TODO(taylor): weight -> matrix

  // compute cost gradient wrt configuration
  if (gradient) {
    // scale gradient by weight
    mju_scl(norm_gradient_force_.data(), norm_gradient_force_.data(),
            weight_force_, dim_residual);

    // compute total gradient wrt configuration: drdq' * dndr
    mju_mulMatTVec(gradient, jacobian_force_.data(),
                   norm_gradient_force_.data(), dim_residual, dim_update);
  }

  // compute cost Hessian wrt configuration
  if (hessian) {
    // scale Hessian by weight
    mju_scl(norm_hessian_force_.data(), norm_hessian_force_.data(),
            weight_force_, dim_residual * dim_residual);

    // compute total Hessian (Gauss-Newton approximation):
    // hessian = drdq * d2ndr2 * drdq

    // step 1: scratch = d2ndr2 * drdq
    mju_mulMatMat(cost_scratch_force_.data(), norm_hessian_force_.data(),
                  jacobian_force_.data(), dim_residual, dim_residual,
                  dim_update);

    // step 2: hessian = drdq' * scratch
    mju_mulMatTMat(hessian, jacobian_force_.data(), cost_scratch_force_.data(),
                   dim_residual, dim_update, dim_update);
  }

  return cost;
}

// force residual
void Estimator::ResidualForce() {
  // dimension
  int nv = model_->nv;

  // loop over force
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // terms
    double* rt = residual_force_.data() + t * nv;
    double* ft_actuator = force_measurement_.data() + t * nv;
    double* ft_inverse_ = force_prediction_.data() + t * nv;

    // force difference
    mju_sub(rt, ft_inverse_, ft_actuator, nv);
  }
}

// force Jacobian
void Estimator::JacobianForce() {
  // velocity dimension
  int nv = model_->nv;

  // residual dimension
  int dim_residual = nv * (configuration_length_ - 2);

  // update dimension
  int dim_update = nv * configuration_length_;

  // reset Jacobian to zero
  mju_zero(jacobian_force_.data(), dim_residual * dim_update);

  // loop over force
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // dqdf
    double* dqdf = block_force_configuration_.data() + t * nv * nv;

    // dvdf
    double* dvdf = block_force_velocity_.data() + t * nv * nv;

    // dadf
    double* dadf = block_force_acceleration_.data() + t * nv * nv;

    // indices
    int row = t * nv;
    int col_previous = t * nv;
    int col_current = (t + 1) * nv;
    int col_next = (t + 2) * nv;

    // ----- configuration previous ----- //
    // dvdf' * dvdq0
    double* dvdq0 = block_velocity_previous_configuration_.data() + t * nv * nv;
    mju_mulMatTMat(block_force_scratch_.data(), dvdf, dvdq0, nv, nv, nv);
    AddMatrixInMatrix(jacobian_force_.data(), block_force_scratch_.data(), 1.0,
                      dim_residual, dim_update, nv, nv, row, col_previous);

    // dadf' * dadq0
    double* dadq0 =
        block_acceleration_previous_configuration_.data() + t * nv * nv;
    mju_mulMatTMat(block_force_scratch_.data(), dadf, dadq0, nv, nv, nv);
    AddMatrixInMatrix(jacobian_force_.data(), block_force_scratch_.data(), 1.0,
                      dim_residual, dim_update, nv, nv, row, col_previous);

    // ----- configuration current ----- //
    // dqdf'
    mju_transpose(block_force_scratch_.data(), dqdf, nv, nv);
    AddMatrixInMatrix(jacobian_force_.data(), block_force_scratch_.data(), 1.0,
                      dim_residual, dim_update, nv, nv, row, col_current);

    // dvdf' * dvdq1
    double* dvdq1 = block_velocity_current_configuration_.data() + t * nv * nv;
    mju_mulMatTMat(block_force_scratch_.data(), dvdf, dvdq1, nv, nv, nv);
    AddMatrixInMatrix(jacobian_force_.data(), block_force_scratch_.data(), 1.0,
                      dim_residual, dim_update, nv, nv, row, col_current);

    // dadf' * dadq1
    double* dadq1 =
        block_acceleration_current_configuration_.data() + t * nv * nv;
    mju_mulMatTMat(block_force_scratch_.data(), dadf, dadq1, nv, nv, nv);
    AddMatrixInMatrix(jacobian_force_.data(), block_force_scratch_.data(), 1.0,
                      dim_residual, dim_update, nv, nv, row, col_current);

    // ----- configuration next ----- //

    // dadf' * dadq2
    double* dadq2 = block_acceleration_next_configuration_.data() + t * nv * nv;
    mju_mulMatTMat(block_force_scratch_.data(), dadf, dadq2, nv, nv, nv);
    AddMatrixInMatrix(jacobian_force_.data(), block_force_scratch_.data(), 1.0,
                      dim_residual, dim_update, nv, nv, row, col_next);
  }
}

// compute force
void Estimator::InverseDynamicsPrediction(ThreadPool& pool) {
  // dimension
  int nq = model_->nq, nv = model_->nv, ns = dim_sensor_;

  // threadpool current count
  int count_before = pool.GetCount();

  // loop over time steps
  for (int t = 0; t < configuration_length_ - 2; t++) {
    pool.Schedule(
        [nq, nv, ns, t, &configuration = this->configuration_,
         &velocity = this->velocity_, &acceleration = this->acceleration_,
         &sensor = this->sensor_prediction_, &force = this->force_prediction_,
         &model = this->model_, &data = this->data_]() {
          // terms
          double* qt = configuration.data() + (t + 1) * nq;
          double* vt = velocity.data() + t * nv;
          double* at = acceleration.data() + t * nv;

          // data
          mjData* d = data[ThreadPool::WorkerId()].get();

          // set qt, vt, at
          mju_copy(d->qpos, qt, nq);
          mju_copy(d->qvel, vt, nv);
          mju_copy(d->qacc, at, nv);

          // inverse dynamics
          mj_inverse(model, d);

          // copy sensor
          double* st = sensor.data() + t * ns;
          mju_copy(st, d->sensordata, ns);

          // copy force
          double* ft = force.data() + t * nv;
          mju_copy(ft, d->qfrc_inverse, nv);
        });
  }
  // wait for all time steps
  pool.WaitCount(count_before + configuration_length_ - 2);

  // reset pool count
  pool.ResetCount();
}

// compute inverse dynamics derivatives (via finite difference)
void Estimator::InverseDynamicsDerivatives(ThreadPool& pool) {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // threadpool current count
  int count_before = pool.GetCount();

  // loop over (state, acceleration)
  for (int t = 0; t < configuration_length_ - 2; t++) {
    pool.Schedule([nq, nv, &ns = this->dim_sensor_, t,
                   &configuration = this->configuration_,
                   &velocity = this->velocity_,
                   &acceleration = this->acceleration_,
                   &sq = this->block_sensor_configuration_,
                   &sv = this->block_sensor_velocity_,
                   &sa = this->block_sensor_acceleration_,
                   &fq = this->block_force_configuration_,
                   &fv = this->block_force_velocity_,
                   &fa = this->block_force_acceleration_, &model = this->model_,
                   &data = this->data_, &fd = this->finite_difference_]() {
      // unpack
      double* q = configuration.data() + (t + 1) * nq;
      double* v = velocity.data() + t * nv;
      double* a = acceleration.data() + t * nv;
      double* dqds = sq.data() + t * ns * nv;
      double* dvds = sv.data() + t * ns * nv;
      double* dads = sa.data() + t * ns * nv;
      double* dqdf = fq.data() + t * nv * nv;
      double* dvdf = fv.data() + t * nv * nv;
      double* dadf = fa.data() + t * nv * nv;

      // set (state, acceleration)
      mju_copy(data[ThreadPool::WorkerId()]->qpos, q, nq);
      mju_copy(data[ThreadPool::WorkerId()]->qvel, v, nv);
      mju_copy(data[ThreadPool::WorkerId()]->qacc, a, nv);

      // finite-difference derivatives
      mjd_inverseFD(model, data[ThreadPool::WorkerId()].get(), fd.tolerance,
                    fd.flg_actuation, dqdf, dvdf, dadf, dqds, dvds, dads, NULL);
    });
  }
  // wait for all time steps
  pool.WaitCount(count_before + configuration_length_ - 2);

  // reset pool count
  pool.ResetCount();
}

// update configuration trajectory
void Estimator::UpdateConfiguration(double* candidate,
                                    const double* configuration,
                                    const double* search_direction,
                                    double step_size) {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // copy configuration to candidate
  mju_copy(candidate, configuration, nq * configuration_length_);

  // loop over configurations
  for (int t = 0; t < configuration_length_; t++) {
    // configuration
    double* q = candidate + t * nq;

    // search direction
    const double* dq = search_direction + t * nv;

    // integrate
    mj_integratePos(model_, q, dq, step_size);
  }
}

// convert sequence of configurations to velocities and accelerations
void Estimator::ConfigurationToVelocityAcceleration() {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // velocities: loop over configuration trajectory
  for (int t = 0; t < configuration_length_ - 1; t++) {
    // previous and current configurations
    const double* q0 = configuration_.data() + t * nq;
    const double* q1 = configuration_.data() + (t + 1) * nq;

    // compute velocity
    double* v1 = velocity_.data() + t * nv;
    mj_differentiatePos(model_, v1, model_->opt.timestep, q0, q1);

    // compute acceleration
    if (t > 0) {
      // previous velocity
      const double* v0 = velocity_.data() + (t - 1) * nv;

      // compute acceleration
      double* a1 = acceleration_.data() + (t - 1) * nv;
      mju_sub(a1, v1, v0, nv);
      mju_scl(a1, a1, 1.0 / model_->opt.timestep, nv);
    }
  }
}

// compute finite-difference velocity, acceleration derivatives
void Estimator::VelocityAccelerationDerivatives() {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // ----- velocity derivatives ----- //
  
  // loop over configurations
  for (int t = 0; t < configuration_length_ - 1; t++) {
    // unpack
    double* q1 = configuration_.data() + t * nq;
    double* q2 = configuration_.data() + (t + 1) * nq;
    double* dv2dq1 = block_velocity_previous_configuration_.data() + t * nv * nv;
    double* dv2dq2 = block_velocity_current_configuration_.data() + t * nv * nv;

    // compute velocity Jacobians
    DifferentiateDifferentiatePos(dv2dq1, dv2dq2, model_, model_->opt.timestep,
                                  q1, q2);

    // compute acceleration Jacobians 
    if (t > 0) {
      // unpack
      double* dadq0 =
          block_acceleration_previous_configuration_.data() + (t - 1) * nv * nv;
      double* dadq1 =
          block_acceleration_current_configuration_.data() + (t - 1) * nv * nv;
      double* dadq2 = block_acceleration_next_configuration_.data() + (t - 1) * nv * nv;

      // previous velocity Jacobians
      double* dv1dq0 =
        block_velocity_previous_configuration_.data() + (t - 1) * nv * nv;
      double* dv1dq1 = block_velocity_current_configuration_.data() + (t - 1) * nv * nv;

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
}

// compute total cost
double Estimator::Cost(double& cost_prior, double& cost_sensor,
                       double& cost_force, ThreadPool& pool) {
  // ----- trajectories ----- //

  // TODO schedule threadpool
  // finite-difference velocities, accelerations
  ConfigurationToVelocityAcceleration();

  // compute sensor and force predictions
  InverseDynamicsPrediction(pool);

  // threadpool current count
  int count_before = pool.GetCount();

  // prior
  pool.Schedule([&cost_prior, &estimator = *this]() {
    estimator.ResidualPrior();
    cost_prior = estimator.CostPrior(NULL, NULL);
  });

  // sensor
  pool.Schedule([&cost_sensor, &estimator = *this]() {
    estimator.ResidualSensor();
    cost_sensor = estimator.CostSensor(NULL, NULL);
  });

  // force 
  pool.Schedule([&cost_force, &estimator = *this]() {
    estimator.ResidualForce();
    cost_force = estimator.CostForce(NULL, NULL);
  });

  // wait for individual cost computation
  pool.WaitCount(count_before + 3);

  // reset pool count
  pool.ResetCount();

  // total cost
  return cost_prior + cost_sensor + cost_force;
}

// optimize trajectory estimate
void Estimator::Optimize(ThreadPool& pool) {
  // resize data
  ResizeMjData(model_, pool.NumThreads());

  // timing
  double timer_model_derivatives = 0.0;
  double timer_velacc_derivatives = 0.0;
  double timer_jacobian_prior = 0.0;
  double timer_jacobian_sensor = 0.0;
  double timer_jacobian_force = 0.0;
  double timer_cost_prior_derivatives = 0.0;
  double timer_cost_sensor_derivatives = 0.0;
  double timer_cost_force_derivatives = 0.0;
  double timer_cost_gradient = 0.0;
  double timer_cost_hessian = 0.0;
  double timer_cost_derivatives = 0.0;
  double timer_search_direction = 0.0;
  double timer_line_search = 0.0;

  // compute cost
  double cost_prior = MAX_ESTIMATOR_COST;
  double cost_sensor = MAX_ESTIMATOR_COST;
  double cost_force = MAX_ESTIMATOR_COST;
  double cost = Cost(cost_prior, cost_sensor, cost_force, pool);

  // dimension
  int dim_con = model_->nq * configuration_length_;
  int dim_vel = model_->nv * configuration_length_;

  // smoother iterations
  int iterations_smoother = 0;
  int iterations_line_search = 0;
  for (; iterations_smoother < max_smoother_iterations_;
       iterations_smoother++) {
    // ----- cost derivatives ----- //

    // start timer (total cost derivatives)
    auto cost_derivatives_start = std::chrono::steady_clock::now();

    // -- compute inverse dynamics derivatives -- //

    // start timer
    auto model_derivatives_start = std::chrono::steady_clock::now();

    // compute derivatives
    InverseDynamicsDerivatives(pool);

    // stop timer
    timer_model_derivatives +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - model_derivatives_start)
            .count();

    // -- compute velocity, acceleration derivatives -- //

    // start timer
    auto velacc_derivatives_start = std::chrono::steady_clock::now();

    // compute derivatives
    VelocityAccelerationDerivatives();

    // stop timer
    timer_velacc_derivatives +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - velacc_derivatives_start)
            .count();

    // -- threadpool current count -- //
    int count_before = pool.GetCount();

    // -- prior derivatives -- //
    pool.Schedule([&estimator = *this, &timer_jacobian_prior,
                   &timer_cost_prior_derivatives]() {
      // start Jacobian timer
      auto jacobian_prior_start = std::chrono::steady_clock::now();

      // compute blocks
      estimator.JacobianPriorBlocks();

      // compute Jacobian
      estimator.JacobianPrior();

      // stop Jacobian timer
      timer_jacobian_prior +=
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - jacobian_prior_start)
              .count();

      // start derivative timer
      auto cost_prior_start = std::chrono::steady_clock::now();

      // compute derivatives
      estimator.CostPrior(estimator.cost_gradient_prior_.data(),
                          estimator.cost_hessian_prior_.data());

      // stop derivative timer
      timer_cost_prior_derivatives +=
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - cost_prior_start)
              .count();
    });

    // -- sensor derivatives -- //
    pool.Schedule([&estimator = *this, &timer_jacobian_sensor,
                   &timer_cost_sensor_derivatives]() {
      // start Jacobian timer
      auto jacobian_sensor_start = std::chrono::steady_clock::now();

      // compute Jacobian
      estimator.JacobianSensor();

      // stop Jacobian timer
      timer_jacobian_sensor +=
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - jacobian_sensor_start)
              .count();

      // start derivative timer
      auto cost_sensor_start = std::chrono::steady_clock::now();

      // compute derivatives
      estimator.CostSensor(estimator.cost_gradient_sensor_.data(),
                           estimator.cost_hessian_sensor_.data());

      // stop derivative timer
      timer_cost_sensor_derivatives +=
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - cost_sensor_start)
              .count();
    });

    // -- force derivatives -- //
    pool.Schedule([&estimator = *this, &timer_jacobian_force,
                   &timer_cost_force_derivatives]() {
      // start Jacobian timer
      auto jacobian_force_start = std::chrono::steady_clock::now();

      // compute Jacobian
      estimator.JacobianForce();

      // stop Jacobian timer
      timer_jacobian_force +=
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - jacobian_force_start)
              .count();

      // start derivative timer
      auto cost_force_start = std::chrono::steady_clock::now();

      // compute derivatives
      estimator.CostForce(estimator.cost_gradient_force_.data(), estimator.cost_hessian_force_.data());

      // stop derivative timer
      timer_cost_force_derivatives +=
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - cost_force_start)
              .count();
    });

    // wait for individual cost computation
    pool.WaitCount(count_before + 3);

    // reset pool count
    pool.ResetCount();

    // -- cumulative gradient -- // 

    // start gradient timer
    auto cost_gradient_start = std::chrono::steady_clock::now();

    // add gradients
    double* gradient = cost_gradient_.data();
    mju_copy(gradient, cost_gradient_prior_.data(), dim_vel);
    mju_addTo(gradient, cost_gradient_sensor_.data(), dim_vel);
    mju_addTo(gradient, cost_gradient_force_.data(), dim_vel);

    // stop gradient timer
    timer_cost_gradient +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - cost_gradient_start)
            .count();

    // -- cumulative Hessian -- //

    // start Hessian timer
    auto cost_hessian_start = std::chrono::steady_clock::now();

    // add Hessians
    double* hessian = cost_hessian_.data();
    mju_copy(hessian, cost_hessian_prior_.data(), dim_vel * dim_vel);
    mju_addTo(hessian, cost_hessian_sensor_.data(), dim_vel * dim_vel);
    mju_addTo(hessian, cost_hessian_force_.data(), dim_vel * dim_vel);

    // stop Hessian timer
    timer_cost_hessian +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - cost_hessian_start)
            .count();

    // stop timer
    timer_cost_derivatives +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - cost_derivatives_start)
            .count();

    // gradient tolerance check
    double gradient_norm = mju_norm(gradient, dim_vel) / dim_vel;
    if (gradient_norm < gradient_tolerance_) break;

    // ----- search direction ----- //

    // start timer
    auto search_direction_start = std::chrono::steady_clock::now();

    // unpack
    double* dq = search_direction_.data();

    // regularize TODO(taylor): LM reg.
    for (int j = 0; j < dim_vel; j++) {
      hessian[j * dim_vel + j] += 1.0e-3;
    }

    // linear system solver
    if (solver_ == kBanded) { // band solver
      // dimensions
      int ntotal = model_->nv * configuration_length_;
      int nband = 3 * model_->nv;
      int ndense = 0;

      // dense to banded
      double* hessian_band = cost_hessian_band_.data();
      mju_dense2Band(hessian_band, cost_hessian_.data(), ntotal, nband, ndense);

      // factorize
      mju_cholFactorBand(hessian_band, ntotal, nband, ndense, 0.0, 0.0);

      // compute search direction
      mju_cholSolveBand(dq, hessian_band, gradient, ntotal, nband, ndense);
    } else {  // dense solver
      // factorize
      mju_cholFactor(hessian, dim_vel, 0.0);

      // compute search direction
      mju_cholSolve(dq, hessian, gradient, dim_vel);
    }

    // end timer
    timer_search_direction +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - search_direction_start)
            .count();

    // ----- line search ----- //
    // TODO(taylor): option for curve search

    // start timer
    auto line_search_start = std::chrono::steady_clock::now();

    // copy configuration
    mju_copy(configuration_copy_.data(), configuration_.data(), dim_con);

    // initialize
    double cost_candidate = cost;
    int iteration_line_search = 0;
    double step_size = 2.0;

    // backtracking until cost decrease
    // TODO(taylor): Armijo, Wolfe conditions
    while (cost_candidate >= cost) {
      // check for max iterations
      if (iteration_line_search > max_line_search_) {
        // reset configuration
        mju_copy(configuration_.data(), configuration_copy_.data(), dim_con);

        // return;
        mju_error("Batch Estimator: Line search failure\n");
      }

      // decrease cost
      step_size *= 0.5;  // TODO(taylor): log schedule

      // candidate
      UpdateConfiguration(configuration_.data(), configuration_copy_.data(), dq,
                          -1.0 * step_size);

      // cost
      cost_candidate = Cost(cost_prior, cost_sensor, cost_force, pool);

      // update iteration
      iteration_line_search++;
    }

    // increment
    iterations_line_search += iteration_line_search;

    // end timer
    timer_line_search +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - line_search_start)
            .count();

    // update cost
    cost = cost_candidate;
  }

  // update cost
  cost_ = cost;
  cost_prior_ = cost_prior;
  cost_sensor_ = cost_sensor;
  cost_force_ = cost_force;

  // set timers
  timer_inverse_dynamics_derivatives_ = timer_model_derivatives;
  timer_velacc_derivatives_ = timer_velacc_derivatives;
  timer_jacobian_prior_ = timer_jacobian_prior;
  timer_jacobian_sensor_ = timer_jacobian_sensor;
  timer_jacobian_force_ = timer_jacobian_force;
  timer_cost_prior_derivatives_ = timer_cost_prior_derivatives;
  timer_cost_sensor_derivatives_ = timer_cost_sensor_derivatives;
  timer_cost_force_derivatives_ = timer_cost_force_derivatives;
  timer_cost_gradient_ = timer_cost_gradient;
  timer_cost_hessian_ = timer_cost_hessian;
  timer_cost_derivatives_ = timer_cost_derivatives;
  timer_search_direction_ = timer_search_direction;
  timer_line_search_ = timer_line_search;

  // status
  iterations_line_search_ = iterations_line_search;
  iterations_smoother_ = iterations_smoother;

  // status
  PrintStatus();
}

// print status
void Estimator::PrintStatus() {
  if (!verbose_) return;

  // title
  printf("Batch Estimator Status:\n\n");

  // timing
  printf("Timing:\n");
  printf("  inverse dynamics derivatives: %.5f (ms) \n",
         1.0e-3 * timer_inverse_dynamics_derivatives_);
  printf("  velacc derivatives: %.5f (ms) \n",
         1.0e-3 * timer_velacc_derivatives_);
  printf("  jacobian prior: %.5f (ms) \n", 1.0e-3 * timer_jacobian_prior_);
  printf("  jacobian sensor: %.5f (ms) \n", 1.0e-3 * timer_jacobian_sensor_);
  printf("  jacobian force: %.5f (ms) \n", 1.0e-3 * timer_jacobian_force_);
  printf("  cost prior derivatives: %.5f (ms) \n",
         1.0e-3 * timer_cost_prior_derivatives_);
  printf("  cost sensor derivatives: %.5f (ms) \n",
         1.0e-3 * timer_cost_sensor_derivatives_);
  printf("  cost force derivatives: %.5f (ms) \n",
         1.0e-3 * timer_cost_force_derivatives_);
  printf("  cost gradient: %.5f (ms) \n", 1.0e-3 * timer_cost_gradient_);
  printf("  cost hessian: %.5f (ms) \n", 1.0e-3 * timer_cost_hessian_);
  printf("  search direction: %.5f (ms) \n", 1.0e-3 * timer_search_direction_);
  printf("  line search: %.5f (ms) \n", 1.0e-3 * timer_line_search_);
  printf("  TOTAL: %.5f (ms) \n",
         1.0e-3 * (timer_cost_derivatives_ + timer_search_direction_ +
                   timer_line_search_));
  printf("\n");

  // status
  printf("Status:\n");
  printf("  iterations line search: %i\n", iterations_line_search_);
  printf("  iterations smoother: %i\n", iterations_smoother_);
}

// resize number of mjData
// TODO(taylor): one method for planner and estimator?
void Estimator::ResizeMjData(const mjModel* model, int num_threads) {
  int new_size = std::max(1, num_threads);
  if (data_.size() > new_size) {
    data_.erase(data_.begin() + new_size, data_.end());
  } else {
    data_.reserve(new_size);
    while (data_.size() < new_size) {
      data_.push_back(MakeUniqueMjData(mj_makeData(model)));
    }
  }
}

}  // namespace mjpc
