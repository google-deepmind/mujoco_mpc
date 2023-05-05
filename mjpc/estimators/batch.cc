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

#include "mjpc/norm.h"
#include "mjpc/utilities.h"

namespace mjpc {

// initialize estimator
void Estimator::Initialize(mjModel* model) {
  // model
  model_ = model;

  // data
  data_ = mj_makeData(model);

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
  cost_prior_ = 0;
  cost_sensor_ = 0;
  cost_force_ = 0;
  cost_ = 0;

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
}

// prior cost
double Estimator::CostPrior(double* gradient, double* hessian) {
  // residual dimension
  int dim = model_->nv * configuration_length_;

  // compute cost
  double cost =
      Norm(gradient ? norm_gradient_prior_.data() : NULL,
           hessian ? norm_hessian_prior_.data() : NULL, residual_prior_.data(),
           norm_parameters_prior_.data(), dim, norm_prior_);

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

    // compute total Hessian (Gauss-Newton approximation):
    // hessian = drdq' * d2ndr2 * drdq

    // step 1: scratch = d2ndr2 * drdq
    mju_mulMatMat(cost_scratch_prior_.data(), norm_hessian_prior_.data(),
                  jacobian_prior_.data(), dim, dim, dim);

    // step 2: hessian = drdq' * scratch
    mju_mulMatTMat(hessian, jacobian_prior_.data(), cost_scratch_prior_.data(),
                   dim, dim, dim);
  }

  // return weighted cost
  return weight_prior_ * cost;  // TODO(taylor): weight -> matrix
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
  int nv = model_->nv, dim = nv * configuration_length_;

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

// sensor cost
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

    // compute total Hessian (Gauss-Newton approximation):
    // hessian = drdq' * d2ndr2 * drdq

    // step 1: scratch = d2ndr2 * drdq
    mju_mulMatMat(cost_scratch_sensor_.data(), norm_hessian_sensor_.data(),
                  jacobian_sensor_.data(), dim_residual, dim_residual,
                  dim_update);

    // step 2: hessian = drdq' * scratch
    mju_mulMatTMat(hessian, jacobian_sensor_.data(),
                   cost_scratch_sensor_.data(), dim_residual, dim_update,
                   dim_update);
  }

  return weight_sensor_ * cost;  // TODO(taylor): weight -> matrix
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

// compute sensors
void Estimator::ComputeSensor() {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // loop over sensor
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // terms
    double* qt = configuration_.data() + (t + 1) * nq;
    double* vt = velocity_.data() + t * nv;
    double* at = acceleration_.data() + t * nv;

    // set qt, vt, at
    mju_copy(data_->qpos, qt, nq);
    mju_copy(data_->qvel, vt, nv);
    mju_copy(data_->qacc, at, nv);

    // sensors
    mj_inverse(model_, data_);

    // copy sensor data
    double* yt = sensor_prediction_.data() + t * dim_sensor_;
    mju_copy(yt, data_->sensordata, dim_sensor_);
  }
}

// force cost
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

  return weight_force_ * cost;  // TODO(taylor): weight -> matrix
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
void Estimator::ComputeForce() {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // loop over force
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // terms
    double* qt = configuration_.data() + (t + 1) * nq;
    double* vt = velocity_.data() + t * nv;
    double* at = acceleration_.data() + t * nv;

    // set qt, vt, at
    mju_copy(data_->qpos, qt, nq);
    mju_copy(data_->qvel, vt, nv);
    mju_copy(data_->qacc, at, nv);

    // force
    mj_inverse(model_, data_);

    // copy force
    double* ft = force_prediction_.data() + t * nv;
    mju_copy(ft, data_->qfrc_inverse, nv);
  }
}

// compute model derivatives (via finite difference)
void Estimator::ModelDerivatives() {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // loop over (state, acceleration)
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // unpack
    double* q = configuration_.data() + (t + 1) * nq;
    double* v = velocity_.data() + t * nv;
    double* a = acceleration_.data() + t * nv;
    double* dqds = block_sensor_configuration_.data() + t * dim_sensor_ * nv;
    double* dvds = block_sensor_velocity_.data() + t * dim_sensor_ * nv;
    double* dads = block_sensor_acceleration_.data() + t * dim_sensor_ * nv;
    double* dqdf = block_force_configuration_.data() + t * nv * nv;
    double* dvdf = block_force_velocity_.data() + t * nv * nv;
    double* dadf = block_force_acceleration_.data() + t * nv * nv;

    // set (state, acceleration)
    mju_copy(data_->qpos, q, nq);
    mju_copy(data_->qvel, v, nv);
    mju_copy(data_->qacc, a, nv);

    // finite-difference derivatives
    mjd_inverseFD(model_, data_, finite_difference_.tolerance,
                  finite_difference_.flg_actuation, dqdf, dvdf, dadf, dqds,
                  dvds, dads, NULL);
  }
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
  }

  // accelerations: loop over velocity trajectory
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // previous and current configurations
    const double* v0 = velocity_.data() + t * nv;
    const double* v1 = velocity_.data() + (t + 1) * nv;

    // compute acceleration
    double* a1 = acceleration_.data() + t * nv;
    mju_sub(a1, v1, v0, nv);
    mju_scl(a1, a1, 1.0 / model_->opt.timestep, nv);
  }
}

// compute finite-difference velocity derivatives
void Estimator::VelocityDerivatives() {
  // dimension
  int nq = model_->nq, nv = model_->nv;

  // loop over configurations
  for (int t = 0; t < configuration_length_ - 1; t++) {
    // unpack
    double* q0 = configuration_.data() + t * nq;
    double* q1 = configuration_.data() + (t + 1) * nq;
    double* dvdq0 = block_velocity_previous_configuration_.data() + t * nv * nv;
    double* dvdq1 = block_velocity_current_configuration_.data() + t * nv * nv;

    // compute Jacobians
    DifferentiateDifferentiatePos(dvdq0, dvdq1, model_, model_->opt.timestep,
                                  q0, q1);
  }
}

// compute finite-difference acceleration derivatives
void Estimator::AccelerationDerivatives() {
  // dimension
  int nv = model_->nv;

  // loop over configurations
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // unpack
    double* dadq0 =
        block_acceleration_previous_configuration_.data() + t * nv * nv;
    double* dadq1 =
        block_acceleration_current_configuration_.data() + t * nv * nv;
    double* dadq2 = block_acceleration_next_configuration_.data() + t * nv * nv;

    // note: velocity Jacobians need to be precomputed
    double* dv1dq0 =
        block_velocity_previous_configuration_.data() + t * nv * nv;
    double* dv1dq1 = block_velocity_current_configuration_.data() + t * nv * nv;

    double* dv2dq1 =
        block_velocity_previous_configuration_.data() + (t + 1) * nv * nv;
    double* dv2dq2 =
        block_velocity_current_configuration_.data() + (t + 1) * nv * nv;

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

// iterations
void Estimator::Iteration() {
  // ----- trajectories ----- //

  // finite-difference velocities, accelerations
  ConfigurationToVelocityAcceleration();

  // compute model sensors
  ComputeSensor();

  // compute model force
  ComputeForce();

  // ----- derivatives ----- //

  // compute model derivatives
  ModelDerivatives();

  // compute velocity derivatives
  VelocityDerivatives();

  // compute acceleration derivatives
  AccelerationDerivatives();

  // ----- residuals ----- //
  ResidualPrior();
  ResidualSensor();
  ResidualForce();

  // ----- residual Jacobians ----- //
  JacobianPrior();
  JacobianSensor();
  JacobianForce();

  // ----- costs ----- //
  double cost_prior =
      CostPrior(cost_gradient_prior_.data(), cost_hessian_prior_.data());
  double cost_sensor =
      CostSensor(cost_gradient_sensor_.data(), cost_hessian_sensor_.data());
  double cost_force =
      CostForce(cost_gradient_force_.data(), cost_hessian_force_.data());

  // ----- total cost ----- //

  // dimension
  int dim = model_->nv * configuration_length_;

  // cost
  double cost = cost_prior + cost_sensor + cost_force;

  // gradient
  double* gradient = cost_gradient_.data();
  mju_copy(gradient, cost_gradient_prior_.data(), dim);
  mju_addTo(gradient, cost_gradient_sensor_.data(), dim);
  mju_addTo(gradient, cost_gradient_force_.data(), dim);

  // Hessian
  double* hessian = cost_hessian_.data();
  mju_copy(hessian, cost_hessian_prior_.data(), dim * dim);
  mju_addTo(hessian, cost_hessian_sensor_.data(), dim * dim);
  mju_addTo(hessian, cost_hessian_force_.data(), dim * dim);

  // ----- search direction ----- //

  // unpack
  double* dq = search_direction_.data();

  if (solver == kBanded) {
    
  } else {    // dense solver

    // factorize
    mju_cholFactor(hessian, dim, 0.0);

    // compute search direction
    mju_cholSolve(dq, hessian, gradient, dim);
  }

  // ----- line search ----- //
  
  // copy configuration 
  mju_copy(configuration_copy_.data(), configuration_.data(), model_->nq * configuration_length_);
  
  // initialize  
  double cost_candidate = cost;
  int iteration = 0;
  double step_size = 2.0;

  // backtracking until cost decrease
  while (cost_candidate >= cost) {
    // check for max iterations
    if (iteration > max_line_search_) {
      mju_error("Batch Estimator: Line search failure\n");
    }

    // decrease cost 
    step_size *= 0.5;

    // candidate
    UpdateConfiguration(configuration_.data(), configuration_copy_.data(), dq,
                        step_size);

    // finite-difference velocities, accelerations
    ConfigurationToVelocityAcceleration();

    // predictions
    ComputeSensor();
    ComputeForce();

    // residuals
    ResidualPrior();
    ResidualSensor();
    ResidualForce();

    // cost
    cost_candidate =
        CostPrior(NULL, NULL) + CostSensor(NULL, NULL) + CostForce(NULL, NULL);

    // update iteration 
    iteration++;
  } 
}

}  // namespace mjpc
