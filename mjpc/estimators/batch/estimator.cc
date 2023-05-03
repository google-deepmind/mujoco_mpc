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

#include "mjpc/estimators/batch/estimator.h"

#include "mjpc/norm.h"
#include "mjpc/utilities.h"

namespace mjpc {

// convert sequence of configurations to velocities
void ConfigurationToVelocity(double* velocity, const double* configuration,
                             int configuration_length, const mjModel* model) {
  // loop over configuration trajectory
  for (int t = 0; t < configuration_length - 1; t++) {
    // previous and current configurations
    const double* q0 = configuration + t * model->nq;
    const double* q1 = configuration + (t + 1) * model->nq;

    // compute velocity
    double* v1 = velocity + t * model->nv;
    mj_differentiatePos(model, v1, model->opt.timestep, q0, q1);
  }
}

// convert sequence of configurations to velocities
void VelocityToAcceleration(double* acceleration, const double* velocity,
                            int velocity_length, const mjModel* model) {
  // loop over velocity trajectory
  for (int t = 0; t < velocity_length - 1; t++) {
    // previous and current configurations
    const double* v0 = velocity + t * model->nv;
    const double* v1 = velocity + (t + 1) * model->nv;

    // compute acceleration
    double* a1 = acceleration + t * model->nv;
    mju_sub(a1, v1, v0, model->nv);
    mju_scl(a1, a1, 1.0 / model->opt.timestep, model->nv);
  }
}

// initialize estimator
void Estimator::Initialize(mjModel* model) {
  // model
  model_ = model;

  // data
  data_ = mj_makeData(model);

  // trajectories
  configuration_length_ = GetNumberOrDefault(10, model, "batch_length");
  configuration_.resize(model->nq * MAX_HISTORY);
  configuration_prior_.resize(model->nq * MAX_HISTORY);
  configuration_copy_.resize(model->nq * MAX_HISTORY);
  velocity_.resize(model->nv * MAX_HISTORY);
  acceleration_.resize(model->nv * MAX_HISTORY);

  // measurement
  dim_measurement_ = model->nsensordata;
  measurement_sensor_.resize(dim_measurement_ * MAX_HISTORY);
  measurement_model_.resize(dim_measurement_ * MAX_HISTORY);

  // qfrc
  qfrc_actuator_.resize(model->nv * MAX_HISTORY);
  qfrc_inverse_.resize(model->nv * MAX_HISTORY);

  // residual
  residual_prior_.resize(model->nv * MAX_HISTORY);
  residual_measurement_.resize(dim_measurement_ * MAX_HISTORY);
  residual_inverse_dynamics_.resize(model->nv * MAX_HISTORY);

  // Jacobian
  jacobian_prior_.resize((model->nv * MAX_HISTORY) * (model->nv * MAX_HISTORY));
  jacobian_measurement_.resize((dim_measurement_ * MAX_HISTORY) *
                               (model->nv * MAX_HISTORY));
  jacobian_inverse_dynamics_.resize((model->nv * MAX_HISTORY) *
                                    (model->nv * MAX_HISTORY));

  // prior Jacobian block
  jacobian_block_prior_configuration_.resize((model->nv * model->nv) *
                                             MAX_HISTORY);

  // measurement Jacobian blocks
  jacobian_block_measurement_configuration_.resize(
      (dim_measurement_ * model->nv) * MAX_HISTORY);
  jacobian_block_measurement_velocity_.resize((dim_measurement_ * model->nv) *
                                              MAX_HISTORY);
  jacobian_block_measurement_acceleration_.resize(
      (dim_measurement_ * model->nv) * MAX_HISTORY);
  jacobian_block_measurement_scratch_.resize((dim_measurement_ * model->nv));

  // inverse dynamics Jacobian blocks
  jacobian_block_inverse_dynamics_configuration_.resize(
      (model->nv * model->nv) * MAX_HISTORY);
  jacobian_block_inverse_dynamics_velocity_.resize((model->nv * model->nv) *
                                                   MAX_HISTORY);
  jacobian_block_inverse_dynamics_acceleration_.resize((model->nv * model->nv) *
                                                       MAX_HISTORY);
  jacobian_block_inverse_dynamics_scratch_.resize((model->nv * model->nv));

  // velocity Jacobian blocks
  jacobian_block_velocity_previous_configuration_.resize(
      (model->nv * model->nv) * MAX_HISTORY);
  jacobian_block_velocity_current_configuration_.resize(
      (model->nv * model->nv) * MAX_HISTORY);

  // acceleration Jacobian blocks
  jacobian_block_acceleration_previous_configuration_.resize(
      (model->nv * model->nv) * MAX_HISTORY);
  jacobian_block_acceleration_current_configuration_.resize(
      (model->nv * model->nv) * MAX_HISTORY);
  jacobian_block_acceleration_next_configuration_.resize(
      (model->nv * model->nv) * MAX_HISTORY);

  // cost gradient
  cost_gradient_prior_.resize(model->nv * MAX_HISTORY);
  cost_gradient_measurement_.resize(model->nv * MAX_HISTORY);
  cost_gradient_inverse_dynamics_.resize(model->nv * MAX_HISTORY);
  cost_gradient_total_.resize(model->nv * MAX_HISTORY);

  // cost Hessian
  cost_hessian_prior_.resize((model->nv * MAX_HISTORY) *
                             (model->nv * MAX_HISTORY));
  cost_hessian_measurement_.resize((model->nv * MAX_HISTORY) *
                                   (model->nv * MAX_HISTORY));
  cost_hessian_inverse_dynamics_.resize((model->nv * MAX_HISTORY) *
                                        (model->nv * MAX_HISTORY));
  cost_hessian_total_.resize((model->nv * MAX_HISTORY) *
                             (model->nv * MAX_HISTORY));

  // weight TODO(taylor): matrices
  weight_prior_ = GetNumberOrDefault(1.0, model, "batch_weight_prior");
  weight_measurement_ =
      GetNumberOrDefault(1.0, model, "batch_weight_measurement");
  weight_inverse_dynamics_ =
      GetNumberOrDefault(1.0, model, "batch_weight_inverse_dynamics");

  // cost norms
  norm_prior_ = (NormType)GetNumberOrDefault(0, model, "batch_norm_prior");
  norm_measurement_ =
      (NormType)GetNumberOrDefault(0, model, "batch_norm_prior");
  norm_inverse_dynamics_ =
      (NormType)GetNumberOrDefault(1.0, model, "batch_norm_prior");

  // cost norm parameters
  norm_parameters_prior_.resize(3);
  norm_parameters_measurement_.resize(3);
  norm_parameters_inverse_dynamics_.resize(3);

  // norm gradient
  norm_gradient_prior_.resize(model->nv * MAX_HISTORY);
  norm_gradient_measurement_.resize(dim_measurement_ * MAX_HISTORY);
  norm_gradient_inverse_dynamics_.resize(model->nv * MAX_HISTORY);

  // norm Hessian
  norm_hessian_prior_.resize((model->nv * MAX_HISTORY) *
                             (model->nv * MAX_HISTORY));
  norm_hessian_measurement_.resize((dim_measurement_ * MAX_HISTORY) *
                                   (dim_measurement_ * MAX_HISTORY));
  norm_hessian_inverse_dynamics_.resize((model->nv * MAX_HISTORY) *
                                        (model->nv * MAX_HISTORY));

  // cost scratch
  cost_scratch_prior_.resize((model->nv * MAX_HISTORY) *
                             (model->nv * MAX_HISTORY));
  cost_scratch_measurement_.resize((dim_measurement_ * MAX_HISTORY) *
                                   (model->nv * MAX_HISTORY));
  cost_scratch_inverse_dynamics_.resize((model->nv * MAX_HISTORY) *
                                        (model->nv * MAX_HISTORY));

  // update
  update_.resize(model->nv * MAX_HISTORY);
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

  // compute cost gradient wrt update
  if (gradient) {
    // scale gradient by weight
    mju_scl(norm_gradient_prior_.data(), norm_gradient_prior_.data(),
            weight_prior_, dim);

    // compute total gradient wrt update: [d (residual) / d (update)]^T * d
    // (norm) / d (residual)
    mju_mulMatTVec(cost_gradient_prior_.data(), jacobian_prior_.data(),
                   norm_gradient_prior_.data(), dim, dim);
  }

  // compute cost Hessian wrt update
  if (hessian) {
    // scale Hessian by weight
    mju_scl(norm_hessian_prior_.data(), norm_hessian_prior_.data(),
            weight_prior_, dim * dim);

    // compute total Hessian (Gauss-Newton approximation):
    // [d (residual) / d (update)]^T * d^2 (norm) / d (residual)^2 * [d
    // (residual) / d (update)]

    // step 1: scratch = d^2 (norm) / d (residual)^2 * [d (residual) / d
    // (update)]
    mju_mulMatMat(cost_scratch_prior_.data(), norm_hessian_prior_.data(),
                  jacobian_prior_.data(), dim, dim, dim);

    // step 2: hessian = [d (residual) / d (update)]^T * scratch
    mju_mulMatTMat(cost_hessian_prior_.data(), jacobian_prior_.data(),
                   cost_scratch_prior_.data(), dim, dim, dim);
  }

  // return weighted cost
  return weight_prior_ * cost;  // TODO(taylor): weight -> matrix
}

// prior residual
void Estimator::ResidualPrior() {
  for (int t = 0; t < configuration_length_; t++) {
    // terms
    double* rt = residual_prior_.data() + t * model_->nv;
    double* qt_prior = configuration_prior_.data() + t * model_->nq;
    double* qt = configuration_.data() + t * model_->nq;

    // configuration difference
    mj_differentiatePos(model_, rt, 1.0, qt_prior, qt);
  }
}

// prior Jacobian
void Estimator::JacobianPrior() {
  // residual dimension
  int dim = model_->nv * configuration_length_;

  // reset Jacobian to zero
  mju_zero(jacobian_prior_.data(), dim * dim);

  // loop over configurations
  for (int t = 0; t < configuration_length_; t++) {
    // unpack
    double* block = jacobian_block_prior_configuration_.data() +
                    t * model_->nv * model_->nv;

    // set block in matrix
    SetMatrixInMatrix(jacobian_prior_.data(), block, 1.0, dim, dim, model_->nv,
                      model_->nv, t * model_->nv, t * model_->nv);
  }
}

// prior Jacobian blocks
void Estimator::JacobianPriorBlocks() {
  // loop over configurations
  for (int t = 0; t < configuration_length_; t++) {
    // unpack
    double* qt = configuration_.data() + t * model_->nq;
    double* qt_prior = configuration_prior_.data() + t * model_->nq;
    double* block = jacobian_block_prior_configuration_.data() +
                    t * model_->nv * model_->nv;

    // compute Jacobian
    DifferentiateDifferentiatePos(NULL, block, model_, 1.0, qt_prior, qt);
  }
}

// measurement cost
double Estimator::CostMeasurement(double* gradient, double* hessian) {
  // residual dimension
  int dim_residual = dim_measurement_ * (configuration_length_ - 2);

  // update dimension
  int dim_update = model_->nv * configuration_length_;

  // compute cost
  double cost =
      Norm(gradient ? norm_gradient_measurement_.data() : NULL,
           hessian ? norm_hessian_measurement_.data() : NULL,
           residual_measurement_.data(), norm_parameters_measurement_.data(),
           dim_residual, norm_measurement_);

  // compute cost gradient wrt update
  if (gradient) {
    // scale gradient by weight
    mju_scl(norm_gradient_measurement_.data(),
            norm_gradient_measurement_.data(), weight_measurement_,
            dim_residual);

    // compute total gradient wrt update: [d (residual) / d (update)]^T * d
    // (norm) / d (residual)
    mju_mulMatTVec(cost_gradient_measurement_.data(),
                   jacobian_measurement_.data(),
                   norm_gradient_measurement_.data(), dim_residual, dim_update);
  }

  // compute cost Hessian wrt update
  if (hessian) {
    // scale Hessian by weight
    mju_scl(norm_hessian_measurement_.data(), norm_hessian_measurement_.data(),
            weight_measurement_, dim_residual * dim_residual);

    // compute total Hessian (Gauss-Newton approximation):
    // [d (residual) / d (update)]^T * d^2 (norm) / d (residual)^2 * [d
    // (residual) / d (update)]

    // step 1: scratch = d^2 (norm) / d (residual)^2 * [d (residual) / d
    // (update)]
    mju_mulMatMat(
        cost_scratch_measurement_.data(), norm_hessian_measurement_.data(),
        jacobian_measurement_.data(), dim_residual, dim_residual, dim_update);

    // step 2: hessian = [d (residual) / d (update)]^T * scratch
    mju_mulMatTMat(
        cost_hessian_measurement_.data(), jacobian_measurement_.data(),
        cost_scratch_measurement_.data(), dim_residual, dim_update, dim_update);
  }

  return weight_measurement_ * cost;  // TODO(taylor): weight -> matrix
}

// measurement residual
void Estimator::ResidualMeasurement() {
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // terms
    double* rt = residual_measurement_.data() + t * dim_measurement_;
    double* yt_sensor = measurement_sensor_.data() + t * dim_measurement_;
    double* yt_model = measurement_model_.data() + t * dim_measurement_;

    // measurement difference
    mju_sub(rt, yt_model, yt_sensor, dim_measurement_);
  }
}

// measurement Jacobian
void Estimator::JacobianMeasurement() {
  // residual dimension
  int dim_residual = dim_measurement_ * (configuration_length_ - 2);

  // update dimension
  int dim_update = model_->nv * configuration_length_;

  // reset Jacobian to zero
  mju_zero(jacobian_measurement_.data(), dim_residual * dim_update);

  // loop over measurements
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // dqds
    double* dqds = jacobian_block_measurement_configuration_.data() +
                   t * dim_measurement_ * model_->nv;

    // dvds
    double* dvds = jacobian_block_measurement_velocity_.data() +
                   t * dim_measurement_ * model_->nv;

    // dads
    double* dads = jacobian_block_measurement_acceleration_.data() +
                   t * dim_measurement_ * model_->nv;

    // indices
    int row = t * dim_measurement_;
    int col_previous = t * model_->nv;
    int col_current = (t + 1) * model_->nv;
    int col_next = (t + 2) * model_->nv;

    // ----- configuration previous ----- //
    // dvds' * dvdq0
    double* dvdq0 = jacobian_block_velocity_previous_configuration_.data() +
                    t * model_->nv * model_->nv;
    mju_mulMatTMat(jacobian_block_measurement_scratch_.data(), dvds, dvdq0,
                   model_->nv, dim_measurement_, model_->nv);
    AddMatrixInMatrix(jacobian_measurement_.data(),
                      jacobian_block_measurement_scratch_.data(), 1.0,
                      dim_residual, dim_update, dim_measurement_, model_->nv,
                      row, col_previous);

    // dads' * dadq0
    double* dadq0 = jacobian_block_acceleration_previous_configuration_.data() +
                    t * model_->nv * model_->nv;
    mju_mulMatTMat(jacobian_block_measurement_scratch_.data(), dads, dadq0,
                   model_->nv, dim_measurement_, model_->nv);
    AddMatrixInMatrix(jacobian_measurement_.data(),
                      jacobian_block_measurement_scratch_.data(), 1.0,
                      dim_residual, dim_update, dim_measurement_, model_->nv,
                      row, col_previous);

    // ----- configuration current ----- //
    // dqds
    mju_transpose(jacobian_block_measurement_scratch_.data(), dqds, model_->nv,
                  dim_measurement_);
    AddMatrixInMatrix(jacobian_measurement_.data(),
                      jacobian_block_measurement_scratch_.data(), 1.0,
                      dim_residual, dim_update, dim_measurement_, model_->nv,
                      row, col_current);

    // dvds' * dvdq1
    double* dvdq1 = jacobian_block_velocity_current_configuration_.data() +
                    t * model_->nv * model_->nv;
    mju_mulMatTMat(jacobian_block_measurement_scratch_.data(), dvds, dvdq1,
                   model_->nv, dim_measurement_, model_->nv);
    AddMatrixInMatrix(jacobian_measurement_.data(),
                      jacobian_block_measurement_scratch_.data(), 1.0,
                      dim_residual, dim_update, dim_measurement_, model_->nv,
                      row, col_current);

    // dads' * dadq1
    double* dadq1 = jacobian_block_acceleration_current_configuration_.data() +
                    t * model_->nv * model_->nv;
    mju_mulMatTMat(jacobian_block_measurement_scratch_.data(), dads, dadq1,
                   model_->nv, dim_measurement_, model_->nv);
    AddMatrixInMatrix(jacobian_measurement_.data(),
                      jacobian_block_measurement_scratch_.data(), 1.0,
                      dim_residual, dim_update, dim_measurement_, model_->nv,
                      row, col_current);

    // ----- configuration next ----- //

    // dads' * dadq2
    double* dadq2 = jacobian_block_acceleration_next_configuration_.data() +
                    t * model_->nv * model_->nv;
    mju_mulMatTMat(jacobian_block_measurement_scratch_.data(), dads, dadq2,
                   model_->nv, dim_measurement_, model_->nv);
    AddMatrixInMatrix(jacobian_measurement_.data(),
                      jacobian_block_measurement_scratch_.data(), 1.0,
                      dim_residual, dim_update, dim_measurement_, model_->nv,
                      row, col_next);
  }
}

// compute measurements
void Estimator::ComputeMeasurements() {
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // terms
    double* qt = configuration_.data() + (t + 1) * model_->nq;
    double* vt = velocity_.data() + t * model_->nv;
    double* at = acceleration_.data() + t * model_->nv;

    // set qt, vt, at
    mju_copy(data_->qpos, qt, model_->nq);
    mju_copy(data_->qvel, vt, model_->nv);
    mju_copy(data_->qacc, at, model_->nv);

    // sensors
    mj_inverse(model_, data_);

    // copy sensor data
    double* yt = measurement_model_.data() + t * dim_measurement_;
    mju_copy(yt, data_->sensordata, dim_measurement_);
  }
}

// inverse dynamics cost
double Estimator::CostInverseDynamics(double* gradient, double* hessian) {
  // residual dimension
  int dim_residual = model_->nv * (configuration_length_ - 2);

  // update dimension
  int dim_update = model_->nv * configuration_length_;

  // compute cost
  double cost = Norm(gradient ? norm_gradient_inverse_dynamics_.data() : NULL,
                     hessian ? norm_hessian_inverse_dynamics_.data() : NULL,
                     residual_inverse_dynamics_.data(),
                     norm_parameters_inverse_dynamics_.data(), dim_residual,
                     norm_inverse_dynamics_);

  // compute cost gradient wrt update
  if (gradient) {
    // scale gradient by weight
    mju_scl(norm_gradient_inverse_dynamics_.data(),
            norm_gradient_inverse_dynamics_.data(), weight_inverse_dynamics_,
            dim_residual);

    // compute total gradient wrt update: [d (residual) / d (update)]^T * d
    // (norm) / d (residual)
    mju_mulMatTVec(cost_gradient_inverse_dynamics_.data(),
                   jacobian_inverse_dynamics_.data(),
                   norm_gradient_inverse_dynamics_.data(), dim_residual,
                   dim_update);
  }

  // compute cost Hessian wrt update
  if (hessian) {
    // scale Hessian by weight
    mju_scl(norm_hessian_inverse_dynamics_.data(),
            norm_hessian_inverse_dynamics_.data(), weight_inverse_dynamics_,
            dim_residual * dim_residual);

    // compute total Hessian (Gauss-Newton approximation):
    // [d (residual) / d (update)]^T * d^2 (norm) / d (residual)^2 * [d
    // (residual) / d (update)]

    // step 1: scratch = d^2 (norm) / d (residual)^2 * [d (residual) / d
    // (update)]
    mju_mulMatMat(cost_scratch_inverse_dynamics_.data(),
                  norm_hessian_inverse_dynamics_.data(),
                  jacobian_inverse_dynamics_.data(), dim_residual, dim_residual,
                  dim_update);

    // step 2: hessian = [d (residual) / d (update)]^T * scratch
    mju_mulMatTMat(cost_hessian_inverse_dynamics_.data(),
                   jacobian_inverse_dynamics_.data(),
                   cost_scratch_inverse_dynamics_.data(), dim_residual,
                   dim_update, dim_update);
  }

  return weight_inverse_dynamics_ * cost;  // TODO(taylor): weight -> matrix
}

// inverse dynamics residual
void Estimator::ResidualInverseDynamics() {
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // terms
    double* rt = residual_inverse_dynamics_.data() + t * model_->nv;
    double* ft_actuator = qfrc_actuator_.data() + t * model_->nv;
    double* ft_inverse_ = qfrc_inverse_.data() + t * model_->nv;

    // qfrc difference
    mju_sub(rt, ft_inverse_, ft_actuator, model_->nv);
  }
}

// inverse dynamics Jacobian
void Estimator::JacobianInverseDynamics() {
  // residual dimension
  int dim_residual = model_->nv * (configuration_length_ - 2);

  // update dimension
  int dim_update = model_->nv * configuration_length_;

  // reset Jacobian to zero
  mju_zero(jacobian_inverse_dynamics_.data(), dim_residual * dim_update);

  // loop over qfrc
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // dqdf
    double* dqdf = jacobian_block_inverse_dynamics_configuration_.data() +
                   t * model_->nv * model_->nv;

    // dvdf
    double* dvdf = jacobian_block_inverse_dynamics_velocity_.data() +
                   t * model_->nv * model_->nv;

    // dadf
    double* dadf = jacobian_block_inverse_dynamics_acceleration_.data() +
                   t * model_->nv * model_->nv;

    // indices
    int row = t * model_->nv;
    int col_previous = t * model_->nv;
    int col_current = (t + 1) * model_->nv;
    int col_next = (t + 2) * model_->nv;

    // ----- configuration previous ----- //
    // dvdf' * dvdq0
    double* dvdq0 = jacobian_block_velocity_previous_configuration_.data() +
                    t * model_->nv * model_->nv;
    mju_mulMatTMat(jacobian_block_inverse_dynamics_scratch_.data(), dvdf, dvdq0,
                   model_->nv, model_->nv, model_->nv);
    AddMatrixInMatrix(jacobian_inverse_dynamics_.data(),
                      jacobian_block_inverse_dynamics_scratch_.data(), 1.0,
                      dim_residual, dim_update, model_->nv, model_->nv,
                      row, col_previous);

    // dadf' * dadq0
    double* dadq0 = jacobian_block_acceleration_previous_configuration_.data() +
                    t * model_->nv * model_->nv;
    mju_mulMatTMat(jacobian_block_inverse_dynamics_scratch_.data(), dadf, dadq0,
                   model_->nv, model_->nv, model_->nv);
    AddMatrixInMatrix(jacobian_inverse_dynamics_.data(),
                      jacobian_block_inverse_dynamics_scratch_.data(), 1.0,
                      dim_residual, dim_update, model_->nv, model_->nv,
                      row, col_previous);

    // ----- configuration current ----- //
    // dqdf'
    mju_transpose(jacobian_block_inverse_dynamics_scratch_.data(), dqdf, model_->nv,
                  model_->nv);
    AddMatrixInMatrix(jacobian_inverse_dynamics_.data(),
                      jacobian_block_inverse_dynamics_scratch_.data(), 1.0,
                      dim_residual, dim_update, model_->nv, model_->nv,
                      row, col_current);

    // dvdf' * dvdq1
    double* dvdq1 = jacobian_block_velocity_current_configuration_.data() +
                    t * model_->nv * model_->nv;
    mju_mulMatTMat(jacobian_block_inverse_dynamics_scratch_.data(), dvdf, dvdq1,
                   model_->nv, model_->nv, model_->nv);
    AddMatrixInMatrix(jacobian_inverse_dynamics_.data(),
                      jacobian_block_inverse_dynamics_scratch_.data(), 1.0,
                      dim_residual, dim_update, model_->nv, model_->nv,
                      row, col_current);

    // dadf' * dadq1
    double* dadq1 = jacobian_block_acceleration_current_configuration_.data() +
                    t * model_->nv * model_->nv;
    mju_mulMatTMat(jacobian_block_inverse_dynamics_scratch_.data(), dadf, dadq1,
                   model_->nv, model_->nv, model_->nv);
    AddMatrixInMatrix(jacobian_inverse_dynamics_.data(),
                      jacobian_block_inverse_dynamics_scratch_.data(), 1.0,
                      dim_residual, dim_update, model_->nv, model_->nv,
                      row, col_current);

    // ----- configuration next ----- //

    // dadf' * dadq2
    double* dadq2 = jacobian_block_acceleration_next_configuration_.data() +
                    t * model_->nv * model_->nv;
    mju_mulMatTMat(jacobian_block_inverse_dynamics_scratch_.data(), dadf, dadq2,
                   model_->nv, model_->nv, model_->nv);
    AddMatrixInMatrix(jacobian_inverse_dynamics_.data(),
                      jacobian_block_inverse_dynamics_scratch_.data(), 1.0,
                      dim_residual, dim_update, model_->nv, model_->nv,
                      row, col_next);
  }
}

// compute inverse dynamics
void Estimator::ComputeInverseDynamics() {
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // terms
    double* qt = configuration_.data() + (t + 1) * model_->nq;
    double* vt = velocity_.data() + t * model_->nv;
    double* at = acceleration_.data() + t * model_->nv;

    // set qt, vt, at
    mju_copy(data_->qpos, qt, model_->nq);
    mju_copy(data_->qvel, vt, model_->nv);
    mju_copy(data_->qacc, at, model_->nv);

    // inverse dynamics
    mj_inverse(model_, data_);

    // copy qfrc
    double* ft = qfrc_inverse_.data() + t * model_->nv;
    mju_copy(ft, data_->qfrc_inverse, model_->nv);
  }
}

// compute model derivatives (via finite difference)
void Estimator::ModelDerivatives() {
  // loop over measurements
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // unpack
    double* q = configuration_.data() + (t + 1) * model_->nq;
    double* v = velocity_.data() + t * model_->nv;
    double* a = acceleration_.data() + t * model_->nv;
    double* dqds = jacobian_block_measurement_configuration_.data() +
                   t * dim_measurement_ * model_->nv;
    double* dvds = jacobian_block_measurement_velocity_.data() +
                   t * dim_measurement_ * model_->nv;
    double* dads = jacobian_block_measurement_acceleration_.data() +
                   t * dim_measurement_ * model_->nv;
    double* dqdf = jacobian_block_inverse_dynamics_configuration_.data() +
                   t * model_->nv * model_->nv;
    double* dvdf = jacobian_block_inverse_dynamics_velocity_.data() +
                   t * model_->nv * model_->nv;
    double* dadf = jacobian_block_inverse_dynamics_acceleration_.data() +
                   t * model_->nv * model_->nv;

    // set (state, acceleration)
    mju_copy(data_->qpos, q, model_->nq);
    mju_copy(data_->qvel, v, model_->nv);
    mju_copy(data_->qacc, a, model_->nv);

    // finite-difference derivatives
    mjd_inverseFD(model_, data_, finite_difference_tolerance_,
                  finite_difference_flg_actuation_, dqdf, dvdf, dadf, dqds,
                  dvds, dads, NULL);
  }
}

// update configuration trajectory
void Estimator::UpdateConfiguration(double* configuration,
                                    const double* update) {
  for (int t = 0; t < configuration_length_; t++) {
    // configuration
    double* q = configuration + t * model_->nq;

    // update
    const double* dq = update + t * model_->nv;

    // integrate
    mj_integratePos(model_, q, dq, 1.0);
  }
}

// update configuration, velocity, acceleration, measurement, and qfrc
// trajectories
void Estimator::UpdateTrajectory(double* configuration, const double* update) {
  // update configuration trajectory using
  UpdateConfiguration(configuration, update);

  // finite-difference velocities
  ConfigurationToVelocity(velocity_.data(), configuration,
                          configuration_length_, model_);

  // finite-difference accelerations
  VelocityToAcceleration(acceleration_.data(), velocity_.data(),
                         configuration_length_ - 1, model_);

  // compute model measurements
  ComputeMeasurements();

  // compute model qfrc
  ComputeInverseDynamics();

  // compute model derivatives
  ModelDerivatives();
}

// velocity Jacobian blocks
void Estimator::VelocityJacobianBlocks() {
  // loop over configurations
  for (int t = 0; t < configuration_length_ - 1; t++) {
    // unpack
    double* q0 = configuration_.data() + t * model_->nq;
    double* q1 = configuration_.data() + (t + 1) * model_->nq;
    double* dvdq0 = jacobian_block_velocity_previous_configuration_.data() +
                    t * model_->nv * model_->nv;
    double* dvdq1 = jacobian_block_velocity_current_configuration_.data() +
                    t * model_->nv * model_->nv;

    // compute Jacobians
    DifferentiateDifferentiatePos(dvdq0, dvdq1, model_, model_->opt.timestep, q0, q1);
  }
}

// acceleration Jacobian blocks
void Estimator::AccelerationJacobianBlocks() {
  // loop over configurations
  for (int t = 0; t < configuration_length_ - 2; t++) {
    // unpack
    double* dadq0 = jacobian_block_acceleration_previous_configuration_.data() +
                    t * model_->nv * model_->nv;
    double* dadq1 = jacobian_block_acceleration_current_configuration_.data() +
                    t * model_->nv * model_->nv;
    double* dadq2 = jacobian_block_acceleration_next_configuration_.data() +
                    t * model_->nv * model_->nv;

    // note: velocity Jacobians need to be precomputed
    double* dv1dq0 = jacobian_block_velocity_previous_configuration_.data() +
                     t * model_->nv * model_->nv;
    double* dv1dq1 = jacobian_block_velocity_current_configuration_.data() +
                     t * model_->nv * model_->nv;

    double* dv2dq1 = jacobian_block_velocity_previous_configuration_.data() +
                     (t + 1) * model_->nv * model_->nv;
    double* dv2dq2 = jacobian_block_velocity_current_configuration_.data() +
                     (t + 1) * model_->nv * model_->nv;

    // dadq0 = -dv1dq0 / h
    mju_copy(dadq0, dv1dq0, model_->nv * model_->nv);
    mju_scl(dadq0, dadq0, -1.0 / model_->opt.timestep, model_->nv * model_->nv);

    // dadq1 = dv2dq1 / h - dv1dq1 / h = (dv2dq1 - dv1dq1) / h
    mju_sub(dadq1, dv2dq1, dv1dq1, model_->nv * model_->nv);
    mju_scl(dadq1, dadq1, 1.0 / model_->opt.timestep, model_->nv * model_->nv);

    // dadq2 = dv2dq2 / h
    mju_copy(dadq2, dv2dq2, model_->nv * model_->nv);
    mju_scl(dadq2, dadq2, 1.0 / model_->opt.timestep, model_->nv * model_->nv);
  }
}

}  // namespace mjpc
