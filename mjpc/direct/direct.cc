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

#include "mjpc/direct/direct.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>

#include "mjpc/direct/trajectory.h"
#include "mjpc/direct/model_parameters.h"
#include "mjpc/norm.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

// constructor
Direct::Direct(const mjModel* model, int length, int max_history)
    : model_parameters_(LoadModelParameters()),
      pool_(NumAvailableHardwareThreads()) {
  // set max history length
  this->max_history_ = (max_history == 0 ? length : max_history);

  // initialize memory
  Initialize(model);

  // set trajectory lengths
  SetConfigurationLength(length);

  // reset memory
  Reset();
}

// initialize direct optimizer
void Direct::Initialize(const mjModel* model) {
  // model
  if (this->model) mj_deleteModel(this->model);
  this->model = mj_copyModel(nullptr, model);

  // set discrete inverse dynamics
  this->model->opt.enableflags |= mjENBL_INVDISCRETE;

  // data
  data_.clear();
  for (int i = 0; i < max_history_; i++) {
    data_.push_back(MakeUniqueMjData(mj_makeData(model)));
  }

  // timestep
  this->model->opt.timestep =
      GetNumberOrDefault(this->model->opt.timestep, model, "direct_timestep");

  // length of configuration trajectory
  configuration_length_ =
      GetNumberOrDefault(3, model, "direct_configuration_length");

  // check configuration length
  if (configuration_length_ > max_history_) {
    mju_error("configuration_length > max_history: increase max history\n");
  }

  // number of parameters
  nparam_ = GetNumberOrDefault(0, model, "direct_num_parameters");

  // model parameters id
  model_parameters_id_ =
      GetNumberOrDefault(-1, model, "direct_model_parameters_id");
  if (model_parameters_id_ == -1 && nparam_ > 0) {
    mju_error("nparam > 0 but model_parameter_id is missing\n");
  }

  // perturbation models
  if (nparam_ > 0) {
    // clear memory
    model_perturb_.clear();

    // add model for each time step (need for threaded evaluation)
    for (int i = 0; i < max_history_; i++) {
      // add model
      model_perturb_.push_back(MakeUniqueMjModel(mj_copyModel(nullptr, model)));

      // set discrete inverse dynamics
      model_perturb_[i].get()->opt.enableflags |= mjENBL_INVDISCRETE;
    }
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

  // allocation dimension
  int nq = model->nq, nv = model->nv, na = model->na;
  int nvel_max = nv * max_history_;
  int nsensor_max = nsensordata_ * max_history_;
  int ntotal_max = nvel_max + nparam_;

  // state dimensions
  nstate_ = nq + nv + na;
  ndstate_ = 2 * nv + na;

  // problem dimensions
  nvel_ = nv * configuration_length_;
  ntotal_ = nvel_ + nparam_;
  nband_ = 3 * nv;

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

  // prior
  configuration_previous.Initialize(nq, configuration_length_);

  // sensor
  sensor_measurement.Initialize(nsensordata_, configuration_length_);
  sensor_prediction.Initialize(nsensordata_, configuration_length_);
  sensor_mask.Initialize(nsensor_, configuration_length_);

  // force
  force_measurement.Initialize(nv, configuration_length_);
  force_prediction.Initialize(nv, configuration_length_);

  // parameters
  parameters.resize(nparam_);
  parameters_previous.resize(nparam_);
  noise_parameter.resize(nparam_);

  // residual
  residual_sensor_.resize(nsensor_max);
  residual_force_.resize(nvel_max);

  // Jacobian
  jacobian_sensor_.resize(settings.assemble_sensor_jacobian * nsensor_max *
                          ntotal_max);
  jacobian_force_.resize(settings.assemble_force_jacobian * nvel_max *
                         ntotal_max);

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
  block_sensor_configurations_.Initialize(nsensordata_ * nband_,
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
  block_force_configurations_.Initialize(nv * nband_, configuration_length_);

  block_force_scratch_.Initialize(nv * nv, configuration_length_);

  // sensor Jacobian blocks wrt parameters
  block_sensor_parameters_.Initialize(model->nsensordata * nparam_,
                                      configuration_length_);
  block_sensor_parametersT_.Initialize(nparam_ * model->nsensordata,
                                       configuration_length_);
  // force Jacobian blocks
  block_force_parameters_.Initialize(nparam_ * nv, configuration_length_);

  // velocity Jacobian blocks wrt parameters
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
  cost_gradient_sensor_.resize(ntotal_max);
  cost_gradient_force_.resize(ntotal_max);
  cost_gradient_.resize(ntotal_max);

  // cost Hessian
  cost_hessian_sensor_band_.resize(nvel_max * nband_ + nparam_ * ntotal_max);
  cost_hessian_force_band_.resize(nvel_max * nband_ + nparam_ * ntotal_max);
  cost_hessian_.resize(settings.assemble_cost_hessian * ntotal_max *
                       ntotal_max);
  cost_hessian_band_.resize(nvel_max * nband_ + nparam_ * ntotal_max);
  cost_hessian_band_factor_.resize(nvel_max * nband_ + nparam_ * ntotal_max);

  // cost norms
  norm_type_sensor.resize(nsensor_);

  // TODO(taylor): method for xml to initial norm
  for (int i = 0; i < nsensor_; i++) {
    norm_type_sensor[i] =
        (NormType)GetNumberOrDefault(0, model, "direct_norm_sensor");

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
  norm_force_.resize(max_history_ - 1);

  // norm gradient
  norm_gradient_sensor_.resize(nsensor_max);
  norm_gradient_force_.resize(ntotal_max);

  // norm Hessian
  norm_hessian_sensor_.resize(settings.assemble_sensor_norm_hessian *
                              nsensor_max * nsensor_max);
  norm_hessian_force_.resize(settings.assemble_force_norm_hessian * ntotal_max *
                             ntotal_max);

  norm_blocks_sensor_.resize(nsensordata_ * nsensor_max);
  norm_blocks_force_.resize(nv * ntotal_max);

  // scratch
  scratch_sensor_.resize(nband_ + nsensordata_ * nband_ + 9 * nv * nv +
                         nparam_ * nband_ + nparam_ * nparam_ +
                         nsensordata_ * nparam_);
  scratch_force_.resize(12 * nv * nv + nparam_ * nband_ + nparam_ * nparam_ +
                        nv * nparam_);
  scratch_expected_.resize(ntotal_max);

  // copy
  configuration_copy_.Initialize(nq, configuration_length_);

  // search direction
  search_direction_.resize(ntotal_max);

  // parameters copy
  parameters_copy_.resize(nparam_ * max_history_);

  // dense cost Hessian rows (for parameter derivatives)
  dense_force_parameter_.resize(nparam_ * ntotal_max);
  dense_sensor_parameter_.resize(nparam_ * ntotal_max);
  dense_parameter_.resize(nparam_ * ntotal_max);

  // regularization
  regularization_ = settings.regularization_initial;

  // search type
  settings.search_type = (SearchType)GetNumberOrDefault(
      static_cast<int>(settings.search_type), model, "direct_search_type");

  // timer
  timer_.sensor_step.resize(max_history_);
  timer_.force_step.resize(max_history_);

  // status
  gradient_norm_ = 0.0;
  search_direction_norm_ = 0.0;
  step_size_ = 1.0;
  solve_status_ = kUnsolved;
}

// reset memory
void Direct::Reset(const mjData* data) {
  // process noise
  double noise_process_scl =
      GetNumberOrDefault(1.0e-4, model, "estimator_process_noise_scale");
  std::fill(noise_process.begin(), noise_process.end(), noise_process_scl);

  // sensor noise
  double noise_sensor_scl =
      GetNumberOrDefault(1.0e-4, model, "estimator_sensor_noise_scale");
  std::fill(noise_sensor.begin(), noise_sensor.end(), noise_sensor_scl);

  // trajectories
  configuration.Reset();
  velocity.Reset();
  acceleration.Reset();
  act.Reset();
  times.Reset();

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

  // parameters
  std::fill(parameters.begin(), parameters.end(), 0.0);
  std::fill(parameters_previous.begin(), parameters_previous.end(), 0.0);

  // parameter weights
  double noise_parameter_scl =
      GetNumberOrDefault(1.0, model, "direct_noise_parameter");
  std::fill(noise_parameter.begin(), noise_parameter.end(),
            noise_parameter_scl);

  // residual
  std::fill(residual_sensor_.begin(), residual_sensor_.end(), 0.0);
  std::fill(residual_force_.begin(), residual_force_.end(), 0.0);

  // Jacobian
  std::fill(jacobian_sensor_.begin(), jacobian_sensor_.end(), 0.0);
  std::fill(jacobian_force_.begin(), jacobian_force_.end(), 0.0);

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

  // sensor Jacobian blocks wrt parameters
  block_sensor_parameters_.Reset();
  block_sensor_parametersT_.Reset();

  // force Jacobian blocks wrt parameters
  block_force_parameters_.Reset();

  // velocity Jacobian blocks
  block_velocity_previous_configuration_.Reset();
  block_velocity_current_configuration_.Reset();

  // acceleration Jacobian blocks
  block_acceleration_previous_configuration_.Reset();
  block_acceleration_current_configuration_.Reset();
  block_acceleration_next_configuration_.Reset();

  // cost
  cost_sensor_ = 0.0;
  cost_force_ = 0.0;
  cost_parameter_ = 0.0;
  cost_ = 0.0;
  cost_initial_ = 0.0;
  cost_previous_ = 1.0e32;

  // cost gradient
  std::fill(cost_gradient_sensor_.begin(), cost_gradient_sensor_.end(), 0.0);
  std::fill(cost_gradient_force_.begin(), cost_gradient_force_.end(), 0.0);
  std::fill(cost_gradient_.begin(), cost_gradient_.end(), 0.0);

  // cost Hessian
  std::fill(cost_hessian_sensor_band_.begin(), cost_hessian_sensor_band_.end(),
            0.0);
  std::fill(cost_hessian_force_band_.begin(), cost_hessian_force_band_.end(),
            0.0);
  std::fill(cost_hessian_.begin(), cost_hessian_.end(), 0.0);
  std::fill(cost_hessian_band_.begin(), cost_hessian_band_.end(), 0.0);
  std::fill(cost_hessian_band_factor_.begin(), cost_hessian_band_factor_.end(),
            0.0);

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
  std::fill(scratch_sensor_.begin(), scratch_sensor_.end(), 0.0);
  std::fill(scratch_force_.begin(), scratch_force_.end(), 0.0);
  std::fill(scratch_expected_.begin(), scratch_expected_.end(), 0.0);

  // candidate
  configuration_copy_.Reset();

  // search direction
  std::fill(search_direction_.begin(), search_direction_.end(), 0.0);

  // parameters copy
  std::fill(parameters_copy_.begin(), parameters_copy_.end(), 0.0);

  // dense cost Hessian rows (for parameter derivatives)
  std::fill(dense_force_parameter_.begin(), dense_force_parameter_.end(), 0.0);
  std::fill(dense_sensor_parameter_.begin(), dense_sensor_parameter_.end(),
            0.0);
  std::fill(dense_parameter_.begin(), dense_parameter_.end(), 0.0);

  // timer
  std::fill(timer_.sensor_step.begin(), timer_.sensor_step.end(), 0.0);
  std::fill(timer_.force_step.begin(), timer_.force_step.end(), 0.0);

  // timing
  ResetTimers();

  // status
  iterations_smoother_ = 0;
  iterations_search_ = 0;
  cost_count_ = 0;
  solve_status_ = kUnsolved;
}

// compute and return dense cost Hessian
double* Direct::GetCostHessian() {
  // resize
  cost_hessian_.resize(ntotal_ * ntotal_);

  // band to dense
  mju_band2Dense(cost_hessian_.data(), cost_hessian_band_.data(), ntotal_,
                 nband_, nparam_, 1);

  // return dense Hessian
  return cost_hessian_.data();
}

// compute and return dense sensor Jacobian
const double* Direct::GetJacobianSensor() {
  // dimension
  int nsensor_max = nsensordata_ * (configuration_length_ - 1);

  // resize
  jacobian_sensor_.resize(nsensor_max * ntotal_);

  // change setting
  int settings_cache = settings.assemble_sensor_jacobian;
  settings.assemble_sensor_jacobian = true;

  // loop over sensors
  for (int t = 0; t < configuration_length_ - 1; t++) {
    BlockSensor(t);
  }

  // TODO
  if (nparam_ > 0) {
    mju_error("parameter Jacobians not implemented\n");
  }

  // restore setting
  settings.assemble_sensor_jacobian = settings_cache;

  // return dense Jacobian
  return jacobian_sensor_.data();
}

// compute and return dense force Jacobian
const double* Direct::GetJacobianForce() {
  // dimensions
  int nv = model->nv;
  int nforcetotal = nv * (configuration_length_ - 2);

  // resize
  jacobian_force_.resize(nforcetotal * ntotal_);

  // change setting
  int settings_cache = settings.assemble_force_jacobian;
  settings.assemble_force_jacobian = true;

  // loop over sensors
  for (int t = 1; t < configuration_length_ - 1; t++) {
    BlockForce(t);
  }

  // TODO
  if (nparam_ > 0) {
    mju_error("parameter Jacobians not implemented\n");
  }

  // restore setting
  settings.assemble_force_jacobian = settings_cache;

  // return dense Jacobian
  return jacobian_force_.data();
}

// compute and return dense sensor norm Hessian
const double* Direct::GetNormHessianSensor() {
  // dimensions
  int nsensor_max = nsensordata_ * configuration_length_;

  // resize
  norm_hessian_sensor_.resize(nsensor_max * nsensor_max);

  // change setting
  int settings_cache = settings.assemble_sensor_norm_hessian;
  settings.assemble_sensor_norm_hessian = true;

  // evalute
  CostSensor(NULL, NULL);

  // restore setting
  settings.assemble_sensor_norm_hessian = settings_cache;

  // return dense Hessian
  return norm_hessian_sensor_.data();
}

// compute and return dense force norm Hessian
const double* Direct::GetNormHessianForce() {
  // dimensions
  int nforcetotal = model->nv * (configuration_length_ - 2);

  // resize
  norm_hessian_force_.resize(nforcetotal * nforcetotal);

  // change setting
  int settings_cache = settings.assemble_force_norm_hessian;
  settings.assemble_force_norm_hessian = true;

  // evalute
  CostForce(NULL, NULL);

  // restore setting
  settings.assemble_force_norm_hessian = settings_cache;

  // return dense Hessian
  return norm_hessian_force_.data();
}

// set configuration length
void Direct::SetConfigurationLength(int length) {
  // check length
  if (length > max_history_) {
    mju_error("length > max_history_\n");
  }

  // set configuration length
  configuration_length_ = std::max(length, kMinDirectHistory);
  nvel_ = model->nv * configuration_length_;
  ntotal_ = nvel_ + nparam_;

  // update trajectory lengths
  configuration.SetLength(configuration_length_);
  configuration_copy_.SetLength(configuration_length_);

  velocity.SetLength(configuration_length_);
  acceleration.SetLength(configuration_length_);
  act.SetLength(configuration_length_);
  times.SetLength(configuration_length_);

  configuration_previous.SetLength(configuration_length_);

  sensor_measurement.SetLength(configuration_length_);
  sensor_prediction.SetLength(configuration_length_);
  sensor_mask.SetLength(configuration_length_);

  force_measurement.SetLength(configuration_length_);
  force_prediction.SetLength(configuration_length_);

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

  block_sensor_parameters_.SetLength(configuration_length_);
  block_sensor_parametersT_.SetLength(configuration_length_);
  block_force_parameters_.SetLength(configuration_length_);

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

// evaluate configurations
void Direct::ConfigurationEvaluation() {
  // finite-difference velocities, accelerations
  ConfigurationToVelocityAcceleration();

  // compute sensor and force predictions
  InverseDynamicsPrediction();
}

// configurations derivatives
void Direct::ConfigurationDerivative() {
  // dimension
  int nv = model->nv;
  int nsen = nsensordata_ * configuration_length_;
  int nforce = nv * (configuration_length_ - 2);

  // operations
  int opsensor = settings.sensor_flag * configuration_length_;
  int opforce = settings.force_flag * (configuration_length_ - 2);

  // inverse dynamics derivatives
  InverseDynamicsDerivatives();

  // velocity, acceleration derivatives
  VelocityAccelerationDerivatives();

  // -- Jacobians -- //
  auto timer_jacobian_start = std::chrono::steady_clock::now();

  // pool count
  int count_begin = pool_.GetCount();

  // individual derivatives
  if (settings.sensor_flag) {
    if (settings.assemble_sensor_jacobian)
      mju_zero(jacobian_sensor_.data(), nsen * ntotal_);
    JacobianSensor();
  }
  if (settings.force_flag) {
    if (settings.assemble_force_jacobian)
      mju_zero(jacobian_force_.data(), nforce * ntotal_);
    JacobianForce();
  }

  // wait
  pool_.WaitCount(count_begin + opsensor + opforce);

  // reset count
  pool_.ResetCount();

  // timers
  timer_.jacobian_sensor += mju_sum(timer_.sensor_step.data(), opsensor);
  timer_.jacobian_force += mju_sum(timer_.force_step.data(), opforce);
  timer_.jacobian_total += GetDuration(timer_jacobian_start);
}

// sensor cost
double Direct::CostSensor(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // update dimension
  int nv = model->nv, ns = nsensordata_;
  int nsen = ns * configuration_length_;

  // residual
  if (!cost_skip_) ResidualSensor();

  // ----- cost ----- //

  // initialize
  double cost = 0.0;

  // zero memory
  if (gradient) mju_zero(gradient, ntotal_);
  if (hessian) mju_zero(hessian, nvel_ * nband_ + nparam_ * ntotal_);
  if (nparam_ > 0) mju_zero(dense_sensor_parameter_.data(), nparam_ * ntotal_);

  // time scaling
  double time_scale = 1.0;
  double time_scale2 = 1.0;
  if (settings.time_scaling_sensor) {
    time_scale = model->opt.timestep * model->opt.timestep;
    time_scale2 = time_scale * time_scale;
  }

  // matrix shift
  int shift_matrix = 0;

  // loop over predictions
  for (int t = 0; t < configuration_length_; t++) {
    // residual
    double* rt = residual_sensor_.data() + ns * t;

    // mask
    int* mask = sensor_mask.Get(t);

    // unpack block
    double* block;
    int block_columns;
    if (t == 0) {  // only position sensors
      block = block_sensor_configuration_.Get(t) + sensor_start_index_ * nv;
      block_columns = nband_ - 2 * nv;
    } else if (t == configuration_length_ - 1) {
      block = block_sensor_configurations_.Get(t);
      block_columns = nband_ - nv;
    } else {  // position, velocity, acceleration sensors
      block = block_sensor_configurations_.Get(t);
      block_columns = nband_;
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

      // dimension
      int nsi = model->sensor_dim[sensor_start_ + i];

      // sensor residual
      double* rti = rt + shift_sensor;

      // weight
      double weight =
          mask[i] ? time_weight / noise_sensor[i] / nsi / configuration_length_
                  : 0.0;

      // first time step
      if (t == 0) weight *= settings.first_step_position_sensors;

      // last time step
      if (t == configuration_length_ - 1)
        weight *= (settings.last_step_position_sensors ||
                   settings.last_step_velocity_sensors);

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

      // gradient wrt configuration: dsidq012' * dndsi
      if (gradient) {
        // sensor block
        double* blocki = block + block_columns * shift_sensor;

        // scratch = dsidq012' * dndsi
        mju_mulMatTVec(scratch_sensor_.data(), blocki, norm_gradient, nsi,
                       block_columns);

        // add
        mju_addToScl(gradient + nv * std::max(0, t - 1), scratch_sensor_.data(),
                     weight, block_columns);

        // parameters
        if (nparam_ > 0) {
          // tmp = dsidp' dndsi
          double* dsidp = block_sensor_parameters_.Get(t) +
                          (sensor_start_index_ + shift_sensor) * nparam_;
          mju_mulMatTVec(scratch_sensor_.data(), dsidp, norm_gradient, nsi,
                         nparam_);
          mju_addToScl(gradient + nvel_, scratch_sensor_.data(), weight,
                       nparam_);
        }
      }

      // Hessian (Gauss-Newton): dsidq012' * d2ndsi2 * dsidq
      if (hessian) {
        // sensor block
        double* blocki = block + block_columns * shift_sensor;

        // step 1: tmp0 = d2ndsi2 * dsidq
        double* tmp0 = scratch_sensor_.data();
        mju_mulMatMat(tmp0, norm_block, blocki, nsi, nsi, block_columns);

        // step 2: hessian = dsidq' * tmp
        double* tmp1 = scratch_sensor_.data() + nsensordata_ * nband_;
        mju_mulMatTMat(tmp1, blocki, tmp0, nsi, block_columns, block_columns);

        // set block in band Hessian
        SetBlockInBand(hessian, tmp1, weight, ntotal_, nband_, block_columns,
                       nv * std::max(0, t - 1));

        // parameters
        if (nparam_ > 0) {
          // parameter Jacobian
          double* dsidp = block_sensor_parameters_.Get(t) +
                          (sensor_start_index_ + shift_sensor) * nparam_;

          // step 1: tmp2 = dsidp' * d2ndsi2
          double* tmp2 = scratch_sensor_.data();
          mju_mulMatTMat(tmp2, dsidp, norm_block, nsi, nparam_, nsi);

          // step 2: tmp3 = tmp2 * dsidp = dsidp' d2ndsi2 dsidp
          double* tmp3 = tmp2 + nparam_ * nsi;
          mju_mulMatMat(tmp3, tmp2, dsidp, nparam_, nsi, nparam_);

          // add dsidp' d2ndsi2 dsidp in dense rows
          AddBlockInMatrix(dense_sensor_parameter_.data(), tmp3, weight,
                           nparam_, ntotal_, nparam_, nparam_, 0, nvel_);

          // step 3: tmp4 = dsidp' * d2ndsi2 * dsidq012
          double* tmp4 = tmp3 + nparam_ * nparam_;
          mju_mulMatTMat(tmp4, dsidp, block, nsi, nparam_, block_columns);

          // add dsidp' * d2ndsi2 * dsidq012 in dense rows
          AddBlockInMatrix(dense_sensor_parameter_.data(), tmp4, weight,
                           nparam_, ntotal_, nparam_, block_columns, 0,
                           nv * std::max(0, t - 1));
        }
      }

      // shift by individual sensor dimension
      shift_sensor += nsi;
      shift_matrix += nsi * nsi;
    }
  }

  // set dense rows in band matrix
  if (hessian && nparam_ > 0) {
    mju_copy(hessian + nvel_ * nband_, dense_sensor_parameter_.data(),
             nparam_ * ntotal_);
  }

  // stop timer
  timer_.cost_sensor_derivatives += GetDuration(start);

  return cost;
}

// sensor residual
void Direct::ResidualSensor() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // loop over predictions
  for (int t = 0; t < configuration_length_; t++) {
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
        if (sensor_stage == mjSTAGE_POS) continue;

        // -- zero memory -- //
        // dimension
        int sensor_dim = model->sensor_dim[sensor_start_ + i];

        // address
        int sensor_adr = model->sensor_adr[sensor_start_ + i];

        // copy sensor data
        mju_zero(rt + sensor_adr - sensor_start_index_, sensor_dim);
      }
    }

    // zero out acceleration sensors at last time step
    if (t == configuration_length_ - 1) {
      // loop over position sensors
      for (int i = 0; i < nsensor_; i++) {
        // sensor stage
        int sensor_stage = model->sensor_needstage[sensor_start_ + i];

        // check for position
        if (sensor_stage == mjSTAGE_POS &&
            settings.last_step_position_sensors) {
          continue;
        }

        // check for velocity
        if (sensor_stage == mjSTAGE_VEL &&
            settings.last_step_velocity_sensors) {
          continue;
        }

        // -- zero memory -- //
        // dimension
        int sensor_dim = model->sensor_dim[sensor_start_ + i];

        // address
        int sensor_adr = model->sensor_adr[sensor_start_ + i];

        // copy sensor data
        mju_zero(rt + sensor_adr - sensor_start_index_, sensor_dim);
      }
    }
  }

  // stop timer
  timer_.residual_sensor += GetDuration(start);
}

// sensor Jacobian blocks (dsdq0, dsdq1, dsdq2)
void Direct::BlockSensor(int index) {
  // dimensions
  int nv = model->nv, ns = nsensordata_;
  int nsen = nsensordata_ * configuration_length_;

  // shift
  int shift = sensor_start_index_ * nv;

  // first time step
  if (index == 0) {
    double* block = block_sensor_configuration_.Get(0) + shift;

    // unpack
    double* dsdq012 = block_sensor_configurations_.Get(0);

    // set dsdq1
    SetBlockInMatrix(dsdq012, block, 1.0, ns, nband_, ns, nv, 0, nv);

    // set block in dense Jacobian
    if (settings.assemble_sensor_jacobian) {
      SetBlockInMatrix(jacobian_sensor_.data(), block, 1.0, nsen, ntotal_,
                       nsensordata_, nv, 0, 0);
    }
    return;
  }

  // last time step
  if (index == configuration_length_ - 1) {
    // dqds
    double* dsdq = block_sensor_configuration_.Get(index) + shift;

    // dvds
    double* dsdv = block_sensor_velocity_.Get(index) + shift;

    // -- configuration previous: dsdq0 = dsdv * dvdq0-- //

    // unpack
    double* dsdq0 = block_sensor_previous_configuration_.Get(index);
    double* tmp = block_sensor_scratch_.Get(index);

    // dsdq0 <- dvds' * dvdq0
    double* dvdq0 = block_velocity_previous_configuration_.Get(index);
    mju_mulMatMat(dsdq0, dsdv, dvdq0, ns, nv, nv);

    // -- configuration current: dsdq1 = dsdq + dsdv * dvdq1 --

    // unpack
    double* dsdq1 = block_sensor_current_configuration_.Get(index);

    // dsdq1 <- dqds'
    mju_copy(dsdq1, dsdq, ns * nv);

    // dsdq1 += dvds' * dvdq1
    double* dvdq1 = block_velocity_current_configuration_.Get(index);
    mju_mulMatMat(tmp, dsdv, dvdq1, ns, nv, nv);
    mju_addTo(dsdq1, tmp, ns * nv);

    // -- assemble dsdq01 block -- //

    // unpack
    double* dsdq01 = block_sensor_configurations_.Get(index);

    // set dfdq0
    SetBlockInMatrix(dsdq01, dsdq0, 1.0, ns, 2 * nv, ns, nv, 0, 0 * nv);

    // set dfdq1
    SetBlockInMatrix(dsdq01, dsdq1, 1.0, ns, 2 * nv, ns, nv, 0, 1 * nv);

    // assemble dense Jacobian
    if (settings.assemble_sensor_jacobian) {
      // set block
      SetBlockInMatrix(jacobian_sensor_.data(), dsdq01, 1.0, nsen, ntotal_,
                       nsensordata_, 2 * nv, index * nsensordata_,
                       (index - 1) * nv);
    }
    return;
  }

  // -- timesteps [1,...,T - 1] -- //

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
  SetBlockInMatrix(dsdq012, dsdq0, 1.0, ns, nband_, ns, nv, 0, 0 * nv);

  // set dfdq1
  SetBlockInMatrix(dsdq012, dsdq1, 1.0, ns, nband_, ns, nv, 0, 1 * nv);

  // set dfdq0
  SetBlockInMatrix(dsdq012, dsdq2, 1.0, ns, nband_, ns, nv, 0, 2 * nv);

  // assemble dense Jacobian
  if (settings.assemble_sensor_jacobian) {
    // set block
    SetBlockInMatrix(jacobian_sensor_.data(), dsdq012, 1.0, nsen, ntotal_,
                     nsensordata_, nband_, index * nsensordata_,
                     (index - 1) * nv);
  }
}

// sensor Jacobian
// note: pool wait is called outside this function
void Direct::JacobianSensor() {
  // loop over predictions
  for (int t = 0; t < configuration_length_; t++) {
    // schedule by time step
    pool_.Schedule([&direct = *this, t]() {
      // start Jacobian timer
      auto jacobian_sensor_start = std::chrono::steady_clock::now();

      // block
      direct.BlockSensor(t);

      // stop Jacobian timer
      direct.timer_.sensor_step[t] = GetDuration(jacobian_sensor_start);
    });
  }
}

// force cost
double Direct::CostForce(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // update dimension
  int nv = model->nv;
  int nforce = nv * (configuration_length_ - 2);

  // residual
  if (!cost_skip_) ResidualForce();

  // ----- cost ----- //

  // initialize
  double cost = 0.0;

  // zero memory
  if (gradient) mju_zero(gradient, ntotal_);
  if (hessian) mju_zero(hessian, nvel_ * nband_ + nparam_ * ntotal_);
  if (nparam_ > 0) mju_zero(dense_force_parameter_.data(), nparam_ * ntotal_);

  // time scaling
  double time_scale2 = 1.0;
  if (settings.time_scaling_force) {
    time_scale2 = model->opt.timestep * model->opt.timestep *
                  model->opt.timestep * model->opt.timestep;
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
    norm_force_[t] = 0.5 * mju_dot(rt, norm_gradient, nv);

    // weighted norm
    cost += norm_force_[t];

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

    // gradient wrt configuration: dfdq012' * dndf
    if (gradient) {
      // scratch = dfdq012' * dndf
      mju_mulMatTVec(scratch_force_.data(), block, norm_gradient, nv, nband_);

      // add
      mju_addToScl(gradient + (t - 1) * nv, scratch_force_.data(), 1.0, nband_);

      // parameters
      if (nparam_ > 0) {
        // tmp = dfdp' dndf
        double* dpdf = block_force_parameters_.Get(t);  // already transposed
        mju_mulMatVec(scratch_force_.data(), dpdf, norm_gradient, nparam_, nv);
        mju_addToScl(gradient + nvel_, scratch_force_.data(), 1.0, nparam_);
      }
    }

    // Hessian (Gauss-Newton): drdq012' * d2ndf2 * dfdq012
    if (hessian) {
      // step 1: tmp0 = d2ndf2 * dfdq012
      double* tmp0 = scratch_force_.data();
      mju_mulMatMat(tmp0, norm_block, block, nv, nv, nband_);

      // step 2: hessian = dfdq012' * tmp
      double* tmp1 = tmp0 + nv * nband_;
      mju_mulMatTMat(tmp1, block, tmp0, nv, nband_, nband_);

      // set block in band Hessian
      SetBlockInBand(hessian, tmp1, 1.0, ntotal_, nband_, nband_, nv * (t - 1));

      // parameters
      if (nparam_ > 0) {
        // parameter Jacobian
        double* dpdf = block_force_parameters_.Get(t);

        // step 1: tmp2 = dpdf * d2ndf2
        double* tmp2 = scratch_force_.data();
        mju_mulMatMat(tmp2, dpdf, norm_block, nparam_, nv, nv);

        // step 2: tmp3 = tmp2 * dpdf' = dpdf * d2ndf2 * dpdf'
        double* tmp3 = tmp2 + nparam_ * nv;
        mju_mulMatMatT(tmp3, tmp2, dpdf, nparam_, nv, nparam_);

        // add dpdf * d2ndf2 * dfdp in dense rows
        AddBlockInMatrix(dense_force_parameter_.data(), tmp3, 1.0, nparam_,
                         ntotal_, nparam_, nparam_, 0, nvel_);

        // step 3: tmp4 = dpdf * d2ndf2 * dfdq012
        double* tmp4 = tmp3 + nparam_ * nparam_;
        mju_mulMatMat(tmp4, tmp2, block, nparam_, nv, nband_);

        // add dpdf * d2ndf2 * dfdq012 in dense rows
        AddBlockInMatrix(dense_force_parameter_.data(), tmp4, 1.0, nparam_,
                         ntotal_, nparam_, nband_, 0, (t - 1) * nv);
      }
    }
  }

  // set dense rows in band Hessian
  if (hessian && nparam_ > 0) {
    mju_copy(hessian + nvel_ * nband_, dense_force_parameter_.data(),
             nparam_ * ntotal_);
  }

  // stop timer
  timer_.cost_force_derivatives += GetDuration(start);

  return cost;
}

// force residual
void Direct::ResidualForce() {
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
void Direct::BlockForce(int index) {
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
  SetBlockInMatrix(dfdq012, dfdq0, 1.0, nv, nband_, nv, nv, 0, 0 * nv);

  // set dfdq1
  SetBlockInMatrix(dfdq012, dfdq1, 1.0, nv, nband_, nv, nv, 0, 1 * nv);

  // set dfdq0
  SetBlockInMatrix(dfdq012, dfdq2, 1.0, nv, nband_, nv, nv, 0, 2 * nv);

  // assemble dense Jacobian
  if (settings.assemble_force_jacobian) {
    // dimensions
    int nv = model->nv;
    int nforce = nv * (configuration_length_ - 2);

    // set block
    SetBlockInMatrix(jacobian_force_.data(), dfdq012, 1.0, nforce, ntotal_, nv,
                     nband_, (index - 1) * nv, (index - 1) * nv);
  }
}

// force Jacobian
// note: pool wait is called outside this function
void Direct::JacobianForce() {
  // loop over predictions
  for (int t = 1; t < configuration_length_ - 1; t++) {
    // schedule by time step
    pool_.Schedule([&direct = *this, t]() {
      // start Jacobian timer
      auto jacobian_force_start = std::chrono::steady_clock::now();

      // block
      direct.BlockForce(t);

      // stop Jacobian timer
      direct.timer_.force_step[t] = GetDuration(jacobian_force_start);
    });
  }
}

// compute force
void Direct::InverseDynamicsPrediction() {
  // compute sensor and force predictions
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model->nq, nv = model->nv, na = model->na, ns = nsensordata_;

  // set parameters
  if (nparam_ > 0) {
    model_parameters_[model_parameters_id_]->Set(model, parameters.data(),
                                                 nparam_);
  }

  // pool count
  int count_before = pool_.GetCount();

  // first time step
  pool_.Schedule([&direct = *this, nq, nv]() {
    // time index
    int t = 0;

    // data
    mjData* d = direct.data_[t].get();

    // terms
    double* q0 = direct.configuration.Get(t);
    double* y0 = direct.sensor_prediction.Get(t);
    mju_zero(y0, direct.nsensordata_);

    // set data
    mju_copy(d->qpos, q0, nq);
    mju_zero(d->qvel, nv);
    mju_zero(d->qacc, nv);
    d->time = direct.times.Get(t)[0];

    // position sensors
    mj_fwdPosition(direct.model, d);
    mj_sensorPos(direct.model, d);
    if (direct.model->opt.enableflags & (mjENBL_ENERGY)) {
      mj_energyPos(direct.model, d);
    }

    // loop over position sensors
    for (int i = 0; i < direct.nsensor_; i++) {
      // sensor stage
      int sensor_stage =
          direct.model->sensor_needstage[direct.sensor_start_ + i];

      // check for position
      if (sensor_stage == mjSTAGE_POS) {
        // dimension
        int sensor_dim = direct.model->sensor_dim[direct.sensor_start_ + i];

        // address
        int sensor_adr = direct.model->sensor_adr[direct.sensor_start_ + i];

        // copy sensor data
        mju_copy(y0 + sensor_adr - direct.sensor_start_index_,
                 d->sensordata + sensor_adr, sensor_dim);
      }
    }
  });

  // loop over predictions
  for (int t = 1; t < configuration_length_ - 1; t++) {
    // schedule
    pool_.Schedule([&direct = *this, nq, nv, na, ns, t]() {
      // terms
      double* qt = direct.configuration.Get(t);
      double* vt = direct.velocity.Get(t);
      double* at = direct.acceleration.Get(t);

      // data
      mjData* d = direct.data_[t].get();

      // set qt, vt, at
      mju_copy(d->qpos, qt, nq);
      mju_copy(d->qvel, vt, nv);
      mju_copy(d->qacc, at, nv);

      // inverse dynamics
      mj_inverse(direct.model, d);

      // copy sensor
      double* st = direct.sensor_prediction.Get(t);
      mju_copy(st, d->sensordata + direct.sensor_start_index_, ns);

      // copy force
      double* ft = direct.force_prediction.Get(t);
      mju_copy(ft, d->qfrc_inverse, nv);

      // copy act
      double* act = direct.act.Get(t + 1);
      mju_copy(act, d->act, na);
    });
  }

  // last time step
  pool_.Schedule([&direct = *this, nq, nv]() {
    // time index
    int t = direct.ConfigurationLength() - 1;

    // data
    mjData* d = direct.data_[t].get();

    // terms
    double* qT = direct.configuration.Get(t);
    double* vT = direct.velocity.Get(t);
    double* yT = direct.sensor_prediction.Get(t);
    mju_zero(yT, direct.nsensordata_);

    // set data
    mju_copy(d->qpos, qT, nq);
    mju_copy(d->qvel, vT, nv);
    mju_zero(d->qacc, nv);
    d->time = direct.times.Get(t)[0];

    // position sensors
    mj_fwdPosition(direct.model, d);
    mj_sensorPos(direct.model, d);
    if (direct.model->opt.enableflags & (mjENBL_ENERGY)) {
      mj_energyPos(direct.model, d);
    }

    // velocity sensors
    mj_fwdVelocity(direct.model, d);
    mj_sensorVel(direct.model, d);
    if (direct.model->opt.enableflags & (mjENBL_ENERGY)) {
      mj_energyVel(direct.model, d);
    }

    // loop over position sensors
    for (int i = 0; i < direct.nsensor_; i++) {
      // sensor stage
      int sensor_stage =
          direct.model->sensor_needstage[direct.sensor_start_ + i];

      // check for position
      if (sensor_stage == mjSTAGE_POS || sensor_stage == mjSTAGE_VEL) {
        // dimension
        int sensor_dim = direct.model->sensor_dim[direct.sensor_start_ + i];

        // address
        int sensor_adr = direct.model->sensor_adr[direct.sensor_start_ + i];

        // copy sensor data
        mju_copy(yT + sensor_adr - direct.sensor_start_index_,
                 d->sensordata + sensor_adr, sensor_dim);
      }
    }
  });

  // wait
  pool_.WaitCount(count_before + configuration_length_);
  pool_.ResetCount();

  // stop timer
  timer_.cost_prediction += GetDuration(start);
}

// compute inverse dynamics derivatives (via finite difference)
void Direct::InverseDynamicsDerivatives() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model->nq, nv = model->nv;

  // set parameters
  if (nparam_ > 0) {
    model_parameters_[model_parameters_id_]->Set(model, parameters.data(),
                                                 nparam_);
  }

  // pool count
  int count_before = pool_.GetCount();

  // first time step
  pool_.Schedule([&direct = *this, nq, nv]() {
    // time index
    int t = 0;

    // data
    mjData* d = direct.data_[t].get();

    // terms
    double* q0 = direct.configuration.Get(t);
    double* dsdq = direct.block_sensor_configuration_.Get(t);

    // set data
    mju_copy(d->qpos, q0, nq);
    mju_zero(d->qvel, nv);
    mju_zero(d->qacc, nv);
    d->time = direct.times.Get(t)[0];

    // finite-difference derivatives
    double* dqds = direct.block_sensor_configurationT_.Get(t);
    mjd_inverseFD(direct.model, d, direct.finite_difference.tolerance,
                  direct.finite_difference.flg_actuation, NULL, NULL, NULL,
                  dqds, NULL, NULL, NULL);
    // transpose
    mju_transpose(dsdq, dqds, nv, direct.model->nsensordata);

    // parameters
    if (direct.nparam_ > 0) {
      direct.ParameterJacobian(t);
    }

    // loop over position sensors
    for (int i = 0; i < direct.nsensor_; i++) {
      // sensor stage
      int sensor_stage =
          direct.model->sensor_needstage[direct.sensor_start_ + i];

      // dimension
      int sensor_dim = direct.model->sensor_dim[direct.sensor_start_ + i];

      // address
      int sensor_adr = direct.model->sensor_adr[direct.sensor_start_ + i];

      // check for position
      if (sensor_stage != mjSTAGE_POS) {
        // zero remaining rows
        mju_zero(dsdq + sensor_adr * nv, sensor_dim * nv);

        // parameter Jacobian
        if (direct.nparam_) {
          mju_zero(direct.block_sensor_parameters_.Get(t) +
                       sensor_adr * direct.nparam_,
                   sensor_dim * direct.nparam_);
        }
      }
    }
  });

  // loop over predictions
  for (int t = 1; t < configuration_length_ - 1; t++) {
    // schedule
    pool_.Schedule([&direct = *this, nq, nv, t]() {
      // unpack
      double* q = direct.configuration.Get(t);
      double* v = direct.velocity.Get(t);
      double* a = direct.acceleration.Get(t);

      double* dsdq = direct.block_sensor_configuration_.Get(t);
      double* dsdv = direct.block_sensor_velocity_.Get(t);
      double* dsda = direct.block_sensor_acceleration_.Get(t);
      double* dqds = direct.block_sensor_configurationT_.Get(t);
      double* dvds = direct.block_sensor_velocityT_.Get(t);
      double* dads = direct.block_sensor_accelerationT_.Get(t);
      double* dqdf = direct.block_force_configuration_.Get(t);
      double* dvdf = direct.block_force_velocity_.Get(t);
      double* dadf = direct.block_force_acceleration_.Get(t);
      mjData* data = direct.data_[t].get();  // TODO(taylor): WorkerID

      // set state, acceleration
      mju_copy(data->qpos, q, nq);
      mju_copy(data->qvel, v, nv);
      mju_copy(data->qacc, a, nv);

      // finite-difference derivatives
      mjd_inverseFD(direct.model, data, direct.finite_difference.tolerance,
                    direct.finite_difference.flg_actuation, dqdf, dvdf, dadf,
                    dqds, dvds, dads, NULL);

      // transpose
      mju_transpose(dsdq, dqds, nv, direct.model->nsensordata);
      mju_transpose(dsdv, dvds, nv, direct.model->nsensordata);
      mju_transpose(dsda, dads, nv, direct.model->nsensordata);

      // parameters
      if (direct.nparam_ > 0) {
        direct.ParameterJacobian(t);
      }
    });
  }

  // last time step
  pool_.Schedule([&direct = *this, nq, nv]() {
    // time index
    int t = direct.ConfigurationLength() - 1;

    // data
    mjData* d = direct.data_[t].get();

    // terms
    double* qT = direct.configuration.Get(t);
    double* vT = direct.velocity.Get(t);
    double* dsdq = direct.block_sensor_configuration_.Get(t);
    double* dsdv = direct.block_sensor_velocity_.Get(t);

    // set data
    mju_copy(d->qpos, qT, nq);
    mju_copy(d->qvel, vT, nv);
    mju_zero(d->qacc, nv);
    d->time = direct.times.Get(t)[0];

    // finite-difference derivatives
    double* dqds = direct.block_sensor_configurationT_.Get(t);
    double* dvds = direct.block_sensor_velocityT_.Get(t);
    mjd_inverseFD(direct.model, d, direct.finite_difference.tolerance,
                  direct.finite_difference.flg_actuation, NULL, NULL, NULL,
                  dqds, dvds, NULL, NULL);
    // transpose
    mju_transpose(dsdq, dqds, nv, direct.model->nsensordata);
    mju_transpose(dsdv, dvds, nv, direct.model->nsensordata);

    // parameters
    if (direct.nparam_ > 0) {
      direct.ParameterJacobian(t);
    }

    // loop over position sensors
    for (int i = 0; i < direct.nsensor_; i++) {
      // sensor stage
      int sensor_stage =
          direct.model->sensor_needstage[direct.sensor_start_ + i];

      // dimension
      int sensor_dim = direct.model->sensor_dim[direct.sensor_start_ + i];

      // address
      int sensor_adr = direct.model->sensor_adr[direct.sensor_start_ + i];

      // check for position
      if (sensor_stage == mjSTAGE_ACC) {
        // zero remaining rows
        mju_zero(dsdq + sensor_adr * nv, sensor_dim * nv);
        mju_zero(dsdv + sensor_adr * nv, sensor_dim * nv);

        // parameter Jacobian
        if (direct.nparam_) {
          mju_zero(direct.block_sensor_parameters_.Get(t) +
                       sensor_adr * direct.nparam_,
                   sensor_dim * direct.nparam_);
        }
      }
    }
  });

  // wait
  pool_.WaitCount(count_before + configuration_length_);

  // reset pool count
  pool_.ResetCount();

  // stop timer
  timer_.inverse_dynamics_derivatives += GetDuration(start);
}

// update configuration trajectory
void Direct::UpdateConfiguration(DirectTrajectory<double>& candidate,
                                 const DirectTrajectory<double>& configuration,
                                 const double* search_direction,
                                 double step_size) {
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

  // stop timer
  timer_.configuration_update += GetDuration(start);
}

// convert sequence of configurations to velocities and accelerations
void Direct::ConfigurationToVelocityAcceleration() {
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
void Direct::VelocityAccelerationDerivatives() {
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

// compute total cost
double Direct::Cost(double* gradient, double* hessian) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // evaluate configurations
  if (!cost_skip_) ConfigurationEvaluation();

  // derivatives
  if (gradient || hessian) {
    ConfigurationDerivative();
  }

  // start cost derivative timer
  auto start_cost_derivatives = std::chrono::steady_clock::now();

  // pool count
  int count_begin = pool_.GetCount();

  bool gradient_flag = (gradient ? true : false);
  bool hessian_flag = (hessian ? true : false);

  // -- individual cost derivatives -- //

  // sensor
  if (settings.sensor_flag) {
    pool_.Schedule([&direct = *this, gradient_flag, hessian_flag]() {
      direct.cost_sensor_ = direct.CostSensor(
          gradient_flag ? direct.cost_gradient_sensor_.data() : NULL,
          hessian_flag ? direct.cost_hessian_sensor_band_.data() : NULL);
    });
  }

  // force
  if (settings.force_flag) {
    pool_.Schedule([&direct = *this, gradient_flag, hessian_flag]() {
      direct.cost_force_ = direct.CostForce(
          gradient_flag ? direct.cost_gradient_force_.data() : NULL,
          hessian_flag ? direct.cost_hessian_force_band_.data() : NULL);
    });
  }

  // wait
  pool_.WaitCount(count_begin + settings.sensor_flag + settings.force_flag);
  pool_.ResetCount();

  // total cost
  double cost = cost_sensor_ + cost_force_;

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

  // parameters
  if (nparam_ > 0) {
    // zero dense rows
    mju_zero(dense_parameter_.data(), nparam_ * ntotal_);

    // zero parameter cost_
    cost_parameter_ = 0.0;

    // loop over parameters
    for (int i = 0; i < nparam_; i++) {
      // parameter difference
      double parameter_diff = parameters[i] - parameters_previous[i];

      // weight
      double weight = 1.0 / noise_parameter[i] / nparam_;

      // cost
      cost_parameter_ += 0.5 * weight * parameter_diff * parameter_diff;

      // gradient
      if (gradient) {
        gradient[nvel_ + i] = weight * parameter_diff;
      }

      // Hessian
      if (hessian) {
        dense_parameter_[i * ntotal_ + nvel_ + i] = weight;
      }
    }

    // total cost
    cost += cost_parameter_;

    // set dense rows in band Hessian
    if (hessian) {
      mju_copy(hessian + nvel_ * nband_, dense_parameter_.data(),
               nparam_ * ntotal_);
    }
  }

  // total cost
  return cost;
}

// compute total gradient
void Direct::TotalGradient(double* gradient) {
  if (!gradient) return;

  // start gradient timer
  auto start = std::chrono::steady_clock::now();

  // zero memory
  mju_zero(gradient, ntotal_);

  // individual gradients
  if (settings.sensor_flag) {
    mju_addTo(gradient, cost_gradient_sensor_.data(), ntotal_);
  }
  if (settings.force_flag) {
    mju_addTo(gradient, cost_gradient_force_.data(), ntotal_);
  }

  // stop gradient timer
  timer_.cost_gradient += GetDuration(start);
}

// compute total Hessian
void Direct::TotalHessian(double* hessian) {
  if (!hessian) return;

  // start Hessian timer
  auto start = std::chrono::steady_clock::now();

  // zero memory
  mju_zero(hessian, nvel_ * nband_ + nparam_ * ntotal_);

  // individual Hessians
  if (settings.sensor_flag) {
    mju_addTo(hessian, cost_hessian_sensor_band_.data(),
              nvel_ * nband_ + nparam_ * ntotal_);
  }

  if (settings.force_flag) {
    mju_addTo(hessian, cost_hessian_force_band_.data(),
              nvel_ * nband_ + nparam_ * ntotal_);
  }

  // stop Hessian timer
  timer_.cost_hessian += GetDuration(start);
}

// optimize configuration trajectory
void Direct::Optimize() {
  // start timer
  auto start_optimize = std::chrono::steady_clock::now();

  // set status
  gradient_norm_ = 0.0;
  search_direction_norm_ = 0.0;
  solve_status_ = kUnsolved;

  // reset timers
  ResetTimers();

  // initial cost
  cost_count_ = 0;
  cost_skip_ = false;
  cost_ = Cost(NULL, NULL);
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
    Cost(cost_gradient_.data(), cost_hessian_band_.data());

    // start timer
    auto start_search = std::chrono::steady_clock::now();

    // -- gradient -- //
    double* gradient = cost_gradient_.data();

    // gradient tolerance check
    gradient_norm_ = mju_norm(gradient, ntotal_) / ntotal_;
    if (gradient_norm_ < settings.gradient_tolerance) {
      break;
    }

    // ----- line / curve search ----- //

    // copy configuration
    mju_copy(configuration_copy_.Data(), configuration.Data(),
             model->nq * configuration_length_);

    // copy parameters
    mju_copy(parameters_copy_.data(), parameters.data(), nparam_);

    // initialize
    double cost_candidate = cost_;
    int iteration_search = 0;
    step_size_ = 1.0;
    regularization_ = settings.regularization_initial;
    improvement_ = -1.0;

    // -- search direction -- //

    // check regularization
    if (regularization_ >= kMaxDirectRegularization - 1.0e-6) {
      // set solve status
      solve_status_ = kMaxRegularizationFailure;

      // failure
      return;
    }

    // compute initial search direction
    if (!SearchDirection()) {
      return;  // failure
    }

    // check small search direction
    if (search_direction_norm_ < settings.search_direction_tolerance) {
      // set solve status
      solve_status_ = kSmallDirectionFailure;

      // failure
      return;
    }

    // backtracking until cost decrease
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
            if (!SearchDirection()) {
              return;  // failure
            }

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

      // candidate configurations
      UpdateConfiguration(configuration, configuration_copy_,
                          search_direction_.data(), -1.0 * step_size_);

      // candidate parameters
      if (nparam_ > 0) {
        mju_copy(parameters.data(), parameters_copy_.data(), nparam_);
        mju_addToScl(parameters.data(), search_direction_.data() + nvel_,
                     -1.0 * step_size_, nparam_);
      }

      // cost
      cost_skip_ = false;
      cost_candidate = Cost(NULL, NULL);

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
          mju_dot(cost_gradient_.data(), search_direction_.data(), ntotal_);

      // tmp = H * d
      double* tmp = scratch_expected_.data();
      mju_bandMulMatVec(tmp, cost_hessian_band_.data(),
                        search_direction_.data(), ntotal_, nband_, nparam_, 1,
                        true);

      // expected += 0.5 d' tmp
      expected_ += 0.5 * mju_dot(search_direction_.data(), tmp, ntotal_);

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
            mju_max(kMinDirectRegularization,
                    regularization_ / settings.regularization_scaling);
      } else if (reduction_ratio_ < 0.25) {
        // increase
        regularization_ =
            mju_min(kMaxDirectRegularization,
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
bool Direct::SearchDirection() {
  // start timer
  auto search_direction_start = std::chrono::steady_clock::now();

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
    if (regularization_ >= kMaxDirectRegularization) {
      printf("min diag = %f\n", min_diag);
      printf("cost Hessian factorization failure: MAX REGULARIZATION\n");
      solve_status_ = kMaxRegularizationFailure;
      return false;
    }

    // copy
    mju_copy(hessian_band_factor, hessian_band,
             nvel_ * nband_ + nparam_ * ntotal_);

    // factorize
    min_diag = mju_cholFactorBand(hessian_band_factor, ntotal_, nband_, nparam_,
                                  regularization_, 0.0);

    // increase regularization
    if (min_diag <= 0.0) {
      IncreaseRegularization();
    }
  }

  // compute search direction
  mju_cholSolveBand(direction, hessian_band_factor, gradient, ntotal_, nband_,
                    nparam_);

  // search direction norm
  search_direction_norm_ = InfinityNorm(direction, ntotal_);

  // set regularization
  if (regularization_ > 0.0) {
    // configurations
    for (int i = 0; i < ntotal_; i++) {
      hessian_band[i * nband_ + nband_ - 1] += regularization_;
    }

    // parameters
    for (int i = 0; i < nparam_; i++) {
      hessian_band[nvel_ * nband_ + i * ntotal_ + nvel_ + i] += regularization_;
    }
  }

  // end timer
  timer_.search_direction += GetDuration(search_direction_start);
  return true;
}

// print Optimize status
void Direct::PrintOptimize() {
  if (!settings.verbose_optimize) return;

  // title
  printf("Direct::Optimize Status:\n\n");

  // timing
  printf("Timing:\n");

  printf("\n");
  printf("  cost : %.3f (ms) \n", 1.0e-3 * timer_.cost / cost_count_);
  printf("    - sensor: %.3f (ms) \n",
         1.0e-3 * timer_.cost_sensor / cost_count_);
  printf("    - force: %.3f (ms) \n", 1.0e-3 * timer_.cost_force / cost_count_);
  printf("    - qpos -> qvel, qacc: %.3f (ms) \n",
         1.0e-3 * timer_.cost_config_to_velacc / cost_count_);
  printf("    - prediction: %.3f (ms) \n",
         1.0e-3 * timer_.cost_prediction / cost_count_);
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
  printf("      < sensor: %.3f (ms) \n", 1.0e-3 * timer_.jacobian_sensor);
  printf("      < force: %.3f (ms) \n", 1.0e-3 * timer_.jacobian_force);
  printf("    - gradient, hessian [total]: %.3f (ms) \n",
         1.0e-3 * timer_.cost_total_derivatives);
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
  printf("    - sensor: %.3f\n", cost_sensor_);
  printf("    - force: %.3f\n", cost_force_);
  printf("  <initial: %.3f>\n", cost_initial_);
  printf("\n");

  fflush(stdout);
}

// print cost
void Direct::PrintCost() {
  if (settings.verbose_cost) {
    printf("cost (total): %.3f\n", cost_);
    printf("  sensor: %.3f\n", cost_sensor_);
    printf("  force: %.3f\n", cost_force_);
    printf("  [initial: %.3f]\n", cost_initial_);
    fflush(stdout);
  }
}

// reset timers
void Direct::ResetTimers() {
  timer_.inverse_dynamics_derivatives = 0.0;
  timer_.velacc_derivatives = 0.0;
  timer_.jacobian_sensor = 0.0;
  timer_.jacobian_force = 0.0;
  timer_.jacobian_total = 0.0;
  timer_.cost_sensor_derivatives = 0.0;
  timer_.cost_force_derivatives = 0.0;
  timer_.cost_total_derivatives = 0.0;
  timer_.cost_gradient = 0.0;
  timer_.cost_hessian = 0.0;
  timer_.cost_derivatives = 0.0;
  timer_.cost = 0.0;
  timer_.cost_sensor = 0.0;
  timer_.cost_force = 0.0;
  timer_.cost_config_to_velacc = 0.0;
  timer_.cost_prediction = 0.0;
  timer_.residual_sensor = 0.0;
  timer_.residual_force = 0.0;
  timer_.search_direction = 0.0;
  timer_.search = 0.0;
  timer_.configuration_update = 0.0;
  timer_.optimize = 0.0;
  timer_.update_trajectory = 0.0;
}

// direct status string
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

// increase regularization
void Direct::IncreaseRegularization() {
  regularization_ = mju_min(kMaxDirectRegularization,
                            regularization_ * settings.regularization_scaling);
}

// derivatives of sensor model wrt parameters
void Direct::ParameterJacobian(int index) {
  // unpack
  mjModel* model_perturb = model_perturb_[index].get();
  mjData* data = data_[index].get();
  double* dsdp = block_sensor_parameters_.Get(index);
  double* dpds = block_sensor_parametersT_.Get(index);
  double* dpdf = block_force_parameters_.Get(index);
  double* param = parameters_copy_.data() + index * nparam_;
  mju_copy(param, parameters.data(), nparam_);

  // loop over parameters
  for (int i = 0; i < nparam_; i++) {
    // unpack
    double* dpids = dpds + i * model->nsensordata;
    double* dpidf = dpdf + i * model->nv;

    // nudge
    param[i] += finite_difference.tolerance;

    // set parameters
    model_parameters_[model_parameters_id_]->Set(model_perturb, param, nparam_);

    // inverse dynamics
    mj_inverse(model_perturb, data);

    // sensor difference
    mju_sub(dpids, data->sensordata, sensor_prediction.Get(index),
            model->nsensordata);

    // force difference
    mju_sub(dpidf, data->qfrc_inverse, force_prediction.Get(index), model->nv);

    // scale
    mju_scl(dpids, dpids, 1.0 / finite_difference.tolerance,
            model->nsensordata);
    mju_scl(dpidf, dpidf, 1.0 / finite_difference.tolerance, model->nv);

    // restore
    param[i] -= finite_difference.tolerance;
  }

  // transpose
  mju_transpose(dsdp, dpds, nparam_, model->nsensordata);
}

}  // namespace mjpc
