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

#include "mjpc/array_safety.h"
#include "mjpc/estimators/buffer.h"
#include "mjpc/norm.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;

// initialize batch estimator
void Batch::Initialize(const mjModel* model) {
  // model
  // if (this->model) mj_deleteModel(this->model);
  this->model = mj_copyModel(nullptr, model);

  // data
  for (int i = 0; i < MAX_HISTORY; i++) {
    data_.push_back(MakeUniqueMjData(mj_makeData(model)));
  }

  // timestep
  this->model->opt.timestep = GetNumberOrDefault(this->model->opt.timestep,
                                                 model, "estimator_timestep");

  // dimension
  int nq = model->nq, nv = model->nv, na = model->na;
  nstate_ = nq + nv + na;
  ndstate_ = 2 * nv + na;

  // check for na > 0 
  // TODO(taylor)
  if (na > 0) {
    mju_error("na > 0: act not supported\n");
  }

  // sensor start index
  sensor_start = GetNumberOrDefault(0, model, "estimator_sensor_start");

  // number of sensors
  nsensor =
      GetNumberOrDefault(model->nsensor, model, "estimator_number_sensor");

  // sensor dimension
  nsensordata_ = 0;
  for (int i = 0; i < nsensor; i++) {
    nsensordata_ += model->sensor_dim[sensor_start + i];
  }

  // sensor start index
  sensor_start_index_ = 0;
  for (int i = 0; i < sensor_start; i++) {
    sensor_start_index_ += model->sensor_dim[i];
  }

  // state
  state.resize(nstate_);

  // covariance
  covariance.resize(ndstate_ * ndstate_);

  // process noise
  noise_process.resize(ndstate_);

  // sensor noise
  noise_sensor.resize(nsensor);

  // length of configuration trajectory
  configuration_length_ =
      GetNumberOrDefault(3, model, "batch_configuration_length");

  // number of predictions
  prediction_length_ = configuration_length_ - 2;

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
  sensor_mask.Initialize(nsensor, configuration_length_);

  // force
  force_measurement.Initialize(nv, configuration_length_);
  force_prediction.Initialize(nv, configuration_length_);

  // residual
  residual_prior_.resize(nv * MAX_HISTORY);
  residual_sensor_.resize(nsensordata_ * MAX_HISTORY);
  residual_force_.resize(nv * MAX_HISTORY);

  // Jacobian
  jacobian_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  jacobian_sensor_.resize((nsensordata_ * MAX_HISTORY) * (nv * MAX_HISTORY));
  jacobian_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // prior Jacobian block
  block_prior_current_configuration_.Initialize(nv * nv, configuration_length_);

  // sensor Jacobian blocks
  block_sensor_configuration_.Initialize(model->nsensordata * nv, prediction_length_);
  block_sensor_velocity_.Initialize(model->nsensordata * nv, prediction_length_);
  block_sensor_acceleration_.Initialize(model->nsensordata * nv, prediction_length_);
  block_sensor_configurationT_.Initialize(model->nsensordata * nv, prediction_length_);
  block_sensor_velocityT_.Initialize(model->nsensordata * nv, prediction_length_);
  block_sensor_accelerationT_.Initialize(model->nsensordata * nv, prediction_length_);

  block_sensor_previous_configuration_.Initialize(nsensordata_ * nv,
                                                  prediction_length_);
  block_sensor_current_configuration_.Initialize(nsensordata_ * nv,
                                                 prediction_length_);
  block_sensor_next_configuration_.Initialize(nsensordata_ * nv,
                                              prediction_length_);
  block_sensor_configurations_.Initialize(nsensordata_ * 3 * nv,
                                          prediction_length_);

  block_sensor_scratch_.Initialize(
      mju_max(nv, nsensordata_) * mju_max(nv, nsensordata_),
      prediction_length_);

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
  cost_hessian_band_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_band_factor_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  cost_hessian_factor_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // prior weights
  scale_prior = GetNumberOrDefault(1.0, model, "batch_scale_prior");
  weight_prior.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  weight_prior_band_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch_prior_weight_.resize(2 * nv * nv);

  // cost norms
  norm_type_sensor.resize(nsensor);

  // TODO(taylor): method for xml to initial norm
  for (int i = 0; i < nsensor; i++) {
    norm_type_sensor[i] =
        (NormType)GetNumberOrDefault(0, model, "batch_norm_sensor");
  }

  // cost norm parameters
  norm_parameters_sensor.resize(nsensor * MAX_NORM_PARAMETERS);

  // TODO(taylor): initialize norm parameters from xml
  std::fill(norm_parameters_sensor.begin(), norm_parameters_sensor.end(), 0.0);

  // norm
  norm_sensor_.resize(nsensor * MAX_HISTORY);
  norm_force_.resize(nv * MAX_HISTORY);

  // norm gradient
  norm_gradient_sensor_.resize(nsensordata_ * MAX_HISTORY);
  norm_gradient_force_.resize(nv * MAX_HISTORY);

  // norm Hessian
  norm_hessian_sensor_.resize((nsensordata_ * MAX_HISTORY) *
                              (nsensordata_ * MAX_HISTORY));
  norm_hessian_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  norm_blocks_sensor_.resize(nsensordata_ * nsensordata_ * MAX_HISTORY);
  norm_blocks_force_.resize(nv * nv * MAX_HISTORY);

  // scratch
  scratch0_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch1_prior_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  scratch0_sensor_.resize(mju_max(nv, nsensordata_) *
                          mju_max(nv, nsensordata_) * MAX_HISTORY);
  scratch1_sensor_.resize(mju_max(nv, nsensordata_) *
                          mju_max(nv, nsensordata_) * MAX_HISTORY);

  scratch0_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch1_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch2_force_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  scratch_expected_.resize(nv * MAX_HISTORY);

  // copy
  configuration_copy_.Initialize(nq, configuration_length_);

  // search direction
  search_direction_.resize(nv * MAX_HISTORY);

  // covariance
  prior_matrix_factor_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch0_covariance_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));
  scratch1_covariance_.resize((nv * MAX_HISTORY) * (nv * MAX_HISTORY));

  // regularization
  regularization_ = settings.regularization_initial;

  // search type
  settings.search_type = (SearchType)GetNumberOrDefault(
      (int)settings.search_type, model, "batch_search_type");

  // timer
  timer_.prior_step.resize(MAX_HISTORY);
  timer_.sensor_step.resize(MAX_HISTORY);
  timer_.force_step.resize(MAX_HISTORY);

  // status
  hessian_factor_ = false;
  num_new_ = configuration_length_;
  gradient_norm_ = 0.0;
  search_direction_norm_ = 0.0;
  solve_status_ = kUnsolved;

  // settings
  settings.band_prior =
      (bool)GetNumberOrDefault(1, model, "batch_band_covariance");
}

// reset memory
void Batch::Reset() {
  // dimension
  int nq = model->nq, nv = model->nv, na = model->na;

  // data
  mjData* d = data_[0].get();

  // set home keyframe
  int home_id = mj_name2id(model, mjOBJ_KEY, "home");
  if (home_id >= 0) mj_resetDataKeyframe(model, d, home_id);

  // state
  mju_copy(state.data(), d->qpos, nq);
  mju_copy(state.data() + nq, d->qvel, nv);
  mju_copy(state.data() + nq + nv, d->act, na);
  d->time = 0.0;
  time = 0.0;

  // covariance
  mju_eye(covariance.data(), ndstate_);
  double covariance_scl =
      GetNumberOrDefault(1.0, model, "estimator_covariance_initial_scale");
  mju_scl(covariance.data(), covariance.data(), covariance_scl,
          ndstate_ * ndstate_);

  // process noise
  double noise_process_scl =
      GetNumberOrDefault(1.0e4, model, "estimator_process_noise_scale");
  std::fill(noise_process.begin(), noise_process.end(), noise_process_scl);

  // sensor noise
  double noise_sensor_scl =
      GetNumberOrDefault(1.0e4, model, "estimator_sensor_noise_scale");
  std::fill(noise_sensor.begin(), noise_sensor.end(), noise_sensor_scl);

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
  for (int i = 0; i < nsensor * configuration_length_; i++) {
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
  cost_prior = 0.0;
  cost_sensor = 0.0;
  cost_force = 0.0;
  cost = 0.0;
  cost_initial = 0.0;
  cost_previous = 1.0e32;

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
  std::fill(cost_hessian_band_factor_.begin(), cost_hessian_band_factor_.end(),
            0.0);
  std::fill(cost_hessian_factor_.begin(), cost_hessian_factor_.end(), 0.0);

  // weight
  std::fill(weight_prior.begin(), weight_prior.end(), 0.0);
  std::fill(weight_prior_band_.begin(), weight_prior_band_.end(), 0.0);
  std::fill(scratch_prior_weight_.begin(), scratch_prior_weight_.end(), 0.0);

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
  std::fill(scratch0_prior_.begin(), scratch0_prior_.end(), 0.0);
  std::fill(scratch1_prior_.begin(), scratch1_prior_.end(), 0.0);

  std::fill(scratch0_sensor_.begin(), scratch0_sensor_.end(), 0.0);
  std::fill(scratch1_sensor_.begin(), scratch1_sensor_.end(), 0.0);

  std::fill(scratch0_force_.begin(), scratch0_force_.end(), 0.0);
  std::fill(scratch1_force_.begin(), scratch1_force_.end(), 0.0);
  std::fill(scratch2_force_.begin(), scratch2_force_.end(), 0.0);

  std::fill(scratch_expected_.begin(), scratch_expected_.end(), 0.0);

  // candidate
  configuration_copy_.Reset();

  // search direction
  std::fill(search_direction_.begin(), search_direction_.end(), 0.0);

  // covariance
  std::fill(prior_matrix_factor_.begin(), prior_matrix_factor_.end(), 0.0);
  std::fill(scratch0_covariance_.begin(), scratch0_covariance_.end(), 0.0);
  std::fill(scratch1_covariance_.begin(), scratch1_covariance_.end(), 0.0);

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
  initialized_ = false;
  solve_status_ = kUnsolved;

  // -- initialize -- //

  settings.gradient_tolerance = 1.0e-6;
  settings.max_smoother_iterations = 10;
  settings.max_search_iterations = 1000;
  settings.force_residual_timestep_scale = false;
  settings.prior_flag = true;
  settings.sensor_flag = true;
  settings.force_flag = true;

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
  double current_time = -1.0 * timestep;
  times.Set(&current_time, 0);
  for (int i = 1; i < configuration_length_; i++) {
    // increment time
    current_time += timestep;

    // set 
    times.Set(&current_time, i);
  }

  // prior weight 
  for (int i = 0; i < ndstate_; i++) {
    weight_prior[nq * configuration_length_ * i + i] = 1.0 / covariance[ndstate_ * i + i];
  }
}

// update
void Batch::Update(const double* ctrl, const double* sensor) {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimensions 
  int nq = model->nq, nv = model->nv, nu = model->nu;

  // current time index 
  int t = prediction_length_;

  // set ctrl 
  this->ctrl.Set(ctrl, t);

  // set sensor 
  sensor_measurement.Set(sensor + sensor_start_index_, t);

  // -- next qpos -- //

  // data
  mjData* d = data_[0].get();

  // set state 
  double* q0 = configuration.Get(t - 1);
  double* q1 = configuration.Get(t);
  mju_copy(q1, state.data(), nq);
  mju_copy(q0, q1, nq);
  mj_integratePos(model, q0, state.data() + nq, -model->opt.timestep);

  mju_copy(d->qpos, q1, model->nq);
  mj_differentiatePos(model, d->qvel, model->opt.timestep, q0, q1);
  // TODO(taylor): set act
  // d->time = times.Get(t)[0]; // TODO(taylor): time as input?

  // set ctrl 
  mju_copy(d->ctrl, ctrl, nu);

  // set state 
  // mju_copy(d->qpos, state.data(), nq);
  // mju_copy(d->qvel, state.data() + nq, nv);

  // copy configuration
  // mju_copy(q1, d->qpos, nq);
  // mju_copy(q0, q1, nq);
  // mj_integratePos(model, q0, d->qvel, -1.0 * model->opt.timestep);

  // configuration_previous.Set(q0, t - 1);
  // configuration_previous.Set(q1, t);

  // forward step 
  mj_step(model, d);

  // set next qpos 
  configuration.Set(d->qpos, t + 1);
  configuration_previous.Set(d->qpos, t + 1);

  // TODO(taylor): set act

  // set next time
  times.Set(&d->time, t + 1);

  // set force measurement 
  force_measurement.Set(d->qfrc_actuator, t);

  // optimize (measurement update)
  // ThreadPool pool(1);
  // Optimize(pool);

  // update state
  // mju_copy(state.data(), configuration.Get(t + 1), nq);
  // mju_copy(state.data() + nq, velocity.Get(t + 1), nv);
  // TODO(taylor): act 
  // time = times.Get(t + 1)[0];

  // mju_copy(state.data(), d->qpos, nq);
  // mju_copy(state.data() + nq, d->qvel, nv);
  // time = d->time;

  // -- update prior weight -- //
  // std::vector<double> res((2 * nq) * (2 * nq)); 
  // std::vector<double> mat00(nq * nq);
  // std::vector<double> mat10((2 * nq) * nq);
  // std::vector<double> mat11((2 * nq) * (2 * nq));
  // std::vector<double> tmp0(nq * (2 * nq));
  // std::vector<double> tmp1((2 * nq) * (2 * nq));

  // // condition cost Hessian
  // ConditionMatrix(res.data(), cost_hessian.data(), mat00.data(), mat10.data(),
  //                 mat11.data(), tmp0.data(), tmp1.data(),
  //                 nq * configuration_length_, nq, 2 * nq);

  // // set conditioned block in prior weight matrix
  // SetBlockInMatrix(weight_prior.data(), res.data(), 1.0,
  //                  nq * configuration_length_, nq * configuration_length_,
  //                  nq * 2, nq * 2, 0, 0);

  ThreadPool pool(1);
  Optimize(pool);

  mju_copy(state.data(), configuration.Get(t + 1), nq);
  mju_copy(state.data() + nq, velocity.Get(t + 1), nv);
  time = d->time;

  // printf("cost: (%f) -> (%f)\n", cost_initial, cost);

  // printf("  prior: (%f)\n", cost_prior);
  // printf("  sensor: (%f)\n", cost_sensor);
  // printf("  force: (%f)\n", cost_force);

  // printf("qpos (sim) = \n");
  // mju_printMat(state.data(), 1, nq);
  // printf("qpos (batch) = \n");
  // mju_printMat(configuration.Get(t + 1), 1, nq);

  // printf("qvel (sim) = \n");
  // mju_printMat(state.data() + nq, 1, nv);
  // printf("qvel (batch) = \n");
  // mju_printMat(velocity.Get(t + 1), 1, nv);

  // printf("residual (force) = \n");
  // mju_printMat(residual_force_.data(), 1, nv);

  // std::vector<double> vel(nv);
  // mj_differentiatePos(model, vel.data(), model->opt.timestep, configuration.Get(t), configuration.Get(t+1));
  // printf("qvel (fd) = \n");
  // mju_printMat(vel.data(), 1, nv);

  // shift trajectories
  Shift(1);

  // stop timer
  timer_.update = 1.0e-3 * GetDuration(start);
}

// set configuration length
void Batch::SetConfigurationLength(int length) {
  // initialize 
  // TODO(taylor): remove 
  Initialize(model);
  Reset();

  // set configuration length
  configuration_length_ = mju_max(length, MIN_HISTORY);

  // set prediction length
  prediction_length_ = configuration_length_ - 2;

  // update trajectory lengths
  configuration.SetLength(length);
  configuration_copy_.SetLength(length);

  velocity.SetLength(length);
  acceleration.SetLength(length);
  act.SetLength(length);
  times.SetLength(length);

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
  block_sensor_configurationT_.SetLength(prediction_length_);
  block_sensor_velocityT_.SetLength(prediction_length_);
  block_sensor_accelerationT_.SetLength(prediction_length_);

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
  search_direction_norm_ = 0.0;
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

  block_prior_current_configuration_.Shift(shift);

  block_sensor_configurationT_.Shift(shift);
  block_sensor_velocityT_.Shift(shift);
  block_sensor_accelerationT_.Shift(shift);

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
  int nsen = nsensordata_ * prediction_length_;
  int nforce = nsensordata_ * prediction_length_;

  // operations
  int opprior = settings.prior_flag * configuration_length_;
  int opsensor = settings.sensor_flag * prediction_length_;
  int opforce = settings.force_flag * prediction_length_;

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
  double* P =
      (settings.band_prior ? weight_prior_band_.data() : weight_prior.data());
  double* tmp = scratch0_prior_.data();

  // initial cost
  double cost = cost_prior;

  // compute cost
  if (!cost_skip_) {
    // residual
    ResidualPrior();

    if (settings.band_prior) {  // approximate covariance
      // dimensions
      int ntotal = dim;
      int nband = 3 * model->nv;
      int ndense = 0;

      // multiply: tmp = P * r
      mju_bandMulMatVec(tmp, P, r, ntotal, nband, ndense, 1, true);
    } else {  // exact covariance
      // multiply: tmp = P * r
      mju_mulMatVec(tmp, P, r, dim, dim);
    }

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
    if (hessian && settings.band_prior) {
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
        BlockFromMatrix(bbij, weight_prior.data(), nv, nv, dim, dim, t * nv,
                        j * nv);
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
  if (hessian && !settings.band_prior) {
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
  // start index
  int start_index =
      settings.reuse_data * mju_max(0, configuration_length_ - num_new_);

  // loop over predictions
  for (int t = 0; t < configuration_length_; t++) {
    // schedule by time step
    pool.Schedule([&batch = *this, start_index, t]() {
      // start Jacobian timer
      auto jacobian_prior_start = std::chrono::steady_clock::now();

      // block
      if (t >= start_index) batch.BlockPrior(t);

      // // assemble
      // if (batch.settings.assemble_prior_jacobian) {
      //   batch.SetBlockPrior(t);
      // }

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
  int nsen = ns * prediction_length_;

  // residual
  if (!cost_skip_) ResidualSensor();

  // ----- cost ----- //

  // initialize
  double cost = 0.0;

  // zero memory
  if (gradient) mju_zero(gradient, nvar);
  if (hessian) mju_zero(hessian, nvar * nvar);

  // matrix shift
  int shift_matrix = 0;

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // residual
    double* rk = residual_sensor_.data() + ns * k;

    // mask
    // int* mask = sensor_mask.Get(t);

    // unpack block
    double* block = block_sensor_configurations_.Get(k);

    // shift
    int shift_sensor = 0;

    // loop over sensors
    for (int i = 0; i < nsensor; i++) {
      // start cost timer
      auto start_cost = std::chrono::steady_clock::now();

      // check mask, skip if missing measurement
      // if (!mask[i]) continue;

      // dimension
      int nsi = model->sensor_dim[sensor_start + i];

      // sensor residual
      double* rki = rk + shift_sensor;

      // weight
      double weight = 1.0 / noise_sensor[i] / nsi / prediction_length_;

      // parameters
      double* pi = norm_parameters_sensor.data() + MAX_NORM_PARAMETERS * i;

      // norm
      NormType normi = norm_type_sensor[i];

      // norm gradient
      double* norm_gradient =
          norm_gradient_sensor_.data() + ns * k + shift_sensor;

      // norm Hessian
      double* norm_block = norm_blocks_sensor_.data() + shift_matrix;

      // ----- cost ----- //

      // norm
      norm_sensor_[nsensor * k + i] =
          Norm(gradient ? norm_gradient : NULL, hessian ? norm_block : NULL,
               rki, pi, nsi, normi);

      // weighted norm
      cost += weight * norm_sensor_[nsensor * k + i];

      // stop cost timer
      timer_.cost_sensor += GetDuration(start_cost);

      // assemble dense norm Hessian
      if (settings.assemble_sensor_norm_hessian) {
        // reset memory
        if (i == 0 && k == 0)
          mju_zero(norm_hessian_sensor_.data(), nsen * nsen);

        // set norm block
        SetBlockInMatrix(norm_hessian_sensor_.data(), norm_block, weight,
                         nsen, nsen, nsi, nsi, ns * k + shift_sensor,
                         ns * k + shift_sensor);
      }

      // gradient wrt configuration: dridq012' * dndri
      if (gradient) {
        // sensor block
        double* blocki = block + (3 * nv) * shift_sensor;

        // scratch = dridq012' * dndri
        mju_mulMatTVec(scratch0_sensor_.data(), blocki, norm_gradient, nsi,
                       3 * nv);

        // add
        mju_addToScl(gradient + k * nv, scratch0_sensor_.data(), weight,
                     3 * nv);
      }

      // Hessian (Gauss-Newton): drdq' * d2ndr2 * drdq
      if (hessian) {
        // sensor block
        double* blocki = block + (3 * nv) * shift_sensor;

        // step 1: tmp0 = d2ndri2 * dridq
        double* tmp0 = scratch0_sensor_.data();
        mju_mulMatMat(tmp0, norm_block, blocki, nsi, nsi, 3 * nv);

        // step 2: hessian = dridq' * tmp
        double* tmp1 = scratch1_sensor_.data();
        mju_mulMatTMat(tmp1, blocki, tmp0, nsi, 3 * nv, 3 * nv);

        // add
        AddBlockInMatrix(hessian, tmp1, weight, nvar, nvar, 3 * nv, 3 * nv,
                         nv * k, nv * k);
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
  int nforce = nv * prediction_length_;

  // residual
  if (!cost_skip_) ResidualForce();

  // ----- cost ----- //

  // initialize
  double cost = 0.0;

  // zero memory
  if (gradient) mju_zero(gradient, nvar);
  if (hessian) mju_zero(hessian, nvar * nvar);

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // unpack block
    double* block = block_force_configurations_.Get(k);

    // start cost timer
    auto start_cost = std::chrono::steady_clock::now();

    // residual
    double* rk = residual_force_.data() + k * nv;

    // norm gradient
    double* norm_gradient = norm_gradient_force_.data() + k * nv;

    // norm block
    double* norm_block = norm_blocks_force_.data() + k * nv * nv;
    mju_zero(norm_block, nv * nv);

    // ----- cost ----- //

    // quadratic cost
    for (int i = 0; i < nv; i++) {
      // weight 
      double weight = 1.0 / noise_process[i] / nv / prediction_length_;

      // gradient 
      norm_gradient[i] = weight * rk[i];

      // Hessian 
      norm_block[nv * i + i] = weight;
    }

    // norm
    norm_sensor_[k] = 0.5 * mju_dot(rk, norm_gradient, nv);

    // weighted norm
    cost += norm_sensor_[k];

    // stop cost timer
    timer_.cost_force += GetDuration(start_cost);

    // assemble dense norm Hessian
    if (settings.assemble_force_norm_hessian) {
      // zero memory
      if (k == 0) mju_zero(norm_hessian_force_.data(), nforce * nforce);

      // set block
      SetBlockInMatrix(norm_hessian_force_.data(), norm_block, 1.0, nforce,
                       nforce, nv, nv, k * nv, k * nv);
    }

    // gradient wrt configuration: dridq012' * dndri
    if (gradient) {
      // scratch = dridq012' * dndri
      mju_mulMatTVec(scratch0_force_.data(), block, norm_gradient, nv, 3 * nv);

      // add
      mju_addToScl(gradient + k * nv, scratch0_force_.data(), 1.0, 3 * nv);
    }

    // Hessian (Gauss-Newton): drdq' * d2ndr2 * drdq
    if (hessian) {
      // step 1: tmp0 = d2ndri2 * dridq
      double* tmp0 = scratch0_force_.data();
      mju_mulMatMat(tmp0, norm_block, block, nv, nv, 3 * nv);

      // step 2: hessian = dridq' * tmp
      double* tmp1 = scratch1_force_.data();
      mju_mulMatTMat(tmp1, block, tmp0, nv, 3 * nv, 3 * nv);

      // add
      AddBlockInMatrix(hessian, tmp1, 1.0, nvar, nvar, 3 * nv, 3 * nv,
                       nv * k, nv * k);
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
  for (int k = 0; k < prediction_length_; k++) {
    // time index
    int t = k + 1;

    // terms
    double* rk = residual_sensor_.data() + k * nsensordata_;
    double* yt_sensor = sensor_measurement.Get(t);
    double* yt_model = sensor_prediction.Get(t);

    // sensor difference
    mju_sub(rk, yt_model, yt_sensor, nsensordata_);
  }

  // stop timer
  timer_.residual_sensor += GetDuration(start);
}

// sensor Jacobian blocks (dsdq0, dsdq1, dsdq2)
void Batch::BlockSensor(int index) {
  // dimensions
  int nv = model->nv, ns = nsensordata_;

  // shift 
  int shift = sensor_start_index_ * nv;

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
    // dimension
    int nv = model->nv;
    int nvar = nv * configuration_length_;
    int nsen = nsensordata_ * prediction_length_;

    // set block
    SetBlockInMatrix(jacobian_sensor_.data(), dsdq012, 1.0, nsen, nvar,
                     nsensordata_, 3 * nv, index * nsensordata_, index * nv);
  }
}

// sensor Jacobian
// note: pool wait is called outside this function
void Batch::JacobianSensor(ThreadPool& pool) {
  // start index
  int start_index =
      settings.reuse_data * mju_max(0, prediction_length_ - num_new_);

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // schedule by time step
    pool.Schedule([&batch = *this, start_index, k]() {
      // start Jacobian timer
      auto jacobian_sensor_start = std::chrono::steady_clock::now();

      // block
      if (k >= start_index) batch.BlockSensor(k);

      // stop Jacobian timer
      batch.timer_.sensor_step[k] = GetDuration(jacobian_sensor_start);
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
  for (int k = 0; k < prediction_length_; k++) {
    // time index
    int t = k + 1;

    // terms
    double* rk = residual_force_.data() + k * nv;
    double* ft_actuator = force_measurement.Get(t);
    double* ft_inverse = force_prediction.Get(t);

    // force difference
    mju_sub(rk, ft_inverse, ft_actuator, nv);

    // scale force residual by timestep
    if (settings.force_residual_timestep_scale) {
      mju_scl(rk, rk, model->opt.timestep, nv);
    }
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

  // scale
  if (settings.force_residual_timestep_scale) {
    mju_scl(dfdq0, dfdq0, model->opt.timestep, nv * nv);
  }

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

  // scale
  if (settings.force_residual_timestep_scale) {
    mju_scl(dfdq1, dfdq1, model->opt.timestep, nv * nv);
  }

  // -- configuration next: dfdq2 = dfda * dadq2 -- //

  // unpack
  double* dfdq2 = block_force_next_configuration_.Get(index);

  // dfdq2 = dadf' * dadq2
  double* dadq2 = block_acceleration_next_configuration_.Get(index);
  mju_mulMatTMat(dfdq2, dadf, dadq2, nv, nv, nv);

  // scale
  if (settings.force_residual_timestep_scale) {
    mju_scl(dfdq2, dfdq2, model->opt.timestep, nv * nv);
  }

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
    int nforce = nv * prediction_length_;

    // set block
    SetBlockInMatrix(jacobian_force_.data(), dfdq012, 1.0, nforce, nvar, nv,
                     3 * nv, index * nv, index * nv);
  }
}

// force Jacobian
// note: pool wait is called outside this function
void Batch::JacobianForce(ThreadPool& pool) {
  // start index
  int start_index =
      settings.reuse_data * mju_max(0, prediction_length_ - num_new_);

  // loop over predictions
  for (int k = 0; k < prediction_length_; k++) {
    // schedule by time step
    pool.Schedule([&batch = *this, start_index, k]() {
      // start Jacobian timer
      auto jacobian_force_start = std::chrono::steady_clock::now();

      // block
      if (k >= start_index) batch.BlockForce(k);

      // stop Jacobian timer
      batch.timer_.force_step[k] = GetDuration(jacobian_force_start);
    });
  }
}

// compute force
// TODO(taylor): combine with Jacobian method
void Batch::InverseDynamicsPrediction(ThreadPool& pool) {
  // compute sensor and force predictions
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nq = model->nq, nv = model->nv, nu = model->nu, ns = nsensordata_;

  // start index
  int start_index =
      settings.reuse_data * mju_max(0, prediction_length_ - num_new_);

  // pool count
  int count_before = pool.GetCount();

  // loop over predictions
  for (int k = start_index; k < prediction_length_; k++) {
    // schedule
    pool.Schedule([&batch = *this, nq, nv, ns, nu, k]() {
      // time index
      int t = k + 1;

      // terms
      double* qt = batch.configuration.Get(t);
      double* vt = batch.velocity.Get(t);
      double* at = batch.acceleration.Get(t);
      double* ct = batch.ctrl.Get(t);

      // data
      mjData* d = batch.data_[k].get();

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
    });
  }

  // wait
  pool.WaitCount(count_before + (prediction_length_ - start_index));
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

  // start index
  int start_index =
      settings.reuse_data * mju_max(0, prediction_length_ - num_new_);

  // loop over predictions
  for (int k = start_index; k < prediction_length_; k++) {
    // schedule
    pool.Schedule([&batch = *this, nq, nv, nu, k]() {
      // time index
      int t = k + 1;

      // unpack
      double* q = batch.configuration.Get(t);
      double* v = batch.velocity.Get(t);
      double* a = batch.acceleration.Get(t);
      double* c = batch.ctrl.Get(t);

      double* dsdq = batch.block_sensor_configuration_.Get(k);
      double* dsdv = batch.block_sensor_velocity_.Get(k);
      double* dsda = batch.block_sensor_acceleration_.Get(k);
      double* dqds = batch.block_sensor_configurationT_.Get(k);
      double* dvds = batch.block_sensor_velocityT_.Get(k);
      double* dads = batch.block_sensor_accelerationT_.Get(k);
      double* dqdf = batch.block_force_configuration_.Get(k);
      double* dvdf = batch.block_force_velocity_.Get(k);
      double* dadf = batch.block_force_acceleration_.Get(k);
      mjData* data = batch.data_[k].get();  // TODO(taylor): WorkerID

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
  pool.WaitCount(count_before + (prediction_length_ - start_index));

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

  // stop timer
  timer_.configuration_update += GetDuration(start);
}

// convert sequence of configurations to velocities and accelerations
void Batch::ConfigurationToVelocityAcceleration() {
  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model->nv;

  // start index
  int start_index =
      settings.reuse_data * mju_max(0, (configuration_length_ - 1) - num_new_);

  // loop over configurations
  for (int k = start_index; k < configuration_length_ - 1; k++) {
    // time index
    int t = k + 1;

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

  // start index
  int start_index =
      settings.reuse_data * mju_max(0, (configuration_length_ - 1) - num_new_);

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
    DifferentiateDifferentiatePos(dv2dq1, dv2dq2, model, model->opt.timestep,
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
// TODO(taylor): fix timers
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

  // individual derivatives
  if (settings.prior_flag) {
    pool.Schedule([&batch = *this, gradient_flag, hessian_flag]() {
      batch.cost_prior = batch.CostPrior(
          gradient_flag ? batch.cost_gradient_prior_.data() : NULL,
          hessian_flag ? batch.cost_hessian_prior_.data() : NULL);
    });
  }
  if (settings.sensor_flag) {
    pool.Schedule([&batch = *this, gradient_flag, hessian_flag]() {
      batch.cost_sensor = batch.CostSensor(
          gradient_flag ? batch.cost_gradient_sensor_.data() : NULL,
          hessian_flag ? batch.cost_hessian_sensor_.data() : NULL);
    });
  }
  if (settings.force_flag) {
    pool.Schedule([&batch = *this, gradient_flag, hessian_flag]() {
      batch.cost_force = batch.CostForce(
          gradient_flag ? batch.cost_gradient_force_.data() : NULL,
          hessian_flag ? batch.cost_hessian_force_.data() : NULL);
    });
  }

  // wait
  pool.WaitCount(count_begin + settings.prior_flag + settings.sensor_flag +
                 settings.force_flag);
  pool.ResetCount();

  // total cost
  double cost = cost_prior + cost_sensor + cost_force;

  // total gradient, hessian
  if (gradient) TotalGradient();
  if (hessian) TotalHessian();

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
void Batch::TotalGradient() {
  // start gradient timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int dim = configuration_length_ * model->nv;

  // unpack
  double* gradient = cost_gradient.data();

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
void Batch::TotalHessian() {
  // start Hessian timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int dim = configuration_length_ * model->nv;

  // unpack
  double* hessian = cost_hessian.data();

  if (settings.band_copy) {
    // zero memory
    mju_zero(hessian, dim * dim);

    // individual Hessians
    if (settings.prior_flag)
      SymmetricBandMatrixCopy(hessian, cost_hessian_prior_.data(), model->nv, 3,
                              dim, configuration_length_, 0, 0, 0, 0,
                              scratch0_prior_.data());
    if (settings.sensor_flag)
      SymmetricBandMatrixCopy(hessian, cost_hessian_sensor_.data(), model->nv,
                              3, dim, configuration_length_, 0, 0, 0, 0,
                              scratch0_sensor_.data());
    if (settings.force_flag)
      SymmetricBandMatrixCopy(hessian, cost_hessian_force_.data(), model->nv, 3,
                              dim, configuration_length_, 0, 0, 0, 0,
                              scratch0_force_.data());
  } else {
    // individual Hessians
    if (settings.prior_flag) {
      mju_copy(hessian, cost_hessian_prior_.data(), dim * dim);
    } else {
      mju_zero(hessian, dim * dim);
    }
    if (settings.sensor_flag)
      mju_addTo(hessian, cost_hessian_sensor_.data(), dim * dim);
    if (settings.force_flag)
      mju_addTo(hessian, cost_hessian_force_.data(), dim * dim);
  }

  // stop Hessian timer
  timer_.cost_hessian += GetDuration(start);
}

// covariance update
void Batch::PriorWeightUpdate(ThreadPool& pool) {
  // // skip
  // if (settings.skip_update_prior_weight) return;

  // start timer
  auto start = std::chrono::steady_clock::now();

  // dimension
  int nv = model->nv;
  int ntotal = nv * configuration_length_;

  // ----- update prior weights ----- //
  // start timer
  auto start_set_weight = std::chrono::steady_clock::now();

  // weight
  double* weight = weight_prior.data();
  double* weight_band = weight_prior_band_.data();

  // Hessian
  // double* hessian = cost_hessian.data();

  // zero memory
  mju_zero(weight, ntotal * ntotal);

  // copy Hessian block to upper left
  // if (configuration_length_ - num_new_ > 0 &&
  // settings.update_prior_weight) {
  //   SymmetricBandMatrixCopy(weight, hessian, nv, nv, ntotal,
  //                           configuration_length_ - num_new_, 0, 0,
  // num_new_,
  //                           num_new_, scratch_prior_weight_.data());
  // }

  // set s * I to lower right
  for (int i = 0; i < ntotal; i++) {
    weight[ntotal * i + i] = scale_prior;
  }

  // dense to band
  if (settings.band_prior) {
    mju_dense2Band(weight_band, weight, ntotal, 3 * nv, 0);
  }

  // stop timer
  timer_.prior_set_weight += GetDuration(start_set_weight);

  // stop timer
  timer_.prior_weight_update += GetDuration(start);

  // status
  PrintPriorWeightUpdate();
}

// optimize trajectory estimate
void Batch::Optimize(ThreadPool& pool) {
  // start timer
  auto start_optimize = std::chrono::steady_clock::now();

  // set status
  solve_status_ = kUnsolved;

  // dimensions
  int nconfig = model->nq * configuration_length_;
  int nvar = model->nv * configuration_length_;

  // reset timers
  ResetTimers();

  // prior update
  // PriorWeightUpdate(pool);

  // initial cost
  cost_count_ = 0;
  cost = Cost(NULL, NULL, pool);
  cost_initial = cost;

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
    Cost(cost_gradient.data(), cost_hessian.data(), pool);

    // start timer
    auto start_search = std::chrono::steady_clock::now();

    // reset num_new_
    num_new_ = configuration_length_;  // update all data now

    // -- gradient -- //
    double* gradient = cost_gradient.data();

    // gradient tolerance check
    gradient_norm_ = mju_norm(gradient, nvar) / nvar;
    if (gradient_norm_ < settings.gradient_tolerance) {
      break;
    }

    // ----- line / curve search ----- //

    // copy configuration
    mju_copy(configuration_copy_.Data(), configuration.Data(), nconfig);

    // initialize
    double cost_candidate = cost;
    int iteration_search = 0;
    step_size_ = 1.0;
    regularization_ = settings.regularization_initial;
    improvement_ = -1.0;

    // -- search direction -- //

    // check regularization
    if (regularization_ >= MAX_REGULARIZATION - 1.0e-6) {
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
    while (cost_candidate >= cost) {
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
            regularization_ =
                mju_min(MAX_REGULARIZATION,
                        regularization_ * settings.regularization_scaling);

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
      improvement_ = cost - cost_candidate;

      // update iteration
      iteration_search++;
    }

    // increment
    iterations_search_ += iteration_search;

    // update cost
    cost_previous = cost;
    cost = cost_candidate;

    // check cost difference
    cost_difference_ = std::abs(cost - cost_previous);
    if (cost_difference_ < settings.cost_tolerance) {
      // set status
      solve_status_ = kCostDifferenceFailure;

      // failure
      return;
    }

    // curve search
    if (settings.search_type == kCurveSearch) {
      // expected = g' d + 0.5 d' H d

      // expected = g' * d
      expected_ = mju_dot(cost_gradient.data(), search_direction_.data(), nvar);

      // tmp = H * d
      double* tmp = scratch_expected_.data();
      if (settings.band_prior) {
        mju_bandMulMatVec(tmp, cost_hessian_band_.data(),
                          search_direction_.data(), nvar, 3 * model->nv, 0, 1,
                          true);
      } else {
        mju_mulMatVec(tmp, cost_hessian.data(), search_direction_.data(), nvar,
                      nvar);
      }

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
            mju_max(MIN_REGULARIZATION,
                    regularization_ / settings.regularization_scaling);
      } else if (reduction_ratio_ < 0.25) {
        // increase
        regularization_ =
            mju_min(MAX_REGULARIZATION,
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

// regularize Hessian
void Batch::Regularize() {
  // dimension
  int nvar = configuration_length_ * model->nv;

  // H + reg * I
  for (int j = 0; j < nvar; j++) {
    cost_hessian[j * nvar + j] += regularization_;
  }
}

// search direction
void Batch::SearchDirection() {
  // start timer
  auto search_direction_start = std::chrono::steady_clock::now();

  // dimensions
  int ntotal = configuration_length_ * model->nv;
  int nband = 3 * model->nv;
  int ndense = 0;

  // regularize
  Regularize();

  // -- band Hessian -- //

  // unpack
  double* direction = search_direction_.data();
  double* gradient = cost_gradient.data();
  double* hessian = cost_hessian.data();
  double* hessian_band = cost_hessian_band_.data();
  double* hessian_band_factor = cost_hessian_band_factor_.data();

  // -- linear system solver -- //

  // select solver
  if (settings.band_prior) {  // band solver
    // dense to band
    mju_dense2Band(hessian_band, cost_hessian.data(), ntotal, nband, ndense);

    // copy
    // TODO(taylor): efficient copy
    mju_copy(hessian_band_factor, hessian_band, ntotal * ntotal);

    // factorize
    mju_cholFactorBand(hessian_band_factor, ntotal, nband, ndense, 0.0, 0.0);

    // compute search direction
    mju_cholSolveBand(direction, hessian_band_factor, gradient, ntotal, nband,
                      ndense);
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

  // search direction norm
  search_direction_norm_ = InfinityNorm(direction, ntotal);

  // end timer
  timer_.search_direction += GetDuration(search_direction_start);
}

// // covariance
// void Batch::Covariance(ThreadPool& pool) {
//   // dimension
//   int nvar = model->nv * configuration_length_;
//   int nv3 = model->nv * 3;

//   // identity
//   double* I = scratch0_covariance_.data();
//   mju_eye(I, nvar);

//   // factorize
//   double* factor = prior_matrix_factor_.data();
//   if (settings.band_prior) {
//     // factorize prior matrix
//     mju_dense2Band(factor, weight_prior.data(), nvar, nv3, 0);
//     mju_cholFactorBand(factor, nvar, nv3, 0, 0.0, 0.0);
//   } else {
//     // factorize prior matrix
//     mju_copy(factor, weight_prior.data(), nvar * nvar);
//     mju_cholFactor(factor, nvar, 0.0);
//   }

//   // -- covariance = L \ I --//

//   // get initial pool count
//   int count_begin = pool.GetCount();

//   // compute covariance
//   for (int i = 0; i < nvar; i++) {
//     pool.Schedule([&batch = *this, nvar, nv3, i]() {
//       if (batch.settings.band_prior) {
//         mju_cholSolveBand(batch.covariance_.data() + nvar * i,
//                           batch.prior_matrix_factor_.data(),
//                           batch.scratch0_covariance_.data() + nvar * i, nvar,
//                           nv3, 0);
//       } else {
//         mju_cholSolve(batch.covariance_.data() + nvar * i,
//                       batch.prior_matrix_factor_.data(),
//                       batch.scratch0_covariance_.data() + nvar * i, nvar);
//       }
//     });
//   }

//   // wait
//   pool.WaitCount(count_begin + nvar);
//   pool.ResetCount();
// }

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

  PrintPriorWeightUpdate();

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
  printf("  final: %.3f\n", cost);
  printf("    - prior: %.3f\n", cost_prior);
  printf("    - sensor: %.3f\n", cost_sensor);
  printf("    - force: %.3f\n", cost_force);
  printf("  <initial: %.3f>\n", cost_initial);
  printf("\n");

  fflush(stdout);
}

// print cost
void Batch::PrintCost() {
  if (settings.verbose_cost) {
    printf("cost (total): %.3f\n", cost);
    printf("  prior: %.3f\n", cost_prior);
    printf("  sensor: %.3f\n", cost_sensor);
    printf("  force: %.3f\n", cost_force);
    printf("  [initial: %.3f]\n", cost_initial);
    fflush(stdout);
  }
}

// print prior weight update status
// print Optimize status
void Batch::PrintPriorWeightUpdate() {
  if (!settings.verbose_prior) return;

  // timing
  printf("  prior weight update [total]: %.3f (ms) \n",
         1.0e-3 * timer_.prior_weight_update);
  printf("    - set weight: %.3f (ms) \n", 1.0e-3 * timer_.prior_set_weight);
  printf("\n");
  fflush(stdout);
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

// // initialize trajectories
// void Batch::InitializeTrajectories(
//     const EstimatorTrajectory<double>& measurement,
//     const EstimatorTrajectory<int>& measurement_mask,
//     const EstimatorTrajectory<double>& ctrl,
//     const EstimatorTrajectory<double>& time) {
//   // start timer
//   auto start = std::chrono::steady_clock::now();

//   // set num new
//   num_new_ = configuration_length_;

//   // -- set initial configurations -- //

//   // set first configuration
//   double* q0 = configuration.Get(0);
//   mju_copy(q0, qpos0.data(), model->nq);
//   mj_integratePos(model, q0, qvel0.data(), -1.0 * model->opt.timestep);

//   // set second configuration
//   configuration.Set(qpos0.data(), 1);

//   // set initial time
//   this->times.Set(times.Get(0), 0);

//   // data
//   mjData* data = data_[0].get();

//   // set state
//   mju_copy(data->qpos, qpos0.data(), model->nq);
//   mju_copy(data->qvel, qvel0.data(), model->nv);
//   data->time = times.Get(1)[0];

//   // set new measurements, ctrl -> qfrc_actuator, rollout new configurations,
//   // new time
//   for (int i = 1; i < configuration_length_ - 1; i++) {
//     // buffer index
//     int buffer_index = times.Length() - (configuration_length_ - 1) + i;

//     // get time
//     this->times.Set(&data->time, i);

//     // set/get ctrl
//     const double* ui = ctrl.Get(buffer_index);
//     this->ctrl.Set(ui, i);
//     mju_copy(data->ctrl, ui, model->nu);

//     // step dynamics
//     mj_step(model, data);

//     // set measurement
//     const double* yi = measurement.Get(buffer_index);
//     sensor_measurement.Set(yi, i);

//     // set mask
//     const int* mi = measurement_mask.Get(buffer_index);
//     sensor_mask.Set(mi, i);

//     // copy qfrc_actuator
//     force_measurement.Set(data->qfrc_actuator, i);

//     // copy configuration
//     configuration.Set(data->qpos, i + 1);
//   }

//   // set last time
//   this->times.Set(&data->time, configuration_length_ - 1);

//   // copy configuration to prior
//   mju_copy(configuration_previous.Data(), configuration.Data(),
//            model->nq * configuration_length_);

//   // stop timer
//   timer_.update_trajectory += GetDuration(start);
// }

// // update trajectories
// int Batch::UpdateTrajectories_(int num_new,
//                                const EstimatorTrajectory<double>&
//                                measurement, const EstimatorTrajectory<int>&
//                                measurement_mask, const
//                                EstimatorTrajectory<double>& ctrl, const
//                                EstimatorTrajectory<double>& time) {
//   // start timer
//   auto start = std::chrono::steady_clock::now();

//   // set number of new elements
//   num_new_ = num_new;

//   // shift trajectory heads
//   Shift(num_new);

//   // get data
//   mjData* data = data_[0].get();

//   // set new measurements, ctrl -> qfrc_actuator, rollout new configurations,
//   // new time
//   for (int i = 0; i < num_new; i++) {
//     // time index
//     int t = i + configuration_length_ - num_new - 1;

//     // buffer index
//     int b = i + measurement.Length() - num_new;

//     // set measurement
//     const double* yi = measurement.Get(b);
//     sensor_measurement.Set(yi, t);

//     // set measurement mask
//     const int* mi = measurement_mask.Get(b);
//     sensor_mask.Set(mi, t);

//     // set time
//     const double* ti = times.Get(b);
//     this->times.Set(ti, t);

//     // ----- forward dynamics ----- //

//     // set ctrl
//     const double* ui = ctrl.Get(b);
//     this->ctrl.Set(ui, t);
//     mju_copy(data->ctrl, ui, model->nu);

//     // set qpos
//     double* q0 = configuration.Get(t - 1);
//     double* q1 = configuration.Get(t);
//     mju_copy(data->qpos, q1, model->nq);

//     // set qvel
//     mj_differentiatePos(model, data->qvel, model->opt.timestep, q0, q1);

//     // set time
//     data->time = times.Get(b)[0];

//     // step dynamics
//     mj_step(model, data);

//     // copy qfrc_actuator
//     force_measurement.Set(data->qfrc_actuator, t);

//     // copy configuration
//     configuration.Set(data->qpos, t + 1);
//   }

//   // set last time
//   this->times.Set(&data->time, configuration_length_ - 1);

//   // copy configuration to prior
//   mju_copy(configuration_previous.Data(), configuration.Data(),
//            model->nq * configuration_length_);

//   // stop timer
//   timer_.update_trajectory += GetDuration(start);

//   return num_new;
// }

// // update trajectories
// int Batch::UpdateTrajectories(const EstimatorTrajectory<double>& measurement,
//                               const EstimatorTrajectory<int>&
//                               measurement_mask, const
//                               EstimatorTrajectory<double>& ctrl, const
//                               EstimatorTrajectory<double>& time) {
//   // lastest buffer time
//   double time_buffer_last = *times.Get(times.Length() - 1);

//   // latest estimator time
//   double time_estimator_last =
//       *times.Get(times.Length() - 2);  // index to latest measurement time

//   // compute number of new elements
//   int num_new =
//       std::round(mju_max(0.0, time_buffer_last - time_estimator_last) /
//                  model->opt.timestep);

//   UpdateTrajectories_(num_new, measurement, measurement_mask, ctrl, time);

//   return num_new;
// }

// // update
// // int Batch::Update(const Buffer& buffer, ThreadPool& pool) {
// //   int num_new = 0;
// //   if (buffer.Length() >= configuration_length_ - 1) {
// //     num_new_ = configuration_length_;
// //     if (!initialized_) {
// //       InitializeTrajectories(buffer.sensor, buffer.sensor_mask,
// buffer.ctrl,
// //                              buffer.time);
// //       initialized_ = true;
// //     } else {
// //       num_new_ = UpdateTrajectories(buffer.sensor, buffer.sensor_mask,
// //                                     buffer.ctrl, buffer.time);
// //     }
// //     num_new = num_new_;

// //     // optimize
// //     Optimize(pool);
// //   }
// //   return num_new;
// // }

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

// condition matrix: res = mat11 - mat10 * mat00 \ mat10^T; return rank of
// mat00
// TODO(taylor): thread
void ConditionMatrix(double* res, const double* mat, double* mat00,
                     double* mat10, double* mat11, double* tmp0, double* tmp1,
                     int n, int n0, int n1, double* bandfactor, int nband) {
  // unpack mat
  BlockFromMatrix(mat00, mat, n0, n0, n, n, 0, 0);
  BlockFromMatrix(mat10, mat, n1, n0, n, n, n0, 0);
  BlockFromMatrix(mat11, mat, n1, n1, n, n, n0, n0);

  // factorize mat00, solve mat00 \ mat10^T
  if (nband > 0 && bandfactor) {
    mju_dense2Band(bandfactor, mat00, n0, nband, 0);

    // factorize mat00
    mju_cholFactorBand(bandfactor, n0, nband, 0, 0.0, 0.0);

    // tmp0 = mat00 \ mat01 = (mat00^-1 mat01)^T
    for (int i = 0; i < n1; i++) {
      mju_cholSolveBand(tmp0 + n0 * i, bandfactor, mat10 + n0 * i, n0, nband,
                        0);
    }
  } else {
    // factorize mat00
    mju_cholFactor(mat00, n0, 0.0);

    // tmp0 = mat00 \ mat01 = (mat00^-1 mat01)^T
    for (int i = 0; i < n1; i++) {
      mju_cholSolve(tmp0 + n0 * i, mat00, mat10 + n0 * i, n0);
    }
  }

  // tmp1 = mat10 * (mat00 \ mat01)
  mju_mulMatMatT(tmp1, tmp0, mat10, n1, n0, n1);

  // res = mat11 - mat10 * (mat00 \ mat01)
  mju_sub(res, mat11, tmp1, n1 * n1);
}

// // compute skew symmetric matrix
// void SkewSymmetricMatrix(double* mat, const double* x) {
//   mat[0] = 0.0;
//   mat[1] = -x[2];
//   mat[2] = x[1];
//   mat[3] = x[2];
//   mat[4] = 0.0;
//   mat[5] = -x[0];
//   mat[6] = -x[1];
//   mat[7] = x[0];
//   mat[8] = 0.0;
// }

// // Jacobians of mju_quatIntegrate wrt quat, vel
// // http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
// // https://arxiv.org/pdf/1812.01537.pdf
// void DifferentiateQuatIntegrate(double* jacquat, double* jacvel,
//                                 const double* quat, const double* vel,
//                                 double scale) {
//   // integrate
//   double quati[4];
//   mju_copy3(quati, quat);
//   mju_quatIntegrate(quati, vel, scale);

//   // Jacobian wrt quat
//   if (jacquat) {
//     // quaternion -> rotation matrix
//     double mat[9];
//     mju_quat2Mat(mat, quati);
//     mju_transpose(jacquat, mat, 3, 3);
//   }

//   // Jacobian wrt vel
//   if (jacvel) {
//     // scaled vel
//     double s_vel[3];
//     mju_scl3(s_vel, vel, scale);

//     // norm of scaled rotation
//     double n = mju_norm3(s_vel);

//     // check small norm
//     if (n < 1.0e-8) {
//       mju_zero(jacvel, 9);
//       return;
//     }

//     // coefficients
//     double n2 = n * n;
//     double n3 = n2 * n;
//     double s0 = (1.0 - mju_cos(n)) / n2;
//     double s1 = (n - mju_sin(n)) / n3;

//     // skew symmetric matrix
//     double skew[9];
//     SkewSymmetricMatrix(skew, s_vel);
//     double skew2[9];
//     mju_mulMatMat(skew2, skew, skew, 3, 3, 3);

//     // jacvel = I - s0 * skew + s1 * skew^2
//     mju_eye(jacvel, 3);
//     mju_addToScl(jacvel, skew, -s0, 9);
//     mju_addToScl(jacvel, skew2, s1, 9);
//   }
// }

// // compute slerp between quat0 and quat1 for t in [0, 1]
// // optionally compute Jacobians wrt quat0, quat1
// void Slerp(double* res, const double* quat0, const double* quat1, double t,
//            double* jac0, double* jac1) {
//   // quaternion difference
//   double dq[3];
//   mju_subQuat(dq, quat1, quat0);

//   // integrate
//   mju_copy4(res, quat0);
//   mju_quatIntegrate(res, dq, t);

//   // slerp Jacobian
//   if (jac0 || jac1) {
//     // differentiate subQuat
//     double dvdq0[9];
//     double dvdq1[9];
//     DifferentiateSubQuat(dvdq1, dvdq0, quat1, quat0);

//     // differentiate quatIntegratae
//     double dqdq[9];
//     double dqdv[9];
//     DifferentiateQuatIntegrate(dqdq, dqdv, quat0, dq, t);

//     // Jacobian wrt quat0: dqdv * dvdq0 * t + dqdq
//     if (jac0) {
//       mju_mulMatMat(jac0, dqdv, dvdq0, 3, 3, 3);
//       mju_scl(jac0, jac0, t, 9);
//       mju_addTo(jac0, dqdq, 9);
//     }

//     // Jacobian wrt quat1: dqdv * dvdq * t
//     if (jac1) {
//       mju_mulMatMat(jac1, dqdv, dvdq1, 3, 3, 3);
//       mju_scl(jac1, jac1, t, 9);
//     }
//   }
// }

// estimator-specific GUI elements
void Batch::GUI(mjUI& ui, double* process_noise, double* sensor_noise,
                double& timestep, int& integrator) {
  // ----- estimator ------ //
  mjuiDef defEstimator[] = {
      {mjITEM_SECTION, "Estimator Settings", 1, nullptr,
       "AP"},  // needs new section to satisfy mjMAXUIITEM
      {mjITEM_BUTTON, "Reset", 2, nullptr, ""},
      {mjITEM_SLIDERNUM, "Timestep", 2, &timestep, "1.0e-3 0.1"},
      {mjITEM_SELECT, "Integrator", 2, &integrator,
       "Euler\nRK4\nImplicit\nFastImplicit"},
      {mjITEM_END}};

  // add estimator
  mjui_add(&ui, defEstimator);

  // -- process noise -- //
  int nv = model->nv;
  int process_noise_shift = 0;
  mjuiDef defProcessNoise[kMaxProcessNoise + 2];

  // separator
  defProcessNoise[0] = {mjITEM_SEPARATOR, "Process Noise Covariance", 1};
  process_noise_shift++;

  // add UI elements
  for (int i = 0; i < nv; i++) {
    // element
    defProcessNoise[process_noise_shift] = {mjITEM_SLIDERNUM, "", 2,
                                            process_noise + i, "1.0e-8 0.01"};

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
  defSensorNoise[0] = {mjITEM_SEPARATOR, "Sensor Noise Covariance", 1};
  sensor_noise_shift++;

  // loop over sensors
  std::string sensor_str;
  for (int i = 0; i < nsensor; i++) {
    std::string name_sensor(model->names +
                            model->name_sensoradr[sensor_start + i]);

    // element
    defSensorNoise[sensor_noise_shift] = {mjITEM_SLIDERNUM, "", 2,
                                          sensor_noise + sensor_noise_shift - 1,
                                          "1.0e-8 0.01"};

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

// estimator-specific plots
void Batch::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                  int planner_shift, int timer_shift, int planning,
                  int* shift) {
  // Batch info
  double estimator_bounds[2] = {-6, 6};

  // covariance trace
  double trace = Trace(covariance.data(), DimensionProcess());
  mjpc::PlotUpdateData(fig_planner, estimator_bounds,
                       fig_planner->linedata[planner_shift + 0][0] + 1,
                       mju_log10(trace), 100, planner_shift + 0, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[planner_shift + 0], "Covariance Trace");

  // Batch timers
  double timer_bounds[2] = {0.0, 1.0};

  // update
  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[timer_shift + 0][0] + 1, timer_.update,
                 100, timer_shift + 0, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[timer_shift + 0], "Update");
}

}  // namespace mjpc