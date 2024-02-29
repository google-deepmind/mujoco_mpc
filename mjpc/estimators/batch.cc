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
#include "mjpc/direct/direct.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;

// batch smoother constructor
Batch::Batch(const mjModel* model, int length, int max_history)
    : Direct::Direct(model, length, max_history) {
  // initialize memory
  Initialize(model);

  // set trajectory lengths
  SetConfigurationLength(length);

  // reset memory
  Reset();
}

// initialize batch estimator
void Batch::Initialize(const mjModel* model) {
  // base method
  Direct::Initialize(model);

  // set batch size
  int batch_size = std::min(
      std::max(GetNumberOrDefault(3, model, "batch_configuration_length"),
               kMinDirectHistory),
      kMaxFilterHistory);

  if (batch_size != configuration_length_) {
    SetConfigurationLength(batch_size);
  }

  // allocation dimension
  int nq = model->nq, nv = model->nv, na = model->na;
  int nvel_max = nv * max_history_;

  // int nsensor_max = nsensordata_ * max_history_;
  int ntotal_max = nvel_max + nparam_;

  // state dimensions
  nstate_ = nq + nv + na;
  ndstate_ = 2 * nv + na;

  // state
  state.resize(nstate_);

  // covariance
  covariance.resize(ndstate_ * ndstate_);

  // residual
  residual_prior_.resize(nvel_max);

  // Jacobian
  jacobian_prior_.resize(filter_settings.assemble_prior_jacobian * nvel_max *
                         ntotal_max);

  // prior Jacobian block
  block_prior_current_configuration_.Initialize(nv * nv, configuration_length_);

  // cost gradient
  cost_gradient_prior_.resize(ntotal_max);

  // cost Hessian
  cost_hessian_.resize(ntotal_max * ntotal_max);
  cost_hessian_prior_band_.resize(nvel_max * nband_ + nparam_ * ntotal_max);

  // prior weights
  scale_prior = GetNumberOrDefault(1.0, model, "batch_scale_prior");
  weight_prior_.resize(ntotal_max * ntotal_max);
  weight_prior_band_.resize(nvel_max * nband_ + nparam_ * ntotal_max);

  // scratch
  scratch_prior_.resize(ntotal_max + 12 * nv * nv);

  // conditioned matrix
  mat00_.resize(ntotal_max * ntotal_max);
  mat10_.resize(ntotal_max * ntotal_max);
  mat11_.resize(ntotal_max * ntotal_max);
  condmat_.resize(ntotal_max * ntotal_max);
  scratch0_condmat_.resize(ntotal_max * ntotal_max);
  scratch1_condmat_.resize(ntotal_max * ntotal_max);

  // timer
  filter_timer_.prior_step.resize(max_history_);

  // -- trajectory cache -- //
  configuration_cache_.Initialize(nq, max_history_);
  velocity_cache_.Initialize(nv, max_history_);
  acceleration_cache_.Initialize(nv, max_history_);
  act_cache_.Initialize(na, max_history_);
  times_cache_.Initialize(1, max_history_);

  // prior
  configuration_previous_cache_.Initialize(nq, max_history_);

  // sensor
  sensor_measurement_cache_.Initialize(nsensordata_, max_history_);
  sensor_prediction_cache_.Initialize(nsensordata_, max_history_);
  sensor_mask_cache_.Initialize(nsensor_, max_history_);

  // force
  force_measurement_cache_.Initialize(nv, max_history_);
  force_prediction_cache_.Initialize(nv, max_history_);

  // -- GUI data -- //
  // time step
  gui_timestep_ = model->opt.timestep;

  // integrator
  gui_integrator_ = model->opt.integrator;

  // process noise
  gui_process_noise_.resize(ndstate_);

  // sensor noise
  gui_sensor_noise_.resize(nsensordata_);

  // scale prior
  gui_scale_prior_ = scale_prior;

  // estimation horizon
  gui_horizon_ = configuration_length_;

  cost_difference_ = improvement_ = expected_ = reduction_ratio_ = 0.0;
}

// reset memory
void Batch::Reset(const mjData* data) {
  // base method
  Direct::Reset(data);

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

  // scale prior
  scale_prior = GetNumberOrDefault(1.0e-1, model, "batch_scale_prior");

  // residual
  std::fill(residual_prior_.begin(), residual_prior_.end(), 0.0);

  // Jacobian
  std::fill(jacobian_prior_.begin(), jacobian_prior_.end(), 0.0);

  // prior Jacobian block
  block_prior_current_configuration_.Reset();

  // cost
  cost_prior_ = 0.0;

  // cost gradient
  std::fill(cost_gradient_prior_.begin(), cost_gradient_prior_.end(), 0.0);

  // cost Hessian
  std::fill(cost_hessian_prior_band_.begin(), cost_hessian_prior_band_.end(),
            0.0);

  // weight
  std::fill(weight_prior_.begin(), weight_prior_.end(), 0.0);
  std::fill(weight_prior_band_.begin(), weight_prior_band_.end(), 0.0);

  // scratch
  std::fill(scratch_prior_.begin(), scratch_prior_.end(), 0.0);

  // conditioned matrix
  std::fill(mat00_.begin(), mat00_.end(), 0.0);
  std::fill(mat10_.begin(), mat10_.end(), 0.0);
  std::fill(mat11_.begin(), mat11_.end(), 0.0);
  std::fill(condmat_.begin(), condmat_.end(), 0.0);
  std::fill(scratch0_condmat_.begin(), scratch0_condmat_.end(), 0.0);
  std::fill(scratch1_condmat_.begin(), scratch1_condmat_.end(), 0.0);

  // timer
  std::fill(filter_timer_.prior_step.begin(), filter_timer_.prior_step.end(),
            0.0);

  InitializeFilter();

  // trajectory cache
  configuration_cache_.Reset();
  velocity_cache_.Reset();
  acceleration_cache_.Reset();
  act_cache_.Reset();
  times_cache_.Reset();

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

  // -- GUI data -- //
  // time step
  gui_timestep_ = model->opt.timestep;

  // integrator
  gui_integrator_ = model->opt.integrator;

  // process noise
  double noise_process_scl =
      GetNumberOrDefault(1.0e-4, model, "estimator_process_noise_scale");
  std::fill(gui_process_noise_.begin(), gui_process_noise_.end(),
            noise_process_scl);

  // sensor noise
  double noise_sensor_scl =
      GetNumberOrDefault(1.0e-4, model, "estimator_sensor_noise_scale");
  std::fill(gui_sensor_noise_.begin(), gui_sensor_noise_.end(),
            noise_sensor_scl);

  // scale prior
  gui_scale_prior_ = scale_prior;

  // estimation horizon
  gui_horizon_ = configuration_length_;
}

// update
void Batch::Update(const double* ctrl, const double* sensor, int mode) {
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

  // set sensor
  sensor_measurement.Set(sensor + sensor_start_index_, t);

  // set force measurement
  force_measurement.Set(d->qfrc_actuator, t);

  // -- measurement update -- //

  // cache configuration length
  int configuration_length_cache = configuration_length_;

  // set reduced configuration length for optimization
  configuration_length_ = current_time_index_ + 2;
  nvel_ = nv * configuration_length_;
  ntotal_ = nvel_ + nparam_;
  if (configuration_length_ != configuration_length_cache) {
    ShiftResizeTrajectory(0, configuration_length_);
  }

  // optimize measurement corrected state
  Optimize();

  // update state
  mju_copy(state.data(), configuration.Get(t + 1), nq);
  mju_copy(state.data() + nq, velocity.Get(t + 1), nv);
  mju_copy(state.data() + nq + nv, act.Get(t + 1), na);
  time = times.Get(t + 1)[0];

  // -- update prior weights -- //

  // prior weights
  double* weights = weight_prior_.data();

  // recursive update
  if (filter_settings.recursive_prior_update &&
      configuration_length_ == configuration_length_cache) {
    // condition dimension
    int ncondition = nvel_ - nv;

    // band to dense cost Hessian
    mju_band2Dense(cost_hessian_.data(), cost_hessian_band_.data(), ntotal_,
                   nband_, nparam_, 1);

    // compute conditioned matrix
    double* condmat = condmat_.data();
    ConditionMatrix(condmat, cost_hessian_.data(), mat00_.data(), mat10_.data(),
                    mat11_.data(), scratch0_condmat_.data(),
                    scratch1_condmat_.data(), ntotal_, nv, ncondition);

    // zero memory
    mju_zero(weights, ntotal_ * ntotal_);

    // set conditioned matrix in prior weights
    SetBlockInMatrix(weights, condmat, 1.0, ntotal_, ntotal_, ncondition,
                     ncondition, 0, 0);

    // set bottom right to scale_prior * I
    for (int i = ncondition; i < ntotal_; i++) {
      weights[ntotal_ * i + i] = scale_prior;
    }

    // make block band
    DenseToBlockBand(weights, ntotal_, nv, 3);
  } else {
    // dimension
    int nvar_new = ntotal_;
    if (current_time_index_ < configuration_length_ - 2) {
      nvar_new += nv;
    }

    // check same size
    if (ntotal_ != nvar_new) {
      // get previous weights
      double* previous_weights = scratch0_condmat_.data();
      mju_copy(previous_weights, weights, ntotal_ * ntotal_);

      // set previous in new weights (dimension may have increased)
      mju_zero(weights, nvar_new * nvar_new);
      SetBlockInMatrix(weights, previous_weights, 1.0, nvar_new, nvar_new,
                       ntotal_, ntotal_, 0, 0);

      // scale_prior * I
      for (int i = ntotal_; i < nvar_new; i++) {
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
  nvel_ = nv * configuration_length_;
  ntotal_ = nvel_ + nparam_;

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

// set state
void Batch::SetState(const double* state) {
  // state
  mju_copy(this->state.data(), state, ndstate_);

  // -- configurations -- //
  int nq = model->nq;
  int t = 1;

  // q1
  configuration.Set(state, t);

  // q0
  double* q0 = configuration.Get(t - 1);
  mju_copy(q0, state, nq);
  mj_integratePos(model, q0, state + nq, -1.0 * model->opt.timestep);
}

// set time
void Batch::SetTime(double time) {
  // copy
  double time_copy = time;

  // t1
  times.Set(&time_copy, 1);

  // t0
  time_copy -= model->opt.timestep;
  times.Set(&time_copy, 0);

  // reset current time index
  current_time_index_ = 1;
}

// compute and return dense prior Jacobian
const double* Batch::GetJacobianPrior() {
  // resize
  jacobian_prior_.resize(ntotal_ * ntotal_);

  // change setting
  int settings_cache = filter_settings.assemble_prior_jacobian;
  filter_settings.assemble_prior_jacobian = true;

  // loop over configurations to assemble Jacobian
  for (int t = 0; t < configuration_length_; t++) {
    BlockPrior(t);
  }

  // restore setting
  filter_settings.assemble_prior_jacobian = settings_cache;

  // return dense Jacobian
  return jacobian_prior_.data();
}

// set prior weights
void Batch::SetPriorWeights(const double* weights, double scale) {
  // dimension
  int nv = model->nv;

  // allocate memory
  weight_prior_.resize(nvel_ * nvel_);
  weight_prior_band_.resize(nvel_ * nband_);

  // set weights
  mju_copy(weight_prior_.data(), weights, nvel_ * nvel_);

  // make block band
  DenseToBlockBand(weight_prior_.data(), nvel_, nv, 3);

  // dense to band
  mju_dense2Band(weight_prior_band_.data(), weight_prior_.data(), nvel_, nband_,
                 0);

  // set scaling
  scale_prior = scale;
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

  configuration_previous.Shift(shift);

  sensor_measurement.Shift(shift);
  sensor_prediction.Shift(shift);
  sensor_mask.Shift(shift);

  force_measurement.Shift(shift);
  force_prediction.Shift(shift);
}

// prior cost
double Batch::CostPrior(double* gradient, double* hessian) {
  // start timer
  auto start_cost = std::chrono::steady_clock::now();

  // total scaling
  double scale = scale_prior / ntotal_;

  // dimension
  int nv = model->nv;

  // unpack
  double* r = residual_prior_.data();
  double* tmp = scratch_prior_.data();

  // zero memory
  if (gradient) mju_zero(gradient, ntotal_);
  if (hessian) mju_zero(hessian, nvel_ * nband_ + nparam_ * ntotal_);

  // initial cost
  double cost = 0.0;

  // dense2band
  mju_dense2Band(weight_prior_band_.data(), weight_prior_.data(), nvel_, nband_,
                 0);

  // compute cost
  if (!cost_skip_) {
    // residual
    ResidualPrior();

    // multiply: tmp = P * r
    mju_bandMulMatVec(tmp, weight_prior_band_.data(), r, nvel_, nband_, 0, 1,
                      true);

    // weighted quadratic: 0.5 * w * r' * tmp
    cost = 0.5 * scale * mju_dot(r, tmp, nvel_);

    // stop cost timer
    filter_timer_.cost_prior += GetDuration(start_cost);
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
        double* mat = tmp + ntotal_;
        mju_zero(mat, nband_ * nband_);

        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            if (j <= i) {
              // unpack
              double* bbij = mat + nband_ * nband_;
              double* tmp0 = bbij + nv * nv;
              double* tmp1 = tmp0 + nv * nv;

              // get matrices
              BlockFromMatrix(bbij, weight_prior_.data(), nv, nv, nvel_, nvel_,
                              (i + t) * nv, (j + t) * nv);
              const double* bdi = block_prior_current_configuration_.Get(i + t);
              const double* bdj = block_prior_current_configuration_.Get(j + t);

              // -- bdi' * bbij * bdj -- //

              // tmp0 = bbij * bdj
              mju_mulMatMat(tmp0, bbij, bdj, nv, nv, nv);

              // tmp1 = bdi' * tmp0
              mju_mulMatTMat(tmp1, bdi, tmp0, nv, nv, nv);

              // set scaled block in matrix
              SetBlockInMatrix(mat, tmp1, scale, nband_, nband_, nv, nv, i * nv,
                               j * nv);
            }
          }
        }
        // set mat in band Hessian
        SetBlockInBand(hessian, mat, 1.0, nvel_, nband_, nband_, t * nv, 0,
                       false);
      }
    }
  }

  // stop derivatives timer
  filter_timer_.cost_prior_derivatives += GetDuration(start_derivatives);

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
  filter_timer_.residual_prior += GetDuration(start);
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
  if (filter_settings.assemble_prior_jacobian) {
    // dimensions
    int nv = model->nv;

    // set block
    SetBlockInMatrix(jacobian_prior_.data(), block, 1.0, ntotal_, ntotal_, nv,
                     nv, nv * index, nv * index);
  }
}

// prior Jacobian
// note: pool wait is called outside this function
void Batch::JacobianPrior() {
  // loop over predictions
  for (int t = 0; t < configuration_length_; t++) {
    // schedule by time step
    pool_.Schedule([&batch = *this, t]() {
      // start Jacobian timer
      auto jacobian_prior_start = std::chrono::steady_clock::now();

      // block
      batch.BlockPrior(t);

      // stop Jacobian timer
      batch.filter_timer_.prior_step[t] = GetDuration(jacobian_prior_start);
    });
  }
}

// initialize filter mode
void Batch::InitializeFilter() {
  // dimensions
  int nq = model->nq;

  // filter mode status
  current_time_index_ = 1;

  // filter settings
  settings.gradient_tolerance = 1.0e-6;
  settings.max_smoother_iterations = 1;
  settings.max_search_iterations = 10;
  filter_settings.recursive_prior_update = true;
  settings.first_step_position_sensors = true;
  settings.last_step_position_sensors = false;
  settings.last_step_velocity_sensors = false;

  // check for number of parameters
  if (nparam_ != 0) {
    mju_error("filter mode requires nparam_ == 0\n");
  }

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
  if (model->opt.enableflags & (mjENBL_ENERGY)) {
    mj_energyPos(model, data);
  }

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
  for (int i = 0; i < ntotal_; i++) {
    weight_prior_[ntotal_ * i + i] = scale_prior;
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
    sensor_measurement.Set(sensor_measurement_cache_.Get(new_head + i), i);
    sensor_prediction.Set(sensor_prediction_cache_.Get(new_head + i), i);
    sensor_mask.Set(sensor_mask_cache_.Get(new_head + i), i);
    force_measurement.Set(force_measurement_cache_.Get(new_head + i), i);
    force_prediction.Set(force_prediction_cache_.Get(new_head + i), i);
  }
}

// compute total cost
double Batch::Cost(double* gradient, double* hessian) {
  // base method
  double cost = Direct::Cost(gradient, hessian);

  // prior Jacobian derivatives
  if (gradient || hessian) {
    // start timer for prior Jacobian
    auto timer_jacobian_start = std::chrono::steady_clock::now();

    // individual derivatives
    if (filter_settings.assemble_prior_jacobian) {
      mju_zero(jacobian_prior_.data(), ntotal_ * ntotal_);
    }

    // pool count
    int count_begin = pool_.GetCount();

    // compute Jacobian of prior cost
    JacobianPrior();

    // wait
    pool_.WaitCount(count_begin + configuration_length_);

    // reset count
    pool_.ResetCount();

    // timers
    filter_timer_.jacobian_prior +=
        mju_sum(filter_timer_.prior_step.data(), configuration_length_);
    timer_.jacobian_total += GetDuration(timer_jacobian_start);
  }

  // prior cost
  cost_prior_ = CostPrior(gradient ? cost_gradient_prior_.data() : NULL,
                          hessian ? cost_hessian_prior_band_.data() : NULL);

  // total cost
  cost += cost_prior_;

  // total gradient, hessian
  if (gradient) {
    // start gradient timer
    auto start = std::chrono::steady_clock::now();

    // add prior gradient
    mju_addTo(gradient, cost_gradient_prior_.data(), ntotal_);

    // stop gradient timer
    timer_.cost_gradient += GetDuration(start);
  }

  if (hessian) {
    // start Hessian timer
    auto start = std::chrono::steady_clock::now();

    // add prior Hessian
    mju_addTo(hessian, cost_hessian_prior_band_.data(),
              nvel_ * nband_ + nparam_ * ntotal_);

    // stop Hessian timer
    timer_.cost_hessian += GetDuration(start);
  }

  // total cost
  return cost;
}

// reset timers
void Batch::ResetTimers() {
  // base method
  Direct::ResetTimers();
  filter_timer_.jacobian_prior = 0.0;
  filter_timer_.cost_prior_derivatives = 0.0;
  filter_timer_.cost_prior = 0.0;
  filter_timer_.residual_prior = 0.0;
  filter_timer_.prior_weight_update = 0.0;
  filter_timer_.prior_set_weight = 0.0;
}

// estimator-specific GUI elements
void Batch::GUI(mjUI& ui) {
  // ----- estimator ------ //
  mjuiDef defEstimator[] = {
      {mjITEM_SECTION, "Estimator", 1, nullptr,
       "AP"},  // needs new section to satisfy mjMAXUIITEM
      {mjITEM_BUTTON, "Reset", 2, nullptr, ""},
      {mjITEM_SLIDERNUM, "Timestep", 2, &gui_timestep_, "1.0e-3 0.1"},
      {mjITEM_SELECT, "Integrator", 2, &gui_integrator_,
       "Euler\nRK4\nImplicit\nFastImplicit"},
      {mjITEM_SLIDERINT, "Horizon", 2, &gui_horizon_, "3 3"},
      {mjITEM_SLIDERNUM, "Prior Scale", 2, &gui_scale_prior_, "1.0e-8 0.1"},
      {mjITEM_END}};

  // set estimation horizon limits
  mju::sprintf_arr(defEstimator[4].other, "%i %i", kMinDirectHistory,
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
        gui_sensor_noise_.data() + sensor_noise_shift - 1, "1.0e-8 0.01"};

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
void Batch::SetGUIData() {
  // time step
  model->opt.timestep = gui_timestep_;

  // integrator
  model->opt.integrator = gui_integrator_;

  // TODO(taylor): update models if nparam > 0

  // noise
  mju_copy(noise_process.data(), gui_process_noise_.data(), DimensionProcess());
  mju_copy(noise_sensor.data(), gui_sensor_noise_.data(), DimensionSensor());

  // scale prior
  scale_prior = gui_scale_prior_;

  // store estimation horizon
  int horizon = gui_horizon_;

  // changing horizon cases
  if (horizon > configuration_length_) {  // increase horizon
    // -- prior weights resize -- //
    int ntotal_new = model->nv * horizon;

    // get previous weights
    double* weights = weight_prior_.data();
    double* previous_weights = scratch0_condmat_.data();
    mju_copy(previous_weights, weights, ntotal_ * ntotal_);

    // set previous in new weights (dimension may have increased)
    mju_zero(weights, ntotal_new * ntotal_new);
    SetBlockInMatrix(weights, previous_weights, 1.0, ntotal_new, ntotal_new,
                     ntotal_, ntotal_, 0, 0);

    // scale_prior * I
    for (int i = ntotal_; i < ntotal_new; i++) {
      weights[ntotal_new * i + i] = scale_prior;
    }

    // modify trajectories
    ShiftResizeTrajectory(0, horizon);

    // update configuration length
    configuration_length_ = horizon;
    nvel_ = model->nv * configuration_length_;
    ntotal_ = nvel_ + nparam_;
  } else if (horizon < configuration_length_) {  // decrease horizon
    // -- prior weights resize -- //
    int ntotal_new = model->nv * horizon;

    // get previous weights
    double* weights = weight_prior_.data();
    double* previous_weights = scratch0_condmat_.data();
    BlockFromMatrix(previous_weights, weights, ntotal_new, ntotal_new, ntotal_,
                    ntotal_, 0, 0);

    // set previous in new weights (dimension may have increased)
    mju_zero(weights, ntotal_ * ntotal_);
    mju_copy(weights, previous_weights, ntotal_new * ntotal_new);

    // compute difference in estimation horizons
    int horizon_diff = configuration_length_ - horizon;

    // modify trajectories
    ShiftResizeTrajectory(horizon_diff, horizon);

    // update configuration length and current time index
    configuration_length_ = horizon;
    nvel_ = model->nv * configuration_length_;
    ntotal_ = nvel_ + nparam_;
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

}  // namespace mjpc
