// Copyright 2022 DeepMind Technologies Limited
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

#include "mjpc/trajectory.h"

#include <algorithm>
#include <functional>
#include <iostream>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {
namespace {
// maximum return value
inline constexpr double kMaxReturnValue = 1.0e6;
}

// initialize dimensions
void Trajectory::Initialize(int dim_state, int dim_action, int dim_residual,
                            int num_trace, int horizon) {
  this->horizon = horizon;
  this->dim_state = dim_state;
  this->dim_action = dim_action;
  this->dim_residual = dim_residual;
  this->dim_trace = 3 * num_trace;
  this->failure = false;
}

// allocate memory
void Trajectory::Allocate(int T) {
  // states
  states.resize(dim_state * T);

  // actions
  actions.resize(dim_action * T);

  // costs
  costs.resize(T);

  // residual
  residual.resize(dim_residual * T);

  // times
  times.resize(T);

  // traces
  trace.resize(dim_trace * T);
}

// reset memory to zeros
void Trajectory::Reset(int T) {
  // states
  std::fill(states.begin(), states.begin() + dim_state * T, 0.0);

  // actions
  std::fill(actions.begin(), actions.begin() + dim_action * T, 0.0);

  // times
  std::fill(times.begin(), times.begin() + T, 0.0);

  // costs
  std::fill(costs.begin(), costs.begin() + T, 0.0);
  std::fill(residual.begin(), residual.begin() + dim_residual * T, 0.0);
  total_return = 0.0;
  failure = false;

  // traces
  std::fill(trace.begin(), trace.begin() + dim_trace * T, 0.0);
}

// simulate model forward in time with continuous-time indexed policy
void Trajectory::Rollout(
    std::function<void(double* action, const double* state, double time)>
        policy,
    const Task* task, const mjModel* model, mjData* data, const double* state,
    double time, const double* mocap, const double* userdata, int steps) {
  // reset failure flag
  failure = false;

  // model sizes
  int nq = model->nq;
  int nv = model->nv;
  int na = model->na;
  int nu = model->nu;
  int nmocap = model->nmocap;
  int nuserdata = model->nuserdata;

  // horizon
  horizon = steps;

  // set mocap
  for (int i = 0; i < nmocap; i++) {
    mju_copy(data->mocap_pos + 3 * i, mocap + 7 * i, 3);
    mju_copy(data->mocap_quat + 4 * i, mocap + 7 * i + 3, 4);
  }

  // set userdata
  mju_copy(data->userdata, userdata, nuserdata);

  // set initial state
  mju_copy(states.data(), state, dim_state);
  mju_copy(data->qpos, state, nq);
  mju_copy(data->qvel, state + nq, nv);
  mju_copy(data->act, state + nq + nv, na);

  // set initial time
  times[0] = time;
  data->time = time;

  for (int t = 0; t < horizon - 1; t++) {
    // set action
    policy(DataAt(actions, t * nu), DataAt(states, t * dim_state), data->time);
    mju_copy(data->ctrl, DataAt(actions, t * nu), nu);

    // step
    mj_step(model, data);

    // record residual
    mju_copy(DataAt(residual, t * dim_residual), data->sensordata,
             dim_residual);

    // record trace
    GetTraces(DataAt(trace, t * 3 * task->num_trace), model, data,
              task->num_trace);

    // check for step warnings
    if ((failure |= CheckWarnings(data))) {
      total_return = kMaxReturnValue;
      std::cerr << "Rollout divergence at step\n";
      return;
    }

    // record state
    mju_copy(DataAt(states, (t + 1) * dim_state), data->qpos, nq);
    mju_copy(DataAt(states, (t + 1) * dim_state + nq), data->qvel, nv);
    mju_copy(DataAt(states, (t + 1) * dim_state + nq + nv), data->act, na);
    times[t + 1] = data->time;
  }

  // check for step warnings
  if ((failure |= CheckWarnings(data))) {
    total_return = kMaxReturnValue;
    std::cerr << "Rollout divergence at step\n";
    return;
  }

  // copy final action
  if (horizon > 1) {
    mju_copy(DataAt(actions, (horizon - 1) * dim_action),
             DataAt(actions, (horizon - 2) * dim_action), dim_action);
  } else {
    mju_zero(DataAt(actions, (horizon - 1) * dim_action), dim_action);
  }

  // final forward
  mj_forward(model, data);

  // final residual
  mju_copy(DataAt(residual, (horizon - 1) * dim_residual), data->sensordata,
           dim_residual);

  // final trace
  GetTraces(DataAt(trace, (horizon - 1) * 3 * task->num_trace), model, data,
            task->num_trace);

  // compute return
  UpdateReturn(task);
}

// simulate model forward in time with discrete-time indexed policy
void Trajectory::RolloutDiscrete(
    std::function<void(double* action, const double* state, int index)>
        policy,
    const Task* task, const mjModel* model, mjData* data, const double* state,
    double time, const double* mocap, const double* userdata, int steps) {
  // reset failure flag
  failure = false;

  // model sizes
  int nq = model->nq;
  int nv = model->nv;
  int na = model->na;
  int nu = model->nu;
  int nmocap = model->nmocap;
  int nuserdata = model->nuserdata;

  // horizon
  horizon = steps;

  // set mocap
  for (int i = 0; i < nmocap; i++) {
    mju_copy(data->mocap_pos + 3 * i, mocap + 7 * i, 3);
    mju_copy(data->mocap_quat + 4 * i, mocap + 7 * i + 3, 4);
  }

  // set userdata
  mju_copy(data->userdata, userdata, nuserdata);

  // set initial state
  mju_copy(states.data(), state, dim_state);
  mju_copy(data->qpos, state, nq);
  mju_copy(data->qvel, state + nq, nv);
  mju_copy(data->act, state + nq + nv, na);

  // set initial time
  times[0] = time;
  data->time = time;

  for (int t = 0; t < horizon - 1; t++) {
    // set action
    policy(DataAt(actions, t * nu), DataAt(states, t * dim_state), t);
    mju_copy(data->ctrl, DataAt(actions, t * nu), nu);

    // step
    mj_step(model, data);

    // record residual
    mju_copy(DataAt(residual, t * dim_residual), data->sensordata,
             dim_residual);

    // record trace
    GetTraces(DataAt(trace, t * 3 * task->num_trace), model, data,
              task->num_trace);

    // check for step warnings
    if ((failure |= CheckWarnings(data))) {
      total_return = kMaxReturnValue;
      std::cerr << "Rollout divergence at step\n";
      return;
    }

    // record state
    mju_copy(DataAt(states, (t + 1) * dim_state), data->qpos, nq);
    mju_copy(DataAt(states, (t + 1) * dim_state + nq), data->qvel, nv);
    mju_copy(DataAt(states, (t + 1) * dim_state + nq + nv), data->act, na);
    times[t + 1] = data->time;
  }

  // check for step warnings
  if ((failure |= CheckWarnings(data))) {
    total_return = kMaxReturnValue;
    std::cerr << "Rollout divergence at step\n";
    return;
  }

  // copy final action
  if (horizon > 1) {
    mju_copy(DataAt(actions, (horizon - 1) * dim_action),
             DataAt(actions, (horizon - 2) * dim_action), dim_action);
  } else {
    mju_zero(DataAt(actions, (horizon - 1) * dim_action), dim_action);
  }

  // final forward
  mj_forward(model, data);

  // final residual
  mju_copy(DataAt(residual, (horizon - 1) * dim_residual), data->sensordata,
           dim_residual);

  // final trace
  GetTraces(DataAt(trace, (horizon - 1) * 3 * task->num_trace), model, data,
            task->num_trace);

  // compute return
  UpdateReturn(task);
}

// calculates total_return and costs
void Trajectory::UpdateReturn(const Task* task) {
  // reset
  total_return = 0;

  for (int t = 0; t < horizon; t++) {
    // compute stage cost
    costs[t] = task->CostValue(DataAt(residual, t * task->num_residual));

    // update total return
    total_return += costs[t];
  }

  // normalize return by trajectory horizon
  total_return /= mju_max(horizon, 1);
}

}  // namespace mjpc
